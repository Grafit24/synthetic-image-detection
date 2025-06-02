from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List

from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.sana_transformer import SanaAttnProcessor2_0, SanaTransformer2DModel
from diffusers.pipelines.sana import SanaPipeline
from safetensors.torch import save_file, load_file
from dataclasses import dataclass, field
import itertools


@dataclass
class SanaClassifierParameters:
    proj_dim: int = 256
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    num_classes: int = 1
    drop_p: float = 0.2
    n_heads: int = 4


@dataclass
class SanaText2ImgParameters:
    prompt: str
    t: float = .25


class SanaClassifierHead(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dims: List[int] = [256, 128, 64], num_classes: int = 1, drop_p: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(drop_p))
            prev_dim = h_dim

        # final linear classifier
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class CrossAttentionProjection(nn.Module):
    """
    Given attention weights of shape (batch, heads, seq_len, seq_len), compute
    a per-layer projection vector of size (batch, proj_d).
    """
    def __init__(self, d_model: int, proj_d: int):
        super().__init__()
        # d_model = heads * seq_len
        self.layer_norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, proj_d)
        self.attn_weights = None

    def get_proj_states(self) -> torch.Tensor:
        return self.forward(self.attn_weights)

    def forward(self, attn_weights: torch.Tensor) -> torch.Tensor:
        batch_size, head_num, head_dim, seq_len = attn_weights.shape
        spatial_attn_l2 = torch.norm(attn_weights, dim=2)
        spatial_attn_l2 = spatial_attn_l2.view(batch_size, -1) / head_dim ** .5
        spatial_attn_l2 = self.layer_norm(spatial_attn_l2)
        proj_states = self.proj(spatial_attn_l2)
        return proj_states


class LayerSelfAttentionAggregator(nn.Module):
    def __init__(self, proj_dim: int, num_layers: int, n_heads: int = 4, dropout: float = .1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=n_heads,
            dim_feedforward=proj_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.ln_out = nn.LayerNorm(proj_dim)

    def forward(self, h_stack: torch.Tensor) -> torch.Tensor:
        """
        h_stack: (B, L, D)
        returns: (B, D) pooled representation
        """
        B, _, proj_dim = h_stack.size()
        cls = self.cls_token.expand(B, -1, proj_dim) # (B,1,D)
        x = torch.cat([cls, h_stack], dim=1) # (B,L+1,D)
        x = self.encoder(x)
        pooled = self.ln_out(x[:, 0]) # CLS output
        return pooled


class SanaClassifier(nn.Module):
    """
    Aggregates per-layer CrossAttentionProjection outputs via a learnable weighted sum (alpha),
    then feeds into a small MLP head to produce a final (binary) classification logit.
    """
    proj_module = CrossAttentionProjection
    agg_module = LayerSelfAttentionAggregator
    clf_head_module = SanaClassifierHead

    def __init__(
        self,
        num_layers: int,
        clf_params: SanaClassifierParameters
    ):
        super().__init__()
        self.num_layers = num_layers
        self.proj_dim = clf_params.proj_dim
        self.hidden_dims = clf_params.hidden_dims
        self.n_heads = clf_params.n_heads
        self.num_classes = clf_params.num_classes
        self.drop_p = clf_params.drop_p

        self.clf_head = self.clf_head_module(
            input_dim=self.proj_dim,
            hidden_dims=self.hidden_dims,
            num_classes=self.num_classes,
            drop_p=self.drop_p,
        )

        self.agg_block = self.agg_module(
            proj_dim=self.proj_dim, 
            num_layers=self.num_layers, 
            n_heads=self.n_heads, 
            dropout=self.drop_p
        )
        self.projections = nn.ModuleList()

    def create_proj(self, d_model: int) -> CrossAttentionProjection:
        """
        Called during register_model to create one CrossAttentionProjection for an attention layer.
        """
        proj = self.proj_module(d_model, self.proj_dim)
        self.projections.append(proj)
        return proj

    def forward(self) -> torch.Tensor:
        """
        After running a forward pass through the transformer with the modified attention processors,
        each CrossAttentionProjection inside self.projections will have stored its per-layer projection.
        We then collect them, weight by softmax(alpha), sum, and run through the final MLP head.
        """
        if len(self.projections) != self.num_layers:
            missing = self.num_layers - len(self.projections)
            raise RuntimeError(f"Expected projections for {self.num_layers} layers, but found {len(self.projections)} (missing {missing}).")

        pooled_feats: List[torch.Tensor] = []
        for proj in self.projections:
            h = proj.get_proj_states()  # shape: (batch, proj_dim)
            pooled_feats.append(h)

        H = self.agg_block(torch.stack(pooled_feats, dim=1))
        logits = self.clf_head(H)  # (batch, num_classes)
        return logits


class ClassifierSanaAttnProcessor(SanaAttnProcessor2_0):
    """
    Wraps the original SanaAttnProcessor2_0 but replaces the 'value' input to scaled_dot_product_attention
    with an identity, so that the attention output equals the attention weights. Then stores those weights
    into the provided CrossAttentionProjection module.
    """
    def __init__(self, proj: CrossAttentionProjection):
        super().__init__()
        self.proj = proj

    @staticmethod
    def get_v_identical(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Build a 'value' tensor of shape (batch, heads, seq_len, seq_len) that is identity along the
        last two dimensions. This ensures that scaled_dot_product_attention returns exactly the attention weights.
        """
        B, H, _, _ = query.shape
        S = key.shape[-2]
        # create an identity matrix of shape (S, S)
        eye_S = torch.eye(S, device=query.device, dtype=query.dtype)  # (S, S)
        # expand to (B, H, S, S)
        value_identity = eye_S.unsqueeze(0).unsqueeze(0).expand(B, H, S, S)
        return value_identity

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        value_identical = ClassifierSanaAttnProcessor.get_v_identical(query, key)
        attn_weights = F.scaled_dot_product_attention(
            query, key, value_identical, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        # now attn_weights has shape (batch, heads, seq_len, seq_len),
        # which we store into the projection module
        self.proj.attn_weights = attn_weights

        hidden_states = attn_weights @ value  # (batch, heads, seq_len, head_dim)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class SanaClassifierPipeline(SanaPipeline):
    """
    Extends the standard SanaPipeline to insert a binary classification head on top of the
    Transformer’s cross-attention maps. During 'register_model', we replace each SanaAttnProcessor2_0
    in selected transformer layers with ClassifierSanaAttnProcessor, which captures its attention weights.
    After a forward pass, we aggregate all per-layer projections through SanaClassifier and return logits.
    """
    clf_model: Optional[SanaClassifier] = None
    _registered: bool = False
    _processors: Dict[str, ClassifierSanaAttnProcessor] = {}
    max_sequence_length: int = 300
    t = .25
    prompt = ""

    def register_model(self, 
            transformer_blocks_ids: List[int], 
            clf_params: SanaClassifierParameters
        ) -> None:
        """
        Replace the SanaAttnProcessor2_0 in each specified transformer block with
        ClassifierSanaAttnProcessor, tied to a newly created CrossAttentionProjection.
        transformer_blocks_ids: list of layer indices (0-based) in self.transformer.transformer_blocks
        where we want to extract attention maps.
        """
        if self._registered:
            raise RuntimeError("Model has already been registered for classification.")

        num_hooks = len(transformer_blocks_ids)
        self.clf_model = SanaClassifier(
            num_layers=num_hooks,
            clf_params=clf_params
        )
        existing_procs = self.transformer.attn_processors
        full_procs: Dict[str, nn.Module] = {}

        hook_idx = 0
        # iterate over each registered processor (names include '.processor' suffix)
        for name, proc in existing_procs.items():
            parts = name.split(".")
            if len(parts) >= 3 and parts[0] == "transformer_blocks":
                layer_idx = int(parts[1])
                if layer_idx in transformer_blocks_ids and isinstance(proc, SanaAttnProcessor2_0):
                    attn_path = name[: -len(".processor")]
                    attn_module = self._get_module_by_path(self.transformer, attn_path)
                    seq_len = self.max_sequence_length
                    d_model = attn_module.heads * seq_len
                    proj = self.clf_model.create_proj(d_model)
                    new_proc = ClassifierSanaAttnProcessor(proj=proj)
                    full_procs[name] = new_proc
                    self._processors[name] = new_proc
                    hook_idx += 1
                    continue

            full_procs[name] = proc

        if hook_idx != num_hooks:
            raise RuntimeError(
                f"Expected to hook {num_hooks} layers, but only replaced {hook_idx}. "
                f"Check that the transformer_blocks_ids match actual layers."
            )
        self.transformer.set_attn_processor(full_procs)
        self.clf_model = self.clf_model.to(self.transformer.device, self.transformer.dtype)
        self._registered = True

    @classmethod
    def prepare_model(
        cls,
        pretrained_model: str,
        transformer_layers: List[int],
        clf_params: SanaClassifierParameters,
        t2i_params: SanaText2ImgParameters,
        device: str = "cuda",
        dtype = torch.bfloat16
    ):
        """
        Загружает SanaClassifierPipeline с заданной базовой моделью,
        переводит в нужные dtype и регистрирует классификатор.
        """
        device_obj = torch.device(device)
        try:
            pipe = cls.from_pretrained(pretrained_model, torch_dtype=torch.float32)
        except Exception:
            from transformers import modeling_utils
            if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
                modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]
            pipe = cls.from_pretrained(pretrained_model, torch_dtype=torch.float32)
        pipe.to(device_obj)
        pipe.text_encoder = pipe.text_encoder.to(dtype)
        pipe.transformer = pipe.transformer.to(dtype)
        pipe.register_model(transformer_blocks_ids=transformer_layers, clf_params=clf_params)
        pipe.prompt = t2i_params.prompt
        pipe.t = t2i_params.t
        return pipe

    def forward(self, image: Union[Image.Image, List[Image.Image]], prompt: str = None, t: float = None) -> torch.Tensor:
        """
        Run one pass through the SANa U-Net, capturing the attention projections, then
        compute and return classification logits.
        - image: either a PIL Image or a preprocessed tensor
        - prompt: text prompt for conditioning
        - t: a float in [0,1] giving the fraction of the noise schedule
        """
        prompt = prompt or self.prompt
        t = t or self.t
        if not self._registered or self.clf_model is None:
            raise RuntimeError("Classifier pipeline not registered. Call register_model(...) first.")
        if prompt is None or len(prompt) == 0 or t is None:
            raise RuntimeError("`prompt` and `t` is empty. Set it explicity or with register_model.")

        self.sana_forward(image, prompt, t)
        logits = self.clf_model()  # shape: (batch_size, num_classes)
        return logits

    def save_clf(self, save_path: Union[str, Path]):
        """
        Save the classifier's state_dict into a safetensors file.
        """
        if self.clf_model is None:
            raise RuntimeError("No classifier model to save. Ensure register_model(...) has been called.")
        path = str(save_path)
        state_dict = self.clf_model.state_dict()
        save_file(state_dict, path)

    def load_clf(self, load_path: Union[str, Path], map_location: Optional[Union[str, torch.device]] = None):
        """
        Load classifier weights from a safetensors file.
        """
        if self.clf_model is None:
            raise RuntimeError("No classifier model instantiated. Call register_model(...) first.")

        path = str(load_path)
        state_dict = load_file(path, device=map_location if map_location is not None else "cpu")
        self.clf_model.load_state_dict(state_dict)

    def _get_module_by_path(self, root_module: nn.Module, path: str) -> nn.Module:
        """
        Given a dot-separated path like "transformer_blocks.3.attn1", traverse
        the attributes (and indices for numeric components) to return the final module.
        """
        parts = path.split(".")
        cur: nn.Module = root_module
        for p in parts:
            if p.isdigit():
                idx = int(p)
                cur = cur[idx]
            else:
                cur = getattr(cur, p)
        return cur

    def train(self):
        self.clf_model.train()

    def eval(self):
        self.clf_model.eval()

    @torch.no_grad()
    def sana_forward(self, image: Union[Image.Image, torch.Tensor], prompt: str, t: float):
        """
        Exactly the same as SanaPipeline.sana_forward, but leaves any attention
        processors attached to the transformer in place, so that our custom
        processors store their projections. In particular, this writes:
          self.noisy_latent and self.noise_pred
        and invokes the transformer with our ClassifierSanaAttnProcessor instances,
        which populate self.clf_model.projections.
        """
        device = self._execution_device

        pixel_values = self.image_processor.preprocess(image).to(device)
        latents = self.vae.encode(pixel_values).latent
        batch_size = latents.size()[0]
        
        prompt_embeds, prompt_attention_mask, _, _ = self.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=False,
            negative_prompt="",
            num_images_per_prompt=latents.size()[0],
            device=device,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            prompt_attention_mask=None,
            negative_prompt_attention_mask=None,
            clean_caption=False,
            max_sequence_length=self.max_sequence_length,
            complex_human_instruction=None,
            lora_scale=None,
        )

        num_train_timesteps = self.scheduler.config.num_train_timesteps
        t_clamped = float(t)
        t_clamped = max(0.0, min(1.0, t_clamped))
        timestep_idx = int(t_clamped * (num_train_timesteps - 1))
        timestep_idxs = torch.tensor([timestep_idx] * batch_size, device=device, dtype=torch.long)

        noise = torch.randn_like(latents, device=device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timestep_idxs)

        latent_model_input = noisy_latents.to(self.transformer.dtype)
        timestep_tensor = timestep_idxs.to(latent_model_input.dtype)
        timestep_tensor = timestep_tensor * self.transformer.config.timestep_scale

        self.transformer(
            latent_model_input,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            timestep=timestep_tensor,
            return_dict=False,
            attention_kwargs=None,
        )

    def named_parameters(self):
        return itertools.chain(
            self.clf_model.named_parameters(),
            self.transformer.named_parameters(),
            self.vae.named_parameters(),
            self.text_encoder.named_parameters(),
        )

    def parameters(self):
        return itertools.chain(
            self.clf_model.parameters(),
            self.transformer.parameters(),
            self.vae.parameters(),
            self.text_encoder.parameters(),
        )

    def __call__(self, image: Union[Image.Image, List[Image.Image]], prompt: str = None, t: float = None) -> torch.Tensor:
        return self.forward(image, prompt, t)
