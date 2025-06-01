from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from tqdm import tqdm

from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.sana_transformer import SanaAttnProcessor2_0
from diffusers.models.transformers.sana_transformer import SanaTransformer2DModel
import torch
from diffusers.pipelines.sana import SanaPipeline


class SanaClassifierHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dims=[256, 128, 64], num_classes=1, drop_p=0.2):
        super().__init__()
        layers = []

        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(drop_p))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)


class CrossAttentionProjection(nn.Module):
    def __init__(self, d_model, proj_d):
        super().__init__()
        self.proj = nn.Linear(d_model, proj_d)

    def forward(x: torch.Tensor) -> torch.Tensor:
        b, ... = x.shape
        x = x.view(...)
        x = self.proj(x)
        return x


class SanaClassifier(nn.Module):
    def __init__(self, d_model: int, num_layers: int, proj_dim=256, hidden_dims=[256, 128, 64], num_classes=1, drop_p=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.clf_head = self.SanaClassifierHead(
            input_dim=proj_dim, 
            hidden_dims=hidden_dims, 
            num_classes=num_classes, 
            drop_p=drop_p
        )
        init_vals = torch.full((num_layers,), 1.0 / num_layers)
        self.alpha = nn.Parameter(init_vals)
        self.projections = nn.ModuleList([
            nn.Linear(d_model, proj_dim) for _ in range(num_layers)
        ])

    def forward(self, outputs_from_cross_attns):
        pooled_feats = []
        for i, feat in enumerate(outputs_from_transformer):
            # Не знаю как пока в начале обработать
            # f = feat.mean(dim=1)
            h = self.projections[i](f)
            pooled_feats.append(h)

        alphas = torch.softmax(self.alpha, dim=0)
        H = sum(alphas[i] * pooled_feats[i] for i in range(self.num_layers))
        logits = self.classifier(H)
        return logits


class ClassifierSanaAttnProcessor(SanaAttnProcessor2_0):
    def __init__(self, proj: CrossAttentionProjection):
        super().__init__()
        self.proj_out = None
        self.proj = proj

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
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
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

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        
        self.proj_out = self.proj(hidden_states).to(query.dtype)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class ClassifierSanaPipeline(SanaPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clf_mlp = nn.Identity()
        self.hidden_states = []

    def forward(self, x):
        # collect hidden states from hooks
        # aggreate features
        # get logits
        ...

    def init_clf_mlp(self, hidden_size: int = 64, dropout: float = 0.1):
        ...

    def register_model(self, transformer_blocks_ids):
        ...

    def _make_hook(self, processor_name: str):
        def hook_fn(module, input_, output):
            value = module.processor.attn_states
            self.hidden_states.append(value)
        return hook_fn

    @torch.no_grad()
    def hooks_collect(self, image, prompt: str, t: float):
        device = self._execution_device

        pixel_values = self.image_processor.preprocess(image).to(device)
        latents = self.vae.encode(pixel_values).latent

        prompt_embeds, prompt_attention_mask, _, _ = self.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=False,
            negative_prompt="",
            num_images_per_prompt=1,
            device=device,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            prompt_attention_mask=None,
            negative_prompt_attention_mask=None,
            clean_caption=False,
            max_sequence_length=300,
            complex_human_instruction=None,
            lora_scale=None,
        )

        num_train_timesteps = self.scheduler.config.num_train_timesteps
        t_clamped = float(t)
        t_clamped = max(0.0, min(1.0, t_clamped))
        timestep_idx = int(t_clamped * (num_train_timesteps - 1))

        timestep_tensor = torch.tensor([timestep_idx], device=device)

        noise = torch.randn_like(latents, device=device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timestep_tensor)
        self.noisy_latent = noisy_latents

        latent_model_input = noisy_latents.to(self.transformer.dtype)

        timestep_tensor = torch.tensor([timestep_idx], device=device, dtype=latent_model_input.dtype)
        timestep_tensor = timestep_tensor * self.transformer.config.timestep_scale

        noise_pred = self.transformer(
            latent_model_input,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            timestep=timestep_tensor,
            return_dict=False,
            attention_kwargs=None
        )[0]

        self.noise_pred = noise_pred