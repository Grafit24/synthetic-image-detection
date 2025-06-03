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

from .pipelines import AlchemistSanaPipeline



class AlchemistSanaAttnProcessor(SanaAttnProcessor2_0):
    def __init__(self):
        super().__init__()
        self._n_stat = None

    @property
    def n_stat(self) -> torch.Tensor:
        return self._n_stat

    @staticmethod
    def get_v_identical(query, key):
        B, H, _, _ = query.shape
        S = key.shape[-2]
        eye_S = torch.eye(S, device=query.device, dtype=query.dtype) 
        value_identity = eye_S.unsqueeze(0).unsqueeze(0).expand(B, H, S, S)
        return value_identity
    
    def calc_n_stat(self, attn_weights):
        n_stat = torch.norm(attn_weights, dim=(1, 2))
        self._n_stat = n_stat
        self.batch_size = attn_weights.shape[0]

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
        value_identity = self.get_v_identical(query=query, key=key)

        attn_weights = F.scaled_dot_product_attention(
            query, key, value_identity, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = attn_weights @ value

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        self.calc_n_stat(attn_weights)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class AlchemistStatsCollector:
    attn_prc = AlchemistSanaAttnProcessor

    def __init__(self):
        self._processors: Dict[str, AlchemistSanaAttnProcessor] = {}
        self.stats: Dict[str, float] = {}
        self._registered = False
        self._transformer = None
        self._hooks = []
        self._n_samples = 0

    def register_transformer(self, transformer: "SanaTransformer2DModel"):
        """
        Заменяет все SanaAttnProcessor2_0 на AlchemistSanaAttnProcessor и вешает forward-hook,
        который после каждого прохода копит processor.n_stat.
        """
        if self._registered:
            raise RuntimeError("AlchemistStatsCollector: уже был выполнен register_transformer для этой модели.")

        existing_procs = transformer.attn_processors

        full_procs: Dict[str, nn.Module] = {}
        for name, proc in existing_procs.items():
            if isinstance(proc, SanaAttnProcessor2_0):
                new_proc = self.attn_prc()
                full_procs[name] = new_proc
                self._processors[name] = new_proc
                self.stats[name] = []
            else:
                full_procs[name] = proc

        transformer.set_attn_processor(full_procs)

        for full_name in self._processors.keys():
            module_path = full_name.rsplit(".", 1)[0]
            attention_module = self._get_module_by_path(transformer, module_path)

            if not isinstance(attention_module, nn.Module):
                raise RuntimeError(f"Модуль по пути '{module_path}' не является nn.Module.")

            handle = attention_module.register_forward_hook(self._make_hook(full_name))
            self._hooks.append(handle)

        self._transformer = transformer
        self._registered = True

    def _get_module_by_path(self, root_module: nn.Module, path: str) -> nn.Module:
        """
        По строке вида "transformer_blocks.3.attn2" спускаемся по атрибутам из root_module
        и возвращаем конечный модуль.
        """
        parts = path.split(".")
        cur = root_module
        for p in parts:
            if p.isdigit():
                idx = int(p)
                cur = cur[idx]
            else:
                cur = getattr(cur, p)
        return cur

    def _make_hook(self, processor_name: str):
        def hook_fn(module, input_, output):
            value = module.processor.n_stat.detach().cpu()
            self._n_samples += module.processor.batch_size
            self.stats[processor_name].append(value)
        return hook_fn

    def get_collected(self) -> Dict[str, float]:
        return {prc_name: torch.concat(val, dim=0) for prc_name, val in self.stats.items()}

    def clear_stats(self):
        for key in self.stats:
            self.stats[key] = []

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._registered = False
        self._transformer = None
        self._processors.clear()
        self.stats.clear()


class AlchemistDataset(Dataset):
    def __init__(self, file_paths: list[Path], max_image_side=1024):
        self.file_paths = file_paths
        self.max_image_side = max_image_side

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Image.Image:
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        max_side = max(w, h)
        if max_side > self.max_image_side:
            scale = self.max_image_side / max_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
        return image
    
    @staticmethod
    def collate_pil_images(batch: list[Image.Image]) -> list[Image.Image]:
        return batch[0]


class Alchemist:
    stat_collector = AlchemistStatsCollector
    pipeline = AlchemistSanaPipeline
    dataset = AlchemistDataset

    def __init__(self, model_id: str, K: int, prompt: str, t: float = .25):
        self.collector = self.stat_collector()
        try:
            self.pipe = self.pipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        except:
            from transformers import modeling_utils
            if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
                modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']
            self.pipe = self.pipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        finally:
            self.pipe.to("cuda")
            self.pipe.text_encoder = self.pipe.text_encoder.to(torch.bfloat16)
            self.pipe.transformer = self.pipe.transformer.to(torch.bfloat16)
        
        self.collector.register_transformer(self.pipe.transformer)
        self._separation_scores = {}
        self._topk_separation_scores = None
        self.K = K
        self.prompt = prompt
        self.t = t

    @property
    def separation_scores(self):
        if len(self._separation_scores) == 0:
            raise RuntimeError("No separaion scores collected!")
        all_scores: list[tuple[str, int, float]] = []
        for prc_name, s_l in self._separation_scores.items():
            for token_idx in range(s_l.shape[0]):
                all_scores.append((prc_name, token_idx, float(s_l[token_idx].item())))

        all_scores = {
            (prc_name, token_idx): score for (prc_name, token_idx, score) in all_scores
        }
        return all_scores
    
    @property
    def topk_separation_scores(self):
        if self._topk_separation_scores is None:
            raise RuntimeError("No separaion scores collected!")
        return self._topk_separation_scores

    def collect_separation_scores(self, x_positive: list[Path], x_negative: list[Path], max_image_side=1024) -> torch.Tensor:
        x_pos_ds = self.dataset(x_positive, max_image_side=max_image_side)
        x_neg_ds = self.dataset(x_negative, max_image_side=max_image_side)
        pos_loader = DataLoader(
            x_pos_ds,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            collate_fn=x_pos_ds.collate_pil_images
        )
        neg_loader = DataLoader(
            x_neg_ds,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            collate_fn=x_neg_ds.collate_pil_images
        )
        self.collector.clear_stats()
        for imgs in tqdm(pos_loader):
            self.pipe(imgs, prompt=self.prompt, t=self.t)
        
        pos_stats = self.collector.get_collected()

        self.collector.clear_stats()
        for imgs in tqdm(neg_loader):
            self.pipe(imgs, prompt=self.prompt, t=self.t)
        
        neg_stats = self.collector.get_collected()

        assert set(pos_stats) == set(neg_stats)

        for prc_name in pos_stats:
            pos_stat = pos_stats[prc_name]
            neg_stat = neg_stats[prc_name]
            cmp_matrix = pos_stat.unsqueeze(1) > neg_stat.unsqueeze(0)
            s_l = cmp_matrix.sum(dim=(0, 1)) 
            self._separation_scores[prc_name] = s_l

        self._topk_separation_scores = self.get_topK_separation_scores()

        return self._separation_scores
    
    def get_topK_separation_scores(self, K: int = None) -> list[dict[tuple[str, int], float]]:
        all_scores = list(self.separation_scores.items())
        all_scores.sort(key=lambda x: x[1], reverse=True)
        K = K or self.K
        topk = all_scores[:K]

        result = { (prc_name, token_idx): score for (prc_name, token_idx), score in topk}

        return result
    
    def score(self, img: Image.Image):
        self.collector.clear_stats()
        self.pipe(img, prompt=self.prompt, t=self.t)
        stats = self.collector.get_collected()
        score = 0
        for (prc_name, token_idx), _ in self.topk_separation_scores.items():
            score += stats[prc_name].squeeze(0)[token_idx]
        return score