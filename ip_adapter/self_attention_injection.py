from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Callable, Dict, Iterable, Literal, Tuple

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from PIL import Image


DEFAULT_INJECTION_LAYERS = tuple(
    f"up_blocks.{block}.attentions.{attention}.transformer_blocks.0.attn1.processor"
    for block in (2, 3)
    for attention in range(3)
)

FeatureKey = Tuple[str, int]


@dataclass
class QKVFeatureBank:
    """CPU-backed feature storage for one content/style inversion pair."""

    content_q: Dict[FeatureKey, torch.Tensor] = field(default_factory=dict)
    style_k: Dict[FeatureKey, torch.Tensor] = field(default_factory=dict)
    style_v: Dict[FeatureKey, torch.Tensor] = field(default_factory=dict)
    write_observer: Callable[[dict], None] | None = field(default=None, repr=False)

    def clear(self) -> None:
        self.content_q.clear()
        self.style_k.clear()
        self.style_v.clear()

    def _store(
        self,
        tensor: torch.Tensor,
        feature_type: str,
        layer_name: str,
        timestep: int,
    ) -> torch.Tensor:
        started = time.perf_counter()
        stored = tensor.detach().to(device="cpu", dtype=torch.float16).contiguous()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if self.write_observer is not None:
            self.write_observer(
                {
                    "feature_type": feature_type,
                    "layer_name": layer_name,
                    "timestep": int(timestep),
                    "source_shape": list(tensor.shape),
                    "source_dtype": str(tensor.dtype),
                    "source_device": str(tensor.device),
                    "cache_shape": list(stored.shape),
                    "cache_dtype": str(stored.dtype),
                    "cache_device": str(stored.device),
                    "bytes": stored.numel() * stored.element_size(),
                    "write_ms": elapsed_ms,
                    "gpu_allocated_mib": (
                        torch.cuda.memory_allocated(tensor.device) / 1024**2
                        if tensor.is_cuda
                        else 0.0
                    ),
                }
            )
        return stored

    def record_content(self, layer_name: str, timestep: int, query: torch.Tensor) -> None:
        self.content_q[(layer_name, timestep)] = self._store(
            query, "Q_c", layer_name, timestep
        )

    def record_style(
        self,
        layer_name: str,
        timestep: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        feature_key = (layer_name, timestep)
        self.style_k[feature_key] = self._store(key, "K_s", layer_name, timestep)
        self.style_v[feature_key] = self._store(value, "V_s", layer_name, timestep)

    def summary(self) -> dict:
        return {
            "content_q": len(self.content_q),
            "style_k": len(self.style_k),
            "style_v": len(self.style_v),
        }


class SelfAttentionQKVController:
    """Coordinates QKV recording during inversion and KV injection during generation."""

    def __init__(self, layer_names: Iterable[str] = DEFAULT_INJECTION_LAYERS):
        self.layer_names = tuple(layer_names)
        self.bank = QKVFeatureBank()
        self.mode: Literal["disabled", "record_content", "record_style", "inject"] = "disabled"
        self.timestep: int | None = None
        self.step_index: int | None = None
        self.lambda_q = 0.7
        self.missing_features: set[FeatureKey] = set()
        self.total_steps: int | None = None
        self.injection_start = 0.0
        self.injection_end = 1.0
        self.applied_steps: set[int] = set()
        self.record_timesteps: set[int] | None = None
        self.injection_observer: Callable[[dict], None] | None = None

    def install(self, unet) -> None:
        processors = dict(unet.attn_processors)
        missing = [name for name in self.layer_names if name not in processors]
        if missing:
            raise ValueError("UNet 中找不到自注意力注入层：\n" + "\n".join(missing))
        for name in self.layer_names:
            processors[name] = QKVInjectionAttnProcessor2_0(self, name)
        unet.set_attn_processor(processors)

    def clear(self) -> None:
        self.bank.clear()
        self.missing_features.clear()
        self.applied_steps.clear()
        self.record_timesteps = None
        self.disable()

    def set_step(self, step_index: int, timestep: int | torch.Tensor) -> None:
        self.step_index = int(step_index)
        self.timestep = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)

    def record_content(self) -> None:
        self.mode = "record_content"

    def record_style(self) -> None:
        self.mode = "record_style"

    def configure_injection_window(
        self,
        total_steps: int,
        injection_start: float = 0.0,
        injection_end: float = 1.0,
    ) -> None:
        if int(total_steps) < 1:
            raise ValueError("total_steps 必须大于 0")
        if not 0.0 <= float(injection_start) <= float(injection_end) <= 1.0:
            raise ValueError("自注意力注入时间窗必须满足 0 <= start <= end <= 1")
        self.total_steps = int(total_steps)
        self.injection_start = float(injection_start)
        self.injection_end = float(injection_end)

    def prepare_record_timesteps(self, generation_timesteps: torch.Tensor) -> None:
        total_steps = len(generation_timesteps)
        start_step = min(total_steps, math.floor(self.injection_start * total_steps))
        end_step = min(total_steps, math.ceil(self.injection_end * total_steps))
        self.record_timesteps = {
            int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
            for timestep in generation_timesteps[start_step:end_step]
        }

    @property
    def recording_active(self) -> bool:
        return self.record_timesteps is None or (
            self.timestep is not None and self.timestep in self.record_timesteps
        )

    def inject(
        self,
        lambda_q: float,
        total_steps: int | None = None,
        injection_start: float = 0.0,
        injection_end: float = 1.0,
    ) -> None:
        if not 0.0 <= float(lambda_q) <= 1.0:
            raise ValueError("lambda_q 必须位于 [0, 1]")
        if total_steps is not None:
            self.configure_injection_window(total_steps, injection_start, injection_end)
        self.lambda_q = float(lambda_q)
        self.missing_features.clear()
        self.applied_steps.clear()
        self.mode = "inject"

    @property
    def injection_active(self) -> bool:
        if self.mode != "inject":
            return False
        if self.total_steps is None or self.step_index is None:
            return True
        start_step = min(
            self.total_steps,
            math.floor(self.injection_start * self.total_steps),
        )
        end_step = min(
            self.total_steps,
            math.ceil(self.injection_end * self.total_steps),
        )
        return start_step <= self.step_index < end_step

    def disable(self) -> None:
        self.mode = "disabled"
        self.timestep = None
        self.step_index = None

    @staticmethod
    def _expand_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        if tensor.shape[0] == batch_size:
            return tensor
        if batch_size % tensor.shape[0] != 0:
            raise ValueError(
                f"记录特征 batch={tensor.shape[0]} 无法扩展到生成 batch={batch_size}"
            )
        return tensor.repeat_interleave(batch_size // tensor.shape[0], dim=0)

    def feature(
        self,
        storage: Dict[FeatureKey, torch.Tensor],
        layer_name: str,
        reference: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.timestep is None:
            return None
        feature_key = (layer_name, self.timestep)
        tensor = storage.get(feature_key)
        if tensor is None:
            self.missing_features.add(feature_key)
            return None
        tensor = tensor.to(device=reference.device, dtype=reference.dtype)
        tensor = self._expand_batch(tensor, reference.shape[0])
        if tensor.shape[1:] != reference.shape[1:]:
            raise ValueError(
                f"{layer_name} 在 timestep={self.timestep} 的特征尺寸不一致："
                f"记录 {tuple(tensor.shape)}，当前 {tuple(reference.shape)}。"
                "请确保内容图、风格图和生成分辨率一致。"
            )
        return tensor


class QKVInjectionAttnProcessor2_0(torch.nn.Module):
    """SDPA self-attention processor with inversion recording and style KV injection."""

    def __init__(self, controller: SelfAttentionQKVController, layer_name: str):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("QKV injection requires torch scaled_dot_product_attention")
        self.controller = controller
        self.layer_name = layer_name

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]
            channel = height = width = None

        sequence_length = hidden_states.shape[1]
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        source_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if encoder_hidden_states is not None and attn.norm_cross:
            source_states = attn.norm_encoder_hidden_states(source_states)
        key = attn.to_k(source_states)
        value = attn.to_v(source_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        timestep = self.controller.timestep
        injection_profile = None
        if timestep is not None:
            if self.controller.mode == "record_content" and self.controller.recording_active:
                self.controller.bank.record_content(self.layer_name, timestep, query)
            elif self.controller.mode == "record_style" and self.controller.recording_active:
                self.controller.bank.record_style(self.layer_name, timestep, key, value)
            elif self.controller.mode == "inject" and self.controller.injection_active:
                if self.controller.step_index is not None:
                    self.controller.applied_steps.add(self.controller.step_index)
                profile_enabled = self.controller.injection_observer is not None and query.is_cuda
                if profile_enabled:
                    total_start = torch.cuda.Event(enable_timing=True)
                    transfer_start = torch.cuda.Event(enable_timing=True)
                    transfer_end = torch.cuda.Event(enable_timing=True)
                    total_start.record()
                    transfer_start.record()
                content_query = self.controller.feature(
                    self.controller.bank.content_q, self.layer_name, query
                )
                style_key = self.controller.feature(
                    self.controller.bank.style_k, self.layer_name, key
                )
                style_value = self.controller.feature(
                    self.controller.bank.style_v, self.layer_name, value
                )
                if profile_enabled:
                    transfer_end.record()
                if content_query is not None and style_key is not None and style_value is not None:
                    query = self.controller.lambda_q * query + (1.0 - self.controller.lambda_q) * content_query
                    key = style_key
                    value = style_value
                    if profile_enabled:
                        attention_start = torch.cuda.Event(enable_timing=True)
                        attention_end = torch.cuda.Event(enable_timing=True)
                        total_end = torch.cuda.Event(enable_timing=True)
                        attention_start.record()
                        feature_key = (self.layer_name, timestep)
                        cached_tensors = (
                            self.controller.bank.content_q.get(feature_key),
                            self.controller.bank.style_k.get(feature_key),
                            self.controller.bank.style_v.get(feature_key),
                        )
                        injection_profile = {
                            "layer_name": self.layer_name,
                            "timestep": int(timestep),
                            "step_index": self.controller.step_index,
                            "lambda_q": self.controller.lambda_q,
                            "mode": "replace_style_kv_mix_content_q",
                            "query_shape": list(query.shape),
                            "key_shape": list(key.shape),
                            "value_shape": list(value.shape),
                            "dtype": str(query.dtype),
                            "cache_location": "cpu",
                            "gpu_allocated_mib": torch.cuda.memory_allocated(query.device) / 1024**2,
                            "read_bytes": sum(
                                item.numel() * item.element_size()
                                for item in cached_tensors
                                if item is not None
                            ),
                            "events": {
                                "total_start": total_start,
                                "transfer_start": transfer_start,
                                "transfer_end": transfer_end,
                                "attention_start": attention_start,
                                "attention_end": attention_end,
                                "total_end": total_end,
                            },
                        }

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        if injection_profile is not None:
            injection_profile["events"]["attention_end"].record()
            injection_profile["events"]["total_end"].record()
            self.controller.injection_observer(injection_profile)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def adaptive_instance_normalization(
    content_latent: torch.Tensor,
    style_latent: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if content_latent.shape != style_latent.shape:
        raise ValueError(
            f"AdaIN latent 尺寸不一致：content={tuple(content_latent.shape)}, "
            f"style={tuple(style_latent.shape)}"
        )
    content_mean = content_latent.mean(dim=(2, 3), keepdim=True)
    content_std = content_latent.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(eps)
    style_mean = style_latent.mean(dim=(2, 3), keepdim=True)
    style_std = style_latent.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(eps)
    return (content_latent - content_mean) / content_std * style_std + style_mean


def blend_adain_latent(
    content_latent: torch.Tensor,
    style_latent: torch.Tensor,
    strength: float = 1.0,
) -> torch.Tensor:
    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("AdaIN strength 必须位于 [0, 1]")
    adain_latent = adaptive_instance_normalization(content_latent, style_latent)
    return content_latent.lerp(adain_latent, strength)


class DDIMQKVInverter:
    """Deterministic DDIM inversion that records features at generation timesteps."""

    def __init__(self, pipe, controller: SelfAttentionQKVController):
        self.pipe = pipe
        self.controller = controller

    def make_scheduler(self) -> DDIMScheduler:
        return DDIMScheduler.from_config(
            self.pipe.scheduler.config,
            clip_sample=False,
            set_alpha_to_one=True,
        )

    @staticmethod
    def _next_latent(
        sample: torch.Tensor,
        noise_pred: torch.Tensor,
        alpha_current: torch.Tensor,
        alpha_next: torch.Tensor,
    ) -> torch.Tensor:
        pred_original = (sample - (1.0 - alpha_current).sqrt() * noise_pred) / alpha_current.sqrt()
        return alpha_next.sqrt() * pred_original + (1.0 - alpha_next).sqrt() * noise_pred

    def _prompt_embeddings(self, prompt: str, dtype: torch.dtype) -> torch.Tensor:
        prompt_embeds, _ = self.pipe.encode_prompt(
            prompt,
            device=self.pipe._execution_device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
        )
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        # StyleShot's attn2 processors reserve the last six tokens for style.
        zero_style_tokens = torch.zeros(
            (prompt_embeds.shape[0], 6, prompt_embeds.shape[-1]),
            device=prompt_embeds.device,
            dtype=prompt_embeds.dtype,
        )
        return torch.cat([prompt_embeds, zero_style_tokens], dim=1)

    @torch.inference_mode()
    def invert(
        self,
        image: Image.Image,
        num_inference_steps: int,
        target: Literal["content", "style"],
        prompt: str = "",
    ) -> torch.Tensor:
        if target not in ("content", "style"):
            raise ValueError("target 必须是 content 或 style")
        device = self.pipe._execution_device
        dtype = self.pipe.unet.dtype
        image_tensor = self.pipe.image_processor.preprocess(image).to(device=device, dtype=dtype)
        latent = self.pipe.vae.encode(image_tensor).latent_dist.mode()
        latent = latent * self.pipe.vae.config.scaling_factor

        scheduler = self.make_scheduler()
        scheduler.set_timesteps(num_inference_steps, device=device)
        self.controller.prepare_record_timesteps(scheduler.timesteps)
        ascending_timesteps = list(reversed(scheduler.timesteps))
        alphas = scheduler.alphas_cumprod.to(device=device, dtype=dtype)
        prompt_embeds = self._prompt_embeddings(prompt, dtype)

        # Start from clean latent (alpha=1), then visit every timestep used by
        # generation. The UNet call after each update records features at x_t.
        current_alpha = torch.ones((), device=device, dtype=dtype)
        initial_timestep = torch.zeros((), device=device, dtype=torch.long)
        self.controller.disable()
        self.controller.set_step(-1, 0)
        noise_pred = self.pipe.unet(
            latent,
            initial_timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        if target == "content":
            self.controller.record_content()
        else:
            self.controller.record_style()

        for step_index, timestep in enumerate(ascending_timesteps):
            timestep_int = int(timestep.item())
            next_alpha = alphas[timestep_int]
            latent = self._next_latent(latent, noise_pred, current_alpha, next_alpha)
            current_alpha = next_alpha
            self.controller.set_step(step_index, timestep_int)
            noise_pred = self.pipe.unet(
                latent,
                timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

        self.controller.disable()
        return latent.detach()
