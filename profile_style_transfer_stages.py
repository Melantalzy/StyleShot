from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil
import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image, ImageOps
from transformers import CLIPTextModel, CLIPTokenizer

from ip_adapter.attention_processor import IPAttnProcessor2_0
from ip_adapter.ip_adapter import StyleProcessor, StyleShot
from ip_adapter.self_attention_injection import (
    SelfAttentionQKVController,
    adaptive_instance_normalization,
)


ROOT = Path(__file__).resolve().parent
BASE_MODEL = ROOT / "runwayml" / "stable-diffusion-v1-5"
CLIP_VISION_MODEL = ROOT / "laion" / "CLIP-ViT-H-14-laion2B-s32B-b79K"
STYLESHOT_MODEL = ROOT / "Gaojunyao" / "StyleShot" / "pretrained_weight"
OUTPUT_ROOT = ROOT / "outputs" / "stage_cost_profiles"

STAGE_NAMES = {
    "01": "输入准备",
    "02": "模型权重加载或卸载",
    "03": "文本编码",
    "04": "初始噪声采样",
    "05": "内容图VAE编码",
    "06": "风格图VAE编码",
    "07": "内容图反演",
    "08": "风格图反演",
    "09": "参考特征提取与缓存写入",
    "10": "风格图像编码与适配器投影",
    "11": "AdaIN起点构造",
    "12": "生成去噪",
    "13": "自注意力KV注入",
    "14": "适配器交叉注意力注入",
    "15": "VAE解码",
    "16": "输出回传与后处理",
    "17": "整次端到端运行",
}

GROUP_INFO = {
    "A": {"name": "纯文本生成", "expected_unet_calls": 50},
    "B": {"name": "普通反演重建", "expected_unet_calls": 100},
    "C": {"name": "KV注入迁移", "expected_unet_calls": 150},
    "D": {"name": "适配器迁移_随机起点", "expected_unet_calls": 50},
    "E": {"name": "适配器迁移_AdaIN起点", "expected_unet_calls": 150},
    "F": {"name": "KV加适配器双路迁移", "expected_unet_calls": 150},
}

CSV_FIELDS = [
    "run_id",
    "group",
    "group_name",
    "stage_code",
    "stage_name",
    "occurrence",
    "parent_stage",
    "started_at",
    "duration_ms",
    "gpu_start_mib",
    "gpu_peak_mib",
    "gpu_end_mib",
    "gpu_reserved_start_mib",
    "gpu_reserved_peak_mib",
    "gpu_reserved_end_mib",
    "cpu_start_mib",
    "cpu_end_mib",
    "details_json",
]


def mib(value: int | float) -> float:
    return round(float(value) / 1024**2, 4)


def module_bytes(module: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        pointer = tensor.data_ptr()
        if pointer and pointer not in seen:
            seen.add(pointer)
            total += tensor.numel() * tensor.element_size()
    return total


def folder_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def tensor_id(tensor: torch.Tensor) -> str:
    array = tensor.detach().to(device="cpu", dtype=torch.float16).contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def file_id(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class ResourceMonitor:
    def __init__(self, recorder: "StageRecorder", interval: float = 0.005):
        self.recorder = recorder
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.cpu_peak_bytes = 0
        self.gpu_peak_bytes = 0
        self.gpu_peak_stage = None

    def start(self) -> None:
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            try:
                cpu = self.process.memory_info().rss
                gpu = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                self.cpu_peak_bytes = max(self.cpu_peak_bytes, cpu)
                if gpu > self.gpu_peak_bytes:
                    self.gpu_peak_bytes = gpu
                    self.gpu_peak_stage = self.recorder.current_stage
            except Exception:
                pass
            self.stop_event.wait(self.interval)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)


class StageRecorder:
    def __init__(self, run_id: str, group: str):
        self.run_id = run_id
        self.group = group
        self.rows: list[dict] = []
        self.current_stage: str | None = None
        self.counts: dict[str, int] = {}
        self.process = psutil.Process(os.getpid())

    def _gpu(self) -> tuple[int, int]:
        if not torch.cuda.is_available():
            return 0, 0
        return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()

    def _append(
        self,
        code: str,
        duration_ms: float,
        gpu_start: int,
        gpu_peak: int,
        gpu_end: int,
        reserved_start: int,
        reserved_peak: int,
        reserved_end: int,
        cpu_start: int,
        cpu_end: int,
        details: dict,
        parent_stage: str | None,
        started_at: str,
    ) -> dict:
        self.counts[code] = self.counts.get(code, 0) + 1
        row = {
            "run_id": self.run_id,
            "group": self.group,
            "group_name": GROUP_INFO[self.group]["name"],
            "stage_code": code,
            "stage_name": STAGE_NAMES[code],
            "occurrence": self.counts[code],
            "parent_stage": parent_stage or "",
            "started_at": started_at,
            "duration_ms": round(float(duration_ms), 4),
            "gpu_start_mib": mib(gpu_start),
            "gpu_peak_mib": mib(gpu_peak),
            "gpu_end_mib": mib(gpu_end),
            "gpu_reserved_start_mib": mib(reserved_start),
            "gpu_reserved_peak_mib": mib(reserved_peak),
            "gpu_reserved_end_mib": mib(reserved_end),
            "cpu_start_mib": mib(cpu_start),
            "cpu_end_mib": mib(cpu_end),
            "details": json_safe(details),
        }
        self.rows.append(row)
        return row

    @contextmanager
    def stage(self, code: str, details: dict | None = None, parent_stage: str | None = None):
        details = details if details is not None else {}
        previous = self.current_stage
        self.current_stage = code
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        gpu_start, reserved_start = self._gpu()
        cpu_start = self.process.memory_info().rss
        started_at = datetime.now().isoformat(timespec="milliseconds")
        started = time.perf_counter()
        try:
            yield details
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gpu_peak = torch.cuda.max_memory_allocated()
                reserved_peak = torch.cuda.max_memory_reserved()
            else:
                gpu_peak = gpu_start
                reserved_peak = reserved_start
            duration_ms = (time.perf_counter() - started) * 1000.0
            gpu_end, reserved_end = self._gpu()
            cpu_end = self.process.memory_info().rss
            self._append(
                code,
                duration_ms,
                gpu_start,
                gpu_peak,
                gpu_end,
                reserved_start,
                reserved_peak,
                reserved_end,
                cpu_start,
                cpu_end,
                details,
                parent_stage,
                started_at,
            )
            self.current_stage = previous

    def aggregate(
        self,
        code: str,
        duration_ms: float,
        gpu_values_mib: list[float],
        details: dict,
        parent_stage: str | None,
    ) -> dict:
        values = gpu_values_mib or [mib(torch.cuda.memory_allocated())]
        cpu = self.process.memory_info().rss
        return self._append(
            code,
            duration_ms,
            int(values[0] * 1024**2),
            int(max(values) * 1024**2),
            int(values[-1] * 1024**2),
            torch.cuda.memory_reserved(),
            torch.cuda.memory_reserved(),
            torch.cuda.memory_reserved(),
            cpu,
            cpu,
            details,
            parent_stage,
            datetime.now().isoformat(timespec="milliseconds"),
        )

    def write(self, folder: Path) -> None:
        json_rows = []
        for row in self.rows:
            item = dict(row)
            item["details_json"] = json.dumps(item.pop("details"), ensure_ascii=False)
            json_rows.append(item)
        with (folder / "stages.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(json_rows)
        (folder / "stages.json").write_text(
            json.dumps(self.rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )


@dataclass
class Runtime:
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    scheduler: DDIMScheduler
    pipe: StableDiffusionPipeline
    style_model: StyleShot | None
    controller: SelfAttentionQKVController | None
    device: torch.device
    dtype: torch.dtype


def model_device(module: torch.nn.Module) -> str:
    try:
        return str(next(module.parameters()).device)
    except StopIteration:
        return "none"


def load_component(
    recorder: StageRecorder,
    module_name: str,
    loader: Callable[[], Any],
    byte_hint: int | None = None,
) -> Any:
    details = {
        "module": module_name,
        "direction": "disk_to_cpu",
        "device_before": "disk",
        "device_after": "cpu",
        "unet_resident_until_denoise_end": module_name == "unet",
    }
    with recorder.stage("02", details):
        component = loader()
        if isinstance(component, torch.nn.Module):
            details["bytes"] = module_bytes(component)
        elif byte_hint is not None:
            details["bytes"] = byte_hint
        else:
            details["bytes"] = 0
    return component


def move_module(
    recorder: StageRecorder,
    name: str,
    module: torch.nn.Module,
    device: torch.device | str,
    dtype: torch.dtype | None,
    direction: str,
) -> None:
    details = {
        "module": name,
        "direction": direction,
        "bytes": module_bytes(module),
        "device_before": model_device(module),
        "device_after": str(device),
        "unet_resident_until_denoise_end": name == "unet" and direction == "cpu_to_gpu",
    }
    with recorder.stage("02", details):
        if dtype is None:
            module.to(device=device)
        else:
            module.to(device=device, dtype=dtype)
        details["device_after"] = model_device(module)


def build_runtime(group: str, recorder: StageRecorder, device: torch.device, dtype: torch.dtype) -> Runtime:
    tokenizer = load_component(
        recorder,
        "tokenizer",
        lambda: CLIPTokenizer.from_pretrained(
            str(BASE_MODEL), subfolder="tokenizer", local_files_only=True
        ),
        byte_hint=folder_bytes(BASE_MODEL / "tokenizer"),
    )
    scheduler = load_component(
        recorder,
        "ddim_scheduler",
        lambda: DDIMScheduler.from_pretrained(
            str(BASE_MODEL), subfolder="scheduler", local_files_only=True
        ),
        byte_hint=folder_bytes(BASE_MODEL / "scheduler"),
    )
    text_encoder = load_component(
        recorder,
        "text_encoder",
        lambda: CLIPTextModel.from_pretrained(
            str(BASE_MODEL),
            subfolder="text_encoder",
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ),
    )
    vae = load_component(
        recorder,
        "vae",
        lambda: AutoencoderKL.from_pretrained(
            str(BASE_MODEL),
            subfolder="vae",
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ),
    )
    unet = load_component(
        recorder,
        "unet",
        lambda: UNet2DConditionModel.from_pretrained(
            str(BASE_MODEL),
            subfolder="unet",
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ),
    )

    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        image_encoder=None,
        requires_safety_checker=False,
    )

    style_model = None
    if group in {"D", "E", "F"}:
        before_unet_bytes = module_bytes(unet)
        details = {
            "module": "styleshot_adapter_bundle",
            "direction": "disk_to_cpu",
            "device_before": "disk",
            "device_after": "cpu",
        }
        with recorder.stage("02", details):
            style_model = StyleShot(
                torch.device("cpu"),
                pipe,
                str(STYLESHOT_MODEL / "ip.bin"),
                str(STYLESHOT_MODEL / "style_aware_encoder.bin"),
                str(CLIP_VISION_MODEL),
                weight_dtype=dtype,
            )
            details["bytes"] = (
                module_bytes(unet)
                - before_unet_bytes
                + module_bytes(style_model.style_aware_encoder)
                + module_bytes(style_model.style_image_proj_modules)
            )

    controller = None
    if group in {"C", "F"}:
        controller = SelfAttentionQKVController()
        controller.install(unet)

    move_module(recorder, "text_encoder", text_encoder, device, dtype, "cpu_to_gpu")
    move_module(recorder, "vae", vae, device, dtype, "cpu_to_gpu")
    move_module(recorder, "unet", unet, device, dtype, "cpu_to_gpu")
    if style_model is not None:
        move_module(
            recorder,
            "style_aware_encoder",
            style_model.style_aware_encoder,
            device,
            dtype,
            "cpu_to_gpu",
        )
        move_module(
            recorder,
            "style_image_projection",
            style_model.style_image_proj_modules,
            device,
            dtype,
            "cpu_to_gpu",
        )
        style_model.device = device
        style_model.pipe = pipe

    text_encoder.eval()
    vae.eval()
    unet.eval()
    if style_model is not None:
        style_model.eval()
        style_model.style_aware_encoder.eval()
        style_model.style_image_proj_modules.eval()

    return Runtime(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        pipe=pipe,
        style_model=style_model,
        controller=controller,
        device=device,
        dtype=dtype,
    )


def unload_runtime(runtime: Runtime, recorder: StageRecorder) -> None:
    if runtime.style_model is not None:
        move_module(
            recorder,
            "style_image_projection",
            runtime.style_model.style_image_proj_modules,
            "cpu",
            runtime.dtype,
            "gpu_to_cpu",
        )
        move_module(
            recorder,
            "style_aware_encoder",
            runtime.style_model.style_aware_encoder,
            "cpu",
            runtime.dtype,
            "gpu_to_cpu",
        )
    move_module(recorder, "unet", runtime.unet, "cpu", runtime.dtype, "gpu_to_cpu")
    move_module(recorder, "vae", runtime.vae, "cpu", runtime.dtype, "gpu_to_cpu")
    move_module(
        recorder, "text_encoder", runtime.text_encoder, "cpu", runtime.dtype, "gpu_to_cpu"
    )
    details = {
        "module": "cuda_cached_blocks",
        "direction": "release_gpu_cache",
        "bytes": torch.cuda.memory_reserved(),
        "device_before": "cuda",
        "device_after": "released",
        "unet_resident_until_denoise_end": True,
    }
    with recorder.stage("02", details):
        runtime.controller = None
        runtime.style_model = None
        gc.collect()
        torch.cuda.empty_cache()


def prepare_pil(image: Image.Image, resolution: int) -> Image.Image:
    return ImageOps.fit(
        ImageOps.exif_transpose(image).convert("RGB"),
        (resolution, resolution),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


def pil_to_vae_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def encode_text(runtime: Runtime, prompt: str, negative_prompt: str, recorder: StageRecorder):
    details: dict = {}
    with recorder.stage("03", details):
        prompts = [negative_prompt, prompt]
        inputs = runtime.tokenizer(
            prompts,
            padding="max_length",
            max_length=runtime.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(runtime.device)
        with torch.inference_mode():
            embeds = runtime.text_encoder(input_ids)[0].to(dtype=runtime.dtype)
        uncond, cond = embeds.chunk(2)
        token_counts = (input_ids != runtime.tokenizer.pad_token_id).sum(dim=1).tolist()
        details.update(
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "token_counts_uncond_cond": token_counts,
                "feature_shape_each": list(cond.shape),
                "dtype": str(cond.dtype),
                "cache_hit": False,
                "shared_by_inversion_and_denoising": True,
                "cfg_unconditional_branch": True,
            }
        )
    return cond, uncond


def vae_encode(
    runtime: Runtime,
    image_tensor_cpu: torch.Tensor,
    recorder: StageRecorder,
    code: str,
    image_id: str,
) -> torch.Tensor:
    details: dict = {"image_id": image_id}
    with recorder.stage(code, details):
        tensor = image_tensor_cpu.to(runtime.device, dtype=runtime.dtype)
        with torch.inference_mode():
            posterior = runtime.vae.encode(tensor).latent_dist
            latent = posterior.mode() * runtime.vae.config.scaling_factor
        details.update(
            {
                "input_shape": list(tensor.shape),
                "latent_shape": list(latent.shape),
                "dtype": str(latent.dtype),
                "posterior_selection": "mode",
                "batched_with_other_image": False,
                "actual_batch": int(tensor.shape[0]),
                "latent_bytes": tensor_bytes(latent),
            }
        )
    return latent


def set_adapter_scale(runtime: Runtime, scale: float) -> None:
    for processor in runtime.unet.attn_processors.values():
        if hasattr(processor, "to_k_ip") and hasattr(processor, "scale"):
            processor.scale = float(scale)


def inversion_embeddings(runtime: Runtime, cond: torch.Tensor) -> torch.Tensor:
    if runtime.style_model is None:
        return cond
    zeros = torch.zeros(
        (cond.shape[0], 6, cond.shape[-1]),
        device=cond.device,
        dtype=cond.dtype,
    )
    return torch.cat([cond, zeros], dim=1)


@torch.inference_mode()
def invert_latent(
    runtime: Runtime,
    latent_z0: torch.Tensor,
    cond: torch.Tensor,
    steps: int,
    recorder: StageRecorder,
    stage_code: str,
    target: str,
    unet_counter: dict,
) -> torch.Tensor:
    details: dict = {}
    scheduler = DDIMScheduler.from_config(
        runtime.scheduler.config, clip_sample=False, set_alpha_to_one=True
    )
    scheduler.set_timesteps(steps, device=runtime.device)
    ascending = list(reversed(scheduler.timesteps))
    alphas = scheduler.alphas_cumprod.to(device=runtime.device, dtype=runtime.dtype)
    embeddings = inversion_embeddings(runtime, cond)
    set_adapter_scale(runtime, 0.0)

    controller = runtime.controller
    if controller is not None:
        controller.configure_injection_window(steps, 0.0, 1.0)
        controller.prepare_record_timesteps(scheduler.timesteps)
        if target == "content":
            controller.record_content()
        elif target == "style":
            controller.record_style()
        else:
            controller.disable()

    latent = latent_z0.clone()
    per_step_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    with recorder.stage(stage_code, details):
        for ascending_index, timestep in enumerate(ascending):
            generation_index = steps - 1 - ascending_index
            if controller is not None:
                controller.set_step(generation_index, timestep)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            noise_pred = runtime.unet(
                latent,
                timestep,
                encoder_hidden_states=embeddings,
                return_dict=False,
            )[0]
            end.record()
            per_step_events.append((start, end))
            unet_counter["inversion"] += 1
            if ascending_index + 1 < len(ascending):
                current_t = int(timestep.item())
                next_t = int(ascending[ascending_index + 1].item())
                alpha_current = alphas[current_t]
                alpha_next = alphas[next_t]
                pred_original = (
                    latent - (1.0 - alpha_current).sqrt() * noise_pred
                ) / alpha_current.sqrt()
                latent = (
                    alpha_next.sqrt() * pred_original
                    + (1.0 - alpha_next).sqrt() * noise_pred
                )
        torch.cuda.synchronize()
        step_times = [start.elapsed_time(end) for start, end in per_step_events]
        if controller is not None:
            controller.disable()
        details.update(
            {
                "target": target,
                "steps": steps,
                "unet_forward_calls": len(per_step_events),
                "effective_batch": 1,
                "terminal_shape": list(latent.shape),
                "terminal_bytes": tensor_bytes(latent),
                "per_step_ms": [round(value, 4) for value in step_times],
                "q_capture": target == "content" and controller is not None,
                "kv_capture": target == "style" and controller is not None,
                "capture_layers": list(controller.layer_names) if controller is not None else [],
                "inversion_convention": "N-call DDIM; z0 is treated as the lowest scheduled timestep",
            }
        )
    return latent


def encode_style_tokens(
    runtime: Runtime,
    vision_patches_cpu: tuple[torch.Tensor, ...],
    recorder: StageRecorder,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert runtime.style_model is not None
    details: dict = {}
    with recorder.stage("10", details):
        patches = tuple(
            item.to(runtime.device, dtype=runtime.dtype) for item in vision_patches_cpu
        )
        enc_start = torch.cuda.Event(enable_timing=True)
        enc_end = torch.cuda.Event(enable_timing=True)
        proj_start = torch.cuda.Event(enable_timing=True)
        proj_end = torch.cuda.Event(enable_timing=True)
        enc_start.record()
        with torch.inference_mode():
            style_embeds = runtime.style_model.style_aware_encoder(patches).to(
                runtime.device, dtype=runtime.dtype
            )
        enc_end.record()
        proj_start.record()
        style_tokens = []
        uncond_tokens = []
        with torch.inference_mode():
            for index, embed in enumerate(
                [style_embeds[:, 0, :], style_embeds[:, 1, :], style_embeds[:, 2, :]]
            ):
                projection = runtime.style_model.style_image_proj_modules[index]
                style_tokens.append(projection(embed))
                uncond_tokens.append(projection(torch.zeros_like(embed)))
        style_tokens_tensor = torch.cat(style_tokens, dim=1)
        uncond_tokens_tensor = torch.cat(uncond_tokens, dim=1)
        proj_end.record()
        torch.cuda.synchronize()
        details.update(
            {
                "vision_encoder": str(CLIP_VISION_MODEL),
                "vision_input_shapes": [list(item.shape) for item in patches],
                "encoder_ms": round(enc_start.elapsed_time(enc_end), 4),
                "projection_ms": round(proj_start.elapsed_time(proj_end), 4),
                "token_count": int(style_tokens_tensor.shape[1]),
                "token_dim": int(style_tokens_tensor.shape[2]),
                "token_shape": list(style_tokens_tensor.shape),
                "dtype": str(style_tokens_tensor.dtype),
                "conditional_token_bytes": tensor_bytes(style_tokens_tensor),
                "unconditional_token_bytes": tensor_bytes(uncond_tokens_tensor),
                "cache_hit": False,
                "precomputed_per_layer_kv": False,
            }
        )
    return style_tokens_tensor, uncond_tokens_tensor


def resolve_qkv_injection_records(records: list[dict]) -> list[dict]:
    if not records:
        return []
    torch.cuda.synchronize()
    resolved = []
    for record in records:
        item = dict(record)
        events = item.pop("events")
        item["transfer_ms"] = events["transfer_start"].elapsed_time(events["transfer_end"])
        item["attention_ms"] = events["attention_start"].elapsed_time(events["attention_end"])
        item["total_ms"] = events["total_start"].elapsed_time(events["total_end"])
        resolved.append(json_safe(item))
    return resolved


def resolve_adapter_records(records: list[dict]) -> list[dict]:
    if not records:
        return []
    torch.cuda.synchronize()
    resolved = []
    for record in records:
        item = dict(record)
        events = item.pop("events")
        item["projection_ms"] = events["projection_start"].elapsed_time(
            events["projection_end"]
        )
        item["attention_ms"] = events["attention_start"].elapsed_time(
            events["attention_end"]
        )
        item["total_ms"] = events["total_start"].elapsed_time(events["total_end"])
        resolved.append(json_safe(item))
    return resolved


@torch.inference_mode()
def denoise(
    runtime: Runtime,
    start_latent: torch.Tensor,
    cond: torch.Tensor,
    uncond: torch.Tensor,
    style_tokens: torch.Tensor | None,
    uncond_style_tokens: torch.Tensor | None,
    steps: int,
    guidance_scale: float,
    style_scale: float,
    lambda_q: float,
    group: str,
    recorder: StageRecorder,
    unet_counter: dict,
    qkv_injection_events: list[dict],
    adapter_events: list[dict],
) -> torch.Tensor:
    scheduler = DDIMScheduler.from_config(
        runtime.scheduler.config, clip_sample=False, set_alpha_to_one=True
    )
    scheduler.set_timesteps(steps, device=runtime.device)

    adapter_enabled = group in {"D", "E", "F"}
    self_enabled = group in {"C", "F"}
    if adapter_enabled:
        assert style_tokens is not None and uncond_style_tokens is not None
        cond_input = torch.cat([cond, style_tokens], dim=1)
        uncond_input = torch.cat([uncond, uncond_style_tokens], dim=1)
        set_adapter_scale(runtime, style_scale)
    else:
        cond_input = cond
        uncond_input = uncond
        set_adapter_scale(runtime, 0.0)
    encoder_hidden_states = torch.cat([uncond_input, cond_input], dim=0)

    controller = runtime.controller
    if self_enabled:
        assert controller is not None
        controller.injection_observer = qkv_injection_events.append
        controller.inject(lambda_q, total_steps=steps, injection_start=0.0, injection_end=1.0)
    elif controller is not None:
        controller.disable()

    for layer_name, processor in runtime.unet.attn_processors.items():
        if isinstance(processor, IPAttnProcessor2_0):
            processor.profile_layer_name = layer_name
            processor.profile_observer = adapter_events.append if adapter_enabled else None

    details: dict = {}
    latents = start_latent.clone()
    step_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    with recorder.stage("12", details):
        for index, timestep in enumerate(scheduler.timesteps):
            if controller is not None:
                controller.set_step(index, timestep)
            for processor in runtime.unet.attn_processors.values():
                if isinstance(processor, IPAttnProcessor2_0):
                    processor.profile_timestep = int(timestep.item())
            latent_input = torch.cat([latents, latents], dim=0)
            latent_input = scheduler.scale_model_input(latent_input, timestep)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            noise_pred = runtime.unet(
                latent_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
            end.record()
            step_events.append((start, end))
            unet_counter["generation"] += 1
        torch.cuda.synchronize()
        per_step = [start.elapsed_time(end) for start, end in step_events]
        details.update(
            {
                "steps": steps,
                "unet_forward_calls": len(step_events),
                "effective_batch": 2,
                "cfg": True,
                "guidance_scale": guidance_scale,
                "attention_backend": "torch.scaled_dot_product_attention",
                "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
                "memory_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "per_step_ms": [round(value, 4) for value in per_step],
                "mean_step_ms": round(float(np.mean(per_step)), 4),
                "p95_step_ms": round(float(np.percentile(per_step, 95)), 4),
                "self_attention_injection_steps": steps if self_enabled else 0,
                "adapter_style_steps": steps if adapter_enabled else 0,
                "dual_route_steps": steps if group == "F" else 0,
                "start_latent_shape": list(start_latent.shape),
                "final_latent_shape": list(latents.shape),
            }
        )

    if controller is not None:
        controller.disable()
        controller.injection_observer = None
    for processor in runtime.unet.attn_processors.values():
        if isinstance(processor, IPAttnProcessor2_0):
            processor.profile_observer = None
    return latents


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(json_safe(record), ensure_ascii=False) + "\n")


def choose_image(title: str) -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError(
            "tkinter不可用，请通过 --content 和 --style 指定图像路径"
        ) from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(
        title=title,
        filetypes=[
            ("图像文件", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
            ("所有文件", "*.*"),
        ],
    )
    root.destroy()
    if not selected:
        raise SystemExit(f"已取消：{title}")
    return Path(selected).resolve()


def aggregate_feature_records(
    recorder: StageRecorder,
    feature_records: list[dict],
    output_dir: Path,
) -> None:
    write_jsonl(output_dir / "feature_cache_records.jsonl", feature_records)
    for parent in ("07", "08"):
        items = [item for item in feature_records if item.get("parent_stage") == parent]
        if not items:
            continue
        total_bytes = sum(int(item["bytes"]) for item in items)
        recorder.aggregate(
            "09",
            sum(float(item["write_ms"]) for item in items),
            [float(item["gpu_allocated_mib"]) for item in items],
            {
                "feature_types": sorted({item["feature_type"] for item in items}),
                "layers": sorted({item["layer_name"] for item in items}),
                "timesteps": len({item["timestep"] for item in items}),
                "tensor_records": len(items),
                "cache_location": "cpu_fp16",
                "cumulative_write_bytes": total_bytes,
                "cache_resident_peak_bytes": total_bytes,
                "write_ms_sum": sum(float(item["write_ms"]) for item in items),
                "details_file": "feature_cache_records.jsonl",
            },
            parent,
        )


def aggregate_injection_records(
    recorder: StageRecorder,
    qkv_records: list[dict],
    adapter_records: list[dict],
    output_dir: Path,
) -> None:
    resolved_qkv = resolve_qkv_injection_records(qkv_records)
    resolved_adapter = resolve_adapter_records(adapter_records)
    write_jsonl(output_dir / "self_attention_injection_records.jsonl", resolved_qkv)
    write_jsonl(output_dir / "adapter_attention_records.jsonl", resolved_adapter)
    if resolved_qkv:
        recorder.aggregate(
            "13",
            sum(float(item["total_ms"]) for item in resolved_qkv),
            [float(item["gpu_allocated_mib"]) for item in resolved_qkv],
            {
                "calls": len(resolved_qkv),
                "layers": sorted({item["layer_name"] for item in resolved_qkv}),
                "timesteps": len({item["timestep"] for item in resolved_qkv}),
                "mode": "replace_style_kv_mix_content_q",
                "lambda_q": resolved_qkv[0]["lambda_q"],
                "cache_location": "cpu",
                "read_bytes": sum(int(item["read_bytes"]) for item in resolved_qkv),
                "cpu_to_gpu_transfer_ms_sum": sum(
                    float(item["transfer_ms"]) for item in resolved_qkv
                ),
                "attention_ms_sum": sum(float(item["attention_ms"]) for item in resolved_qkv),
                "details_file": "self_attention_injection_records.jsonl",
            },
            "12",
        )
    if resolved_adapter:
        recorder.aggregate(
            "14",
            sum(float(item["total_ms"]) for item in resolved_adapter),
            [float(item["gpu_allocated_mib"]) for item in resolved_adapter],
            {
                "style_branch_calls": len(resolved_adapter),
                "layers": sorted({item["layer_name"] for item in resolved_adapter}),
                "timesteps": len({item["timestep"] for item in resolved_adapter}),
                "style_token_count": resolved_adapter[0]["token_count"],
                "token_dim": resolved_adapter[0]["token_dim"],
                "projection_each_step": True,
                "projection_ms_sum": sum(
                    float(item["projection_ms"]) for item in resolved_adapter
                ),
                "style_attention_ms_sum": sum(
                    float(item["attention_ms"]) for item in resolved_adapter
                ),
                "details_file": "adapter_attention_records.jsonl",
            },
            "12",
        )


def run_group(
    group: str,
    args,
    session_dir: Path,
    raw_content: Image.Image,
    raw_style: Image.Image,
) -> dict:
    run_id = f"{group}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    output_dir = session_dir / f"group_{group}__{GROUP_INFO[group]['name']}__{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)
    recorder = StageRecorder(run_id, group)
    monitor = ResourceMonitor(recorder)
    device = torch.device("cuda")
    dtype = torch.float16
    unet_counter = {"inversion": 0, "generation": 0}
    feature_records: list[dict] = []
    qkv_injection_events: list[dict] = []
    adapter_events: list[dict] = []
    cache_hits = {
        "text_initial_encode": False,
        "text_reused_for_inversion_and_generation": True,
        "style_tokens_initial_encode": False,
        "style_tokens_reused_for_all_generation_steps": group in {"D", "E", "F"},
        "qkv_initial_capture": False,
        "qkv_reused_during_generation": group in {"C", "F"},
        "model_os_page_cache": "unknown_and_order_dependent",
    }

    # Stage 17 starts from raw PIL images already resident in CPU memory.
    e2e_started = time.perf_counter()
    monitor.start()
    content_image = None
    style_image = None
    content_tensor = None
    style_tensor = None
    style_vision_patches_cpu = None
    input_details: dict = {"prompt": args.prompt}
    if group in {"B", "C", "E", "F"}:
        input_details["content_id"] = file_id(args.content)
        input_details["content_original_size"] = list(raw_content.size)
    if group in {"C", "D", "E", "F"}:
        input_details["style_id"] = file_id(args.style)
        input_details["style_original_size"] = list(raw_style.size)
    with recorder.stage("01", input_details):
        if group in {"B", "C", "E", "F"}:
            content_image = prepare_pil(raw_content, args.resolution)
            content_tensor = pil_to_vae_tensor(content_image)
            input_details["content_crop_resize"] = {
                "mode": "center_crop_lanczos",
                "output_size": list(content_image.size),
                "vae_input_shape": list(content_tensor.shape),
            }
        if group in {"C", "E", "F"}:
            style_image = prepare_pil(raw_style, args.resolution)
            style_tensor = pil_to_vae_tensor(style_image)
            input_details["style_vae_route"] = {
                "output_size": list(style_image.size),
                "vae_input_shape": list(style_tensor.shape),
            }
        elif group == "D":
            style_image = prepare_pil(raw_style, args.resolution)
        if group in {"D", "E", "F"}:
            # StyleProcessor shuffles multi-scale patches; reset the CPU RNG so
            # paired D/E/F runs receive exactly the same style condition.
            torch.manual_seed(args.seed)
            style_vision_patches_cpu = StyleProcessor(style_image, torch.device("cpu"))
            input_details["style_vision_route"] = {
                "source_size": list(style_image.size),
                "StyleProcessor_target": [512, 512],
                "encoder_input_shapes": [
                    list(item.shape) for item in style_vision_patches_cpu
                ],
                "dtype": [str(item.dtype) for item in style_vision_patches_cpu],
            }

    runtime = build_runtime(group, recorder, device, dtype)
    cond, uncond = encode_text(runtime, args.prompt, args.negative_prompt, recorder)

    start_latent = None
    noise_logical_id = None
    if group in {"A", "D"}:
        details: dict = {}
        with recorder.stage("04", details):
            generator = torch.Generator(device=device).manual_seed(args.seed)
            start_latent = torch.randn(
                (1, 4, args.resolution // 8, args.resolution // 8),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            noise_logical_id = tensor_id(start_latent)
            details.update(
                {
                    "seed": args.seed,
                    "shape": list(start_latent.shape),
                    "dtype": str(start_latent.dtype),
                    "device": str(start_latent.device),
                    "noise_tensor_id": noise_logical_id,
                    "paired_A_D_same_seed_shape_dtype": True,
                }
            )

    content_z0 = None
    style_z0 = None
    if group in {"B", "C", "E", "F"}:
        content_z0 = vae_encode(
            runtime, content_tensor, recorder, "05", input_details["content_id"]
        )
    if group in {"C", "E", "F"}:
        style_z0 = vae_encode(runtime, style_tensor, recorder, "06", input_details["style_id"])

    feature_parent = {"value": None}
    if runtime.controller is not None:
        def feature_observer(item: dict) -> None:
            record = dict(item)
            record["parent_stage"] = feature_parent["value"]
            feature_records.append(record)

        runtime.controller.bank.write_observer = feature_observer
        runtime.controller.clear()
        runtime.controller.configure_injection_window(args.steps, 0.0, 1.0)

    content_zT = None
    style_zT = None
    if group in {"B", "C", "E", "F"}:
        feature_parent["value"] = "07"
        content_zT = invert_latent(
            runtime,
            content_z0,
            cond,
            args.steps,
            recorder,
            "07",
            "content" if group in {"C", "F"} else "none",
            unet_counter,
        )
    if group in {"C", "E", "F"}:
        feature_parent["value"] = "08"
        style_zT = invert_latent(
            runtime,
            style_z0,
            cond,
            args.steps,
            recorder,
            "08",
            "style" if group in {"C", "F"} else "none",
            unet_counter,
        )
    if feature_records:
        aggregate_feature_records(recorder, feature_records, output_dir)

    style_tokens = None
    uncond_style_tokens = None
    style_token_id = None
    if group in {"D", "E", "F"}:
        style_tokens, uncond_style_tokens = encode_style_tokens(
            runtime, style_vision_patches_cpu, recorder
        )
        style_token_id = tensor_id(style_tokens)

    adain_id = None
    if group in {"C", "E", "F"}:
        details: dict = {}
        with recorder.stage("11", details):
            start_latent = adaptive_instance_normalization(
                content_zT, style_zT, eps=args.adain_epsilon
            )
            if args.adain_strength != 1.0:
                start_latent = content_zT.lerp(start_latent, args.adain_strength)
            adain_id = tensor_id(start_latent)
            details.update(
                {
                    "content_terminal_id": tensor_id(content_zT),
                    "style_terminal_id": tensor_id(style_zT),
                    "statistics_dims": [2, 3],
                    "epsilon": args.adain_epsilon,
                    "strength": args.adain_strength,
                    "output_shape": list(start_latent.shape),
                    "dtype": str(start_latent.dtype),
                    "start_tensor_id": adain_id,
                    "paired_C_E_F_should_match": True,
                }
            )
    elif group == "B":
        start_latent = content_zT

    final_latent = denoise(
        runtime,
        start_latent,
        cond,
        uncond,
        style_tokens,
        uncond_style_tokens,
        args.steps,
        args.guidance_scale,
        args.style_scale,
        args.lambda_q,
        group,
        recorder,
        unet_counter,
        qkv_injection_events,
        adapter_events,
    )
    aggregate_injection_records(
        recorder, qkv_injection_events, adapter_events, output_dir
    )

    decode_details: dict = {}
    with recorder.stage("15", decode_details):
        decoded_gpu = runtime.vae.decode(
            final_latent / runtime.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        decode_details.update(
            {
                "output_shape": list(decoded_gpu.shape),
                "resolution": [args.resolution, args.resolution],
                "dtype": str(decoded_gpu.dtype),
                "sliced_decode": False,
            }
        )

    post_details: dict = {}
    with recorder.stage("16", post_details):
        transfer_started = time.perf_counter()
        image_cpu = decoded_gpu.detach().float().cpu()
        transfer_ms = (time.perf_counter() - transfer_started) * 1000.0
        format_started = time.perf_counter()
        image_cpu = (image_cpu / 2 + 0.5).clamp(0, 1)
        array = (
            image_cpu[0].permute(1, 2, 0).numpy() * 255.0
        ).round().astype(np.uint8)
        output_image = Image.fromarray(array, mode="RGB")
        format_ms = (time.perf_counter() - format_started) * 1000.0
        post_details.update(
            {
                "gpu_to_cpu_bytes": tensor_bytes(decoded_gpu),
                "gpu_to_cpu_transfer_ms": round(transfer_ms, 4),
                "format_conversion_ms": round(format_ms, 4),
                "cpu_image_size": list(output_image.size),
                "cpu_image_mode": output_image.mode,
                "png_write_in_separate_stage_row": True,
            }
        )

    # End-to-end boundary ends when the final image exists in CPU memory.
    monitor.stop()
    e2e_duration_ms = (time.perf_counter() - e2e_started) * 1000.0
    all_stage_peaks = [row["gpu_peak_mib"] for row in recorder.rows]
    peak_stage = monitor.gpu_peak_stage
    recorder.aggregate(
        "17",
        e2e_duration_ms,
        [0.0, mib(monitor.gpu_peak_bytes), mib(torch.cuda.memory_allocated())],
        {
            "total_unet_forward_calls": sum(unet_counter.values()),
            "inversion_unet_calls": unet_counter["inversion"],
            "generation_unet_calls": unet_counter["generation"],
            "expected_unet_calls_for_configured_steps": (
                {"A": 1, "B": 2, "C": 3, "D": 1, "E": 3, "F": 3}[group]
                * args.steps
            ),
            "gpu_absolute_peak_mib": max(all_stage_peaks + [mib(monitor.gpu_peak_bytes)]),
            "gpu_peak_stage_sampled": peak_stage,
            "cpu_rss_peak_mib": mib(monitor.cpu_peak_bytes),
            "cache_hits": cache_hits,
            "success": True,
            "oom": False,
        },
        None,
    )

    output_path = output_dir / "output.png"
    png_details = {"output_path": str(output_path)}
    with recorder.stage("16", png_details):
        started = time.perf_counter()
        output_image.save(output_path, format="PNG")
        png_details.update(
            {
                "operation": "png_encode_and_disk_write",
                "png_encode_write_ms": round((time.perf_counter() - started) * 1000.0, 4),
                "file_bytes": output_path.stat().st_size,
            }
        )

    summary = {
        "run_id": run_id,
        "group": group,
        "group_name": GROUP_INFO[group]["name"],
        "output": str(output_path),
        "steps": args.steps,
        "resolution": args.resolution,
        "unet_calls": dict(unet_counter),
        "total_unet_calls": sum(unet_counter.values()),
        "expected_total_unet_calls": (
            {"A": 1, "B": 2, "C": 3, "D": 1, "E": 3, "F": 3}[group]
            * args.steps
        ),
        "noise_tensor_id": noise_logical_id,
        "adain_tensor_id": adain_id,
        "style_token_id": style_token_id,
        "feature_cache_summary": (
            runtime.controller.bank.summary() if runtime.controller is not None else None
        ),
        "stage_rows": None,
        "e2e_ms": round(e2e_duration_ms, 4),
        "gpu_peak_mib": max(row["gpu_peak_mib"] for row in recorder.rows),
        "cpu_peak_mib": mib(monitor.cpu_peak_bytes),
        "success": True,
    }

    unload_runtime(runtime, recorder)
    summary["stage_rows"] = len(recorder.rows)
    recorder.write(output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile stage-level time and memory costs for style-transfer groups A-F."
    )
    parser.add_argument(
        "--content",
        type=Path,
        default=None,
        help="内容图路径；省略时弹出文件选择框",
    )
    parser.add_argument(
        "--style",
        type=Path,
        default=None,
        help="风格图路径；省略时弹出文件选择框",
    )
    parser.add_argument("--groups", default="A,B,C,D,E,F")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative-prompt", default="low quality, blurry, artifacts")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--style-scale", type=float, default=1.0)
    parser.add_argument("--lambda-q", type=float, default=0.7)
    parser.add_argument("--adain-strength", type=float, default=1.0)
    parser.add_argument("--adain-epsilon", type=float, default=1e-6)
    parser.add_argument("--minimum-free-vram-gb", type=float, default=5.0)
    return parser.parse_args()


def validate_args(args) -> list[str]:
    args.content = args.content.resolve() if args.content is not None else choose_image("选择内容图")
    args.style = args.style.resolve() if args.style is not None else choose_image("选择风格图")
    args.output_root = args.output_root.resolve()
    if not args.content.is_file() or not args.style.is_file():
        raise FileNotFoundError("Both --content and --style must point to image files")
    groups = [item.strip().upper() for item in args.groups.split(",") if item.strip()]
    invalid = [item for item in groups if item not in GROUP_INFO]
    if invalid:
        raise ValueError(f"Unsupported groups: {invalid}; supported A-F. Group G is not defined.")
    if args.steps < 2:
        raise ValueError("--steps must be at least 2")
    if args.resolution < 64 or args.resolution % 64:
        raise ValueError("--resolution must be a multiple of 64")
    if not 0.0 <= args.lambda_q <= 1.0:
        raise ValueError("--lambda-q must be in [0,1]")
    if not 0.0 <= args.adain_strength <= 1.0:
        raise ValueError("--adain-strength must be in [0,1]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes / 1024**3 < args.minimum_free_vram_gb:
        raise RuntimeError(
            f"Only {free_bytes / 1024**3:.2f}/{total_bytes / 1024**3:.2f} GiB VRAM is free. "
            "Close/unload other model processes before profiling."
        )
    return groups


def main() -> int:
    args = parse_args()
    groups = validate_args(args)
    raw_content = Image.open(args.content).convert("RGB")
    raw_style = Image.open(args.style).convert("RGB")
    if args.prompt is None:
        args.prompt = f"{args.content.stem}, best quality, high quality"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = args.session_name or (
        f"session_{timestamp}__steps{args.steps}__res{args.resolution}"
    )
    session_dir = args.output_root / session_name
    session_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "content": str(args.content),
        "style": str(args.style),
        "groups": groups,
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "counting_convention": {
            "A_D": "N generation U-Net calls",
            "B": "N inversion + N generation calls",
            "C_E_F": "N content inversion + N style inversion + N generation calls",
            "inversion": (
                "N-call DDIM: z0 is treated as the lowest scheduled timestep. "
                "This intentionally matches the requested 50/100/150 count; the GUI inverter "
                "currently performs one additional bootstrap prediction per image."
            ),
        },
        "memory_units": "MiB",
        "gpu_memory_definition": "PyTorch absolute allocated/reserved memory, not per-stage delta",
        "method_scope": {
            "controlnet_enabled": False,
            "reason": (
                "The requested A-F table and stages isolate U-Net text, inversion, self-attention KV, "
                "and adapter cross-attention costs. HED/ControlNet is therefore excluded from every group."
            ),
            "adapter_definition": "StyleShot six image-style tokens in U-Net cross-attention",
            "self_attention_definition": "content-Q/style-KV replacement in six up-block attn1 layers",
        },
        "group_G": "not executed because no group G method was defined in the requested group table",
    }
    (session_dir / "session_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summaries = []
    for group in groups:
        print(f"\n===== Group {group}: {GROUP_INFO[group]['name']} =====", flush=True)
        try:
            summaries.append(
                run_group(group, args, session_dir, raw_content.copy(), raw_style.copy())
            )
            gc.collect()
            torch.cuda.empty_cache()
        except torch.OutOfMemoryError as exc:
            error = {
                "group": group,
                "success": False,
                "oom": True,
                "error": str(exc),
            }
            summaries.append(error)
            torch.cuda.empty_cache()
            print(json.dumps(error, ensure_ascii=False), flush=True)
            break
        except Exception as exc:
            error = {
                "group": group,
                "success": False,
                "oom": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            summaries.append(error)
            torch.cuda.empty_cache()
            print(json.dumps(error, ensure_ascii=False), flush=True)
            break

    # Cross-group pairing checks.
    by_group = {item["group"]: item for item in summaries if item.get("success")}
    noise_ids = {group: by_group[group].get("noise_tensor_id") for group in ("A", "D") if group in by_group}
    adain_ids = {
        group: by_group[group].get("adain_tensor_id")
        for group in ("C", "E", "F")
        if group in by_group
    }
    style_token_ids = {
        group: by_group[group].get("style_token_id")
        for group in ("D", "E", "F")
        if group in by_group
    }
    comparison = {
        "session_dir": str(session_dir),
        "runs": summaries,
        "pairing_checks": {
            "A_D_same_noise": len(set(noise_ids.values())) == 1 if len(noise_ids) == 2 else None,
            "noise_ids": noise_ids,
            "C_E_F_same_adain": len(set(adain_ids.values())) == 1 if len(adain_ids) >= 2 else None,
            "adain_ids": adain_ids,
            "D_E_F_same_style_tokens": (
                len(set(style_token_ids.values())) == 1
                if len(style_token_ids) >= 2
                else None
            ),
            "style_token_ids": style_token_ids,
        },
    }
    (session_dir / "comparison_summary.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (session_dir / "comparison_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fields = [
            "group",
            "group_name",
            "success",
            "total_unet_calls",
            "expected_total_unet_calls",
            "e2e_ms",
            "gpu_peak_mib",
            "cpu_peak_mib",
            "output",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"\nProfile results: {session_dir}")
    return 0 if all(item.get("success") for item in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
