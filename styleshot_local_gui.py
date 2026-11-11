from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import threading
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
for proxy_bypass_key in ("NO_PROXY", "no_proxy"):
    proxy_bypass = os.environ.get(proxy_bypass_key, "")
    local_hosts = [host for host in ("127.0.0.1", "localhost") if host not in proxy_bypass]
    if local_hosts:
        os.environ[proxy_bypass_key] = ",".join(filter(None, [proxy_bypass, *local_hosts]))

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageOps
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)

from annotator.hed import SOFT_HEDdetector
from annotator.util import resize_image
from ip_adapter import StyleContentStableDiffusionControlNetPipeline, StyleShot
from ip_adapter.self_attention_injection import (
    DDIMQKVInverter,
    SelfAttentionQKVController,
    blend_adain_latent,
)


ROOT = Path(__file__).resolve().parent
BASE_MODEL = ROOT / "runwayml" / "stable-diffusion-v1-5"
CLIP_MODEL = ROOT / "laion" / "CLIP-ViT-H-14-laion2B-s32B-b79K"
STYLESHOT_MODEL = ROOT / "Gaojunyao" / "StyleShot"
STYLEBENCH = ROOT / "StyleBench"
CONTENT_ROOT = STYLEBENCH / "content"
STYLE_ROOT = STYLEBENCH / "style"
OUTPUT_ROOT = ROOT / "outputs" / "gui"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_POSITIVE = "best quality, high quality, extremely detailed"
DEFAULT_NEGATIVE = (
    "lowres, worst quality, low quality, blurry, bad anatomy, bad hands, "
    "missing fingers, extra fingers, cropped, watermark, text"
)

MODEL_LOCK = threading.Lock()
MODEL: StyleShot | None = None
DETECTOR: SOFT_HEDdetector | None = None
QKV_CONTROLLER: SelfAttentionQKVController | None = None
QKV_INVERTER: DDIMQKVInverter | None = None
MODEL_LOAD_SECONDS: float | None = None


def image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )


CONTENT_FILES = image_files(CONTENT_ROOT)
STYLE_FILES = {
    folder.name: image_files(folder)
    for folder in sorted(STYLE_ROOT.iterdir(), key=lambda path: path.name.lower())
    if folder.is_dir() and image_files(folder)
} if STYLE_ROOT.is_dir() else {}


def required_files() -> list[Path]:
    return [
        BASE_MODEL / "model_index.json",
        BASE_MODEL / "unet" / "diffusion_pytorch_model.safetensors",
        BASE_MODEL / "vae" / "diffusion_pytorch_model.safetensors",
        BASE_MODEL / "text_encoder" / "model.safetensors",
        CLIP_MODEL / "config.json",
        CLIP_MODEL / "model.safetensors",
        STYLESHOT_MODEL / "pretrained_weight" / "ip.bin",
        STYLESHOT_MODEL / "pretrained_weight" / "style_aware_encoder.bin",
        ROOT / "annotator" / "ckpts" / "ControlNetHED.pth",
    ]


def validate_installation() -> None:
    missing = [str(path) for path in required_files() if not path.is_file()]
    if missing:
        raise RuntimeError("缺少运行文件：\n" + "\n".join(missing))
    if not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 未检测到 CUDA；StyleShot 图形界面需要 NVIDIA GPU。")
    if not CONTENT_FILES:
        raise RuntimeError(f"StyleBench 内容图目录为空：{CONTENT_ROOT}")
    if not STYLE_FILES:
        raise RuntimeError(f"StyleBench 风格图目录为空：{STYLE_ROOT}")


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()


def content_names() -> list[str]:
    return [path.name for path in CONTENT_FILES]


def style_names(category: str) -> list[str]:
    return [path.name for path in STYLE_FILES.get(category, [])]


def load_content_choice(name: str | None) -> Image.Image | None:
    if not name:
        return None
    path = CONTENT_ROOT / name
    return open_rgb(path) if path.is_file() else None


def load_style_choice(category: str | None, name: str | None) -> Image.Image | None:
    if not category or not name:
        return None
    path = STYLE_ROOT / category / name
    return open_rgb(path) if path.is_file() else None


def change_style_category(category: str):
    names = style_names(category)
    selected = names[0] if names else None
    image = load_style_choice(category, selected)
    return gr.update(choices=names, value=selected), image


def scheduler_from_name(name: str, config):
    if name == "DDIM":
        return DDIMScheduler.from_config(config)
    if name == "Euler":
        return EulerDiscreteScheduler.from_config(config)
    if name == "Euler a":
        return EulerAncestralDiscreteScheduler.from_config(config)
    if name == "DPM++ 2M Karras":
        return DPMSolverMultistepScheduler.from_config(
            config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
    return PNDMScheduler.from_config(config)


def load_models(progress=gr.Progress(track_tqdm=True)) -> str:
    global MODEL, DETECTOR, QKV_CONTROLLER, QKV_INVERTER, MODEL_LOAD_SECONDS
    if MODEL is not None and DETECTOR is not None:
        return model_status()

    with MODEL_LOCK:
        if MODEL is not None and DETECTOR is not None:
            return model_status()

        validate_installation()
        started = time.perf_counter()
        dtype = torch.float16
        device = torch.device("cuda")

        progress(0.05, desc="加载轮廓检测器")
        detector = SOFT_HEDdetector()

        progress(0.15, desc="加载 SD1.5 UNet")
        unet = UNet2DConditionModel.from_pretrained(
            str(BASE_MODEL),
            subfolder="unet",
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        progress(0.30, desc="构建内容融合 ControlNet")
        controlnet = ControlNetModel.from_unet(unet).to(dtype=dtype)

        progress(0.45, desc="组装 Stable Diffusion 管线")
        pipe = StyleContentStableDiffusionControlNetPipeline.from_pretrained(
            str(BASE_MODEL),
            unet=unet,
            controlnet=controlnet,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        pipe.enable_vae_slicing()

        progress(0.65, desc="加载 StyleShot 风格与内容权重")
        model = StyleShot(
            device,
            pipe,
            str(STYLESHOT_MODEL / "pretrained_weight" / "ip.bin"),
            str(STYLESHOT_MODEL / "pretrained_weight" / "style_aware_encoder.bin"),
            str(CLIP_MODEL),
            weight_dtype=dtype,
        )
        model.eval()
        model.style_aware_encoder.eval()
        model.style_image_proj_modules.eval()
        model.pipe.unet.eval()
        model.pipe.controlnet.eval()

        qkv_controller = SelfAttentionQKVController()
        qkv_controller.install(model.pipe.unet)
        model.pipe.qkv_controller = qkv_controller
        qkv_inverter = DDIMQKVInverter(model.pipe, qkv_controller)

        MODEL = model
        DETECTOR = detector
        QKV_CONTROLLER = qkv_controller
        QKV_INVERTER = qkv_inverter
        MODEL_LOAD_SECONDS = time.perf_counter() - started
        torch.cuda.synchronize()
        progress(1.0, desc="模型加载完成")
        return model_status()


def unload_models() -> str:
    global MODEL, DETECTOR, QKV_CONTROLLER, QKV_INVERTER, MODEL_LOAD_SECONDS
    with MODEL_LOCK:
        MODEL = None
        DETECTOR = None
        QKV_CONTROLLER = None
        QKV_INVERTER = None
        MODEL_LOAD_SECONDS = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return "模型已卸载，显存缓存已释放。下次生成时会自动重新加载。"


def model_status() -> str:
    if MODEL is None:
        return "模型尚未加载；点击“加载模型”或直接点击“开始风格迁移”。"
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    seconds = MODEL_LOAD_SECONDS or 0.0
    return (
        f"模型已加载：`{torch.cuda.get_device_name(0)}`，FP16；"
        f"加载用时 {seconds:.1f}s；显存 {allocated:.2f}GB / 保留 {reserved:.2f}GB。"
    )


def prepare_content(image: Image.Image, resolution: int, resize_mode: str) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if resize_mode == "中心裁剪为正方形":
        square = ImageOps.fit(
            image,
            (resolution, resolution),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        return np.asarray(square)

    resized = resize_image(np.asarray(image), resolution)
    height, width = resized.shape[:2]
    max_side = max(768, resolution)
    if max(height, width) > max_side:
        scale = max_side / max(height, width)
        new_width = max(64, int(round(width * scale / 64)) * 64)
        new_height = max(64, int(round(height * scale / 64)) * 64)
        resized = cv2.resize(resized, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized


def save_run(
    style_image: Image.Image,
    content_image: Image.Image,
    control_image: Image.Image,
    results: list[Image.Image],
    metadata: dict,
) -> tuple[Path, list[str]]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = OUTPUT_ROOT / f"{stamp}_seed{metadata['seed']}"
    run_dir.mkdir(parents=True, exist_ok=False)

    style_image.save(run_dir / "style.png")
    content_image.save(run_dir / "content.png")
    control_image.save(run_dir / "control.png")
    output_paths = []
    for index, image in enumerate(results, start=1):
        path = run_dir / f"result_{index:02d}.png"
        image.save(path)
        output_paths.append(str(path))

    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir, output_paths + [str(metadata_path)]


@torch.inference_mode()
def generate(
    style_image: Image.Image | None,
    content_image: Image.Image | None,
    prompt: str,
    negative_prompt: str,
    num_samples: int,
    resolution: int,
    resize_mode: str,
    style_scale: float,
    condition_scale: float,
    steps: int,
    guidance_scale: float,
    seed: int,
    scheduler_name: str,
    contour_threshold: int,
    control_start: float,
    control_end: float,
    guess_mode: bool,
    enable_qkv: bool,
    lambda_q: float,
    adain_strength: float,
    self_attention_start: float,
    self_attention_end: float,
    enable_style_cross_attention: bool,
    style_cross_attention_start: float,
    style_cross_attention_end: float,
    progress=gr.Progress(track_tqdm=True),
):
    if style_image is None:
        raise gr.Error("请上传或从 StyleBench 选择风格图。")
    if content_image is None:
        raise gr.Error("请上传或从 StyleBench 选择内容图。")
    if control_start >= control_end:
        raise gr.Error("Control Start 必须小于 Control End。")
    if enable_qkv and float(self_attention_start) >= float(self_attention_end):
        raise gr.Error("启用 QKV 时，自注意力 Start 必须小于 End。")
    if enable_style_cross_attention and float(style_cross_attention_start) >= float(style_cross_attention_end):
        raise gr.Error("启用交叉注意力风格条件时，Cross-Attention Start 必须小于 End。")

    actual_seed = int(seed)
    if actual_seed < 0:
        actual_seed = random.SystemRandom().randint(0, 2_147_483_647)

    load_models(progress)
    assert MODEL is not None and DETECTOR is not None
    assert QKV_CONTROLLER is not None and QKV_INVERTER is not None

    style_image = ImageOps.exif_transpose(style_image).convert("RGB")
    content_image = ImageOps.exif_transpose(content_image).convert("RGB")
    content_array = prepare_content(content_image, int(resolution), resize_mode)

    progress(0.05, desc="提取内容轮廓")
    control_array = DETECTOR(content_array, threshold=int(contour_threshold))
    control_image = Image.fromarray(control_array).convert("RGB")

    positive = (prompt or "").strip() or DEFAULT_POSITIVE
    negative = (negative_prompt or "").strip()
    started = time.perf_counter()

    qkv_summary = None
    effective_scheduler = scheduler_name
    with MODEL_LOCK:
        try:
            generation_kwargs = {
                "style_cross_attention_enabled": bool(enable_style_cross_attention),
                "style_cross_attention_start": float(style_cross_attention_start),
                "style_cross_attention_end": float(style_cross_attention_end),
            }
            if enable_qkv:
                effective_scheduler = "DDIM（QKV 反演强制）"
                QKV_CONTROLLER.clear()
                QKV_CONTROLLER.configure_injection_window(
                    int(steps),
                    float(self_attention_start),
                    float(self_attention_end),
                )
                inversion_content = Image.fromarray(content_array).convert("RGB")
                inversion_style = ImageOps.fit(
                    style_image,
                    inversion_content.size,
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5),
                )

                progress(0.12, desc="DDIM 反演风格图并记录 K/V")
                style_latent = QKV_INVERTER.invert(
                    inversion_style,
                    num_inference_steps=int(steps),
                    target="style",
                    prompt="",
                )
                progress(0.42, desc="DDIM 反演内容图并记录 Q")
                content_latent = QKV_INVERTER.invert(
                    inversion_content,
                    num_inference_steps=int(steps),
                    target="content",
                    prompt="",
                )
                start_latent = blend_adain_latent(
                    content_latent,
                    style_latent,
                    strength=float(adain_strength),
                )
                start_latent = start_latent.repeat(int(num_samples), 1, 1, 1)
                generation_kwargs["latents"] = start_latent
                MODEL.pipe.scheduler = QKV_INVERTER.make_scheduler()
                QKV_CONTROLLER.inject(
                    float(lambda_q),
                    total_steps=int(steps),
                    injection_start=float(self_attention_start),
                    injection_end=float(self_attention_end),
                )
                qkv_summary = QKV_CONTROLLER.bank.summary()
                progress(0.72, desc="使用 AdaIN latent 和风格 K/V 生成")
            else:
                # A full 512px/50-step feature bank can occupy substantial CPU
                # memory. A baseline run does not need the previous inversion.
                QKV_CONTROLLER.clear()
                MODEL.pipe.scheduler = scheduler_from_name(scheduler_name, MODEL.pipe.scheduler.config)
                progress(0.12, desc="执行 StyleShot 风格迁移")

            generated = MODEL.generate(
                style_image=style_image,
                prompt=positive,
                negative_prompt=negative,
                scale=float(style_scale),
                num_samples=int(num_samples),
                seed=actual_seed,
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(steps),
                content_image=control_image,
                controlnet_conditioning_scale=float(condition_scale),
                control_guidance_start=float(control_start),
                control_guidance_end=float(control_end),
                guess_mode=bool(guess_mode),
                **generation_kwargs,
            )
        finally:
            QKV_CONTROLLER.disable()

    results = generated[0]
    elapsed = time.perf_counter() - started

    def attention_window(enabled: bool, start: float, end: float) -> dict:
        start_step = min(int(steps), math.floor(float(start) * int(steps)))
        end_step = min(int(steps), math.ceil(float(end) * int(steps)))
        active = bool(enabled) and end_step > start_step
        return {
            "enabled": bool(enabled),
            "start_fraction": float(start),
            "end_fraction": float(end),
            "start_step_1_based": start_step + 1 if active else None,
            "end_step_1_based": end_step if active else None,
            "active_step_count": end_step - start_step if active else 0,
        }

    self_window = attention_window(
        bool(enable_qkv), float(self_attention_start), float(self_attention_end)
    )
    cross_window = attention_window(
        bool(enable_style_cross_attention),
        float(style_cross_attention_start),
        float(style_cross_attention_end),
    )
    self_window["applied_steps_1_based"] = [
        step + 1 for step in sorted(QKV_CONTROLLER.applied_steps)
    ]
    cross_window["applied_steps_1_based"] = [
        step + 1
        for step in getattr(MODEL.pipe, "_last_style_cross_attention_steps", [])
    ]
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": actual_seed,
        "prompt": positive,
        "negative_prompt": negative,
        "num_samples": int(num_samples),
        "resolution": int(resolution),
        "output_size": [results[0].width, results[0].height],
        "resize_mode": resize_mode,
        "style_scale": float(style_scale),
        "condition_scale": float(condition_scale),
        "steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "scheduler": effective_scheduler,
        "contour_threshold": int(contour_threshold),
        "control_start": float(control_start),
        "control_end": float(control_end),
        "guess_mode": bool(guess_mode),
        "qkv_enabled": bool(enable_qkv),
        "lambda_q": float(lambda_q),
        "adain_strength": float(adain_strength),
        "self_attention_window": self_window,
        "style_cross_attention_window": cross_window,
        "qkv_features": qkv_summary,
        "qkv_missing_features": len(QKV_CONTROLLER.missing_features),
        "elapsed_seconds": round(elapsed, 3),
    }
    run_dir, saved_files = save_run(style_image, content_image, control_image, results, metadata)

    allocated = torch.cuda.memory_allocated() / 1024**3
    status = (
        f"完成：生成 {len(results)} 张，种子 `{actual_seed}`，耗时 {elapsed:.1f}s，"
        f"当前显存 {allocated:.2f}GB。  \n保存目录：`{run_dir}`"
    )
    if enable_qkv:
        status += (
            f"  \nQKV：`lambda_q={float(lambda_q):.2f}`，"
            f"AdaIN `{float(adain_strength):.2f}`，特征 `{qkv_summary}`，"
            f"缺失 `{len(QKV_CONTROLLER.missing_features)}`，"
            f"自注意力生效步 `{self_window['start_step_1_based']}–{self_window['end_step_1_based']}`。"
        )
    if enable_style_cross_attention:
        status += (
            f"  \nStyleShot 交叉注意力风格条件生效步："
            f"`{cross_window['start_step_1_based']}–{cross_window['end_step_1_based']}`。"
        )
    else:
        status += "  \nStyleShot 交叉注意力风格条件：`关闭`。"
    gallery = [(image, f"结果 {index} · seed {actual_seed}") for index, image in enumerate(results, 1)]
    progress(1.0, desc="已保存结果与参数")
    return control_image, gallery, saved_files, status, actual_seed


def build_ui() -> gr.Blocks:
    default_content_name = content_names()[0]
    default_category = "Watercolor" if "Watercolor" in STYLE_FILES else next(iter(STYLE_FILES))
    default_style_name = style_names(default_category)[0]

    with gr.Blocks(title="StyleShot 本地风格迁移", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# StyleShot 本地风格迁移\n"
            "选择内容图和风格图，调整风格条件与内容轮廓强度。模型仅从本地目录加载，结果会自动保存。"
        )

        with gr.Row():
            load_button = gr.Button("加载模型", variant="secondary")
            unload_button = gr.Button("卸载模型并释放显存", variant="secondary")
        model_message = gr.Markdown(model_status())

        with gr.Row(equal_height=True):
            with gr.Column():
                gr.Markdown("### 内容图")
                content_choice = gr.Dropdown(
                    choices=content_names(),
                    value=default_content_name,
                    label="StyleBench 内容图",
                    filterable=True,
                )
                content_image = gr.Image(
                    value=load_content_choice(default_content_name),
                    type="pil",
                    sources=["upload", "clipboard"],
                    label="内容图（可以直接上传覆盖）",
                    height=420,
                )

            with gr.Column():
                gr.Markdown("### 风格图")
                style_category = gr.Dropdown(
                    choices=list(STYLE_FILES),
                    value=default_category,
                    label="StyleBench 风格类别",
                    filterable=True,
                )
                style_choice = gr.Dropdown(
                    choices=style_names(default_category),
                    value=default_style_name,
                    label="该类别中的参考图",
                )
                style_image = gr.Image(
                    value=load_style_choice(default_category, default_style_name),
                    type="pil",
                    sources=["upload", "clipboard"],
                    label="风格图（可以直接上传覆盖）",
                    height=366,
                )

        prompt = gr.Textbox(label="正向提示词", value=DEFAULT_POSITIVE, lines=2)
        negative_prompt = gr.Textbox(label="负向提示词", value=DEFAULT_NEGATIVE, lines=2)

        with gr.Row():
            style_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Style Strength（风格条件强度）")
            condition_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Content/Control Strength（内容约束）")
            seed = gr.Number(value=42, precision=0, label="Seed（-1 为随机）")

        with gr.Accordion("自注意力 QKV 注入", open=True):
            enable_qkv = gr.Checkbox(
                value=False,
                label="启用：DDIM 反演 + 内容 Q / 风格 K,V + AdaIN latent",
            )
            gr.Markdown(
                "启用后使用与生成相同的步数分别反演风格图和内容图，并强制使用 DDIM。"
                "注入层固定为 `up_blocks.2/3` 中的六个自注意力层。"
            )
            with gr.Row():
                lambda_q = gr.Slider(
                    0.0,
                    1.0,
                    value=0.7,
                    step=0.05,
                    label="lambda_q：1=当前生成Q，0=内容图Q",
                )
                adain_strength = gr.Slider(
                    0.0,
                    1.0,
                    value=1.0,
                    step=0.05,
                    label="AdaIN latent 强度",
                )
            with gr.Row():
                self_attention_start = gr.Slider(
                    0.0,
                    0.95,
                    value=0.0,
                    step=0.05,
                    label="自注意力注入 Start",
                )
                self_attention_end = gr.Slider(
                    0.05,
                    1.0,
                    value=1.0,
                    step=0.05,
                    label="自注意力注入 End",
                )

        with gr.Accordion("交叉注意力风格条件与时间步", open=True):
            enable_style_cross_attention = gr.Checkbox(
                value=True,
                label="启用 StyleShot 交叉注意力风格条件",
            )
            gr.Markdown(
                "这里只控制风格图编码得到的 6 个风格 token，不关闭文本提示词的交叉注意力。"
                "时间轴按去噪进度表示：`0` 是第一步（高噪声），`1` 是最后一步（低噪声）。"
            )
            with gr.Row():
                style_cross_attention_start = gr.Slider(
                    0.0,
                    0.95,
                    value=0.0,
                    step=0.05,
                    label="交叉注意力风格 Start",
                )
                style_cross_attention_end = gr.Slider(
                    0.05,
                    1.0,
                    value=1.0,
                    step=0.05,
                    label="交叉注意力风格 End",
                )

        with gr.Accordion("高级参数", open=False):
            with gr.Row():
                num_samples = gr.Slider(1, 4, value=1, step=1, label="一次生成数量")
                resolution = gr.Slider(256, 768, value=512, step=64, label="输出基准分辨率")
                resize_mode = gr.Radio(
                    ["中心裁剪为正方形", "保持原始比例"],
                    value="中心裁剪为正方形",
                    label="内容图尺寸处理",
                )
            with gr.Row():
                steps = gr.Slider(5, 100, value=30, step=1, label="采样步数")
                guidance_scale = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="CFG Guidance Scale")
                scheduler = gr.Dropdown(
                    ["DDIM", "PNDM", "Euler", "Euler a", "DPM++ 2M Karras"],
                    value="DDIM",
                    label="采样调度器",
                )
            with gr.Row():
                contour_threshold = gr.Slider(0, 255, value=200, step=1, label="轮廓阈值（越高轮廓越稀疏）")
                control_start = gr.Slider(0.0, 0.95, value=0.0, step=0.05, label="Control Start")
                control_end = gr.Slider(0.05, 1.0, value=1.0, step=0.05, label="Control End")
                guess_mode = gr.Checkbox(value=False, label="Guess Mode")

        generate_button = gr.Button("开始风格迁移", variant="primary", size="lg")
        run_status = gr.Markdown("等待生成。")

        with gr.Row():
            control_preview = gr.Image(label="内容控制图/轮廓预览", type="pil", height=420)
            result_gallery = gr.Gallery(
                label="生成结果",
                columns=2,
                rows=2,
                object_fit="contain",
                height=520,
            )
        saved_files = gr.File(label="已保存文件（结果图与 metadata.json）", file_count="multiple")

        content_choice.change(load_content_choice, inputs=content_choice, outputs=content_image)
        style_category.change(
            change_style_category,
            inputs=style_category,
            outputs=[style_choice, style_image],
        )
        style_choice.change(
            load_style_choice,
            inputs=[style_category, style_choice],
            outputs=style_image,
        )
        load_button.click(load_models, outputs=model_message, concurrency_limit=1)
        unload_button.click(unload_models, outputs=model_message, concurrency_limit=1)
        generate_button.click(
            generate,
            inputs=[
                style_image,
                content_image,
                prompt,
                negative_prompt,
                num_samples,
                resolution,
                resize_mode,
                style_scale,
                condition_scale,
                steps,
                guidance_scale,
                seed,
                scheduler,
                contour_threshold,
                control_start,
                control_end,
                guess_mode,
                enable_qkv,
                lambda_q,
                adain_strength,
                self_attention_start,
                self_attention_end,
                enable_style_cross_attention,
                style_cross_attention_start,
                style_cross_attention_end,
            ],
            outputs=[control_preview, result_gallery, saved_files, run_status, seed],
            concurrency_limit=1,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="StyleShot local Gradio interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio link")
    parser.add_argument("--inbrowser", action="store_true", help="Open the browser after launch")
    parser.add_argument("--check", action="store_true", help="Validate files and exit")
    args = parser.parse_args()

    validate_installation()
    if args.check:
        print(f"CHECK_OK content={len(CONTENT_FILES)} style_categories={len(STYLE_FILES)} style_images={sum(map(len, STYLE_FILES.values()))}")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    app = build_ui()
    app.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
        show_error=True,
        allowed_paths=[str(STYLEBENCH), str(OUTPUT_ROOT)],
    )


if __name__ == "__main__":
    main()
