from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

import styleshot_local_gui as gui
from ip_adapter.self_attention_injection import blend_adain_latent


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PHASE_NAMES = {
    "02": "phase_02_nonoverlap",
    "03": "phase_03_symmetric_overlap",
    "04_R1": "phase_04_R1_reverse_order",
}
MANIFEST_FIELDS = [
    "case_id",
    "content_name",
    "style_name",
    "phase",
    "config_id",
    "description",
    "output_file",
    "cross_window",
    "self_window",
    "expected_cross_steps",
    "actual_cross_steps",
    "expected_self_steps",
    "actual_self_steps",
    "qkv_missing_features",
    "width",
    "height",
    "pixel_mean",
    "pixel_std",
    "pixel_p01",
    "pixel_p99",
    "sha256",
    "elapsed_seconds",
    "validation",
    "messages",
]


@dataclass(frozen=True)
class Schedule:
    phase: str
    config_id: str
    description: str
    cross_enabled: bool
    cross_start: float
    cross_end: float
    self_enabled: bool
    self_start: float
    self_end: float

    def active_steps(self, mechanism: str, total_steps: int) -> list[int]:
        enabled = self.cross_enabled if mechanism == "cross" else self.self_enabled
        if not enabled:
            return []
        start = self.cross_start if mechanism == "cross" else self.self_start
        end = self.cross_end if mechanism == "cross" else self.self_end
        first = min(total_steps, math.floor(start * total_steps))
        stop = min(total_steps, math.ceil(end * total_steps))
        return list(range(first + 1, stop + 1))


@dataclass(frozen=True)
class TestCase:
    case_id: str
    folder: Path
    content_path: Path
    style_path: Path


class ConsoleProgress:
    def __call__(self, value, desc=None, **_kwargs):
        if desc:
            print(f"[model {float(value):.0%}] {desc}", flush=True)


def slug(text: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_-]+", "_", text.strip()).strip("_")
    return value or "image"


def schedules() -> list[Schedule]:
    result: list[Schedule] = []

    # Phase 02: cross attention first, self-attention injection second.
    for index, split in enumerate((0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)):
        result.append(
            Schedule(
                phase="02",
                config_id=f"S{index}",
                description=f"nonoverlap split={split:.2f}",
                cross_enabled=split > 0.0,
                cross_start=0.0,
                cross_end=split,
                self_enabled=split < 1.0,
                self_start=split,
                self_end=1.0,
            )
        )

    # Phase 03: equal-duration, center-symmetric windows. O0 duplicates S3
    # intentionally and is used as a deterministic reset consistency check.
    for index, duration in enumerate((0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
        result.append(
            Schedule(
                phase="03",
                config_id=f"O{index}",
                description=f"symmetric duration={duration:.2f}",
                cross_enabled=True,
                cross_start=0.0,
                cross_end=duration,
                self_enabled=True,
                self_start=round(1.0 - duration, 10),
                self_end=1.0,
            )
        )

    # Phase 04 R1: hypothesis direction reversed: self early, cross late.
    result.append(
        Schedule(
            phase="04_R1",
            config_id="R1",
            description="reverse order: self first half, cross second half",
            cross_enabled=True,
            cross_start=0.5,
            cross_end=1.0,
            self_enabled=True,
            self_start=0.0,
            self_end=0.5,
        )
    )
    return result


def discover_cases(input_root: Path, selected: str) -> list[TestCase]:
    wanted = None
    if selected.strip().lower() != "all":
        wanted = {item.strip() for item in selected.split(",") if item.strip()}

    cases: list[TestCase] = []
    for folder in sorted(
        (item for item in input_root.iterdir() if item.is_dir() and item.name != "results"),
        key=lambda item: (not item.name.isdigit(), int(item.name) if item.name.isdigit() else item.name),
    ):
        if wanted is not None and folder.name not in wanted:
            continue
        images = [
            item for item in folder.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        ]
        styles = [item for item in images if item.stem.lower() == "style"]
        contents = [item for item in images if item.stem.lower() != "style"]
        if len(styles) != 1 or len(contents) != 1:
            raise ValueError(
                f"{folder}: expected one content image and one style image; "
                f"found content={len(contents)}, style={len(styles)}"
            )
        cases.append(TestCase(folder.name, folder, contents[0], styles[0]))

    if not cases:
        raise ValueError(f"No test cases found under {input_root}")
    return cases


def range_name(schedule: Schedule, mechanism: str, total_steps: int) -> str:
    steps = schedule.active_steps(mechanism, total_steps)
    if not steps:
        return "off"
    return f"{steps[0]:03d}-{steps[-1]:03d}"


def output_filename(schedule: Schedule, total_steps: int) -> str:
    cross_steps = schedule.active_steps("cross", total_steps)
    self_steps = schedule.active_steps("self", total_steps)
    overlap = len(set(cross_steps).intersection(self_steps))
    return (
        f"{schedule.config_id}__cross_{range_name(schedule, 'cross', total_steps)}"
        f"__self_{range_name(schedule, 'self', total_steps)}"
        f"__overlap_{overlap:03d}.png"
    )


def steps_text(steps: list[int]) -> str:
    if not steps:
        return "off"
    return f"{steps[0]}-{steps[-1]} ({len(steps)})"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output(
    path: Path,
    expected_size: tuple[int, int],
    expected_cross: list[int],
    actual_cross: list[int],
    expected_self: list[int],
    actual_self: list[int],
    missing_features: int,
) -> tuple[dict, str, list[str]]:
    messages: list[str] = []
    status = "PASS"
    try:
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.float32)
            size = rgb.size
    except Exception as exc:
        return {}, "FAIL", [f"image decode failed: {exc}"]

    if size != expected_size:
        status = "FAIL"
        messages.append(f"size {size} != expected {expected_size}")
    if expected_cross != actual_cross:
        status = "FAIL"
        messages.append("cross-attention active steps mismatch")
    if expected_self != actual_self:
        status = "FAIL"
        messages.append("self-attention active steps mismatch")
    if missing_features:
        status = "FAIL"
        messages.append(f"missing QKV features={missing_features}")

    pixel_std = float(array.std())
    p01, p99 = (float(value) for value in np.percentile(array, (1, 99)))
    if pixel_std < 2.0 or p99 - p01 < 10.0:
        if status == "PASS":
            status = "WARN"
        messages.append("possible blank or near-uniform output")

    stats = {
        "width": size[0],
        "height": size[1],
        "pixel_mean": round(float(array.mean()), 4),
        "pixel_std": round(pixel_std, 4),
        "pixel_p01": round(p01, 4),
        "pixel_p99": round(p99, 4),
        "sha256": sha256_file(path),
    }
    return stats, status, messages


def write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(items: list[tuple[str, Image.Image]], output_path: Path, columns: int = 4) -> None:
    if not items:
        return
    thumb_size = 256
    label_height = 34
    rows = math.ceil(len(items) / columns)
    canvas = Image.new("RGB", (columns * thumb_size, rows * (thumb_size + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(items):
        row, column = divmod(index, columns)
        thumb = ImageOps.contain(image.convert("RGB"), (thumb_size, thumb_size), Image.Resampling.LANCZOS)
        x = column * thumb_size + (thumb_size - thumb.width) // 2
        y = row * (thumb_size + label_height) + (thumb_size - thumb.height) // 2
        canvas.paste(thumb, (x, y))
        draw.text((column * thumb_size + 6, row * (thumb_size + label_height) + thumb_size + 8), label, fill="black")
    canvas.save(output_path, quality=92)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def check_free_vram(minimum_gb: float, ignore: bool) -> None:
    if ignore:
        return
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / 1024**3
    total_gb = total_bytes / 1024**3
    if free_gb < minimum_gb:
        raise RuntimeError(
            f"Free VRAM is {free_gb:.2f}/{total_gb:.2f} GB; at least {minimum_gb:.2f} GB is required. "
            "Close or unload the running StyleShot GUI, then run this script again."
        )


def run_case(
    case: TestCase,
    run_dir: Path,
    experiment_schedules: list[Schedule],
    args,
    manifest_rows: list[dict],
) -> dict:
    assert gui.MODEL is not None and gui.DETECTOR is not None
    assert gui.QKV_CONTROLLER is not None and gui.QKV_INVERTER is not None
    model = gui.MODEL
    controller = gui.QKV_CONTROLLER
    inverter = gui.QKV_INVERTER

    case_slug = slug(case.content_path.stem)
    case_number = f"{int(case.case_id):02d}" if case.case_id.isdigit() else slug(case.case_id)
    case_dir = run_dir / f"case_{case_number}__{case_slug}"
    inputs_dir = case_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    for phase_name in PHASE_NAMES.values():
        (case_dir / phase_name).mkdir(parents=True, exist_ok=True)

    content_source = Image.open(case.content_path).convert("RGB")
    style_source = Image.open(case.style_path).convert("RGB")
    content_array = gui.prepare_content(content_source, args.resolution, "中心裁剪为正方形")
    content_image = Image.fromarray(content_array).convert("RGB")
    style_fitted = ImageOps.fit(
        style_source,
        content_image.size,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    control_array = gui.DETECTOR(content_array, threshold=args.contour_threshold)
    control_image = Image.fromarray(control_array).convert("RGB")

    content_source.save(inputs_dir / "content_source.png")
    content_image.save(inputs_dir / "content_processed.png")
    style_source.save(inputs_dir / "style_source.png")
    style_fitted.save(inputs_dir / "style_fitted.png")
    control_image.save(inputs_dir / "control.png")

    custom_prompt = getattr(args, "prompt", None)
    prompt = custom_prompt or (
        f"{case.content_path.stem}, {gui.DEFAULT_POSITIVE}"
        if args.prompt_mode == "filename"
        else gui.DEFAULT_POSITIVE
    )

    print(f"\n[{case.case_id}] {case.content_path.name} + {case.style_path.name}", flush=True)
    print("  inverting style/content once and recording the full Q/K/V bank...", flush=True)
    controller.clear()
    controller.configure_injection_window(args.steps, 0.0, 1.0)
    style_latent = inverter.invert(style_fitted, args.steps, target="style", prompt="")
    content_latent = inverter.invert(content_image, args.steps, target="content", prompt="")
    adain_latent = blend_adain_latent(content_latent, style_latent, args.adain_strength)
    feature_summary = controller.bank.summary()
    expected_features = 6 * args.steps
    if any(feature_summary[key] != expected_features for key in ("content_q", "style_k", "style_v")):
        raise RuntimeError(
            f"Incomplete full feature bank for case {case.case_id}: {feature_summary}; "
            f"expected {expected_features} per feature type"
        )

    torch.save(
        {
            "content_latent": content_latent.detach().cpu().half(),
            "style_latent": style_latent.detach().cpu().half(),
            "adain_latent": adain_latent.detach().cpu().half(),
        },
        inputs_dir / "inversion_latents.pt",
    )
    style_tokens, uncond_style_tokens = model.get_image_embeds(style_source)

    case_meta = {
        "case_id": case.case_id,
        "content_source": str(case.content_path),
        "style_source": str(case.style_path),
        "prompt": prompt,
        "negative_prompt": args.negative_prompt,
        "feature_bank": feature_summary,
        "processed_size": list(content_image.size),
    }
    save_json(case_dir / "case_config.json", case_meta)

    contact_items: list[tuple[str, Image.Image]] = [
        ("CONTENT", content_image.copy()),
        ("STYLE", style_fitted.copy()),
    ]
    phase_items: dict[str, list[tuple[str, Image.Image]]] = {key: [] for key in PHASE_NAMES}
    output_hashes: dict[str, str] = {}

    for schedule in experiment_schedules:
        phase_dir = case_dir / PHASE_NAMES[schedule.phase]
        filename = output_filename(schedule, args.steps)
        output_path = phase_dir / filename
        expected_cross = schedule.active_steps("cross", args.steps)
        expected_self = schedule.active_steps("self", args.steps)
        print(
            f"  {schedule.phase}/{schedule.config_id}: "
            f"cross={steps_text(expected_cross)}, self={steps_text(expected_self)}",
            flush=True,
        )
        started = time.perf_counter()
        row = {
            "case_id": case.case_id,
            "content_name": case.content_path.name,
            "style_name": case.style_path.name,
            "phase": schedule.phase,
            "config_id": schedule.config_id,
            "description": schedule.description,
            "output_file": str(output_path.relative_to(run_dir)),
            "cross_window": f"{schedule.cross_start:.2f}-{schedule.cross_end:.2f}" if schedule.cross_enabled else "off",
            "self_window": f"{schedule.self_start:.2f}-{schedule.self_end:.2f}" if schedule.self_enabled else "off",
            "expected_cross_steps": steps_text(expected_cross),
            "expected_self_steps": steps_text(expected_self),
        }
        try:
            model.set_scale(args.style_scale)
            model.pipe.scheduler = inverter.make_scheduler()
            if schedule.self_enabled:
                controller.inject(
                    args.lambda_q,
                    total_steps=args.steps,
                    injection_start=schedule.self_start,
                    injection_end=schedule.self_end,
                )
            else:
                controller.disable()
                controller.applied_steps.clear()
                controller.missing_features.clear()

            images = model.samples(
                style_tokens,
                uncond_style_tokens,
                1,
                model.device,
                prompt,
                args.negative_prompt,
                args.seed,
                args.guidance_scale,
                args.steps,
                control_image,
                latents=adain_latent.clone(),
                controlnet_conditioning_scale=args.condition_scale,
                control_guidance_start=0.0,
                control_guidance_end=1.0,
                guess_mode=False,
                style_cross_attention_enabled=schedule.cross_enabled,
                style_cross_attention_start=schedule.cross_start,
                style_cross_attention_end=schedule.cross_end,
            )
            image = images[0]
            image.save(output_path)
            actual_self = [step + 1 for step in sorted(controller.applied_steps)]
            actual_cross = [
                step + 1
                for step in getattr(model.pipe, "_last_style_cross_attention_steps", [])
            ]
            missing = len(controller.missing_features)
            stats, validation, messages = validate_output(
                output_path,
                content_image.size,
                expected_cross,
                actual_cross,
                expected_self,
                actual_self,
                missing,
            )
            row.update(stats)
            row.update(
                {
                    "actual_cross_steps": steps_text(actual_cross),
                    "actual_self_steps": steps_text(actual_self),
                    "qkv_missing_features": missing,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "validation": validation,
                    "messages": "; ".join(messages),
                }
            )
            output_hashes[schedule.config_id] = stats["sha256"]
            contact_items.append((schedule.config_id, image.copy()))
            phase_items[schedule.phase].append((schedule.config_id, image.copy()))
        except Exception as exc:
            row.update(
                {
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "validation": "FAIL",
                    "messages": f"generation failed: {type(exc).__name__}: {exc}",
                }
            )
            if isinstance(exc, torch.OutOfMemoryError):
                manifest_rows.append(row)
                write_manifest(run_dir / "manifest.csv", manifest_rows)
                raise
        finally:
            controller.disable()

        manifest_rows.append(row)
        write_manifest(run_dir / "manifest.csv", manifest_rows)

    make_contact_sheet(contact_items, case_dir / "comparison_all.jpg")
    for phase, items in phase_items.items():
        make_contact_sheet(items, case_dir / f"comparison_{PHASE_NAMES[phase]}.jpg")

    duplicate_check = {
        "S3_vs_O0_identical": output_hashes.get("S3") == output_hashes.get("O0"),
        "S3_sha256": output_hashes.get("S3"),
        "O0_sha256": output_hashes.get("O0"),
    }
    save_json(case_dir / "determinism_check.json", duplicate_check)
    controller.clear()
    gc.collect()
    return duplicate_check


def parse_args():
    default_input = Path(__file__).resolve().parent.parent / "small_test"
    parser = argparse.ArgumentParser(
        description="Run StyleShot attention-timestep experiments on small_test cases."
    )
    parser.add_argument("--input-root", type=Path, default=default_input)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--cases", default="all", help="all or comma-separated case IDs, e.g. 0,3,6")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--lambda-q", type=float, default=0.7)
    parser.add_argument("--adain-strength", type=float, default=1.0)
    parser.add_argument("--style-scale", type=float, default=1.0)
    parser.add_argument("--condition-scale", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--contour-threshold", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-mode", choices=("filename", "generic"), default="filename")
    parser.add_argument("--prompt", default=None, help="Optional prompt overriding --prompt-mode")
    parser.add_argument("--negative-prompt", default=gui.DEFAULT_NEGATIVE)
    parser.add_argument("--minimum-free-vram-gb", type=float, default=5.0)
    parser.add_argument("--ignore-vram-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List cases/configurations without loading models")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else args.input_root / "results" / "attention_timestep"
    )
    experiment_schedules = schedules()
    cases = discover_cases(args.input_root, args.cases)

    print(f"Cases: {len(cases)}; schedules per case: {len(experiment_schedules)}")
    for case in cases:
        print(f"  case {case.case_id}: {case.content_path.name} + {case.style_path.name}")
    for schedule in experiment_schedules:
        print(
            f"  {schedule.phase}/{schedule.config_id}: "
            f"cross={steps_text(schedule.active_steps('cross', args.steps))}; "
            f"self={steps_text(schedule.active_steps('self', args.steps))}"
        )
    if args.dry_run:
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    check_free_vram(args.minimum_free_vram_gb, args.ignore_vram_check)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or (
        f"run_{timestamp}__steps{args.steps}__res{args.resolution}__lambdaq{args.lambda_q:.2f}"
    )
    run_dir = args.output_root / slug(run_name)
    run_dir.mkdir(parents=True, exist_ok=False)

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(gui.ROOT),
        "input_root": str(args.input_root),
        "output_root": str(run_dir),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "cases": [
            {
                "case_id": case.case_id,
                "content": str(case.content_path),
                "style": str(case.style_path),
            }
            for case in cases
        ],
        "schedules": [asdict(schedule) for schedule in experiment_schedules],
        "notes": [
            "All schedules use the same per-case DDIM inversion and AdaIN latent.",
            "Cross attention means StyleShot image-style condition; text cross-attention remains enabled.",
            "Self attention means cross-image content-Q/style-KV injection in six up-block attn1 layers.",
        ],
    }
    save_json(run_dir / "run_config.json", run_config)

    manifest_rows: list[dict] = []
    duplicate_checks: dict[str, dict] = {}
    started = time.perf_counter()
    print("\nLoading local StyleShot models...", flush=True)
    gui.load_models(ConsoleProgress())
    assert gui.MODEL is not None
    gui.MODEL.pipe.set_progress_bar_config(disable=True)

    for case in cases:
        duplicate_checks[case.case_id] = run_case(
            case, run_dir, experiment_schedules, args, manifest_rows
        )

    passed = sum(row.get("validation") == "PASS" for row in manifest_rows)
    warned = sum(row.get("validation") == "WARN" for row in manifest_rows)
    failed = sum(row.get("validation") == "FAIL" for row in manifest_rows)
    summary = {
        "run_dir": str(run_dir),
        "expected_outputs": len(cases) * len(experiment_schedules),
        "manifest_rows": len(manifest_rows),
        "passed": passed,
        "warnings": warned,
        "failed": failed,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "determinism_checks": duplicate_checks,
    }
    save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nResults: {run_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
