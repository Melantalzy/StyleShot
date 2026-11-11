from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

import run_small_test_attention_experiments as experiment


def choose_image(title: str) -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError(
            "tkinter is unavailable; provide --content and --style on the command line"
        ) from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(
        title=title,
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not selected:
        raise SystemExit(f"Cancelled: {title}")
    return Path(selected).resolve()


def image_path(value: str | None, title: str) -> Path:
    path = Path(value).resolve() if value else choose_image(title)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in experiment.IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {path.suffix}")
    return path


def parse_args():
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Run phase 02, phase 03 and phase 04/R1 attention-timestep experiments "
            "for one content/style image pair. If image paths are omitted, file dialogs are shown."
        )
    )
    parser.add_argument("--content", default=None, help="Content image path")
    parser.add_argument("--style", default=None, help="Style image path")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "outputs" / "attention_timestep_pairs",
    )
    parser.add_argument("--run-name", default=None)
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
    parser.add_argument("--prompt", default=None, help="Optional custom positive prompt")
    parser.add_argument("--negative-prompt", default=experiment.gui.DEFAULT_NEGATIVE)
    parser.add_argument("--minimum-free-vram-gb", type=float, default=5.0)
    parser.add_argument("--ignore-vram-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    content_path = image_path(args.content, "选择内容图")
    style_path = image_path(args.style, "选择风格图")
    if content_path == style_path:
        raise ValueError("Content image and style image must be different files")

    if args.steps < 2:
        raise ValueError("--steps must be at least 2")
    if args.resolution < 64 or args.resolution % 64 != 0:
        raise ValueError("--resolution must be a multiple of 64")
    if not 0.0 <= args.lambda_q <= 1.0:
        raise ValueError("--lambda-q must be in [0, 1]")
    if not 0.0 <= args.adain_strength <= 1.0:
        raise ValueError("--adain-strength must be in [0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    experiment.check_free_vram(args.minimum_free_vram_gb, args.ignore_vram_check)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = (
        f"pair_{experiment.slug(content_path.stem)}__style_{experiment.slug(style_path.stem)}"
        f"__{timestamp}__steps{args.steps}__res{args.resolution}__lambdaq{args.lambda_q:.2f}"
    )
    run_dir = args.output_root.resolve() / experiment.slug(args.run_name or default_name)
    run_dir.mkdir(parents=True, exist_ok=False)

    case = experiment.TestCase(
        case_id="0",
        folder=content_path.parent,
        content_path=content_path,
        style_path=style_path,
    )
    schedules = experiment.schedules()
    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "single_content_style_pair",
        "content": str(content_path),
        "style": str(style_path),
        "output": str(run_dir),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "schedules": [asdict(schedule) for schedule in schedules],
        "notes": [
            "All 14 outputs share one DDIM inversion, one full QKV bank and one AdaIN latent.",
            "Phase 02 contains seven non-overlap schedules.",
            "Phase 03 contains six symmetric-overlap schedules.",
            "Phase 04/R1 uses self-attention first and style cross-attention second.",
        ],
    }
    experiment.save_json(run_dir / "run_config.json", run_config)

    print(f"Content: {content_path}")
    print(f"Style:   {style_path}")
    print(f"Output:  {run_dir}")
    print(f"Schedules: {len(schedules)}")
    print("\nLoading local StyleShot models...", flush=True)
    started = time.perf_counter()
    experiment.gui.load_models(experiment.ConsoleProgress())
    assert experiment.gui.MODEL is not None
    experiment.gui.MODEL.pipe.set_progress_bar_config(disable=True)

    rows: list[dict] = []
    duplicate_check = experiment.run_case(case, run_dir, schedules, args, rows)
    passed = sum(row.get("validation") == "PASS" for row in rows)
    warned = sum(row.get("validation") == "WARN" for row in rows)
    failed = sum(row.get("validation") == "FAIL" for row in rows)
    summary = {
        "run_dir": str(run_dir),
        "expected_outputs": len(schedules),
        "manifest_rows": len(rows),
        "passed": passed,
        "warnings": warned,
        "failed": failed,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "determinism_check": duplicate_check,
    }
    experiment.save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nResults: {run_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
