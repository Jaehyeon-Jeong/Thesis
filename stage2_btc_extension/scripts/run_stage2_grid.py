#!/usr/bin/env python
"""Run a Stage 2 BTC experiment grid in Kaggle.

역할:
    Stage 2 BTC baseline을 여러 image window, return horizon, image spec,
    seed 조합으로 반복 실행한다. 실제 학습/평가/Grad-CAM 구현은 기존 Stage 2
    script들을 그대로 호출하고, 이 파일은 grid orchestration만 담당한다.

중요:
    이 script는 모델 구조를 바꾸지 않는다. 각 experiment는
    `run_stage2_btc_baseline.py`, `evaluate_stage2_predictions.py`,
    `evaluate_stage2_trading.py`, `generate_stage2_gradcam.py`,
    `check_stage2_outputs.py`를 순서대로 호출한다.

Output 보존:
    각 experiment/seed의 주요 단계가 끝날 때마다 `backup_root` 아래에 해당
    experiment만 zip으로 저장한다. Kaggle에서 다음 run이 project folder를 다시
    만들더라도 완료된 결과를 잃지 않기 위한 장치다.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


IMAGE_WINDOWS = (5, 20, 60)
RETURN_HORIZONS = (5, 20, 60)
IMAGE_SPECS = ("ohlc", "ohlc_vb", "ohlc_ma", "ohlc_ma_vb")


@dataclass
class RunRecord:
    """Grid run 한 줄의 실행 결과를 저장한다."""

    experiment_name: str
    image_window: int
    image_spec: str
    return_horizon: int
    run_seed: int
    status: str
    elapsed_seconds: float
    message: str


def parse_args() -> argparse.Namespace:
    """CLI 인자를 읽는다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env_kaggle.yaml")
    parser.add_argument("--image-windows", nargs="+", type=int, default=list(IMAGE_WINDOWS))
    parser.add_argument("--return-horizons", nargs="+", type=int, default=list(RETURN_HORIZONS))
    parser.add_argument("--image-specs", nargs="+", default=list(IMAGE_SPECS))
    parser.add_argument("--run-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--split", default="test")
    parser.add_argument("--gradcam-samples-per-class", type=int, default=2)
    parser.add_argument("--skip-gradcam", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--backup-root", default="/kaggle/working/stage2_saved_outputs")
    parser.add_argument("--summary-name", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-validation-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def experiment_name(image_window: int, image_spec: str, return_horizon: int) -> str:
    """Stage 2 experiment directory name을 만든다."""

    return f"stage2_i{image_window}_{image_spec}_r{return_horizon}"


def load_yaml(path: Path) -> dict[str, Any]:
    """YAML config를 읽는다."""

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_command(cmd: list[str], cwd: Path) -> None:
    """subprocess를 실행하고 stdout/stderr를 Kaggle cell에 즉시 흘린다."""

    print("\n$ " + " ".join(str(item) for item in cmd), flush=True)
    subprocess.run([str(item) for item in cmd], cwd=str(cwd), check=True)


def output_paths(output_root: Path, experiment: str, seed: int, split: str, samples_per_class: int) -> dict[str, Path]:
    """한 experiment/seed의 핵심 output path를 계산한다."""

    seed_name = f"seed_{seed}"
    return {
        "checkpoint": output_root / "checkpoints" / experiment / seed_name / "best.pt",
        "train_history": output_root / "metrics" / experiment / seed_name / "train_history.csv",
        "train_metadata": output_root / "metrics" / experiment / seed_name / "train_metadata.json",
        "predictions": output_root / "predictions" / experiment / seed_name / f"{split}_predictions.csv",
        "metrics": output_root / "metrics" / experiment / seed_name / f"{split}_metrics.json",
        "trading": output_root / "metrics" / experiment / seed_name / f"{split}_trading_metrics.json",
        "gradcam": (
            output_root
            / "figures"
            / experiment
            / seed_name
            / "gradcam"
            / split
            / f"btc_gradcam_{split}_{samples_per_class}perclass.png"
        ),
    }


def is_complete(paths: dict[str, Path], require_gradcam: bool) -> bool:
    """이미 완료된 experiment인지 판단한다."""

    required = ["checkpoint", "train_history", "train_metadata", "predictions", "metrics", "trading"]
    if require_gradcam:
        required.append("gradcam")
    return all(paths[name].exists() for name in required)


def backup_experiment_outputs(
    project_root: Path,
    output_root: Path,
    backup_root: Path,
    experiment: str,
    seed: int,
    phase: str,
) -> Path | None:
    """현재 experiment/seed 관련 output만 zip으로 저장한다.

    전체 `outputs/stage2`를 매번 압축하면 grid run에서 zip이 너무 커진다. 따라서
    checkpoint, metrics, predictions, figures, run manifest 중 현재 experiment/seed에
    해당하는 파일만 모은다.
    """

    seed_name = f"seed_{seed}"
    candidates = [
        output_root / "checkpoints" / experiment / seed_name,
        output_root / "metrics" / experiment / seed_name,
        output_root / "predictions" / experiment / seed_name,
        output_root / "figures" / experiment / seed_name,
        output_root / "run_manifests" / experiment / seed_name,
        project_root / "reports" / "figures" / "gradcam",
    ]
    files: list[Path] = []
    for candidate in candidates:
        if candidate.is_file():
            files.append(candidate)
        elif candidate.is_dir():
            for file_path in candidate.rglob("*"):
                if file_path.is_file() and (experiment in str(file_path) or "reports" not in file_path.parts):
                    files.append(file_path)

    if not files:
        print(f"[backup:{phase}] skip: no files yet for {experiment}/seed_{seed}", flush=True)
        return None

    backup_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    zip_path = backup_root / f"{experiment}_seed{seed}_{phase}_{timestamp}_outputs.zip"
    receipt_path = backup_root / f"{experiment}_seed{seed}_{phase}_{timestamp}_receipt.json"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(set(files)):
            try:
                arcname = file_path.relative_to(project_root)
            except ValueError:
                arcname = Path("external") / file_path.name
            archive.write(file_path, arcname=str(arcname))

    receipt = {
        "experiment": experiment,
        "run_seed": seed,
        "phase": phase,
        "created_utc": timestamp,
        "archive_path": str(zip_path),
        "archive_size_mb": round(zip_path.stat().st_size / (1024 * 1024), 3),
        "num_files": len(files),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(
        f"[backup:{phase}] saved {zip_path} ({receipt['archive_size_mb']} MB), "
        f"receipt={receipt_path}",
        flush=True,
    )
    return zip_path


def passthrough_limits(args: argparse.Namespace) -> list[str]:
    """smoke/diagnostic row limit 인자를 downstream script로 넘긴다."""

    result: list[str] = []
    if args.max_train_rows is not None:
        result.extend(["--max-train-rows", str(args.max_train_rows)])
    if args.max_validation_rows is not None:
        result.extend(["--max-validation-rows", str(args.max_validation_rows)])
    if args.max_test_rows is not None:
        result.extend(["--max-test-rows", str(args.max_test_rows)])
    return result


def main() -> int:
    """Stage 2 grid run entrypoint."""

    args = parse_args()
    stage_root = Path(__file__).resolve().parents[1]
    config_path = stage_root / args.config
    config = load_yaml(config_path)
    output_root = Path(config["paths"]["output_root"])
    backup_root = Path(args.backup_root)
    reports_root = Path(config["paths"].get("reports_root", stage_root / "reports"))
    tables_root = Path(config["paths"].get("tables_root", reports_root / "tables"))
    tables_root.mkdir(parents=True, exist_ok=True)

    run_gradcam = not args.skip_gradcam
    records: list[RunRecord] = []
    total = len(args.image_windows) * len(args.return_horizons) * len(args.image_specs) * len(args.run_seeds)
    counter = 0

    print(
        json.dumps(
            {
                "status": "start",
                "total_runs": total,
                "image_windows": args.image_windows,
                "return_horizons": args.return_horizons,
                "image_specs": args.image_specs,
                "run_seeds": args.run_seeds,
                "run_gradcam": run_gradcam,
                "skip_completed": args.skip_completed,
            },
            indent=2,
        ),
        flush=True,
    )

    limit_args = passthrough_limits(args)
    for seed in args.run_seeds:
        for image_window in args.image_windows:
            for return_horizon in args.return_horizons:
                for image_spec in args.image_specs:
                    counter += 1
                    experiment = experiment_name(image_window, image_spec, return_horizon)
                    paths = output_paths(
                        output_root=output_root,
                        experiment=experiment,
                        seed=seed,
                        split=args.split,
                        samples_per_class=args.gradcam_samples_per_class,
                    )
                    print(
                        f"\n[{counter}/{total}] experiment={experiment} seed={seed}",
                        flush=True,
                    )
                    start = time.time()

                    if args.skip_completed and is_complete(paths, require_gradcam=run_gradcam):
                        records.append(
                            RunRecord(
                                experiment,
                                image_window,
                                image_spec,
                                return_horizon,
                                seed,
                                "skipped_complete",
                                0.0,
                                "all required files already exist",
                            )
                        )
                        print("[skip] completed output already exists", flush=True)
                        continue

                    if args.dry_run:
                        records.append(
                            RunRecord(
                                experiment,
                                image_window,
                                image_spec,
                                return_horizon,
                                seed,
                                "dry_run",
                                0.0,
                                "not executed",
                            )
                        )
                        continue

                    try:
                        if not paths["checkpoint"].exists():
                            train_cmd = [
                                sys.executable,
                                "-u",
                                "scripts/run_stage2_btc_baseline.py",
                                "--config",
                                args.config,
                                "--image-window",
                                str(image_window),
                                "--image-spec",
                                image_spec,
                                "--return-horizon",
                                str(return_horizon),
                                "--run-seed",
                                str(seed),
                            ]
                            if args.max_epochs is not None:
                                train_cmd.extend(["--max-epochs", str(args.max_epochs)])
                            train_cmd.extend(limit_args)
                            run_command(train_cmd, cwd=stage_root)
                            backup_experiment_outputs(stage_root, output_root, backup_root, experiment, seed, "after_train")

                        if not paths["metrics"].exists() or not paths["predictions"].exists():
                            run_command(
                                [
                                    sys.executable,
                                    "-u",
                                    "scripts/evaluate_stage2_predictions.py",
                                    "--config",
                                    args.config,
                                    "--image-window",
                                    str(image_window),
                                    "--image-spec",
                                    image_spec,
                                    "--return-horizon",
                                    str(return_horizon),
                                    "--run-seed",
                                    str(seed),
                                    "--split",
                                    args.split,
                                ]
                                + limit_args,
                                cwd=stage_root,
                            )
                            backup_experiment_outputs(stage_root, output_root, backup_root, experiment, seed, "after_prediction_eval")

                        if not paths["trading"].exists():
                            run_command(
                                [
                                    sys.executable,
                                    "-u",
                                    "scripts/evaluate_stage2_trading.py",
                                    "--config",
                                    args.config,
                                    "--image-window",
                                    str(image_window),
                                    "--image-spec",
                                    image_spec,
                                    "--return-horizon",
                                    str(return_horizon),
                                    "--run-seed",
                                    str(seed),
                                    "--split",
                                    args.split,
                                ],
                                cwd=stage_root,
                            )
                            backup_experiment_outputs(stage_root, output_root, backup_root, experiment, seed, "after_trading_eval")

                        if run_gradcam and not paths["gradcam"].exists():
                            run_command(
                                [
                                    sys.executable,
                                    "-u",
                                    "scripts/generate_stage2_gradcam.py",
                                    "--config",
                                    args.config,
                                    "--image-window",
                                    str(image_window),
                                    "--image-spec",
                                    image_spec,
                                    "--return-horizon",
                                    str(return_horizon),
                                    "--run-seed",
                                    str(seed),
                                    "--split",
                                    args.split,
                                    "--samples-per-class",
                                    str(args.gradcam_samples_per_class),
                                    "--write-report-copy",
                                ]
                                + limit_args,
                                cwd=stage_root,
                            )
                            backup_experiment_outputs(stage_root, output_root, backup_root, experiment, seed, "after_gradcam")

                        if run_gradcam:
                            run_command(
                                [
                                    sys.executable,
                                    "-u",
                                    "scripts/check_stage2_outputs.py",
                                    "--config",
                                    args.config,
                                    "--image-window",
                                    str(image_window),
                                    "--image-spec",
                                    image_spec,
                                    "--return-horizon",
                                    str(return_horizon),
                                    "--run-seed",
                                    str(seed),
                                    "--split",
                                    args.split,
                                    "--gradcam-samples-per-class",
                                    str(args.gradcam_samples_per_class),
                                ],
                                cwd=stage_root,
                            )
                        backup_experiment_outputs(stage_root, output_root, backup_root, experiment, seed, "after_output_check")
                        records.append(
                            RunRecord(
                                experiment,
                                image_window,
                                image_spec,
                                return_horizon,
                                seed,
                                "ok",
                                round(time.time() - start, 3),
                                "",
                            )
                        )
                    except Exception as exc:  # noqa: BLE001 - grid should record failures.
                        backup_experiment_outputs(stage_root, output_root, backup_root, experiment, seed, "after_error")
                        records.append(
                            RunRecord(
                                experiment,
                                image_window,
                                image_spec,
                                return_horizon,
                                seed,
                                "error",
                                round(time.time() - start, 3),
                                repr(exc),
                            )
                        )
                        print(f"[error] {experiment}/seed_{seed}: {exc!r}", flush=True)
                        if not args.continue_on_error:
                            raise

    summary_name = args.summary_name or (
        "stage2_grid_five_seed" if len(args.run_seeds) > 1 else "stage2_grid_single_seed"
    )
    csv_path = tables_root / f"{summary_name}_run_summary.csv"
    json_path = tables_root / f"{summary_name}_run_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()) if records else list(RunRecord.__annotations__.keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    json_path.write_text(json.dumps([asdict(record) for record in records], indent=2), encoding="utf-8")
    print(json.dumps({"status": "done", "summary_csv": str(csv_path), "summary_json": str(json_path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
