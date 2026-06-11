#!/usr/bin/env python
"""Training entrypoint for RUN_0003a (nnU-Net v2).

Orchestrates the full nnU-Net v2 CLI workflow:

1. Prepare dataset in nnU-Net raw format.
2. Run ``nnUNetv2_plan_and_preprocess``.
3. Run ``nnUNetv2_train``.
4. Symlink the best checkpoint into ``outputs/checkpoints/RUN_0003a/``.

Exit codes follow project conventions (0 = success, 1 = error).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _nnunet_env(config: dict) -> dict:
    """Return a copy of ``os.environ`` with nnU-Net variables set.

    Uses existing shell variables when present; otherwise defaults to
    project-local directories so that local execution is self-contained.
    """
    env = os.environ.copy()
    out_cfg = config.get("output", {})

    if "nnUNet_raw" not in env:
        env["nnUNet_raw"] = str(PROJECT_ROOT / "nnUNet_raw")
    if "nnUNet_preprocessed" not in env:
        env["nnUNet_preprocessed"] = str(PROJECT_ROOT / "nnUNet_preprocessed")
    if "nnUNet_results" not in env:
        env["nnUNet_results"] = str(PROJECT_ROOT / "nnUNet_results")

    return env


def _run(cmd: list[str], env: dict | None = None, check: bool = True) -> int:
    """Print and run a subprocess command, exiting on failure."""
    print(f"  $ {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, env=env, check=check)
        return result.returncode
    except subprocess.CalledProcessError as exc:
        print(f"\n[ERROR] Command failed (return code {exc.returncode}).")
        sys.exit(1)
    except FileNotFoundError:
        print(
            f"\n[ERROR] Command not found: {cmd[0]}. "
            "Is nnunetv2 installed? (pip install nnunetv2)"
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RUN_0003a nnU-Net v2 training launcher."
    )
    parser.add_argument("--config", required=True, help="Path to config YAML.")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", dest="smoke_test_dash")
    args, _ = parser.parse_known_args()

    with open(args.config, "r") as fh:
        config = yaml.safe_load(fh) or {}

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    out_cfg = config.get("output", {})

    dataset_id = int(data_cfg.get("nnunet_dataset_id", 501))
    dataset_name = data_cfg.get("nnunet_dataset_name", "LISA2026_Task2")
    configuration = model_cfg.get("configuration", "3d_fullres")
    fold = int(model_cfg.get("fold", 0))

    data_root = data_cfg.get("data_root", os.environ.get("LISA_DATA_ROOT"))
    split_pkl = data_cfg.get("split_pkl", "results/splits/task2_fixed.pkl")

    if not data_root:
        print(
            "[ERROR] No data root found. Set data.data_root in config or "
            "LISA_DATA_ROOT environment variable."
        )
        sys.exit(1)

    env = _nnunet_env(config)
    raw_dir = Path(env["nnUNet_raw"])
    results_dir = Path(env["nnUNet_results"])

    out_checkpoint_dir = Path(out_cfg.get("checkpoint_dir", "outputs/checkpoints/RUN_0003a"))

    print("=" * 60)
    print("  RUN_0003a — nnU-Net v2 Training")
    print(f"  Dataset  : {dataset_id:03d} ({dataset_name})")
    print(f"  Config   : {configuration}")
    print(f"  Fold     : {fold}")
    print(f"  Data     : {data_root}")
    print(f"  nnUNet_raw          : {raw_dir}")
    print(f"  nnUNet_preprocessed : {env['nnUNet_preprocessed']}")
    print(f"  nnUNet_results      : {results_dir}")
    print("=" * 60)

    # ── 1. Prepare dataset ─────────────────────────────────────────────────
    dataset_root = raw_dir / f"Dataset{dataset_id:03d}_{dataset_name}"
    if not (dataset_root / "dataset.json").exists():
        print("\n── Preparing nnU-Net dataset ──────────────────────────────")
        _run(
            [
                sys.executable,
                "-m",
                "src.prepare_nnunet_dataset",
                "--data-root",
                str(data_root),
                "--split-pkl",
                str(split_pkl),
                "--output-root",
                str(dataset_root),
            ],
            env=env,
        )
    else:
        print(f"\n[INFO] Dataset already prepared: {dataset_root}")

    # ── 2. Plan & preprocess ───────────────────────────────────────────────
    preprocessed_flag = Path(env["nnUNet_preprocessed"]) / f"Dataset{dataset_id:03d}_{dataset_name}"
    if not preprocessed_flag.exists():
        print("\n── nnUNetv2_plan_and_preprocess ───────────────────────────")
        _run(
            [
                "nnUNetv2_plan_and_preprocess",
                "-d",
                str(dataset_id),
                "--verify_dataset_integrity",
                "-np",
                "8",
                "-nps",
                "8",
            ],
            env=env,
        )
    else:
        print(f"\n[INFO] Preprocessing already done: {preprocessed_flag}")

    # ── 3. Train ───────────────────────────────────────────────────────────
    print("\n── nnUNetv2_train ─────────────────────────────────────────")
    trainer = model_cfg.get("trainer", "nnUNetTrainer")
    plans = model_cfg.get("plans", "nnUNetPlans")
    _run(
        [
            "nnUNetv2_train",
            str(dataset_id),
            configuration,
            str(fold),
        ],
        env=env,
    )

    # ── 4. Checkpoint symlink ──────────────────────────────────────────────
    checkpoint_source = (
        results_dir
        / f"Dataset{dataset_id:03d}_{dataset_name}"
        / f"{trainer}__{plans}__{configuration}"
        / f"fold_{fold}"
        / "checkpoint_final.pth"
    )
    out_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_link = out_checkpoint_dir / "task2_nnunet_best.pt"

    if checkpoint_source.exists():
        if checkpoint_link.exists() or checkpoint_link.is_symlink():
            checkpoint_link.unlink()
        checkpoint_link.symlink_to(checkpoint_source.resolve())
        print(f"\n[INFO] Checkpoint linked: {checkpoint_link} -> {checkpoint_source}")

    print("\n[OK] Training completed.")


if __name__ == "__main__":
    main()
