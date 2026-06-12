"""Automatic run discovery from YAML configuration files.

Scans ``configs/run_*.yaml`` and builds a unified registry of training and
evaluation runs.  No hard-coded mappings are needed — each YAML file is the
single source of truth for its run.

Usage
-----
    from src.cli.registry import discover_runs, list_runs

    registry = discover_runs()
"""

from __future__ import annotations

from functools import lru_cache
import logging
import sys
import copy
import importlib.util
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # lisa2026/
CONFIG_DIR = PROJECT_ROOT / "configs"

# Mapping from (task, model_type_or_none) → (train_module, eval_module)
# model_type is extracted from config["model"]["type"] when present.
MODULE_MAP: dict[tuple[str, str | None], tuple[str, str]] = {
    ("1a", None): ("src.runners.train_task1a", "src.runners.evaluate_task1a"),
    ("1b", None): ("src.runners.train_task1b", "src.runners.evaluate_task1b"),
    ("2", None): ("src.runners.train_task2_dynunet", "src.runners.evaluate_task2_dynunet"),
    ("2", "nnunetv2"): ("src.runners.train_task2_nnunet", "src.runners.evaluate_task2_nnunet"),
    ("2", "medsam2"): ("src.runners.train_task2_medsam2", "src.runners.evaluate_task2_medsam2"),
    ("1a+1b+2", None): ("src.runners.train_multitask", "src.runners.evaluate_multitask"),
}

# Suffixes to skip (Jean Zay overrides — not distinct runs)
IGNORE_SUFFIXES = ("_jeanzay",)

# Official Task 1a artifact names (used to infer task from legacy configs)
ARTIFACT_TASK_NAMES = {
    "Noise",
    "Zipper",
    "Positioning",
    "Banding",
    "Motion",
    "Contrast",
    "Distortion",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_run_id_from_filename(path: Path) -> str | None:
    """Derive run_id from a config filename.

    Examples
    --------
    >>> _extract_run_id_from_filename(Path("run_0001_baseline.yaml"))
    '0001'
    >>> _extract_run_id_from_filename(Path("run_0003a_task2_nnunet.yaml"))
    '0003a'
    >>> _extract_run_id_from_filename(Path("run_0002_jeanzay.yaml"))
    None   (ignored)
    """
    stem = path.stem  # e.g. "run_0001_baseline"
    if stem.lower().endswith(IGNORE_SUFFIXES):
        return None
    # Strip "run_" prefix
    if stem.lower().startswith("run_"):
        rest = stem[4:]
        # Take the first segment before "_" (e.g. "0001" from "0001_baseline")
        return rest.split("_")[0]
    return None


def _extract_run_id_from_config(config: dict[str, Any]) -> str | None:
    """Try to get run_id from the config dict itself."""
    # Top-level key
    if "run_id" in config:
        return str(config["run_id"])
    # Nested under "run"
    if isinstance(config.get("run"), dict):
        return str(config["run"].get("id", ""))
    return None


def _extract_run_name(config: dict[str, Any]) -> str:
    """Try to get run_name from the config dict itself."""
    if "run_name" in config:
        return str(config["run_name"])
    if isinstance(config.get("run"), dict):
        name = config["run"].get("name", "")
        if name:
            return str(name)
    return ""


def _extract_task(config: dict[str, Any], filename: str) -> str:
    """Derive the task string from config or filename."""
    if "task" in config:
        return str(config["task"])
    if "tasks" in config:
        tasks = config["tasks"]
        if isinstance(tasks, list):
            # If tasks is a list of official artifact names, it's Task 1a.
            if all(str(t) in ARTIFACT_TASK_NAMES for t in tasks):
                return "1a"
            return "+".join(str(t) for t in tasks)
        return str(tasks)
    # Fallback from filename
    name_lower = filename.lower()
    if "task1a" in name_lower or "baseline" in name_lower:
        return "1a"
    if "task1b" in name_lower or "cyclegan" in name_lower:
        return "1b"
    if "task2" in name_lower:
        return "2"
    if "multitask" in name_lower:
        return "1a+1b+2"
    return "unknown"


def _extract_model_type(config: dict[str, Any]) -> str | None:
    """Extract model type from config (e.g. 'nnunetv2', 'medsam2', 'hybrid')."""
    model = config.get("model", {})
    if isinstance(model, dict):
        return model.get("type")
    return None


def _infer_modules(task: str, model_type: str | None) -> tuple[str, str] | None:
    """Look up (train_module, eval_module) from MODULE_MAP."""
    key = (task, model_type)
    result = MODULE_MAP.get(key)
    if result is None:
        logger.warning(
            "No module mapping for task=%r model_type=%r — run will be skipped.",
            task,
            model_type,
        )
    return result


def _module_exists(module_path: str) -> bool:
    """Return True if an importable module exists for *module_path*."""
    return importlib.util.find_spec(module_path) is not None


def _infer_mode(config: dict[str, Any]) -> str:
    """Determine run mode: 'per_task', 'nnunet', or 'single'."""
    if "tasks" in config and isinstance(config["tasks"], list):
        return "per_task"
    model = config.get("model", {})
    if isinstance(model, dict) and model.get("type") == "nnunetv2":
        return "nnunet"
    return "single"


def _parse_config(path: Path) -> dict[str, Any] | None:
    """Load and return a YAML config, or None on error."""
    try:
        with open(path, "r") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.warning("Failed to load %s: %s", path.name, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _discover_raw() -> dict[str, dict[str, Any]]:
    """Internal: scan configs and return raw dict[run_id, entry].

    Each entry has keys: run_id, task, module, eval_module, config, mode,
    tasks, supports_smoke_test, run_name, date, description.
    """
    registry: dict[str, dict[str, Any]] = {}

    if not CONFIG_DIR.is_dir():
        logger.error("Config directory not found: %s", CONFIG_DIR)
        return registry

    yaml_files = sorted(CONFIG_DIR.glob("run_*.yaml"))
    if not yaml_files:
        logger.warning("No config files matching 'run_*.yaml' in %s", CONFIG_DIR)
        return registry

    for config_path in yaml_files:
        # Skip Jean Zay overrides
        if config_path.stem.lower().endswith(IGNORE_SUFFIXES):
            continue

        config = _parse_config(config_path)
        if config is None:
            continue

        # Derive run_id
        run_id = _extract_run_id_from_config(config)
        if run_id is None:
            run_id = _extract_run_id_from_filename(config_path)
        if run_id is None:
            logger.warning("Cannot derive run_id from %s — skipping.", config_path.name)
            continue

        # Check for duplicates
        if run_id in registry:
            logger.warning(
                "Duplicate run_id '%s' — %s overrides %s",
                run_id,
                config_path.name,
                registry[run_id]["config"],
            )

        task = _extract_task(config, config_path.name)
        model_type = _extract_model_type(config)
        modules = _infer_modules(task, model_type)
        if modules is None:
            logger.warning("Skipping run '%s' — no module mapping found.", run_id)
            continue
        train_module, eval_module = modules

        if not _module_exists(train_module):
            logger.warning(
                "Skipping run '%s' — training module '%s' not found.",
                run_id,
                train_module,
            )
            continue
        if not _module_exists(eval_module):
            logger.warning(
                "Skipping run '%s' — evaluation module '%s' not found.",
                run_id,
                eval_module,
            )
            continue

        entry: dict[str, Any] = {
            "run_id": run_id,
            "task": task,
            "module": train_module,
            "eval_module": eval_module,
            "config": str(config_path),
            "mode": _infer_mode(config),
            "tasks": config.get("tasks", []),
            "supports_smoke_test": model_type != "nnunetv2",
            "run_name": _extract_run_name(config),
            "date": config.get("date", ""),
            "description": config.get("description", ""),
        }
        registry[run_id] = entry

    return registry


def discover_runs() -> dict[str, dict[str, Any]]:
    """Discover training/evaluation runs from YAML configs.

    Returns a dict mapping run_id → entry dict.
    """
    # Return a deep copy so callers cannot mutate the cached registry.
    return copy.deepcopy(_discover_raw())


def discover_eval_runs() -> dict[str, dict[str, Any]]:
    """Alias for :func:`discover_runs`.

    Kept for backward compatibility — both training and evaluation share
    the same registry because every run carries ``module`` and ``eval_module``.
    """
    return discover_runs()


def list_runs() -> None:
    """Pretty-print the discovered runs to stdout."""
    registry = _discover_raw()
    if not registry:
        print("[WARNING] No runs discovered.", file=sys.stderr)
        return

    # Column widths
    id_w = max(len(r) for r in registry) if registry else 4
    id_w = max(id_w, 4)
    task_w = 6
    name_w = max(len(e.get("run_name", "")) for e in registry.values()) if registry else 10
    name_w = max(name_w, 10)
    date_w = 12

    header = f"{'Run':>{id_w}}  {'Task':>{task_w}}  {'Name':<{name_w}}  {'Date':>{date_w}}"
    print(header)
    print("-" * len(header))
    for run_id in sorted(registry.keys()):
        entry = registry[run_id]
        name = entry.get("run_name", "—") or "—"
        date = entry.get("date", "—") or "—"
        print(f"{run_id:>{id_w}}  {entry['task']:>{task_w}}  {name:<{name_w}}  {date:>{date_w}}")


# ---------------------------------------------------------------------------
# CLI entry-point (python -m src.cli.registry)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    list_runs()
