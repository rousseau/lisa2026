"""Standardised metrics.json I/O for all LISA 2026 tasks.

Every evaluator should produce an envelope with at least:

    {
      "run_id": "0001",
      "task": "task1a",
      "model": "ordinal",
      "date": "2026-06-10T14:30:00",
      "status": "final",
      "global": { ... },
      "per_class": { ... }   // optional
    }
"""

import datetime
import json
import os
from typing import Any


# Required keys for a compliant metrics.json
REQUIRED_KEYS = {"run_id", "task", "date", "global", "status"}


class MetricsSchemaError(ValueError):
    """Raised when a metrics dict does not conform to the required schema."""
    pass


def validate_metrics(payload: dict) -> None:
    """Raise ``MetricsSchemaError`` if *payload* is missing required keys."""
    missing = REQUIRED_KEYS - set(payload.keys())
    if missing:
        raise MetricsSchemaError(f"metrics.json missing keys: {missing}")


def build_payload(
    run_id: str,
    task: str,
    global_metrics: dict[str, Any],
    *,
    model: str = "",
    status: str = "final",
    per_class: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standard metrics.json payload.

    Parameters
    ----------
    run_id:
        Run identifier (e.g. "0001", "RUN_0002").
    task:
        Task name: ``"task1a"``, ``"task1b"``, ``"task2"``, or composite.
    global_metrics:
        Dict of global metric values.
    model:
        Model type string (e.g. "ordinal", "dynunet", "medsam2").
    status:
        ``"final"`` for challenge-ground-truth metrics,
        ``"proxy"`` for self-consistency / validation proxy metrics.
    per_class:
        Optional per-class metrics (e.g. list of dicts for Task 2).
    extra:
        Any additional keys to include in the payload.

    Returns
    -------
    A dict conforming to the standard envelope.
    """
    payload: dict[str, Any] = {
        "run_id": run_id,
        "task": task,
        "model": model,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "global": global_metrics,
    }
    if per_class is not None:
        payload["per_class"] = per_class
    if extra is not None:
        payload.update(extra)
    return payload


def write_metrics(
    payload: dict[str, Any],
    results_dir: str,
    filename: str = "metrics.json",
) -> str:
    """Validate and write a metrics payload to disk.

    Returns the absolute path to the written file.
    """
    validate_metrics(payload)
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return os.path.abspath(path)


def read_metrics(results_dir: str, filename: str = "metrics.json") -> dict[str, Any]:
    """Read and return a metrics.json payload, or raise FileNotFoundError."""
    path = os.path.join(results_dir, filename)
    with open(path, "r") as fh:
        return json.load(fh)
