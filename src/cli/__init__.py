"""CLI sub-package for LISA 2026 — run discovery and registry management."""

from src.cli.registry import discover_runs, discover_eval_runs, list_runs

__all__ = ["discover_runs", "discover_eval_runs", "list_runs"]
