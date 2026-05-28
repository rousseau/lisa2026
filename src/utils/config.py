"""Configuration loading and environment-variable overrides."""

import os

import yaml


def load_config(config_path: str) -> dict:
    """Load a YAML config file and return it as a dict.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Parsed config dict.
    """
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def apply_env_overrides(config: dict) -> dict:
    """Override data-path config keys from environment variables.

    Supported variables:
        LISA_DATA_ROOT  → config["data"]["data_root"]  and  config["data"]["bids_root"]
        LISA_CSV_PATH   → config["data"]["csv_path"]

    If a variable is not set, the corresponding config value is left unchanged.
    The config dict is modified **in-place** and also returned for convenience.

    Args:
        config: Loaded config dict (must contain a ``"data"`` key).

    Returns:
        The same dict after applying overrides.
    """
    data = config.setdefault("data", {})

    root_override = os.getenv("LISA_DATA_ROOT")
    if root_override:
        data["data_root"] = root_override
        # bids_root may be the same as data_root for some runs
        if "bids_root" in data:
            data["bids_root"] = root_override

    csv_override = os.getenv("LISA_CSV_PATH")
    if csv_override:
        data["csv_path"] = csv_override

    return config
