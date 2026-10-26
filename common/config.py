"""Repo-wide paths and config-file helpers."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_DATA_CONFIG = CONFIG_DIR / "data_config.json"
DEFAULT_MODEL_SETTINGS = CONFIG_DIR / "model_settings.json"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_data_config(path=None):
    """Loads configs/data_config.json (or the file given by `path`)."""
    return load_json(path or DEFAULT_DATA_CONFIG)


def dataset_dir(data_config, dataset):
    """Folder holding formatted_transactions.csv (+ the *_map.json files) for `dataset`."""
    return f"{data_config['paths']['aml_data']}/{dataset}"
