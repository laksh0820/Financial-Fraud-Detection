"""Small helpers shared by prediction_model and explanation_model."""
import json
import logging
import os
import random
import sys

import numpy as np


def setup_logging(log_dir="logs"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],  # log to stdout
    )


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
    except ImportError:  # the LLM / dataset code paths work without torch being importable
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set as {seed}")


def write_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
