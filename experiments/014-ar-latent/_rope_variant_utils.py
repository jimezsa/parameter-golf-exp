from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


_THIS_DIR = Path(__file__).resolve().parent
_BASELINE_PATH = _THIS_DIR / "train_gpt.py"


def load_base(module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, _BASELINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load baseline module from {_BASELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def float_env(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def str_env(name: str, default: str) -> str:
    return os.environ.get(name, default)
