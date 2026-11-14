"""共用小工具：亂數種子、裝置選擇、JSON 存取、路徑常數。"""
import json
import random
from pathlib import Path

import numpy as np
import torch

# 專案根目錄（src/ 的上一層）
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def set_seed(seed: int) -> None:
    """固定所有亂數來源，讓實驗可重現。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> torch.device:
    """pref = auto / cuda / cpu。auto：有 GPU 就用 GPU，否則用 CPU。"""
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 --device cuda，但偵測不到 CUDA GPU")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
