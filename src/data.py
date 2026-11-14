"""資料載入：FB15k-237 三元組、id 對照、filtered 評估用的「已知正確答案」索引、
以及非監督分析用的實體名稱與類型標籤。

檔案格式（每行一個三元組，tab 分隔）：  head \t relation \t tail
"""
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .utils import DATA_DIR

FB_DIR = DATA_DIR / "FB15k-237"
TKRL_TYPE_FILE = DATA_DIR / "tkrl_raw" / "v2" / "entity2type.txt"


def read_triples(path):
    with open(path, encoding="utf-8") as f:
        return [tuple(line.rstrip("\n").split("\t")) for line in f if line.strip()]


@dataclass
class KGData:
    ent2id: dict
    rel2id: dict
    train: torch.Tensor   # [N, 3] LongTensor，欄位順序 (h, r, t)
    valid: torch.Tensor
    test: torch.Tensor
    # filtered 設定用：所有 split 中出現過的正確三元組
    tails_of: dict = field(default_factory=dict)   # (h, r) -> {t}
    heads_of: dict = field(default_factory=dict)   # (r, t) -> {h}

    @property
    def n_ent(self):
        return len(self.ent2id)

    @property
    def n_rel(self):
        return len(self.rel2id)

    @property
    def id2ent(self):
        return [e for e, _ in sorted(self.ent2id.items(), key=lambda x: x[1])]


def load_fb15k237(data_dir: Path = FB_DIR) -> KGData:
    splits = {s: read_triples(Path(data_dir) / f"{s}.txt") for s in ("train", "valid", "test")}

    # 用三個 split 的聯集建 id（排序後編號，結果固定可重現）
    ents = sorted({x for trips in splits.values() for h, _, t in trips for x in (h, t)})
    rels = sorted({r for trips in splits.values() for _, r, _ in trips})
    ent2id = {e: i for i, e in enumerate(ents)}
    rel2id = {r: i for i, r in enumerate(rels)}

    def to_tensor(trips):
        return torch.tensor([[ent2id[h], rel2id[r], ent2id[t]] for h, r, t in trips], dtype=torch.long)

    tensors = {s: to_tensor(v) for s, v in splits.items()}

    # 論文 4.2 節 "filt." 設定：排名時，把「本身也是正確三元組」的候選實體移除
    tails_of, heads_of = defaultdict(set), defaultdict(set)
    for tensor in tensors.values():
        for h, r, t in tensor.tolist():
            tails_of[(h, r)].add(t)
            heads_of[(r, t)].add(h)

    return KGData(ent2id, rel2id, tensors["train"], tensors["valid"], tensors["test"],
                  dict(tails_of), dict(heads_of))


def build_filter_index(triples: torch.Tensor, data: KGData):
    """為評估集預先算好「要被過濾掉的其他正確答案」，存成 CSR 格式（offsets + cols），
    評估時就能整批在 GPU 上把這些位置設成 +inf，不必逐筆跑 Python 迴圈。

    回傳 {"tail": (offsets, cols), "head": (offsets, cols)}；
    第 i 筆三元組要過濾的實體 = cols[offsets[i]:offsets[i+1]]（不含正確答案本身）。
    """
    out = {}
    for side in ("tail", "head"):
        offsets, cols = [0], []
        for h, r, t in triples.tolist():
            if side == "tail":
                others = data.tails_of[(h, r)] - {t}
            else:
                others = data.heads_of[(r, t)] - {h}
            cols.extend(others)
            offsets.append(len(cols))
        out[side] = (torch.tensor(offsets, dtype=torch.long), torch.tensor(cols, dtype=torch.long))
    return out


# ---------------- 非監督分析用：名稱與類型 ----------------

def load_entity_names(ent2id, path=FB_DIR / "entity2text.txt"):
    """Freebase MID -> 人看得懂的名稱（例：/m/0c94fn -> Gary Rydstrom）。缺的用 MID 代替。"""
    names = {}
    if Path(path).exists():
        for line in open(path, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                names[parts[0]] = parts[1]
    return [names.get(e, e) for e, _ in sorted(ent2id.items(), key=lambda x: x[1])]


# 粗類型的判斷規則：依序檢查，第一個符合的就是該實體的類型。
# （Freebase 的 type 清單沒有固定的「主要類型」順序，例如演員常把 award/award_winner 排在前面，
#   所以不能直接取第一個 type，要用優先順序。順序的理由：學校、球隊同時也是 organization / location，
#   要先判斷比較具體的類型。）
TYPE_RULES = [
    ("person",       {"people/person"}),
    ("film",         {"film/film"}),
    ("tv_program",   {"tv/tv_program"}),
    ("sports_team",  {"sports/sports_team"}),
    ("school",       {"education/educational_institution"}),
    ("organization", {"organization/organization"}),
    ("location",     {"location/location"}),
    ("award",        {"award/award_category"}),
    ("genre",        {"music/genre", "film/film_genre", "media_common/netflix_genre", "tv/tv_genre"}),
    ("profession",   {"people/profession"}),
]


def load_entity_types(ent2id, path=TKRL_TYPE_FILE):
    """從 TKRL (Xie et al., 2016) 的 entity2type.txt 為每個實體指定一個粗類型（見 TYPE_RULES），
    都不符合的標為 "other"。回傳與 entity id 對齊的 numpy 陣列；找不到檔案時回傳 None。"""
    if not Path(path).exists():
        return None
    types = {}
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        types[parts[0]] = set(parts[1:])
    labels = []
    for e, _ in sorted(ent2id.items(), key=lambda x: x[1]):
        ts = types.get(e, set())
        labels.append(next((name for name, rule in TYPE_RULES if ts & rule), "other"))
    return np.array(labels, dtype=object)
