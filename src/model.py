"""模型：TransE 與 Unstructured baseline。

TransE 的核心假設（論文 Section 2）：若 (h, ℓ, t) 成立，則 h + ℓ ≈ t，
也就是 t 應該是 h + ℓ 的最近鄰。距離 d 用 L1 或 L2（論文兩種都試）。

Unstructured（論文 Section 4.1 的 baseline）＝ 把 ℓ 拿掉（ℓ = 0）：
d(h, t) = ||h - t||，所有關係共用同一個「h 和 t 要靠近」的目標，
它只能學到「哪些實體常一起出現」，無法區分不同關係。
"""
import math

import torch
import torch.nn as nn


class TransE(nn.Module):
    def __init__(self, n_ent: int, n_rel: int, dim: int = 50, p: int = 1, use_relation: bool = True):
        super().__init__()
        assert p in (1, 2), "距離函數只支援 L1 (p=1) 或 L2 (p=2)"
        self.p, self.dim, self.use_relation = p, dim, use_relation
        self.ent = nn.Embedding(n_ent, dim)
        self.rel = nn.Embedding(n_rel, dim)

        # Algorithm 1 第 1、3 行：ℓ, e ~ uniform(-6/√k, 6/√k)（Glorot & Bengio 的初始化）
        bound = 6.0 / math.sqrt(dim)
        nn.init.uniform_(self.ent.weight, -bound, bound)
        nn.init.uniform_(self.rel.weight, -bound, bound)
        # Algorithm 1 第 2 行：ℓ ← ℓ / ||ℓ||（關係向量只在初始化時正規化一次）
        with torch.no_grad():
            self.rel.weight.div_(self.rel.weight.norm(dim=1, keepdim=True))

        if not use_relation:
            # Unstructured 不使用關係向量，凍結它，避免被當成參數
            self.rel.weight.requires_grad_(False)

    @torch.no_grad()
    def normalize_entities(self):
        """Algorithm 1 第 5 行：e ← e / ||e||。
        論文的限制是實體向量 L2 norm = 1，防止模型靠把向量放大來「作弊」降低 loss。"""
        w = self.ent.weight
        w.div_(w.norm(dim=1, keepdim=True).clamp_min(1e-12))

    def _query(self, h, r):
        """回傳 h + ℓ（Unstructured 時 ℓ = 0，回傳 h）"""
        q = self.ent(h)
        return q + self.rel(r) if self.use_relation else q

    def distance(self, triples: torch.Tensor) -> torch.Tensor:
        """d(h + ℓ, t)，triples 形狀 [B, 3]，回傳 [B]"""
        h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
        diff = self._query(h, r) - self.ent(t)
        return torch.linalg.vector_norm(diff, ord=self.p, dim=1)

    # ---------- 評估用：對「所有」實體打分（距離越小越可能是答案） ----------
    @torch.no_grad()
    def dist_all_tails(self, h, r):
        """預測尾實體：對每個候選 t' 算 d(h + ℓ, t')，回傳 [B, n_ent]"""
        return torch.cdist(self._query(h, r), self.ent.weight, p=self.p)

    @torch.no_grad()
    def dist_all_heads(self, r, t):
        """預測頭實體：d(h' + ℓ, t) = ||h' - (t - ℓ)||，回傳 [B, n_ent]"""
        target = self.ent(t) - self.rel(r) if self.use_relation else self.ent(t)
        return torch.cdist(target, self.ent.weight, p=self.p)


def build_model(name: str, n_ent: int, n_rel: int, dim: int, p: int) -> TransE:
    if name == "transe":
        return TransE(n_ent, n_rel, dim, p, use_relation=True)
    if name == "unstructured":
        return TransE(n_ent, n_rel, dim, p, use_relation=False)
    raise ValueError(f"未知模型 {name}")
