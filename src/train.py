"""訓練：依論文 Algorithm 1 實作 TransE 的學習流程，加上用驗證集做 early stopping。

Algorithm 1（對應下方程式碼的註解編號）：
  1-3  初始化 ℓ、e ~ uniform(-6/√k, 6/√k)，ℓ ← ℓ/||ℓ||          → model.py
  4    loop
  5      e ← e/||e||（每個實體正規化）
  6      S_batch ← sample(S, b)（抽 minibatch）
  7      T_batch ← ∅
  8-9    對每個 (h,ℓ,t) 抽一個 corrupted triplet (h',ℓ,t')       → corrupt()
  10     T_batch ← T_batch ∪ {((h,ℓ,t), (h',ℓ,t'))}
  12     依梯度 ∇ Σ [γ + d(h+ℓ,t) − d(h'+ℓ,t')]_+ 更新 embedding  → Eq.(1) 的 loss
  13   end loop
"""
import copy
import time
from dataclasses import asdict, dataclass

import torch

from .evaluate import evaluate, format_metrics
from .model import build_model


@dataclass
class TrainConfig:
    model: str = "transe"       # transe / unstructured
    dim: int = 50               # 向量維度 k
    lr: float = 0.01            # 學習率 λ
    margin: float = 1.0         # margin γ
    p: int = 1                  # 距離 L1 (1) / L2 (2)
    batch_size: int = 2048      # minibatch 大小 b
    epochs: int = 1000          # 最多跑幾輪（一輪 = 看過所有訓練三元組一次）
    eval_every: int = 25        # 每幾輪在驗證集評估一次
    patience: int = 5           # 連續幾次驗證沒進步就提早停止
    optimizer: str = "sgd"      # 論文用 SGD；adam 為額外選項
    seed: int = 42
    select_metric: str = "filt_mr"   # 選模型 / early stopping 的依據（論文用驗證集 mean rank）

    def name(self):
        return f"{self.model}_k{self.dim}_lr{self.lr}_m{self.margin}_L{self.p}"


def corrupt(pos: torch.Tensor, n_ent: int) -> torch.Tensor:
    """Algorithm 1 第 9 行 / 論文 Eq.(2)：
    S'_(h,ℓ,t) = {(h',ℓ,t) | h' ∈ E} ∪ {(h,ℓ,t') | t' ∈ E}
    每個正樣本隨機換掉 head「或」tail（機率各 1/2，不會同時換），換成均勻抽樣的實體。"""
    neg = pos.clone()
    replace_head = torch.rand(len(pos), device=pos.device) < 0.5
    rand_ent = torch.randint(0, n_ent, (len(pos),), device=pos.device)
    neg[replace_head, 0] = rand_ent[replace_head]
    neg[~replace_head, 2] = rand_ent[~replace_head]
    return neg


def train(cfg: TrainConfig, data, valid_filter, device, log=print):
    torch.manual_seed(cfg.seed)
    model = build_model(cfg.model, data.n_ent, data.n_rel, cfg.dim, cfg.p).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=cfg.lr)
    else:
        opt = torch.optim.Adam(params, lr=cfg.lr)

    train_trips = data.train.to(device)
    N = len(train_trips)
    history = {"epoch": [], "loss": [], "eval_epoch": [], "valid": []}
    lower_is_better = cfg.select_metric.endswith("_mr")
    best_score, best_state, best_epoch, best_metrics, bad = None, None, 0, None, 0
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):                       # 第 4 行 loop
        perm = torch.randperm(N, device=device)
        total = 0.0
        for s in range(0, N, cfg.batch_size):
            model.normalize_entities()                           # 第 5 行：e ← e/||e||
            pos = train_trips[perm[s:s + cfg.batch_size]]        # 第 6 行：S_batch
            neg = corrupt(pos, data.n_ent)                       # 第 8-9 行：corrupted triplet
            d_pos, d_neg = model.distance(pos), model.distance(neg)
            # 論文 Eq.(1)：L = Σ [γ + d(h+ℓ,t) − d(h'+ℓ,t')]_+   （[x]_+ = max(0,x)）
            loss = torch.relu(cfg.margin + d_pos - d_neg).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()                                           # 第 12 行：梯度下降更新
            total += loss.item()
        history["epoch"].append(epoch)
        history["loss"].append(total / N)                        # 每個樣本的平均 loss

        # ---- 用驗證集評估 + early stopping ----
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            model.normalize_entities()
            m = evaluate(model, data.valid, valid_filter, device)
            history["eval_epoch"].append(epoch)
            history["valid"].append(m)
            score = m[cfg.select_metric]
            improved = best_score is None or (score < best_score if lower_is_better else score > best_score)
            if improved:
                best_score, best_epoch, best_metrics, bad = score, epoch, m, 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                bad += 1
            log(f"[{cfg.name()}] ep {epoch:4d} loss {total / N:.4f} | valid {format_metrics(m)}"
                f"{' *' if improved else ''} | {time.time() - t0:.0f}s")
            if bad >= cfg.patience:
                log(f"[{cfg.name()}] early stop at epoch {epoch}（最佳 epoch {best_epoch}）")
                break

    model.load_state_dict(best_state)
    return {
        "config": asdict(cfg),
        "model": model,
        "best_epoch": best_epoch,
        "best_valid": best_metrics,
        "history": history,
        "train_seconds": time.time() - t0,
    }
