"""Link prediction 評估（論文 Section 4.2）。

對每個測試三元組 (h, ℓ, t)：
  1. 把 head 換成「所有」實體，算每個候選的距離 d(h' + ℓ, t)，看正確的 h 排第幾名
  2. tail 同理
指標：
  - Mean Rank (MR)：正確答案的平均名次（越低越好）
  - Hits@10：正確答案排在前 10 名的比例（越高越好）
  - MRR：名次倒數的平均（論文沒有，但現在的論文都會報，順便附上）
兩種設定：
  - raw：直接排名
  - filtered：排名前先把「本身也是正確三元組」的候選拿掉（它們排在前面不算錯）
"""
import torch


@torch.no_grad()
def evaluate(model, triples: torch.Tensor, filter_index: dict, device, batch_size: int = 512,
             return_ranks: bool = False):
    """return_ranks=True 時另外回傳每筆三元組的 filtered 名次 {"head": [N], "tail": [N]}"""
    model.eval()
    triples = triples.to(device)
    sums = {f"{s}_{m}": 0.0 for s in ("raw", "filt") for m in ("mr", "mrr", "h10")}
    n_total = 0
    ranks_out = {}

    for side in ("tail", "head"):
        offsets, cols = (x.to(device) for x in filter_index[side])
        side_ranks = []
        for s in range(0, len(triples), batch_size):
            batch = triples[s:s + batch_size]
            B = len(batch)
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
            if side == "tail":
                dist, target = model.dist_all_tails(h, r), t
            else:
                dist, target = model.dist_all_heads(r, t), h

            target_d = dist.gather(1, target[:, None])               # 正確答案的距離 [B,1]
            # 名次 = 1 + 距離比正確答案小的候選數
            raw_rank = (dist < target_d).sum(1) + 1

            # filtered：把其他正確答案的位置設為 +inf，讓它們不會排在前面
            lo, hi = offsets[s].item(), offsets[s + B].item()
            if hi > lo:
                counts = offsets[s + 1:s + B + 1] - offsets[s:s + B]
                rows = torch.repeat_interleave(torch.arange(B, device=device), counts)
                dist[rows, cols[lo:hi]] = float("inf")
            filt_rank = (dist < target_d).sum(1) + 1
            side_ranks.append(filt_rank.cpu())

            for name, rank in (("raw", raw_rank), ("filt", filt_rank)):
                rank = rank.double()
                sums[f"{name}_mr"] += rank.sum().item()
                sums[f"{name}_mrr"] += (1.0 / rank).sum().item()
                sums[f"{name}_h10"] += (rank <= 10).sum().item()
            n_total += B
        ranks_out[side] = torch.cat(side_ranks)

    model.train()
    metrics = {k: v / n_total for k, v in sums.items()}   # head 與 tail 預測的平均
    return (metrics, ranks_out) if return_ranks else metrics


def relation_categories(train: torch.Tensor, n_rel: int, threshold: float = 1.5):
    """論文 Section 4.3 / Table 4：依關係的「對應型態」分成 1-1、1-N、N-1、N-N。
    tph = 每個 head 平均有幾個 tail；hpt = 每個 tail 平均有幾個 head。
    值 < 1.5 視為「1」，否則為「N」（與論文相同門檻）。"""
    from collections import defaultdict
    tails, heads = defaultdict(set), defaultdict(set)
    for h, r, t in train.tolist():
        tails[(r, h)].add(t)
        heads[(r, t)].add(h)
    tph_sum, tph_cnt, hpt_sum, hpt_cnt = [0] * n_rel, [0] * n_rel, [0] * n_rel, [0] * n_rel
    for (r, _), s in tails.items():
        tph_sum[r] += len(s); tph_cnt[r] += 1
    for (r, _), s in heads.items():
        hpt_sum[r] += len(s); hpt_cnt[r] += 1
    cats = []
    for r in range(n_rel):
        tph = tph_sum[r] / max(tph_cnt[r], 1)
        hpt = hpt_sum[r] / max(hpt_cnt[r], 1)
        cats.append(("1" if hpt < threshold else "N") + "-" + ("1" if tph < threshold else "N"))
    return cats


def hits10_by_category(triples: torch.Tensor, ranks: dict, cats: list):
    """回傳 {類別: {"n": 三元組數, "head_h10": 預測 head 的 filtered Hits@10, "tail_h10": ...}}"""
    rel_cat = [cats[r] for r in triples[:, 1].tolist()]
    out = {}
    for c in ("1-1", "1-N", "N-1", "N-N"):
        idx = torch.tensor([i for i, x in enumerate(rel_cat) if x == c], dtype=torch.long)
        if len(idx) == 0:
            continue
        out[c] = {
            "n": len(idx),
            "head_h10": (ranks["head"][idx] <= 10).double().mean().item(),
            "tail_h10": (ranks["tail"][idx] <= 10).double().mean().item(),
        }
    return out


def format_metrics(m: dict) -> str:
    return (f"raw MR {m['raw_mr']:.1f} H@10 {m['raw_h10']*100:.2f}% | "
            f"filt MR {m['filt_mr']:.1f} H@10 {m['filt_h10']*100:.2f}% MRR {m['filt_mrr']:.4f}")
