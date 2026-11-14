"""完整實驗流程（README 的主要指令）：

  階段 A  超參數搜尋：對 grid 中每組參數訓練 screen_epochs 輪，看驗證集 filtered MR
  階段 B  正式訓練：最佳參數訓練到 final_epochs（驗證集 early stopping）
  階段 C  測試集評估：raw / filtered 的 MR、Hits@10，以及依關係類型（1-1…N-N）的 Hits@10
  TransE 與 Unstructured baseline 各跑一次完整流程。

用法：
  python -m src.tune                      # 完整版（RTX 4060 約 1 小時）
  python -m src.tune --quick              # 小規模試跑（幾分鐘，用來確認環境）
已完成的 grid 結果會存在 results/grid/，中斷後重跑會自動跳過。
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import torch

from .data import build_filter_index, load_fb15k237
from .evaluate import evaluate, format_metrics, hits10_by_category, relation_categories
from .plots import MODEL_COLOR, MODEL_LABEL, plt, save
from .train import TrainConfig, train
from .utils import RESULTS_DIR, get_device, load_json, save_json, set_seed


def log_to(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        f.write(msg + "\n")
        f.flush()
    return log


def grid_search(model_name, grid, data, valid_filter, device, args, log):
    keys = ["dim", "margin", "p", "lr"]
    results = []
    for values in itertools.product(*(grid[k] for k in keys)):
        cfg = TrainConfig(model=model_name, epochs=args.screen_epochs, eval_every=args.eval_every,
                          patience=args.patience, batch_size=args.batch_size,
                          optimizer=args.optimizer, seed=args.seed, **dict(zip(keys, values)))
        out_path = args.out / "grid" / f"{cfg.name()}_e{cfg.epochs}.json"
        if out_path.exists():
            r = load_json(out_path)
            log(f"[skip] {cfg.name()} 已有結果：valid filt MR {r['best_valid']['filt_mr']:.1f}")
        else:
            res = train(cfg, data, valid_filter, device, log)
            r = {k: res[k] for k in ("config", "best_epoch", "best_valid", "train_seconds")}
            save_json(r, out_path)
        results.append(r)
    results.sort(key=lambda r: r["best_valid"][args.select_metric])
    return results


def plot_curves(histories, out_dir):
    # 圖 1：訓練 loss 曲線
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for name, h in histories.items():
        ax.plot(h["epoch"], h["loss"], color=MODEL_COLOR[name], label=MODEL_LABEL[name])
        ax.annotate(MODEL_LABEL[name], (h["epoch"][-1], h["loss"][-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set(xlabel="Epoch", ylabel="Margin ranking loss (per triple)", title="Training loss")
    ax.legend()
    save(fig, out_dir / "loss_curve.png")

    # 圖 2、3：驗證集指標（分開畫兩張，不用雙 y 軸）
    for key, ylabel, fname in (("filt_mr", "Filtered mean rank (lower is better)", "valid_mr.png"),
                               ("filt_h10", "Filtered Hits@10 (%)", "valid_hits10.png")):
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for name, h in histories.items():
            ys = [m[key] * (100 if key.endswith("h10") else 1) for m in h["valid"]]
            ax.plot(h["eval_epoch"], ys, color=MODEL_COLOR[name], marker="o", markersize=4,
                    label=MODEL_LABEL[name])
        ax.set(xlabel="Epoch", ylabel=ylabel, title="Validation " + ylabel.split(" (")[0])
        ax.legend()
        save(fig, out_dir / fname)


def write_tables(final, grids, cat_counts, out_dir):
    rows = []
    for name, r in final.items():
        t = r["test"]
        rows.append((MODEL_LABEL[name], r["config"], t))
    md = ["# 實驗結果（FB15k-237 測試集）", "",
          "## Link prediction（head / tail 預測平均）", "",
          "| 模型 | k | γ | 距離 | lr | 最佳 epoch | Raw MR | Raw Hits@10 | Filt. MR | Filt. Hits@10 | Filt. MRR |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, c, t in rows:
        name = c["model"]
        md.append(f"| {label} | {c['dim']} | {c['margin']} | L{c['p']} | {c['lr']} | {final[name]['best_epoch']} | "
                  f"{t['raw_mr']:.0f} | {t['raw_h10']*100:.1f}% | {t['filt_mr']:.0f} | {t['filt_h10']*100:.1f}% | {t['filt_mrr']:.3f} |")

    md += ["", "## 依關係類型的 filtered Hits@10（對應論文 Table 4）", "",
           "關係類型由訓練集計算：hpt / tph（每個 tail 平均幾個 head / 每個 head 平均幾個 tail）< 1.5 視為「1」。", "",
           "| 模型 | 預測 | " + " | ".join(f"{c}（{n} 關係）" for c, n in cat_counts.items()) + " |",
           "|---|---|" + "---|" * len(cat_counts)]
    for name, r in final.items():
        for side in ("head", "tail"):
            cells = [f"{r['by_category'][c][f'{side}_h10']*100:.1f}%" if c in r["by_category"] else "-"
                     for c in cat_counts]
            md.append(f"| {MODEL_LABEL[name]} | {side} | " + " | ".join(cells) + " |")

    md += ["", "## 超參數搜尋（驗證集，依 filtered MR 排序）", ""]
    for name, g in grids.items():
        md += [f"### {MODEL_LABEL[name]}", "", "| k | γ | 距離 | lr | 最佳 epoch | Valid filt. MR | Valid filt. Hits@10 | 秒 |",
               "|---|---|---|---|---|---|---|---|"]
        for r in g:
            c, v = r["config"], r["best_valid"]
            md.append(f"| {c['dim']} | {c['margin']} | L{c['p']} | {c['lr']} | {r['best_epoch']} | "
                      f"{v['filt_mr']:.0f} | {v['filt_h10']*100:.1f}% | {r['train_seconds']:.0f} |")
        md.append("")
    (out_dir / "metrics.md").write_text("\n".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="configs/grid.json")
    ap.add_argument("--models", nargs="+", default=["transe", "unstructured"])
    ap.add_argument("--screen-epochs", type=int, default=100)
    ap.add_argument("--final-epochs", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    ap.add_argument("--select-metric", default="filt_mr")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="小規模試跑：grid 只取 2 組、epoch 很少")
    args = ap.parse_args()

    grid = load_json(args.grid)
    if args.quick:
        grid = {"dim": [50], "margin": [1.0], "p": [1, 2], "lr": [0.01]}
        args.screen_epochs, args.final_epochs, args.eval_every = 10, 20, 5
    out = RESULTS_DIR / "quick" if args.quick else RESULTS_DIR
    args.out = out

    set_seed(args.seed)
    device = get_device(args.device)
    log = log_to(out / "train_log.txt")
    log(f"device = {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    data = load_fb15k237()
    log(f"FB15k-237: {data.n_ent} entities, {data.n_rel} relations, "
        f"train/valid/test = {len(data.train)}/{len(data.valid)}/{len(data.test)}")
    valid_filter = build_filter_index(data.valid, data)
    test_filter = build_filter_index(data.test, data)
    cats = relation_categories(data.train, data.n_rel)
    cat_counts = {c: cats.count(c) for c in ("1-1", "1-N", "N-1", "N-N")}
    log(f"relation categories: {cat_counts}")

    final, grids, histories = {}, {}, {}
    emb_dir = out / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    for model_name in args.models:
        log(f"\n===== {model_name}: 階段 A 超參數搜尋 =====")
        grids[model_name] = grid_search(model_name, grid, data, valid_filter, device, args, log)
        best = grids[model_name][0]["config"]
        log(f"最佳參數：{best}")

        log(f"\n===== {model_name}: 階段 B 正式訓練 =====")
        cfg = TrainConfig(**{**best, "epochs": args.final_epochs, "eval_every": args.eval_every,
                             "patience": args.patience})
        res = train(cfg, data, valid_filter, device, log)
        model = res["model"]
        model.normalize_entities()

        log(f"\n===== {model_name}: 階段 C 測試集評估 =====")
        test_m, ranks = evaluate(model, data.test, test_filter, device, return_ranks=True)
        by_cat = hits10_by_category(data.test, ranks, cats)
        log(f"TEST {format_metrics(test_m)}")
        log(f"by category: {by_cat}")

        torch.save(model.state_dict(), out / f"{model_name}_best.pt")
        np.save(emb_dir / f"{model_name}_entity.npy", model.ent.weight.detach().cpu().numpy())
        np.save(emb_dir / f"{model_name}_relation.npy", model.rel.weight.detach().cpu().numpy())
        final[model_name] = {"config": res["config"], "best_epoch": res["best_epoch"],
                             "best_valid": res["best_valid"], "test": test_m, "by_category": by_cat,
                             "train_seconds": res["train_seconds"]}
        histories[model_name] = res["history"]
        save_json({"final": final, "histories": histories}, out / "final_results.json")

    (emb_dir / "entities.txt").write_text("\n".join(data.id2ent), encoding="utf-8")
    plot_curves(histories, out / "figures")
    write_tables(final, grids, cat_counts, out)
    log(f"\n完成。結果表：{out / 'metrics.md'}")


if __name__ == "__main__":
    main()
