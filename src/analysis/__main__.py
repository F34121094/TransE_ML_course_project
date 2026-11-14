"""非監督分析主程式：python -m src.analysis

讀入 src.tune 存下的實體向量（results/embeddings/*_entity.npy），對每個模型：
  1. K-means 掃 k，用 elbow / silhouette 選 k
  2. 以 silhouette 選出的 k，以及 k = 類型數，各做一次分群，算 purity / NMI / ARI
  3. PCA、t-SNE 降到 2D，依「真實類型」與「K-means 群」各畫一組 small multiples
  4. 輸出每群代表實體（質性分析）
最後寫出 results/analysis/analysis.md 比較 TransE 與 Unstructured。
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from ..data import load_entity_names, load_entity_types, load_fb15k237
from ..plots import MODEL_LABEL
from ..utils import RESULTS_DIR, save_json
from .cluster import cluster_report, external_scores, plot_k_scan, scan_k
from .visualize import cluster_panels, facet_plot, run_pca, run_tsne, type_panels


def analyze(model_name, X, names, labels, out, args):
    label = MODEL_LABEL[model_name]
    print(f"\n===== {label}: K-means 掃 k =====", flush=True)
    ks = list(range(args.k_min, args.k_max + 1, args.k_step))
    rows = scan_k(X, ks, seed=args.seed)
    for r in rows:
        print(f"  k={r['k']:3d} inertia={r['inertia']:.1f} silhouette={r['silhouette']:.4f}", flush=True)
    best_k = max(rows, key=lambda r: r["silhouette"])["k"]
    plot_k_scan(rows, best_k, f"{label}: choosing k for K-means", out / "kmeans_k_scan.png")

    n_types = len(set(labels.tolist())) if labels is not None else best_k
    result = {"k_scan": rows, "best_k_silhouette": best_k, "clusterings": {}}
    reports = {}
    for tag, k in (("silhouette", best_k), ("n_types", n_types)):
        km = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit(X)
        rep = cluster_report(X, km, names, labels)
        reports[tag] = (k, km, rep)
        scores = external_scores(km.labels_, labels) if labels is not None else {}
        result["clusterings"][tag] = {"k": k, **scores}
        print(f"  [{tag}] k={k} {scores}", flush=True)

    # 對照組：隨機分群（同樣的 k=類型數）的 purity / NMI，當作「完全沒學到東西」的下限
    if labels is not None:
        rng = np.random.default_rng(args.seed)
        result["random_baseline"] = external_scores(rng.integers(0, n_types, len(X)), labels)

    print(f"===== {label}: PCA / t-SNE =====", flush=True)
    pca_xy, evr = run_pca(X, args.seed)
    result["pca_explained_variance"] = evr
    tsne_xy = run_tsne(X, args.seed)
    np.save(out / "pca_2d.npy", pca_xy)
    np.save(out / "tsne_2d.npy", tsne_xy)

    k, km, rep = reports["n_types"]
    for method, xy in (("PCA", pca_xy), ("t-SNE", tsne_xy)):
        fname = method.lower().replace("-", "")
        if labels is not None:
            facet_plot(xy, labels, type_panels(labels),
                       f"{label} entity embeddings, {method}: colored by Freebase type",
                       out / f"{fname}_by_type.png")
        facet_plot(xy, km.labels_, cluster_panels(rep),
                   f"{label} entity embeddings, {method}: K-means clusters (k={k})",
                   out / f"{fname}_by_cluster.png")

    # 質性分析：每群代表實體
    md = [f"# {label}：K-means 各群的代表實體", ""]
    for tag, (k, _, rep) in reports.items():
        md += [f"## k = {k}（{'silhouette 最佳' if tag == 'silhouette' else '等於類型數'}）", ""]
        for r in rep:
            comp = "、".join(f"{t} {p*100:.0f}%" for t, p in r["types"])
            md += [f"**Cluster {r['cluster']}**（{r['size']} 個實體；類型組成：{comp}）  ",
                   "最靠近中心：" + "；".join(r["examples"]), ""]
    (out / "clusters.md").write_text("\n".join(md), encoding="utf-8")
    save_json(result, out / "analysis.json")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["transe", "unstructured"])
    ap.add_argument("--results", default=str(RESULTS_DIR), help="src.tune 的輸出資料夾")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=40)
    ap.add_argument("--k-step", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    res_dir = Path(args.results)
    data = load_fb15k237()
    names = load_entity_names(data.ent2id)
    labels = load_entity_types(data.ent2id)
    if labels is None:
        print("找不到 TKRL 類型檔，只做 silhouette 與質性分析")

    summary = {}
    for m in args.models:
        path = res_dir / "embeddings" / f"{m}_entity.npy"
        if not path.exists():
            print(f"找不到 {path}，請先執行 python -m src.tune")
            continue
        X = np.load(path)
        out = res_dir / "analysis" / m
        out.mkdir(parents=True, exist_ok=True)
        summary[m] = analyze(m, X, names, labels, out, args)

    md = ["# 非監督分析總表", "",
          "| 模型 | silhouette 最佳 k | 分群設定 | k | Purity | NMI | ARI |", "|---|---|---|---|---|---|---|"]
    for m, r in summary.items():
        for tag, c in r["clusterings"].items():
            md.append(f"| {MODEL_LABEL[m]} | {r['best_k_silhouette']} | {tag} | {c['k']} | "
                      f"{c.get('purity', float('nan')):.3f} | {c.get('nmi', float('nan')):.3f} | {c.get('ari', float('nan')):.3f} |")
        if "random_baseline" in r:
            b = r["random_baseline"]
            md.append(f"| 隨機分群（對照） | - | random | - | {b['purity']:.3f} | {b['nmi']:.3f} | {b['ari']:.3f} |")
    md += ["", "PCA 前兩主成分解釋的變異比例："]
    for m, r in summary.items():
        md.append(f"- {MODEL_LABEL[m]}：{', '.join(f'{v*100:.1f}%' for v in r['pca_explained_variance'])}")
    (res_dir / "analysis" / "analysis.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n完成：{res_dir / 'analysis' / 'analysis.md'}")


if __name__ == "__main__":
    main()
