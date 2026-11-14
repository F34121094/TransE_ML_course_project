"""非監督分析 (2)：PCA 與 t-SNE 降到 2D 視覺化。

- PCA：線性投影到變異量最大的兩個方向，保留「全域」結構，但 2 維通常只解釋一小部分變異
- t-SNE：非線性，盡量保留「鄰居關係」，適合看群聚；但群與群的距離、群的大小沒有意義

類型有 10 多種，全畫在同一張散佈圖上顏色會分不清楚，所以用 small multiples：
每個小圖只把一種類型上色，其他點畫成灰色當背景。
"""
import math
from collections import Counter

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ..plots import MUTED_POINT, SERIES, plt, save


def run_pca(X, seed=42):
    pca = PCA(n_components=2, random_state=seed)
    return pca.fit_transform(X), pca.explained_variance_ratio_.tolist()


def run_tsne(X, seed=42, perplexity=30):
    # 先用 PCA 降到 50 維再做 t-SNE（常見做法，去雜訊也加速）
    X50 = PCA(n_components=min(50, X.shape[1]), random_state=seed).fit_transform(X)
    return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                random_state=seed).fit_transform(X50)


def facet_plot(coords, groups, panel_titles, suptitle, path, highlight=SERIES[0]):
    """groups：每個點所屬的群組（類型或群編號）；panel_titles：{群組: 小圖標題}，依序各畫一格。"""
    keys = list(panel_titles)
    ncol = 4
    nrow = math.ceil(len(keys) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, key in zip(axes.flat, keys):
        ax.set_axis_on()
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        mask = groups == key
        ax.scatter(coords[~mask, 0], coords[~mask, 1], s=1.5, c=MUTED_POINT, linewidths=0, rasterized=True)
        ax.scatter(coords[mask, 0], coords[mask, 1], s=3, c=highlight, linewidths=0, rasterized=True)
        ax.set_title(panel_titles[key], fontsize=9, fontweight="normal")
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    save(fig, path)


def type_panels(labels):
    cnt = Counter(labels.tolist())
    order = [t for t, _ in cnt.most_common() if t != "other"] + (["other"] if "other" in cnt else [])
    return {t: f"{t} (n={cnt[t]})" for t in order}


def cluster_panels(report, max_panels=12):
    titles = {}
    for r in report[:max_panels]:
        top = f"{r['types'][0][0]} {r['types'][0][1]*100:.0f}%" if r["types"] else ""
        titles[r["cluster"]] = f"cluster {r['cluster']} (n={r['size']})\nmostly {top}"
    return titles
