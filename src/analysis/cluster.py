"""非監督分析 (1)：對學到的實體向量做 K-means 分群。

- 選 k：elbow method（看 inertia 下降何時趨緩）與 silhouette score（越高代表群內緊密、群間分離）
- 外部評估：拿 TKRL 的實體類型當「標準答案」，計算
    purity = Σ_群 (群內最多數類型的數量) / N           → 每群有多「純」
    NMI    = 正規化互資訊，0 = 分群與類型無關，1 = 完全一致
    ARI    = 調整後 Rand index，隨機分群的期望值為 0
  注意：類型標籤「完全沒有」參與訓練或分群，只在事後用來評估，所以整個流程仍是非監督的。
- 質性分析：每群列出最靠近群中心的實體名稱，以及群內的類型組成
"""
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from ..plots import SERIES, TEXT_2, plt, save


def scan_k(X, ks, seed=42, sil_sample=5000):
    """對每個 k 跑 K-means，回傳 inertia 與 silhouette。
    silhouette 是 O(N²)，用 5000 個點抽樣估計（固定亂數種子）。"""
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=min(sil_sample, len(X)), random_state=seed)
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)})
    return rows


def plot_k_scan(rows, best_k, title, path):
    ks = [r["k"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(ks, [r["inertia"] for r in rows], color=SERIES[0], marker="o", markersize=4)
    axes[0].set(xlabel="k (number of clusters)", ylabel="Inertia (within-cluster SSE)", title="Elbow method")
    axes[1].plot(ks, [r["silhouette"] for r in rows], color=SERIES[0], marker="o", markersize=4)
    best = next(r for r in rows if r["k"] == best_k)
    axes[1].scatter([best_k], [best["silhouette"]], s=80, facecolors="none", edgecolors=TEXT_2, zorder=3)
    axes[1].annotate(f"best k = {best_k}", (best_k, best["silhouette"]), xytext=(8, 6),
                     textcoords="offset points", fontsize=9, color=TEXT_2)
    axes[1].set(xlabel="k (number of clusters)", ylabel="Silhouette score", title="Silhouette score")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    save(fig, path)


def purity(clusters, labels):
    total = 0
    for c in np.unique(clusters):
        total += Counter(labels[clusters == c]).most_common(1)[0][1]
    return total / len(labels)


def external_scores(clusters, labels):
    return {"purity": purity(clusters, labels),
            "nmi": float(normalized_mutual_info_score(labels, clusters)),
            "ari": float(adjusted_rand_score(labels, clusters))}


def cluster_report(X, km, names, labels, n_show=10):
    """每群：大小、類型組成前 3 名、最靠近中心的 n_show 個實體。"""
    out = []
    dists = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
    for c in range(km.n_clusters):
        idx = np.where(km.labels_ == c)[0]
        comp = Counter(labels[idx]).most_common(3) if labels is not None else []
        nearest = idx[np.argsort(dists[idx])[:n_show]]
        out.append({"cluster": c, "size": int(len(idx)),
                    "types": [(t, n / len(idx)) for t, n in comp],
                    "examples": [names[i] for i in nearest]})
    out.sort(key=lambda r: -r["size"])
    return out
