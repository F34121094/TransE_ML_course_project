# 非監督分析總表

| 模型 | silhouette 最佳 k | 分群設定 | k | Purity | NMI | ARI |
|---|---|---|---|---|---|---|
| TransE | 40 | silhouette | 40 | 0.891 | 0.600 | 0.204 |
| TransE | 40 | n_types | 11 | 0.801 | 0.644 | 0.412 |
| 隨機分群（對照） | - | random | - | 0.310 | 0.002 | -0.000 |
| Unstructured | 2 | silhouette | 2 | 0.391 | 0.199 | 0.155 |
| Unstructured | 2 | n_types | 11 | 0.595 | 0.374 | 0.230 |
| 隨機分群（對照） | - | random | - | 0.310 | 0.002 | -0.000 |

PCA 前兩主成分解釋的變異比例：
- TransE：8.7%, 5.8%
- Unstructured：11.2%, 6.3%