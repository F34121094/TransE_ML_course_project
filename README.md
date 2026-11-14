# TransE 從零實作 + 非監督分析（FB15k-237）

以 PyTorch 從零實作 Bordes et al. (2013) *Translating Embeddings for Modeling Multi-relational Data*（TransE），
在 FB15k-237 上做 link prediction，並對學到的實體向量做 K-means、PCA、t-SNE 等非監督分析。
沒有使用任何現成的 KG embedding 套件。

## 檔案結構

```
TransE/
├─ README.md
├─ requirements.txt
├─ configs/grid.json          超參數搜尋範圍
├─ scripts/
│  ├─ download_data.ps1       下載資料
│  └─ run_all.ps1             一鍵跑完整流程
├─ src/
│  ├─ utils.py                亂數種子、裝置（CUDA / CPU）、JSON
│  ├─ data.py                 讀三元組、建 id、filtered 評估索引、實體名稱與類型
│  ├─ model.py                TransE 與 Unstructured（ℓ = 0）baseline
│  ├─ train.py                論文 Algorithm 1 + 驗證集 early stopping
│  ├─ evaluate.py             MR / Hits@10 / MRR（raw、filtered），依關係類型分析
│  ├─ tune.py                 主流程：grid search → 正式訓練 → 測試
│  ├─ plots.py                圖表樣式
│  └─ analysis/               非監督分析（python -m src.analysis）
│     ├─ cluster.py           K-means、elbow / silhouette、purity / NMI / ARI、代表實體
│     └─ visualize.py         PCA、t-SNE、small-multiples 散佈圖
├─ data/                      下載的資料（不進版控）
├─ results/                   所有輸出
└─ docs/report_outline.md     報告大綱
```

## 1. 環境安裝（Windows）

需要 Python 3.10 以上（測試環境：Python 3.13.5、RTX 4060 Laptop、驅動 551.76）。

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
# 有 NVIDIA GPU（驅動支援 CUDA 12.4 以上）：
.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# 沒有 GPU 的話改成：
# .venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install numpy pandas scikit-learn matplotlib tqdm
```

確認 GPU 可用：`.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"`

## 2. 下載資料

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_data.ps1
```

| 檔案 | 來源 | 用途 |
|---|---|---|
| `data/FB15k-237/{train,valid,test}.txt` | Toutanova & Chen (2015) 的標準切分（GitHub 鏡像 villmow/datasets_knowledge_embedding） | 訓練 / 評估 |
| `data/FB15k-237/entity2text.txt` | KG-BERT repo | MID → 實體名稱（質性分析） |
| `data/tkrl_raw/v2/entity2type.txt` | TKRL (Xie et al., 2016) | 實體的 Freebase types（只用來事後評估分群） |

## 3. 執行

```powershell
# 先小規模試跑（約 2 分鐘），確認環境沒問題
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Quick

# 完整流程（RTX 4060 Laptop 實測：訓練 + 評估約 20 分鐘，非監督分析約 5 分鐘）
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
```

也可以分開跑：

```powershell
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe -m src.tune          # 訓練 + 評估
.venv\Scripts\python.exe -m src.analysis      # 非監督分析（需先跑完 tune）
```

`src.tune` 的常用參數：

| 參數 | 預設 | 說明 |
|---|---|---|
| `--models` | `transe unstructured` | 要跑的模型 |
| `--screen-epochs` | 100 | grid search 每組訓練幾輪 |
| `--final-epochs` | 1000 | 正式訓練最多幾輪（有 early stopping） |
| `--eval-every` / `--patience` | 25 / 5 | 每 25 輪驗證一次；連續 5 次沒進步就停止 |
| `--batch-size` | 2048 | minibatch 大小 |
| `--optimizer` | `sgd` | 論文用 SGD；`adam` 收斂較快 |
| `--device` | `auto` | `cuda` / `cpu` |

grid search 每組的結果會存在 `results/grid/`。中斷後重跑會跳過已完成的組合。

### 時間太長時怎麼縮小規模

- 縮小 grid：編輯 `configs/grid.json`，例如只留 `"p": [1]`、`"dim": [50]`
- 少跑幾輪：`--screen-epochs 50 --final-epochs 300`
- 只跑 TransE：`--models transe`
- 只用 CPU 時：L1 距離比 L2 慢，建議先用 `--quick` 測速度，再估算完整時間

## 4. 預期輸出

```
results/
├─ metrics.md               ★ 測試集結果表、依關係類型的 Hits@10、grid search 結果
├─ final_results.json       同上（JSON），含每輪 loss 與驗證指標
├─ train_log.txt            訓練紀錄
├─ grid/*.json              每組超參數的驗證結果
├─ transe_best.pt / unstructured_best.pt
├─ embeddings/              實體、關係向量（.npy）與 entities.txt
├─ figures/
│  ├─ loss_curve.png        訓練 loss
│  ├─ valid_mr.png          驗證集 filtered MR
│  └─ valid_hits10.png      驗證集 filtered Hits@10
└─ analysis/
   ├─ analysis.md           ★ purity / NMI / ARI 總表
   └─ transe/, unstructured/
      ├─ kmeans_k_scan.png  elbow 與 silhouette
      ├─ pca_by_type.png, tsne_by_type.png        依真實類型上色
      ├─ pca_by_cluster.png, tsne_by_cluster.png  依 K-means 群上色
      └─ clusters.md        ★ 每群的類型組成與代表實體（質性分析）
```

### 實測結果摘要（seed 42，FB15k-237 測試集）

| 模型 | 最佳參數 | Raw MR | Raw Hits@10 | Filt. MR | Filt. Hits@10 | Filt. MRR |
|---|---|---|---|---|---|---|
| TransE | k=50, γ=2, L1, lr=0.01（第 325 epoch） | 429 | 28.5% | 265 | 37.8% | 0.219 |
| Unstructured | k=50, γ=2, L1, lr=0.01（第 175 epoch） | 1411 | 2.9% | 1255 | 4.0% | 0.020 |

K-means（k = 11 = 類型數）與 Freebase 類型的一致程度：

| 實體向量 | Purity | NMI | ARI |
|---|---|---|---|
| TransE | 0.801 | 0.644 | 0.412 |
| Unstructured | 0.595 | 0.374 | 0.230 |
| 隨機分群（對照） | 0.310 | 0.002 | 0.000 |

完整表格見 `results/metrics.md` 與 `results/analysis/analysis.md`。換 GPU / 驅動版本時數字可能有些微差異。

## 5. 實作細節與論文對應

| 論文 | 程式位置 |
|---|---|
| Eq.(1) margin ranking loss `Σ [γ + d(h+ℓ,t) − d(h'+ℓ,t')]_+` | `src/train.py` 訓練迴圈 |
| Eq.(2) corrupted triplet（隨機換 head 或 tail，不同時換） | `src/train.py: corrupt()` |
| Algorithm 1 第 1–3 行：uniform(−6/√k, 6/√k) 初始化，ℓ 正規化 | `src/model.py: TransE.__init__` |
| Algorithm 1 第 5 行：每次迭代 e ← e/‖e‖ | `src/model.py: normalize_entities()`，在每個 minibatch 前呼叫 |
| Section 4.2 raw / filtered、Mean Rank、Hits@10 | `src/evaluate.py: evaluate()` |
| Section 4.3 / Table 4：1-1、1-N、N-1、N-N 分析 | `src/evaluate.py: relation_categories()` |
| Unstructured baseline（ℓ = 0） | `src/model.py`（`use_relation=False`） |

我自己做的選擇（論文沒有寫清楚的部分）：

- **loss 用 sum：** 一個 minibatch 的 loss 是逐筆加總，照 Eq.(1)，不取平均。
- **評估名次：** 名次 = 1 + 距離「嚴格小於」正確答案的候選數。同分很少見，這裡忽略。
- **選模型的依據：** 驗證集 filtered MR。論文寫的是 mean rank；改依據用 `--select-metric filt_h10`。
- **實體類型標籤：** Freebase type 清單沒有固定的「主要類型」，所以依 `src/data.py: TYPE_RULES` 的優先順序指定。例如有 `people/person` 就算 person；學校先於 organization、location 判斷。共 10 類加上 other。
