"""共用的圖表樣式（matplotlib）。圖上文字用英文，避免 Windows 中文字型缺字。"""
import matplotlib

matplotlib.use("Agg")   # 不開視窗，直接存檔
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
MUTED_POINT = "#d6d5d0"
# 類別色（固定順序；同一個實體／模型永遠同一色）
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLOR = {"transe": SERIES[0], "unstructured": SERIES[1]}
MODEL_LABEL = {"transe": "TransE", "unstructured": "Unstructured"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "text.color": TEXT,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "font.size": 10, "lines.linewidth": 2,
    "legend.frameon": False, "figure.dpi": 110,
})


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
