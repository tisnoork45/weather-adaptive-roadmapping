"""Shared figure style for the validation graphs (300-DPI PNG, light paper background)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, GRAPHS = ROOT / "data", ROOT / "graphs"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
RISK = {"SAFE": "#0ca30c", "CAUTION": "#fab219", "WARNING": "#ec835a", "CRITICAL": "#d03b3b"}
RISK_ORDER = ["SAFE", "CAUTION", "WARNING", "CRITICAL"]

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 300, "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10.5,
    "font.family": "DejaVu Sans", "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def note(fig, text):
    """Provenance / caveat line under the plot."""
    fig.text(0.01, 0.01, text, fontsize=7.5, color=INK2, ha="left", va="bottom", wrap=True)


def save(fig, name):
    GRAPHS.mkdir(parents=True, exist_ok=True)
    fig.savefig(GRAPHS / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", GRAPHS / name)
