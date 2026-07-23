"""Shared publication styling for MICCAI/ISIC workshop figures."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIGURES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURES_DIR.parent
DATA_DIR = FIGURES_DIR / "data"
OUTPUT_DIR = FIGURES_DIR / "output"

# ---------------------------------------------------------------------------
# Model / class metadata
# ---------------------------------------------------------------------------
MODELS = ["Late Fusion", "GMU Baseline", "Cross-Attention T→V", "Cross-Attention V→T"]
MODEL_SHORT = {
    "Image-Only": "Img-Only",
    "Text-Only": "Txt-Only",
    "Late Fusion": "Late Fusion",
    "GMU Baseline": "GMU",
    "Cross-Attention T→V": "Cross-Attn(T->V)",
    "Cross-Attention V→T": "Cross-Attn(V->T)",
}
MODEL_COLORS = {
    "Image-Only": "#888888",
    "Text-Only": "#AAAAAA",
    "Late Fusion": "#4477AA",
    "GMU Baseline": "#CC6677",
    "Cross-Attention T→V": "#228833",
    "Cross-Attention V→T": "#999933",
}
CLASS_NAMES = ["MEL", "BCC", "SCC", "ACK", "NEV", "SEK"]
CLASS_COLORS = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#A65628",
]

# ---------------------------------------------------------------------------
# Typography & export defaults
# ---------------------------------------------------------------------------
DPI = 300
FONT_SIZE = 20
TITLE_SIZE = 22
LABEL_SIZE = 20
TICK_SIZE = 18
LEGEND_SIZE = 16


def apply_style():
    """Apply matplotlib rcParams for publication-quality figures."""
    mpl.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "axes.linewidth": 1.2,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig, stem: str):
    """Save figure as PNG and PDF to figures/output/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"  Saved: {png_path.name}, {pdf_path.name}")
    plt.close(fig)
