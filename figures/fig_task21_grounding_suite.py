"""Publication figures for the Task 21 matched MEL/NEV grounding audit."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from viz_style import apply_style, save_figure


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
PAIRS_CSV = RESULTS_DIR / "task21_matched_grounding_audit_pairs.csv"
SUMMARY_CSV = RESULTS_DIR / "task21_matched_grounding_audit_summary.csv"

MEL = "#B2182B"
NEV = "#2166AC"
REAL = "#4C78A8"
BLANK = "#8E8E8E"
GOOD = "#1B9E77"
BAD = "#D95F02"
NEUTRAL = "#4D4D4D"


def apply_task21_style():
    apply_style()
    mpl.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.3,
    })


def load_data():
    pairs = pd.read_csv(PAIRS_CSV)
    summary = pd.read_csv(SUMMARY_CSV)
    return pairs, summary


def save_both(fig, stem):
    save_figure(fig, stem)
    out_dir = Path(__file__).resolve().parent / "output"
    root_dir = Path(__file__).resolve().parent
    for ext in ("pdf", "png"):
        src = out_dir / f"{stem}.{ext}"
        dst = root_dir / f"{stem}.{ext}"
        dst.write_bytes(src.read_bytes())


def draw_box(ax, xy, text, color, width=0.25, height=0.14, fontsize=10):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.4,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def add_arrow(ax, start, end, color=NEUTRAL):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.5,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def fig_design(summary):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.5,
        0.95,
        "Matched 2x2 Grounding Audit",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.90,
        "The same MEL/NEV image is paired with diagnosis-aligned and contradictory clinical text",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#333333",
    )

    y_rows = [0.68, 0.43]
    labels = ["Real lesion image", "Blank image control"]
    colors = [REAL, BLANK]
    for y, label, color in zip(y_rows, labels, colors):
        draw_box(ax, (0.04, y), label, color, width=0.20, height=0.12)
        draw_box(ax, (0.34, y + 0.07), "MEL-consistent\nhistory", MEL, width=0.20, height=0.10)
        draw_box(ax, (0.34, y - 0.07), "NEV-consistent\nhistory", NEV, width=0.20, height=0.10)
        draw_box(ax, (0.67, y), "Cross-Attn\nT->V", "#333333", width=0.15, height=0.12)
        draw_box(ax, (0.88, y), "MEL-NEV\nlogit margin", "#333333", width=0.10, height=0.12, fontsize=9)
        add_arrow(ax, (0.25, y + 0.06), (0.34, y + 0.12), color)
        add_arrow(ax, (0.25, y + 0.06), (0.34, y - 0.01), color)
        add_arrow(ax, (0.55, y + 0.12), (0.67, y + 0.07), MEL)
        add_arrow(ax, (0.55, y - 0.01), (0.67, y + 0.04), NEV)
        add_arrow(ax, (0.82, y + 0.06), (0.88, y + 0.06), "#333333")

    real_all = summary[(summary["image_condition"] == "real_image") & (summary["true_class"] == "ALL")].iloc[0]
    blank_all = summary[(summary["image_condition"] == "blank_image") & (summary["true_class"] == "ALL")].iloc[0]
    ax.text(
        0.5,
        0.23,
        "Primary estimand: aligned true-class margin - contradictory true-class margin",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.16,
        (
            f"Expected if grounded: positive shift.  Observed on real images: "
            f"{real_all['mean_aligned_minus_contradictory_margin']:+.4f}, "
            f"flip rate {100 * real_all['contradiction_flip_rate']:.2f}%"
        ),
        ha="center",
        va="center",
        fontsize=10,
        color=BAD,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.09,
        (
            f"Blank-image control: {blank_all['mean_aligned_minus_contradictory_margin']:+.4f}, "
            "showing a tiny text-only offset but no decision movement"
        ),
        ha="center",
        va="center",
        fontsize=9.2,
        color="#555555",
    )
    save_both(fig, "fig_task21_grounding_design")


def fig_heatmap(summary):
    ordered_rows = [
        ("blank_image", "MEL"),
        ("blank_image", "NEV"),
        ("real_image", "MEL"),
        ("real_image", "NEV"),
    ]
    values = []
    labels = []
    pvals = []
    flips = []
    for image_condition, true_class in ordered_rows:
        row = summary[
            (summary["image_condition"] == image_condition)
            & (summary["true_class"] == true_class)
        ].iloc[0]
        values.append(row["mean_aligned_minus_contradictory_margin"])
        labels.append(f"{image_condition.replace('_', ' ').title()}\n{true_class}")
        pvals.append(row["wilcoxon_p_vs_zero"])
        flips.append(row["contradiction_flip_rate"])

    arr = np.array(values).reshape(4, 1)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    vmax = max(abs(arr.min()), abs(arr.max()))
    im = ax.imshow(arr, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks([0])
    ax.set_xticklabels(["Mean aligned -\ncontradictory margin"])
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(labels)
    ax.set_title("Semantic Text Grounding Moves the Margin the Wrong Way", pad=8)

    for i, val in enumerate(values):
        text_color = "white" if abs(val) > vmax * 0.50 else "#222222"
        ax.text(
            0,
            i,
            f"{val:+.4f}\np={pvals[i]:.1e}\nflip={100*flips[i]:.1f}%",
            ha="center",
            va="center",
            fontsize=8.5,
            color=text_color,
            fontweight="bold",
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
    cbar.set_label("Positive = diagnosis-consistent text helps", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    ax.tick_params(axis="both", length=0)
    save_both(fig, "fig_task21_margin_heatmap")


def fig_diagonal_scatter(pairs):
    real = pairs[pairs["image_condition"] == "real_image"].copy()
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    for cls, color in [("MEL", MEL), ("NEV", NEV)]:
        sub = real[real["true_class"] == cls]
        ax.scatter(
            sub["contradictory_true_margin"],
            sub["aligned_true_margin"],
            s=38,
            alpha=0.76,
            label=cls,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

    lo = min(real["contradictory_true_margin"].min(), real["aligned_true_margin"].min())
    hi = max(real["contradictory_true_margin"].max(), real["aligned_true_margin"].max())
    pad = (hi - lo) * 0.08
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], color="#222222", linewidth=1.5, linestyle="--", label="No text effect")
    ax.fill_between([lo, hi], [lo, hi], [hi, hi], color=GOOD, alpha=0.08, label="Grounded region")
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], color=BAD, alpha=0.08, label="Wrong-way/no-help region")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True-class margin with contradictory text")
    ax.set_ylabel("True-class margin with aligned text")
    ax.set_title("Case-Level Margins Cluster on the No-Effect Diagonal")
    ax.legend(loc="best", frameon=False, fontsize=7.5)
    save_both(fig, "fig_task21_diagonal_scatter")


def fig_shift_distribution(pairs):
    plot_rows = [
        ("Blank MEL", pairs[(pairs["image_condition"] == "blank_image") & (pairs["true_class"] == "MEL")]),
        ("Blank NEV", pairs[(pairs["image_condition"] == "blank_image") & (pairs["true_class"] == "NEV")]),
        ("Real MEL", pairs[(pairs["image_condition"] == "real_image") & (pairs["true_class"] == "MEL")]),
        ("Real NEV", pairs[(pairs["image_condition"] == "real_image") & (pairs["true_class"] == "NEV")]),
    ]
    data = [row["aligned_minus_contradictory_margin"].to_numpy() for _, row in plot_rows]
    labels = [name for name, _ in plot_rows]
    colors = [BLANK, BLANK, MEL, NEV]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    parts = ax.violinplot(data, positions=np.arange(len(data)), showmeans=False, showmedians=False, widths=0.74)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.22)
    for key in ("cbars", "cmins", "cmaxes"):
        parts[key].set_color("#777777")
        parts[key].set_linewidth(1)

    for idx, (vals, color) in enumerate(zip(data, colors)):
        jitter = np.linspace(-0.18, 0.18, len(vals))
        ax.scatter(
            np.full(len(vals), idx) + jitter,
            vals,
            s=18,
            alpha=0.70,
            color=color,
            edgecolor="white",
            linewidth=0.25,
        )
        mean = float(np.mean(vals))
        ax.hlines(mean, idx - 0.30, idx + 0.30, color="#111111", linewidth=2.2)
        ax.text(idx, mean + 0.006, f"{mean:+.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(0, color="#222222", linewidth=1.3, linestyle="--")
    ax.text(
        0.02,
        0.96,
        "Grounded response should lie above the dashed zero line",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color=GOOD,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.5},
    )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Aligned - contradictory true-class margin")
    ax.set_title("Matched Text Rarely Helps and Real Images Shift Negative")
    save_both(fig, "fig_task21_shift_distribution")


def fig_prediction_stability(pairs):
    groups = [
        ("Blank\nMEL", "blank_image", "MEL"),
        ("Blank\nNEV", "blank_image", "NEV"),
        ("Real\nMEL", "real_image", "MEL"),
        ("Real\nNEV", "real_image", "NEV"),
    ]
    x = np.arange(len(groups))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    for offset, condition, alpha in [(-width / 2, "aligned_pred", 0.95), (width / 2, "contradictory_pred", 0.58)]:
        mel_rates = []
        nev_rates = []
        for _, image_condition, true_class in groups:
            sub = pairs[(pairs["image_condition"] == image_condition) & (pairs["true_class"] == true_class)]
            mel_rates.append((sub[condition] == "MEL").mean())
            nev_rates.append((sub[condition] == "NEV").mean())
        ax.bar(x + offset, mel_rates, width, color=MEL, alpha=alpha, label="MEL pred" if condition == "aligned_pred" else None)
        ax.bar(
            x + offset,
            nev_rates,
            width,
            bottom=mel_rates,
            color=NEV,
            alpha=alpha,
            label="NEV pred" if condition == "aligned_pred" else None,
        )
        for idx, (m, n) in enumerate(zip(mel_rates, nev_rates)):
            ax.text(
                idx + offset,
                1.04,
                "A" if condition == "aligned_pred" else "C",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#333333",
            )
            if m > 0.08:
                ax.text(
                    idx + offset,
                    m / 2,
                    f"{100*m:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
            if n > 0.08:
                ax.text(
                    idx + offset,
                    m + n / 2,
                    f"{100*n:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("Prediction fraction")
    ax.set_title("Contradictory Text Almost Never Changes the Binary Decision")
    ax.text(0.5, -0.18, "A = aligned text, C = contradictory text", transform=ax.transAxes, ha="center", fontsize=9)
    ax.legend(loc="upper center", ncol=2, frameon=False)
    save_both(fig, "fig_task21_prediction_stability")


def main():
    apply_task21_style()
    pairs, summary = load_data()
    fig_design(summary)
    fig_heatmap(summary)
    fig_diagonal_scatter(pairs)
    fig_shift_distribution(pairs)
    fig_prediction_stability(pairs)


if __name__ == "__main__":
    main()
