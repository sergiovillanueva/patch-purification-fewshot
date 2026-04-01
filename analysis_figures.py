"""
Analysis figures and tables for Paper 3: Dirty Few-Shot Self-Purification.

Generates all figures and LaTeX tables for the paper:
1. AUROC vs contamination rate curves (Fig 2)
2. Per-dataset performance chart (Fig 3)
3. Risk-Reward scatter plot (Fig 4)
4. Recovery vs contamination extended curve (Fig 5)
5. Ablation: percentile threshold (Fig 6)
6. TL comparison (Fig 7)
7. DINOv3 vs CLIP backbone comparison (Fig 8)
8. LaTeX tables (main results, ablation, timing)

Usage:
    python analysis_figures.py                  # Generate all figures
    python analysis_figures.py --debug          # Generate only from available data
    python analysis_figures.py --figures-only   # Skip LaTeX tables
    python analysis_figures.py --tables-only    # Skip figures
"""

import argparse
import csv
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Color scheme for methods (colorblind-friendly)
METHOD_COLORS = {
    "none": "#888888",
    "random_patch": "#AAAAAA",
    "image_level_loo": "#E69F00",
    "loo_patch": "#0072B2",
    "mahalanobis_patch": "#009E73",
    "cosine_loo": "#CC79A7",
    "ensemble_loo_mahal": "#D55E00",
    "oracle_patch": "#000000",
    "adaptive_loo_patch": "#56B4E9",
    "lof_patch": "#7570B3",
}

METHOD_LABELS = {
    "none": "No purification (dirty)",
    "random_patch": "Random removal (control)",
    "image_level_loo": "Image-level LOO",
    "loo_patch": "LOO (Euclidean)",
    "mahalanobis_patch": "Mahalanobis",
    "cosine_loo": "LOO (Cosine)",
    "ensemble_loo_mahal": "Ensemble (LOO+Mahal)",
    "oracle_patch": "Oracle (supervised)",
    "adaptive_loo_patch": "Adaptive LOO",
    "lof_patch": "LOF (patch)",
}

METHOD_MARKERS = {
    "none": "s",
    "random_patch": "v",
    "image_level_loo": "^",
    "loo_patch": "o",
    "mahalanobis_patch": "D",
    "cosine_loo": "P",
    "ensemble_loo_mahal": "*",
    "oracle_patch": "X",
    "lof_patch": "h",
}

BENCHMARK_NAMES = {
    "mvtec_AD": "MVTec AD",
    "VisA": "VisA",
    "btad": "BTAD",
    "mvtec_loco_AD": "MVTec LOCO",
}

LOCO_DATASETS = [
    "mvtec_loco_AD/breakfast_box",
    "mvtec_loco_AD/juice_bottle",
    "mvtec_loco_AD/pushpins",
    "mvtec_loco_AD/screw_bag",
    "mvtec_loco_AD/splicing_connectors",
]

# Paper figure settings
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})


# ============================================================================
# DATA LOADING
# ============================================================================

def load_csv(csv_path: str, delimiter: str = ";") -> list[dict]:
    """Load CSV results file."""
    if not Path(csv_path).exists():
        print(f"  [SKIP] {csv_path} not found")
        return []
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            rows.append(row)
    print(f"  Loaded {len(rows)} rows from {csv_path}")
    return rows


def get_benchmark(dataset: str) -> str:
    """Get benchmark name from dataset path."""
    return dataset.split("/")[0]


def aggregate(
    rows: list[dict],
    group_keys: list[str],
    metric: str = "img_auroc",
    agg_func=np.mean,
) -> dict[tuple, float]:
    """
    Aggregate metric by group keys.
    Returns: {(key1, key2, ...): aggregated_value}
    """
    values = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        try:
            values[key].append(float(row[metric]))
        except (ValueError, KeyError):
            continue
    return {k: agg_func(v) for k, v in values.items()}


def aggregate_with_std(
    rows: list[dict],
    group_keys: list[str],
    metric: str = "img_auroc",
) -> dict[tuple, tuple[float, float]]:
    """Returns: {key: (mean, std)}"""
    values = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        try:
            values[key].append(float(row[metric]))
        except (ValueError, KeyError):
            continue
    return {k: (np.mean(v), np.std(v)) for k, v in values.items()}


# ============================================================================
# FIGURE 2: AUROC vs Contamination Rate Curves
# ============================================================================

def fig_auroc_vs_contamination(rows, output_dir, tl="10"):
    """AUROC vs contamination rate, per method."""
    methods = ["none", "random_patch", "loo_patch", "mahalanobis_patch",
               "cosine_loo", "ensemble_loo_mahal", "oracle_patch"]

    # Aggregate by (cont_rate, method) -> mean AUROC over datasets and seeds
    data = aggregate_with_std(
        [r for r in rows if r["train_limit"] == tl and r["purification_method"] in methods],
        ["contamination_rate", "purification_method"],
    )

    cont_rates = sorted(set(float(r["contamination_rate"]) for r in rows
                            if r["train_limit"] == tl))

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    for method in methods:
        means = []
        stds = []
        xs = []
        for cr in cont_rates:
            key = (str(cr), method)
            if key in data:
                m, s = data[key]
                means.append(m)
                stds.append(s)
                xs.append(cr)

        if not means:
            continue

        means = np.array(means)
        stds = np.array(stds)
        color = METHOD_COLORS.get(method, "#333333")
        label = METHOD_LABELS.get(method, method)
        marker = METHOD_MARKERS.get(method, "o")
        lw = 2.5 if method in ("ensemble_loo_mahal", "oracle_patch", "none") else 1.5
        ls = "--" if method in ("oracle_patch", "random_patch") else "-"

        ax.plot(xs, means, color=color, marker=marker, label=label,
                linewidth=lw, linestyle=ls, markersize=5)
        ax.fill_between(xs, means - stds, means + stds, alpha=0.1, color=color)

    ax.set_xlabel("Contamination Rate")
    ax.set_ylabel("Image AUROC")
    ax.set_title(f"AUROC vs Contamination Rate (TL={tl})")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8)
    ax.set_xlim(-0.01, max(cont_rates) + 0.01)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    path = output_dir / f"fig2_auroc_vs_cont_tl{tl}.pdf"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 3: Per-dataset Performance Chart
# ============================================================================

def fig_per_dataset_improvement(rows, output_dir, tl="10"):
    """Bar chart: AUROC improvement per dataset at cont=30%."""
    methods_to_show = ["ensemble_loo_mahal", "loo_patch", "mahalanobis_patch"]

    # Aggregate by (dataset, method) at cont=0.3
    data = aggregate(
        [r for r in rows if r["contamination_rate"] == "0.3" and r["train_limit"] == tl],
        ["dataset", "purification_method"],
    )

    # Also get 'none' baseline per dataset
    none_data = aggregate(
        [r for r in rows if r["contamination_rate"] == "0.3" and
         r["train_limit"] == tl and r["purification_method"] == "none"],
        ["dataset"],
    )

    datasets = sorted(set(k[0] for k in data.keys() if k[1] == "none"))

    # Sort datasets by improvement of ensemble
    def get_improvement(ds):
        none_val = none_data.get((ds,), 0)
        ens_val = data.get((ds, "ensemble_loo_mahal"), none_val)
        return ens_val - none_val

    datasets = sorted(datasets, key=get_improvement, reverse=True)

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    n_methods = len(methods_to_show)
    width = 0.8 / n_methods
    x = np.arange(len(datasets))

    for midx, method in enumerate(methods_to_show):
        improvements = []
        for ds in datasets:
            none_val = none_data.get((ds,), 0)
            method_val = data.get((ds, method), none_val)
            improvements.append(method_val - none_val)

        color = METHOD_COLORS.get(method, "#333333")
        label = METHOD_LABELS.get(method, method)
        offset = (midx - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, improvements, width, color=color, label=label, alpha=0.85)

    # Mark LOCO datasets
    for i, ds in enumerate(datasets):
        if ds in LOCO_DATASETS:
            ax.axvspan(i - 0.45, i + 0.45, alpha=0.08, color="red")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("AUROC Improvement vs. No Purification")
    ax.set_title(f"Per-Dataset AUROC Improvement at 30% Contamination (TL={tl})")
    ax.set_xticks(x)
    ax.set_xticklabels([d.split("/")[-1] for d in datasets], rotation=45, ha="right", fontsize=7)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    path = output_dir / f"fig3_per_dataset_improvement_tl{tl}.pdf"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 4: Risk-Reward Scatter Plot (Insurance Argument)
# ============================================================================

def fig_risk_reward_scatter(rows, output_dir):
    """Scatter: clean penalty (x) vs contamination benefit (y) per dataset."""
    method = "ensemble_loo_mahal"

    # Get clean penalty per dataset (cont=0.0)
    clean_data = aggregate(
        [r for r in rows if r["contamination_rate"] == "0.0"],
        ["dataset", "purification_method"],
    )

    # Get contaminated benefit per dataset (cont=0.3)
    dirty_data = aggregate(
        [r for r in rows if r["contamination_rate"] == "0.3"],
        ["dataset", "purification_method"],
    )

    datasets = sorted(set(k[0] for k in clean_data.keys() if k[1] == "none"))

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    for ds in datasets:
        # Clean penalty = method(cont=0) - none(cont=0)
        clean_none = clean_data.get((ds, "none"), 0)
        clean_method = clean_data.get((ds, method), 0)
        penalty = clean_method - clean_none  # negative = penalty

        # Contamination benefit = method(cont=0.3) - none(cont=0.3)
        dirty_none = dirty_data.get((ds, "none"), 0)
        dirty_method = dirty_data.get((ds, method), 0)
        benefit = dirty_method - dirty_none  # positive = benefit

        benchmark = get_benchmark(ds)
        color = {"mvtec_AD": "#0072B2", "VisA": "#009E73", "btad": "#D55E00",
                 "mvtec_loco_AD": "#CC79A7"}.get(benchmark, "#888888")
        marker = {"mvtec_AD": "o", "VisA": "s", "btad": "D",
                  "mvtec_loco_AD": "^"}.get(benchmark, "o")

        ax.scatter(penalty * 100, benefit * 100, c=color, marker=marker, s=60,
                   alpha=0.8, edgecolors="white", linewidths=0.5)

    # Add quadrant labels
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")

    # Insurance zone
    ax.fill_between([-5, 0], [0, 0], [15, 15], alpha=0.03, color="green")
    ax.fill_between([0, 3], [0, 0], [15, 15], alpha=0.03, color="yellow")

    ax.text(0.02, 0.98, "Win-Win\n(purify always)",
            transform=ax.transAxes, fontsize=8, va="top", ha="left",
            color="green", alpha=0.6)
    ax.text(0.98, 0.98, "Net positive\n(worth the cost)",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            color="orange", alpha=0.6)

    # Legend for benchmarks
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#0072B2", label="MVTec AD", markersize=8),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#009E73", label="VisA", markersize=8),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#D55E00", label="BTAD", markersize=8),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#CC79A7", label="MVTec LOCO", markersize=8),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    ax.set_xlabel("Clean Penalty (% AUROC)")
    ax.set_ylabel("Contamination Benefit (% AUROC)")
    ax.set_title("Risk-Reward Analysis: Ensemble Purification")
    ax.grid(True, alpha=0.3)

    path = output_dir / "fig4_risk_reward_scatter.pdf"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 5: Extended Contamination Curve (0.0 - 0.5)
# ============================================================================

def fig_extended_contamination(rows_main, rows_extra, output_dir):
    """AUROC vs contamination 0-50% using combined main + extra data."""
    methods = [
        "none", "random_patch", "lof_patch",
        "cosine_loo", "mahalanobis_patch", "loo_patch",
        "ensemble_loo_mahal", "oracle_patch",
    ]

    all_rows = rows_main + rows_extra

    # Aggregate by (cont_rate, method) -> mean AUROC
    data = aggregate_with_std(
        [r for r in all_rows if r["purification_method"] in methods],
        ["contamination_rate", "purification_method"],
    )

    cont_rates = sorted(set(float(r["contamination_rate"]) for r in all_rows
                            if r["purification_method"] in methods))

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    for method in methods:
        means = []
        xs = []
        for cr in cont_rates:
            key = (str(cr), method)
            if key in data:
                m, _ = data[key]
                means.append(m)
                xs.append(cr)

        if not means:
            continue

        means = np.array(means)
        color = METHOD_COLORS.get(method, "#333333")
        label = METHOD_LABELS.get(method, method)
        marker = METHOD_MARKERS.get(method, "o")
        lw = 3.0 if method in ("ensemble_loo_mahal", "oracle_patch", "none") else 2.0
        ls = "--" if method == "oracle_patch" else "-"
        ms = 9 if method in ("ensemble_loo_mahal", "oracle_patch", "none") else 7

        ax.plot(xs, means, color=color, marker=marker, label=label,
                linewidth=lw, linestyle=ls, markersize=ms,
                markeredgecolor=color, markerfacecolor=color,
                markeredgewidth=1.2, zorder=3 if method in ("ensemble_loo_mahal",) else 2)

    ax.set_xlabel("Contamination Rate", fontsize=12)
    ax.set_ylabel("Image AUROC", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=9, edgecolor="#CCCCCC")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    path = output_dir / "fig5_extended_contamination.pdf"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 6: Ablation - Percentile Threshold
# ============================================================================

def fig_ablation_percentile(rows, output_dir):
    """Ablation study: AUROC at different percentile thresholds."""
    # Find purification method (ensemble_loo_mahal or loo_patch)
    purif_methods = sorted(set(r["purification_method"] for r in rows
                               if r["purification_method"] != "none"))
    if not purif_methods:
        print("  [SKIP] No ablation percentile data (no purification methods)")
        return
    purif_method = purif_methods[0]
    print(f"  Ablation method: {purif_method}")

    # Aggregate by (percentile, cont_rate) -> mean AUROC for purification method
    data = aggregate(
        [r for r in rows if r["purification_method"] == purif_method],
        ["percentile_threshold", "contamination_rate"],
    )

    # Also aggregate 'none' baseline (same across percentiles)
    data_none = aggregate(
        [r for r in rows if r["purification_method"] == "none"],
        ["contamination_rate"],
    )

    percentiles = sorted(set(r["percentile_threshold"] for r in rows
                             if r["purification_method"] == purif_method))

    if not percentiles:
        print("  [SKIP] No ablation percentile data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: AUROC at different contamination rates
    cont_rates = ["0.0", "0.1", "0.2", "0.3"]
    cont_colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    for cr, color in zip(cont_rates, cont_colors):
        aurocs = []
        pcts = []
        for p in percentiles:
            key = (p, cr)
            if key in data:
                aurocs.append(data[key])
                pcts.append(int(p))
        if aurocs:
            ax1.plot(pcts, aurocs, color=color, marker="o", label=f"cont={cr}",
                     linewidth=2, markersize=6)

    # Add 'none' baseline horizontal lines
    for cr, color in zip(cont_rates, cont_colors):
        key = (cr,)
        if key in data_none:
            ax1.axhline(data_none[key], color=color, linewidth=1, linestyle="--", alpha=0.4)

    ax1.set_xlabel("Percentile Threshold")
    ax1.set_ylabel("Image AUROC")
    ax1.set_title("Effect of Percentile Threshold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: Clean penalty vs contamination benefit per percentile
    penalties = []
    benefits = []
    pct_labels = []
    for p in percentiles:
        clean_none = data.get(("95", "0.0"), None)  # reference
        p_clean = data.get((p, "0.0"), None)
        p_dirty = data.get((p, "0.3"), None)
        none_dirty = aggregate(
            [r for r in rows if r["purification_method"] == "none" and r["contamination_rate"] == "0.3"],
            [],
        )

        # Get none baseline at 0.3
        none_at_03 = None
        none_vals = [float(r["img_auroc"]) for r in rows
                     if r["purification_method"] == "none" and r["contamination_rate"] == "0.3"]
        if none_vals:
            none_at_03 = np.mean(none_vals)

        none_at_00 = None
        none_clean = [float(r["img_auroc"]) for r in rows
                      if r["purification_method"] == "none" and r["contamination_rate"] == "0.0"]
        if none_clean:
            none_at_00 = np.mean(none_clean)

        if p_clean is not None and p_dirty is not None and none_at_03 is not None and none_at_00 is not None:
            penalty = (p_clean - none_at_00) * 100
            benefit = (p_dirty - none_at_03) * 100
            penalties.append(penalty)
            benefits.append(benefit)
            pct_labels.append(f"p={p}")

    if penalties:
        colors_pct = plt.cm.viridis(np.linspace(0.2, 0.8, len(penalties)))
        for i in range(len(penalties)):
            ax2.scatter(penalties[i], benefits[i], c=[colors_pct[i]], s=100, zorder=5)
            ax2.annotate(pct_labels[i], (penalties[i], benefits[i]),
                         textcoords="offset points", xytext=(8, 5), fontsize=8)

        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.axvline(0, color="black", linewidth=0.5)
        ax2.set_xlabel("Clean Penalty (% AUROC)")
        ax2.set_ylabel("Contamination Benefit (% AUROC)")
        ax2.set_title("Percentile: Cost vs Benefit")
        ax2.grid(True, alpha=0.3)

    path = output_dir / "fig6_ablation_percentile.pdf"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 7: Train Limit Comparison
# ============================================================================

def fig_train_limit_comparison(rows_main, rows_tl5, output_dir):
    """Compare purification across N=5, 10, 20."""
    methods = ["none", "ensemble_loo_mahal", "oracle_patch"]
    all_rows = rows_main + rows_tl5

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    tls = ["5", "10", "20"]
    cont_rates = sorted(set(float(r["contamination_rate"]) for r in all_rows))
    cont_rates = [cr for cr in cont_rates if cr >= 0.0]

    short_labels = {
        "none": "Dirty",
        "ensemble_loo_mahal": "Ensemble",
        "oracle_patch": "Oracle",
    }

    for method in methods:
        for tl in tls:
            means = []
            xs = []
            for cr in cont_rates:
                vals = [float(r["img_auroc"]) for r in all_rows
                        if r["purification_method"] == method
                        and r["train_limit"] == tl
                        and r["contamination_rate"] == str(cr)]
                if vals:
                    means.append(np.mean(vals))
                    xs.append(cr)

            if not means:
                continue

            color = METHOD_COLORS.get(method, "#333333")
            label = f"{short_labels[method]} (N={tl})"
            ls = {"5": ":", "10": "-", "20": "--"}.get(tl, "-")
            marker = {"5": "v", "10": "o", "20": "s"}.get(tl, "o")
            lw = 2.8 if method == "ensemble_loo_mahal" else 2.2
            ms = 8

            ax.plot(xs, means, color=color, marker=marker, label=label,
                    linewidth=lw, linestyle=ls, markersize=ms,
                    markeredgecolor=color, markerfacecolor=color,
                    markeredgewidth=1.2)

    ax.set_xlabel("Contamination Rate", fontsize=12)
    ax.set_ylabel("Image AUROC", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(loc="lower left", fontsize=8, ncol=3, framealpha=0.95, edgecolor="#CCCCCC")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))

    path = output_dir / "fig7_train_limit_comparison.pdf"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 8: DINOv3 vs CLIP Backbone
# ============================================================================

def fig_backbone_comparison(rows_dino, rows_clip, output_dir):
    """Compare DINOv3 and CLIP backbones."""
    methods = ["none", "ensemble_loo_mahal", "oracle_patch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: AUROC vs contamination for both backbones
    cont_rates = [0.0, 0.1, 0.2, 0.3]

    for backbone_name, rows, ls in [("DINOv2", rows_dino, "-"), ("CLIP", rows_clip, "--")]:
        for method in methods:
            means = []
            xs = []
            for cr in cont_rates:
                vals = [float(r["img_auroc"]) for r in rows
                        if r["purification_method"] == method
                        and r["contamination_rate"] == str(cr)
                        and r["train_limit"] in ("10", "20")]
                if vals:
                    means.append(np.mean(vals))
                    xs.append(cr)

            if not means:
                continue

            color = METHOD_COLORS.get(method, "#333333")
            label = f"{backbone_name} — {METHOD_LABELS.get(method, method)}"
            ax1.plot(xs, means, color=color, linestyle=ls, marker="o",
                     linewidth=2, markersize=5, label=label)

    ax1.set_xlabel("Contamination Rate")
    ax1.set_ylabel("Image AUROC")
    ax1.set_title("AUROC: DINOv2 vs CLIP Backbones")
    ax1.legend(fontsize=7, loc="lower left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    # Right: Recovery comparison
    for backbone_name, rows, color in [("DINOv2", rows_dino, "#0072B2"), ("CLIP", rows_clip, "#D55E00")]:
        # Calculate recovery per cont_rate for ensemble
        recoveries = []
        xs = []
        for cr in [0.1, 0.2, 0.3]:
            method_vals = [float(r["img_auroc"]) for r in rows
                           if r["purification_method"] == "ensemble_loo_mahal"
                           and r["contamination_rate"] == str(cr)
                           and r["train_limit"] in ("10", "20")]
            none_vals = [float(r["img_auroc"]) for r in rows
                         if r["purification_method"] == "none"
                         and r["contamination_rate"] == str(cr)
                         and r["train_limit"] in ("10", "20")]
            clean_vals = [float(r["img_auroc"]) for r in rows
                          if r["purification_method"] == "none"
                          and r["contamination_rate"] == "0.0"
                          and r["train_limit"] in ("10", "20")]

            if method_vals and none_vals and clean_vals:
                m = np.mean(method_vals)
                n = np.mean(none_vals)
                c = np.mean(clean_vals)
                if c - n != 0:
                    rec = (m - n) / (c - n) * 100
                    recoveries.append(rec)
                    xs.append(cr)

        if recoveries:
            ax2.plot(xs, recoveries, color=color, marker="o",
                     linewidth=2.5, markersize=8, label=backbone_name)

    ax2.set_xlabel("Contamination Rate")
    ax2.set_ylabel("Recovery (%)")
    ax2.set_title("Damage Recovery: DINOv2 vs CLIP")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    path = output_dir / "fig8_backbone_comparison.pdf"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 9: Per-Benchmark Box Plot
# ============================================================================

def fig_benchmark_boxplot(rows, output_dir):
    """Box plot of AUROC improvement per benchmark at cont=0.3."""
    method = "ensemble_loo_mahal"

    # Get per-dataset improvements
    data_method = aggregate(
        [r for r in rows if r["contamination_rate"] == "0.3"
         and r["purification_method"] == method],
        ["dataset"],
    )
    data_none = aggregate(
        [r for r in rows if r["contamination_rate"] == "0.3"
         and r["purification_method"] == "none"],
        ["dataset"],
    )

    improvements = defaultdict(list)
    for ds in data_method:
        if ds in data_none:
            diff = data_method[ds] - data_none[ds]
            benchmark = get_benchmark(ds[0]) if isinstance(ds, tuple) else get_benchmark(ds)
            improvements[benchmark].append(diff * 100)

    if not improvements:
        print("  [SKIP] No data for benchmark boxplot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    benchmarks = ["mvtec_AD", "VisA", "btad", "mvtec_loco_AD"]
    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
    plot_data = []
    labels = []
    box_colors = []

    for bm, color in zip(benchmarks, colors):
        if bm in improvements:
            plot_data.append(improvements[bm])
            labels.append(BENCHMARK_NAMES.get(bm, bm))
            box_colors.append(color)

    bp = ax.boxplot(plot_data, labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(0, color="red", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_ylabel("AUROC Improvement (% points)")
    ax.set_title("Purification Benefit by Benchmark (Ensemble, cont=30%)")
    ax.grid(True, axis="y", alpha=0.3)

    path = output_dir / "fig9_benchmark_boxplot.pdf"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# LATEX TABLES
# ============================================================================

def generate_main_results_table(rows, output_dir, tl="10"):
    """Generate LaTeX table for main results."""
    methods = ["none", "random_patch", "loo_patch", "mahalanobis_patch",
               "cosine_loo", "ensemble_loo_mahal", "oracle_patch"]

    cont_rates = ["0.0", "0.1", "0.2", "0.3"]

    # Aggregate by (cont_rate, method)
    data = aggregate_with_std(
        [r for r in rows if r["train_limit"] == tl and r["purification_method"] in methods],
        ["contamination_rate", "purification_method"],
    )

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Image AUROC across contamination rates (TL=" + tl + "). Mean $\\pm$ std over 35 datasets and 5 seeds.}")
    lines.append("\\label{tab:main_results}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l" + "c" * len(cont_rates) + "}")
    lines.append("\\toprule")
    lines.append("Method & " + " & ".join(f"$c={c}$" for c in cont_rates) + " \\\\")
    lines.append("\\midrule")

    for method in methods:
        label = METHOD_LABELS.get(method, method)
        if method == "oracle_patch":
            lines.append("\\midrule")
        vals = []
        for cr in cont_rates:
            key = (cr, method)
            if key in data:
                m, s = data[key]
                vals.append(f"{m:.4f} $\\pm$ {s:.4f}")
            else:
                vals.append("---")
        line = f"{label} & " + " & ".join(vals) + " \\\\"
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    path = output_dir / f"table_main_results_tl{tl}.tex"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")


def generate_benchmark_table(rows, output_dir):
    """Generate LaTeX table: per-benchmark results at cont=0.3."""
    methods = ["none", "loo_patch", "mahalanobis_patch",
               "ensemble_loo_mahal", "oracle_patch"]
    benchmarks = ["mvtec_AD", "VisA", "btad", "mvtec_loco_AD"]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Image AUROC per benchmark at $c=0.3$. Mean over seeds and TL.}")
    lines.append("\\label{tab:benchmark_results}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l" + "c" * len(benchmarks) + "}")
    lines.append("\\toprule")
    lines.append("Method & " + " & ".join(BENCHMARK_NAMES.get(b, b) for b in benchmarks) + " \\\\")
    lines.append("\\midrule")

    for method in methods:
        label = METHOD_LABELS.get(method, method)
        if method == "oracle_patch":
            lines.append("\\midrule")
        vals = []
        for bm in benchmarks:
            bm_vals = [float(r["img_auroc"]) for r in rows
                       if r["purification_method"] == method
                       and r["contamination_rate"] == "0.3"
                       and r["dataset"].startswith(bm + "/")]
            if bm_vals:
                vals.append(f"{np.mean(bm_vals):.4f}")
            else:
                vals.append("---")
        line = f"{label} & " + " & ".join(vals) + " \\\\"
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    path = output_dir / "table_benchmark_results.tex"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")


def generate_statistical_table(output_dir):
    """Generate LaTeX table from statistical test results."""
    wilcoxon_path = output_dir / "wilcoxon_dinov3_cont30.csv"
    if not wilcoxon_path.exists():
        print(f"  [SKIP] {wilcoxon_path} not found — run analysis_statistical_tests.py first")
        return

    rows = []
    with open(wilcoxon_path, "r") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            rows.append(row)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Statistical tests: each method vs.\\ no purification at $c=0.3$. Holm-Bonferroni corrected.}")
    lines.append("\\label{tab:statistical_tests}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Method & Mean $\\Delta$ & 95\\% CI & Wilcoxon $p$ & Cliff's $\\delta$ & A12 \\\\")
    lines.append("\\midrule")

    for row in rows:
        label = METHOD_LABELS.get(row["method"], row["method"])
        md = float(row["mean_diff"])
        ci_lo = float(row["boot_ci_lower"])
        ci_hi = float(row["boot_ci_upper"])
        p = float(row["p_adjusted"])
        d = float(row["cliffs_delta"])
        a12 = float(row["a12"])
        interp = row["cliffs_interp"]

        p_str = f"${p:.1e}$" if p < 0.001 else f"${p:.3f}$"
        sig_marker = "$^{***}$" if p < 0.001 else ("$^{**}$" if p < 0.01 else ("$^{*}$" if p < 0.05 else ""))

        line = (f"{label} & {md:+.4f} & [{ci_lo:+.4f}, {ci_hi:+.4f}] "
                f"& {p_str}{sig_marker} & {d:+.3f} ({interp}) & {a12:.3f} \\\\")
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    path = output_dir / "table_statistical_tests.tex"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures and tables")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--figures-only", action="store_true", help="Skip LaTeX tables")
    parser.add_argument("--tables-only", action="store_true", help="Skip figures")
    args = parser.parse_args()

    output_dir = Path("output/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"  GENERATING PAPER FIGURES AND TABLES")
    print(f"{'#' * 70}\n")

    # Load all data
    print("Loading data...")
    rows_main = load_csv("output/exp_p3_002_full/results_v2.csv")
    rows_tl5 = load_csv("output/exp_p3_002_full/results_tl5.csv")
    rows_extra = load_csv("output/exp_p3_002_full/results_extra_cont.csv")
    rows_ablation_pct = load_csv("output/exp_p3_002_full/ablation_full_percentile.csv")
    rows_clip = load_csv("output/exp_p3_002_full_clip/results_v2.csv")

    if not rows_main:
        print("[ERROR] No main results found. Run experiments first.")
        return

    # ================================================================
    # FIGURES
    # ================================================================
    if not args.tables_only:
        print(f"\n{'=' * 50}")
        print("  Generating figures...")
        print(f"{'=' * 50}")

        print("\n[Fig 2] AUROC vs Contamination Rate")
        fig_auroc_vs_contamination(rows_main, output_dir, tl="10")
        fig_auroc_vs_contamination(rows_main, output_dir, tl="20")

        print("\n[Fig 3] Per-Dataset Improvement")
        fig_per_dataset_improvement(rows_main, output_dir, tl="10")

        print("\n[Fig 4] Risk-Reward Scatter")
        fig_risk_reward_scatter(rows_main, output_dir)

        if rows_extra:
            print("\n[Fig 5] Extended Contamination (0-50%)")
            fig_extended_contamination(rows_main, rows_extra, output_dir)

        if rows_ablation_pct:
            print("\n[Fig 6] Ablation Percentile")
            fig_ablation_percentile(rows_ablation_pct, output_dir)

        if rows_tl5:
            print("\n[Fig 7] Train Limit Comparison")
            fig_train_limit_comparison(rows_main, rows_tl5, output_dir)

        if rows_clip:
            print("\n[Fig 8] Backbone Comparison (DINOv3 vs CLIP)")
            fig_backbone_comparison(rows_main, rows_clip, output_dir)

        print("\n[Fig 9] Benchmark Boxplot")
        fig_benchmark_boxplot(rows_main, output_dir)

    # ================================================================
    # TABLES
    # ================================================================
    if not args.figures_only:
        print(f"\n{'=' * 50}")
        print("  Generating LaTeX tables...")
        print(f"{'=' * 50}")

        print("\n[Table 1] Main Results")
        generate_main_results_table(rows_main, output_dir, tl="10")
        generate_main_results_table(rows_main, output_dir, tl="20")

        print("\n[Table 2] Benchmark Results")
        generate_benchmark_table(rows_main, output_dir)

        print("\n[Table 3] Statistical Tests")
        generate_statistical_table(output_dir)

    print(f"\n{'#' * 70}")
    print(f"  ALL DONE — figures and tables saved to {output_dir}")
    print(f"{'#' * 70}\n")


if __name__ == "__main__":
    main()
