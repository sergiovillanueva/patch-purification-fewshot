"""
Statistical tests for Paper 3: Dirty Few-Shot Self-Purification.

Runs all statistical analyses required for a Q1 Pattern Recognition paper:
1. Friedman test (omnibus non-parametric test across methods)
2. Nemenyi post-hoc (pairwise comparison with CD diagram)
3. Wilcoxon signed-rank (pairwise vs dirty baseline)
4. Cliff's delta / Vargha-Delaney A12 (effect sizes)
5. Bootstrap 95% CI on recovery and AUROC delta
6. Critical Difference (CD) diagram generation

Unit of analysis: dataset (n=35 or n=30 excl. LOCO), aggregated over seeds.

Usage:
    python analysis_statistical_tests.py                     # Full analysis
    python analysis_statistical_tests.py --debug             # Quick test (prints only, no saves)
    python analysis_statistical_tests.py --exclude-loco      # Exclude LOCO datasets
    python analysis_statistical_tests.py --backbone clip     # CLIP results
"""

import argparse
import csv
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

# Fix Windows encoding for Unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Suppress warnings for clean output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================

CORE_METHODS = [
    "none",
    "loo_patch",
    "mahalanobis_patch",
    "cosine_loo",
    "ensemble_loo_mahal",
    "oracle_patch",
]

# For Friedman/Nemenyi: only methods with FULL 35-dataset coverage
# (exclude oracle, image_level_loo, adaptive, global_iforest, pca — partial coverage)
RANKABLE_METHODS = [
    "none",
    "random_patch",
    "lof_patch",
    "loo_patch",
    "mahalanobis_patch",
    "cosine_loo",
    "ensemble_loo_mahal",
]

# Methods to compare vs 'none' baseline in Wilcoxon tests (full coverage only)
PAIRWISE_METHODS = [
    "loo_patch",
    "mahalanobis_patch",
    "cosine_loo",
    "ensemble_loo_mahal",
    "lof_patch",
    "random_patch",
]

LOCO_DATASETS = [
    "mvtec_loco_AD/breakfast_box",
    "mvtec_loco_AD/juice_bottle",
    "mvtec_loco_AD/pushpins",
    "mvtec_loco_AD/screw_bag",
    "mvtec_loco_AD/splicing_connectors",
]

CONT_RATES_MAIN = ["0.3"]  # Primary analysis at 30% contamination
TRAIN_LIMITS_MAIN = ["10", "20"]  # Both TLs


# ============================================================================
# DATA LOADING
# ============================================================================

def load_results(csv_path: str, delimiter: str = ";") -> list[dict]:
    """Load CSV results file."""
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            rows.append(row)
    print(f"  Loaded {len(rows)} rows from {csv_path}")
    return rows


def aggregate_by_dataset(
    rows: list[dict],
    methods: list[str],
    cont_rates: list[str],
    train_limits: list[str],
    exclude_loco: bool = False,
    metric: str = "img_auroc",
) -> dict[str, dict[str, float]]:
    """
    Aggregate metric by dataset, averaging over seeds, TLs, and cont_rates.

    Returns: {dataset: {method: mean_metric}}
    """
    # Collect all values per (dataset, method)
    values = defaultdict(lambda: defaultdict(list))

    for row in rows:
        method = row["purification_method"]
        if method not in methods:
            continue
        if row["contamination_rate"] not in cont_rates:
            continue
        if row["train_limit"] not in train_limits:
            continue
        dataset = row["dataset"]
        if exclude_loco and dataset in LOCO_DATASETS:
            continue

        try:
            val = float(row[metric])
            values[dataset][method].append(val)
        except (ValueError, KeyError):
            continue

    # Average over seeds/TLs/cont_rates per (dataset, method)
    result = {}
    for dataset in sorted(values.keys()):
        method_means = {}
        valid = True
        for method in methods:
            vals = values[dataset][method]
            if len(vals) == 0:
                valid = False
                break
            method_means[method] = np.mean(vals)
        if valid:
            result[dataset] = method_means

    return result


def aggregate_paired(
    rows: list[dict],
    method_a: str,
    method_b: str,
    cont_rates: list[str],
    train_limits: list[str],
    exclude_loco: bool = False,
    metric: str = "img_auroc",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Get paired observations (one per dataset) for two methods.

    Returns: (values_a, values_b, dataset_names)
    """
    values_a = defaultdict(list)
    values_b = defaultdict(list)

    for row in rows:
        method = row["purification_method"]
        if method not in (method_a, method_b):
            continue
        if row["contamination_rate"] not in cont_rates:
            continue
        if row["train_limit"] not in train_limits:
            continue
        dataset = row["dataset"]
        if exclude_loco and dataset in LOCO_DATASETS:
            continue

        try:
            val = float(row[metric])
        except (ValueError, KeyError):
            continue

        if method == method_a:
            values_a[dataset].append(val)
        else:
            values_b[dataset].append(val)

    # Only keep datasets with both methods
    common = sorted(set(values_a.keys()) & set(values_b.keys()))
    a = np.array([np.mean(values_a[d]) for d in common])
    b = np.array([np.mean(values_b[d]) for d in common])
    return a, b, common


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def friedman_test(data: dict[str, dict[str, float]], methods: list[str]) -> dict:
    """
    Run Friedman test across methods.

    data: {dataset: {method: mean_auroc}}
    Returns: dict with statistic, p_value, n_datasets, n_methods, mean_ranks
    """
    datasets = sorted(data.keys())
    n = len(datasets)
    k = len(methods)

    # Build matrix: n_datasets x n_methods
    matrix = np.zeros((n, k))
    for i, ds in enumerate(datasets):
        for j, m in enumerate(methods):
            matrix[i, j] = data[ds][m]

    # Friedman test
    stat, p = stats.friedmanchisquare(*[matrix[:, j] for j in range(k)])

    # Compute mean ranks (higher AUROC = rank 1)
    ranks = np.zeros_like(matrix)
    for i in range(n):
        # Rank from highest (1) to lowest (k)
        ranks[i] = stats.rankdata(-matrix[i])

    mean_ranks = {methods[j]: np.mean(ranks[:, j]) for j in range(k)}

    return {
        "statistic": stat,
        "p_value": p,
        "n_datasets": n,
        "n_methods": k,
        "mean_ranks": mean_ranks,
        "rank_matrix": ranks,
        "value_matrix": matrix,
    }


def nemenyi_posthoc(friedman_result: dict, methods: list[str], alpha: float = 0.05) -> dict:
    """
    Nemenyi post-hoc test for pairwise method comparison.

    Returns: dict with CD value, pairwise p-values
    """
    try:
        import scikit_posthocs as sp
    except ImportError:
        print("  [WARNING] scikit-posthocs not installed. Skipping Nemenyi.")
        return {"error": "scikit-posthocs not installed"}

    matrix = friedman_result["value_matrix"]
    n, k = matrix.shape

    # Nemenyi post-hoc: returns p-value matrix
    # scikit_posthocs expects observations in columns
    p_values = sp.posthoc_nemenyi_friedman(matrix)

    # Critical Difference (Demsar 2006)
    # CD = q_alpha * sqrt(k * (k + 1) / (6 * n))
    # q_alpha values from studentized range table
    from scipy.stats import studentized_range
    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

    return {
        "p_values": p_values,
        "cd": cd,
        "alpha": alpha,
        "n_datasets": n,
        "n_methods": k,
        "q_alpha": q_alpha,
    }


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Wilcoxon signed-rank test for paired samples.
    Tests if b > a (one-sided: method b is better than method a).
    """
    diff = b - a
    n_nonzero = np.sum(diff != 0)

    if n_nonzero < 5:
        return {"statistic": np.nan, "p_value": 1.0, "n_pairs": len(a), "n_nonzero": n_nonzero}

    try:
        stat, p_two = stats.wilcoxon(a, b, alternative="two-sided")
        _, p_greater = stats.wilcoxon(a, b, alternative="greater")  # b > a
    except ValueError:
        return {"statistic": np.nan, "p_value": 1.0, "n_pairs": len(a), "n_nonzero": n_nonzero}

    return {
        "statistic": stat,
        "p_value_twosided": p_two,
        "p_value_greater": p_greater,
        "n_pairs": len(a),
        "n_nonzero": n_nonzero,
        "median_diff": np.median(diff),
        "mean_diff": np.mean(diff),
    }


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, str]:
    """
    Cliff's delta effect size (non-parametric).

    Ranges: [-1, 1]. Thresholds (Romano et al. 2006):
    |d| < 0.147: negligible
    |d| < 0.33: small
    |d| < 0.474: medium
    |d| >= 0.474: large
    """
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return 0.0, "negligible"

    # Count dominance pairs
    more = 0
    less = 0
    for ai in a:
        for bi in b:
            if bi > ai:
                more += 1
            elif bi < ai:
                less += 1

    d = (more - less) / (n_a * n_b)

    # Interpret
    abs_d = abs(d)
    if abs_d < 0.147:
        interpretation = "negligible"
    elif abs_d < 0.33:
        interpretation = "small"
    elif abs_d < 0.474:
        interpretation = "medium"
    else:
        interpretation = "large"

    return d, interpretation


def vargha_delaney_a12(a: np.ndarray, b: np.ndarray) -> float:
    """
    Vargha-Delaney A12 measure.
    A12 > 0.5 means b tends to be larger than a.
    A12 = (Cliff's delta + 1) / 2
    """
    d, _ = cliffs_delta(a, b)
    return (d + 1) / 2


def bootstrap_ci(
    data: np.ndarray,
    statistic=np.mean,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval.

    Returns: (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = statistic(sample)

    alpha = (1 - ci) / 2
    ci_lower = np.percentile(boot_stats, 100 * alpha)
    ci_upper = np.percentile(boot_stats, 100 * (1 - alpha))
    point = statistic(data)

    return point, ci_lower, ci_upper


# ============================================================================
# CD DIAGRAM
# ============================================================================

def _find_cliques(sorted_methods: list[tuple[str, float]], cd: float) -> list[list[int]]:
    """
    Find maximal cliques of methods not significantly different (Demsar 2006).

    A clique is a maximal contiguous range [i..j] such that
    rank[j] - rank[i] < cd. This greedy approach matches the standard
    CD diagram convention.
    """
    k = len(sorted_methods)
    ranks = [r for _, r in sorted_methods]

    # Find all maximal contiguous non-significant groups
    cliques = []
    for i in range(k):
        # Extend as far right as possible
        j = i
        while j + 1 < k and ranks[j + 1] - ranks[i] < cd:
            j += 1
        if j > i:  # At least 2 methods
            group = list(range(i, j + 1))
            # Only add if not a subset of an existing clique
            is_subset = False
            for existing in cliques:
                if set(group).issubset(set(existing)):
                    is_subset = True
                    break
            if not is_subset:
                # Remove any existing cliques that are subsets of this one
                cliques = [c for c in cliques if not set(c).issubset(set(group))]
                cliques.append(group)

    return cliques


def plot_cd_diagram(
    mean_ranks: dict[str, float],
    cd: float,
    n_datasets: int,
    output_path: str,
    title: str = "Critical Difference Diagram",
    alpha: float = 0.05,
):
    """
    Plot a Critical Difference (CD) diagram (Demsar 2006 style).
    Generates vector PDF for publication quality.

    Layout:
    - Top: axis with rank numbers + CD bar
    - Left side: best methods (lowest rank), with names
    - Right side: worst methods (highest rank), with names
    - Bottom: thick horizontal bars connecting methods NOT significantly different
    """
    # Sort methods by mean rank (best first)
    sorted_methods = sorted(mean_ranks.items(), key=lambda x: x[1])
    k = len(sorted_methods)

    # Clean method names for display
    name_map = {
        "none": "No purification",
        "random_patch": "Random (control)",
        "image_level_loo": "Image-level LOO",
        "loo_patch": "LOO (Euclidean)",
        "mahalanobis_patch": "Mahalanobis",
        "lof_patch": "LOF (patch)",
        "cosine_loo": "LOO (Cosine)",
        "ensemble_loo_mahal": "Ensemble (LOO+Mahal)",
        "oracle_patch": "Oracle",
    }

    # Find cliques
    cliques = _find_cliques(sorted_methods, cd)
    n_cliques = len(cliques)

    # --- Layout parameters ---
    low_rank = 1
    high_rank = k
    x_pad = 0.6            # horizontal padding for labels
    y_axis = 0.0           # horizontal axis y-position
    label_spacing = 0.50   # vertical spacing between labels
    clique_spacing = 0.30  # vertical spacing between clique bars

    # Split methods: left (best) and right (worst)
    n_left = (k + 1) // 2
    left_methods = sorted_methods[:n_left]
    right_methods = sorted_methods[n_left:]

    # Compute vertical extents
    y_top = y_axis + 1.2                                          # CD bar + numbers
    y_label_bottom = y_axis - 0.6 - max(n_left, len(right_methods)) * label_spacing
    y_clique_bottom = y_label_bottom - 0.5 - n_cliques * clique_spacing

    fig, ax = plt.subplots(1, 1, figsize=(12, 5.0))
    ax.set_xlim(low_rank - x_pad - 3.2, high_rank + x_pad + 3.2)
    ax.set_ylim(y_clique_bottom - 0.3, y_top + 0.2)
    ax.axis("off")

    # White background bbox for text labels (so lines don't cross through text)
    text_bbox = dict(boxstyle="round,pad=0.08", facecolor="white",
                     edgecolor="none", alpha=1.0)

    # === AXIS LINE ===
    ax.hlines(y_axis, low_rank, high_rank, colors="black", linewidth=2.0)
    for r in range(1, k + 1):
        ax.vlines(r, y_axis - 0.10, y_axis + 0.10, colors="black", linewidth=2.0)
        ax.text(r, y_axis + 0.18, str(r), ha="center", va="bottom", fontsize=15,
                fontweight="bold")

    # === CD BAR (above axis) ===
    cd_y = y_axis + 0.75
    cd_start = low_rank
    cd_end = low_rank + cd
    ax.hlines(cd_y, cd_start, cd_end, colors="#CC0000", linewidth=3.5)
    ax.vlines(cd_start, cd_y - 0.10, cd_y + 0.10, colors="#CC0000", linewidth=3.5)
    ax.vlines(cd_end, cd_y - 0.10, cd_y + 0.10, colors="#CC0000", linewidth=3.5)
    ax.text(
        (cd_start + cd_end) / 2, cd_y + 0.15,
        f"CD = {cd:.2f}",
        ha="center", va="bottom", fontsize=15, color="#CC0000", fontweight="bold",
    )

    # === LEFT SIDE: best methods ===
    x_label_left = low_rank - x_pad - 3.1
    for idx, (method, rank) in enumerate(left_methods):
        y_pos = y_axis - 0.6 - idx * label_spacing
        label = name_map.get(method, method)

        # Horizontal line from label to rank tick (draw BEFORE text)
        ax.hlines(y_pos, x_label_left + 0.05, rank, colors="#AAAAAA",
                  linewidth=0.9, zorder=1)
        # Vertical line from label height up to axis
        ax.vlines(rank, y_pos, y_axis, colors="#AAAAAA", linewidth=0.9, zorder=1)
        # Dot on axis
        ax.plot(rank, y_axis, "ko", markersize=7, zorder=5)
        # Label text (with white background so lines don't cross through)
        ax.text(
            x_label_left, y_pos, f"{label} ({rank:.2f})",
            ha="left", va="center", fontsize=15,
            fontweight="bold" if idx == 0 else "normal",
            bbox=text_bbox, zorder=4,
        )

    # === RIGHT SIDE: worst methods ===
    x_label_right = high_rank + x_pad + 3.1
    for idx, (method, rank) in enumerate(right_methods):
        y_pos = y_axis - 0.6 - idx * label_spacing
        label = name_map.get(method, method)

        ax.hlines(y_pos, rank, x_label_right - 0.05, colors="#AAAAAA",
                  linewidth=0.9, zorder=1)
        ax.vlines(rank, y_pos, y_axis, colors="#AAAAAA", linewidth=0.9, zorder=1)
        ax.plot(rank, y_axis, "ko", markersize=7, zorder=5)
        ax.text(
            x_label_right, y_pos, f"({rank:.2f}) {label}",
            ha="right", va="center", fontsize=15,
            bbox=text_bbox, zorder=4,
        )

    # === CLIQUE BARS (below labels) ===
    clique_colors = ["#2C3E50", "#7F8C8D"]  # alternating dark colors
    clique_y_start = y_label_bottom - 0.4
    for cidx, clique in enumerate(cliques):
        min_rank = sorted_methods[clique[0]][1]
        max_rank = sorted_methods[clique[-1]][1]
        bar_y = clique_y_start - cidx * clique_spacing
        bar_color = clique_colors[cidx % len(clique_colors)]
        ax.plot(
            [min_rank, max_rank], [bar_y, bar_y],
            color=bar_color, linewidth=5.5, solid_capstyle="round", zorder=3,
        )

    plt.tight_layout()
    fmt = "pdf" if output_path.endswith(".pdf") else "png"
    plt.savefig(output_path, format=fmt, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  CD diagram saved to {output_path}")


# ============================================================================
# REPORTING
# ============================================================================

def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[tuple[float, float, bool]]:
    """
    Holm-Bonferroni correction for multiple comparisons.

    Returns: [(original_p, adjusted_p, significant), ...]
    """
    k = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    adjusted = [0.0] * k
    max_adj = 0.0
    for rank, (orig_idx, p) in enumerate(indexed, 1):
        adj_p = min(p * (k - rank + 1), 1.0)
        adj_p = max(adj_p, max_adj)  # Ensure monotonicity
        max_adj = adj_p
        adjusted[orig_idx] = adj_p

    return [(p_values[i], adjusted[i], adjusted[i] < alpha) for i in range(k)]


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def run_all_tests(args):
    """Main analysis pipeline."""

    # Determine paths
    if args.backbone == "clip":
        csv_path = "output/exp_p3_002_full_clip/results_v2.csv"
        output_dir = Path("output/analysis_clip")
        backbone_label = "CLIP"
    else:
        csv_path = "output/exp_p3_002_full/results_v2.csv"
        output_dir = Path("output/analysis")
        backbone_label = "DINOv3"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"  STATISTICAL ANALYSIS — {backbone_label} backbone")
    print(f"  Exclude LOCO: {args.exclude_loco}")
    print(f"{'#' * 70}")

    # Load data
    print_section("1. Loading data")
    rows = load_results(csv_path)

    # ================================================================
    # 2. FRIEDMAN TEST
    # ================================================================
    print_section("2. Friedman Test (omnibus)")

    # Filter methods that exist in the data
    available_methods = set(r["purification_method"] for r in rows)
    rankable = [m for m in RANKABLE_METHODS if m in available_methods]
    print(f"  Rankable methods ({len(rankable)}): {rankable}")

    data = aggregate_by_dataset(
        rows, rankable, CONT_RATES_MAIN, TRAIN_LIMITS_MAIN,
        exclude_loco=args.exclude_loco,
    )
    print(f"  Datasets for Friedman: {len(data)}")

    friedman = friedman_test(data, rankable)
    print(f"\n  Friedman chi2 = {friedman['statistic']:.2f}")
    print(f"  p-value       = {friedman['p_value']:.2e}")
    print(f"  n_datasets    = {friedman['n_datasets']}")
    print(f"  n_methods     = {friedman['n_methods']}")
    print(f"\n  Mean ranks (lower = better):")
    for method, rank in sorted(friedman["mean_ranks"].items(), key=lambda x: x[1]):
        print(f"    {method:30s}  {rank:.3f}")

    # ================================================================
    # 3. NEMENYI POST-HOC + CD DIAGRAM
    # ================================================================
    print_section("3. Nemenyi Post-Hoc + CD Diagram")

    nemenyi = nemenyi_posthoc(friedman, rankable)
    if "error" not in nemenyi:
        print(f"\n  Critical Difference (CD) = {nemenyi['cd']:.3f}")
        print(f"  q_alpha = {nemenyi['q_alpha']:.3f}, alpha = {nemenyi['alpha']}")

        # Print pairwise p-values (significant pairs only)
        p_df = nemenyi["p_values"]
        print(f"\n  Significant differences (p < 0.05):")
        sig_found = False
        for i in range(len(rankable)):
            for j in range(i + 1, len(rankable)):
                p = p_df.iloc[i, j]
                if p < 0.05:
                    sig_found = True
                    print(f"    {rankable[i]:25s} vs {rankable[j]:25s}  p = {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*'}")
        if not sig_found:
            print("    (none at alpha=0.05)")

        # CD Diagram
        if not args.debug:
            loco_str = "_no_loco" if args.exclude_loco else ""
            cd_path = output_dir / f"cd_diagram_{backbone_label.lower()}{loco_str}_cont30.pdf"
            plot_cd_diagram(
                friedman["mean_ranks"],
                nemenyi["cd"],
                friedman["n_datasets"],
                str(cd_path),
                title=f"Critical Difference Diagram — {backbone_label}, cont=30%"
                      + (" (excl. LOCO)" if args.exclude_loco else ""),
            )
    else:
        print(f"  Skipped: {nemenyi['error']}")

    # ================================================================
    # 4. WILCOXON SIGNED-RANK (vs 'none' baseline)
    # ================================================================
    print_section("4. Wilcoxon Signed-Rank Tests (method vs none)")

    pairwise = [m for m in PAIRWISE_METHODS if m in available_methods]
    wilcoxon_results = {}
    p_values_for_correction = []

    for method in pairwise:
        a, b, ds = aggregate_paired(
            rows, "none", method, CONT_RATES_MAIN, TRAIN_LIMITS_MAIN,
            exclude_loco=args.exclude_loco,
        )
        result = wilcoxon_signed_rank(a, b)
        wilcoxon_results[method] = result
        p_values_for_correction.append(result.get("p_value_twosided", 1.0))

    # Holm-Bonferroni correction
    corrected = holm_bonferroni(p_values_for_correction)

    print(f"\n  {'Method':30s} {'Median Δ':>10s} {'Mean Δ':>10s} {'p (2-sided)':>12s} {'p (adj)':>10s} {'Sig?':>6s} {'n':>5s}")
    print(f"  {'-'*85}")
    for i, method in enumerate(pairwise):
        r = wilcoxon_results[method]
        orig_p, adj_p, sig = corrected[i]
        md = r.get("median_diff", 0)
        mn = r.get("mean_diff", 0)
        print(f"  {method:30s} {md:+10.4f} {mn:+10.4f} {orig_p:12.2e} {adj_p:10.2e} {'YES' if sig else 'no':>6s} {r['n_pairs']:5d}")

    # ================================================================
    # 5. EFFECT SIZES (Cliff's delta + A12)
    # ================================================================
    print_section("5. Effect Sizes (Cliff's delta + Vargha-Delaney A12)")

    print(f"\n  {'Method':30s} {'Cliff d':>10s} {'Interp':>12s} {'A12':>8s}")
    print(f"  {'-'*65}")
    for method in pairwise:
        a, b, ds = aggregate_paired(
            rows, "none", method, CONT_RATES_MAIN, TRAIN_LIMITS_MAIN,
            exclude_loco=args.exclude_loco,
        )
        d, interp = cliffs_delta(a, b)
        a12 = vargha_delaney_a12(a, b)
        print(f"  {method:30s} {d:+10.4f} {interp:>12s} {a12:8.4f}")

    # ================================================================
    # 6. BOOTSTRAP 95% CI
    # ================================================================
    print_section("6. Bootstrap 95% CI on AUROC improvement (vs none)")

    print(f"\n  {'Method':30s} {'Mean Δ':>10s} {'95% CI lower':>14s} {'95% CI upper':>14s}")
    print(f"  {'-'*72}")
    bootstrap_results = {}
    for method in pairwise:
        a, b, ds = aggregate_paired(
            rows, "none", method, CONT_RATES_MAIN, TRAIN_LIMITS_MAIN,
            exclude_loco=args.exclude_loco,
        )
        diff = b - a
        point, ci_lo, ci_hi = bootstrap_ci(diff, n_bootstrap=10000)
        bootstrap_results[method] = {"point": point, "ci_lower": ci_lo, "ci_upper": ci_hi}
        print(f"  {method:30s} {point:+10.4f} [{ci_lo:+14.4f}, {ci_hi:+14.4f}]")

    # ================================================================
    # 7. WIN RATES
    # ================================================================
    print_section("7. Win Rates (method beats none, per dataset)")

    print(f"\n  {'Method':30s} {'Wins':>6s} {'Ties':>6s} {'Losses':>8s} {'Win%':>8s}")
    print(f"  {'-'*62}")
    for method in pairwise:
        a, b, ds = aggregate_paired(
            rows, "none", method, CONT_RATES_MAIN, TRAIN_LIMITS_MAIN,
            exclude_loco=args.exclude_loco,
        )
        diff = b - a
        wins = np.sum(diff > 0.001)
        ties = np.sum(np.abs(diff) <= 0.001)
        losses = np.sum(diff < -0.001)
        win_pct = 100 * wins / len(diff) if len(diff) > 0 else 0
        print(f"  {method:30s} {wins:6d} {ties:6d} {losses:8d} {win_pct:7.1f}%")

    # ================================================================
    # 8. SAVE RESULTS
    # ================================================================
    if not args.debug:
        print_section("8. Saving results")

        loco_str = "_no_loco" if args.exclude_loco else ""
        prefix = f"{backbone_label.lower()}{loco_str}_cont30"

        # Save Friedman + rankings
        rankings_path = output_dir / f"rankings_{prefix}.csv"
        with open(rankings_path, "w", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["method", "mean_rank"])
            for method, rank in sorted(friedman["mean_ranks"].items(), key=lambda x: x[1]):
                w.writerow([method, f"{rank:.4f}"])
        print(f"  Saved: {rankings_path}")

        # Save Wilcoxon + effect sizes
        tests_path = output_dir / f"wilcoxon_{prefix}.csv"
        with open(tests_path, "w", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow([
                "method", "median_diff", "mean_diff",
                "p_twosided", "p_adjusted", "significant",
                "cliffs_delta", "cliffs_interp", "a12",
                "boot_mean", "boot_ci_lower", "boot_ci_upper",
                "n_pairs",
            ])
            for i, method in enumerate(pairwise):
                r = wilcoxon_results[method]
                orig_p, adj_p, sig = corrected[i]
                a, b, ds = aggregate_paired(
                    rows, "none", method, CONT_RATES_MAIN, TRAIN_LIMITS_MAIN,
                    exclude_loco=args.exclude_loco,
                )
                d, interp = cliffs_delta(a, b)
                a12 = vargha_delaney_a12(a, b)
                br = bootstrap_results.get(method, {})
                w.writerow([
                    method,
                    f"{r.get('median_diff', 0):.6f}",
                    f"{r.get('mean_diff', 0):.6f}",
                    f"{orig_p:.2e}",
                    f"{adj_p:.2e}",
                    "yes" if sig else "no",
                    f"{d:.4f}",
                    interp,
                    f"{a12:.4f}",
                    f"{br.get('point', 0):.6f}",
                    f"{br.get('ci_lower', 0):.6f}",
                    f"{br.get('ci_upper', 0):.6f}",
                    r["n_pairs"],
                ])
        print(f"  Saved: {tests_path}")

        # Save summary text
        summary_path = output_dir / f"summary_{prefix}.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Statistical Analysis Summary — {backbone_label}\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(f"Exclude LOCO: {args.exclude_loco}\n")
            f.write(f"Contamination rate: 30%\n")
            f.write(f"Train limits: {TRAIN_LIMITS_MAIN}\n")
            f.write(f"N datasets: {friedman['n_datasets']}\n\n")

            f.write(f"Friedman chi2 = {friedman['statistic']:.2f}, p = {friedman['p_value']:.2e}\n")
            if "error" not in nemenyi:
                f.write(f"Critical Difference (CD) = {nemenyi['cd']:.3f}\n\n")

            f.write("Mean ranks:\n")
            for method, rank in sorted(friedman["mean_ranks"].items(), key=lambda x: x[1]):
                f.write(f"  {method:30s}  {rank:.3f}\n")

            f.write(f"\nWilcoxon tests (vs none):\n")
            for i, method in enumerate(pairwise):
                r = wilcoxon_results[method]
                orig_p, adj_p, sig = corrected[i]
                f.write(f"  {method:30s}  Δ={r.get('mean_diff', 0):+.4f}  p={orig_p:.2e}  p_adj={adj_p:.2e}  {'SIG' if sig else ''}\n")

        print(f"  Saved: {summary_path}")

    # ================================================================
    # ALSO RUN FOR cont=0.0 (clean penalty analysis)
    # ================================================================
    print_section("BONUS: Clean Penalty Analysis (cont=0.0)")

    clean_methods = [m for m in ["none", "loo_patch", "mahalanobis_patch", "cosine_loo", "ensemble_loo_mahal"] if m in available_methods]

    print(f"\n  {'Method':30s} {'Mean Δ':>10s} {'95% CI':>30s} {'Wilcoxon p':>12s}")
    print(f"  {'-'*85}")
    for method in clean_methods[1:]:  # Skip 'none'
        a, b, ds = aggregate_paired(
            rows, "none", method, ["0.0"], TRAIN_LIMITS_MAIN,
            exclude_loco=args.exclude_loco,
        )
        diff = b - a
        point, ci_lo, ci_hi = bootstrap_ci(diff, n_bootstrap=10000)
        wres = wilcoxon_signed_rank(a, b)
        p = wres.get("p_value_twosided", 1.0)
        print(f"  {method:30s} {point:+10.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}] {p:12.2e}")

    print(f"\n{'#' * 70}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'#' * 70}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Statistical tests for Paper 3")
    parser.add_argument("--debug", action="store_true", help="Debug mode: print only, no file saves")
    parser.add_argument("--exclude-loco", action="store_true", help="Exclude LOCO datasets")
    parser.add_argument("--backbone", choices=["dinov3", "clip"], default="dinov3", help="Backbone")
    args = parser.parse_args()

    # Run full analysis
    run_all_tests(args)

    # Also run with LOCO excluded if not already
    if not args.exclude_loco and not args.debug:
        print("\n\n" + "=" * 70)
        print("  RE-RUNNING WITH LOCO EXCLUDED")
        print("=" * 70)
        args_no_loco = argparse.Namespace(**vars(args))
        args_no_loco.exclude_loco = True
        run_all_tests(args_no_loco)


if __name__ == "__main__":
    main()
