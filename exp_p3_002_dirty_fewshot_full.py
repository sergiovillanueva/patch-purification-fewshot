"""
P3_002: Core few-shot memory-bank experiment harness (contaminated support sets).
================================================================================
This is the base experiment engine. It builds few-shot memory banks from cached
features, injects contamination under two protocols (shared pool and disjoint
pool), applies the patch-level baselines (leave-one-out consistency, Mahalanobis,
their ensemble, LOF, random-patch and oracle references), and computes the
image-level metrics used throughout the paper.

It produces the two reference CSVs that the analysis scripts read:
  - output/exp_p3_002_full/results_v2.csv            (shared-pool protocol)
  - output/exp_p3_002_full/results_leakage_check.csv (disjoint-pool protocol)

The paper's finding is that the patch-level baselines do not recover the damage
under the disjoint-pool protocol; the deployable remedy is the image-level audit
in exp_p4_image_level.py. This script is kept as the shared infrastructure and
as the source of the protocol-comparison exhibit.

REQUIRES: run precache_features.py first to extract DINOv3 features to disk.
CPU-only (no GPU needed).

Run modes:
  python exp_p3_002_dirty_fewshot_full.py                       # Main: 35 datasets, 5 seeds, TL=5,10,20
  python exp_p3_002_dirty_fewshot_full.py --debug               # Debug: 2 datasets, 1 seed, TL=10
  python exp_p3_002_dirty_fewshot_full.py --ablation percentile # Ablation: vary percentile threshold
  python exp_p3_002_dirty_fewshot_full.py --ablation knn_k      # Ablation: vary KNN K

Backbone: DINOv3 (facebook/dinov3-vitl16-pretrain-lvd1689m), features pre-cached.
"""

import os
import sys
os.environ["PYTHONWARNINGS"] = "ignore"

import gc
import csv
import time
import argparse
import warnings
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import faiss
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from scipy.stats import skew, t as t_dist, median_abs_deviation
from scipy.ndimage import zoom as ndimage_zoom
from sklearn.mixture import GaussianMixture
from PIL import Image


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="P3_002 Full Validation: Dirty Few-Shot Self-Purification (CPU-only, cached features)"
    )
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: 2 datasets, 1 seed, 1 train_limit")
    parser.add_argument("--ablation", choices=["percentile", "knn_k"], default=None,
                        help="Run ablation study instead of main experiment")
    parser.add_argument("--gate-check", action="store_true",
                        help="Gate check: compare all methods on 4 Quick Test datasets, 2 seeds, TL=10, cont=0+0.3")
    parser.add_argument("--extra-cont", action="store_true",
                        help="Extra contamination rates (0.05, 0.4, 0.5) for degradation curve. "
                             "35 datasets, 5 seeds, TL=10. Uses --methods tier (~11h with core).")
    parser.add_argument("--leakage-check", action="store_true",
                        help="Data leakage verification: split test anomalies 50/50 "
                             "(contamination pool vs evaluation). 6 QT datasets, 5 seeds.")
    parser.add_argument("--tl5", action="store_true",
                        help="TL=5 few-shot experiment: 35 datasets, 5 seeds, TL=5, "
                             "cont={0,0.2,0.3} (m=0,1,2 images). 4 methods (~2h).")
    parser.add_argument("--ablation-full-pct", action="store_true",
                        help="Ablation percentile on 35 datasets: pct={90,95,99}, "
                             "cont=0.3 only, TL=10, ensemble only. 5 seeds (~1h).")
    parser.add_argument("--pixel-auroc", action="store_true",
                        help="Pixel-level AUROC experiment: compute patch score maps "
                             "and compare vs GT masks. 35 datasets, 5 seeds, TL=10, "
                             "cont={0,0.3}, 3 methods (none/ensemble/oracle) (~5h).")
    parser.add_argument("--methods", choices=["fast", "core", "extended", "all"], default="fast",
                        help="Method tier for main/extra-cont/leakage. "
                             "fast=none+loo_patch (default), "
                             "core=7 methods (~24h main), "
                             "extended=11 methods (~40h), "
                             "all=17 methods (~60h+)")
    parser.add_argument("--backbone", choices=["dino", "clip"], default="dino",
                        help="Feature backbone: dino (DINOv3-ViT-L, 784 patches) or "
                             "clip (CLIP ViT-L/14, 256 patches). Selects feature cache dir.")
    # Backward compat: --all-methods is equivalent to --methods all
    parser.add_argument("--all-methods", action="store_true",
                        help="(deprecated) Alias for --methods all")
    args = parser.parse_args()
    if args.all_methods:
        args.methods = "all"
    return args


# =============================================================================
# CONFIGURATION
# =============================================================================

# NOTE: CACHE_DIR is set dynamically in main() based on --backbone argument.
# Default shown here for module-level references; overridden at runtime.
CACHE_DIR = Path("output/feature_cache")
OUTPUT_DIR = Path("output/exp_p3_002_full")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Defaults
DEFAULT_KNN_K = 5
DEFAULT_PERCENTILE_THRESHOLD = 95

# KNN parallelism (best-effort; sklearn APIs differ by version)
DEFAULT_N_JOBS: Optional[int] = os.cpu_count() or None

# 35 datasets (MVTec AD + VisA + MVTec LOCO AD + BTAD)
ALL_DATASETS = [
    # MVTec AD (15)
    "mvtec_AD/bottle", "mvtec_AD/cable", "mvtec_AD/capsule", "mvtec_AD/carpet",
    "mvtec_AD/grid", "mvtec_AD/hazelnut", "mvtec_AD/leather", "mvtec_AD/metal_nut",
    "mvtec_AD/pill", "mvtec_AD/screw", "mvtec_AD/tile", "mvtec_AD/toothbrush",
    "mvtec_AD/transistor", "mvtec_AD/wood", "mvtec_AD/zipper",
    # VisA (12)
    "VisA/candle", "VisA/capsules", "VisA/cashew", "VisA/chewinggum", "VisA/fryum",
    "VisA/macaroni1", "VisA/macaroni2", "VisA/pcb1", "VisA/pcb2", "VisA/pcb3",
    "VisA/pcb4", "VisA/pipe_fryum",
    # MVTec LOCO AD (5)
    "mvtec_loco_AD/breakfast_box", "mvtec_loco_AD/juice_bottle",
    "mvtec_loco_AD/pushpins", "mvtec_loco_AD/screw_bag",
    "mvtec_loco_AD/splicing_connectors",
    # BTAD (3)
    "btad/01", "btad/02", "btad/03",
]

# Quick Test subset (representative categories - used for ablations)
QUICK_TEST_DATASETS = [
    "mvtec_AD/bottle", "mvtec_AD/cable", "mvtec_AD/screw",
    "VisA/pcb1", "mvtec_loco_AD/breakfast_box", "btad/01",
]

# Experiment settings
SEEDS = [0, 1, 2, 3, 4]
TRAIN_LIMITS = [10, 20]  # TL=5 excluded: cont=10% gives 0 contaminated images
CONTAMINATION_RATES = [0.0, 0.1, 0.2, 0.3]

# Purification methods
# - none: baseline (no purification)
# - loo_patch: LOO patch consistency (our method)
# - adaptive_loo_patch: LOO patch only if contamination detected (skewness gate)
# - random_patch: random removal of 5% patches (ablation control)
# - mahalanobis_patch: Mahalanobis distance to centroid (Ledoit-Wolf shrinkage)
# - image_level_loo: remove entire images by LOO image score
# - isolation_forest_patch: IsolationForest on patch features
# - global_iforest: IsolationForest on global (mean-pooled) image features
# - oracle_patch: remove exactly the contaminated image patches (upper bound)
PURIFICATION_METHODS = [
    "none", "loo_patch", "adaptive_loo_patch",
    "random_patch", "mahalanobis_patch", "image_level_loo",
    "isolation_forest_patch", "global_iforest", "oracle_patch",
    # Additional methods
    "ensemble_loo_mahal",   # Ensemble LOO + Mahalanobis scores (normalized + combined)
    "iterative_loo",        # Multi-round LOO: remove worst, recompute, repeat
    "cosine_loo",           # LOO with cosine distance instead of euclidean
    "lof_patch",            # Local Outlier Factor on patches
    "pca_patch",            # PCA reconstruction error as anomaly score
    "grubbs_adaptive_loo",  # LOO with Grubbs test for contamination detection
    # Extended methods
    "gmm_adaptive_loo",    # LOO with GMM+BIC contamination detection
    "mad_adaptive_loo",    # LOO with MAD z-score contamination detection
]

# Method tiers (ordered by paper importance)
# - fast: minimal for quick iteration (none + loo_patch)
# - core: 7 methods essential for paper tables (~5h full run)
# - extended: 11 methods for complete paper + supplementary (~8h)
# - all: 17 methods including experimental/adaptive (~25h)
METHODS_FAST = ["none", "loo_patch"]

METHODS_CORE = [
    "none",                    # dirty baseline
    "cosine_loo",              # BEST method (74.6% recovery in gate check)
    "loo_patch",               # euclidean LOO (reference, shows cosine > euclidean)
    "mahalanobis_patch",       # strong baseline (reviewer will ask "why not Mahalanobis?")
    "ensemble_loo_mahal",      # complementarity story (LOO + Mahal combined)
    "random_patch",            # negative control ("purification is not trivial")
    "oracle_patch",            # upper bound (contextualizes recovery %)
]

METHODS_EXTENDED = METHODS_CORE + [
    "image_level_loo",         # patch vs image granularity argument
    "global_iforest",          # off-the-shelf outlier detector baseline
    "adaptive_loo_patch",      # conditional purification (eliminates clean penalty)
    "pca_patch",               # reconstruction-based alternative
]

METHODS_ALL = PURIFICATION_METHODS  # all 17

def get_methods_for_tier(tier: str) -> list:
    """Return method list for the given tier name."""
    tiers = {
        "fast": METHODS_FAST,
        "core": METHODS_CORE,
        "extended": METHODS_EXTENDED,
        "all": METHODS_ALL,
    }
    methods = tiers[tier]
    # Ensure ordering follows PURIFICATION_METHODS (none first, then by original order)
    ordered = [m for m in PURIFICATION_METHODS if m in methods]
    return ordered

# Ablation settings (run on Quick Test datasets only)
ABLATION_PERCENTILE_VALUES = [80, 90, 95, 98, 99]
ABLATION_KNN_K_VALUES = [1, 3, 5, 7, 10]
ABLATION_CONTAMINATION_RATES = [0.0, 0.3]  # Clean baseline + worst case
ABLATION_TRAIN_LIMIT = 10

# Extra contamination rates for degradation curve
EXTRA_CONTAMINATION_RATES = [0.0, 0.05, 0.4, 0.5]  # 0.0 needed as clean baseline

# TL=5 few-shot experiment
# cont=10% with TL=5 gives round(5*0.1)=0 contaminated images → skip
# Use cont={0, 0.2, 0.3} which gives m={0, 1, 2} contaminated images
TL5_CONTAMINATION_RATES = [0.0, 0.2, 0.3]
TL5_METHODS = [
    "none", "loo_patch", "mahalanobis_patch", "ensemble_loo_mahal",
]

# Ablation percentile on all datasets
ABLATION_FULL_PCT_VALUES = [90, 95, 99]
ABLATION_FULL_PCT_METHODS = ["ensemble_loo_mahal"]  # Only ensemble (best method)

# Pixel AUROC experiment
# Patch grid: 28x28 = 784 patches, upsample to 448x448 for pixel-level comparison
PIXEL_AUROC_EVAL_SHAPE = (448, 448)  # Standard evaluation resolution (same as Anomalib)
PIXEL_AUROC_PATCH_GRID = 28  # sqrt(784 patches)
PIXEL_AUROC_CONTAMINATION_RATES = [0.0, 0.3]
PIXEL_AUROC_METHODS = [  # Only 3 essential, enough for localization table
    "none",                    # dirty baseline
    "ensemble_loo_mahal",      # best method
    "oracle_patch",            # upper bound
]

assert CONTAMINATION_RATES[0] == 0.0, "CONTAMINATION_RATES must start with 0.0 (clean baseline)"
assert PURIFICATION_METHODS[0] == "none", "PURIFICATION_METHODS must start with 'none'"


# =============================================================================
# CACHED FEATURE LOADING
# =============================================================================

def safe_name(dataset_name: str) -> str:
    """Convert 'mvtec_AD/bottle' -> 'mvtec_AD__bottle'"""
    return dataset_name.replace("/", "__")


def load_cached_dataset(dataset_name: str) -> dict:
    """Load pre-cached features from disk.

    Returns dict with keys:
        all_train_features: (n_all_train, 784, 1024)
        all_train_paths: list[str]
        test_features: (n_test, 784, 1024)
        test_paths: list[str]
        test_labels: (n_test,) int array
    """
    ds_cache = CACHE_DIR / safe_name(dataset_name)

    if not ds_cache.exists():
        raise FileNotFoundError(
            f"Feature cache not found for {dataset_name}. "
            f"Run precache_features.py first. Expected: {ds_cache}"
        )

    data = {}

    # Train
    train_feat_path = ds_cache / "all_train_features.npy"
    train_paths_path = ds_cache / "all_train_paths.txt"
    if train_feat_path.exists():
        data["all_train_features"] = np.load(str(train_feat_path))
        with open(train_paths_path, "r", encoding="utf-8") as f:
            data["all_train_paths"] = [line.strip() for line in f if line.strip()]
    else:
        raise FileNotFoundError(f"Train features not cached: {train_feat_path}")

    # Test
    test_feat_path = ds_cache / "test_features.npy"
    test_labels_path = ds_cache / "test_labels.npy"
    test_paths_path = ds_cache / "test_paths.txt"
    if test_feat_path.exists():
        data["test_features"] = np.load(str(test_feat_path))
        data["test_labels"] = np.load(str(test_labels_path))
        with open(test_paths_path, "r", encoding="utf-8") as f:
            data["test_paths"] = [line.strip() for line in f if line.strip()]
    else:
        raise FileNotFoundError(f"Test features not cached: {test_feat_path}")

    return data


def prepare_experiment_data(
    cached_data: dict,
    train_limit: int,
    seed: int,
) -> tuple[np.ndarray, list[int], np.ndarray, np.ndarray, np.ndarray]:
    """From cached data, prepare train/test split for one experiment.

    Returns:
        train_features: (train_limit, n_patches, dim) - subsampled normal train features
        anomaly_indices_in_test: list of indices into test_features where label==1
        test_features: (n_test, n_patches, dim) - full test set
        test_labels: (n_test,) - binary labels
        all_train_features: (n_all_train, n_patches, dim) - full train set (for anomaly feature lookup)
    """
    all_train_features = cached_data["all_train_features"]
    test_features = cached_data["test_features"]
    test_labels = cached_data["test_labels"]

    n_all_train = all_train_features.shape[0]

    # Subsample train to train_limit
    rng = np.random.RandomState(seed)
    if n_all_train > train_limit:
        indices = rng.choice(n_all_train, train_limit, replace=False)
        train_features = all_train_features[sorted(indices)]
    else:
        train_features = all_train_features  # No copy needed; contaminate_train_features copies when needed

    # Anomaly indices in test set (for contamination source)
    anomaly_indices = np.where(test_labels == 1)[0].tolist()

    return train_features, anomaly_indices, test_features, test_labels, all_train_features


def _kneighbors(knn: NearestNeighbors, x: np.ndarray, n_jobs: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
    """Compat wrapper: sklearn has `n_jobs` in different places depending on version."""
    try:
        return knn.kneighbors(x, n_jobs=n_jobs)
    except TypeError:
        return knn.kneighbors(x)


def _faiss_knn_l2(bank: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Fast exact L2 kNN using FAISS. Returns euclidean distances (n_queries, k)."""
    bank_f32 = np.ascontiguousarray(bank, dtype=np.float32)
    queries_f32 = np.ascontiguousarray(queries, dtype=np.float32)
    index = faiss.IndexFlatL2(bank_f32.shape[1])
    index.add(bank_f32)
    sq_dists, _ = index.search(queries_f32, k)
    # FAISS returns squared L2 distances; convert to euclidean
    return np.sqrt(np.maximum(0, sq_dists))


def _faiss_knn_cosine(bank: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Fast exact cosine kNN using FAISS (via inner product on L2-normalized vectors).

    Returns cosine distances (1 - cosine_similarity) with shape (n_queries, k).
    """
    # IMPORTANT: copy before normalize_L2, it mutates in-place
    bank_f32 = np.ascontiguousarray(bank, dtype=np.float32).copy()
    queries_f32 = np.ascontiguousarray(queries, dtype=np.float32).copy()
    # L2-normalize for cosine similarity via inner product
    faiss.normalize_L2(bank_f32)
    faiss.normalize_L2(queries_f32)
    index = faiss.IndexFlatIP(bank_f32.shape[1])
    index.add(bank_f32)
    similarities, _ = index.search(queries_f32, k)
    # cosine_distance = 1 - cosine_similarity
    return np.maximum(0, 1.0 - similarities)


# =============================================================================
# CONTAMINATION
# =============================================================================

def contaminate_train_features(
    train_features: np.ndarray,
    anomaly_features_pool: np.ndarray,
    contamination_rate: float,
    seed: int,
) -> tuple[np.ndarray, list[int]]:
    """Replace some normal train features with anomaly features.

    Args:
        train_features: (n_train, n_patches, dim) - clean normal features
        anomaly_features_pool: (n_anomalies, n_patches, dim) - anomaly features to pick from
        contamination_rate: fraction of train to replace
        seed: random seed

    Returns: (contaminated_features, is_contaminated_flags)
    """
    n_total = train_features.shape[0]
    n_contaminate = int(round(n_total * contamination_rate))

    if n_contaminate == 0:
        return train_features.copy(), [0] * n_total

    rng = np.random.RandomState(seed + 2000)

    # Select which normal positions to replace
    replace_idx = rng.choice(n_total, n_contaminate, replace=False)

    # Select which anomalies to insert
    n_anomalies = anomaly_features_pool.shape[0]
    if n_anomalies < n_contaminate:
        anomaly_selection = rng.choice(n_anomalies, n_contaminate, replace=True)
    else:
        anomaly_selection = rng.choice(n_anomalies, n_contaminate, replace=False)

    contaminated = train_features.copy()
    is_contaminated = [0] * n_total
    contaminated[replace_idx] = anomaly_features_pool[anomaly_selection]
    for idx in replace_idx:
        is_contaminated[idx] = 1

    return contaminated, is_contaminated


# =============================================================================
# PURIFICATION: LOO PATCH CONSISTENCY
# =============================================================================

def compute_loo_consistency(
    features: np.ndarray,
    k: int = DEFAULT_KNN_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute leave-one-image-out consistency scores.

    features: (n_images, n_patches, dim)
    Returns:
    - patch_scores: (n_images, n_patches) inconsistency per patch
    - image_scores: (n_images,) mean inconsistency per image
    """
    n_images, n_patches, dim = features.shape
    if n_images < 2:
        raise ValueError(f"LOO consistency requires at least 2 images, got {n_images}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    patch_scores = np.zeros((n_images, n_patches))

    for i in range(n_images):
        # Bank: all patches from other images
        mask = np.ones(n_images, dtype=bool)
        mask[i] = False
        bank = features[mask].reshape(-1, dim)

        if bank.shape[0] < k:
            raise ValueError(
                f"LOO bank too small for kNN (bank_patches={bank.shape[0]} < k={k}). "
                f"Try smaller k or larger train_limit."
            )

        # Distance of each patch in image i to its nearest neighbors in the bank
        patches = features[i]  # (n_patches, dim)
        distances = _faiss_knn_l2(bank, patches, k)
        patch_scores[i] = distances.mean(axis=1)  # Mean of k distances

    image_scores = patch_scores.mean(axis=1)  # Mean inconsistency per image

    return patch_scores, image_scores


def purify_loo_patch(
    features: np.ndarray,
    patch_scores: np.ndarray,
    percentile_threshold: float = DEFAULT_PERCENTILE_THRESHOLD,
) -> tuple[np.ndarray, int]:
    """Remove most inconsistent patches by excluding them from the bank.

    Returns flat bank (n_remaining_patches, dim) instead of (n_images, n_patches, dim).
    This way removed patches simply don't exist in the memory bank.

    Returns: (purified_bank_flat, n_patches_removed)
    """
    threshold = np.percentile(patch_scores, percentile_threshold)
    remove_mask = patch_scores > threshold  # True = inconsistent = remove

    n_removed = int(remove_mask.sum())

    # Flatten to (n_total_patches, dim) and keep only good patches
    n_images, n_patches, dim = features.shape
    flat_features = features.reshape(-1, dim)
    flat_mask = ~remove_mask.ravel()
    purified_bank = flat_features[flat_mask]  # (n_remaining, dim)

    return purified_bank, n_removed


# =============================================================================
# PURIFICATION: BASELINES AND VARIANTS
# =============================================================================

# Adaptive purification: combined detector using skewness + gap ratio
# Only purify when BOTH conditions suggest contamination, reducing false positives
# Skewness: right-skewed distribution suggests outlier images
# Gap ratio: large gap between max and median score suggests outliers
ADAPTIVE_SKEWNESS_THRESHOLD = 0.5
ADAPTIVE_GAP_RATIO_THRESHOLD = 0.3  # (max - median) / median


def detect_contamination(image_scores: np.ndarray) -> tuple[bool, dict]:
    """Detect if the support set is likely contaminated using LOO image score distribution.

    Uses two signals:
    1. Skewness > threshold: right-skewed distribution suggests outliers
    2. Gap ratio > threshold: (max - median) / median measures how extreme the max is

    Purification activates when BOTH conditions are met (AND logic, conservative).

    Returns: (is_contaminated, diagnostics_dict)
    """
    diagnostics = {"skewness": 0.0, "gap_ratio": 0.0, "triggered_by": "none"}

    if len(image_scores) < 3:
        return False, diagnostics

    s = float(skew(image_scores))
    median_score = float(np.median(image_scores))
    max_score = float(np.max(image_scores))

    gap_ratio = (max_score - median_score) / median_score if median_score > 1e-8 else 0.0

    diagnostics["skewness"] = s
    diagnostics["gap_ratio"] = gap_ratio

    skewness_triggered = s > ADAPTIVE_SKEWNESS_THRESHOLD
    gap_triggered = gap_ratio > ADAPTIVE_GAP_RATIO_THRESHOLD

    # Use AND logic: both conditions must be met to reduce false positives
    # With few samples (10 images), either signal alone has high FP rate
    if skewness_triggered and gap_triggered:
        diagnostics["triggered_by"] = "both"
        return True, diagnostics
    else:
        diagnostics["triggered_by"] = "none"
        return False, diagnostics


def purify_adaptive_loo_patch(
    features: np.ndarray,
    patch_scores: np.ndarray,
    image_scores: np.ndarray,
    percentile_threshold: float = DEFAULT_PERCENTILE_THRESHOLD,
) -> tuple[np.ndarray, int, bool, dict]:
    """LOO patch purification, but only if contamination is detected.

    Returns: (bank_features, n_patches_removed, did_purify, diagnostics)
    - If contamination detected: purifies like loo_patch
    - If no contamination: returns flat bank with 0 patches removed
    """
    contamination_detected, diagnostics = detect_contamination(image_scores)

    if contamination_detected:
        bank, n_removed = purify_loo_patch(features, patch_scores, percentile_threshold)
        return bank, n_removed, True, diagnostics
    else:
        # No purification: return flat bank as-is
        n_images, n_patches, dim = features.shape
        flat_bank = features.reshape(-1, dim)
        return flat_bank, 0, False, diagnostics


def purify_random_patch(
    features: np.ndarray,
    percentile_threshold: float = DEFAULT_PERCENTILE_THRESHOLD,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """Remove random patches (ablation control).

    Removes the same fraction as loo_patch would (100 - percentile_threshold)%
    but randomly instead of by LOO score.

    Returns: (purified_bank_flat, n_patches_removed)
    """
    n_images, n_patches, dim = features.shape
    total_patches = n_images * n_patches
    n_to_remove = int(total_patches * (100 - percentile_threshold) / 100.0)

    rng = np.random.RandomState(seed + 5000)
    flat_features = features.reshape(-1, dim)

    if n_to_remove >= total_patches - 1:
        n_to_remove = total_patches - 1  # Keep at least 1 patch

    remove_indices = rng.choice(total_patches, n_to_remove, replace=False)
    keep_mask = np.ones(total_patches, dtype=bool)
    keep_mask[remove_indices] = False

    return flat_features[keep_mask], n_to_remove


def purify_mahalanobis_patch(
    features: np.ndarray,
    percentile_threshold: float = DEFAULT_PERCENTILE_THRESHOLD,
) -> tuple[np.ndarray, int]:
    """Remove patches with highest Mahalanobis distance to centroid.

    Uses Ledoit-Wolf shrinkage for covariance estimation (required when
    dim=1024 >> n_patches, making empirical covariance singular).

    Returns: (purified_bank_flat, n_patches_removed)
    """
    n_images, n_patches, dim = features.shape
    flat_features = features.reshape(-1, dim)

    # Ledoit-Wolf shrinkage covariance
    try:
        lw = LedoitWolf()
        lw.fit(flat_features)
        precision_matrix = lw.get_precision()
        centroid = flat_features.mean(axis=0)

        # Mahalanobis distance for each patch
        diff = flat_features - centroid
        # d_i = sqrt(diff_i @ precision @ diff_i^T)
        # Vectorized: (n, dim) @ (dim, dim) -> (n, dim), then sum
        mahal_scores = np.sqrt(np.maximum(0, np.sum(diff @ precision_matrix * diff, axis=1)))
    except Exception:
        # Fallback to Euclidean if Ledoit-Wolf fails
        centroid = flat_features.mean(axis=0)
        mahal_scores = np.linalg.norm(flat_features - centroid, axis=1)

    threshold = np.percentile(mahal_scores, percentile_threshold)
    keep_mask = mahal_scores <= threshold
    n_removed = int((~keep_mask).sum())

    return flat_features[keep_mask], n_removed


def purify_image_level_loo(
    features: np.ndarray,
    image_scores: np.ndarray,
    contamination_rate: float,
) -> tuple[np.ndarray, int, int]:
    """Remove entire images with highest LOO image scores.

    Removes ceil(n_images * contamination_rate) images or at least 1
    (when cont > 0). This mirrors what an ideal image-level detector would do.

    Returns: (purified_bank_flat, n_patches_removed, n_images_removed)
    """
    n_images, n_patches, dim = features.shape

    # Number of images to remove: same as number of contaminated images
    n_to_remove = int(np.ceil(n_images * contamination_rate))
    if n_to_remove == 0:
        # For cont=0, still remove top 1 to measure clean penalty
        # Actually no: for cont=0, removing is harmful. Only remove if cont>0.
        flat_bank = features.reshape(-1, dim)
        return flat_bank, 0, 0

    # Don't remove more than n_images - 1
    n_to_remove = min(n_to_remove, n_images - 1)

    # Remove images with highest LOO score
    sorted_indices = np.argsort(image_scores)[::-1]  # Descending
    remove_indices = set(sorted_indices[:n_to_remove].tolist())

    keep_mask = np.array([i not in remove_indices for i in range(n_images)])
    kept_features = features[keep_mask]  # (n_kept, n_patches, dim)
    n_patches_removed = n_to_remove * n_patches

    flat_bank = kept_features.reshape(-1, dim)
    return flat_bank, n_patches_removed, n_to_remove


def purify_isolation_forest_patch(
    features: np.ndarray,
    percentile_threshold: float = DEFAULT_PERCENTILE_THRESHOLD,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """Remove patches flagged as outliers by Isolation Forest.

    Uses contamination=(100-percentile_threshold)/100 to match the same
    removal rate as loo_patch.

    Returns: (purified_bank_flat, n_patches_removed)
    """
    n_images, n_patches, dim = features.shape
    flat_features = features.reshape(-1, dim)

    contamination_frac = (100.0 - percentile_threshold) / 100.0
    contamination_frac = max(0.001, min(contamination_frac, 0.5))  # IForest bounds

    clf = IsolationForest(
        contamination=contamination_frac,
        random_state=seed + 3000,
        n_estimators=100,
        n_jobs=-1,
    )
    preds = clf.fit_predict(flat_features)  # 1=inlier, -1=outlier

    keep_mask = preds == 1
    n_removed = int((~keep_mask).sum())

    # Guard: keep at least 1 patch
    if keep_mask.sum() == 0:
        keep_mask[0] = True
        n_removed = flat_features.shape[0] - 1

    return flat_features[keep_mask], n_removed


def purify_global_iforest(
    features: np.ndarray,
    contamination_rate: float,
    seed: int = 0,
) -> tuple[np.ndarray, int, int]:
    """Remove entire images flagged by IsolationForest on global (mean-pooled) features.

    Each image is represented by its mean patch feature (1024-dim).

    Returns: (purified_bank_flat, n_patches_removed, n_images_removed)
    """
    n_images, n_patches, dim = features.shape

    # Global features: mean pool over patches
    global_features = features.mean(axis=1)  # (n_images, dim)

    if contamination_rate <= 0 or n_images < 3:
        flat_bank = features.reshape(-1, dim)
        return flat_bank, 0, 0

    contamination_frac = max(0.001, min(contamination_rate, 0.5))

    clf = IsolationForest(
        contamination=contamination_frac,
        random_state=seed + 4000,
        n_estimators=100,
        n_jobs=-1,
    )
    preds = clf.fit_predict(global_features)  # 1=inlier, -1=outlier

    keep_mask = preds == 1

    # Guard: keep at least 1 image
    if keep_mask.sum() == 0:
        keep_mask[0] = True

    n_images_removed = int((~keep_mask).sum())
    kept_features = features[keep_mask]
    n_patches_removed = n_images_removed * n_patches

    flat_bank = kept_features.reshape(-1, dim)
    return flat_bank, n_patches_removed, n_images_removed


def purify_oracle_patch(
    features: np.ndarray,
    is_contaminated: list[int],
) -> tuple[np.ndarray, int]:
    """Remove ALL patches from contaminated images (perfect upper bound).

    Returns: (purified_bank_flat, n_patches_removed)
    """
    n_images, n_patches, dim = features.shape

    keep_mask = np.array([not flag for flag in is_contaminated])

    if keep_mask.sum() == 0:
        # All contaminated, keep at least something
        flat_bank = features.reshape(-1, dim)
        return flat_bank, 0

    kept_features = features[keep_mask]
    n_removed = int((~keep_mask).sum()) * n_patches

    flat_bank = kept_features.reshape(-1, dim)
    return flat_bank, n_removed


# =============================================================================
# PURIFICATION: ADDITIONAL METHODS
# =============================================================================

def purify_ensemble_loo_mahal(
    features: np.ndarray,
    patch_scores_loo: np.ndarray,
    percentile_threshold: float = DEFAULT_PERCENTILE_THRESHOLD,
    alpha: float = 0.5,
) -> tuple[np.ndarray, int]:
    """Ensemble LOO + Mahalanobis patch scoring.

    Combines LOO consistency (local inconsistency) with Mahalanobis distance
    (global distributional outlier). Each is min-max normalized to [0,1] before
    combining with weight alpha.

    score_i = alpha * norm(LOO_i) + (1 - alpha) * norm(Mahal_i)

    Returns: (purified_bank_flat, n_patches_removed)
    """
    n_images, n_patches, dim = features.shape
    flat_features = features.reshape(-1, dim)

    # LOO scores (already computed, flatten)
    loo_flat = patch_scores_loo.ravel()

    # Mahalanobis scores
    try:
        lw = LedoitWolf()
        lw.fit(flat_features)
        precision_matrix = lw.get_precision()
        centroid = flat_features.mean(axis=0)
        diff = flat_features - centroid
        mahal_flat = np.sqrt(np.maximum(0, np.sum(diff @ precision_matrix * diff, axis=1)))
    except Exception:
        centroid = flat_features.mean(axis=0)
        mahal_flat = np.linalg.norm(flat_features - centroid, axis=1)

    # Normalize both to [0, 1]
    def _minmax(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    loo_norm = _minmax(loo_flat)
    mahal_norm = _minmax(mahal_flat)

    # Combine
    ensemble_scores = alpha * loo_norm + (1 - alpha) * mahal_norm

    threshold = np.percentile(ensemble_scores, percentile_threshold)
    keep_mask = ensemble_scores <= threshold
    n_removed = int((~keep_mask).sum())

    return flat_features[keep_mask], n_removed


def purify_iterative_loo(
    features: np.ndarray,
    k: int = DEFAULT_KNN_K,
    percentile_threshold: float = DEFAULT_PERCENTILE_THRESHOLD,
    max_rounds: int = 3,
) -> tuple[np.ndarray, int]:
    """Multi-round LOO purification.

    Round 1: Remove worst 5% patches by LOO score.
    Round 2+: Recompute LOO on remaining, remove worst again.
    Each round removes patches relative to the CURRENT bank size.

    After removing the most obvious outliers in round 1, subtler contaminants
    become detectable in round 2+ because they no longer hide behind the
    removed extreme outliers.

    Returns: (purified_bank_flat, total_n_patches_removed)
    """
    n_images, n_patches, dim = features.shape
    total_patches = n_images * n_patches

    # Work with flat features + track which patches remain
    flat_features = features.reshape(-1, dim)
    remaining_mask = np.ones(total_patches, dtype=bool)

    # For LOO we need image structure. Keep track of image assignments.
    # patch j belongs to image j // n_patches
    image_ids = np.repeat(np.arange(n_images), n_patches)

    total_removed = 0
    frac_to_remove = (100.0 - percentile_threshold) / 100.0

    for round_i in range(max_rounds):
        # Get current remaining patches and their image IDs
        current_indices = np.where(remaining_mask)[0]
        current_features = flat_features[current_indices]
        current_image_ids = image_ids[current_indices]

        # Need at least 2 images with patches
        unique_images = np.unique(current_image_ids)
        if len(unique_images) < 2:
            break

        n_current = len(current_indices)
        n_to_remove = max(1, int(n_current * frac_to_remove))

        # Compute LOO-like scores: for each patch, distance to nearest neighbors
        # excluding patches from the SAME image
        scores = np.zeros(n_current)
        for img_id in unique_images:
            img_mask = current_image_ids == img_id
            other_mask = ~img_mask
            if other_mask.sum() < k:
                continue
            bank = current_features[other_mask]
            queries = current_features[img_mask]
            knn = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='brute')
            knn.fit(bank)
            distances, _ = _kneighbors(knn, queries, DEFAULT_N_JOBS)
            scores[img_mask] = distances.mean(axis=1)

        # Find top outliers
        if n_to_remove >= n_current - k:
            break  # Can't remove this many

        # Get indices of worst patches (highest scores)
        worst_idx = np.argsort(scores)[-n_to_remove:]
        global_worst = current_indices[worst_idx]
        remaining_mask[global_worst] = False
        total_removed += n_to_remove

    purified_bank = flat_features[remaining_mask]
    return purified_bank, total_removed


def compute_loo_consistency_cosine(
    features: np.ndarray,
    k: int = DEFAULT_KNN_K,
) -> tuple[np.ndarray, np.ndarray]:
    """LOO consistency using cosine distance instead of euclidean.

    DINOv2/v3 features live on a high-dimensional hypersphere where cosine
    distance may separate anomalous patches better than euclidean.

    Returns: (patch_scores, image_scores) same shape as compute_loo_consistency
    """
    n_images, n_patches, dim = features.shape
    patch_scores = np.zeros((n_images, n_patches))

    for i in range(n_images):
        mask = np.ones(n_images, dtype=bool)
        mask[i] = False
        bank = features[mask].reshape(-1, dim)

        if bank.shape[0] < k:
            continue

        distances = _faiss_knn_cosine(bank, features[i], k)
        patch_scores[i] = distances.mean(axis=1)

    image_scores = patch_scores.mean(axis=1)
    return patch_scores, image_scores


def purify_cosine_loo(
    features: np.ndarray,
    k: int = DEFAULT_KNN_K,
    percentile_threshold: float = DEFAULT_PERCENTILE_THRESHOLD,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """LOO purification using cosine distance.

    Returns: (purified_bank_flat, n_patches_removed, cosine_patch_scores, cosine_image_scores)
    """
    patch_scores, image_scores = compute_loo_consistency_cosine(features, k)
    bank, n_removed = purify_loo_patch(features, patch_scores, percentile_threshold)
    return bank, n_removed, patch_scores, image_scores


def purify_lof_patch(
    features: np.ndarray,
    percentile_threshold: float = DEFAULT_PERCENTILE_THRESHOLD,
) -> tuple[np.ndarray, int]:
    """Remove patches with highest Local Outlier Factor score.

    LOF measures local density deviation: a patch surrounded by dense neighbors
    but itself in a sparse region gets a high LOF score. This captures different
    outlier structure than Mahalanobis (which is purely distributional).

    Returns: (purified_bank_flat, n_patches_removed)
    """
    n_images, n_patches, dim = features.shape
    flat_features = features.reshape(-1, dim)

    n_neighbors = min(20, flat_features.shape[0] - 1)
    if n_neighbors < 2:
        return flat_features, 0

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False, n_jobs=-1)
    lof.fit(flat_features)
    # negative_outlier_factor_: more negative = more outlier
    lof_scores = -lof.negative_outlier_factor_  # Higher = more outlier

    threshold = np.percentile(lof_scores, percentile_threshold)
    keep_mask = lof_scores <= threshold
    n_removed = int((~keep_mask).sum())

    if keep_mask.sum() == 0:
        keep_mask[0] = True
        n_removed = flat_features.shape[0] - 1

    return flat_features[keep_mask], n_removed


def purify_pca_patch(
    features: np.ndarray,
    percentile_threshold: float = DEFAULT_PERCENTILE_THRESHOLD,
    n_components: float = 0.95,
) -> tuple[np.ndarray, int]:
    """Remove patches with highest PCA reconstruction error.

    Patches that cannot be well-reconstructed by the principal components of
    the bank are likely outliers (anomalous patches from contaminated images).

    Returns: (purified_bank_flat, n_patches_removed)
    """
    n_images, n_patches, dim = features.shape
    flat_features = features.reshape(-1, dim)

    # Limit n_components to avoid issues with small sample sizes
    max_components = min(flat_features.shape[0], flat_features.shape[1]) - 1
    if max_components < 2:
        return flat_features, 0

    try:
        pca = PCA(n_components=n_components, svd_solver='auto')
        transformed = pca.fit_transform(flat_features)
        reconstructed = pca.inverse_transform(transformed)
        recon_errors = np.mean((flat_features - reconstructed) ** 2, axis=1)
    except Exception:
        return flat_features, 0

    threshold = np.percentile(recon_errors, percentile_threshold)
    keep_mask = recon_errors <= threshold
    n_removed = int((~keep_mask).sum())

    if keep_mask.sum() == 0:
        keep_mask[0] = True
        n_removed = flat_features.shape[0] - 1

    return flat_features[keep_mask], n_removed


def detect_contamination_grubbs(image_scores: np.ndarray, alpha: float = 0.05) -> tuple[bool, dict]:
    """Detect contamination using Grubbs' test for outliers.

    Grubbs' test has proper statistical foundation for detecting a single outlier
    in a sample, using the t-distribution critical value. More principled than
    ad-hoc skewness thresholds.

    Tests the maximum value as potential outlier. If significant, contamination
    is detected and purification activates.

    Args:
        image_scores: (n_images,) LOO image scores
        alpha: significance level (default 0.05)

    Returns: (is_contaminated, diagnostics_dict)
    """
    diagnostics = {"grubbs_G": 0.0, "grubbs_G_crit": 0.0, "triggered_by": "none"}
    n = len(image_scores)

    if n < 3:
        return False, diagnostics

    mean_s = float(np.mean(image_scores))
    std_s = float(np.std(image_scores, ddof=1))

    if std_s < 1e-10:
        return False, diagnostics

    max_s = float(np.max(image_scores))
    G = (max_s - mean_s) / std_s

    # Critical value from t-distribution
    t_crit = t_dist.ppf(1 - alpha / (2 * n), n - 2)
    G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))

    diagnostics["grubbs_G"] = G
    diagnostics["grubbs_G_crit"] = G_crit

    if G > G_crit:
        diagnostics["triggered_by"] = "grubbs"
        return True, diagnostics
    else:
        diagnostics["triggered_by"] = "none"
        return False, diagnostics


def purify_grubbs_adaptive_loo(
    features: np.ndarray,
    patch_scores: np.ndarray,
    image_scores: np.ndarray,
    percentile_threshold: float = DEFAULT_PERCENTILE_THRESHOLD,
    alpha: float = 0.05,
) -> tuple[np.ndarray, int, bool, dict]:
    """LOO patch purification gated by Grubbs' statistical test.

    Only purifies if Grubbs' test detects a significant outlier in LOO image
    scores. More principled than skewness + gap_ratio thresholds.

    Returns: (bank_features, n_patches_removed, did_purify, diagnostics)
    """
    contamination_detected, diagnostics = detect_contamination_grubbs(image_scores, alpha)

    if contamination_detected:
        bank, n_removed = purify_loo_patch(features, patch_scores, percentile_threshold)
        return bank, n_removed, True, diagnostics
    else:
        n_images, n_patches, dim = features.shape
        flat_bank = features.reshape(-1, dim)
        return flat_bank, 0, False, diagnostics


def detect_contamination_gmm_bic(image_scores: np.ndarray, bic_threshold: float = 0.0) -> tuple[bool, dict]:
    """Detect contamination using GMM 1D + BIC model selection.

    Fits GMM with k=1 and k=2 components on LOO image scores.
    If BIC improves with 2 components (BIC_2 < BIC_1 - threshold) AND the
    second component has a clearly higher mean, contamination is detected.

    More robust than skewness/gap for small n (~10 images).

    Args:
        image_scores: (n_images,) LOO image scores
        bic_threshold: minimum BIC improvement to prefer k=2 (default 0.0 = any improvement)

    Returns: (is_contaminated, diagnostics_dict)
    """
    diagnostics = {"bic_1": 0.0, "bic_2": 0.0, "bic_delta": 0.0,
                   "gmm_weight_high": 0.0, "triggered_by": "none"}
    n = len(image_scores)

    if n < 4:  # Need at least 4 samples for meaningful GMM(k=2)
        return False, diagnostics

    scores_2d = image_scores.reshape(-1, 1)

    try:
        gmm1 = GaussianMixture(n_components=1, random_state=42)
        gmm1.fit(scores_2d)
        bic1 = gmm1.bic(scores_2d)

        gmm2 = GaussianMixture(n_components=2, random_state=42)
        gmm2.fit(scores_2d)
        bic2 = gmm2.bic(scores_2d)
    except Exception:
        return False, diagnostics

    diagnostics["bic_1"] = float(bic1)
    diagnostics["bic_2"] = float(bic2)
    diagnostics["bic_delta"] = float(bic1 - bic2)  # Positive means k=2 is better

    # Check if k=2 is significantly better
    if bic2 < bic1 - bic_threshold:
        # Identify the "contaminated" component (higher mean)
        means = gmm2.means_.ravel()
        weights = gmm2.weights_.ravel()
        high_idx = np.argmax(means)
        weight_high = weights[high_idx]
        diagnostics["gmm_weight_high"] = float(weight_high)

        # Only trigger if the high-mean component has reasonable weight (not too small or too large)
        # Too small (<5%) = noise, too large (>60%) = not contamination, just bimodal
        if 0.05 <= weight_high <= 0.60:
            diagnostics["triggered_by"] = "gmm_bic"
            return True, diagnostics

    return False, diagnostics


def purify_gmm_adaptive_loo(
    features: np.ndarray,
    patch_scores: np.ndarray,
    image_scores: np.ndarray,
    percentile_threshold: float = DEFAULT_PERCENTILE_THRESHOLD,
) -> tuple[np.ndarray, int, bool, dict]:
    """LOO patch purification gated by GMM+BIC contamination detector.

    Only purifies if GMM model selection detects bimodality in LOO image scores.
    More adaptive than Grubbs or skewness tests.

    Returns: (bank_features, n_patches_removed, did_purify, diagnostics)
    """
    contamination_detected, diagnostics = detect_contamination_gmm_bic(image_scores)

    if contamination_detected:
        bank, n_removed = purify_loo_patch(features, patch_scores, percentile_threshold)
        return bank, n_removed, True, diagnostics
    else:
        n_images, n_patches, dim = features.shape
        flat_bank = features.reshape(-1, dim)
        return flat_bank, 0, False, diagnostics


def detect_contamination_mad(image_scores: np.ndarray, z_threshold: float = 3.5) -> tuple[bool, dict]:
    """Detect contamination using MAD (Median Absolute Deviation) z-score.

    Robust outlier detection: z_i = (s_i - median(s)) / MAD(s).
    Activates if max(z) > z_threshold. Very conservative → near-zero FP on clean data.

    Uses scipy.stats.median_abs_deviation with scale='normal' for consistency
    with standard normal distribution (MAD * 1.4826).

    Args:
        image_scores: (n_images,) LOO image scores
        z_threshold: activation threshold for max z-score (default 3.5)

    Returns: (is_contaminated, diagnostics_dict)
    """
    diagnostics = {"mad_max_z": 0.0, "mad_threshold": z_threshold, "triggered_by": "none"}
    n = len(image_scores)

    if n < 3:
        return False, diagnostics

    mad = median_abs_deviation(image_scores, scale='normal')

    if mad < 1e-10:
        return False, diagnostics

    med = float(np.median(image_scores))
    z_scores = (image_scores - med) / mad
    max_z = float(np.max(z_scores))

    diagnostics["mad_max_z"] = max_z

    if max_z > z_threshold:
        diagnostics["triggered_by"] = "mad"
        return True, diagnostics

    return False, diagnostics


def purify_mad_adaptive_loo(
    features: np.ndarray,
    patch_scores: np.ndarray,
    image_scores: np.ndarray,
    percentile_threshold: float = DEFAULT_PERCENTILE_THRESHOLD,
    z_threshold: float = 3.5,
) -> tuple[np.ndarray, int, bool, dict]:
    """LOO patch purification gated by MAD z-score contamination detector.

    Only purifies if MAD-based z-score detects significant outlier in LOO image scores.
    Very conservative → minimal clean penalty.

    Returns: (bank_features, n_patches_removed, did_purify, diagnostics)
    """
    contamination_detected, diagnostics = detect_contamination_mad(image_scores, z_threshold)

    if contamination_detected:
        bank, n_removed = purify_loo_patch(features, patch_scores, percentile_threshold)
        return bank, n_removed, True, diagnostics
    else:
        n_images, n_patches, dim = features.shape
        flat_bank = features.reshape(-1, dim)
        return flat_bank, 0, False, diagnostics


# =============================================================================
# SCORING
# =============================================================================

def compute_knn_scores(
    train_features: np.ndarray,
    test_features: np.ndarray,
    k: int = DEFAULT_KNN_K,
    chunk_size: int = 200,
) -> np.ndarray:
    """Compute image-level KNN anomaly scores (vectorized, chunked for memory).

    train_features: (n_train, n_patches, dim) OR (n_patches_flat, dim) for purified
    test_features: (n_test, n_patches, dim)
    chunk_size: process this many test images at a time (controls RAM usage)
    Returns: (n_test,) image-level scores
    """
    if train_features.ndim == 3:
        dim = train_features.shape[2]
        bank = train_features.reshape(-1, dim)
    else:
        bank = train_features
        dim = train_features.shape[-1]

    n_test = test_features.shape[0]
    n_patches = test_features.shape[1]

    if bank.shape[0] < k:
        raise ValueError(f"Bank too small for kNN (bank_patches={bank.shape[0]} < k={k})")

    # Build FAISS index once, query in chunks
    bank_f32 = np.ascontiguousarray(bank, dtype=np.float32)
    index = faiss.IndexFlatL2(dim)
    index.add(bank_f32)

    # Pre-reshape test features to 2D for efficient chunked queries
    test_flat = test_features.reshape(-1, dim)  # (n_test * n_patches, dim)
    # Index for the 95% order statistic (discrete quantile, "higher" style)
    # This is a speed optimization vs np.percentile; note it may differ slightly from interpolated percentiles.
    pct_idx = int(np.ceil(0.95 * n_patches)) - 1
    pct_idx = max(0, min(pct_idx, n_patches - 1))

    scores = np.zeros(n_test)
    for start in range(0, n_test, chunk_size):
        end = min(start + chunk_size, n_test)
        chunk = np.ascontiguousarray(test_flat[start * n_patches : end * n_patches], dtype=np.float32)
        sq_dists, _ = index.search(chunk, k)
        distances = np.sqrt(np.maximum(0, sq_dists))
        patch_scores = distances.mean(axis=1).reshape(end - start, n_patches)
        # Use np.partition (O(n)) instead of np.percentile (O(n log n))
        partitioned = np.partition(patch_scores, pct_idx, axis=1)
        scores[start:end] = partitioned[:, pct_idx]

    return scores


def compute_knn_scores_with_maps(
    train_features: np.ndarray,
    test_features: np.ndarray,
    k: int = DEFAULT_KNN_K,
    chunk_size: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Like compute_knn_scores but also returns pixel-level anomaly maps.

    Returns:
        scores: (n_test,) image-level scores (percentile 95 of patch scores)
        patch_score_maps: (n_test, n_patches) per-patch anomaly scores
    """
    if train_features.ndim == 3:
        dim = train_features.shape[2]
        bank = train_features.reshape(-1, dim)
    else:
        bank = train_features
        dim = train_features.shape[-1]

    n_test = test_features.shape[0]
    n_patches = test_features.shape[1]

    if bank.shape[0] < k:
        raise ValueError(f"Bank too small for kNN (bank_patches={bank.shape[0]} < k={k})")

    bank_f32 = np.ascontiguousarray(bank, dtype=np.float32)
    index = faiss.IndexFlatL2(dim)
    index.add(bank_f32)

    test_flat = test_features.reshape(-1, dim)
    pct_idx = int(np.ceil(0.95 * n_patches)) - 1
    pct_idx = max(0, min(pct_idx, n_patches - 1))

    scores = np.zeros(n_test)
    all_patch_scores = np.zeros((n_test, n_patches), dtype=np.float32)

    for start in range(0, n_test, chunk_size):
        end = min(start + chunk_size, n_test)
        chunk = np.ascontiguousarray(test_flat[start * n_patches : end * n_patches], dtype=np.float32)
        sq_dists, _ = index.search(chunk, k)
        distances = np.sqrt(np.maximum(0, sq_dists))
        patch_scores = distances.mean(axis=1).reshape(end - start, n_patches)
        all_patch_scores[start:end] = patch_scores
        partitioned = np.partition(patch_scores, pct_idx, axis=1)
        scores[start:end] = partitioned[:, pct_idx]

    return scores, all_patch_scores


# =============================================================================
# PIXEL-LEVEL METRICS (mask loading + pixel AUROC)
# =============================================================================

def resolve_mask_path(test_path: str, dataset_name: str) -> dict:
    """Given a test image path (relative), resolve its GT mask path.

    Returns dict with keys: mask_path (str|None), mask_dir (str|None for LOCO multi-mask).
    Follows same logic as galad_train_all.py load_mvtec_dataset/load_loco_dataset/load_visa_dataset.
    """
    # test_path is relative like "data\\mvtec_AD\\bottle\\test\\broken_large\\000.png"
    p = Path(test_path)
    parts = p.parts  # ('data', 'mvtec_AD', 'bottle', 'test', 'broken_large', '000.png')

    # Check if this is a normal image (good category) → no mask
    # For MVTec/LOCO/BTAD: category is parts[-2] (folder name)
    # For VisA: "Normal" vs "Anomaly" in path
    if "VisA" in dataset_name:
        if "Normal" in test_path:
            return {"mask_path": None, "mask_dir": None}
        # VisA anomaly: data/VisA/pcb1/Data/Images/Anomaly/000.JPG
        # Mask:         data/VisA/pcb1/Data/Masks/Anomaly/000.png
        mask_path = Path(test_path.replace("Images", "Masks")).with_suffix(".png")
        if mask_path.exists():
            return {"mask_path": str(mask_path), "mask_dir": None}
        return {"mask_path": None, "mask_dir": None}

    # MVTec AD / BTAD / LOCO: check if category == "good"
    category = parts[-2]  # e.g., "broken_large" or "good" or "defect"
    if category == "good":
        return {"mask_path": None, "mask_dir": None}

    # Derive data_root: path up to the dataset folder
    # For MVTec: data/mvtec_AD/bottle/test/broken_large/000.png → data_root = data/mvtec_AD/bottle
    # test is at parts[-3], so data_root is up to parts[-4]
    test_idx = None
    for i, part in enumerate(parts):
        if part == "test":
            test_idx = i
            break
    if test_idx is None:
        return {"mask_path": None, "mask_dir": None}

    data_root = Path(*parts[:test_idx])
    gt_dir = data_root / "ground_truth"
    stem = p.stem

    if "mvtec_loco_AD" in dataset_name:
        # LOCO: ground_truth/<category>/<stem>/ contains multiple PNGs
        mask_subdir = gt_dir / category / stem
        if mask_subdir.exists():
            masks = list(mask_subdir.glob("*.png"))
            if masks:
                return {"mask_path": str(masks[0]), "mask_dir": str(mask_subdir)}
        return {"mask_path": None, "mask_dir": None}

    # MVTec AD / BTAD: ground_truth/<category>/<stem>.png or <stem>_mask.png
    if gt_dir.exists():
        mask_dir = gt_dir / category
        if mask_dir.exists():
            for ext in [".png", "_mask.png"]:
                mp = mask_dir / f"{stem}{ext}"
                if mp.exists():
                    return {"mask_path": str(mp), "mask_dir": None}

    return {"mask_path": None, "mask_dir": None}


def load_mask(mask_info: dict) -> np.ndarray | None:
    """Load GT mask as numpy array. Handles LOCO multi-mask (combine with np.maximum).

    Follows same logic as galad_train_all.py load_mask().
    Returns: (H, W) uint8 array with values 0/255, or None if no mask.
    """
    if not mask_info.get("mask_path"):
        return None

    if mask_info.get("mask_dir"):
        # LOCO: combine multiple masks
        mask_dir = Path(mask_info["mask_dir"])
        combined = None
        for mp in mask_dir.glob("*.png"):
            m = np.array(Image.open(mp).convert("L"))
            combined = m if combined is None else np.maximum(combined, m)
        return combined

    return np.array(Image.open(mask_info["mask_path"]).convert("L"))


def preload_gt_masks(
    test_paths: list[str],
    test_labels: np.ndarray,
    dataset_name: str,
    eval_shape: tuple = PIXEL_AUROC_EVAL_SHAPE,
) -> tuple[np.ndarray, int]:
    """Pre-load and cache all GT masks for a dataset's test set.

    Returns:
        gt_masks: (n_test, H, W) uint8 binary masks (0 for normals)
        n_with_mask: number of test images that have actual masks
    """
    n_test = len(test_paths)
    gt_masks = np.zeros((n_test, eval_shape[0], eval_shape[1]), dtype=np.uint8)
    n_with_mask = 0

    for i, test_path in enumerate(test_paths):
        mask_info = resolve_mask_path(test_path, dataset_name)
        mask = load_mask(mask_info)
        if mask is not None:
            gt = (mask > 0).astype(np.uint8)
            if gt.shape != eval_shape:
                zy = eval_shape[0] / gt.shape[0]
                zx = eval_shape[1] / gt.shape[1]
                gt = (ndimage_zoom(gt.astype(float), (zy, zx), order=0) > 0.5).astype(np.uint8)
            gt_masks[i] = gt
            n_with_mask += 1

    return gt_masks, n_with_mask


def compute_pixel_auroc(
    patch_score_maps: np.ndarray,
    test_paths: list[str],
    test_labels: np.ndarray,
    dataset_name: str,
    patch_grid: int = PIXEL_AUROC_PATCH_GRID,
    eval_shape: tuple = PIXEL_AUROC_EVAL_SHAPE,
    preloaded_masks: np.ndarray = None,
    preloaded_n_masks: int = None,
) -> dict:
    """Compute pixel-level AUROC from patch score maps and GT masks.

    Args:
        patch_score_maps: (n_test, n_patches) per-patch anomaly scores
        test_paths: list of relative paths for test images
        test_labels: (n_test,) binary labels
        dataset_name: e.g., "mvtec_AD/bottle"
        patch_grid: side of the patch grid (28 for 784 patches)
        eval_shape: (H, W) to upsample to (448, 448)
        preloaded_masks: optional (n_test, H, W) pre-loaded GT masks
        preloaded_n_masks: optional count of images with real masks

    Returns: dict with pix_auroc, pix_aupr (or None if no masks)
    """
    n_test = len(test_paths)
    zoom_factor = eval_shape[0] / patch_grid  # 448 / 28 = 16.0

    # Use preloaded masks if available (huge speedup: avoids re-loading from disk)
    if preloaded_masks is not None:
        n_with_mask = preloaded_n_masks if preloaded_n_masks is not None else int((preloaded_masks.sum(axis=(1, 2)) > 0).sum())
    else:
        # Fallback: load masks on the fly
        preloaded_masks, n_with_mask = preload_gt_masks(test_paths, test_labels, dataset_name, eval_shape)

    if n_with_mask == 0:
        return {"pix_auroc": None, "pix_aupr": None, "n_masks_found": 0}

    # Batch upsample all anomaly maps at once
    am_2d = patch_score_maps.reshape(n_test, patch_grid, patch_grid)
    # Vectorized upsample: (n_test, patch_grid, patch_grid) → (n_test, H, W)
    all_am_up = np.zeros((n_test, eval_shape[0], eval_shape[1]), dtype=np.float32)
    for i in range(n_test):
        all_am_up[i] = ndimage_zoom(am_2d[i], zoom_factor, order=1)

    pix_s = all_am_up.ravel()
    pix_l = preloaded_masks.ravel()

    if len(np.unique(pix_l)) < 2:
        return {"pix_auroc": None, "pix_aupr": None, "n_masks_found": n_with_mask}

    pix_auroc = float(roc_auc_score(pix_l, pix_s))
    pix_aupr = float(average_precision_score(pix_l, pix_s))

    return {"pix_auroc": pix_auroc, "pix_aupr": pix_aupr, "n_masks_found": n_with_mask}


def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Compute standard AD metrics."""
    if len(np.unique(labels)) < 2:
        return {"img_auroc": 0.0, "img_aupr": 0.0, "img_f1max": 0.0, "tpr_fpr1": 0.0, "tpr_fpr5": 0.0}

    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)

    # F1-max (efficient): evaluate thresholds from the PR curve (O(n log n))
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denom = (precision + recall)
    f1_all = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    f1_max = float(np.max(f1_all))

    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    if len(anomaly_scores) > 0 and len(normal_scores) > 0:
        fpr1_thresh = np.percentile(normal_scores, 99)
        fpr5_thresh = np.percentile(normal_scores, 95)
        tpr_fpr1 = float((anomaly_scores >= fpr1_thresh).mean())
        tpr_fpr5 = float((anomaly_scores >= fpr5_thresh).mean())
    else:
        tpr_fpr1 = tpr_fpr5 = 0.0

    return {"img_auroc": auroc, "img_aupr": aupr, "img_f1max": f1_max,
            "tpr_fpr1": tpr_fpr1, "tpr_fpr5": tpr_fpr5}


# =============================================================================
# RESUMABILITY & CSV OUTPUT
# =============================================================================

FIELDNAMES = [
    "timestamp", "dataset", "config", "contamination_rate", "purification_method",
    "train_limit", "n_train_original", "n_train_after_purification",
    "n_contaminated", "n_patches_removed",
    "n_test", "n_normal_test", "n_anomaly_test", "seed",
    "knn_k", "percentile_threshold",
    "img_auroc", "img_aupr", "img_f1max", "tpr_fpr1", "tpr_fpr5",
    "auroc_delta_vs_clean", "recovery_rate",
    "contaminant_detection_auroc",  # AUROC of LOO image scores for detecting contaminated images
    "contaminant_precision_at_1",  # Is the top-1 most suspicious image a contaminant?
    "contaminant_precision_at_k",  # Precision when reviewing exactly k=n_contaminated top images
    "contaminant_recall_at_k",  # Recall when reviewing exactly k=n_contaminated top images
    "adaptive_did_purify",  # 1 if adaptive method decided to purify, 0 if not
    "adaptive_skewness",  # Skewness of LOO image scores
    "inference_time_s",
]


def get_completed_experiments(csv_path: Path) -> set[tuple]:
    """Read CSV and return set of completed experiment keys."""
    completed = set()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                key = (
                    row["dataset"],
                    row["config"],
                    row["contamination_rate"],
                    int(row["seed"]),
                    str(row.get("train_limit", "10")),
                    str(row.get("knn_k", "5")),
                    str(row.get("percentile_threshold", "95")),
                )
                completed.add(key)
    return completed


def save_result(result: dict, csv_path: Path) -> None:
    """Append one result row to CSV."""
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=";")
        if not file_exists:
            writer.writeheader()
        row = {k: result.get(k, "") for k in FIELDNAMES}
        writer.writerow(row)


# =============================================================================
# LOGGING
# =============================================================================

SCRIPT_START_TIME = None


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = ""
    if SCRIPT_START_TIME:
        delta = (datetime.now() - SCRIPT_START_TIME).total_seconds()
        hours = delta / 3600
        elapsed = f" [{hours:.1f}h]"
    line = f"[{ts}]{elapsed} {msg}"
    print(line)
    log_path = OUTPUT_DIR / "log.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_error(dataset: str, config: str, seed: int, exc: Exception) -> None:
    error_log_path = OUTPUT_DIR / "errors.txt"
    with open(error_log_path, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Time: {datetime.now().isoformat()}\n")
        f.write(f"Dataset: {dataset}, Config: {config}, Seed: {seed}\n")
        f.write(f"Error: {type(exc).__name__}: {exc}\n")
        f.write(traceback.format_exc())
        f.write("\n")


# =============================================================================
# CORE EXPERIMENT FUNCTION
# =============================================================================

def run_dataset_experiment(
    dataset_name: str,
    seed: int,
    train_limit: int,
    knn_k: int,
    percentile_threshold: float,
    cached_data: dict,
    completed: set[tuple],
    csv_path: Path,
    contamination_rates: list[float],
    purification_methods: list[str],
    is_debug: bool = False,
) -> list[str]:
    """Run all contamination/purification configs for one dataset+seed+train_limit.

    Uses pre-cached features (no GPU needed).

    Returns: list of error strings
    """
    errors = []

    # Prepare data from cache
    train_features, anomaly_indices, test_features, test_labels, _ = prepare_experiment_data(
        cached_data, train_limit, seed
    )

    n_test = test_features.shape[0]
    n_normal_test = int((test_labels == 0).sum())
    n_anomaly_test = int((test_labels == 1).sum())

    # Get anomaly features pool from test set (for contamination)
    anomaly_features_pool = test_features[anomaly_indices]  # (n_anomalies, n_patches, dim)

    print(f"    n_train={train_features.shape[0]}, anomaly_pool={len(anomaly_indices)}, "
          f"n_test={n_test} (norm={n_normal_test}, anom={n_anomaly_test})")

    if len(anomaly_indices) == 0:
        print(f"    [WARN] No anomalies in test set, cannot contaminate. Skipping.")
        return errors

    # Track clean baseline AUROC and dirty AUROCs for delta/recovery computation
    clean_auroc = None
    dirty_aurocs = {}  # {cont_rate: auroc_without_purification}

    for cont_rate in contamination_rates:
        # Contaminate using features directly (no image loading needed)
        if cont_rate == 0.0:
            contaminated_features = train_features  # No copy needed - contaminate_train_features copies later anyway
            is_contaminated = [0] * train_features.shape[0]
        else:
            contaminated_features, is_contaminated = contaminate_train_features(
                train_features, anomaly_features_pool, cont_rate, seed
            )
        n_contaminated = sum(is_contaminated)
        config_tag = f"cont_{int(cont_rate*100)}pct"

        if is_debug:
            for idx, flag in enumerate(is_contaminated):
                status = "CONTAMINATED" if flag else "clean"
                print(f"      img[{idx}]: {status}")

        # Precompute LOO consistency once per contamination level (shared across loo_patch, adaptive, image_level_loo)
        loo_patch_scores = None
        loo_image_scores = None

        # Cache for "none" baseline scores to avoid redundant KNN scoring
        none_scores_cache = None  # Will hold test scores for the unpurified bank

        for method_name in purification_methods:
            full_config = f"{config_tag}_{method_name}"

            # Check if already done
            key = (dataset_name, full_config, str(cont_rate), seed,
                   str(train_limit), str(knn_k), str(percentile_threshold))
            if key in completed:
                # Still need clean_auroc and dirty_aurocs for downstream rows
                if method_name == "none":
                    all_done = all(
                        (dataset_name, f"{config_tag}_{m}", str(cont_rate), seed,
                         str(train_limit), str(knn_k), str(percentile_threshold)) in completed
                        for m in purification_methods
                    )
                    if all_done:
                        print(f"    {config_tag}: all methods SKIP (already done)")
                        break  # Skip this contamination level entirely
                    # Otherwise, need to compute baseline for remaining methods
                    none_scores_cache = compute_knn_scores(contaminated_features, test_features, k=knn_k)
                    metrics_baseline = compute_metrics(test_labels, none_scores_cache)
                    if cont_rate == 0.0:
                        clean_auroc = metrics_baseline["img_auroc"]
                    dirty_aurocs[cont_rate] = metrics_baseline["img_auroc"]
                    print(f"    {full_config}: SKIP (already done, computed AUROC for downstream)")
                    continue
                else:
                    print(f"    {full_config}: SKIP (already done)")
                    continue

            try:
                t0 = time.time()

                n_patches_removed = 0
                n_train_after = contaminated_features.shape[0]
                extra_info = {}  # For method-specific metadata
                use_none_scores = False  # Flag: reuse cached "none" scores

                if method_name == "none":
                    # No purification: use raw bank
                    bank_features = contaminated_features
                    n_train_after = contaminated_features.shape[0]

                elif method_name == "loo_patch":
                    # Compute LOO consistency (once per contamination level)
                    if loo_patch_scores is None:
                        loo_patch_scores, loo_image_scores = compute_loo_consistency(
                            contaminated_features, k=knn_k
                        )
                        if is_debug:
                            for idx in range(len(loo_image_scores)):
                                status = "CONTAMINATED" if is_contaminated[idx] else "clean"
                                print(f"      img[{idx}] ({status}): inconsistency={loo_image_scores[idx]:.4f}")

                    bank_features, n_patches_removed = purify_loo_patch(
                        contaminated_features, loo_patch_scores, percentile_threshold
                    )
                    n_train_after = contaminated_features.shape[0]

                elif method_name == "adaptive_loo_patch":
                    # Compute LOO consistency (once per contamination level)
                    if loo_patch_scores is None:
                        loo_patch_scores, loo_image_scores = compute_loo_consistency(
                            contaminated_features, k=knn_k
                        )
                    bank_features, n_patches_removed, did_purify, diagnostics = purify_adaptive_loo_patch(
                        contaminated_features, loo_patch_scores, loo_image_scores, percentile_threshold
                    )
                    n_train_after = contaminated_features.shape[0]
                    extra_info["adaptive_did_purify"] = int(did_purify)
                    extra_info["adaptive_skewness"] = f"{diagnostics['skewness']:.4f}"
                    # If adaptive decided NOT to purify, bank is identical to "none"
                    if not did_purify:
                        use_none_scores = True

                elif method_name == "random_patch":
                    bank_features, n_patches_removed = purify_random_patch(
                        contaminated_features, percentile_threshold, seed=seed
                    )
                    n_train_after = contaminated_features.shape[0]

                elif method_name == "mahalanobis_patch":
                    bank_features, n_patches_removed = purify_mahalanobis_patch(
                        contaminated_features, percentile_threshold
                    )
                    n_train_after = contaminated_features.shape[0]

                elif method_name == "image_level_loo":
                    # Compute LOO consistency (once per contamination level)
                    if loo_patch_scores is None:
                        loo_patch_scores, loo_image_scores = compute_loo_consistency(
                            contaminated_features, k=knn_k
                        )
                    bank_features, n_patches_removed, n_imgs_removed = purify_image_level_loo(
                        contaminated_features, loo_image_scores, cont_rate
                    )
                    n_train_after = contaminated_features.shape[0] - n_imgs_removed
                    extra_info["n_images_removed"] = n_imgs_removed
                    # If no images removed (cont=0), bank is identical to "none"
                    if n_imgs_removed == 0:
                        use_none_scores = True

                elif method_name == "isolation_forest_patch":
                    bank_features, n_patches_removed = purify_isolation_forest_patch(
                        contaminated_features, percentile_threshold, seed=seed
                    )
                    n_train_after = contaminated_features.shape[0]

                elif method_name == "global_iforest":
                    bank_features, n_patches_removed, n_imgs_removed = purify_global_iforest(
                        contaminated_features, cont_rate, seed=seed
                    )
                    n_train_after = contaminated_features.shape[0] - n_imgs_removed
                    extra_info["n_images_removed"] = n_imgs_removed
                    # If no images removed (cont=0), bank is identical to "none"
                    if n_imgs_removed == 0:
                        use_none_scores = True

                elif method_name == "oracle_patch":
                    bank_features, n_patches_removed = purify_oracle_patch(
                        contaminated_features, is_contaminated
                    )
                    n_oracle_removed = sum(is_contaminated)
                    n_train_after = contaminated_features.shape[0] - n_oracle_removed
                    # If no contaminated images (cont=0), bank is identical to "none"
                    if n_oracle_removed == 0:
                        use_none_scores = True

                elif method_name == "ensemble_loo_mahal":
                    # Needs LOO patch scores
                    if loo_patch_scores is None:
                        loo_patch_scores, loo_image_scores = compute_loo_consistency(
                            contaminated_features, k=knn_k
                        )
                    bank_features, n_patches_removed = purify_ensemble_loo_mahal(
                        contaminated_features, loo_patch_scores, percentile_threshold
                    )
                    n_train_after = contaminated_features.shape[0]

                elif method_name == "iterative_loo":
                    bank_features, n_patches_removed = purify_iterative_loo(
                        contaminated_features, k=knn_k,
                        percentile_threshold=percentile_threshold, max_rounds=3,
                    )
                    n_train_after = contaminated_features.shape[0]

                elif method_name == "cosine_loo":
                    bank_features, n_patches_removed, cosine_pscores, cosine_iscores = purify_cosine_loo(
                        contaminated_features, k=knn_k,
                        percentile_threshold=percentile_threshold,
                    )
                    n_train_after = contaminated_features.shape[0]
                    # Use cosine image scores for contaminant detection AUROC if LOO not computed
                    if loo_image_scores is None:
                        loo_image_scores = cosine_iscores

                elif method_name == "lof_patch":
                    bank_features, n_patches_removed = purify_lof_patch(
                        contaminated_features, percentile_threshold
                    )
                    n_train_after = contaminated_features.shape[0]

                elif method_name == "pca_patch":
                    bank_features, n_patches_removed = purify_pca_patch(
                        contaminated_features, percentile_threshold
                    )
                    n_train_after = contaminated_features.shape[0]

                elif method_name == "grubbs_adaptive_loo":
                    # Needs LOO patch scores
                    if loo_patch_scores is None:
                        loo_patch_scores, loo_image_scores = compute_loo_consistency(
                            contaminated_features, k=knn_k
                        )
                    bank_features, n_patches_removed, did_purify, diagnostics = purify_grubbs_adaptive_loo(
                        contaminated_features, loo_patch_scores, loo_image_scores, percentile_threshold
                    )
                    n_train_after = contaminated_features.shape[0]
                    extra_info["adaptive_did_purify"] = int(did_purify)
                    extra_info["adaptive_skewness"] = f"{diagnostics['grubbs_G']:.4f}"
                    if not did_purify:
                        use_none_scores = True

                elif method_name == "gmm_adaptive_loo":
                    # Needs LOO patch + image scores
                    if loo_patch_scores is None:
                        loo_patch_scores, loo_image_scores = compute_loo_consistency(
                            contaminated_features, k=knn_k
                        )
                    bank_features, n_patches_removed, did_purify, diagnostics = purify_gmm_adaptive_loo(
                        contaminated_features, loo_patch_scores, loo_image_scores, percentile_threshold
                    )
                    n_train_after = contaminated_features.shape[0]
                    extra_info["adaptive_did_purify"] = int(did_purify)
                    extra_info["adaptive_skewness"] = f"{diagnostics.get('bic_delta', 0.0):.4f}"
                    if not did_purify:
                        use_none_scores = True

                elif method_name == "mad_adaptive_loo":
                    # Needs LOO patch + image scores
                    if loo_patch_scores is None:
                        loo_patch_scores, loo_image_scores = compute_loo_consistency(
                            contaminated_features, k=knn_k
                        )
                    bank_features, n_patches_removed, did_purify, diagnostics = purify_mad_adaptive_loo(
                        contaminated_features, loo_patch_scores, loo_image_scores, percentile_threshold
                    )
                    n_train_after = contaminated_features.shape[0]
                    extra_info["adaptive_did_purify"] = int(did_purify)
                    extra_info["adaptive_skewness"] = f"{diagnostics.get('mad_max_z', 0.0):.4f}"
                    if not did_purify:
                        use_none_scores = True

                else:
                    raise ValueError(f"Unknown purification method: {method_name}")

                # Score test set (reuse cached scores when bank is identical to "none")
                if use_none_scores and none_scores_cache is not None:
                    scores = none_scores_cache
                else:
                    # Guard: ensure bank has enough patches for KNN
                    n_bank_patches = bank_features.shape[0] if bank_features.ndim == 2 \
                        else bank_features.reshape(-1, bank_features.shape[-1]).shape[0]
                    if n_bank_patches < knn_k:
                        print(f"    [WARN] {full_config}: bank too small ({n_bank_patches} < {knn_k}), skip")
                        continue

                    scores = compute_knn_scores(bank_features, test_features, k=knn_k)

                    # Cache "none" baseline scores for reuse
                    if method_name == "none":
                        none_scores_cache = scores

                elapsed = time.time() - t0

                # Compute metrics
                metrics = compute_metrics(test_labels, scores)

                # Track clean baseline AUROC
                if cont_rate == 0.0 and method_name == "none":
                    clean_auroc = metrics["img_auroc"]

                # Track dirty (unpurified) AUROC per contamination rate
                if method_name == "none":
                    dirty_aurocs[cont_rate] = metrics["img_auroc"]

                # Compute delta vs clean and recovery rate
                auroc_delta = ""
                recovery_rate = ""
                if clean_auroc is not None:
                    if cont_rate == 0.0 and method_name == "none":
                        pass  # This IS the baseline
                    else:
                        auroc_delta = f"{metrics['img_auroc'] - clean_auroc:.4f}"

                    # Recovery rate: (purified - dirty) / (clean - dirty)
                    if method_name != "none" and cont_rate > 0.0 and cont_rate in dirty_aurocs:
                        dirty_auroc = dirty_aurocs[cont_rate]
                        damage = clean_auroc - dirty_auroc
                        if abs(damage) > 1e-6:
                            recovery = (metrics["img_auroc"] - dirty_auroc) / damage
                            recovery_rate = f"{recovery:.4f}"
                        else:
                            recovery_rate = "1.0000"  # No damage to recover from

                # Contaminant detection AUROC (using LOO image scores)
                contaminant_det_auroc = ""
                p_at_1 = ""
                p_at_k = ""
                r_at_k = ""
                if loo_image_scores is not None and cont_rate > 0.0 and n_contaminated > 0:
                    # Can we distinguish contaminated vs clean images by LOO score?
                    is_cont_array = np.array(is_contaminated)
                    if len(np.unique(is_cont_array)) == 2:
                        contaminant_det_auroc = f"{roc_auc_score(is_cont_array, loo_image_scores):.4f}"

                    # Precision@k / Recall@k (operational metrics)
                    # Higher LOO score = more suspicious
                    ranked_indices = np.argsort(loo_image_scores)[::-1]  # descending
                    ranked_labels = is_cont_array[ranked_indices]

                    # P@1: is the most suspicious image a contaminant?
                    p_at_1 = f"{ranked_labels[0]:.0f}"

                    # P@k and R@k where k = n_contaminated
                    k = n_contaminated
                    top_k_labels = ranked_labels[:k]
                    true_positives = int(top_k_labels.sum())
                    p_at_k = f"{true_positives / k:.4f}" if k > 0 else ""
                    r_at_k = f"{true_positives / n_contaminated:.4f}" if n_contaminated > 0 else ""

                result = {
                    "timestamp": datetime.now().isoformat(),
                    "dataset": dataset_name,
                    "config": full_config,
                    "contamination_rate": str(cont_rate),
                    "purification_method": method_name,
                    "train_limit": train_limit,
                    "n_train_original": contaminated_features.shape[0],
                    "n_train_after_purification": n_train_after,
                    "n_contaminated": n_contaminated,
                    "n_patches_removed": n_patches_removed,
                    "n_test": n_test,
                    "n_normal_test": n_normal_test,
                    "n_anomaly_test": n_anomaly_test,
                    "seed": seed,
                    "knn_k": knn_k,
                    "percentile_threshold": percentile_threshold,
                    "img_auroc": f"{metrics['img_auroc']:.4f}",
                    "img_aupr": f"{metrics['img_aupr']:.4f}",
                    "img_f1max": f"{metrics['img_f1max']:.4f}",
                    "tpr_fpr1": f"{metrics['tpr_fpr1']:.4f}",
                    "tpr_fpr5": f"{metrics['tpr_fpr5']:.4f}",
                    "auroc_delta_vs_clean": auroc_delta,
                    "recovery_rate": recovery_rate,
                    "contaminant_detection_auroc": contaminant_det_auroc,
                    "contaminant_precision_at_1": p_at_1,
                    "contaminant_precision_at_k": p_at_k,
                    "contaminant_recall_at_k": r_at_k,
                    "adaptive_did_purify": extra_info.get("adaptive_did_purify", ""),
                    "adaptive_skewness": extra_info.get("adaptive_skewness", ""),
                    "inference_time_s": f"{elapsed:.2f}",
                }
                save_result(result, csv_path)

                recovery_str = f" recovery={recovery_rate}" if recovery_rate else ""
                try:
                    print(f"    {full_config}: AUROC={metrics['img_auroc']:.3f} "
                          f"(delta={auroc_delta if auroc_delta else 'baseline'}) "
                          f"patches_removed={n_patches_removed}{recovery_str}")
                except OSError:
                    pass  # Windows console encoding issue; result already saved

            except Exception as e:
                print(f"    [ERROR] {full_config}: {e}")
                log_error(dataset_name, full_config, seed, e)
                errors.append(f"{dataset_name}/{full_config}: {e}")
                if is_debug:
                    traceback.print_exc()

    return errors


# =============================================================================
# RUN MODES
# =============================================================================

def run_main_experiment(args) -> None:
    """Main experiment: 35 datasets, 5 seeds, TL=10,20, 4 contamination rates."""
    datasets = QUICK_TEST_DATASETS[:2] if args.debug else ALL_DATASETS
    seeds = [0] if args.debug else SEEDS
    train_limits = [10] if args.debug else TRAIN_LIMITS
    contamination_rates = [0.0, 0.2] if args.debug else CONTAMINATION_RATES

    csv_path = OUTPUT_DIR / "results_v2.csv"
    completed = get_completed_experiments(csv_path)

    methods = get_methods_for_tier(args.methods)

    total_experiments = (len(datasets) * len(seeds) * len(train_limits) *
                        len(contamination_rates) * len(methods))

    print("=" * 70)
    print("P3_002 FULL: Dirty Few-Shot Self-Purification (CPU-only, cached features)")
    print(f"  Mode: {'DEBUG' if args.debug else 'MAIN'} | Methods: {args.methods} ({len(methods)} methods)")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Seeds: {seeds}")
    print(f"  Train limits: {train_limits}")
    print(f"  Contamination rates: {contamination_rates}")
    print(f"  Methods: {methods}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Already completed: {len(completed)}")
    print(f"  Output: {csv_path}")
    print("=" * 70)

    all_errors = []

    for ds_idx, dataset_name in enumerate(datasets):
        ds_start = time.time()
        log(f"[{ds_idx+1}/{len(datasets)}] Dataset: {dataset_name}")

        # Load cached features ONCE per dataset
        try:
            cached_data = load_cached_dataset(dataset_name)
            print(f"  Loaded cache: train={cached_data['all_train_features'].shape}, "
                  f"test={cached_data['test_features'].shape}")
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            log_error(dataset_name, "cache_load", 0, e)
            all_errors.append(f"{dataset_name}: {e}")
            continue

        for seed in seeds:
            print(f"\n  seed={seed}")

            for train_limit in train_limits:
                print(f"  train_limit={train_limit}")
                try:
                    errors = run_dataset_experiment(
                        dataset_name, seed, train_limit,
                        knn_k=DEFAULT_KNN_K,
                        percentile_threshold=DEFAULT_PERCENTILE_THRESHOLD,
                        cached_data=cached_data,
                        completed=completed,
                        csv_path=csv_path,
                        contamination_rates=contamination_rates,
                        purification_methods=methods,
                        is_debug=args.debug,
                    )
                    all_errors.extend(errors)
                except Exception as e:
                    print(f"    [ERROR] Fatal: {e}")
                    log_error(dataset_name, "all", seed, e)
                    all_errors.append(f"{dataset_name}: {e}")
                    if args.debug:
                        traceback.print_exc()

        # Free memory after each dataset
        del cached_data
        gc.collect()

        ds_elapsed = time.time() - ds_start
        log(f"  {dataset_name} done in {ds_elapsed/60:.1f} min")

    # Summary
    print("\n" + "=" * 70)
    log(f"DONE. Errors: {len(all_errors)}")
    for err in all_errors:
        print(f"  - {err}")


def run_ablation_percentile(args) -> None:
    """Ablation: vary percentile_threshold (80, 90, 95, 98, 99).
    Fixed: 6 Quick Test datasets, 5 seeds, cont=0.0+0.3, TL=10, K=5.
    """
    datasets = QUICK_TEST_DATASETS[:2] if args.debug else QUICK_TEST_DATASETS
    seeds = [0] if args.debug else SEEDS
    percentile_values = [90, 95] if args.debug else ABLATION_PERCENTILE_VALUES

    csv_path = OUTPUT_DIR / "ablation_percentile.csv"
    completed = get_completed_experiments(csv_path)

    methods = get_methods_for_tier(args.methods)

    total = (len(datasets) * len(seeds) * len(percentile_values) *
             len(ABLATION_CONTAMINATION_RATES) * len(methods))

    print("=" * 70)
    print("P3_002 ABLATION: Percentile Threshold (CPU-only, cached features)")
    print(f"  Mode: {'DEBUG' if args.debug else 'ABLATION'} | Methods: {args.methods} ({len(methods)} methods)")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Seeds: {seeds}")
    print(f"  Percentile values: {percentile_values}")
    print(f"  Train limit: {ABLATION_TRAIN_LIMIT}")
    print(f"  Contamination: {ABLATION_CONTAMINATION_RATES}")
    print(f"  Methods: {methods}")
    print(f"  Total experiments: {total}")
    print(f"  Already completed: {len(completed)}")
    print(f"  Output: {csv_path}")
    print("=" * 70)

    all_errors = []

    for ds_idx, dataset_name in enumerate(datasets):
        ds_start = time.time()
        log(f"[{ds_idx+1}/{len(datasets)}] Ablation percentile: {dataset_name}")

        try:
            cached_data = load_cached_dataset(dataset_name)
            print(f"  Loaded cache: train={cached_data['all_train_features'].shape}, "
                  f"test={cached_data['test_features'].shape}")
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            log_error(dataset_name, "cache_load", 0, e)
            all_errors.append(f"{dataset_name}: {e}")
            continue

        for seed in seeds:
            print(f"\n  seed={seed}")

            for pct in percentile_values:
                print(f"  percentile_threshold={pct}")
                try:
                    errors = run_dataset_experiment(
                        dataset_name, seed,
                        train_limit=ABLATION_TRAIN_LIMIT,
                        knn_k=DEFAULT_KNN_K,
                        percentile_threshold=pct,
                        cached_data=cached_data,
                        completed=completed,
                        csv_path=csv_path,
                        contamination_rates=ABLATION_CONTAMINATION_RATES,
                        purification_methods=methods,
                        is_debug=args.debug,
                    )
                    all_errors.extend(errors)
                except Exception as e:
                    print(f"    [ERROR] Fatal: {e}")
                    log_error(dataset_name, f"abl_pct_{pct}", seed, e)
                    all_errors.append(f"{dataset_name}/pct_{pct}: {e}")

        del cached_data
        gc.collect()

        ds_elapsed = time.time() - ds_start
        log(f"  {dataset_name} done in {ds_elapsed/60:.1f} min")

    print("\n" + "=" * 70)
    log(f"ABLATION PERCENTILE DONE. Errors: {len(all_errors)}")
    for err in all_errors:
        print(f"  - {err}")


def run_ablation_knn_k(args) -> None:
    """Ablation: vary KNN K (1, 3, 5, 7, 10).
    Fixed: 6 Quick Test datasets, 5 seeds, cont=0.0+0.3, TL=10, percentile=95.
    """
    datasets = QUICK_TEST_DATASETS[:2] if args.debug else QUICK_TEST_DATASETS
    seeds = [0] if args.debug else SEEDS
    k_values = [3, 5] if args.debug else ABLATION_KNN_K_VALUES

    csv_path = OUTPUT_DIR / "ablation_knn_k.csv"
    completed = get_completed_experiments(csv_path)

    methods = get_methods_for_tier(args.methods)

    total = (len(datasets) * len(seeds) * len(k_values) *
             len(ABLATION_CONTAMINATION_RATES) * len(methods))

    print("=" * 70)
    print("P3_002 ABLATION: KNN K (CPU-only, cached features)")
    print(f"  Mode: {'DEBUG' if args.debug else 'ABLATION'} | Methods: {args.methods} ({len(methods)} methods)")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Seeds: {seeds}")
    print(f"  K values: {k_values}")
    print(f"  Train limit: {ABLATION_TRAIN_LIMIT}")
    print(f"  Percentile: {DEFAULT_PERCENTILE_THRESHOLD}")
    print(f"  Contamination: {ABLATION_CONTAMINATION_RATES}")
    print(f"  Methods: {methods}")
    print(f"  Total experiments: {total}")
    print(f"  Already completed: {len(completed)}")
    print(f"  Output: {csv_path}")
    print("=" * 70)

    all_errors = []

    for ds_idx, dataset_name in enumerate(datasets):
        ds_start = time.time()
        log(f"[{ds_idx+1}/{len(datasets)}] Ablation KNN K: {dataset_name}")

        try:
            cached_data = load_cached_dataset(dataset_name)
            print(f"  Loaded cache: train={cached_data['all_train_features'].shape}, "
                  f"test={cached_data['test_features'].shape}")
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            log_error(dataset_name, "cache_load", 0, e)
            all_errors.append(f"{dataset_name}: {e}")
            continue

        for seed in seeds:
            print(f"\n  seed={seed}")

            for k_val in k_values:
                print(f"  knn_k={k_val}")
                try:
                    errors = run_dataset_experiment(
                        dataset_name, seed,
                        train_limit=ABLATION_TRAIN_LIMIT,
                        knn_k=k_val,
                        percentile_threshold=DEFAULT_PERCENTILE_THRESHOLD,
                        cached_data=cached_data,
                        completed=completed,
                        csv_path=csv_path,
                        contamination_rates=ABLATION_CONTAMINATION_RATES,
                        purification_methods=methods,
                        is_debug=args.debug,
                    )
                    all_errors.extend(errors)
                except Exception as e:
                    print(f"    [ERROR] Fatal: {e}")
                    log_error(dataset_name, f"abl_k_{k_val}", seed, e)
                    all_errors.append(f"{dataset_name}/k_{k_val}: {e}")

        del cached_data
        gc.collect()

        ds_elapsed = time.time() - ds_start
        log(f"  {dataset_name} done in {ds_elapsed/60:.1f} min")

    print("\n" + "=" * 70)
    log(f"ABLATION KNN_K DONE. Errors: {len(all_errors)}")
    for err in all_errors:
        print(f"  - {err}")


# =============================================================================
# GATE CHECK MODE
# =============================================================================

# Gate check datasets: representative subset (skip LOCO for gate check)
GATE_CHECK_DATASETS = [
    "mvtec_AD/bottle", "mvtec_AD/cable", "VisA/pcb1", "btad/01",
]


def run_gate_check(args) -> None:
    """Gate check: compare ALL purification methods on 4 datasets, 2 seeds.

    Purpose: verify LOO beats mahalanobis/random/IForest BEFORE scaling up.
    If LOO doesn't clearly win → STOP and rethink.

    Quick run: 4 datasets × 2 seeds × 2 cont × 9 methods = 144 experiments.
    """
    datasets = GATE_CHECK_DATASETS
    seeds = [0, 1]
    contamination_rates = [0.0, 0.3]
    train_limit = 10

    csv_path = OUTPUT_DIR / "gate_check.csv"
    completed = get_completed_experiments(csv_path)

    methods = PURIFICATION_METHODS

    total = len(datasets) * len(seeds) * len(contamination_rates) * len(methods)

    print("=" * 70)
    print("P3_002 GATE CHECK: LOO vs Baselines (CPU-only)")
    print(f"  Datasets: {datasets}")
    print(f"  Seeds: {seeds}")
    print(f"  Train limit: {train_limit}")
    print(f"  Contamination: {contamination_rates}")
    print(f"  Methods: {methods}")
    print(f"  Total experiments: {total}")
    print(f"  Already completed: {len(completed)}")
    print(f"  Output: {csv_path}")
    print("=" * 70)

    all_errors = []

    for ds_idx, dataset_name in enumerate(datasets):
        ds_start = time.time()
        log(f"[{ds_idx+1}/{len(datasets)}] Gate check: {dataset_name}")

        try:
            cached_data = load_cached_dataset(dataset_name)
            print(f"  Loaded cache: train={cached_data['all_train_features'].shape}, "
                  f"test={cached_data['test_features'].shape}")
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            log_error(dataset_name, "cache_load", 0, e)
            all_errors.append(f"{dataset_name}: {e}")
            continue

        for seed in seeds:
            print(f"\n  seed={seed}")
            try:
                errors = run_dataset_experiment(
                    dataset_name, seed, train_limit,
                    knn_k=DEFAULT_KNN_K,
                    percentile_threshold=DEFAULT_PERCENTILE_THRESHOLD,
                    cached_data=cached_data,
                    completed=completed,
                    csv_path=csv_path,
                    contamination_rates=contamination_rates,
                    purification_methods=methods,
                    is_debug=False,
                )
                all_errors.extend(errors)
            except Exception as e:
                print(f"    [ERROR] Fatal: {e}")
                log_error(dataset_name, "all", seed, e)
                all_errors.append(f"{dataset_name}: {e}")
                traceback.print_exc()

        del cached_data
        gc.collect()

        ds_elapsed = time.time() - ds_start
        log(f"  {dataset_name} done in {ds_elapsed/60:.1f} min")

    # Summary analysis
    print("\n" + "=" * 70)
    log(f"GATE CHECK DONE. Errors: {len(all_errors)}")
    for err in all_errors:
        print(f"  - {err}")

    # Auto-analyze results
    if csv_path.exists():
        print("\n" + "=" * 70)
        print("GATE CHECK ANALYSIS")
        print("=" * 70)
        analyze_gate_check(csv_path)


def analyze_gate_check(csv_path: Path) -> None:
    """Analyze gate check results and print comparison table."""
    # Avoid pandas dependency; parse CSV directly.
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            try:
                r["contamination_rate"] = float(r["contamination_rate"])
                r["img_auroc"] = float(r["img_auroc"])
            except Exception:
                continue
            rows.append(r)

    dirty_rows = [r for r in rows if abs(r["contamination_rate"] - 0.3) < 1e-9]
    if not dirty_rows:
        print("  No cont=0.3 results found.")
        return

    clean_rows = [r for r in rows if abs(r["contamination_rate"] - 0.0) < 1e-9 and r.get("purification_method") == "none"]
    clean_auroc = (float(np.mean([r["img_auroc"] for r in clean_rows])) if clean_rows else None)

    # Mean/std AUROC per method
    by_method = defaultdict(list)
    for r in dirty_rows:
        by_method[r["purification_method"]].append(r["img_auroc"])

    method_means = {m: float(np.mean(v)) for m, v in by_method.items()}
    method_stds = {m: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for m, v in by_method.items()}

    print(f"\n  Clean baseline AUROC: {clean_auroc:.4f}" if clean_auroc is not None else "  Clean baseline: N/A")
    print(f"\n  Method comparison at cont=30%:")
    print(f"  {'Method':<25} {'AUROC':>8} {'±std':>8} {'Recovery':>10}")
    print(f"  {'-'*55}")

    for method in PURIFICATION_METHODS:
        if method in method_means:
            auroc = method_means[method]
            std = method_stds[method]
            recovery = ""
            if clean_auroc is not None and method != "none":
                dirty_auroc = method_means.get("none", None)
                if dirty_auroc is not None and clean_auroc > dirty_auroc + 1e-6:
                    rec = (auroc - dirty_auroc) / (clean_auroc - dirty_auroc)
                    recovery = f"{rec*100:.1f}%"
            print(f"  {method:<25} {auroc:.4f}   ±{std:.4f}   {recovery:>10}")

    # Per-dataset breakdown
    print(f"\n  Per-dataset AUROC at cont=30%:")
    datasets = sorted({r["dataset"] for r in dirty_rows})
    methods_present = [m for m in PURIFICATION_METHODS if m in by_method]

    header = f"  {'Dataset':<25}" + "".join(f" {m[:12]:>12}" for m in methods_present)
    print(header)
    print(f"  {'-'*len(header)}")

    by_ds_method = defaultdict(list)
    for r in dirty_rows:
        by_ds_method[(r["dataset"], r["purification_method"])].append(r["img_auroc"])

    for ds in datasets:
        line = f"  {ds:<25}"
        for m in methods_present:
            vals = by_ds_method.get((ds, m), [])
            if vals:
                line += f" {float(np.mean(vals)):>12.4f}"
            else:
                line += f" {'N/A':>12}"
        print(line)

    # GATE CHECK VERDICT
    print(f"\n  {'='*55}")
    loo_auroc = method_means.get("loo_patch", None)
    mahal_auroc = method_means.get("mahalanobis_patch", None)
    random_auroc = method_means.get("random_patch", None)

    if loo_auroc is not None:
        ensemble_auroc = method_means.get("ensemble_loo_mahal", None)
        verdict_parts = []
        if random_auroc is not None:
            diff = loo_auroc - random_auroc
            verdict_parts.append(f"LOO vs random: {diff:+.4f} ({'WIN' if diff > 0 else 'LOSE'})")
        if mahal_auroc is not None:
            diff = loo_auroc - mahal_auroc
            verdict_parts.append(f"LOO vs mahalanobis: {diff:+.4f} ({'WIN' if diff > 0 else 'LOSE'})")
        if ensemble_auroc is not None:
            diff_vs_loo = ensemble_auroc - loo_auroc
            diff_vs_mahal = ensemble_auroc - (mahal_auroc or 0)
            verdict_parts.append(f"Ensemble vs LOO: {diff_vs_loo:+.4f} ({'WIN' if diff_vs_loo > 0 else 'LOSE'})")
            verdict_parts.append(f"Ensemble vs mahalanobis: {diff_vs_mahal:+.4f} ({'WIN' if diff_vs_mahal > 0 else 'LOSE'})")

        print(f"  VERDICT:")
        for v in verdict_parts:
            print(f"    {v}")

        # Best method (excluding oracle and none)
        non_oracle_means = {m: v for m, v in method_means.items() if m not in ("oracle_patch", "none")}
        best_method = max(non_oracle_means, key=non_oracle_means.get) if non_oracle_means else "?"
        best_auroc = non_oracle_means.get(best_method, 0)
        oracle_auroc = method_means.get("oracle_patch", None)
        print(f"\n  Best non-oracle method: {best_method} (AUROC={best_auroc:.4f})")
        if oracle_auroc is not None:
            pct = (best_auroc - method_means.get("none", 0)) / (oracle_auroc - method_means.get("none", 0)) * 100 if oracle_auroc > method_means.get("none", 0) + 1e-6 else 0
            print(f"  Oracle: {oracle_auroc:.4f} ({pct:.1f}% of oracle gap achieved)")

        if ensemble_auroc is not None and ensemble_auroc >= loo_auroc and ensemble_auroc >= (mahal_auroc or 0):
            print(f"\n  PASS: Ensemble beats both LOO and Mahalanobis. Framework narrative supported.")
        elif mahal_auroc is not None and loo_auroc > mahal_auroc:
            print(f"\n  PASS: LOO beats Mahalanobis baseline.")
        elif mahal_auroc is not None:
            print(f"\n  NOTE: LOO does NOT beat Mahalanobis. Ensemble/framework narrative needed.")
        else:
            print(f"\n  ? GATE CHECK INCONCLUSIVE: Mahalanobis results missing.")


# =============================================================================
# EXTRA CONTAMINATION RATES (degradation curve + break-even)
# =============================================================================

def run_extra_cont(args) -> None:
    """Extra contamination rates (0.05, 0.4, 0.5) for degradation curve.

    Same as main experiment but with different cont rates.
    Writes to separate CSV (results_extra_cont.csv) to avoid mixing with main results.
    Core methods only (7), 35 datasets, 5 seeds, TL=10+20.
    """
    datasets = QUICK_TEST_DATASETS[:2] if args.debug else ALL_DATASETS
    seeds = [0] if args.debug else SEEDS
    train_limits = [10] if args.debug else [10]  # TL=10 only (TL=20 adds 2x time, marginal info)
    contamination_rates = [0.0, 0.4] if args.debug else EXTRA_CONTAMINATION_RATES

    csv_path = OUTPUT_DIR / "results_extra_cont.csv"
    completed = get_completed_experiments(csv_path)

    methods = get_methods_for_tier(args.methods)

    total_experiments = (len(datasets) * len(seeds) * len(train_limits) *
                        len(contamination_rates) * len(methods))

    print("=" * 70)
    print("P3_002 EXTRA CONT RATES: Degradation curve + break-even analysis")
    print(f"  Mode: {'DEBUG' if args.debug else 'FULL'} | Methods: {args.methods} ({len(methods)} methods)")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Seeds: {seeds}")
    print(f"  Train limits: {train_limits}")
    print(f"  Contamination rates: {contamination_rates}")
    print(f"  Methods: {methods}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Already completed: {len(completed)}")
    print(f"  Output: {csv_path}")
    print("=" * 70)

    all_errors = []

    for ds_idx, dataset_name in enumerate(datasets):
        ds_start = time.time()
        log(f"[{ds_idx+1}/{len(datasets)}] Dataset: {dataset_name}")

        try:
            cached_data = load_cached_dataset(dataset_name)
            print(f"  Loaded cache: train={cached_data['all_train_features'].shape}, "
                  f"test={cached_data['test_features'].shape}")
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            log_error(dataset_name, "cache_load", 0, e)
            all_errors.append(f"{dataset_name}: {e}")
            continue

        for seed in seeds:
            print(f"\n  seed={seed}")

            for train_limit in train_limits:
                print(f"  train_limit={train_limit}")
                try:
                    errors = run_dataset_experiment(
                        dataset_name, seed, train_limit,
                        knn_k=DEFAULT_KNN_K,
                        percentile_threshold=DEFAULT_PERCENTILE_THRESHOLD,
                        cached_data=cached_data,
                        completed=completed,
                        csv_path=csv_path,
                        contamination_rates=contamination_rates,
                        purification_methods=methods,
                        is_debug=args.debug,
                    )
                    all_errors.extend(errors)
                except Exception as e:
                    print(f"    [ERROR] Fatal: {e}")
                    log_error(dataset_name, "all", seed, e)
                    all_errors.append(f"{dataset_name}: {e}")
                    if args.debug:
                        traceback.print_exc()

        del cached_data
        gc.collect()

        ds_elapsed = time.time() - ds_start
        log(f"  {dataset_name} done in {ds_elapsed/60:.1f} min")

    print("\n" + "=" * 70)
    log(f"EXTRA CONT DONE. Errors: {len(all_errors)}")
    for err in all_errors:
        print(f"  - {err}")


# =============================================================================
# TL=5 FEW-SHOT EXPERIMENT
# =============================================================================

def run_tl5(args) -> None:
    """TL=5 few-shot experiment: minimal support set with m={0,1,2} contaminated images.

    Demonstrates purification works even with the smallest practical support set.
    Writes to separate CSV (results_tl5.csv).
    """
    datasets = QUICK_TEST_DATASETS[:2] if args.debug else ALL_DATASETS
    seeds = [0] if args.debug else SEEDS
    train_limits = [5]
    contamination_rates = [0.0, 0.3] if args.debug else TL5_CONTAMINATION_RATES
    methods = TL5_METHODS

    csv_path = OUTPUT_DIR / "results_tl5.csv"
    completed = get_completed_experiments(csv_path)

    total_experiments = (len(datasets) * len(seeds) * len(train_limits) *
                        len(contamination_rates) * len(methods))

    print("=" * 70)
    print("P3_002 TL=5: Few-shot minimal support set experiment")
    print(f"  Mode: {'DEBUG' if args.debug else 'FULL'}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Seeds: {seeds}")
    print(f"  Train limits: {train_limits}")
    print(f"  Contamination rates: {contamination_rates} (m=0,1,2 images)")
    print(f"  Methods: {methods}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Already completed: {len(completed)}")
    print(f"  Output: {csv_path}")
    print("=" * 70)

    all_errors = []

    for ds_idx, dataset_name in enumerate(datasets):
        ds_start = time.time()
        log(f"[{ds_idx+1}/{len(datasets)}] Dataset: {dataset_name}")

        try:
            cached_data = load_cached_dataset(dataset_name)
            print(f"  Loaded cache: train={cached_data['all_train_features'].shape}, "
                  f"test={cached_data['test_features'].shape}")
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            log_error(dataset_name, "cache_load", 0, e)
            all_errors.append(f"{dataset_name}: {e}")
            continue

        for seed in seeds:
            print(f"\n  seed={seed}")

            for train_limit in train_limits:
                print(f"  train_limit={train_limit}")
                try:
                    errors = run_dataset_experiment(
                        dataset_name, seed, train_limit,
                        knn_k=DEFAULT_KNN_K,
                        percentile_threshold=DEFAULT_PERCENTILE_THRESHOLD,
                        cached_data=cached_data,
                        completed=completed,
                        csv_path=csv_path,
                        contamination_rates=contamination_rates,
                        purification_methods=methods,
                        is_debug=args.debug,
                    )
                    all_errors.extend(errors)
                except Exception as e:
                    print(f"    [ERROR] Fatal: {e}")
                    log_error(dataset_name, "all", seed, e)
                    all_errors.append(f"{dataset_name}: {e}")
                    if args.debug:
                        traceback.print_exc()

        del cached_data
        gc.collect()

        ds_elapsed = time.time() - ds_start
        log(f"  {dataset_name} done in {ds_elapsed/60:.1f} min")

    print("\n" + "=" * 70)
    log(f"TL=5 DONE. Errors: {len(all_errors)}")
    for err in all_errors:
        print(f"  - {err}")


# =============================================================================
# ABLATION PERCENTILE ON ALL DATASETS
# =============================================================================

def run_ablation_full_pct(args) -> None:
    """Ablation percentile threshold on all 35 datasets.

    Fixed: cont=0.3, TL=10, ensemble only. Varies pct={90,95,99}.
    Much stronger than the 6-dataset ablation (avoids gate-check bias).
    Writes to separate CSV (ablation_full_percentile.csv).
    """
    datasets = QUICK_TEST_DATASETS[:2] if args.debug else ALL_DATASETS
    seeds = [0] if args.debug else SEEDS
    train_limits = [10]
    # cont=0.0 shows clean penalty sensitivity to pct, cont=0.3 shows recovery sensitivity
    contamination_rates = [0.0, 0.3] if args.debug else [0.0, 0.3]
    methods = ABLATION_FULL_PCT_METHODS
    pct_values = ABLATION_FULL_PCT_VALUES

    csv_path = OUTPUT_DIR / "ablation_full_percentile.csv"
    completed = get_completed_experiments(csv_path)

    total_experiments = (len(datasets) * len(seeds) * len(train_limits) *
                        len(contamination_rates) * len(methods) * len(pct_values))
    # Add 'none' baseline per cont rate (1 per dataset×seed×cont)
    total_experiments += len(datasets) * len(seeds) * len(contamination_rates)

    print("=" * 70)
    print("P3_002 ABLATION FULL PCT: Percentile threshold on 35 datasets")
    print(f"  Mode: {'DEBUG' if args.debug else 'FULL'}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Seeds: {seeds}")
    print(f"  Contamination rates: {contamination_rates}")
    print(f"  Percentile values: {pct_values}")
    print(f"  Methods: ['none'] + {methods}")
    print(f"  Total experiments: ~{total_experiments}")
    print(f"  Already completed: {len(completed)}")
    print(f"  Output: {csv_path}")
    print("=" * 70)

    all_errors = []

    for ds_idx, dataset_name in enumerate(datasets):
        ds_start = time.time()
        log(f"[{ds_idx+1}/{len(datasets)}] Dataset: {dataset_name}")

        try:
            cached_data = load_cached_dataset(dataset_name)
            print(f"  Loaded cache: train={cached_data['all_train_features'].shape}, "
                  f"test={cached_data['test_features'].shape}")
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            log_error(dataset_name, "cache_load", 0, e)
            all_errors.append(f"{dataset_name}: {e}")
            continue

        for seed in seeds:
            print(f"\n  seed={seed}")

            for pct in pct_values:
                print(f"    pct={pct}")
                for train_limit in train_limits:
                    try:
                        # Run with none + ensemble at this percentile
                        errors = run_dataset_experiment(
                            dataset_name, seed, train_limit,
                            knn_k=DEFAULT_KNN_K,
                            percentile_threshold=pct,
                            cached_data=cached_data,
                            completed=completed,
                            csv_path=csv_path,
                            contamination_rates=contamination_rates,
                            purification_methods=["none"] + methods,
                            is_debug=args.debug,
                        )
                        all_errors.extend(errors)
                    except Exception as e:
                        print(f"    [ERROR] Fatal: {e}")
                        log_error(dataset_name, "all", seed, e)
                        all_errors.append(f"{dataset_name}: {e}")
                        if args.debug:
                            traceback.print_exc()

        del cached_data
        gc.collect()

        ds_elapsed = time.time() - ds_start
        log(f"  {dataset_name} done in {ds_elapsed/60:.1f} min")

    print("\n" + "=" * 70)
    log(f"ABLATION FULL PCT DONE. Errors: {len(all_errors)}")
    for err in all_errors:
        print(f"  - {err}")


# =============================================================================
# PIXEL AUROC EXPERIMENT (patch score maps vs GT masks)
# =============================================================================

PIXEL_FIELDNAMES = [
    "timestamp", "dataset", "config", "contamination_rate", "purification_method",
    "train_limit", "n_train_original", "n_contaminated", "n_patches_removed",
    "n_test", "seed",
    "knn_k", "percentile_threshold",
    "img_auroc", "pix_auroc", "pix_aupr",
    "n_masks_found",
    "auroc_delta_vs_clean", "recovery_rate",
    "pix_auroc_delta_vs_clean", "pix_recovery_rate",
    "inference_time_s",
]


def get_completed_pixel_experiments(csv_path: Path) -> set[tuple]:
    """Read pixel AUROC CSV and return set of completed experiment keys."""
    completed = set()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                key = (
                    row["dataset"],
                    row["config"],
                    row["contamination_rate"],
                    int(row["seed"]),
                    str(row.get("train_limit", "10")),
                )
                completed.add(key)
    return completed


def save_pixel_result(result: dict, csv_path: Path) -> None:
    """Append one pixel AUROC result row to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PIXEL_FIELDNAMES, delimiter=";")
        if not file_exists:
            writer.writeheader()
        row = {k: result.get(k, "") for k in PIXEL_FIELDNAMES}
        writer.writerow(row)


def run_pixel_auroc(args) -> None:
    """Pixel-level AUROC experiment: compute patch score maps and compare vs GT masks.

    For each dataset/seed/cont_rate/method:
      1. Purify memory bank (same as main experiments)
      2. Score test set with compute_knn_scores_with_maps (returns patch_score_maps)
      3. Load GT masks from disk
      4. Upsample patch maps → pixel level → compute pixel AUROC/AUPR

    Writes to results_pixel_auroc.csv.
    """
    datasets = QUICK_TEST_DATASETS[:2] if args.debug else ALL_DATASETS
    seeds = [0] if args.debug else SEEDS
    train_limit = 10
    contamination_rates = [0.0, 0.3] if args.debug else PIXEL_AUROC_CONTAMINATION_RATES
    methods = PIXEL_AUROC_METHODS
    knn_k = DEFAULT_KNN_K
    pct = DEFAULT_PERCENTILE_THRESHOLD

    csv_path = OUTPUT_DIR / "results_pixel_auroc.csv"
    completed = get_completed_pixel_experiments(csv_path)

    total_experiments = len(datasets) * len(seeds) * len(contamination_rates) * len(methods)

    print("=" * 70)
    print("P3_002 PIXEL AUROC: Patch score maps vs GT masks")
    print(f"  Mode: {'DEBUG' if args.debug else 'FULL'}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Seeds: {seeds}")
    print(f"  Train limit: {train_limit}")
    print(f"  Contamination rates: {contamination_rates}")
    print(f"  Methods: {methods}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Already completed: {len(completed)}")
    print(f"  Output: {csv_path}")
    print("=" * 70)

    all_errors = []

    for ds_idx, dataset_name in enumerate(datasets):
        ds_start = time.time()
        log(f"[{ds_idx+1}/{len(datasets)}] Dataset: {dataset_name}")

        try:
            cached_data = load_cached_dataset(dataset_name)
            print(f"  Loaded cache: train={cached_data['all_train_features'].shape}, "
                  f"test={cached_data['test_features'].shape}")
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            log_error(dataset_name, "cache_load", 0, e)
            all_errors.append(f"{dataset_name}: {e}")
            continue

        # Determine patch grid from feature shape (784→28×28 for DINO, 256→16×16 for CLIP)
        n_patches = cached_data["test_features"].shape[1]
        patch_grid = int(np.sqrt(n_patches))
        assert patch_grid * patch_grid == n_patches, f"Non-square patch grid: {n_patches}"
        print(f"  Patch grid: {patch_grid}×{patch_grid} ({n_patches} patches)")

        test_paths = cached_data.get("test_paths", [])
        if not test_paths:
            print(f"  [WARN] No test_paths in cache, skipping pixel AUROC for {dataset_name}")
            del cached_data
            gc.collect()
            continue

        # Pre-load ALL GT masks once for this dataset (avoid re-loading per method)
        test_labels_ds = cached_data["test_labels"]
        print(f"  Pre-loading GT masks...")
        t_mask = time.time()
        gt_masks_preloaded, n_masks_found = preload_gt_masks(
            test_paths, test_labels_ds, dataset_name,
            eval_shape=PIXEL_AUROC_EVAL_SHAPE,
        )
        print(f"  Masks loaded: {n_masks_found} in {time.time()-t_mask:.1f}s")

        if n_masks_found == 0:
            print(f"  [WARN] No GT masks found for {dataset_name}, skipping pixel AUROC")
            del cached_data
            gc.collect()
            continue

        for seed in seeds:
            print(f"\n  seed={seed}")

            # Prepare data
            train_features, anomaly_indices, test_features, test_labels, _ = prepare_experiment_data(
                cached_data, train_limit, seed
            )

            n_test = test_features.shape[0]
            anomaly_features_pool = test_features[anomaly_indices]

            if len(anomaly_indices) == 0:
                print(f"    [WARN] No anomalies in test set, cannot contaminate. Skipping.")
                continue

            # Track clean baselines for delta/recovery
            clean_img_auroc = None
            clean_pix_auroc = None
            dirty_img_aurocs = {}
            dirty_pix_aurocs = {}

            # Precompute LOO once per contamination level
            loo_cache = {}  # {cont_rate: (loo_patch_scores, loo_image_scores)}

            for cont_rate in contamination_rates:
                if cont_rate == 0.0:
                    contaminated_features = train_features
                    is_contaminated = [0] * train_features.shape[0]
                else:
                    contaminated_features, is_contaminated = contaminate_train_features(
                        train_features, anomaly_features_pool, cont_rate, seed
                    )
                n_contaminated = sum(is_contaminated)
                config_tag = f"cont_{int(cont_rate*100)}pct"

                # LOO cache for this contamination level
                loo_patch_scores = None
                loo_image_scores = None
                none_scores_cache = None
                none_maps_cache = None

                for method_name in methods:
                    full_config = f"{config_tag}_{method_name}"

                    # Check if already done
                    key = (dataset_name, full_config, str(cont_rate), seed, str(train_limit))
                    if key in completed:
                        if method_name == "none":
                            # Need to compute baseline for downstream recovery
                            all_done = all(
                                (dataset_name, f"{config_tag}_{m}", str(cont_rate), seed, str(train_limit)) in completed
                                for m in methods
                            )
                            if all_done:
                                print(f"    {config_tag}: all methods SKIP")
                                break
                            # Compute baseline scores for downstream
                            scores, patch_maps = compute_knn_scores_with_maps(
                                contaminated_features, test_features, k=knn_k
                            )
                            metrics = compute_metrics(test_labels, scores)
                            pix_metrics = compute_pixel_auroc(
                                patch_maps, test_paths, test_labels, dataset_name,
                                patch_grid=patch_grid,
                                preloaded_masks=gt_masks_preloaded,
                                preloaded_n_masks=n_masks_found,
                            )
                            if cont_rate == 0.0:
                                clean_img_auroc = metrics["img_auroc"]
                                clean_pix_auroc = pix_metrics.get("pix_auroc")
                            dirty_img_aurocs[cont_rate] = metrics["img_auroc"]
                            dirty_pix_aurocs[cont_rate] = pix_metrics.get("pix_auroc")
                            none_scores_cache = scores
                            none_maps_cache = patch_maps
                            print(f"    {full_config}: SKIP (computed baseline for downstream)")
                            continue
                        print(f"    {full_config}: SKIP")
                        continue

                    try:
                        t0 = time.time()
                        n_patches_removed = 0
                        use_none = False

                        if method_name == "none":
                            bank_features = contaminated_features

                        elif method_name == "loo_patch":
                            if loo_patch_scores is None:
                                loo_patch_scores, loo_image_scores = compute_loo_consistency(
                                    contaminated_features, k=knn_k
                                )
                            bank_features, n_patches_removed = purify_loo_patch(
                                contaminated_features, loo_patch_scores, pct
                            )

                        elif method_name == "cosine_loo":
                            bank_features, n_patches_removed, _, _ = purify_cosine_loo(
                                contaminated_features, k=knn_k, percentile_threshold=pct
                            )

                        elif method_name == "ensemble_loo_mahal":
                            if loo_patch_scores is None:
                                loo_patch_scores, loo_image_scores = compute_loo_consistency(
                                    contaminated_features, k=knn_k
                                )
                            bank_features, n_patches_removed = purify_ensemble_loo_mahal(
                                contaminated_features, loo_patch_scores, pct
                            )

                        elif method_name == "oracle_patch":
                            bank_features, n_patches_removed = purify_oracle_patch(
                                contaminated_features, is_contaminated
                            )
                            if sum(is_contaminated) == 0:
                                use_none = True

                        else:
                            raise ValueError(f"Unknown pixel AUROC method: {method_name}")

                        # Score with maps
                        if use_none and none_scores_cache is not None:
                            scores = none_scores_cache
                            patch_maps = none_maps_cache
                        else:
                            n_bank = bank_features.shape[0] if bank_features.ndim == 2 \
                                else bank_features.reshape(-1, bank_features.shape[-1]).shape[0]
                            if n_bank < knn_k:
                                print(f"    [WARN] {full_config}: bank too small ({n_bank} < {knn_k}), skip")
                                continue

                            scores, patch_maps = compute_knn_scores_with_maps(
                                bank_features, test_features, k=knn_k
                            )
                            if method_name == "none":
                                none_scores_cache = scores
                                none_maps_cache = patch_maps

                        elapsed = time.time() - t0

                        # Compute image metrics
                        metrics = compute_metrics(test_labels, scores)

                        # Compute pixel metrics (uses pre-loaded masks for speed)
                        pix_metrics = compute_pixel_auroc(
                            patch_maps, test_paths, test_labels, dataset_name,
                            patch_grid=patch_grid,
                            preloaded_masks=gt_masks_preloaded,
                            preloaded_n_masks=n_masks_found,
                        )

                        # Track baselines
                        if cont_rate == 0.0 and method_name == "none":
                            clean_img_auroc = metrics["img_auroc"]
                            clean_pix_auroc = pix_metrics.get("pix_auroc")
                        if method_name == "none":
                            dirty_img_aurocs[cont_rate] = metrics["img_auroc"]
                            dirty_pix_aurocs[cont_rate] = pix_metrics.get("pix_auroc")

                        # Compute deltas and recovery
                        img_delta = ""
                        img_recovery = ""
                        pix_delta = ""
                        pix_recovery = ""

                        if clean_img_auroc is not None:
                            if not (cont_rate == 0.0 and method_name == "none"):
                                img_delta = f"{metrics['img_auroc'] - clean_img_auroc:.4f}"
                            if method_name != "none" and cont_rate > 0.0 and cont_rate in dirty_img_aurocs:
                                damage = clean_img_auroc - dirty_img_aurocs[cont_rate]
                                if abs(damage) > 1e-6:
                                    img_recovery = f"{(metrics['img_auroc'] - dirty_img_aurocs[cont_rate]) / damage:.4f}"
                                else:
                                    img_recovery = "1.0000"

                        if clean_pix_auroc is not None and pix_metrics.get("pix_auroc") is not None:
                            if not (cont_rate == 0.0 and method_name == "none"):
                                pix_delta = f"{pix_metrics['pix_auroc'] - clean_pix_auroc:.4f}"
                            if method_name != "none" and cont_rate > 0.0 and cont_rate in dirty_pix_aurocs:
                                dirty_pix = dirty_pix_aurocs[cont_rate]
                                if dirty_pix is not None:
                                    damage_pix = clean_pix_auroc - dirty_pix
                                    if abs(damage_pix) > 1e-6:
                                        pix_recovery = f"{(pix_metrics['pix_auroc'] - dirty_pix) / damage_pix:.4f}"
                                    else:
                                        pix_recovery = "1.0000"

                        result = {
                            "timestamp": datetime.now().isoformat(),
                            "dataset": dataset_name,
                            "config": full_config,
                            "contamination_rate": str(cont_rate),
                            "purification_method": method_name,
                            "train_limit": train_limit,
                            "n_train_original": contaminated_features.shape[0],
                            "n_contaminated": n_contaminated,
                            "n_patches_removed": n_patches_removed,
                            "n_test": n_test,
                            "seed": seed,
                            "knn_k": knn_k,
                            "percentile_threshold": pct,
                            "img_auroc": f"{metrics['img_auroc']:.4f}",
                            "pix_auroc": f"{pix_metrics['pix_auroc']:.4f}" if pix_metrics.get("pix_auroc") is not None else "",
                            "pix_aupr": f"{pix_metrics['pix_aupr']:.4f}" if pix_metrics.get("pix_aupr") is not None else "",
                            "n_masks_found": pix_metrics.get("n_masks_found", 0),
                            "auroc_delta_vs_clean": img_delta,
                            "recovery_rate": img_recovery,
                            "pix_auroc_delta_vs_clean": pix_delta,
                            "pix_recovery_rate": pix_recovery,
                            "inference_time_s": f"{elapsed:.2f}",
                        }
                        save_pixel_result(result, csv_path)

                        pix_str = f"pix={pix_metrics['pix_auroc']:.4f}" if pix_metrics.get("pix_auroc") is not None else "pix=N/A"
                        print(f"    {full_config}: img={metrics['img_auroc']:.3f} {pix_str} "
                              f"masks={pix_metrics.get('n_masks_found', 0)} "
                              f"patches_rm={n_patches_removed} {elapsed:.1f}s")

                    except Exception as e:
                        err_msg = f"{dataset_name} s{seed} {full_config}: {type(e).__name__}: {e}"
                        print(f"    [ERROR] {err_msg}")
                        log_error(dataset_name, full_config, seed, e)
                        all_errors.append(err_msg)
                        if args.debug:
                            traceback.print_exc()

        del cached_data
        gc.collect()

        ds_elapsed = time.time() - ds_start
        log(f"  {dataset_name} done in {ds_elapsed/60:.1f} min")

    print("\n" + "=" * 70)
    log(f"PIXEL AUROC DONE. Errors: {len(all_errors)}")
    for err in all_errors:
        print(f"  - {err}")


# =============================================================================
# DATA LEAKAGE VERIFICATION (split anomalies 50/50)
# =============================================================================

def run_leakage_check(args) -> None:
    """Data leakage verification: split test anomalies 50/50.

    Pool A = contamination source, Pool B = evaluation only.
    Runs on Quick Test datasets (6), 5 seeds, TL=10+20, cont=0+0.1+0.2+0.3.
    Core methods only. Writes to results_leakage_check.csv.
    """
    datasets = QUICK_TEST_DATASETS[:2] if args.debug else QUICK_TEST_DATASETS
    seeds = [0] if args.debug else SEEDS
    train_limits = [10] if args.debug else TRAIN_LIMITS
    contamination_rates = [0.0, 0.2] if args.debug else CONTAMINATION_RATES

    csv_path = OUTPUT_DIR / "results_leakage_check.csv"
    completed = get_completed_experiments(csv_path)

    methods = get_methods_for_tier(args.methods)

    total_experiments = (len(datasets) * len(seeds) * len(train_limits) *
                        len(contamination_rates) * len(methods))

    print("=" * 70)
    print("P3_002 LEAKAGE CHECK: Split anomalies 50/50 (contamination vs evaluation)")
    print(f"  Mode: {'DEBUG' if args.debug else 'FULL'} | Methods: {args.methods} ({len(methods)} methods)")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Seeds: {seeds}")
    print(f"  Train limits: {train_limits}")
    print(f"  Contamination rates: {contamination_rates}")
    print(f"  Methods: {methods}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Already completed: {len(completed)}")
    print(f"  Output: {csv_path}")
    print("=" * 70)

    all_errors = []

    for ds_idx, dataset_name in enumerate(datasets):
        ds_start = time.time()
        log(f"[{ds_idx+1}/{len(datasets)}] Dataset: {dataset_name}")

        try:
            cached_data = load_cached_dataset(dataset_name)
            print(f"  Loaded cache: train={cached_data['all_train_features'].shape}, "
                  f"test={cached_data['test_features'].shape}")
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            log_error(dataset_name, "cache_load", 0, e)
            all_errors.append(f"{dataset_name}: {e}")
            continue

        for seed in seeds:
            print(f"\n  seed={seed}")

            for train_limit in train_limits:
                print(f"  train_limit={train_limit}")
                try:
                    errors = run_leakage_experiment(
                        dataset_name, seed, train_limit,
                        knn_k=DEFAULT_KNN_K,
                        percentile_threshold=DEFAULT_PERCENTILE_THRESHOLD,
                        cached_data=cached_data,
                        completed=completed,
                        csv_path=csv_path,
                        contamination_rates=contamination_rates,
                        purification_methods=methods,
                        is_debug=args.debug,
                    )
                    all_errors.extend(errors)
                except Exception as e:
                    print(f"    [ERROR] Fatal: {e}")
                    log_error(dataset_name, "all", seed, e)
                    all_errors.append(f"{dataset_name}: {e}")
                    if args.debug:
                        traceback.print_exc()

        del cached_data
        gc.collect()

        ds_elapsed = time.time() - ds_start
        log(f"  {dataset_name} done in {ds_elapsed/60:.1f} min")

    print("\n" + "=" * 70)
    log(f"LEAKAGE CHECK DONE. Errors: {len(all_errors)}")
    for err in all_errors:
        print(f"  - {err}")


def run_leakage_experiment(
    dataset_name: str,
    seed: int,
    train_limit: int,
    knn_k: int,
    percentile_threshold: float,
    cached_data: dict,
    completed: set[tuple],
    csv_path: Path,
    contamination_rates: list[float],
    purification_methods: list[str],
    is_debug: bool = False,
) -> list[str]:
    """Like run_dataset_experiment but with 50/50 anomaly split.

    Split test anomalies into:
    - Pool A (50%): used ONLY as contamination source
    - Pool B (50%): used ONLY for evaluation (along with all normal test images)

    This eliminates data leakage: contaminating images never appear in the test set.
    """
    errors = []

    # Prepare data from cache
    train_features, anomaly_indices, test_features, test_labels, _ = prepare_experiment_data(
        cached_data, train_limit, seed
    )

    # Split anomaly indices 50/50 deterministically per seed
    rng_split = np.random.RandomState(seed + 5000)
    anomaly_indices_arr = np.array(anomaly_indices)
    rng_split.shuffle(anomaly_indices_arr)
    n_anom = len(anomaly_indices_arr)
    n_pool_a = n_anom // 2  # contamination source
    pool_a_indices = anomaly_indices_arr[:n_pool_a]  # for contamination
    pool_b_indices = anomaly_indices_arr[n_pool_a:]  # for evaluation

    # Build evaluation test set: all normals + pool_b anomalies
    normal_indices = np.where(test_labels == 0)[0]
    eval_indices = np.sort(np.concatenate([normal_indices, pool_b_indices]))
    eval_test_features = test_features[eval_indices]
    eval_test_labels = test_labels[eval_indices]

    n_test = eval_test_features.shape[0]
    n_normal_test = int((eval_test_labels == 0).sum())
    n_anomaly_test = int((eval_test_labels == 1).sum())

    # Contamination pool: only pool_a anomalies
    anomaly_features_pool = test_features[pool_a_indices]

    print(f"    LEAKAGE CHECK: {n_anom} anomalies -> pool_A={len(pool_a_indices)} (contam), pool_B={len(pool_b_indices)} (eval)")
    print(f"    n_train={train_features.shape[0]}, n_eval_test={n_test} (norm={n_normal_test}, anom={n_anomaly_test})")

    if len(pool_a_indices) == 0:
        print(f"    [WARN] No anomalies for contamination pool. Skipping.")
        return errors
    if n_anomaly_test == 0:
        print(f"    [WARN] No anomalies in evaluation set. Skipping.")
        return errors

    clean_auroc = None
    dirty_aurocs = {}

    for cont_rate in contamination_rates:
        if cont_rate == 0.0:
            contaminated_features = train_features
            is_contaminated = [0] * train_features.shape[0]
        else:
            contaminated_features, is_contaminated = contaminate_train_features(
                train_features, anomaly_features_pool, cont_rate, seed
            )
        n_contaminated = sum(is_contaminated)
        config_tag = f"cont_{int(cont_rate*100)}pct"

        loo_patch_scores = None
        loo_image_scores = None
        none_scores_cache = None

        for method_name in purification_methods:
            full_config = f"{config_tag}_{method_name}"

            key = (dataset_name, full_config, str(cont_rate), seed,
                   str(train_limit), str(knn_k), str(percentile_threshold))
            if key in completed:
                if method_name == "none":
                    all_done = all(
                        (dataset_name, f"{config_tag}_{m}", str(cont_rate), seed,
                         str(train_limit), str(knn_k), str(percentile_threshold)) in completed
                        for m in purification_methods
                    )
                    if all_done:
                        print(f"    {config_tag}: all methods SKIP (already done)")
                        break
                    none_scores_cache = compute_knn_scores(contaminated_features, eval_test_features, k=knn_k)
                    metrics_baseline = compute_metrics(eval_test_labels, none_scores_cache)
                    if cont_rate == 0.0:
                        clean_auroc = metrics_baseline["img_auroc"]
                    dirty_aurocs[cont_rate] = metrics_baseline["img_auroc"]
                    print(f"    {full_config}: SKIP (already done, computed AUROC for downstream)")
                    continue
                else:
                    print(f"    {full_config}: SKIP (already done)")
                    continue

            try:
                t0 = time.time()

                n_patches_removed = 0
                n_train_after = contaminated_features.shape[0]
                extra_info = {}
                use_none_scores = False

                # Apply purification (same logic as run_dataset_experiment)
                if method_name == "none":
                    bank_features = contaminated_features
                elif method_name == "loo_patch":
                    if loo_patch_scores is None:
                        loo_patch_scores, loo_image_scores = compute_loo_consistency(
                            contaminated_features, k=knn_k)
                    bank_features, n_patches_removed = purify_loo_patch(
                        contaminated_features, loo_patch_scores, percentile_threshold)
                    n_train_after = contaminated_features.shape[0]
                elif method_name == "cosine_loo":
                    cosine_scores, cosine_img_scores = compute_loo_consistency_cosine(
                        contaminated_features, k=knn_k)
                    bank_features, n_patches_removed = purify_loo_patch(
                        contaminated_features, cosine_scores, percentile_threshold)
                    n_train_after = contaminated_features.shape[0]
                    if loo_image_scores is None:
                        loo_image_scores = cosine_img_scores
                elif method_name == "mahalanobis_patch":
                    bank_features, n_patches_removed = purify_mahalanobis_patch(
                        contaminated_features, percentile_threshold)
                    n_train_after = contaminated_features.shape[0]
                elif method_name == "ensemble_loo_mahal":
                    if loo_patch_scores is None:
                        loo_patch_scores, loo_image_scores = compute_loo_consistency(
                            contaminated_features, k=knn_k)
                    bank_features, n_patches_removed = purify_ensemble_loo_mahal(
                        contaminated_features, loo_patch_scores, percentile_threshold)
                    n_train_after = contaminated_features.shape[0]
                elif method_name == "random_patch":
                    bank_features, n_patches_removed = purify_random_patch(
                        contaminated_features, percentile_threshold, seed)
                    n_train_after = contaminated_features.shape[0]
                elif method_name == "oracle_patch":
                    bank_features, n_patches_removed = purify_oracle_patch(
                        contaminated_features, is_contaminated)
                    n_train_after = contaminated_features.shape[0] - sum(is_contaminated)
                    if sum(is_contaminated) == 0:
                        use_none_scores = True
                else:
                    raise ValueError(f"Unknown method for leakage check: {method_name}")

                # Score using EVALUATION test set (no leakage)
                if use_none_scores and none_scores_cache is not None:
                    test_scores = none_scores_cache
                else:
                    # Guard: ensure bank has enough patches for KNN
                    n_bank_patches = bank_features.shape[0] if bank_features.ndim == 2 \
                        else bank_features.reshape(-1, bank_features.shape[-1]).shape[0]
                    if n_bank_patches < knn_k:
                        print(f"    [WARN] {full_config}: bank too small ({n_bank_patches} < {knn_k}), skip")
                        continue

                    test_scores = compute_knn_scores(bank_features, eval_test_features, k=knn_k)

                    if method_name == "none":
                        none_scores_cache = test_scores

                metrics = compute_metrics(eval_test_labels, test_scores)
                elapsed = time.time() - t0

                if method_name == "none":
                    if cont_rate == 0.0:
                        clean_auroc = metrics["img_auroc"]
                    dirty_aurocs[cont_rate] = metrics["img_auroc"]

                auroc_delta = None
                recovery = None
                if clean_auroc is not None and cont_rate > 0.0:
                    dirty_base = dirty_aurocs.get(cont_rate)
                    if dirty_base is not None:
                        auroc_delta = metrics["img_auroc"] - clean_auroc
                        damage = clean_auroc - dirty_base
                        recovery = (metrics["img_auroc"] - dirty_base) / damage if abs(damage) > 1e-9 else None

                # Contaminant detection AUROC
                det_auroc = None
                if method_name != "oracle_patch" and cont_rate > 0 and loo_image_scores is not None:
                    if sum(is_contaminated) > 0 and sum(is_contaminated) < len(is_contaminated):
                        det_auroc = float(roc_auc_score(is_contaminated, loo_image_scores))

                row = {
                    "timestamp": datetime.now().isoformat(),
                    "dataset": dataset_name,
                    "config": full_config,
                    "contamination_rate": cont_rate,
                    "purification_method": method_name,
                    "train_limit": train_limit,
                    "n_train_original": train_features.shape[0],
                    "n_train_after_purification": n_train_after,
                    "n_contaminated": n_contaminated,
                    "n_patches_removed": n_patches_removed,
                    "n_test": n_test,
                    "n_normal_test": n_normal_test,
                    "n_anomaly_test": n_anomaly_test,
                    "seed": seed,
                    "knn_k": knn_k,
                    "percentile_threshold": percentile_threshold,
                    **metrics,
                    "auroc_delta_vs_clean": auroc_delta,
                    "recovery_rate": recovery,
                    "contaminant_detection_auroc": det_auroc,
                    "contaminant_precision_at_1": "",
                    "contaminant_precision_at_k": "",
                    "contaminant_recall_at_k": "",
                    "adaptive_did_purify": extra_info.get("adaptive_did_purify", ""),
                    "adaptive_skewness": extra_info.get("adaptive_skewness", ""),
                    "inference_time_s": elapsed,
                }
                save_result(row, csv_path)
                completed.add(key)

                rec_str = f"rec={recovery:.1%}" if recovery is not None else ""
                delta_str = f"d={auroc_delta:+.4f}" if auroc_delta is not None else ""
                print(f"    {full_config}: AUROC={metrics['img_auroc']:.4f} {delta_str} {rec_str} "
                      f"rem={n_patches_removed} [{elapsed:.1f}s]")

            except Exception as e:
                print(f"    [ERROR] {full_config}: {e}")
                log_error(dataset_name, full_config, seed, e)
                errors.append(f"{dataset_name}/{full_config}: {e}")
                if is_debug:
                    traceback.print_exc()

    return errors


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    global SCRIPT_START_TIME, CACHE_DIR, OUTPUT_DIR
    SCRIPT_START_TIME = datetime.now()

    args = parse_args()

    # Set cache dir and output dir based on backbone
    if args.backbone == "clip":
        CACHE_DIR = Path("output/feature_cache_clip")
        OUTPUT_DIR = Path("output/exp_p3_002_full_clip")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[BACKBONE] CLIP ViT-L/14, cache: {CACHE_DIR}, output: {OUTPUT_DIR}")
        if not CACHE_DIR.exists() or not any(CACHE_DIR.iterdir()):
            print(f"\n[ERROR] CLIP feature cache not found at {CACHE_DIR}")
            print(f"  Run first: uv run precache_features.py --backbone clip")
            sys.exit(1)
    else:
        CACHE_DIR = Path("output/feature_cache")
        OUTPUT_DIR = Path("output/exp_p3_002_full")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.gate_check:
        run_gate_check(args)
    elif args.ablation == "percentile":
        run_ablation_percentile(args)
    elif args.ablation == "knn_k":
        run_ablation_knn_k(args)
    elif args.extra_cont:
        run_extra_cont(args)
    elif args.leakage_check:
        run_leakage_check(args)
    elif args.tl5:
        run_tl5(args)
    elif args.ablation_full_pct:
        run_ablation_full_pct(args)
    elif args.pixel_auroc:
        run_pixel_auroc(args)
    else:
        run_main_experiment(args)


if __name__ == "__main__":
    main()
