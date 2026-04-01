"""
Run LOF (SoftPatch-style) scoring on all 35 datasets.
Appends results to the existing results CSV.

This script reuses the experiment infrastructure from run_experiments.py
but only runs the 'lof_patch' method (+ 'none' for baselines, which will be skipped
since they already exist in the CSV).

Usage:
    python run_lof_baseline.py            # Full run: 35 datasets, 5 seeds, TL=10,20
    python run_lof_baseline.py --debug    # Debug: 2 datasets, 1 seed, TL=10
"""
import os
import sys
import gc
import time
import csv
import warnings
from datetime import datetime
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np

# Import everything we need from the main experiment module
from run_experiments import (
    ALL_DATASETS, QUICK_TEST_DATASETS, SEEDS, TRAIN_LIMITS, CONTAMINATION_RATES,
    DEFAULT_KNN_K, DEFAULT_PERCENTILE_THRESHOLD,
    FIELDNAMES,
    load_cached_dataset, prepare_experiment_data,
    contaminate_train_features, purify_lof_patch,
    compute_knn_scores, compute_metrics,
    get_completed_experiments, save_result,
)

# Configuration
CACHE_DIR = Path("output/feature_cache")
OUTPUT_DIR = Path("output/exp_p3_002_full")
CSV_PATH = OUTPUT_DIR / "results_v2.csv"
METHODS = ["none", "lof_patch"]  # none needed for baseline, will be skipped if already done


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run LOF-only experiment")
    parser.add_argument("--debug", action="store_true", help="Debug: 2 datasets, 1 seed, TL=10")
    args = parser.parse_args()

    datasets = QUICK_TEST_DATASETS[:2] if args.debug else ALL_DATASETS
    seeds = [0] if args.debug else SEEDS
    train_limits = [10] if args.debug else TRAIN_LIMITS
    contamination_rates = [0.0, 0.2] if args.debug else CONTAMINATION_RATES

    # Load completed experiments for skip logic
    completed = get_completed_experiments(CSV_PATH)

    total_needed = len(datasets) * len(seeds) * len(train_limits) * len(contamination_rates) * len(METHODS)
    already_done = 0
    for ds in datasets:
        for seed in seeds:
            for tl in train_limits:
                for cr in contamination_rates:
                    for m in METHODS:
                        key = (ds, f"cont_{int(cr*100)}pct_{m}", str(cr), seed, str(tl),
                               str(DEFAULT_KNN_K), str(DEFAULT_PERCENTILE_THRESHOLD))
                        if key in completed:
                            already_done += 1

    print("=" * 70)
    print("LOF-ONLY EXPERIMENT (SoftPatch baseline for paper tables)")
    print(f"  Mode: {'DEBUG' if args.debug else 'FULL'}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Seeds: {seeds}")
    print(f"  Train limits: {train_limits}")
    print(f"  Contamination rates: {contamination_rates}")
    print(f"  Methods: {METHODS}")
    print(f"  Total experiments: {total_needed}")
    print(f"  Already completed: {already_done}")
    print(f"  Remaining: {total_needed - already_done}")
    print(f"  Output: {CSV_PATH}")
    print("=" * 70)
    sys.stdout.flush()

    start_time = time.time()
    n_completed = 0
    n_errors = 0

    for ds_idx, dataset_name in enumerate(datasets):
        ds_start = time.time()
        elapsed_h = (time.time() - start_time) / 3600
        print(f"\n[{ds_idx+1}/{len(datasets)}] [{elapsed_h:.1f}h] Dataset: {dataset_name}")
        sys.stdout.flush()

        try:
            cached_data = load_cached_dataset(dataset_name)
            print(f"  Loaded cache: train={cached_data['all_train_features'].shape}, "
                  f"test={cached_data['test_features'].shape}")
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            n_errors += 1
            continue

        for seed in seeds:
            for train_limit in train_limits:
                # Prepare data
                train_features, anomaly_indices, test_features, test_labels, _ = \
                    prepare_experiment_data(cached_data, train_limit, seed)

                if len(anomaly_indices) == 0:
                    print(f"  [WARN] seed={seed} tl={train_limit}: no anomalies, skip")
                    continue

                anomaly_features_pool = test_features[anomaly_indices]
                n_test = test_features.shape[0]
                n_normal_test = int((test_labels == 0).sum())
                n_anomaly_test = int((test_labels == 1).sum())

                # Track baselines per contamination rate
                clean_auroc = None
                dirty_aurocs = {}

                for cont_rate in contamination_rates:
                    # Contaminate
                    if cont_rate == 0.0:
                        contaminated_features = train_features
                        is_contaminated = [0] * train_features.shape[0]
                    else:
                        contaminated_features, is_contaminated = contaminate_train_features(
                            train_features, anomaly_features_pool, cont_rate, seed
                        )
                    n_contaminated = sum(is_contaminated)

                    for method_name in METHODS:
                        config_tag = f"cont_{int(cont_rate*100)}pct"
                        full_config = f"{config_tag}_{method_name}"

                        # Skip if already done
                        key = (dataset_name, full_config, str(cont_rate), seed,
                               str(train_limit), str(DEFAULT_KNN_K), str(DEFAULT_PERCENTILE_THRESHOLD))
                        if key in completed:
                            # But we still need baseline AUROCs for recovery computation
                            if method_name == "none":
                                scores = compute_knn_scores(contaminated_features, test_features, k=DEFAULT_KNN_K)
                                metrics = compute_metrics(test_labels, scores)
                                if cont_rate == 0.0:
                                    clean_auroc = metrics["img_auroc"]
                                dirty_aurocs[cont_rate] = metrics["img_auroc"]
                            continue

                        try:
                            t0 = time.time()

                            if method_name == "none":
                                bank_features = contaminated_features
                                n_patches_removed = 0
                            elif method_name == "lof_patch":
                                bank_features, n_patches_removed = purify_lof_patch(
                                    contaminated_features, DEFAULT_PERCENTILE_THRESHOLD
                                )
                            else:
                                continue

                            # Score test set
                            scores = compute_knn_scores(bank_features, test_features, k=DEFAULT_KNN_K)
                            elapsed = time.time() - t0

                            metrics = compute_metrics(test_labels, scores)

                            # Track baselines
                            if method_name == "none" and cont_rate == 0.0:
                                clean_auroc = metrics["img_auroc"]
                            if method_name == "none":
                                dirty_aurocs[cont_rate] = metrics["img_auroc"]

                            # Compute delta and recovery
                            auroc_delta = ""
                            recovery_rate = ""
                            if clean_auroc is not None:
                                if not (cont_rate == 0.0 and method_name == "none"):
                                    auroc_delta = f"{metrics['img_auroc'] - clean_auroc:.4f}"
                                if method_name != "none" and cont_rate > 0.0 and cont_rate in dirty_aurocs:
                                    damage = clean_auroc - dirty_aurocs[cont_rate]
                                    if abs(damage) > 1e-6:
                                        recovery_rate = f"{(metrics['img_auroc'] - dirty_aurocs[cont_rate]) / damage:.4f}"

                            result = {
                                "timestamp": datetime.now().isoformat(),
                                "dataset": dataset_name,
                                "config": full_config,
                                "contamination_rate": str(cont_rate),
                                "purification_method": method_name,
                                "train_limit": train_limit,
                                "n_train_original": contaminated_features.shape[0],
                                "n_train_after_purification": contaminated_features.shape[0],
                                "n_contaminated": n_contaminated,
                                "n_patches_removed": n_patches_removed,
                                "n_test": n_test,
                                "n_normal_test": n_normal_test,
                                "n_anomaly_test": n_anomaly_test,
                                "seed": seed,
                                "knn_k": DEFAULT_KNN_K,
                                "percentile_threshold": DEFAULT_PERCENTILE_THRESHOLD,
                                "img_auroc": f"{metrics['img_auroc']:.4f}",
                                "img_aupr": f"{metrics['img_aupr']:.4f}",
                                "img_f1max": f"{metrics['img_f1max']:.4f}",
                                "tpr_fpr1": f"{metrics['tpr_fpr1']:.4f}",
                                "tpr_fpr5": f"{metrics['tpr_fpr5']:.4f}",
                                "auroc_delta_vs_clean": auroc_delta,
                                "recovery_rate": recovery_rate,
                                "contaminant_detection_auroc": "",
                                "contaminant_precision_at_1": "",
                                "contaminant_precision_at_k": "",
                                "contaminant_recall_at_k": "",
                                "adaptive_did_purify": "",
                                "adaptive_skewness": "",
                                "inference_time_s": f"{elapsed:.2f}",
                            }
                            save_result(result, CSV_PATH)
                            completed.add(key)
                            n_completed += 1

                            rec_str = f" rec={recovery_rate}" if recovery_rate else ""
                            print(f"    {full_config}: AUROC={metrics['img_auroc']:.3f} "
                                  f"removed={n_patches_removed}{rec_str} [{elapsed:.1f}s]")
                            sys.stdout.flush()

                        except Exception as e:
                            print(f"    [ERROR] {full_config}: {e}")
                            n_errors += 1

        # Free memory
        del cached_data
        gc.collect()

        ds_elapsed = time.time() - ds_start
        print(f"  {dataset_name} done in {ds_elapsed/60:.1f} min")
        sys.stdout.flush()

    total_elapsed = (time.time() - start_time) / 3600
    print(f"\n{'=' * 70}")
    print(f"DONE. Completed: {n_completed}, Errors: {n_errors}, Time: {total_elapsed:.1f}h")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
