# Training-free reference auditing for few-shot industrial visual inspection

Code and pre-computed results for the paper:

> **Cleaning Contaminated Reference Sets for Few-Shot Visual Anomaly
> Detection**
> Sergio Villanueva Lopez, Emilio Soria-Olivas, Manuel Sanchez-Montanes
> (under review, 2026)

Few-shot memory-bank inspection systems are commissioned on the factory floor
from 5 to 20 reference images of conforming parts. If a defective or atypical
part slips into that set (a displaced label, a smudge, an operator's hand in the
frame), it shifts the quality standard of the whole station. This repository
contains the full evaluation of that damage and of a **training-free audit** that
flags suspicious reference images at commissioning so they can be removed,
reviewed by the operator, or recaptured.

The audit scores each reference image by the **largest leave-one-out patch
distance** it contains (defects are localized, so the worst patch carries the
signal while averaging over patches dilutes it) and flags images whose score
exceeds a robust threshold (`median + 1.5 * 1.4826 * MAD`).

## Key results

35 categories from MVTec AD, VisA, BTAD and MVTec LOCO, DINOv3 features, a
protocol with **disjoint** contamination and test pools, paired Wilcoxon over
categories (each category is one unit, seeds averaged first).

Damage of contamination: **0.6 to 2.1 AUROC points** for 10 to 20 references.

Acting on the flags (Setup B, N=10, c=0.3, DINOv3 detector):

| Intervention | Mean AUROC | Damage recovered | p |
|---|---|---|---|
| Contaminated, no action | 0.8415 | reference | reference |
| Remove flagged images | 0.8480 | 37% | 2.0e-3 |
| Recapture flagged images | 0.8517 | 58% | 1.2e-4 |
| Oracle removal (ceiling of deletion) | 0.8518 | 59% | 3.0e-3 |
| Clean reference set | 0.8591 | 100% | reference |

The flagging has 0.94 removal precision and 0.52 recall, and its mean cost on
**clean** reference sets is statistically indistinguishable from zero
(+0.0009, 95% CI [-0.0018, +0.0033]).

Two further findings:

- **The auditor can be decoupled from the detector.** A fixed DINOv3 auditor
  applied to banks built on other backbones improves them by more than each
  detector's own self-audit does: +0.0080 over self-audit on a DINOv2 bank
  (p=4.6e-5) and +0.0052 on a CLIP bank (p=6.7e-3). A plant can keep its
  production detector and still audit its references with a spatially fine
  encoder.

- **Evaluation protocol matters.** Drawing contaminants from the same pool used
  for testing (a "shared pool" design) overstates the damage by a factor of 2.7
  (4.68 vs 1.76 AUROC points) and turns a null patch-level filtering result
  (-5%, p=0.59 under the disjoint protocol) into an apparent 51% recovery
  (p=1.9e-7). All headline results here use the disjoint protocol; we report the
  shared-pool comparison as a caution for future work.

## Repository layout

Scripts:

| File | Role |
|------|------|
| `precache_features.py` | Feature extraction (DINOv3 / CLIP / DINOv2) to `.npy` caches (GPU) |
| `exp_p3_002_dirty_fewshot_full.py` | Core harness: protocols, contamination, patch-level baselines, metrics |
| `exp_p4_image_level.py` | Image-level audit under both protocols (aggregations, thresholds; resumable) |
| `exp_p5_assurance.py` | Decoupled auditor, recapture, review budget, global-embedding baselines |
| `analysis_p4_final.py` | Reproduces the damage / screening / protocol numbers |
| `analysis_p5.py` | Reproduces the transfer, recapture and review-budget numbers |

Pre-computed results (`;`-separated CSVs; load with `pandas.read_csv(path, sep=";")`):

| File | Content |
|------|---------|
| `output/exp_p3_002_full/results_v2.csv` | Shared-pool protocol (Setup A) |
| `output/exp_p3_002_full/results_leakage_check.csv` | Disjoint-pool protocol (Setup B), patch baselines and oracle |
| `output/exp_p4_image_level/results.csv` | Image-level audit, DINOv3 |
| `output/exp_p4_image_level/results_clip.csv` | Image-level audit, CLIP |
| `output/exp_p4_image_level/results_dinov2.csv` | Image-level audit, DINOv2 |
| `output/exp_p4_image_level/FINAL_ANALYSIS.txt` | Full text report produced by `analysis_p4_final.py` |
| `output/exp_p5_assurance/results.csv` | Assurance experiments (transfer, recapture, review, baselines) |

## Reproducing the paper numbers

**Without datasets or a GPU.** Every number in the paper is reproduced from the
pre-computed CSVs above. These two scripts need only `numpy` and `scipy`:

```bash
python analysis_p4_final.py   # damage, screening, protocol-inflation exhibit
python analysis_p5.py         # auditor transfer, recapture, review budget
```

**From scratch.**

1. Download the datasets (MVTec AD, VisA, BTAD, MVTec LOCO) into `data/`.
2. `python precache_features.py` (GPU; also `--backbone clip` and `--backbone dinov2`).
3. `python exp_p3_002_dirty_fewshot_full.py` then `python exp_p4_image_level.py --setup both`
   and `python exp_p5_assurance.py` (CPU only; all resumable).
4. `python analysis_p4_final.py` and `python analysis_p5.py`.

## Requirements

- Python 3.11
- The analysis scripts need only `numpy` and `scipy` (see `requirements.txt`).
- Feature extraction additionally needs `torch`, `torchvision` and
  `transformers` (install PyTorch for your hardware from
  https://pytorch.org/get-started/locally/), plus a CUDA GPU.
- `faiss-cpu` is optional and only speeds up nearest-neighbor search; the code
  falls back to `scikit-learn` without it.

```bash
pip install -r requirements.txt
```

## Datasets

Download and extract into a `data/` directory (not redistributed here; each has
its own license):

| Dataset | Categories | Source |
|---------|-----------|--------|
| MVTec AD | 15 | https://www.mvtec.com/company/research/datasets/mvtec-ad |
| VisA | 12 | https://github.com/amazon-science/spot-diff |
| BTAD | 3 | https://github.com/pankajmishra000/VT-ADL |
| MVTec LOCO AD | 5 | https://www.mvtec.com/company/research/datasets/mvtec-loco |

## A note on the earlier version of this repository

Earlier commits described a patch-level purification method with a 51% recovery
headline. That number was an artifact of a shared-pool evaluation, in which
inserted contaminants are drawn from the same pool used at test time and match
themselves. We identified this ourselves, switched to a disjoint-pool protocol,
and rebuilt the study around the image-level audit reported here. The data that
expose the artifact are in `results_v2.csv` (shared pool) and
`results_leakage_check.csv` (disjoint pool) and were produced and published by
us.

## License

MIT. See [LICENSE](LICENSE).
