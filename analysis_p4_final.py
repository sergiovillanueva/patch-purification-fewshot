"""
FINAL analysis of the P4 image-level screening experiment over all 35 categories.

Produces the numbers for the rewritten paper:
  1. Sanity: P4 'none' must match the official leakage CSV (protocol fidelity).
  2. Setup B (honest): full method table at c=0.3, N=10 and N=20.
  3. Damage and best-method recovery across contamination rates (does it help at low c?).
  4. Per-benchmark breakdown at c=0.3, N=10.
  5. Setup A (shared pool): same table -> the protocol-inflation exhibit.
  6. Quality-gate view: TPR@FPR=1% for the default method vs baseline (Setup B).

Writes report to output/exp_p4_image_level/FINAL_ANALYSIS.txt and prints it.
"""
import csv
import statistics as st
from collections import defaultdict
from scipy.stats import wilcoxon

OUT = "output/exp_p4_image_level/FINAL_ANALYSIS.txt"
lines = []


def say(s=""):
    lines.append(s)
    print(s)


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


# ---------- load + dedup P4 ----------
raw = load("output/exp_p4_image_level/results.csv")
seen = set()
rows = []
for r in raw:
    k = (r["dataset"], r["setup"], r["method"], str(r["param_k"]),
         r["contamination_rate"], r["seed"], r["train_limit"])
    if k in seen:
        continue
    seen.add(k)
    rows.append(r)
say(f"P4 rows: {len(raw)} raw -> {len(rows)} unique after dedup")

LK = load("output/exp_p3_002_full/results_leakage_check.csv")   # Setup B official
V2 = load("output/exp_p3_002_full/results_v2.csv")               # Setup A official


def perds(src, filt, metric="img_auroc"):
    by = defaultdict(list)
    for r in src:
        if all(r.get(k) == v for k, v in filt.items()):
            try:
                by[r["dataset"]].append(float(r[metric]))
            except (ValueError, KeyError):
                pass
    return {d: st.mean(v) for d, v in by.items() if v}


def p4f(setup, method, par, c, tl):
    return {"setup": setup, "method": method, "param_k": str(par),
            "contamination_rate": c, "train_limit": tl}


def lkf(method, c, tl):
    return {"purification_method": method, "contamination_rate": c, "train_limit": tl}


# ---------- 1. sanity ----------
a = perds(rows, p4f("B", "none", "", "0.3", "10"))
b = perds(LK, lkf("none", "0.3", "10"))
common = sorted(set(a) & set(b))
mx = max(abs(a[d] - b[d]) for d in common)
# Threshold 1e-3: one cell (VisA/cashew seed 2) differs by ~2e-4 due to FAISS
# multi-thread reduction order on tied distances; everything else is exact.
say(f"\n[1] SANITY protocol fidelity: max |P4_none - leakage_none| over {len(common)} cats = {mx:.2e}"
    + ("  OK" if mx < 1e-3 else "  MISMATCH!!"))


def method_table(setup, tl, c, extra_lk=True):
    """Full comparison table for one (setup, tl, c)."""
    if setup == "B":
        clean = perds(rows, p4f("B", "none", "", "0.0", tl))
        dirty = perds(rows, p4f("B", "none", "", c, tl))
    else:
        clean = perds(rows, p4f("A", "none", "", "0.0", tl))
        dirty = perds(rows, p4f("A", "none", "", c, tl))
    com = sorted(set(clean) & set(dirty))
    dmg = st.mean([clean[d] - dirty[d] for d in com])
    dmg_p = wilcoxon([clean[d] - dirty[d] for d in com]).pvalue
    say(f"\n### Setup {setup} | N={tl} | c={c} | cats={len(com)} | "
        f"clean={st.mean([clean[d] for d in com]):.4f} dirty={st.mean([dirty[d] for d in com]):.4f} "
        f"damage={dmg:.4f} (p={dmg_p:.1e})")
    say(f"{'method':22s} {'AUROC':>7s} {'gain':>8s} {'recov%':>7s} {'p':>8s} {'wins':>6s} "
        f"{'cleanPen':>9s} {'remP':>5s} {'remR':>5s}")

    def show(name, m3, m0, pr=float("nan"), rc=float("nan")):
        cc = [d for d in com if d in m3]
        if not cc:
            say(f"{name:22s} (no data)")
            return
        diffs = [m3[d] - dirty[d] for d in cc]
        try:
            p = wilcoxon(diffs).pvalue
        except Exception:
            p = float("nan")
        pen = (st.mean([m0[d] - clean[d] for d in com if d in m0]) if m0 else float("nan"))
        say(f"{name:22s} {st.mean([m3[d] for d in cc]):7.4f} {st.mean(diffs):+8.4f} "
            f"{st.mean(diffs)/dmg*100:7.1f} {p:8.1e} {sum(1 for x in diffs if x>0):3d}/{len(cc):2d} "
            f"{pen:+9.4f} {pr:5.2f} {rc:5.2f}")

    def remstats(setup, meth, par, c, tl):
        P, R = [], []
        for r in rows:
            if (r["setup"] != setup or r["method"] != meth or str(r["param_k"]) != str(par)
                    or r["contamination_rate"] != c or r["train_limit"] != tl):
                continue
            if r["removal_precision"] not in ("", None):
                P.append(float(r["removal_precision"]))
            if r["removal_recall"] not in ("", None):
                R.append(float(r["removal_recall"]))
        return (st.mean(P) if P else float("nan"), st.mean(R) if R else float("nan"))

    for meth, par in [("imgmax_mad", "1.0"), ("imgmax_mad", "1.5"), ("imgmax_mad", "2.0"),
                      ("imgq99_mad", "1.5"), ("imgt5_mad", "1.5"),
                      ("imgmax_known_c", ""), ("image_auto_mad", "2.0"), ("image_known_c", "")]:
        m3 = perds(rows, p4f(setup, meth, par, c, tl))
        m0 = perds(rows, p4f(setup, meth, par, "0.0", tl))
        pr, rc = remstats(setup, meth, par, c, tl)
        show(f"{meth}{par}", m3, m0, pr, rc)

    if extra_lk:
        src = LK if setup == "B" else V2
        for lm, label in [("ensemble_loo_mahal", "patch_ensemble"), ("random_patch", "random_patch"),
                          ("oracle_patch", "oracle_images")]:
            m3 = perds(src, lkf(lm, c, tl))
            m0 = perds(src, lkf(lm, "0.0", tl))
            show(label, m3, m0)


# ---------- 2. Setup B main tables ----------
say("\n[2] SETUP B (honest evaluation)")
method_table("B", "10", "0.3")
method_table("B", "20", "0.3")

# ---------- 3. across contamination rates ----------
say("\n[3] Default rule across contamination rates (Setup B, N=10, imgmax_mad)")
for kk in ["1.0", "1.5"]:
    say(f"  --- k={kk} ---")
    clean = perds(rows, p4f("B", "none", "", "0.0", "10"))
    for c in ["0.1", "0.2", "0.3"]:
        dirty = perds(rows, p4f("B", "none", "", c, "10"))
        m = perds(rows, p4f("B", "imgmax_mad", kk, c, "10"))
        com = sorted(set(clean) & set(dirty) & set(m))
        dmg = st.mean([clean[d] - dirty[d] for d in com])
        diffs = [m[d] - dirty[d] for d in com]
        p = wilcoxon(diffs).pvalue
        say(f"  c={c}: damage={dmg:+.4f}  gain={st.mean(diffs):+.4f} "
            f"(recovery {st.mean(diffs)/dmg*100:5.1f}%)  p={p:.1e}")

# ---------- 4. per benchmark ----------
say("\n[4] Per benchmark (Setup B, N=10, c=0.3, imgmax_mad k=1.5 vs oracle)")
clean = perds(rows, p4f("B", "none", "", "0.0", "10"))
dirty = perds(rows, p4f("B", "none", "", "0.3", "10"))
m15 = perds(rows, p4f("B", "imgmax_mad", "1.5", "0.3", "10"))
orc = perds(LK, lkf("oracle_patch", "0.3", "10"))
bench = defaultdict(lambda: {"dmg": [], "g15": [], "gor": []})
for d in sorted(set(clean) & set(dirty) & set(m15) & set(orc)):
    bkey = d.split("/")[0]
    bench[bkey]["dmg"].append(clean[d] - dirty[d])
    bench[bkey]["g15"].append(m15[d] - dirty[d])
    bench[bkey]["gor"].append(orc[d] - dirty[d])
for bkey, v in sorted(bench.items()):
    dmg = st.mean(v["dmg"])
    say(f"  {bkey:16s} damage={dmg:+.4f}  k1.5 gain={st.mean(v['g15']):+.4f} "
        f"({(st.mean(v['g15'])/dmg*100 if abs(dmg)>1e-9 else float('nan')):5.1f}%)  "
        f"oracle gain={st.mean(v['gor']):+.4f} (n={len(v['dmg'])})")

# ---------- 5. Setup A exhibit ----------
say("\n[5] SETUP A (shared pool) - the protocol-inflation exhibit")
method_table("A", "10", "0.3")

# ---------- 6. quality gate ----------
say("\n[6] Quality gate TPR@FPR=1% (Setup B, N=10, c=0.3)")
for meth, par, label in [("none", "", "none"), ("imgmax_mad", "1.0", "imgmax_mad1.0"),
                         ("imgmax_mad", "1.5", "imgmax_mad1.5")]:
    t = perds(rows, p4f("B", meth, par, "0.3", "10"), metric="tpr_fpr1")
    t5 = perds(rows, p4f("B", meth, par, "0.3", "10"), metric="tpr_fpr5")
    say(f"  {label:16s} TPR@1%={st.mean(list(t.values())):.4f}  TPR@5%={st.mean(list(t5.values())):.4f}")
tc = perds(rows, p4f("B", "none", "", "0.0", "10"), metric="tpr_fpr1")
tc5 = perds(rows, p4f("B", "none", "", "0.0", "10"), metric="tpr_fpr5")
say(f"  {'clean reference':16s} TPR@1%={st.mean(list(tc.values())):.4f}  TPR@5%={st.mean(list(tc5.values())):.4f}")

# ---------- 7. operational removal behavior ----------
say("\n[7] Operational behavior: mean images removed per bank (Setup B, N=10)")
for meth, par in [("imgmax_mad", "1.0"), ("imgmax_mad", "1.5"), ("imgmax_mad", "2.0")]:
    for c in ["0.0", "0.3"]:
        vals = [float(r["n_images_removed"]) for r in rows
                if r["setup"] == "B" and r["method"] == meth and str(r["param_k"]) == par
                and r["contamination_rate"] == c and r["train_limit"] == "10"]
        if vals:
            say(f"  {meth} k={par} c={c}: removed {st.mean(vals):.2f} of 10 images "
                f"(contaminated present: {3 if c=='0.3' else 0})")

# ---------- 8. bootstrap CI + Holm over the k grid ----------
say("\n[8] Bootstrap 95% CI of the gain and Holm correction over k grid "
    "(Setup B, c=0.3, gain vs none, per-category units)")
import random
random.seed(0)
for tl in ["10", "20"]:
    clean = perds(rows, p4f("B", "none", "", "0.0", tl))
    dirty = perds(rows, p4f("B", "none", "", "0.3", tl))
    com = sorted(set(clean) & set(dirty))
    pvals = []
    stats_rows = []
    for kk in ["1.0", "1.5", "2.0"]:
        m3 = perds(rows, p4f("B", "imgmax_mad", kk, "0.3", tl))
        cc = [d for d in com if d in m3]
        diffs = [m3[d] - dirty[d] for d in cc]
        boots = []
        for _ in range(10000):
            sample = [random.choice(diffs) for _ in diffs]
            boots.append(st.mean(sample))
        boots.sort()
        lo, hi = boots[249], boots[9749]
        p = wilcoxon(diffs).pvalue
        pvals.append(p)
        stats_rows.append((kk, st.mean(diffs), lo, hi, p, sum(1 for x in diffs if x > 0), len(diffs)))
    # Holm step-down with the cumulative-maximum enforcement: the adjusted
    # p-values must be non-decreasing in the ordering of the raw p-values.
    order = sorted(range(3), key=lambda i: pvals[i])
    holm = [None] * 3
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, pvals[i] * (3 - rank))
        holm[i] = min(1.0, running)
    say(f"  N={tl}:")
    for (kk, g, lo, hi, p, w, n), ph in zip(stats_rows, holm):
        say(f"    k={kk}: gain={g:+.4f}  CI95=[{lo:+.4f},{hi:+.4f}]  p={p:.1e}  "
            f"Holm-p={ph:.1e}  wins={w}/{n}")

# ---------- 9. AUPR (within Setup B only; prevalence differs across setups) ----------
say("\n[9] AUPR within Setup B (N=10, c=0.3)")
for meth, par, label in [("none", "", "none"), ("imgmax_mad", "1.5", "imgmax_mad1.5")]:
    v = perds(rows, p4f("B", meth, par, "0.3", "10"), metric="img_aupr")
    say(f"  {label:16s} AUPR={st.mean(list(v.values())):.4f} (n={len(v)})")
vc = perds(rows, p4f("B", "none", "", "0.0", "10"), metric="img_aupr")
say(f"  {'clean reference':16s} AUPR={st.mean(list(vc.values())):.4f}")

# ---------- 10. N=5 (when the chain run lands) ----------
say("\n[10] N=5 under Setup B (c=0.3)")
try:
    method_table("B", "5", "0.3", extra_lk=False)
except Exception as e:
    say(f"  (no data yet: {e})")

# ---------- 11. CLIP backbone (when the chain run lands) ----------
say("\n[11] CLIP backbone, Setup B (N=10, c=0.3)")
try:
    CLIP = load("output/exp_p4_image_level/results_clip.csv")
    seenc = set()
    crows = []
    for r in CLIP:
        k = (r["dataset"], r["setup"], r["method"], str(r["param_k"]),
             r["contamination_rate"], r["seed"], r["train_limit"])
        if k in seenc:
            continue
        seenc.add(k)
        crows.append(r)
    cclean = perds(crows, p4f("B", "none", "", "0.0", "10"))
    cdirty = perds(crows, p4f("B", "none", "", "0.3", "10"))
    com = sorted(set(cclean) & set(cdirty))
    dmg = st.mean([cclean[d] - cdirty[d] for d in com])
    say(f"  cats={len(com)} clean={st.mean([cclean[d] for d in com]):.4f} "
        f"dirty={st.mean([cdirty[d] for d in com]):.4f} damage={dmg:.4f}")
    for kk in ["1.0", "1.5", "2.0"]:
        m3 = perds(crows, p4f("B", "imgmax_mad", kk, "0.3", "10"))
        cc = [d for d in com if d in m3]
        if not cc:
            continue
        diffs = [m3[d] - cdirty[d] for d in cc]
        p = wilcoxon(diffs).pvalue
        say(f"  imgmax_mad k={kk}: gain={st.mean(diffs):+.4f} "
            f"(recovery {st.mean(diffs)/dmg*100:.1f}%) p={p:.1e} n={len(cc)}")
except FileNotFoundError:
    say("  (results_clip.csv not present yet)")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"\n[saved to {OUT}]")
