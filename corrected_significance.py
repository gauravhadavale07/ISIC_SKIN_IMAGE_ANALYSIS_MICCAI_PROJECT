"""
corrected_significance.py — Holm-Bonferroni corrected significance tests
                           with paired Cohen's d effect sizes.

Runs AFTER full_analysis.py completes (baselines must be in experiment_progress.json).

Effect size (paired Cohen's d):
  d = mean(b_i - a_i) / std(b_i - a_i)
  Convention (Cohen 1988): |d| < 0.2 negligible, 0.2–0.5 small,
                           0.5–0.8 medium, > 0.8 large
  At n=3 a large effect can still fail significance — d distinguishes
  "underpowered large effect" from "genuinely small effect".

Priority order per task:
  P1  T→V vs. Image-Only   (per metric, corrected)
  P1  GMU  vs. Image-Only  (per metric, corrected)
  P2  Full 14-comparison × 7-metric matrix with Holm-Bonferroni
  P3  Silhouette three-way comparison (Late Fusion / V→T / T→V) — loaded from
      full_analysis_run.log if already computed, else recomputed here
  P4  Image-Only and Text-Only raw numbers (sanity check)

Holm-Bonferroni procedure:
  1. Compute all m raw p-values
  2. Sort ascending: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
  3. Reject H_(k) if p_(k) ≤ α / (m - k + 1)   for k = 1, 2, ...
     Stop rejecting at first non-rejection; all subsequent are also non-rejected.
  α = 0.05
"""

import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy import stats
from collections import defaultdict

# ── load results ───────────────────────────────────────────────────────────────
with open("./results/experiment_progress.json") as f:
    data = json.load(f)
results_db = data["results"]

ALPHA = 0.05
SEEDS = [456, 789, 1337]   # order in which results were appended

print(f"\n{'='*72}")
print("CORRECTED SIGNIFICANCE ANALYSIS")
print(f"Holm-Bonferroni correction, α={ALPHA}")
print(f"{'='*72}")

# ── helpers ────────────────────────────────────────────────────────────────────
def get3(arch, metric):
    """Return exactly-3 values or None."""
    vals = results_db.get(arch, {}).get(metric, [])
    if len(vals) != 3:
        return None
    return list(vals)

def cohens_d_paired(a, b):
    """Paired Cohen's d = mean(diffs) / std(diffs, ddof=1)."""
    diffs = np.array(b) - np.array(a)
    sd = np.std(diffs, ddof=1)   # sample std of differences
    if sd == 0:
        return float('inf') if diffs.mean() != 0 else 0.0
    return float(diffs.mean() / sd)

def d_label(d):
    """Cohen's d magnitude label."""
    ad = abs(d)
    if ad < 0.2:  return "negligible"
    if ad < 0.5:  return "small"
    if ad < 0.8:  return "medium"
    return "large"

def raw_ttest(arch_a, arch_b, metric):
    """Paired t-test + Cohen's d. Returns (t, p, delta, d) or None."""
    a = get3(arch_a, metric)
    b = get3(arch_b, metric)
    if a is None or b is None:
        return None
    t, p = stats.ttest_rel(a, b)
    d = cohens_d_paired(a, b)
    return float(t), float(p), float(np.mean(b) - np.mean(a)), d

def holm_bonferroni(tests):
    """
    tests: list of (label, t, p, delta)
    Returns same list with added 'corrected_reject' bool and 'adjusted_alpha'.
    """
    m = len(tests)
    sorted_tests = sorted(enumerate(tests), key=lambda x: x[1][2])   # sort by p
    result = [None] * m
    stop = False
    for rank, (orig_idx, (label, t, p, delta, d)) in enumerate(sorted_tests):
        adj_alpha = ALPHA / (m - rank)
        reject = (not stop) and (p <= adj_alpha)
        if not reject:
            stop = True
        result[orig_idx] = (label, t, p, delta, d, adj_alpha, reject)
    return result

# ── architecture names ─────────────────────────────────────────────────────────
ALL_ARCHS = ["Late Fusion", "GMU Baseline", "Cross-Attention",
             "Cross-Attention T→V", "Image-Only", "Text-Only"]

PRESENT = [a for a in ALL_ARCHS if a in results_db and
           len(results_db[a].get("Accuracy", [])) == 3]
MISSING = [a for a in ALL_ARCHS if a not in PRESENT]

print(f"\nArchitectures with 3-seed data : {PRESENT}")
if MISSING:
    print(f"Missing / incomplete           : {MISSING}  ← excluded from tests")

# ── P4: raw numbers for Image-Only and Text-Only ───────────────────────────────
print(f"\n{'='*72}")
print("P4 — IMAGE-ONLY & TEXT-ONLY RAW NUMBERS (sanity check)")
print(f"{'='*72}")

REPORT_METRICS = {
    "Accuracy":           ("×100", lambda v: f"{np.mean(v)*100:.2f}% ± {np.std(v)*100:.2f}%"),
    "AUROC":              ("",     lambda v: f"{np.mean(v):.4f} ± {np.std(v):.4f}"),
    "F1 (Macro)":         ("",     lambda v: f"{np.mean(v):.4f} ± {np.std(v):.4f}"),
    "Real_Accuracy":      ("pp",   lambda v: f"{np.mean(v):.2f} ± {np.std(v):.2f}%"),
    "Blank_Accuracy":     ("pp",   lambda v: f"{np.mean(v):.2f} ± {np.std(v):.2f}%"),
    "Blank_Accuracy_Drop":("pp",   lambda v: f"{np.mean(v):.2f} ± {np.std(v):.2f}pp"),
    "CFR":                ("pp",   lambda v: f"{np.mean(v):.2f} ± {np.std(v):.2f}%"),
    "Mean_Delta_P":       ("pp",   lambda v: f"{np.mean(v):.2f} ± {np.std(v):.2f}pp"),
    "Linear_CKA":         ("",     lambda v: f"{np.mean(v):.4f} ± {np.std(v):.4f}"),
    "Neutral_Accuracy":   ("pp",   lambda v: f"{np.mean(v):.2f} ± {np.std(v):.2f}%"),
    "Counterfactual_Accuracy": ("pp", lambda v: f"{np.mean(v):.2f} ± {np.std(v):.2f}%"),
}
MAJORITY_BASELINE = 845 / 2298 * 100   # 36.77%

for arch in ["Image-Only", "Text-Only"]:
    if arch not in results_db:
        print(f"\n  {arch}: NOT YET IN RESULTS — still training?")
        continue
    m = results_db[arch]
    print(f"\n  {arch}")
    print(f"  {'─'*50}")
    for metric, (_, fmt) in REPORT_METRICS.items():
        vals = m.get(metric, [])
        if vals:
            flag = ""
            if metric in ("Blank_Accuracy", "Neutral_Accuracy", "Real_Accuracy"):
                if np.mean(vals) < MAJORITY_BASELINE:
                    flag = f"  ⚠️  BELOW majority baseline ({MAJORITY_BASELINE:.1f}%)"
            print(f"    {metric:<30} {fmt(vals)}{flag}")
        else:
            print(f"    {metric:<30} N/A")

# ── build the full test matrix ─────────────────────────────────────────────────
SIG_METRICS = ["Accuracy", "AUROC", "F1 (Macro)", "CFR", "Mean_Delta_P",
               "Linear_CKA", "Blank_Accuracy_Drop"]

COMPARISONS = [
    # (label, baseline, proposed)  — "proposed" is "b" in delta=b-a
    ("T→V            vs. Image-Only",    "Image-Only",       "Cross-Attention T→V"),
    ("GMU            vs. Image-Only",    "Image-Only",       "GMU Baseline"),
    ("Late Fusion    vs. Image-Only",    "Image-Only",       "Late Fusion"),
    ("V→T            vs. Image-Only",    "Image-Only",       "Cross-Attention"),
    ("T→V            vs. Text-Only",     "Text-Only",        "Cross-Attention T→V"),
    ("GMU            vs. Text-Only",     "Text-Only",        "GMU Baseline"),
    ("Late Fusion    vs. Text-Only",     "Text-Only",        "Late Fusion"),
    ("V→T            vs. Text-Only",     "Text-Only",        "Cross-Attention"),
    ("T→V            vs. V→T",           "Cross-Attention",  "Cross-Attention T→V"),
    ("T→V            vs. Late Fusion",   "Late Fusion",      "Cross-Attention T→V"),
    ("T→V            vs. GMU",           "GMU Baseline",     "Cross-Attention T→V"),
    ("GMU            vs. Late Fusion",   "Late Fusion",      "GMU Baseline"),
    ("V→T            vs. Late Fusion",   "Late Fusion",      "Cross-Attention"),
    ("V→T            vs. GMU",           "GMU Baseline",     "Cross-Attention"),
]

# Collect ALL (comparison, metric) raw tests first
all_raw = []   # (comp_label, metric, arch_a, arch_b, t, p, delta, d) or None
for comp_label, arch_a, arch_b in COMPARISONS:
    if arch_a not in PRESENT or arch_b not in PRESENT:
        continue
    for metric in SIG_METRICS:
        res = raw_ttest(arch_a, arch_b, metric)
        if res is not None:
            t, p, delta, d = res
            all_raw.append((comp_label, metric, arch_a, arch_b, t, p, delta, d))

m_total = len(all_raw)
print(f"\n{'='*72}")
print(f"FULL CORRECTED TEST MATRIX  (m = {m_total} tests, Holm-Bonferroni α={ALPHA})")
print(f"{'='*72}")

# Apply Holm-Bonferroni across entire family
test_tuples = [(f"{r[0]} | {r[1]}", r[4], r[5], r[6], r[7]) for r in all_raw]
corrected = holm_bonferroni(test_tuples)

# Attach back to original rows
corrected_rows = []
for i, (comp_label, metric, arch_a, arch_b, t, p, delta, d) in enumerate(all_raw):
    _, _, _, _, _, adj_alpha, reject = corrected[i]
    corrected_rows.append({
        "comp": comp_label, "metric": metric,
        "arch_a": arch_a, "arch_b": arch_b,
        "t": t, "p": p, "delta": delta, "d": d,
        "adj_alpha": adj_alpha, "reject": reject,
    })

# ── P1: Priority comparisons — T→V vs Image-Only and GMU vs Image-Only ─────────
print(f"\n{'─'*72}")
print("P1 — PRIORITY: T→V / GMU / V→T vs. Image-Only")
print(f"     (Holm-Bonferroni corrected within family of {m_total} tests)")
print(f"{'─'*72}")
print(f"  {'Comparison':<35} {'Metric':<22} {'t':>7} {'p-raw':>8} {'adj_α':>8} {'Reject':>6} {'Δ(b-a)':>10} {'d':>7} {'|d|':>12}")
print(f"  {'─'*120}")

priority_comps = {
    "T→V            vs. Image-Only",
    "GMU            vs. Image-Only",
    "V→T            vs. Image-Only",
}
for row in sorted(corrected_rows, key=lambda r: r["p"]):
    if row["comp"].strip() not in {c.strip() for c in priority_comps}:
        continue
    sym = "✅" if row["reject"] else "no"
    dl = d_label(row["d"])
    print(f"  {row['comp']:<35} {row['metric']:<22} {row['t']:>+7.3f} {row['p']:>8.4f} "
          f"{row['adj_alpha']:>8.4f} {sym:>6} {row['delta']:>+10.4f} {row['d']:>+7.3f} {dl:>12}")

# Focused effect-size spotlight for the three key priority comparisons
print(f"\n  — Effect-size spotlight (Accuracy, CFR, Blank_Accuracy_Drop) —")
print(f"  {'Comparison':<35} {'Metric':<22} {'d':>7} {'|d| label':>12} {'Δ(b-a)':>12} {'a vals':>24} {'b vals':>24}")
print(f"  {'─'*130}")
SPOTLIGHT_METRICS = {"Accuracy", "CFR", "Blank_Accuracy_Drop"}
for row in corrected_rows:
    if row["comp"].strip() not in {c.strip() for c in priority_comps}:
        continue
    if row["metric"] not in SPOTLIGHT_METRICS:
        continue
    a_vals = get3(row["arch_a"], row["metric"]) or []
    b_vals = get3(row["arch_b"], row["metric"]) or []
    dl = d_label(row["d"])
    a_str = str([round(v, 4) for v in a_vals])
    b_str = str([round(v, 4) for v in b_vals])
    print(f"  {row['comp']:<35} {row['metric']:<22} {row['d']:>+7.3f} {dl:>12} {row['delta']:>+12.4f} {a_str:>24} {b_str:>24}")

# ── P2: Full table, all comparisons, sorted by p-value ─────────────────────────
print(f"\n{'─'*72}")
print("P2 — FULL CORRECTED TABLE (all comparisons, sorted by raw p)")
print(f"{'─'*72}")
print(f"  {'Comparison':<35} {'Metric':<22} {'t':>7} {'p-raw':>8} {'adj_α':>8} {'Reject':>6} {'Δ(b-a)':>10} {'d':>7} {'|d|':>12}")
print(f"  {'─'*120}")

for row in sorted(corrected_rows, key=lambda r: r["p"]):
    sym = "✅" if row["reject"] else "no"
    dl = d_label(row["d"])
    print(f"  {row['comp']:<35} {row['metric']:<22} {row['t']:>+7.3f} {row['p']:>8.4f} "
          f"{row['adj_alpha']:>8.4f} {sym:>6} {row['delta']:>+10.4f} {row['d']:>+7.3f} {dl:>12}")

# ── Summary: how many survived correction ──────────────────────────────────────
n_reject = sum(1 for r in corrected_rows if r["reject"])
print(f"\n  Survivors after Holm-Bonferroni: {n_reject} / {m_total}")
if n_reject == 0:
    print("  ⚠️  No comparisons survive correction at n=3 seeds.")
    print("  This is an honest, reportable finding: power is insufficient to")
    print("  make corrected claims at this scale. A minimum-seeds power analysis")
    print("  should be included in the paper to quantify the required n.")

# ── P1 focused summary ─────────────────────────────────────────────────────────
print(f"\n{'─'*72}")
print("P1 — FOCUSED NARRATIVE: T→V and GMU vs. Image-Only")
print(f"{'─'*72}")

for comp_name in ["T→V            vs. Image-Only",
                  "GMU            vs. Image-Only",
                  "V→T            vs. Image-Only"]:
    rows = [r for r in corrected_rows if r["comp"].strip() == comp_name.strip()]
    any_reject = any(r["reject"] for r in rows)
    print(f"\n  {comp_name.strip()}:")
    if not rows:
        print("    No data — Image-Only not yet trained.")
        continue

    # Effect-size summary regardless of significance
    large_effects  = [r for r in rows if abs(r["d"]) >= 0.8]
    medium_effects = [r for r in rows if 0.5 <= abs(r["d"]) < 0.8]
    small_effects  = [r for r in rows if 0.2 <= abs(r["d"]) < 0.5]
    negl_effects   = [r for r in rows if abs(r["d"]) < 0.2]
    print(f"    Effect sizes (d): large={len(large_effects)}, medium={len(medium_effects)}, "
          f"small={len(small_effects)}, negligible={len(negl_effects)}")
    if large_effects:
        for r in sorted(large_effects, key=lambda x: -abs(x["d"])):
            direction = "↑" if r["d"] > 0 else "↓"
            print(f"      Large {direction} d={r['d']:+.3f} on {r['metric']} "
                  f"(Δ={r['delta']:+.4f}, p={r['p']:.4f})")

    if any_reject:
        sig_metrics = [r["metric"] for r in rows if r["reject"]]
        print(f"    ✅ Survives Holm-Bonferroni on: {sig_metrics}")
        for r in rows:
            if r["reject"]:
                direction = "better" if r["delta"] > 0 else "worse"
                print(f"       {r['metric']}: Δ={r['delta']:+.4f} ({direction}), d={r['d']:+.3f} [{d_label(r['d'])}], "
                      f"t={r['t']:+.3f}, p={r['p']:.4f} (adj_α={r['adj_alpha']:.4f})")
    else:
        print(f"    ⚠️  NOT significant after Holm-Bonferroni (n=3 seeds, m={m_total} tests).")
        best_p = min(rows, key=lambda r: r["p"])
        best_d = max(rows, key=lambda r: abs(r["d"]))
        print(f"    Best raw p={best_p['p']:.4f} on {best_p['metric']} (d={best_p['d']:+.3f} [{d_label(best_p['d'])}])")
        print(f"    Largest |d|={best_d['d']:+.3f} [{d_label(best_d['d'])}] on {best_d['metric']} — "
              f"{'underpowered large effect' if abs(best_d['d']) >= 0.8 else 'effect too small to claim practical significance at n=3'}")

print(f"\n{'='*72}")
print("CORRECTED SIGNIFICANCE ANALYSIS COMPLETE")
print(f"{'='*72}\n")
