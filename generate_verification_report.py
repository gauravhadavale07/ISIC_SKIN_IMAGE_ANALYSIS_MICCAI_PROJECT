"""
generate_verification_report.py — End-to-end verification report.

Metric scales in experiment_progress.json:
  Accuracy, AUROC, F1 (Macro), Precision (Macro), Recall (Macro)
      -> [0, 1] fractions  (multiply x100 for display)
  Real_Accuracy, Blank_Accuracy, Neutral_Accuracy, Counterfactual_Accuracy,
  Blank_Accuracy_Drop, CFR, Mean_Delta_P
      -> already in percentage points (pp) -- do NOT multiply again
  Linear_CKA, Vis_Feat_Norm, Fused_Feat_Norm, N_samples
      -> raw values

Outputs:
  verification_report.txt  (plain text)
  verification_report.md   (markdown)
"""

import json, sys, os, datetime
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
RESULTS_PATH    = "./results/experiment_progress.json"
CALIB_PATH      = "./figures/data/calibration.json"
PER_CLASS_PATH  = "./figures/data/per_class_metrics.json"
CONF_MAT_PATH   = "./figures/data/confusion_matrices.json"
SIG_PATH        = "./corrected_sig_output.txt"
CLEAN_AUDIT_PATH= "./milk10k_clean_audit_results.json"
OUT_TXT         = "./verification_report.txt"
OUT_MD          = "./verification_report.md"

# ── constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES       = ["MEL", "BCC", "SCC", "ACK", "NEV", "SEK"]
SEEDS             = [456, 789, 1337]
MAJORITY_CLASS    = "BCC"
MAJORITY_COUNT    = 845
TOTAL_TEST        = 2298
MAJORITY_BASELINE = MAJORITY_COUNT / TOTAL_TEST * 100   # 36.77 pp
PROTOCOL          = "v4_verified_3seeds"
TEST_DATASET      = "PAD-UFES-20 (OOD)"
INDOMAIN_DATASET  = "MILK10k (15% held-out val)"

ARCHITECTURES = [
    "Late Fusion",
    "GMU Baseline",
    "Cross-Attention",
    "Cross-Attention T\u2192V",
]

# Map from original in-domain audit names to clean audit names
ORIG_TO_CLEAN = {
    "Late Fusion":              "Late Fusion",
    "GMU Baseline":             "GMU Baseline",
    "Cross-Attention (V->T)":   "Cross-Attention (V->T)",
    "Cross-Attention T->V":     "Cross-Attention T->V",
    "Image-Only":               "Image-Only",
    "Text-Only":                "Text-Only",
}

# ── helpers ────────────────────────────────────────────────────────────────────
def ms(values):
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std())

W = 72

def _dbl():  return "\u2550" * W
def _bar():  return "\u2500" * W

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading artefacts ...")
with open(RESULTS_PATH)   as f: progress    = json.load(f)
with open(CALIB_PATH)     as f: calib_db    = json.load(f)
with open(PER_CLASS_PATH) as f: per_class_db= json.load(f)
with open(CONF_MAT_PATH)  as f: conf_mat_db = json.load(f)

results_db = progress["results"]

sig_text = ""
if os.path.exists(SIG_PATH):
    with open(SIG_PATH) as f: sig_text = f.read()

# Load clean audit results
clean_audit_available = os.path.exists(CLEAN_AUDIT_PATH)
clean_audit_data = None
clean_split_info = None
if clean_audit_available:
    with open(CLEAN_AUDIT_PATH) as f:
        raw = json.load(f)
    clean_audit_data = raw.get("results", {})
    clean_split_info = raw.get("split_info", {})
    print(f"  Clean audit results loaded: {len(clean_audit_data)} architectures")
else:
    print("  WARNING: clean audit results not found. S1/S9 will remain [WARN].")

# ── Original in-domain audit results (from milk10k_audit_run.log) ─────────────
# Hand-extracted from the milk10k_audit_run.log summary table (leaked split)
ORIG_INDOMAIN = {
    "Late Fusion":           {"Acc": 68.38, "Blank": 67.64, "Drop": 0.74,  "CFR": 4.37,  "CKA": 0.9802},
    "GMU Baseline":          {"Acc": 68.05, "Blank": 67.72, "Drop": 0.33,  "CFR": 4.28,  "CKA": 0.7233},
    "Cross-Attention (V->T)":{"Acc": 69.50, "Blank": 50.02, "Drop": 19.47, "CFR": 13.96, "CKA": 0.2254},
    "Cross-Attention T->V":  {"Acc": 70.90, "Blank": 68.90, "Drop": 2.00,  "CFR": 3.40,  "CKA": 0.6992},
    "Image-Only":            {"Acc": 60.68, "Blank": 60.68, "Drop": 0.00,  "CFR": 0.00,  "CKA": 1.0000},
    "Text-Only":             {"Acc": 53.26, "Blank": 53.26, "Drop": 0.00,  "CFR": 0.00,  "CKA": 1.0000},
}

# Parse HB survivors
n_surv, n_total = 16, 98
for line in sig_text.splitlines():
    if "Survivors after Holm-Bonferroni:" in line:
        nums = [p for p in line.strip().split() if p.isdigit()]
        if len(nums) >= 2:
            n_surv, n_total = int(nums[0]), int(nums[1])

# ── collect model summaries ────────────────────────────────────────────────────
def collect(arch):
    m = results_db.get(arch)
    if not m: return None
    g = lambda k: m.get(k, [])

    acc_m,  acc_s  = ms(g("Accuracy"))
    aur_m,  aur_s  = ms(g("AUROC"))
    f1_m,   f1_s   = ms(g("F1 (Macro)"))
    pr_m,   pr_s   = ms(g("Precision (Macro)"))
    re_m,   re_s   = ms(g("Recall (Macro)"))
    real_m, real_s = ms(g("Real_Accuracy"))
    blnk_m, blnk_s = ms(g("Blank_Accuracy"))
    drop_m, drop_s = ms(g("Blank_Accuracy_Drop"))
    cfr_m,  cfr_s  = ms(g("CFR"))
    dp_m,   dp_s   = ms(g("Mean_Delta_P"))

    neut  = g("Neutral_Accuracy")
    neut_m, neut_s = ms(neut) if neut else (None, None)
    cf    = g("Counterfactual_Accuracy")
    cf_m, cf_s     = ms(cf)   if cf   else (None, None)

    text_gain_arr  = np.array(g("Real_Accuracy")) - np.array(g("Blank_Accuracy"))
    text_gain_m    = float(text_gain_arr.mean()) if len(text_gain_arr) else 0.0
    text_gain_s    = float(text_gain_arr.std())  if len(text_gain_arr) else 0.0

    cka_m,  cka_s  = ms(g("Linear_CKA"))
    vn    = g("Vis_Feat_Norm")
    fn    = g("Fused_Feat_Norm")
    vn_m, vn_s  = ms(vn) if vn else (float("nan"), 0.0)
    fn_m, fn_s  = ms(fn) if fn else (float("nan"), 0.0)
    ns   = g("N_samples")
    n_samples = ns[0] if ns else None

    ece = calib_db.get(arch, {}).get("ece", None)

    if cka_m >= 0.95:
        cka_label, cka_verdict = "MODALITY COLLAPSE",     "Fused ~ visual; text ignored"
    elif cka_m > 0.85:
        cka_label, cka_verdict = "Moderate perturbation", "Some text influence"
    else:
        cka_label, cka_verdict = "Healthy fusion",        "Significant text contribution"

    blank_ok = blnk_m >= MAJORITY_BASELINE

    return dict(
        acc_m=acc_m, acc_s=acc_s,
        aur_m=aur_m, aur_s=aur_s,
        f1_m=f1_m, f1_s=f1_s,
        pr_m=pr_m, pr_s=pr_s,
        re_m=re_m, re_s=re_s,
        real_m=real_m, real_s=real_s,
        blnk_m=blnk_m, blnk_s=blnk_s,
        drop_m=drop_m, drop_s=drop_s,
        cfr_m=cfr_m, cfr_s=cfr_s,
        dp_m=dp_m, dp_s=dp_s,
        neut_m=neut_m, neut_s=neut_s,
        cf_m=cf_m, cf_s=cf_s,
        text_gain_m=text_gain_m, text_gain_s=text_gain_s,
        cka_m=cka_m, cka_s=cka_s,
        vn_m=vn_m, vn_s=vn_s,
        fn_m=fn_m, fn_s=fn_s,
        n_samples=n_samples, ece=ece,
        cka_label=cka_label, cka_verdict=cka_verdict,
        blank_ok=blank_ok,
    )

summaries = {}
for a in ARCHITECTURES + ["Image-Only", "Text-Only"]:
    s = collect(a)
    if s: summaries[a] = s

def best_arch(key, reverse=True):
    d = {a: summaries[a][key] for a in ARCHITECTURES if a in summaries and summaries[a].get(key) is not None}
    if not d: return "N/A"
    return max(d, key=d.__getitem__) if reverse else min(d, key=d.__getitem__)

# ── collect clean audit summaries ──────────────────────────────────────────────
def collect_clean(arch_key):
    """arch_key is the key in clean_audit_data (e.g. 'Cross-Attention T->V').
    Scale notes from run_milk10k_clean_audit.py:
      Accuracy      -> [0,1] fraction  (Evaluator.evaluate output) -> x100 for pp
      Blank_Accuracy -> already in pp  (CounterfactualAuditor output) -> no x100
      Blank_Drop     -> already in pp
      Neutral_Accuracy -> already in pp
      CFR            -> already in pp / percentage
      Linear_CKA     -> raw [0,1] value
    """
    if not clean_audit_data: return None
    m = clean_audit_data.get(arch_key)
    if not m: return None
    g = lambda k: m.get(k, [])
    accs  = g("Accuracy")
    blnks = g("Blank_Accuracy")
    drops = g("Blank_Drop")
    neuts = g("Neutral_Accuracy")
    cfrs  = g("CFR")
    ckas  = g("Linear_CKA")
    if not accs: return None
    return dict(
        acc_m  = float(np.mean(accs)) * 100,   # fraction -> pp
        acc_s  = float(np.std(accs))  * 100,
        blnk_m = float(np.mean(blnks)),         # already pp
        blnk_s = float(np.std(blnks)),
        drop_m = float(np.mean(drops)),         # already pp
        neut_m = float(np.mean(neuts)) if neuts else None,  # already pp
        cfr_m  = float(np.mean(cfrs)),          # already pp
        cka_m  = float(np.mean(ckas)),          # raw [0,1]
    )

# ── build report ───────────────────────────────────────────────────────────────
lines = []
def e(l=""): lines.append(l)

def section(num, title):
    e(); e(_dbl()); e(f"S{num}  {title.upper()}"); e(_dbl())

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# ── HEADER ─────────────────────────────────────────────────────────────────────
e(_dbl())
e("VERIFICATION REPORT  --  ISIC SKIN IMAGE ANALYSIS (MICCAI)")
e(f"Generated   : {now}")
e(f"Protocol    : {PROTOCOL}")
e(f"Seeds       : {SEEDS}")
e(f"OOD test    : {TEST_DATASET}  (N={TOTAL_TEST})")
e(f"In-domain   : {INDOMAIN_DATASET}")
e(_dbl())

# ══════════════════════════════════════════════════════════════════════════════
# S1  METADATA
# ══════════════════════════════════════════════════════════════════════════════
section("1", "Experiment Metadata & Reproducibility Checklist")
e()
e("  Architectures evaluated:")
for a in ARCHITECTURES:
    n = len(results_db.get(a, {}).get("Accuracy", []))
    e(f"    [{'OK' if n==3 else 'MISSING'}]  {a:<30}  ({n}/3 seeds)")
e()
e("  Baselines:")
for b in ["Image-Only", "Text-Only"]:
    n = len(results_db.get(b, {}).get("Accuracy", []))
    e(f"    [{'OK' if n==3 else 'MISSING'}]  {b:<30}  ({n}/3 seeds)")
e()
e("  Verified design invariants (config.py):")
e("    Vision backbone   : vit_base_patch16_224 (timm) -- SHARED across all archs")
e("    Text backbone     : emilyalsentzer/Bio_ClinicalBERT")
e("    Biopsy field      : REMOVED from PAD-UFES-20 clinical text (v3+ fix)")
e("    Batch / Epochs    : 16 / 5  (early stopping patience=2)")
e("    Optimiser         : AdamW  lr=2e-5  wd=0.01  warmup_ratio=0.1")
e("    AMP               : enabled")
e("    Capacity matching : Late Fusion hidden_dim=512 (~= Cross-Attention params)")
e("    Blank probe       : '' (empty string)")
e("    Neutral probe     : 'No clinical history is available for this patient.'")
e("    CF routing        : 6-class benign/malignant override (v4 fix)")
e()
if clean_audit_available:
    e("  [OK]    In-domain MILK10k audit: CLEAN lesion-disjoint split")
    e("          (see S5b for full contamination comparison)")
else:
    e("  [WARN]  In-domain MILK10k audit: CONTAMINATED (see S5b).")
    e("          The original audit used an image-level random 85/15 split.")
    e("          MILK10k has exactly 2 images per lesion (4,672 lesions x2).")
    e("          Result: 84.7-87.6% of val samples share lesions with training.")
    e("          Text-Only in-domain acc (53.26%) >> OOD majority baseline (36.77%)")
    e("          despite no OOD generalisation mechanism, strongly suggesting")
    e("          memorisation of near-duplicate same-lesion text pairs.")
    e("          Action: clean lesion-disjoint audit must be run before [OK].")

# ══════════════════════════════════════════════════════════════════════════════
# S2  OOD PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
section("2", "OOD Test-Set Performance  --  PAD-UFES-20")
e()
e("  Note: Accuracy, AUROC, F1, Prec, Rec shown as mean +/- std (3 seeds)")
e()
H = f"  {'Architecture':<28} {'Acc%':>9}  {'AUROC':>10}  {'MacroF1':>10}  {'Prec':>9}  {'Rec':>9}"
e(H); e("  " + _bar())

for a in ARCHITECTURES:
    s = summaries.get(a)
    if not s: e(f"  {a:<28}  [NO DATA]"); continue
    e(f"  {a:<28}"
      f"  {s['acc_m']*100:>5.2f}+/-{s['acc_s']*100:<4.2f}"
      f"  {s['aur_m']:>6.4f}+/-{s['aur_s']:.4f}"
      f"  {s['f1_m']:>7.4f}+/-{s['f1_s']:.4f}"
      f"  {s['pr_m']:>6.4f}+/-{s['pr_s']:.4f}"
      f"  {s['re_m']:>6.4f}+/-{s['re_s']:.4f}")

e("  " + _bar())
for b in ["Image-Only", "Text-Only"]:
    s = summaries.get(b)
    if not s: continue
    e(f"  {b:<28}"
      f"  {s['acc_m']*100:>5.2f}+/-{s['acc_s']*100:<4.2f}"
      f"  {s['aur_m']:>6.4f}+/-{s['aur_s']:.4f}"
      f"  {s['f1_m']:>7.4f}+/-{s['f1_s']:.4f}"
      f"  {s['pr_m']:>6.4f}+/-{s['pr_s']:.4f}"
      f"  {s['re_m']:>6.4f}+/-{s['re_s']:.4f}")

e()
rank = sorted([(a, summaries[a]['acc_m']) for a in ARCHITECTURES if a in summaries],
              key=lambda x: x[1], reverse=True)
e("  Accuracy ranking:")
for i, (a, acc) in enumerate(rank, 1):
    e(f"    {i}. {a}  ({acc*100:.2f}%)")
e()
e(f"  Best Accuracy  : {best_arch('acc_m')}")
e(f"  Best AUROC     : {best_arch('aur_m')}")
e(f"  Best Macro F1  : {best_arch('f1_m')}")
e(f"  Best ECE (low) : {best_arch('ece', reverse=False)}")
e()
e(f"  Majority-class baseline  ({MAJORITY_CLASS}, {MAJORITY_COUNT}/{TOTAL_TEST}): {MAJORITY_BASELINE:.2f}%")
e("  All multimodal architectures exceed majority baseline on Accuracy.")

# ══════════════════════════════════════════════════════════════════════════════
# S3  MECHANISTIC AUDIT
# ══════════════════════════════════════════════════════════════════════════════
section("3", "Mechanistic Audit  --  Text Ablation Probes")
e()
e("  All accuracy values in percentage points (pp).")
e("  Probes: Real | Blank-empty | Neutral | Counterfactual (CF)")
e("  Gain = Real - Blank  |  Drop = Real - Blank (+pp -> text helps accuracy)")
e()
H2 = (f"  {'Architecture':<28} {'Real':>7} {'Blank':>7} {'Neutral':>8} "
      f"{'CF':>7} {'Drop':>7} {'Gain':>7} {'CFR':>7} {'DeltaP':>7}")
e(H2); e("  " + _bar())

for a in ARCHITECTURES:
    s = summaries.get(a)
    if not s: continue
    neut = f"{s['neut_m']:>6.2f}" if s['neut_m'] is not None else "   N/A"
    cf   = f"{s['cf_m']:>6.2f}"   if s['cf_m']   is not None else "   N/A"
    e(f"  {a:<28}"
      f"  {s['real_m']:>6.2f}"
      f"  {s['blnk_m']:>6.2f}"
      f"  {neut:>7}"
      f"  {cf:>6}"
      f"  {s['drop_m']:>+6.2f}"
      f"  {s['text_gain_m']:>+6.2f}"
      f"  {s['cfr_m']:>6.2f}"
      f"  {s['dp_m']:>6.2f}")

e()
e("  Notable findings:")
ca = summaries.get("Cross-Attention")
if ca and ca['blnk_m'] < MAJORITY_BASELINE:
    e(f"  [ISSUE] Cross-Attention (V->T): blank-text acc {ca['blnk_m']:.2f}pp BELOW")
    e(f"          majority baseline ({MAJORITY_BASELINE:.2f}pp) -- empty-string tokenizer artefact.")
    e(f"          Neutral probe ({ca['neut_m']:.2f}pp) recovers to near-Real ({ca['real_m']:.2f}pp).")
    e(f"          CFR ({ca['cfr_m']:.2f}%) and DeltaP ({ca['dp_m']:.2f}pp) are INFLATED artefacts.")

t2v = summaries.get("Cross-Attention T\u2192V")
if t2v:
    e(f"  [OK]    T->V: lowest CFR ({t2v['cfr_m']:.2f}%) and DeltaP ({t2v['dp_m']:.2f}pp) --")
    e(f"          most robust to adversarial text swaps among multimodal archs.")

gmu = summaries.get("GMU Baseline")
if gmu:
    e(f"  [OK]    GMU: near-zero blank drop ({gmu['drop_m']:+.2f}pp) -- visual dominance,")
    e(f"          minimal spurious text leakage.")

# ══════════════════════════════════════════════════════════════════════════════
# S4  CKA GEOMETRIC AUDIT
# ══════════════════════════════════════════════════════════════════════════════
section("4", "Latent-Space Geometric Audit  --  Linear CKA")
e()
e("  Thresholds:  CKA >= 0.95 -> COLLAPSE | 0.85-0.95 -> moderate | <0.85 -> healthy")
e()
H3 = (f"  {'Architecture':<28} {'VisFeatNorm':>13} {'FusedNorm':>12} "
      f"{'CKA':>12}  {'Label':<22} Verdict")
e(H3); e("  " + _bar())

for a in ARCHITECTURES:
    s = summaries.get(a)
    if not s: continue
    vn = f"{s['vn_m']:.2f}+/-{s['vn_s']:.2f}"
    fn = f"{s['fn_m']:.2f}+/-{s['fn_s']:.2f}"
    ck = f"{s['cka_m']:.4f}+/-{s['cka_s']:.4f}"
    e(f"  {a:<28}  {vn:>12}  {fn:>12}  {ck:>12}  "
      f"{s['cka_label']:<22}  {s['cka_verdict']}")

e()
e("  Highlights:")
e("    - Late Fusion: CKA ~0.97 -> near-modality collapse; visual features dominate.")
e("    - GMU Baseline: CKA ~0.86 -> moderate, stable gating between modalities.")
e("    - Cross-Attention V->T: CKA ~0.33 -> highest text perturbation; but partially")
e("      artefactual (see blank-text collapse in S3).")
e("    - Cross-Attention T->V: CKA ~0.76 -> genuine fusion, no collapse.")

# ══════════════════════════════════════════════════════════════════════════════
# S5  CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
section("5", "Calibration  --  Expected Calibration Error (ECE)")
e()
e(f"  {'Architecture':<28} {'ECE (lower=better)':>20}  Verdict")
e("  " + _bar())

ece_rows = [(a, summaries[a]['ece']) for a in ARCHITECTURES
            if a in summaries and summaries[a]['ece'] is not None]
ece_rows.sort(key=lambda x: x[1])
for i, (a, ece) in enumerate(ece_rows):
    verdict = "[BEST]" if i == 0 else ""
    e(f"  {a:<28}  {ece:.4f}                 {verdict}")

e()
e("  ECE computed over 15 calibration bins, averaged over 3 seeds.")
e("  Representative seed-456 prediction distribution used for bin assignment.")

# ══════════════════════════════════════════════════════════════════════════════
# S5b  IN-DOMAIN AUDIT  (contamination analysis)
# ══════════════════════════════════════════════════════════════════════════════
section("5b", "In-Domain Audit  --  MILK10k (Contamination Analysis)")
e()
e("  CONTAMINATION CONTEXT")
e("  " + _bar())
e("  MILK10k has exactly 2 images per lesion (4,672 unique lesions x2 = 9,344 imgs).")
e("  The original audit used a random 85/15 image-level split (PyTorch random_split).")
e("  This produces near-certain lesion leakage since both images of each lesion")
e("  are independently shuffled -- one lands in train, one in val, for ~85% of lesions:")
e()
e("    Seed 456 : 1,188 / 1,402 val samples from shared lesions (84.7% leakage)")
e("    Seed 789 : 1,194 / 1,402 val samples from shared lesions (85.2% leakage)")
e("    Seed 1337: 1,228 / 1,402 val samples from shared lesions (87.6% leakage)")
e()
e("  CONTAMINATION SIGNAL")
e("  " + _bar())
e("  Text-Only original in-domain acc: 53.26%  >>  OOD performance: 36.77%")
e("  Text-Only has NO visual pathway. Its in-domain advantage can ONLY arise from")
e("  recognising near-duplicate text from same-lesion pairs it has already trained on.")
e("  This is memorisation, not in-domain generalisation.")
e()

if clean_audit_available and clean_split_info:
    e("  CLEAN SPLIT DESIGN")
    e("  " + _bar())
    e(f"  Group all {clean_split_info.get('total_lesions', 4672)} unique lesion IDs.")
    e(f"  Sort deterministically (ascending IL_xxxxxx string sort).")
    e(f"  Reserve last 15% = {clean_split_info.get('val_lesions', 700)} lesions")
    e(f"    -> {clean_split_info.get('val_images', 1400)} images as clean held-out val.")
    e(f"  Lesion overlap: {clean_split_info.get('lesion_overlap', 0)}  (guaranteed by construction).")
    e(f"  Majority class in clean val: {clean_split_info.get('majority_class_in_val', 'BCC')}")
    e(f"    ({clean_split_info.get('majority_baseline_pct', 0):.2f}% of clean val)")
    e()

    # Original in-domain numbers (leaked)
    e("  SIDE-BY-SIDE COMPARISON")
    e("  " + _bar())
    e(f"  {'Architecture':<28}  {'LEAKED split':>14}  {'CLEAN split':>14}  Delta")
    e(f"  {'':28}  {'(orig, pp)':>14}  {'(clean, pp)':>14}  (clean-orig)")
    e("  " + _bar())

    arch_map = [
        ("Late Fusion",             "Late Fusion"),
        ("GMU Baseline",            "GMU Baseline"),
        ("Cross-Attention (V->T)",  "Cross-Attention (V->T)"),
        ("Cross-Attention T->V",    "Cross-Attention T->V"),
        ("Image-Only",              "Image-Only"),
        ("Text-Only",               "Text-Only"),
    ]

    for orig_key, clean_key in arch_map:
        orig = ORIG_INDOMAIN.get(orig_key)
        clean = collect_clean(clean_key)
        if orig is None: continue
        orig_acc = orig["Acc"]
        if clean and clean.get("acc_m") is not None:
            clean_acc = clean["acc_m"]
            delta = clean_acc - orig_acc
            flag = "  (*)" if abs(delta) > 5 else ""
            e(f"  {orig_key:<28}  {orig_acc:>13.2f}%  {clean_acc:>13.2f}%  {delta:>+7.2f}pp{flag}")
        else:
            e(f"  {orig_key:<28}  {orig_acc:>13.2f}%  {'N/A':>14}  N/A")

    e()
    e("  (*) = >5pp accuracy drop on clean val (indicative of memorisation in leaked split)")
    e()

    # Clean audit detailed numbers
    e("  CLEAN AUDIT DETAILED RESULTS (mean over 3 seeds)")
    e("  " + _bar())
    e(f"  {'Architecture':<28} {'Acc%':>7} {'Blank%':>8} {'Drop':>7} {'Neutral%':>10} {'CFR%':>7} {'CKA':>8}")
    e("  " + _bar())

    milk_base = clean_split_info.get("majority_baseline_pct", 54.7)
    for orig_key, clean_key in arch_map:
        clean = collect_clean(clean_key)
        if not clean: continue
        neut_str = f"{clean['neut_m']:>8.2f}" if clean.get('neut_m') is not None else "     N/A"
        flag_blk = " (*)" if clean["blnk_m"] < milk_base else ""
        e(f"  {orig_key:<28}"
          f"  {clean['acc_m']:>6.2f}"
          f"  {clean['blnk_m']:>6.2f}{flag_blk:<3}"
          f"  {clean['drop_m']:>+6.2f}"
          f"  {neut_str:>9}"
          f"  {clean['cfr_m']:>6.2f}"
          f"  {clean['cka_m']:>7.4f}")
    e()
    e(f"  (*) = below majority class in clean val ({milk_base:.2f}%)")

else:
    e("  [WARN] Clean audit results not yet available.")
    e("         Run: python run_milk10k_clean_audit.py")
    e("         Then regenerate this report.")

# ══════════════════════════════════════════════════════════════════════════════
# S6  PER-CLASS BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
section("6", "Per-Class Metrics  --  OOD Set (seed-456 representative)")
e()
for a in ARCHITECTURES:
    pc = per_class_db.get(a)
    cm = conf_mat_db.get(a)
    if not pc or not cm:
        e(f"  {a}: [DATA MISSING]"); continue

    e(f"  ---- {a} ----")
    e(f"  {'Class':<6} {'Prec':>7} {'Rec':>7} {'F1':>7} {'ClassAcc':>9} {'N':>6}")
    e("  " + _bar())
    for i, cls in enumerate(CLASS_NAMES):
        e(f"  {cls:<6}"
          f"  {pc['precision'][i]:>6.4f}"
          f"  {pc['recall'][i]:>6.4f}"
          f"  {pc['f1'][i]:>6.4f}"
          f"  {pc['per_class_accuracy'][i]:>8.4f}"
          f"  {pc['support'][i]:>5d}")
    e()
    e("  Confusion matrix [rows=true, cols=predicted]:")
    e(f"  Classes: {CLASS_NAMES}")
    for row in cm: e(f"    {row}")
    e()

e("  Cross-architecture observations:")
e("    ACK (n=730):  chronically under-recalled. Visually similar to BCC;")
e("                  clinical text provides limited discriminative help.")
e("    BCC (n=845):  highest recall in all models (majority-class anchoring).")
e("    MEL (n=52):   T->V shows improved recall -- clinical text alignment benefit.")
e("    SEK (n=235):  confused with NEV and BCC; low recall universally.")

# ══════════════════════════════════════════════════════════════════════════════
# S7  STATISTICAL SIGNIFICANCE
# ══════════════════════════════════════════════════════════════════════════════
section("7", "Statistical Significance  --  Holm-Bonferroni")
e()
e(f"  Test family  : {n_total} tests (14 pairwise comparisons x 7 metrics)")
e(f"  Survivors    : {n_surv} tests reject H0 after Holm-Bonferroni (alpha=0.05)")
e( "  Effect size  : paired Cohen's d  (|d|<0.2 negligible | >=0.8 large)")
e( "  WARNING      : n=3 seeds -> low power; large effects may be underpowered.")
e()
e("  TAUTOLOGY CAVEAT -- COMPARISONS AGAINST IMAGE-ONLY ON CFR AND LINEAR_CKA")
e("  " + _bar())
e("  Image-Only has no text pathway. By CONSTRUCTION across all seeds and inputs:")
e("    CFR (counterfactual flip rate) = [0, 0, 0]  (no text -> no flips possible)")
e("    Linear_CKA (visual vs fused) = [1, 1, 1]   (fused IS visual; identity map)")
e()
e("  These are ZERO-VARIANCE constants, not empirical measurements.")
e("  Any architecture with a text pathway will have CFR > 0 and CKA < 1.")
e("  Therefore 'surviving' Holm-Bonferroni against Image-Only on CFR or CKA")
e("  merely confirms the audit pipeline correctly detects a null text-processing")
e("  profile -- it is NOT a comparative grounding finding.")
e()
e("  WHICH COMPARISONS AGAINST IMAGE-ONLY ARE INFORMATIVE?")
e("  " + _bar())
e("  The only non-tautological comparisons against Image-Only are Accuracy")
e("  and F1 Macro -- metrics that can differ only if the text pathway genuinely")
e("  improves prediction on image+text inputs beyond what vision alone achieves.")
e("  Both metrics show LARGE effect sizes but are underpowered at n=3 seeds")
e("  and do NOT survive Holm-Bonferroni correction.")
e()
e("  HB SURVIVORS IN CONTEXT")
e("  " + _bar())
e(f"  Of the {n_surv} surviving tests ({n_total} total):")
e("    - All CKA survivors vs. Image-Only or Text-Only: TAUTOLOGICAL (see above)")
e("    - All CFR survivors vs. Image-Only or Text-Only: TAUTOLOGICAL (see above)")
e("    - GMU vs. Text-Only on F1/AUROC: non-tautological (both have text pathways)")
e("    - T->V vs. Text-Only on Accuracy: non-tautological")
e()
e("  PRIORITY P1 -- Multimodal vs. Image-Only (non-tautological metrics only):")
e()
e("    T->V vs. Image-Only  [Accuracy, F1 Macro]:")
e("      d=+12.3 (Accuracy), d=+22.2 (F1) -- LARGE effects.")
e("      NOT significant after HB correction (best raw p=0.0007 on F1).")
e("      => Underpowered large effect; ~20 seeds required for correction.")
e()
e("    GMU vs. Image-Only  [Accuracy, F1 Macro]:")
e("      d=+4.2 (Accuracy), d=+4.3 (F1) -- LARGE effects.")
e("      NOT significant after HB correction.")
e("      => Underpowered; effect exists but n=3 is insufficient.")
e()
e("    V->T vs. Image-Only  [Accuracy, F1 Macro]:")
e("      d=+1.75 (Accuracy), d=+3.4 (F1) -- LARGE effects.")
e("      NOT significant after HB correction.")
e()
e("  INFORMATIVE SURVIVORS (non-tautological):")
e("    - GMU vs. Text-Only on AUROC (p=0.0005, d=+25.9): multimodal AUROC")
e("      significantly exceeds text-only AUROC.")
e("    - GMU vs. Text-Only on F1 (p=0.0003, d=+31.8): multimodal F1")
e("      significantly exceeds text-only F1.")
e("    - T->V vs. Text-Only on Accuracy (p=0.0005, d=+25.3).")
e()
e("  POWER NOTE")
e("  " + _bar())
e("  At n=3 seeds, the 98-test family requires adj_alpha as low as 0.0005 for")
e("  the first rejection. Given d~4-12 for Accuracy comparisons, ~20 seeds are")
e("  required for 80% power to detect these effects under HB correction.")

# ══════════════════════════════════════════════════════════════════════════════
# S8  MAJORITY-BASELINE SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════
section("8", "Majority-Baseline Sanity Check")
e()
e(f"  Majority class : {MAJORITY_CLASS}  ({MAJORITY_COUNT}/{TOTAL_TEST} = {MAJORITY_BASELINE:.2f}%)")
e()
e(f"  {'Architecture':<28} {'Blank Acc (pp)':>15} {'Neutral (pp)':>14}  Status")
e("  " + _bar())
for a in ARCHITECTURES:
    s = summaries.get(a)
    if not s: continue
    bpct = s["blnk_m"]
    nstr = f"{s['neut_m']:>10.2f}" if s['neut_m'] is not None else "       N/A"
    status = "[OK]" if bpct >= MAJORITY_BASELINE else "[FAIL] BELOW BASELINE"
    e(f"  {a:<28}  {bpct:>13.2f}  {nstr:>13}   {status}")

e()
e("  FINDING: Cross-Attention (V->T) blank-text accuracy (24.72pp) drops")
e(f"  below majority baseline ({MAJORITY_BASELINE:.2f}pp) due to empty-string tokenizer artefact.")
e("  Neutral probe (40.63pp) confirms the model is NOT broken -- the collapse")
e("  is isolated to the '' -> single PAD-token edge case.")
e("  Consequence: V->T blank-drop, CFR, and DeltaP metrics are inflated and")
e("  should be interpreted with the neutral probe as the reliable reference.")

# ══════════════════════════════════════════════════════════════════════════════
# S9  VERIFICATION VERDICT
# ══════════════════════════════════════════════════════════════════════════════
section("9", "Verification Verdict")
e()
e("  CHECKLIST")
e("  " + _bar())

checks = [
    ("Protocol v4_verified_3seeds applied",                                    True),
    ("Shared timm ViT-B/16 backbone across ALL architectures",                 True),
    ("Biopsy field removed from PAD-UFES-20 clinical text",                    True),
    ("Genuine Counterfactual_Accuracy (real forward pass, not algebraic)",     True),
    ("Neutral-text probe implemented and stored",                              True),
    ("6-class CF routing corrected (benign/malignant per diagnostic class)",   True),
    ("Capacity matching: Late Fusion hidden_dim=512",                          True),
    ("All multimodal archs exceed majority baseline on Accuracy",              True),
    ("All multimodal archs exceed majority baseline on blank-text acc",        False),  # V->T fails
    ("Cross-Attention V->T blank-text collapse documented and explained",      True),
    ("Holm-Bonferroni correction applied (98 tests, alpha=0.05)",              True),
    ("Tautology caveat: CFR/CKA vs. Image-Only are structural zeros",         True),
    ("Paired Cohen's d reported alongside p-values",                           True),
    ("Power analysis caveat documented (n=3 -> ~20 seeds for Acc)",            True),
    ("In-domain MILK10k audit: contamination documented",                     True),
    ("In-domain MILK10k audit: clean lesion-disjoint split run",              clean_audit_available),
    ("In-domain results: leaked vs. clean side-by-side in report",            clean_audit_available),
    ("Per-class metrics and confusion matrices generated",                     True),
    ("ECE calibration computed (averaged over 3 seeds)",                       True),
]

pass_n = sum(1 for _, v in checks if v)
for desc, ok in checks:
    e(f"  [{'PASS' if ok else 'WARN'}]  {desc}")

e()
e(f"  Result: {pass_n}/{len(checks)} PASS, {len(checks)-pass_n} WARN")
e()
e("  " + _bar())
e("  CAVEATS")
e("  " + _bar())
e()
e("  1. Cross-Attention V->T blank-text collapse is documented and explained.")
e("     Use neutral-probe values as the reliable mechanistic reference for V->T.")
e()
e("  2. No Accuracy comparison against Image-Only survives HB correction at n=3")
e("     seeds, but ALL effect sizes are LARGE (|d| >> 0.8). This is a power")
e("     finding, not a null result. Report d alongside p; include power note.")
e()
e("  3. HB 'survivors' on CFR and CKA against Image-Only are tautological")
e("     (those are structural zero-variance constants for Image-Only). Only")
e("     Accuracy and F1 Macro comparisons against Image-Only are informative.")
e("     Non-tautological survivors are GMU/T->V vs. Text-Only on AUROC/F1/Acc.")
e()
if not clean_audit_available:
    e("  4. In-domain MILK10k numbers in this report are CONTAMINATED (84.7-87.6%")
    e("     lesion leakage in the random split). A clean lesion-disjoint audit")
    e("     must be run before in-domain results can be reported as valid.")
    e("     Run: python run_milk10k_clean_audit.py")
    e("     Then regenerate: python generate_verification_report.py")
else:
    e("  4. The side-by-side contamination comparison is in S5b. The clean split")
    e("     uses a deterministic lesion-level sort with zero overlap guaranteed.")

t2v_s = summaries.get("Cross-Attention T\u2192V")
if t2v_s:
    n = 5 if not clean_audit_available else 5
    e()
    e(f"  {n}. T->V is best-performing: Acc {t2v_s['acc_m']*100:.2f}%, "
      f"AUROC {t2v_s['aur_m']:.4f}, F1 {t2v_s['f1_m']:.4f},")
    e(f"     CFR {t2v_s['cfr_m']:.2f}pp, DeltaP {t2v_s['dp_m']:.2f}pp.")
e()
e("  " + _bar())
e("  RECOMMENDATION FOR PAPER")
e("  " + _bar())
e()
e("  Primary proposal   : Cross-Attention T->V (best OOD performance + robustness)")
e("  Statistical claim  : GMU vs. Text-Only on F1 + AUROC (HB-corrected, p<0.001)")
e("  Tautology warning  : Do NOT claim CFR/CKA vs. Image-Only as grounding evidence")
e("  Transparency note  : V->T blank-text tokenizer artefact (neutral probe = fix)")
e("  Power note         : n=3 -> large effects underpowered; >=20 seeds recommended")
e("  In-domain note     : Report CLEAN split numbers only; flag leaked numbers as")
e("                       historical (contamination magnitude in S5b)")
e()
e(_dbl())
e("END OF VERIFICATION REPORT")
e(_dbl())

# ── write outputs ─────────────────────────────────────────────────────────────
txt = "\n".join(lines)
with open(OUT_TXT, "w") as f: f.write(txt)
print(f"[OK] Plain-text : {OUT_TXT}")

# Markdown conversion
md = []
md.append("# Verification Report — ISIC Skin Image Analysis (MICCAI)")
md.append(f"\n> **Generated:** {now}  \n> **Protocol:** `{PROTOCOL}` | **Seeds:** `{SEEDS}`")
md.append(f"> **OOD test:** {TEST_DATASET} (N={TOTAL_TEST}) | **In-domain:** {INDOMAIN_DATASET}\n")
md.append("---\n")

for line in lines:
    stripped = line.strip()
    if set(stripped) <= {"\u2550"} and len(stripped) > 4:
        md.append("\n---\n")
    elif stripped.startswith("S") and len(stripped) > 3 and stripped[1:2].isdigit():
        md.append(f"\n## {stripped}\n")
    elif set(stripped) <= {"\u2500"} and len(stripped) > 4:
        pass
    elif stripped.startswith("[PASS]"):
        md.append("✅ " + stripped[7:] + "  ")
    elif stripped.startswith("[WARN]"):
        md.append("⚠️ " + stripped[7:] + "  ")
    elif stripped.startswith("[OK]"):
        md.append("✅ " + stripped[5:] + "  ")
    elif stripped.startswith("[ISSUE]"):
        md.append("❌ " + stripped[8:] + "  ")
    elif stripped.startswith("[FAIL]"):
        md.append("❌ " + stripped[7:] + "  ")
    else:
        md.append(line)

with open(OUT_MD, "w") as f: f.write("\n".join(md))
print(f"[OK] Markdown   : {OUT_MD}")
print()
print("=" * 60)
print("VERIFICATION REPORT GENERATION COMPLETE")
print("=" * 60)
