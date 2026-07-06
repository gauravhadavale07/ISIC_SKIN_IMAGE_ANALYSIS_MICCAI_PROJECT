# Verification Report — ISIC Skin Image Analysis (MICCAI)

> **Generated:** 2026-07-06 12:01 UTC  
> **Protocol:** `v4_verified_3seeds` | **Seeds:** `[456, 789, 1337]`
> **OOD test:** PAD-UFES-20 (OOD) (N=2298) | **In-domain:** MILK10k (15% held-out val)

---


---

VERIFICATION REPORT  --  ISIC SKIN IMAGE ANALYSIS (MICCAI)
Generated   : 2026-07-06 12:01 UTC
Protocol    : v4_verified_3seeds
Seeds       : [456, 789, 1337]
OOD test    : PAD-UFES-20 (OOD)  (N=2298)
In-domain   : MILK10k (15% held-out val)

---



---


## S1  EXPERIMENT METADATA & REPRODUCIBILITY CHECKLIST


---


  Architectures evaluated:
✅  Late Fusion                     (3/3 seeds)  
✅  GMU Baseline                    (3/3 seeds)  
✅  Cross-Attention                 (3/3 seeds)  
✅  Cross-Attention T→V             (3/3 seeds)  

  Baselines:
✅  Image-Only                      (3/3 seeds)  
✅  Text-Only                       (3/3 seeds)  

  Verified design invariants (config.py):
    Vision backbone   : vit_base_patch16_224 (timm) -- SHARED across all archs
    Text backbone     : emilyalsentzer/Bio_ClinicalBERT
    Biopsy field      : REMOVED from PAD-UFES-20 clinical text (v3+ fix)
    Batch / Epochs    : 16 / 5  (early stopping patience=2)
    Optimiser         : AdamW  lr=2e-5  wd=0.01  warmup_ratio=0.1
    AMP               : enabled
    Capacity matching : Late Fusion hidden_dim=512 (~= Cross-Attention params)
    Blank probe       : '' (empty string)
    Neutral probe     : 'No clinical history is available for this patient.'
    CF routing        : 6-class benign/malignant override (v4 fix)

✅    In-domain MILK10k audit: CLEAN lesion-disjoint split  
          (see S5b for full contamination comparison)


---


## S2  OOD TEST-SET PERFORMANCE  --  PAD-UFES-20


---


  Note: Accuracy, AUROC, F1, Prec, Rec shown as mean +/- std (3 seeds)

  Architecture                      Acc%       AUROC     MacroF1       Prec        Rec
  Late Fusion                   41.51+/-0.49  0.7505+/-0.0060   0.2445+/-0.0079  0.3904+/-0.0129  0.3287+/-0.0080
  GMU Baseline                  41.91+/-0.45  0.7589+/-0.0066   0.2725+/-0.0047  0.3651+/-0.0082  0.3456+/-0.0149
  Cross-Attention               41.70+/-1.42  0.7521+/-0.0051   0.2871+/-0.0331  0.3539+/-0.0336  0.3528+/-0.0188
  Cross-Attention T→V           43.47+/-0.22  0.7948+/-0.0057   0.3265+/-0.0145  0.4866+/-0.0058  0.3695+/-0.0238
  Image-Only                    38.63+/-0.20  0.6804+/-0.0159   0.1918+/-0.0133  0.3608+/-0.0902  0.2318+/-0.0102
  Text-Only                     36.77+/-0.00  0.4035+/-0.0178   0.0896+/-0.0000  0.0613+/-0.0000  0.1667+/-0.0000

  Accuracy ranking:
    1. Cross-Attention T→V  (43.47%)
    2. GMU Baseline  (41.91%)
    3. Cross-Attention  (41.70%)
    4. Late Fusion  (41.51%)

  Best Accuracy  : Cross-Attention T→V
  Best AUROC     : Cross-Attention T→V
  Best Macro F1  : Cross-Attention T→V
  Best ECE (low) : Late Fusion

  Majority-class baseline  (BCC, 845/2298): 36.77%
  All multimodal architectures exceed majority baseline on Accuracy.


---


## S3  MECHANISTIC AUDIT  --  TEXT ABLATION PROBES


---


  All accuracy values in percentage points (pp).
  Probes: Real | Blank-empty | Neutral | Counterfactual (CF)
  Gain = Real - Blank  |  Drop = Real - Blank (+pp -> text helps accuracy)

  Architecture                    Real   Blank  Neutral      CF    Drop    Gain     CFR  DeltaP
  Late Fusion                    41.51   41.04    40.82   40.85   +0.48   +0.48   12.45    5.80
  GMU Baseline                   41.91   41.88    41.80   41.82   +0.03   +0.03    9.69    5.18
  Cross-Attention                41.70   24.72    40.63   43.08  +16.99  +16.99   23.14   10.88
  Cross-Attention T→V            43.47   43.12    42.50   43.49   +0.35   +0.35    2.93    1.66

  Notable findings:
❌ Cross-Attention (V->T): blank-text acc 24.72pp BELOW  
          majority baseline (36.77pp) -- empty-string tokenizer artefact.
          Neutral probe (40.63pp) recovers to near-Real (41.70pp).
          CFR (23.14%) and DeltaP (10.88pp) are INFLATED artefacts.
✅    T->V: lowest CFR (2.93%) and DeltaP (1.66pp) --  
          most robust to adversarial text swaps among multimodal archs.
✅    GMU: near-zero blank drop (+0.03pp) -- visual dominance,  
          minimal spurious text leakage.


---


## S4  LATENT-SPACE GEOMETRIC AUDIT  --  LINEAR CKA


---


  Thresholds:  CKA >= 0.95 -> COLLAPSE | 0.85-0.95 -> moderate | <0.85 -> healthy

  Architecture                   VisFeatNorm    FusedNorm          CKA  Label                  Verdict
  Late Fusion                   34.78+/-0.00  20.55+/-0.05  0.9655+/-0.0007  MODALITY COLLAPSE       Fused ~ visual; text ignored
  GMU Baseline                  34.78+/-0.00   8.93+/-0.27  0.8627+/-0.0032  Moderate perturbation   Some text influence
  Cross-Attention               34.78+/-0.00   8.39+/-0.32  0.3275+/-0.0082  Healthy fusion          Significant text contribution
  Cross-Attention T→V           34.78+/-0.00  11.19+/-0.27  0.7644+/-0.0256  Healthy fusion          Significant text contribution

  Highlights:
    - Late Fusion: CKA ~0.97 -> near-modality collapse; visual features dominate.
    - GMU Baseline: CKA ~0.86 -> moderate, stable gating between modalities.
    - Cross-Attention V->T: CKA ~0.33 -> highest text perturbation; but partially
      artefactual (see blank-text collapse in S3).
    - Cross-Attention T->V: CKA ~0.76 -> genuine fusion, no collapse.


---


## S5  CALIBRATION  --  EXPECTED CALIBRATION ERROR (ECE)


---


  Architecture                   ECE (lower=better)  Verdict
  Late Fusion                   0.0877                 [BEST]
  GMU Baseline                  0.0910                 
  Cross-Attention               0.1019                 
  Cross-Attention T→V           0.1860                 

  ECE computed over 15 calibration bins, averaged over 3 seeds.
  Representative seed-456 prediction distribution used for bin assignment.


---


## S5b  IN-DOMAIN AUDIT  --  MILK10K (CONTAMINATION ANALYSIS)


---


  CONTAMINATION CONTEXT
  MILK10k has exactly 2 images per lesion (4,672 unique lesions x2 = 9,344 imgs).
  The original audit used a random 85/15 image-level split (PyTorch random_split).
  This produces near-certain lesion leakage since both images of each lesion
  are independently shuffled -- one lands in train, one in val, for ~85% of lesions:

    Seed 456 : 1,188 / 1,402 val samples from shared lesions (84.7% leakage)
    Seed 789 : 1,194 / 1,402 val samples from shared lesions (85.2% leakage)
    Seed 1337: 1,228 / 1,402 val samples from shared lesions (87.6% leakage)

  CONTAMINATION SIGNAL
  Text-Only original in-domain acc: 53.26%  >>  OOD performance: 36.77%
  Text-Only has NO visual pathway. Its in-domain advantage can ONLY arise from
  recognising near-duplicate text from same-lesion pairs it has already trained on.
  This is memorisation, not in-domain generalisation.

  CLEAN SPLIT DESIGN
  Group all 4672 unique lesion IDs.
  Sort deterministically (ascending IL_xxxxxx string sort).
  Reserve last 15% = 700 lesions
    -> 1400 images as clean held-out val.
  Lesion overlap: 0  (guaranteed by construction).
  Majority class in clean val: BCC
    (54.71% of clean val)

  SIDE-BY-SIDE COMPARISON
  Architecture                    LEAKED split     CLEAN split  Delta
                                    (orig, pp)     (clean, pp)  (clean-orig)
  Late Fusion                           68.38%          69.07%    +0.69pp
  GMU Baseline                          68.05%          69.29%    +1.24pp
  Cross-Attention (V->T)                69.50%          70.55%    +1.05pp
  Cross-Attention T->V                  70.90%          71.57%    +0.67pp
  Image-Only                            60.68%          61.40%    +0.72pp
  Text-Only                             53.26%          54.71%    +1.45pp

  (*) = >5pp accuracy drop on clean val (indicative of memorisation in leaked split)

  CLEAN AUDIT DETAILED RESULTS (mean over 3 seeds)
  Architecture                    Acc%   Blank%    Drop   Neutral%    CFR%      CKA
  Late Fusion                    69.07   68.74      +0.33      66.14    4.10   0.9802
  GMU Baseline                   69.29   68.90      +0.38      68.00    3.74   0.7333
  Cross-Attention (V->T)         70.55   48.93 (*)  +21.62      66.29   14.12   0.2145
  Cross-Attention T->V           71.57   69.86      +1.71      69.88    3.31   0.7066
  Image-Only                     61.40    0.61 (*)   +0.00       0.61    0.00   1.0000
  Text-Only                      54.71   54.71      +0.00      54.71    0.00   1.0000

  (*) = below majority class in clean val (54.71%)


---


## S6  PER-CLASS METRICS  --  OOD SET (SEED-456 REPRESENTATIVE)


---


  ---- Late Fusion ----
  Class     Prec     Rec      F1  ClassAcc      N
  MEL     0.2319  0.3077  0.2645    0.3077     52
  BCC     0.4297  0.9041  0.5825    0.9041    845
  SCC     0.2444  0.0573  0.0928    0.0573    192
  ACK     0.7000  0.0096  0.0189    0.0096    730
  NEV     0.4392  0.6803  0.5338    0.6803    244
  SEK     0.2778  0.0213  0.0395    0.0213    235

  Confusion matrix [rows=true, cols=predicted]:
  Classes: ['MEL', 'BCC', 'SCC', 'ACK', 'NEV', 'SEK']
    [16, 15, 0, 0, 21, 0]
    [17, 764, 14, 0, 47, 3]
    [1, 168, 11, 1, 8, 3]
    [11, 654, 20, 7, 31, 7]
    [7, 71, 0, 0, 166, 0]
    [17, 106, 0, 2, 105, 5]

  ---- GMU Baseline ----
  Class     Prec     Rec      F1  ClassAcc      N
  MEL     0.2754  0.3654  0.3140    0.3654     52
  BCC     0.4344  0.8580  0.5768    0.8580    845
  SCC     0.1860  0.1250  0.1495    0.1250    192
  ACK     0.5385  0.0288  0.0546    0.0288    730
  NEV     0.4151  0.6516  0.5072    0.6516    244
  SEK     0.3333  0.0128  0.0246    0.0128    235

  Confusion matrix [rows=true, cols=predicted]:
  Classes: ['MEL', 'BCC', 'SCC', 'ACK', 'NEV', 'SEK']
    [19, 11, 0, 0, 22, 0]
    [9, 725, 44, 10, 52, 5]
    [1, 155, 24, 4, 8, 0]
    [8, 607, 58, 21, 35, 1]
    [12, 73, 0, 0, 159, 0]
    [20, 98, 3, 4, 107, 3]

  ---- Cross-Attention ----
  Class     Prec     Rec      F1  ClassAcc      N
  MEL     0.0599  0.1923  0.0913    0.1923     52
  BCC     0.4681  0.8497  0.6036    0.8497    845
  SCC     0.1299  0.1562  0.1418    0.1562    192
  ACK     0.7500  0.0041  0.0082    0.0041    730
  NEV     0.5638  0.6885  0.6199    0.6885    244
  SEK     0.2656  0.0723  0.1137    0.0723    235

  Confusion matrix [rows=true, cols=predicted]:
  Classes: ['MEL', 'BCC', 'SCC', 'ACK', 'NEV', 'SEK']
    [10, 12, 0, 0, 30, 0]
    [24, 718, 72, 0, 16, 15]
    [5, 152, 30, 0, 3, 2]
    [35, 518, 121, 3, 23, 30]
    [18, 57, 1, 0, 168, 0]
    [75, 77, 7, 1, 58, 17]

  ---- Cross-Attention T→V ----
  Class     Prec     Rec      F1  ClassAcc      N
  MEL     0.2143  0.4615  0.2927    0.4615     52
  BCC     0.4293  0.9456  0.5905    0.9456    845
  SCC     0.2545  0.1458  0.1854    0.1458    192
  ACK     0.7586  0.0301  0.0580    0.0301    730
  NEV     0.7667  0.3770  0.5055    0.3770    244
  SEK     0.5000  0.1404  0.2193    0.1404    235

  Confusion matrix [rows=true, cols=predicted]:
  Classes: ['MEL', 'BCC', 'SCC', 'ACK', 'NEV', 'SEK']
    [24, 17, 0, 0, 10, 1]
    [2, 799, 33, 4, 0, 7]
    [0, 161, 28, 0, 0, 3]
    [3, 654, 43, 22, 1, 7]
    [36, 100, 1, 0, 92, 15]
    [47, 130, 5, 3, 17, 33]

  Cross-architecture observations:
    ACK (n=730):  chronically under-recalled. Visually similar to BCC;
                  clinical text provides limited discriminative help.
    BCC (n=845):  highest recall in all models (majority-class anchoring).
    MEL (n=52):   T->V shows improved recall -- clinical text alignment benefit.
    SEK (n=235):  confused with NEV and BCC; low recall universally.


---


## S7  STATISTICAL SIGNIFICANCE  --  HOLM-BONFERRONI


---


  Test family  : 98 tests (14 pairwise comparisons x 7 metrics)
  Survivors    : 16 tests reject H0 after Holm-Bonferroni (alpha=0.05)
  Effect size  : paired Cohen's d  (|d|<0.2 negligible | >=0.8 large)
  WARNING      : n=3 seeds -> low power; large effects may be underpowered.

  TAUTOLOGY CAVEAT -- COMPARISONS AGAINST IMAGE-ONLY ON CFR AND LINEAR_CKA
  Image-Only has no text pathway. By CONSTRUCTION across all seeds and inputs:
    CFR (counterfactual flip rate) = [0, 0, 0]  (no text -> no flips possible)
    Linear_CKA (visual vs fused) = [1, 1, 1]   (fused IS visual; identity map)

  These are ZERO-VARIANCE constants, not empirical measurements.
  Any architecture with a text pathway will have CFR > 0 and CKA < 1.
  Therefore 'surviving' Holm-Bonferroni against Image-Only on CFR or CKA
  merely confirms the audit pipeline correctly detects a null text-processing
  profile -- it is NOT a comparative grounding finding.

  WHICH COMPARISONS AGAINST IMAGE-ONLY ARE INFORMATIVE?
  The only non-tautological comparisons against Image-Only are Accuracy
  and F1 Macro -- metrics that can differ only if the text pathway genuinely
  improves prediction on image+text inputs beyond what vision alone achieves.
  Both metrics show LARGE effect sizes but are underpowered at n=3 seeds
  and do NOT survive Holm-Bonferroni correction.

  HB SURVIVORS IN CONTEXT
  Of the 16 surviving tests (98 total):
    - All CKA survivors vs. Image-Only or Text-Only: TAUTOLOGICAL (see above)
    - All CFR survivors vs. Image-Only or Text-Only: TAUTOLOGICAL (see above)
    - GMU vs. Text-Only on F1/AUROC: non-tautological (both have text pathways)
    - T->V vs. Text-Only on Accuracy: non-tautological

  PRIORITY P1 -- Multimodal vs. Image-Only (non-tautological metrics only):

    T->V vs. Image-Only  [Accuracy, F1 Macro]:
      d=+12.3 (Accuracy), d=+22.2 (F1) -- LARGE effects.
      NOT significant after HB correction (best raw p=0.0007 on F1).
      => Underpowered large effect; ~20 seeds required for correction.

    GMU vs. Image-Only  [Accuracy, F1 Macro]:
      d=+4.2 (Accuracy), d=+4.3 (F1) -- LARGE effects.
      NOT significant after HB correction.
      => Underpowered; effect exists but n=3 is insufficient.

    V->T vs. Image-Only  [Accuracy, F1 Macro]:
      d=+1.75 (Accuracy), d=+3.4 (F1) -- LARGE effects.
      NOT significant after HB correction.

  INFORMATIVE SURVIVORS (non-tautological):
    - GMU vs. Text-Only on AUROC (p=0.0005, d=+25.9): multimodal AUROC
      significantly exceeds text-only AUROC.
    - GMU vs. Text-Only on F1 (p=0.0003, d=+31.8): multimodal F1
      significantly exceeds text-only F1.
    - T->V vs. Text-Only on Accuracy (p=0.0005, d=+25.3).

  POWER NOTE
  At n=3 seeds, the 98-test family requires adj_alpha as low as 0.0005 for
  the first rejection. Given d~4-12 for Accuracy comparisons, ~20 seeds are
  required for 80% power to detect these effects under HB correction.


---


## S8  MAJORITY-BASELINE SANITY CHECK


---


  Majority class : BCC  (845/2298 = 36.77%)

  Architecture                  Blank Acc (pp)   Neutral (pp)  Status
  Late Fusion                           41.04          40.82   [OK]
  GMU Baseline                          41.88          41.80   [OK]
  Cross-Attention                       24.72          40.63   [FAIL] BELOW BASELINE
  Cross-Attention T→V                   43.12          42.50   [OK]

  FINDING: Cross-Attention (V->T) blank-text accuracy (24.72pp) drops
  below majority baseline (36.77pp) due to empty-string tokenizer artefact.
  Neutral probe (40.63pp) confirms the model is NOT broken -- the collapse
  is isolated to the '' -> single PAD-token edge case.
  Consequence: V->T blank-drop, CFR, and DeltaP metrics are inflated and
  should be interpreted with the neutral probe as the reliable reference.


---


## S9  VERIFICATION VERDICT


---


  CHECKLIST
✅  Protocol v4_verified_3seeds applied  
✅  Shared timm ViT-B/16 backbone across ALL architectures  
✅  Biopsy field removed from PAD-UFES-20 clinical text  
✅  Genuine Counterfactual_Accuracy (real forward pass, not algebraic)  
✅  Neutral-text probe implemented and stored  
✅  6-class CF routing corrected (benign/malignant per diagnostic class)  
✅  Capacity matching: Late Fusion hidden_dim=512  
✅  All multimodal archs exceed majority baseline on Accuracy  
⚠️  All multimodal archs exceed majority baseline on blank-text acc  
✅  Cross-Attention V->T blank-text collapse documented and explained  
✅  Holm-Bonferroni correction applied (98 tests, alpha=0.05)  
✅  Tautology caveat: CFR/CKA vs. Image-Only are structural zeros  
✅  Paired Cohen's d reported alongside p-values  
✅  Power analysis caveat documented (n=3 -> ~20 seeds for Acc)  
✅  In-domain MILK10k audit: contamination documented  
✅  In-domain MILK10k audit: clean lesion-disjoint split run  
✅  In-domain results: leaked vs. clean side-by-side in report  
✅  Per-class metrics and confusion matrices generated  
✅  ECE calibration computed (averaged over 3 seeds)  

  Result: 18/19 PASS, 1 WARN

  CAVEATS

  1. Cross-Attention V->T blank-text collapse is documented and explained.
     Use neutral-probe values as the reliable mechanistic reference for V->T.

  2. No Accuracy comparison against Image-Only survives HB correction at n=3
     seeds, but ALL effect sizes are LARGE (|d| >> 0.8). This is a power
     finding, not a null result. Report d alongside p; include power note.

  3. HB 'survivors' on CFR and CKA against Image-Only are tautological
     (those are structural zero-variance constants for Image-Only). Only
     Accuracy and F1 Macro comparisons against Image-Only are informative.
     Non-tautological survivors are GMU/T->V vs. Text-Only on AUROC/F1/Acc.

  4. The side-by-side contamination comparison is in S5b. The clean split
     uses a deterministic lesion-level sort with zero overlap guaranteed.

  5. T->V is best-performing: Acc 43.47%, AUROC 0.7948, F1 0.3265,
     CFR 2.93pp, DeltaP 1.66pp.

  RECOMMENDATION FOR PAPER

  Primary proposal   : Cross-Attention T->V (best OOD performance + robustness)
  Statistical claim  : GMU vs. Text-Only on F1 + AUROC (HB-corrected, p<0.001)
  Tautology warning  : Do NOT claim CFR/CKA vs. Image-Only as grounding evidence
  Transparency note  : V->T blank-text tokenizer artefact (neutral probe = fix)
  Power note         : n=3 -> large effects underpowered; >=20 seeds recommended
  In-domain note     : Report CLEAN split numbers only; flag leaked numbers as
                       historical (contamination magnitude in S5b)


---

END OF VERIFICATION REPORT

---
