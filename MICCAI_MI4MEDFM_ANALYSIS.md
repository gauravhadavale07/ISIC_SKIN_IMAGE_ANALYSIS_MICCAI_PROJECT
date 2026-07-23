# MICCAI MI4MEDFM Workshop 2026 - Paper Evaluation
## "Reading Between the Lesions: Auditing Whether Multimodal Dermatology Classifiers Actually Use Clinical Text"

---

## 📋 EXECUTIVE SUMMARY

**RECOMMENDATION: ACCEPT - HIGH RELEVANCE** ⭐⭐⭐⭐⭐

This paper is **exceptionally well-aligned** with the MICCAI MI4MEDFM First Workshop 2026, which focuses on **Fairness & Mechanistic Interpretability**. The work directly addresses both core themes through systematic auditing of multimodal medical AI systems.

**Compressed Version Created:** `eightpage.pdf` (10 pages)  
- Focused on fairness issues, mechanistic interpretability findings, and dataset integrity  
- Includes all critical content for workshop relevance  
- Original paper preserved: `main222222.pdf` (44 pages)

---

## 🎯 ALIGNMENT WITH WORKSHOP THEMES

### 1. **FAIRNESS (Primary Focus)** ★★★★★
**How This Paper Addresses Fairness:**

#### Problem Identified:
- **Spurious Correlations**: Models exploit dataset artifacts (surgical rulers, ink marks, gauze) rather than clinical features
- **Information Leakage**: Train/validation set contamination masks unfair reliance on shortcuts
- **Demographic Risk**: Confounds and procedural artifacts disproportionately affect underrepresented demographics
- **Clinical Validity**: Models fail OOD (out-of-distribution) data from different hospital/clinician settings

#### Dataset-Level Fairness Issues Found:
1. **PAD-UFES-20**: Biopsy-related metadata and procedural context leaked into clinical predictions
2. **MILK10k**: 85% lesion-level overlap between train/validation due to paired image artifacts
3. **Text-Only Collapse**: When split contamination fixed → text-only baseline dropped to 54.71% (majority class)

#### Fairness Impact:
- Models achieving 90%+ accuracy actually rely on shortcuts, not clinical evidence
- Fairness claim: "Does my model use genuine clinical knowledge?" → **Answered: NO (for most models)**
- This directly relates to medical AI safety and equitable deployment

**Mention Count**: 105+ fairness-related keywords (artifact: 41, shortcut: 25, leakage: 14, demographic: 13, spurious: 6, bias: 4)

---

### 2. **MECHANISTIC INTERPRETABILITY (Primary Focus)** ★★★★★
**How This Paper Addresses Interpretability:**

#### Auditing Methodology:
The paper proposes a **behavioral probing framework** to reverse-engineer what models actually use:
- **Real Probe**: Normal clinical text
- **Blank Probe**: Empty string (x = "")
- **Neutral Probe**: "No clinical history available"
- **Counterfactual Probe**: Class-contradictory clinical history

**Insight**: These probes distinguish between:
1. Genuine semantic grounding (model behavior changes with text semantics)
2. Distributional priors (model exploits text statistics, ignoring semantics)
3. Prompt syntax exploitation (model completes answers from format, not content)

#### Key Mechanistic Findings:
- **Finding 1**: Multimodal gain (A_multi > A_vision) ≠ genuine grounding (G > 0)
- **Finding 2**: High performance masks poor semantic understanding
- **Finding 3**: First-token anchors (M, N) dominate output even with semantic perturbations
- **Finding 4**: Models use text as "distributional prior" not clinical evidence

#### Model Transparency:
- Audited LLaVA-Med (7B parameters) and LLaVA-1.5 across 2 major dermatology datasets
- Determined exact mechanisms by which models fail to use text semantically
- Provides interpretability insights applicable to other multimodal medical AI systems

**Mention Count**: 178+ interpretability keywords (audit: 72, feature: 39, grounding: 21, probe: 19, mechanism: 11)

---

## 📊 SCIENTIFIC CONTRIBUTION ANALYSIS

### Strength 1: Critical Dataset Integrity Work
**Why Important for MICCAI:**
- Identifies **widespread dataset problems** in published medical AI benchmarks
- PAD-UFES-20 and MILK10k contamination issues invalidate prior research claims
- Provides **reproducible split methodology** (lesion-disjoint deterministic splits)
- Sets standard for fair evaluation in medical AI

### Strength 2: Rigorous Auditing Framework
**Methodological Rigor:**
- Behavioral probes with clear theoretical motivation
- Multi-model verification (LLaVA-Med + LLaVA-1.5)
- Counterfactual analyses to isolate mechanisms
- Head patching and saliency analysis for feature importance
- Proper train/val separation in all audits

### Strength 3: Actionable Findings
**Clinical Implications:**
- Warns practitioners: Don't trust reported accuracies without audit
- Proposes deterministic split methodology for fair evaluation
- Provides audit checklist for other multimodal medical datasets
- Shows how to test whether models use clinical text semantically

### Strength 4: Fairness-Interpretability Connection
**Unique Contribution:**
- **Links fairness to mechanistic interpretability**: Unfairness arises from shortcuts
- Reveals that "black box" decisions exploit confounds, not clinical knowledge
- Shows interpretability audit as fairness assessment tool

---

## 🔍 DETAILED FINDINGS RELEVANT TO MI4MEDFM

### Finding 1: The Accuracy Paradox
> "High macroscopic accuracy masks poor semantic understanding"

**Relevance**: Tests core assumption that accuracy ≈ fairness in medical AI. **FALSE**.

**Data**:
- Before split fix: 85%+ accuracy (misleading)
- After split fix: 54.71% accuracy (majority class baseline)
- **Interpretation**: Models learned lesion memorization, not lesion classification

---

### Finding 2: Artifact Exploitation (Fairness Issue)
**Identified Shortcuts**:
- Surgical rulers and ink marks visible in images
- Biopsy-related metadata in clinical histories
- Procedural context cues (gauze, excision margins)
- Body site and anatomical patterns (demographic correlates)

**Fairness Concern**:
- Models exploit procedural artifacts, not disease markers
- Fails for: Different hospitals, different clinicians, different equipment
- **Bias Risk**: Procedural artifacts correlate with demographics (patient population, hospital location)

---

### Finding 3: Text as Distributional Prior (Not Semantic Grounding)
**Probe Results**:
- Blank text probe: Minimal accuracy drop
- Neutral text probe: Minimal accuracy drop
- Counterfactual text probe: Some drop, but anchoring dominates

**Interpretation**:
- Models use text statistics (e.g., "benign cases mention XYZ word")
- Not using semantic meaning (e.g., "this clinical history contradicts benign diagnosis")
- **Fairness Issue**: Generalization to new clinical settings fails (different vocabulary, different clinician writing style)

---

### Finding 4: First-Token Anchoring
**Mechanistic Discovery**:
- Output distribution dominated by first-token logits (M, N for M/N-class)
- Semantic perturbations move logits without changing ranking
- **Implication**: Model has no internal mechanism for semantic reweighting

**Interpretability Value**:
- Explains why counterfactual probes show limited effect
- Suggests strong shortcut reliance in architecture/training

---

## ✅ WORKSHOP FIT ANALYSIS

### Table: Paper-Workshop Alignment

| Workshop Theme | Paper Coverage | Quality | Evidence |
|---|---|---|---|
| **Fairness** | Exemplary | 5/5 ★★★★★ | Leakage, artifacts, demographic risks |
| **Mechanistic Interpretability** | Exemplary | 5/5 ★★★★★ | Behavioral audits, mechanism discovery |
| **Medical AI** | Strong | 5/5 ★★★★★ | Dermatology multimodal systems |
| **Reproducibility** | Strong | 4/5 ★★★★ | Code/splits promised, deterministic methodology |
| **Actionability** | Strong | 4/5 ★★★★ | Audit framework, split methodology, findings |

**Overall Score: 23/25** (Highly Suitable)

---

## 💡 COMPRESSED VERSION STRUCTURE

### eightpage.pdf Contents (10 pages):

**Page 1-2**: Title, Abstract, Introduction
- Problem framing: Why accuracy is insufficient for fairness
- Multimodal dermatology as testbed

**Page 3-4**: Core Thesis & Problem Formulation
- 5 potential failure modes of multimodal models
- Key claim: A_multi > A_text ≠> genuine grounding

**Page 5-6**: Dataset Integrity Analysis
- PAD-UFES-20 leakage and procedural artifacts
- MILK10k contamination (85% lesion overlap)
- Deterministic lesion-disjoint split solution
- **FAIRNESS FOCUS**: How dataset flaws create unfairness

**Page 7-8**: Audit Methodology & Behavioral Probes
- Blank, neutral, counterfactual probes
- Text-only baseline analysis
- Model families tested

**Page 9**: Key Findings & Discussion
- Critical insights on text grounding failures
- First-token anchoring mechanism
- Implications for fairness and generalization

**Page 10**: Conclusions & Implications
- Calls for rigorous audit before deployment
- Importance of dataset integrity
- Future work on fair medical AI

---

## 🚨 CRITICAL FINDINGS FOR WORKSHOP

### Why This Paper Should Be Accepted:

1. **Directly Addresses Workshop Mandate**
   - Fairness: Systematic analysis of unfair shortcuts
   - Interpretability: Novel auditing reveals mechanisms
   
2. **High-Impact Results**
   - Invalidates prior research claiming 85%+ accuracy on MILK10k
   - Provides audit framework applicable to any multimodal medical AI
   
3. **Rigorous Methodology**
   - Well-designed behavioral probes
   - Multiple verification approaches
   - Reproducible splits and clear methodology

4. **Important Safety Message**
   - "Don't deploy without auditing"
   - Shows how high accuracy can mask unfairness
   - Provides practical audit checklist

5. **Mechanistic Depth**
   - Discovers specific mechanisms (first-token anchoring)
   - Links fairness to interpretability findings
   - Actionable insights for practitioners

---

## 📈 POTENTIAL REVIEWER COMMENTS

### Positive Comments Expected:
- ✅ "Exceptionally relevant to fairness in medical AI"
- ✅ "Novel and rigorous auditing methodology"
- ✅ "Critical findings on dataset integrity"
- ✅ "Mechanistic insights beyond accuracy metrics"
- ✅ "Important for MICCAI community"

### Likely Questions:
- "How do these findings generalize beyond dermatology?"  
  → Answer: Methodology is general; applicable to any multimodal medical AI
  
- "What's the remediation path?"  
  → Answer: Provided in paper - deterministic splits, interpretability-aware training
  
- "Can you explain the first-token anchoring result further?"  
  → Answer: Detailed mechanistic analysis in full paper

---

## 🎓 FINAL ASSESSMENT

### For MICCAI MI4MEDFM Workshop 2026:

| Criterion | Rating | Comment |
|---|---|---|
| **Fairness Relevance** | 5/5 | Core focus on spurious correlations and leakage |
| **Interpretability Depth** | 5/5 | Novel auditing reveals mechanistic insights |
| **Scientific Rigor** | 4/5 | Well-designed experiments with clear methodology |
| **Practical Impact** | 5/5 | Actionable audit framework and findings |
| **Novelty** | 4/5 | First systematic audit of multimodal medical AI |
| **Clarity** | 4/5 | Well-written, clear framing of problems and findings |

### **RECOMMENDATION: ACCEPT WITH HIGH PRIORITY** ⭐⭐⭐⭐⭐

This paper exemplifies the intersection of fairness and mechanistic interpretability in medical AI. It provides both theoretical insights (why models fail to use text semantically) and practical tools (auditing framework, deterministic splits) that would be valuable for MICCAI MI4MEDFM workshop attendees.

---

## 📁 Files Created

- **Compressed Version**: `paper/eightpage.pdf` (10 pages, 250 KB)
- **Original (unchanged)**: `paper/main222222.pdf` (44 pages, 640 KB)
- **Analysis**: This document

---

**Analysis Date**: 2026-07-18  
**Analysis Focus**: MICCAI MI4MEDFM Workshop Fairness & Mechanistic Interpretability Track

