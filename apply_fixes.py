import re

with open('paper/main2.txt', 'r') as f:
    text = f.read()

# 1. Delete lines 7-96 (Master Manuscript Draft to "Abstract")
# Let's find "Master Manuscript Draft for MICCAI/LNCS Reduction" up to right before "1\n\nAbstract"
text = re.sub(r'Master Manuscript Draft for MICCAI/LNCS Reduction.*?Abstract', 'Abstract', text, flags=re.DOTALL)
text = re.sub(r'Anonymous\n\nAnonymous Organization\n\n', 'Anonymous\nAnonymous Organization\n\n', text)

# Delete Table 1 (lines 361-397 roughly)
# Starts with "{DROP-IN Table 1: PAD-UFES-20 structural integrity checklist}"
# Ends before "2.3 MILK10k Contamination"
text = re.sub(r'\{DROP-IN Table 1: PAD-UFES-20 structural integrity checklist\}.*?(?=2\.3 MILK10k Contamination)', '', text, flags=re.DOTALL)

# Delete Table 3 (Activation patching)
text = re.sub(r'\{DROP-IN Table 3: Activation patching head recovery\}.*?(?=4\.3 Top-K Sparse Autoencoder)', '', text, flags=re.DOTALL)

# Delete Table 4 (Attention-head ablation)
text = re.sub(r'\{DROP-IN Table 4: Attention-head ablation logit shift\}.*?(?=4\.4 Generative LVLM)', '', text, flags=re.DOTALL)

# Delete Appendices (2025-end)
text = re.sub(r'A\n\nDrop-In Table Templates.*', '', text, flags=re.DOTALL)

# 2. Architecture & Mismatch Fixes
# Scaffold Ablation (1886-1898)
text = text.replace('theorized that the Vision-to-Text fusion model bypasses true semantic alignment', 
                    'theorized that the Cross-Attention T→V fusion model bypasses true semantic alignment')

# Biopsy Leak (1208-1209)
text = text.replace('SAE on the residual/fused stream of Cross-Attention T → V',
                    'SAE on the residual/fused stream of Cross-Attention T → V (note that Task 20 subsequently evaluates biopsy leakage using the V → T architecture)')

# Cropping Overclaim (1909-1920)
text = text.replace('This proves causally that peripheral artifacts', 'This suggests that peripheral artifacts')
text = text.replace('which perfectly preserve the core pathology', 'which attempt to preserve the core pathology')

# LVLM Claims
text = text.replace('syntactic prior collapse and visual grounding failure', 'prompt and answer-order sensitivity')
text = text.replace('syntactic prior collapse', 'prompt sensitivity')
text = text.replace('visual grounding failure', 'answer-order sensitivity')

# 3. Scientific Revisions & Limitations
# We'll add a limitations section at the end (before references if there are any, or just at the end)
limitations = """
7 Limitations and Future Work

While this audit identifies several vulnerabilities in multimodal dermatology classifiers, we acknowledge limitations in our experimental controls. First, the Top-K Sparse Autoencoder (SAE) analysis lacks rigorous reporting on variance explained, dead-feature rates, and test-set splits for activation distributions. Second, our causal interventions (e.g., Feature 1449 knockout) provide preliminary mechanistic evidence but require matched activation-magnitude controls and sign-reversal tests to fully isolate the causal pathways. Finally, the LVLM audit findings (including apparent answer-order sensitivity and length-penalty effects) are constrained by a small sample size (e.g., 52 MEL images) and require more extensive controls—such as no-image baselines, fully randomized answer ordering, and length-normalized likelihoods—to confirm visual grounding failure conclusively.
"""
text = text + "\n" + limitations

# Soften length-penalty claim
text = text.replace('This proves a length-penalty paradox', 'This suggests a plausible length-penalty paradox, though full length-normalized likelihood regression is required to prove it conclusively')

# 4. Data & Metric Inconsistencies
# MILK10k split:
text = text.replace('leakage artificially inflated validation accuracy', 'leakage did not artificially inflate validation accuracy (clean split performance was actually higher)')

# Unify Baselines:
# V->T to 41.70%, T->V to 43.47%
text = text.replace('40.34%', '41.70%')
text = text.replace('43.34%', '43.47%')

# PAD-UFES Table fix
pad_table_broken_regex = r'AUROC\n\nLate Fusion.*?0\.0000\n\n'
pad_table_fixed = """AUROC, Macro-F1, Precision, Recall
Late Fusion: 41.51 ± 0.49 | 0.7505 ± 0.0060 | 0.2445 ± 0.0079 | 0.3904 ± 0.0129 | 0.3287 ± 0.0080
GMU: 41.91 ± 0.45 | 0.7589 ± 0.0066 | 0.2725 ± 0.0047 | 0.3651 ± 0.0082 | 0.3456 ± 0.0149
Cross-Attention V->T: 41.70 ± 1.42 | 0.7521 ± 0.0051 | 0.2871 ± 0.0331 | 0.3539 ± 0.0336 | 0.3528 ± 0.0188
Cross-Attention T->V: 43.47 ± 0.22 | 0.7948 ± 0.0057 | 0.3265 ± 0.0145 | 0.4866 ± 0.0058 | 0.3695 ± 0.0238
Image-Only: 38.63 ± 0.20 | 0.6804 ± 0.0159 | 0.1918 ± 0.0133 | 0.3608 ± 0.0902 | 0.2318 ± 0.0102
Text-Only: 36.77 ± 0.00 | 0.4035 ± 0.0178 | 0.0896 ± 0.0000 | 0.0613 ± 0.0000 | 0.1667 ± 0.0000

"""
text = re.sub(pad_table_broken_regex, pad_table_fixed, text, flags=re.DOTALL)

# DDI dark-skin
text = text.replace('near 10.4%', '10.30%')

# 5. Overclaim Check
text = text.replace('proves', 'suggests').replace('Proves', 'Suggests')
text = text.replace('confirms', 'indicates').replace('Confirms', 'Indicates')
text = text.replace('visual evidence ignored', 'prompt sensitivity')
text = text.replace('causal collapse', 'performance degradation')

# DDI null patching clarify
text = text.replace('shows a complete absence of localized bias', 'shows this matched subset/head-patching design was underpowered or unstable')
text = text.replace('demonstrates absence of localized bias', 'shows this matched subset/head-patching design was underpowered or unstable')

# Patient histories claim
text = text.replace('Real histories should add useful patient-specific evidence', 'Real histories might add patient-specific evidence, though basic age/sex/site text might lack sufficient diagnostic value once biopsy fields are removed')

# 6. Minor Polish
# Define malignant/benign class mapping explicitly
text = text.replace('malignant/benign', 'malignant (MEL, BCC, SCC) / benign (NEV, ACK, SEK, BOD)')

# Remove project prompt reports phrasing
text = text.replace('project prompt reports', 'our empirical findings')

# Placeholders for exact prompt strings and hyperparameters
text = text.replace('Prompt strings:', 'Prompt strings: [INSERT EXACT PROMPTS]')
text = text.replace('hyperparameters:', 'hyperparameters: [INSERT EXACT HYPERPARAMETERS]')

with open('paper/main2.txt', 'w') as f:
    f.write(text)
print("Done applying fixes.")
