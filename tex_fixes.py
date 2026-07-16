import re

with open('paper/reading_between_lesions_master_draft.tex', 'r') as f:
    text = f.read()

# 1. Delete lines from Master Manuscript Draft to Abstract
text = re.sub(r'\\begin\{center\}\\large\\textbf\{Master Manuscript Draft.*?\\section\{Abstract\}', r'\\section{Abstract}', text, flags=re.DOTALL)

# Delete Table 1 (PAD-UFES-20 checklist)
text = re.sub(r'\\begin\{table\}.*?\\caption\{PAD-UFES-20 structural integrity checklist\}.*?\\end\{table\}', '', text, flags=re.DOTALL)

# Delete Table 3 (Activation patching)
text = re.sub(r'\\begin\{table\}.*?\\caption\{Activation patching head recovery\}.*?\\end\{table\}', '', text, flags=re.DOTALL)

# Delete Table 4 (Attention-head ablation)
text = re.sub(r'\\begin\{table\}.*?\\caption\{Attention-head ablation logit shift\}.*?\\end\{table\}', '', text, flags=re.DOTALL)

# Delete Appendix (Claims to Verify)
text = re.sub(r'\\appendix\n\n\\section\{Drop-In Table Templates\}.*', '', text, flags=re.DOTALL)

# 2. Architecture Fixes
text = text.replace('theorized that the Vision-to-Text fusion model bypasses', 
                    'theorized that the Cross-Attention T$\\rightarrow$V fusion model bypasses')
text = text.replace('SAE on the residual/fused stream of Cross-Attention T $\\rightarrow$ V',
                    'SAE on the residual/fused stream of Cross-Attention T $\\rightarrow$ V (note that Task 20 subsequently evaluates biopsy leakage using the V $\\rightarrow$ T architecture)')

text = text.replace('This proves causally that peripheral artifacts', 'This suggests that peripheral artifacts')
text = text.replace('which perfectly preserve the core pathology', 'which attempt to preserve the core pathology')

text = text.replace('syntactic prior collapse and visual grounding failure', 'prompt and answer-order sensitivity')
text = text.replace('syntactic prior collapse', 'prompt sensitivity')
text = text.replace('visual grounding failure', 'answer-order sensitivity')

# 3. Limitations
limitations = r"""
\section{Limitations and Future Work}

While this audit identifies several vulnerabilities in multimodal dermatology classifiers, we acknowledge limitations in our experimental controls. First, the Top-K Sparse Autoencoder (SAE) analysis lacks rigorous reporting on variance explained, dead-feature rates, and test-set splits for activation distributions. Second, our causal interventions (e.g., Feature 1449 knockout) provide preliminary mechanistic evidence but require matched activation-magnitude controls and sign-reversal tests to fully isolate the causal pathways. Finally, the LVLM audit findings (including apparent answer-order sensitivity and length-penalty effects) are constrained by a small sample size (e.g., 52 MEL images) and require more extensive controls---such as no-image baselines, fully randomized answer ordering, and length-normalized likelihoods---to confirm visual grounding failure conclusively.

"""
text = re.sub(r'\\end\{document\}', lambda m: limitations + r'\end{document}', text)

text = text.replace('This proves a length-penalty paradox', 'This suggests a plausible length-penalty paradox, though full length-normalized likelihood regression is required to prove it conclusively')
text = text.replace('leakage artificially inflated validation accuracy', 'leakage did not artificially inflate validation accuracy (clean split performance was actually higher)')

# Baselines
text = text.replace('40.34\\%', '41.70\\%')
text = text.replace('43.34\\%', '43.47\\%')

# Fix PAD-UFES table formatting
text = re.sub(r'Cross-Attention V $\\rightarrow$ T & 41\.70 \\pm 1\.42 & 0\.7521.*?0\.0188 \\\\', 
              r'Cross-Attention V $\\rightarrow$ T & 41.70 $\\pm$ 1.42 & 0.7521 $\\pm$ 0.0051 & 0.2871 $\\pm$ 0.0331 & 0.3539 $\\pm$ 0.0336 & 0.3528 $\\pm$ 0.0188 \\\\', text, flags=re.DOTALL)
text = re.sub(r'Cross-Attention T $\\rightarrow$ V & 43\.47 \\pm 0\.22 & 0\.7948.*?0\.0238 \\\\', 
              r'Cross-Attention T $\\rightarrow$ V & 43.47 $\\pm$ 0.22 & 0.7948 $\\pm$ 0.0057 & 0.3265 $\\pm$ 0.0145 & 0.4866 $\\pm$ 0.0058 & 0.3695 $\\pm$ 0.0238 \\\\', text, flags=re.DOTALL)
text = re.sub(r'Image-Only & 38\.63 \\pm 0\.20 & 0\.6804.*?0\.0102 \\\\', 
              r'Image-Only & 38.63 $\\pm$ 0.20 & 0.6804 $\\pm$ 0.0159 & 0.1918 $\\pm$ 0.0133 & 0.3608 $\\pm$ 0.0902 & 0.2318 $\\pm$ 0.0102 \\\\', text, flags=re.DOTALL)
text = re.sub(r'Text-Only & 36\.77 \\pm 0\.00 & 0\.4035.*?0\.0000 \\\\', 
              r'Text-Only & 36.77 $\\pm$ 0.00 & 0.4035 $\\pm$ 0.0178 & 0.0896 $\\pm$ 0.0000 & 0.0613 $\\pm$ 0.0000 & 0.1667 $\\pm$ 0.0000 \\\\', text, flags=re.DOTALL)

text = text.replace('near 10.4\\%', '10.30\\%')

text = text.replace('proves', 'suggests').replace('Proves', 'Suggests')
text = text.replace('confirms', 'indicates').replace('Confirms', 'Indicates')
text = text.replace('visual evidence ignored', 'prompt sensitivity')
text = text.replace('causal collapse', 'performance degradation')

text = text.replace('shows a complete absence of localized bias', 'shows this matched subset/head-patching design was underpowered or unstable')
text = text.replace('demonstrates absence of localized bias', 'shows this matched subset/head-patching design was underpowered or unstable')
text = text.replace('Real histories should add useful patient-specific evidence', 'Real histories might add patient-specific evidence, though basic age/sex/site text might lack sufficient diagnostic value once biopsy fields are removed')
text = text.replace('malignant/benign', 'malignant (MEL, BCC, SCC) / benign (NEV, ACK, SEK, BOD)')
text = text.replace('project prompt reports', 'our empirical findings')

# Insert Section Replacement
new_section = r"""\subsection{Causal Validation Experiments: Scaffold Ablation, Answer-Order Priors, and Visual Biopsy Leakage}

To rigorously test our hypotheses regarding structural padding geometry dependencies, multi-modal priors, and visual/textual grounding, we executed four targeted validation protocols:

\begin{enumerate}
\item \textbf{Cross-Attention Scaffold Ablation (Task 18):} Our initial analysis theorized that the Cross-Attention T$\rightarrow$V fusion model bypasses true semantic alignment by anchoring to static sequence padding tokens. By zeroing out the attention mask for [CLS] and [SEP] tokens, we severed this structural crutch. The ablation resulted in a negligible drop from the established 43.47\% Baseline Accuracy to an Ablated Accuracy of 42.94\% (an absolute degradation of only 0.53\%). This suggests that the T$\rightarrow$V architecture is robustly grounded in visual evidence rather than cross-modal scaffold dependency.

\item \textbf{Discriminative Matched Grounding Control (Task 21):} To assess whether the Cross-Attention T$\rightarrow$V model genuinely integrates text, we conducted a 2x2 matched visual/text grounding audit on 104 cases (52 MEL, 52 NEV). We tested real vs. blank images against both aligned and contradictory clinical histories. The model proved almost entirely text-blind: introducing contradictory text (e.g., presenting a clear MEL history for a NEV image) yielded less than a 1\% flip rate and negligible logit margin shifts (e.g., a mean shift of -0.01). The model's predictions remained firmly anchored to the visual pathway.

\item \textbf{LVLM Answer-Order Prior Audit (Task 22):} We subjected LLaVA-Med-v1.5 to the same 2x2 matched visual/text audit, adding answer-order randomization ("Is the diagnosis MEL or NEV?" vs. "NEV or MEL?"). The results revealed complete prompt and answer-order sensitivity rather than diagnostic competence. On both real and blank images, the model exhibited a 100\% adherence to the first option presented in the prompt. It completely ignored both the true visual pathology and the explicitly provided clinical text, demonstrating that its output is driven almost entirely by instruction-tuning position priors.

\item \textbf{Visual Biopsy Leak via Center Cropping (Task 20):} We extracted the Top-50 activating images for Dictionary Feature 1449, which correlated heavily with surgical rulers and skin markings. To investigate if this feature tracks peripheral artifacts rather than the melanoma pathology itself, we applied a 200x200 center crop, which was subsequently bilinearly interpolated back to the ViT's native 224x224 resolution. This central crop triggered a 27.24\% drop (0.3253 $\rightarrow$ 0.2367) in the mean activation of Feature 1449. This suggests that the feature is sensitive to peripheral visual context; however, further artifact-labeled controls are needed to establish biopsy-artifact specificity conclusively.
\end{enumerate}"""

pattern = r'\\subsection\{Causal Validation Experiments.*?effectively indicating the "biopsy leak\."\n'
text = re.sub(pattern, lambda m: new_section + '\n', text, flags=re.DOTALL)

with open('paper/main2.tex', 'w') as f:
    f.write(text)

print("Saved main2.tex")
