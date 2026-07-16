import re

with open('paper/main2.txt', 'r') as f:
    text = f.read()

new_section = """6.8 Causal Validation Experiments: Scaffold Ablation, Answer-Order Priors, and Visual Biopsy Leakage

To rigorously test our hypotheses regarding structural padding geometry dependencies, multi-modal priors, and visual/textual grounding, we executed four targeted validation protocols:

1. Cross-Attention Scaffold Ablation (Task 18): Our initial analysis theorized that the Cross-Attention T→V fusion model bypasses true semantic alignment by anchoring to static sequence padding tokens. By zeroing out the attention mask for [CLS] and [SEP] tokens, we severed this structural crutch. The ablation resulted in a negligible drop from the established 43.47% Baseline Accuracy to an Ablated Accuracy of 42.94% (an absolute degradation of only 0.53%). This suggests that the T→V architecture is robustly grounded in visual evidence rather than cross-modal scaffold dependency.

2. Discriminative Matched Grounding Control (Task 21): To assess whether the Cross-Attention T→V model genuinely integrates text, we conducted a 2x2 matched visual/text grounding audit on 104 cases (52 MEL, 52 NEV). We tested real vs. blank images against both aligned and contradictory clinical histories. The model proved almost entirely text-blind: introducing contradictory text (e.g., presenting a clear MEL history for a NEV image) yielded less than a 1% flip rate and negligible logit margin shifts (e.g., a mean shift of -0.01). The model's predictions remained firmly anchored to the visual pathway.

3. LVLM Answer-Order Prior Audit (Task 22): We subjected LLaVA-Med-v1.5 to the same 2x2 matched visual/text audit, adding answer-order randomization ("Is the diagnosis MEL or NEV?" vs. "NEV or MEL?"). The results revealed complete prompt and answer-order sensitivity rather than diagnostic competence. On both real and blank images, the model exhibited a 100% adherence to the first option presented in the prompt. It completely ignored both the true visual pathology and the explicitly provided clinical text, demonstrating that its output is driven almost entirely by instruction-tuning position priors.

4. Visual Biopsy Leak via Center Cropping (Task 20): We extracted the Top-50 activating images for Dictionary Feature 1449, which correlated heavily with surgical rulers and skin markings. To investigate if this feature tracks peripheral artifacts rather than the melanoma pathology itself, we applied a 200x200 center crop, which was subsequently bilinearly interpolated back to the ViT's native 224x224 resolution. This central crop triggered a 27.24% drop (0.3253 → 0.2367) in the mean activation of Feature 1449. This suggests that the feature is sensitive to peripheral visual context; however, further artifact-labeled controls are needed to establish biopsy-artifact specificity conclusively."""

# We will use regex to find the section and replace it.
# The section starts with "6.8 Causal Validation Experiments" and ends right before "8\n\n7. Discussion"
pattern = r'6\.8 Causal Validation Experiments.*?effectively confirming the "biopsy leak\."\n'
text = re.sub(pattern, new_section + '\n', text, flags=re.DOTALL)

with open('paper/main2.txt', 'w') as f:
    f.write(text)

print("Replaced section 6.8")
