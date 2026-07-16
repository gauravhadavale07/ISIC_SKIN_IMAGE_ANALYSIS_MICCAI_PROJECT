import re

with open('paper/main_template.tex', 'r') as f:
    content = f.read()

# 1. Change all [h] and [t] to [htbp]
content = re.sub(r'\\begin\{figure\}\[[ht]\]', r'\\begin{figure}[htbp]', content)
content = re.sub(r'\\begin\{table\}\[[ht]\]', r'\\begin{table}[htbp]', content)

# 2. Resize Fig 1 and Fig 2 to 0.5\textwidth
content = content.replace(r'\includegraphics[width=0.8\textwidth]{figures/fig_biopsy_leak.pdf}', 
                          r'\includegraphics[width=0.5\textwidth]{figures/fig_biopsy_leak.pdf}')
content = content.replace(r'\includegraphics[width=0.8\textwidth]{figures/fig12_cka_visualization.pdf}', 
                          r'\includegraphics[width=0.5\textwidth]{figures/fig12_cka_visualization.pdf}')

# 3. Update Figure 4 caption to specify Cross-Attention (V->T)
old_caption = r"\caption{Counterfactual Semantic Injection. By actively altering the clinical text to present contradictory priors (e.g., changing the age or lesion location), we force the model to flip its prediction, proving its active reliance on the text pathway.}"
new_caption = r"\caption{Counterfactual Semantic Injection (Cross-Attention V$\rightarrow$T). By actively altering the clinical text to present contradictory priors (e.g., changing the age or lesion location), we force the model to flip its prediction, proving its active reliance on the text pathway.}"
content = content.replace(old_caption, new_caption)

# 4. Resize Fig 5 (Cross-Attention visualization) to 0.8\textwidth
content = content.replace(r'\includegraphics[width=\textwidth]{figures/fig16_cross_attention_visualization.pdf}', 
                          r'\includegraphics[width=0.8\textwidth]{figures/fig16_cross_attention_visualization.pdf}')

# 5. Combine Fig 6 and Fig 7 into side-by-side minipages
# Wait, let's use regex to find them properly because they might have [htbp] now.
fig67_pattern = r"\\begin\{figure\}\[htbp\]\s*\\centering\s*\\includegraphics\[width=\\textwidth\]\{figures/fig21_fairness_audit\.pdf\}\s*\\caption\{Skin-Tone Fairness Audit on the DDI dataset\. All multimodal architectures suffer a catastrophic degradation in AUROC on Dark skin \(FST V/VI\), approaching random chance performance\.\}\s*\\label\{fig:fairness_audit\}\s*\\end\{figure\}\s*\\begin\{figure\}\[htbp\]\s*\\centering\s*\\includegraphics\[width=\\textwidth\]\{figures/fig_ddi_examples\.pdf\}\s*\\caption\{Representative samples from the DDI dataset across the Fitzpatrick Skin Type \(FST\) spectrum, illustrating the multimodal architecture's failure \(False Negative\) on dark skin tones\.\}\s*\\label\{fig:ddi_examples\}\s*\\end\{figure\}"

fig67_new = r"""\begin{figure}[htbp]
    \centering
    \begin{minipage}[t]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figures/fig21_fairness_audit.pdf}
        \caption{Skin-Tone Fairness Audit on the DDI dataset. All multimodal architectures suffer a catastrophic degradation in AUROC on Dark skin (FST V/VI).}
        \label{fig:fairness_audit}
    \end{minipage}\hfill
    \begin{minipage}[t]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figures/fig_ddi_examples.pdf}
        \caption{Representative samples from the DDI dataset across the Fitzpatrick Skin Type (FST) spectrum, illustrating failure on dark skin tones.}
        \label{fig:ddi_examples}
    \end{minipage}
\end{figure}"""

content = re.sub(fig67_pattern, lambda _: fig67_new, content)

# 6. Trim Section 5 reproducibility caveats
sec5_old = r"""To ensure maximal transparency, we explicitly report our reproducibility limits: (1) all results reflect evaluated seeds 456, 789, and 1337 (superseding deprecated initial tests on seeds 42, 123, and 999), (2) the execution environment relies on dynamic HuggingFace model hashes and unpinned \texttt{timm} layers which may drift, (3) floating-point calculations leverage \texttt{torch.amp.autocast} without explicitly restrictive \texttt{CUBLAS\_WORKSPACE\_CONFIG} determinism locks, exposing tests to atomic non-determinism, and (4) the Kaggle API used for dataset retrieval is subject to silent upstream drift. We urge the community to mandate mechanistic text ablation controls before publishing claims of multimodal improvements."""

sec5_new = r"""To ensure transparency, we note our reproducibility limits: results reflect seeds 456, 789, and 1337; the environment relies on dynamic HuggingFace hashes and unpinned \texttt{timm} layers which may drift; floating-point \texttt{torch.amp.autocast} lacks strict determinism locks; and the Kaggle API is subject to upstream drift. We urge the community to mandate mechanistic text ablation controls before claiming multimodal improvements."""

content = content.replace(sec5_old, sec5_new)

with open('paper/main_template.tex', 'w') as f:
    f.write(content)
with open('paper/main.tex', 'w') as f:
    f.write(content)

print("Rewrite successful.")
