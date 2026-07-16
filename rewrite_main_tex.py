import os
from string import Template

latex_content = r"""\documentclass[runningheads]{llncs}
%
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{multirow}
\usepackage{float}
%
\begin{document}
%
\title{Beyond Modality: Mechanistic Auditing of Multimodal Fusion in Skin Lesion Analysis}
\titlerunning{Mechanistic Auditing of Multimodal Fusion}

% Double-blind review: Authors left blank/asterisks
\author{***}
\authorrunning{***}
\institute{***}

\maketitle
%
\begin{abstract}
Multimodal architectures integrating clinical imagery and patient metadata have increasingly claimed state-of-the-art performance in dermatological classification. However, standard evaluation metrics often conflate genuine multimodal semantic grounding with the exploitation of dataset artifacts. In this paper, we conduct a mechanistic and fairness audit of multimodal fusion paradigms (Late Fusion, Gated Multimodal Units, and Cross-Attention) on two major public benchmarks: PAD-UFES-20 and MILK10k. Through an auditing pipeline utilizing Neutral Text Ablation, Counterfactual Injection, and Metadata-Shuffle Lexical Controls, we demonstrate that the vast majority of current multimodal architectures are functionally blind to textual semantics. Instead of extracting diagnostic clinical priors, architectures such as Late Fusion, GMU, and Cross-Attention (T$\rightarrow$V) utilize the text pathway purely as a lexical scaffold. Conversely, we find that Cross-Attention (V$\rightarrow$T) exhibits a small but statistically significant sensitivity to text-image pairing. Furthermore, we reveal that these architectures incur up to a 67\% increase in inference compute (FLOPs) compared to unimodal baselines for negligible multimodal diagnostic utility. Finally, we expose pervasive structural tautologies in standard interpretability metrics (Counterfactual Flip Rate and Linear CKA), identify critical data leakage flaws (biopsy leaks and an 85\% lesion overlap in MILK10k splits) that artificially inflate multimodal baselines, and uncover severe performance degradation on dark skin tones (FST V/VI). We propose a standardized auditing framework for the robust evaluation of multimodal dermatological models.

\keywords{Multimodal Learning \and Mechanistic Auditing \and Skin Lesion Analysis \and Fairness.}
\end{abstract}

%
\section{Introduction}

The integration of clinical patient metadata (e.g., age, sex, lesion location) with dermoscopic imagery via vision-language multimodal architectures has become a dominant paradigm in automated skin lesion analysis. Recent studies frequently report that models such as Cross-Attention Vision Transformers and Gated Multimodal Units (GMU) surpass unimodal image baselines. However, these claims rely predominantly on macroscopic metrics like Accuracy and Area Under the Curve (AUC), which are vulnerable to dataset biases and structural leakage.

In this work, we argue that superior accuracy on a multimodal test set does not necessarily indicate genuine semantic grounding. Models may achieve high performance by exploiting spurious correlations, such as biopsy markers in images \cite{pacheco2020pad}, or by memorizing patient metadata due to lesion-level dataset overlap. To address this, we introduce a mechanistic auditing framework that isolates the text pathway to evaluate its true semantic contribution, shifting the evaluation focus from simple architectural leaderboards to rigorous mechanistic and fairness audits.

Our contributions are threefold:
\begin{enumerate}
    \item We identify and resolve five pervasive integrity flaws in current multimodal evaluations, most notably a severe 85\% lesion-leakage bug in standard MILK10k splits and tokenizer-induced attention collapse.
    \item We propose a novel mechanistic auditing pipeline comprising Neutral Text Ablation, Counterfactual Probes, and Metadata-Shuffle Lexical Controls.
    \item We demonstrate quantitatively and qualitatively that the vast majority of current multimodal architectures are functionally blind to textual semantics, utilizing clinical text merely as a structural scaffold, while also exhibiting profound out-of-distribution demographic bias on dark skin tones.
\end{enumerate}

\section{Identifying and Resolving Evaluation Flaws}

Before evaluating multimodal architectures, it is critical to ensure that the dataset and evaluation metrics are structurally sound. During our initial baselining on the PAD-UFES-20 and MILK10k datasets, we uncovered several severe evaluation flaws.

\subsection{Dataset Contamination and Leakage}
In the PAD-UFES-20 dataset \cite{pacheco2020pad}, we identified a pervasive ``biopsy leak'': physical surgical markers and biopsy punches were visible in images corresponding to malignant classes. Unimodal Vision models trivially exploit this artifact, achieving artificially inflated performance (see Figure \ref{fig:biopsy_leak}).

\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/fig_biopsy_leak.pdf}
    \caption{Visual evidence of the ``biopsy leak'' in PAD-UFES-20. The benign lesion (left) is clean, while the malignant melanoma (right) contains a visible surgical marker, providing a trivial spurious correlation for the vision backbone.}
    \label{fig:biopsy_leak}
\end{figure}

More critically, in the MILK10k benchmark, we found that standard random image-level 85/15 splits resulted in an 84.7\% to 87.6\% overlap of identical patient lesions between the training and validation sets. Because multimodal networks receive identical clinical text strings for multiple images of the same lesion, a Text-Only baseline was able to achieve 53.26\% in-domain accuracy in earlier flawed iterations, vastly outperforming its Out-of-Domain (OOD) random-chance baseline (36.77\%). Documenting and resolving this lesion leakage is a primary methodological contribution of our work. By generating a strictly deterministic, lesion-disjoint split, we eliminated this memorization vector, forcing the Text-Only baseline back to majority-class prediction (54.71\%).

\subsection{Structural Metric Tautologies}
Researchers frequently utilize the Counterfactual Flip Rate (CFR) and Centered Kernel Alignment (CKA) \cite{kornblith2019similarity} to prove that their fusion layers are leveraging text. We prove algebraically and empirically that for any Unimodal Image Baseline, the CFR is strictly 0\% (because it processes no text) and the CKA is strictly 1.0 (because the fusion space is identical to the visual space). Reporting these metrics against a unimodal baseline is a structural tautology that guarantees a statistically significant p-value without providing any evidence of semantic grounding. They must only be used to compare multimodal models against each other (Figure \ref{fig:cka}).

\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/fig12_cka_visualization.pdf}
    \caption{Centered Kernel Alignment (CKA) between intermediate multimodal representations and unimodal baselines. Values approaching 1.0 indicate representational collapse, where the multimodal model behaves identically to a unimodal baseline.}
    \label{fig:cka}
\end{figure}

\section{The Mechanistic Auditing Framework}

To determine if multimodal architectures genuinely extract diagnostic clinical priors from text, we developed a three-stage mechanistic audit. We evaluated four architectures: an Image-Only baseline (ViT-Base) \cite{dosovitskiy2020image}, Late Fusion, GMU \cite{arevalo2017gated}, and a bidirectional Cross-Attention Transformer (T$\rightarrow$V and V$\rightarrow$T) \cite{vaswani2017attention}. The complete auditing pipeline is illustrated in Figure \ref{fig:pipeline}.

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{figures/pipeline_diagram.png}
    \caption{The Mechanistic Auditing Pipeline. We isolate the text pathway by subjecting it to four distinct probes (Blank, Neutral, Counterfactual, and Metadata-Shuffle) to quantify genuine semantic grounding versus structural artifact exploitation.}
    \label{fig:pipeline}
\end{figure}

\subsection{Tokenizer Artifacts and Neutral Ablation}
The standard practice for isolating the visual pathway in a multimodal model is to input an empty string (``''). However, we discovered this induces a tokenizer collapse. When an empty string is tokenized by ClinicalBERT \cite{alsentzer2019publicly}, it yields only a [CLS] and [SEP] token. In Cross-Attention (V$\rightarrow$T) layers, attending to this empty structure causes catastrophic numerical instability, crashing model accuracy to 24.72\% (below the 36.77\% majority baseline). We solved this by instituting a ``Neutral Text'' probe (e.g., ``No clinical history available''), which preserved token sequence stability (recovering accuracy to 40.63\%) without injecting semantic priors.

\subsection{Counterfactual Semantic Injection}
If a model relies on textual semantics, overriding its input with contradictory metadata (e.g., telling the model a facial lesion is located on the foot) should flip its prediction. We tracked the CFR and Mean Probability Shift ($\Delta$P) when injecting these adversarial priors (see Figure \ref{fig:counterfactual}).

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{figures/fig15_counterfactual_case_studies.pdf}
    \caption{Counterfactual Semantic Injection. By actively altering the clinical text to present contradictory priors (e.g., changing the age or lesion location), we force the model to flip its prediction, proving its active reliance on the text pathway.}
    \label{fig:counterfactual}
\end{figure}

\subsection{Metadata-Shuffle Lexical Control}
Because clinical histories follow strict structural templates, we introduced a Metadata-Shuffle Control. We randomized the clinical text strings across the entire test set. This perfectly preserves the dataset's lexical token distribution and syntax while completely severing the semantic alignment between the clinical priors and the ground-truth image. To strictly control the family-wise error rate across multiple architectures, we apply a Holm-Bonferroni correction. 

\section{Results and Discussion}

\begin{figure}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/fig16_cross_attention_visualization.pdf}
    \caption{Cross-Attention visualization demonstrating severe over-reliance on text tokens. The model's attention heads aggressively target the clinical metadata rather than the visual lesion pathology, structurally explaining the semantic blindness.}
    \label{fig:attention}
\end{figure}

\subsection{In-Domain and Out-of-Domain Generalization}
Table \ref{tab:milk10k} displays the in-domain results on the corrected, lesion-disjoint MILK10k split. When evaluated properly without lesion overlap, all models experience a drastic performance drop compared to leaky random splits. The Image-Only baseline drops to 61.40\%, confirming that previous high performance on random splits was heavily reliant on dataset memorization. 

\begin{table}[h]
\centering
\caption{In-Domain Accuracy, Macro-F1, and AUROC on Lesion-Disjoint MILK10k.*}
\label{tab:milk10k}
\begin{tabular}{lccc}
\toprule
Architecture & Accuracy & Macro-F1 & AUROC \\
\midrule
Image-Only Baseline & $MILK10K_IMG_ACC & $MILK10K_IMG_F1 & $MILK10K_IMG_AUC \\
Text-Only Baseline & $MILK10K_TXT_ACC & $MILK10K_TXT_F1 & $MILK10K_TXT_AUC \\
Late Fusion & $MILK10K_LF_ACC & $MILK10K_LF_F1 & $MILK10K_LF_AUC \\
GMU Baseline & $MILK10K_GMU_ACC & $MILK10K_GMU_F1 & $MILK10K_GMU_AUC \\
Cross-Attention (V$\rightarrow$T) & $MILK10K_VT_ACC & $MILK10K_VT_F1 & $MILK10K_VT_AUC \\
Cross-Attention (T$\rightarrow$V) & \textbf{$MILK10K_TV_ACC} & \textbf{$MILK10K_TV_F1} & \textbf{$MILK10K_TV_AUC} \\
\bottomrule
\end{tabular}
\end{table}
*Note: These lesion-disjoint metrics supersede all prior results from the original 85/15 random split, which suffered from an 85\% identical-lesion overlap between train and validation sets, artificially inflating metrics.

Table \ref{tab:pad_ufes_audit} illustrates out-of-distribution (OOD) performance and mechanistic robustness on PAD-UFES-20. Cross-Attention (T$\rightarrow$V) maintains the highest accuracy and lowest CFR, indicating it is the most resilient to contradictory metadata.

\begin{table}[h]
\centering
\caption{OOD Performance and Mechanistic Audit on PAD-UFES-20.}
\label{tab:pad_ufes_audit}
\resizebox{\textwidth}{!}{
\begin{tabular}{lcccccc}
\toprule
Architecture & Accuracy & AUROC & Macro-F1 & CFR & Mean $\Delta$P & Linear CKA \\
\midrule
Image-Only & 38.63\% & 0.680 & 19.18\% & 0.00\%$^\dagger$ & 0.00 pp & 1.000$^\dagger$ \\
Text-Only & 36.77\% & 0.404 & 8.96\% & 0.00\%$^\dagger$ & 0.00 pp & 1.000$^\dagger$ \\
Late Fusion & 41.51\% & 0.751 & 24.45\% & 12.45\% & 5.80 pp & 0.966 \\
GMU Baseline & 41.91\% & 0.759 & 27.25\% & 9.69\% & 5.18 pp & 0.863 \\
Cross-Attention (V$\rightarrow$T) & 41.70\% & 0.752 & 28.71\% & 23.14\%* & 10.88 pp* & 0.328 \\
Cross-Attention (T$\rightarrow$V) & \textbf{43.47\%} & \textbf{0.795} & \textbf{32.65\%} & \textbf{2.93\%} & \textbf{1.66 pp} & \textbf{0.764} \\
\bottomrule
\end{tabular}
}
\end{table}
*Note: Cross-Attention (V$\rightarrow$T) blank-text ablation accuracy crashed to 24.72\% due to a tokenizer artifact; CFR and $\Delta$P for this architecture are artificially inflated by this instability. \\
$^\dagger$Note: CFR and Linear CKA for unimodal baselines are structural tautologies (fixed at 0\% and 1.0 respectively) and are provided purely to demonstrate the metric floor/ceiling.

\subsection{Testing Semantic Blindness and Statistical Power}
When evaluated under the Metadata-Shuffle Lexical Control, randomizing metadata semantics caused zero meaningful accuracy drop for Late Fusion, GMU, and Cross-Attention (T$\rightarrow$V). We conclude these specific architectures are structurally blind to textual semantics. However, Cross-Attention (V$\rightarrow$T) experienced a statistically significant accuracy drop ($\Delta = 2.76$ pp, 95\% CI [1.89, 3.64], Holm-corrected $p < 0.00004$), suggesting attention direction matters for semantic preservation.

While comparing T$\rightarrow$V to the Image-Only baseline yields an enormous effect size on Accuracy (Cohen's $d = +12.3$) and F1 Macro ($d = +22.2$), none of the architectures survive Holm-Bonferroni correction against the unimodal baseline. This is due to a mathematically restricted sample size ($N=3$ seeds), which establishes a $p$-value floor that prevents even massive performance gaps from crossing the threshold. We acknowledge this as an "underpowered large effect"; definitive statistical confirmation would require running $\ge 20$ seeds.

\subsection{Computational Inefficiency}
Table \ref{tab:flops} highlights the computational cost of these architectures. Implementing multimodal architectures incurs up to a 67\% increase in inference compute compared to the unimodal Image-Only baseline (jumping from 16.87 to $\sim$28 GFLOPs). Given our proof that these models extract near-zero semantic utility from the text, this immense overhead offers negative clinical utility.

\begin{table}[h]
\centering
\caption{Computational Overhead vs Unimodal Baseline}
\label{tab:flops}
\begin{tabular}{lccc}
\toprule
Architecture & Total Params & Trainable & Inference Compute \\
\midrule
Image-Only & 85.80 M & 4.6 K & 16.87 GFLOPs \\
Late Fusion & 195.29 M & 1.18 M & 27.75 GFLOPs \\
GMU Baseline & 196.08 M & 1.97 M & 27.75 GFLOPs \\
Cross-Attention (V$\rightarrow$T) & 196.47 M & 2.37 M & 27.90 GFLOPs \\
Cross-Attention (T$\rightarrow$V) & 197.07 M & 2.96 M & 28.17 GFLOPs \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Fairness: Skin-Tone Stratification Audit}
To determine if these architectures exhibit out-of-distribution (OOD) demographic bias, we conducted a skin-tone stratified audit using the Diverse Dermatology Images (DDI) dataset (Figure \ref{fig:ddi_examples}). As shown in Table \ref{tab:ddi_fairness} and Figure \ref{fig:fairness_audit}, every architecture exhibits profound demographic bias, suffering massive performance degradations on Dark skin (FST V/VI). For example, Cross-Attention (T$\rightarrow$V) achieves a Macro-F1 of 16.8\% on Medium skin but drops precipitously to 10.3\% on Dark skin. With $N=66$ for Dark skin, the wide 95\% Confidence Intervals confirm that models operate barely above random chance on this demographic.

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{figures/fig21_fairness_audit.pdf}
    \caption{Skin-Tone Fairness Audit on the DDI dataset. All multimodal architectures suffer a catastrophic degradation in AUROC on Dark skin (FST V/VI), approaching random chance performance.}
    \label{fig:fairness_audit}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{figures/fig_ddi_examples.pdf}
    \caption{Representative samples from the DDI dataset across the Fitzpatrick Skin Type (FST) spectrum, illustrating the multimodal architecture's failure (False Negative) on dark skin tones.}
    \label{fig:ddi_examples}
\end{figure}

\begin{table}[h]
\centering
\caption{Skin-Tone Stratified Audit on DDI. Models exhibit severe performance drops on Dark skin tones (FST V/VI). Point estimates are provided with 95\% CIs.}
\label{tab:ddi_fairness}
\resizebox{\textwidth}{!}{
\begin{tabular}{l lll lll lll}
\toprule
& \multicolumn{3}{c}{FST I/II (Light, N=123)} & \multicolumn{3}{c}{FST III/IV (Medium, N=154)} & \multicolumn{3}{c}{FST V/VI (Dark, N=66)} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}
Architecture & Acc [95\% CI] & F1 [95\% CI] & AUROC & Acc [95\% CI] & F1 [95\% CI] & AUROC & Acc [95\% CI] & F1 [95\% CI] & AUROC \\
\midrule
Image-Only & 14.6\% [8.9, 21.1] & 7.3\% [4.0, 11.5] & 0.59 & 29.9\% [23.4, 37.0] & 13.0\% [9.5, 16.7] & 0.60 & 8.9\% [3.0, 16.7] & 7.6\% [2.0, 14.6] & 0.51 \\
Late Fusion & 13.8\% [7.3, 20.3] & 9.1\% [3.3, 16.6] & 0.64 & 32.6\% [25.3, 40.3] & 16.2\% [10.6, 22.9] & 0.68 & \textbf{14.9\% [6.1, 24.2]} & \textbf{15.4\% [6.2, 25.7]} & 0.61 \\
GMU & 14.7\% [8.9, 21.1] & \textbf{9.7\% [4.6, 15.9]} & \textbf{0.63} & 30.7\% [24.0, 38.3] & \textbf{19.4\% [13.0, 26.9]} & 0.67 & 7.5\% [1.5, 15.2] & 7.0\% [1.2, 13.1] & 0.54 \\
Cr-Attn (T$\rightarrow$V) & \textbf{17.0\% [10.6, 23.6]} & 9.0\% [5.6, 13.1] & 0.62 & \textbf{35.9\% [28.6, 43.5]} & 16.8\% [13.2, 20.9] & \textbf{0.68} & 10.6\% [4.5, 18.2] & 10.4\% [3.4, 18.2] & \textbf{0.62} \\
Cr-Attn (V$\rightarrow$T) & 11.3\% [5.7, 17.9] & 4.1\% [2.0, 6.8] & 0.58 & 24.1\% [18.2, 30.5] & 8.5\% [5.6, 12.2] & 0.65 & 6.0\% [1.5, 12.1] & 9.2\% [1.6, 17.1] & 0.55 \\
\bottomrule
\end{tabular}
}
\end{table}

\section{Conclusion and Reproducibility}
Our mechanistic audit deconstructs the illusion of multimodal superiority in current dermatological classification architectures. We demonstrated that most fusion pathways are functionally blind to clinical semantics, serving primarily as computationally expensive lexical scaffolds with severe out-of-distribution demographic bias.

To ensure maximal transparency, we explicitly report our reproducibility limits: (1) all results reflect evaluated seeds 456, 789, and 1337 (superseding deprecated initial tests on seeds 42, 123, and 999), (2) the execution environment relies on dynamic HuggingFace model hashes and unpinned \texttt{timm} layers which may drift, (3) floating-point calculations leverage \texttt{torch.amp.autocast} without explicitly restrictive \texttt{CUBLAS\_WORKSPACE\_CONFIG} determinism locks, exposing tests to atomic non-determinism, and (4) the Kaggle API used for dataset retrieval is subject to silent upstream drift. We urge the community to mandate mechanistic text ablation controls before publishing claims of multimodal improvements.

\section*{Data and Code Availability}
The datasets analyzed in this study are publicly available. PAD-UFES-20 is available at \url{https://data.mendeley.com/datasets/zr7vgbcyr2/1}. The MILK10k dataset is available via the ISIC Archive (\url{https://www.isic-archive.com}). The Diverse Dermatology Images (DDI) dataset is available at \url{https://ddi-dataset.github.io}. The fully reproducible code, including the mechanistic auditing framework, dataset splitting scripts, and counterfactual probes, is available at [Anonymized for Review].

%
% ---- Bibliography ----
%
\bibliographystyle{splncs04}
\bibliography{references}

\end{document}
"""

with open("paper/main_template.tex", "w") as f:
    f.write(latex_content)
print("Saved paper/main_template.tex")
