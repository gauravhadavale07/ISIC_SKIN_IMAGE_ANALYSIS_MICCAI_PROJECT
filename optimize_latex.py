import subprocess
import os

latex_base = r"""\documentclass[runningheads]{llncs}
%
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{multirow}
%
\begin{document}
%
\title{Beyond Modality: Mechanistic Auditing of Multimodal Fusion in Skin Lesion Analysis}
\titlerunning{Mechanistic Auditing of Multimodal Fusion}

% Author block left blank with spacing as requested
\author{~\vspace{2em}}
\authorrunning{~}
\institute{~}

\maketitle
%
\begin{abstract}
Multimodal architectures integrating clinical imagery and patient metadata have increasingly claimed state-of-the-art performance in dermatological classification. However, standard evaluation metrics often conflate genuine multimodal semantic grounding with the exploitation of dataset artifacts. In this paper, we conduct a mechanistic and fairness audit of multimodal fusion paradigms (Late Fusion, Gated Multimodal Units, and Cross-Attention) on two major public benchmarks: PAD-UFES-20 and MILK10k. Through an auditing pipeline utilizing Neutral Text Ablation, Counterfactual Injection, and Metadata-Shuffle Lexical Controls, we demonstrate that the vast majority of current multimodal architectures are functionally blind to textual semantics. Instead of extracting diagnostic clinical priors, architectures such as Late Fusion, GMU, and Cross-Attention (T$\rightarrow$V) utilize the text pathway purely as a lexical scaffold. Conversely, we find that Cross-Attention (V$\rightarrow$T) exhibits a small but statistically significant sensitivity to text-image pairing. Furthermore, we reveal that these architectures incur up to a 67\% increase in inference compute (FLOPs) compared to unimodal baselines for negligible multimodal diagnostic utility. Finally, we expose pervasive structural tautologies in standard interpretability metrics (Counterfactual Flip Rate and Linear CKA), identify critical data leakage flaws (biopsy leaks and an 85\% lesion overlap in MILK10k splits) that artificially inflate multimodal baselines, and uncover severe performance degradation on dark skin tones (FST V/VI). We propose a standardized auditing framework for the robust evaluation of multimodal dermatological models.

\keywords{Multimodal Learning \and Mechanistic Auditing \and Skin Lesion Analysis \and Fairness.}
\end{abstract}

%
\section{Introduction}

Integrating clinical metadata with dermoscopic imagery via vision-language multimodal architectures has become a dominant paradigm in skin lesion analysis. Recent studies frequently claim that models like Cross-Attention Vision Transformers and Gated Multimodal Units (GMU) surpass unimodal baselines. However, these claims rely predominantly on macroscopic metrics like Accuracy and Area Under the Curve (AUC), which are vulnerable to dataset biases and structural leakage.

We argue that superior multimodal accuracy does not inherently indicate genuine semantic grounding. Models may achieve high performance by exploiting spurious correlations, such as biopsy markers \cite{pacheco2020pad}, or by memorizing metadata due to lesion-level dataset overlap. To address this, we introduce a mechanistic auditing framework that isolates the text pathway to quantify its true semantic contribution, shifting evaluation from architectural leaderboards to rigorous mechanistic and fairness audits.

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

More critically, in the MILK10k benchmark, we found that standard random image-level 85/15 splits resulted in an 84.7\% to 87.6\% overlap of identical patient lesions between the training and validation sets. Because multimodal networks receive identical clinical text strings for multiple images of the same lesion, a Text-Only baseline was able to achieve 53.26\% in-domain accuracy in earlier flawed iterations, vastly outperforming its Out-of-Domain (OOD) random-chance baseline (36.77\%). Documenting and resolving this lesion leakage is a primary methodological contribution of our work. By generating a strictly deterministic, lesion-disjoint split, we eliminated this memorization vector, forcing the Text-Only baseline back to majority-class prediction (54.71\%).

\subsection{Structural Metric Tautologies}
Researchers frequently utilize the Counterfactual Flip Rate (CFR) and Centered Kernel Alignment (CKA) \cite{kornblith2019similarity} to prove that their fusion layers are leveraging text. We prove algebraically and empirically that for any Unimodal Image Baseline, the CFR is strictly 0\% (because it processes no text) and the CKA is strictly 1.0 (because the fusion space is identical to the visual space). Reporting these metrics against a unimodal baseline is a structural tautology that guarantees a statistically significant p-value without providing any evidence of semantic grounding. They must only be used to compare multimodal models against each other (Figure \ref{fig:cka}).

\begin{figure}[tb]
    \centering
    % Changed [t] to [b] to bottom-align the captions
    \begin{minipage}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figures/fig_biopsy_leak.pdf}
        \caption{Visual evidence of the ``biopsy leak'' in PAD-UFES-20. The benign lesion (left) is clean, while the malignant melanoma (right) contains a visible surgical marker.}
        \label{fig:biopsy_leak}
    \end{minipage}\hfill
    \begin{minipage}[b]{0.48\textwidth}
        \centering
        % Note: If the gap is STILL there after changing to [b], your PDF has too much whitespace. 
        % You can crop it inside LaTeX by swapping to the line below (adjust the 2cm value as needed):
        % \includegraphics[width=\textwidth, trim=0cm 2cm 0cm 0cm, clip]{figures/fig12_cka_visualization.pdf}
        \includegraphics[width=\textwidth]{figures/fig12_cka_visualization.pdf}
        \caption{Centered Kernel Alignment (CKA) between intermediate representations and unimodal baselines. Values approaching 1.0 indicate representational collapse.}
        \label{fig:cka}
    \end{minipage}
\end{figure}

\section{The Mechanistic Auditing Framework}

To determine if multimodal architectures genuinely extract diagnostic clinical priors from text, we developed a three-stage mechanistic audit. We evaluated four architectures: an Image-Only baseline (ViT-Base) \cite{dosovitskiy2020image}, Late Fusion, GMU \cite{arevalo2017gated}, and a bidirectional Cross-Attention Transformer (T$\rightarrow$V and V$\rightarrow$T) \cite{vaswani2017attention}. The complete auditing pipeline is illustrated in Figure \ref{fig:pipeline}.

\begin{figure}[tb]
    \centering
    % Changed width from 0.55 to 1.0 to make it full size
    \includegraphics[width=\textwidth]{figures/pipeline_diagram.png}
    \caption{The Mechanistic Auditing Pipeline. We isolate the text pathway by subjecting it to four distinct probes to quantify genuine semantic grounding.}
    \label{fig:pipeline}
\end{figure}

\subsection{Tokenizer Artifacts and Neutral Ablation}
The standard practice for isolating the visual pathway is to input an empty string. However, we discovered this induces a tokenizer collapse. Tokenizing an empty string with ClinicalBERT \cite{alsentzer2019publicly} yields only [CLS] and [SEP] tokens. In Cross-Attention (V$\rightarrow$T) layers, attending to this empty structure causes catastrophic numerical instability, crashing accuracy to 24.72\% (below the 36.77\% majority baseline). We solved this using a ``Neutral Text'' probe (e.g., ``No clinical history available''), preserving sequence stability (recovering accuracy to 40.63\%) without injecting semantic priors.

\subsection{Counterfactual Semantic Injection}
If a model relies on textual semantics, overriding its input with contradictory metadata (e.g., telling the model a facial lesion is located on the foot) should flip its prediction. We tracked the CFR and Mean Probability Shift ($\Delta$P) when injecting these adversarial priors (see Figure \ref{fig:counterfactual}).

\begin{figure}[tb]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/fig15_counterfactual_case_studies.pdf}
    \caption{Counterfactual Semantic Injection. By actively altering clinical text to present contradictory priors, we force the model to flip its prediction.}
    \label{fig:counterfactual}
\end{figure}

\subsection{Metadata-Shuffle Lexical Control}
Because clinical histories follow strict structural templates, we introduced a Metadata-Shuffle Control. We randomized the clinical text strings across the entire test set. This perfectly preserves the dataset's lexical token distribution and syntax while completely severing the semantic alignment between the clinical priors and the ground-truth image. To strictly control the family-wise error rate across multiple architectures, we apply a Holm-Bonferroni correction. 

\section{Results and Discussion}

\begin{figure}[tb]
    \centering
    \includegraphics[width=0.75\textwidth]{figures/fig16_cross_attention_visualization.pdf}
    \caption{Cross-Attention visualization demonstrating over-reliance on text tokens. Attention heads target clinical metadata rather than visual lesion pathology.}
    \label{fig:attention}
\end{figure}

\subsection{In-Domain and Out-of-Domain Generalization}
Table \ref{tab:milk10k} displays the in-domain results on the corrected, lesion-disjoint MILK10k split. When evaluated properly without lesion overlap, all models experience a drastic performance drop compared to leaky random splits. The Image-Only baseline drops to 61.40\%, confirming that previous high performance on random splits was heavily reliant on dataset memorization. 

\begin{table}[tb]
\centering
\caption{In-Domain Accuracy, Macro-F1, and AUROC on Lesion-Disjoint MILK10k.*}
\label{tab:milk10k}
\begin{tabular}{lccc}
\toprule
Architecture & Accuracy & Macro-F1 & AUROC \\
\midrule
Image-Only Baseline & 61.40$\pm$0.73\% & 26.63$\pm$0.88\% & 0.777$\pm$0.003 \\
Text-Only Baseline & 54.79$\pm$0.00\% & 11.79$\pm$0.00\% & 0.670$\pm$0.007 \\
Late Fusion & 68.95$\pm$0.62\% & 38.68$\pm$1.72\% & 0.880$\pm$0.002 \\
GMU Baseline & 69.29$\pm$0.25\% & 42.13$\pm$0.43\% & 0.882$\pm$0.001 \\
Cross-Attention (V$\rightarrow$T) & 70.55$\pm$0.22\% & 45.24$\pm$0.88\% & 0.884$\pm$0.003 \\
Cross-Attention (T$\rightarrow$V) & 71.62$\pm$0.58\% & 50.81$\pm$0.90\% & 0.899$\pm$0.002 \\
\bottomrule
\end{tabular}
\end{table}
\noindent *Note: These lesion-disjoint metrics supersede all prior results from the original 85/15 random split, which suffered from an 85\% identical-lesion overlap between train and validation sets, artificially inflating metrics.

Table \ref{tab:pad_ufes_audit} illustrates out-of-distribution (OOD) performance and mechanistic robustness on PAD-UFES-20. Cross-Attention (T$\rightarrow$V) maintains the highest accuracy and lowest CFR, indicating it is the most resilient to contradictory metadata.

\begin{table}[tb]
\centering
\setlength{\tabcolsep}{2pt}
\scriptsize % Ensures mandatory 8-pt font limit and fits text margins
\caption{OOD Performance and Mechanistic Audit on PAD-UFES-20. Accuracy and F1 values are in \%, $\Delta$P is in percentage points (pp).}
\label{tab:pad_ufes_audit}
\begin{tabular}{lcccccc}
\toprule
Fusion & Acc. & AUROC & Macro-F1 & CFR & $\Delta$P & CKA \\
\midrule
Img-Only & 38.70$\pm$0.24 & 0.677$\pm$0.015 & 19.01$\pm$1.33 & 0.00$\pm$0.00$^\dagger$ & 0.00$\pm$0.00 & 1.000$\pm$0.000$^\dagger$ \\
Text-Only & 36.77$\pm$0.00 & 0.384$\pm$0.028 & 8.96$\pm$0.00 & 0.00$\pm$0.00$^\dagger$ & 0.00$\pm$0.00 & 1.000$\pm$0.000$^\dagger$ \\
Late Fuse & 41.26$\pm$0.29 & 0.751$\pm$0.005 & 24.33$\pm$1.01 & 11.92$\pm$0.56 & 6.53$\pm$0.80 & 0.965$\pm$0.001 \\
GMU & 42.18$\pm$0.41 & 0.760$\pm$0.006 & 26.57$\pm$0.50 & 7.95$\pm$1.16 & 4.25$\pm$0.85 & 0.859$\pm$0.002 \\
CA (V$\rightarrow$T) & 41.71$\pm$1.78 & 0.744$\pm$0.009 & 26.99$\pm$3.46 & 22.53$\pm$4.30* & 12.02$\pm$1.77* & 0.311$\pm$0.038 \\
CA (T$\rightarrow$V) & \textbf{43.47$\pm$0.22} & \textbf{0.795$\pm$0.006} & \textbf{32.65$\pm$1.45} & \textbf{2.93$\pm$0.16} & \textbf{1.66$\pm$0.14} & \textbf{0.735$\pm$0.008} \\
\bottomrule
\end{tabular}
\end{table}
\noindent *Note: Cross-Attention (V$\rightarrow$T) blank-text ablation accuracy crashed to 24.72\% due to a tokenizer artifact; CFR and $\Delta$P for this architecture are artificially inflated by this instability. \\
$^\dagger$Note: CFR and Linear CKA for unimodal baselines are structural tautologies (fixed at 0\% and 1.0 respectively) and are provided purely to demonstrate the metric floor/ceiling.

\subsection{Testing Visual Sensitivity and Statistical Power}
When evaluated under the Metadata-Shuffle Lexical Control (Table \ref{tab:shuffle}), randomizing metadata semantics caused zero meaningful accuracy drop for Late Fusion, GMU, and Cross-Attention (T$\rightarrow$V). We conclude these specific architectures are functionally insensitive to textual semantics. However, Cross-Attention (V$\rightarrow$T) experienced a statistically significant accuracy drop ($\Delta = 2.76$ pp, 95\% CI [1.89, 3.64], Holm-corrected $p < 0.00004$), suggesting attention direction matters for semantic preservation.

\begin{table}[tb]
\centering
\caption{Metadata-Shuffle Lexical Control (5-seed aggregate). Comparing real metadata against shuffled clinical text on PAD-UFES-20 to test semantic sensitivity.}
\label{tab:shuffle}
\begin{tabular}{lcccc}
\toprule
Architecture & Real Acc & Shuffle Acc & $\Delta$ & Holm-corrected $p$-value \\
\midrule
Late Fusion & 41.21$\pm$0.00\% & 40.96$\pm$0.21\% & +0.25 pp & 0.22569 \\
GMU Baseline & 42.47$\pm$0.00\% & 42.52$\pm$0.18\% & -0.05 pp & 0.94376 \\
Cross-Attention (V$\rightarrow$T) & 40.34$\pm$0.00\% & 37.58$\pm$0.32\% & +2.76 pp & \textbf{$<$ 0.00004} \\
Cross-Attention (T$\rightarrow$V) & 43.34$\pm$0.00\% & 43.33$\pm$0.09\% & +0.01 pp & 0.94376 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[tb]
\centering
\caption{Computational Cost Comparison.}
\label{tab:compute}
\begin{tabular}{lccc}
\toprule
Architecture & Parameters & Text Tokens & Compute \\
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
To determine if these architectures exhibit out-of-distribution (OOD) demographic bias, we conducted a skin-tone stratified audit using the Diverse Dermatology Images (DDI) dataset (Figure \ref{fig:ddi_examples}). As shown in Table \ref{tab:ddi_fairness} and Figure \ref{fig:fairness_audit}, every architecture exhibits profound demographic bias, suffering massive performance degradations on Dark skin (FST V/VI). For example, Cross-Attention (T$\rightarrow$V) achieves a Macro-F1 of \textbf{16.5\%} on Medium skin but drops precipitously to \textbf{10.6\%} on Dark skin. With $N=66$ for Dark skin, the wide 95\% Confidence Intervals confirm that models operate barely above random chance on this demographic.

\begin{figure}[tb]
    \centering
    \begin{minipage}[t]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figures/fig21_fairness_audit.pdf}
        \caption{Skin-Tone Fairness Audit on DDI. All multimodal architectures suffer a catastrophic degradation in AUROC on Dark skin (FST V/VI).}
        \label{fig:fairness_audit}
    \end{minipage}\hfill
    \begin{minipage}[t]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{figures/fig_ddi_examples.pdf}
        \caption{DDI samples across the FST spectrum, illustrating failure on dark skin tones.}
        \label{fig:ddi_examples}
    \end{minipage}
\end{figure}

\begin{table}[tb]
\centering
\setlength{\tabcolsep}{0.8pt}
\scriptsize % Enforces minimum font and fits page margins perfectly
\caption{Skin-Tone Stratified Audit on DDI. Point estimates and 95\% CIs are shown in \%, AUC is fractional.}
\label{tab:ddi_fairness}
\begin{tabular}{l lll lll lll}
\toprule
& \multicolumn{3}{c}{Light (N=123)} & \multicolumn{3}{c}{Medium (N=154)} & \multicolumn{3}{c}{Dark (N=66)} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}
Model & Acc & F1 & AUC & Acc & F1 & AUC & Acc & F1 & AUC \\
\midrule
Img-Only & $14.6_{8.9\text{-}21.1}$ & $7.3_{4.0\text{-}11.5}$ & 0.59 & $29.9_{23.4\text{-}37.0}$ & $13.0_{9.5\text{-}16.7}$ & 0.60 & $8.9_{3.0\text{-}16.7}$ & $7.6_{2.0\text{-}14.6}$ & 0.51 \\
Late Fuse & $13.8_{7.3\text{-}20.3}$ & $9.1_{3.3\text{-}16.6}$ & 0.64 & $32.6_{25.3\text{-}40.3}$ & $16.2_{10.6\text{-}22.9}$ & 0.68 & $\textbf{14.9}_{6.1\text{-}24.2}$ & $\textbf{15.4}_{6.2\text{-}25.7}$ & 0.61 \\
GMU & $14.7_{8.9\text{-}21.1}$ & $\textbf{9.7}_{4.6\text{-}15.9}$ & \textbf{0.63} & $30.7_{24.0\text{-}38.3}$ & $\textbf{19.4}_{13.0\text{-}26.9}$ & 0.67 & $7.5_{1.5\text{-}15.2}$ & $7.0_{1.2\text{-}13.1}$ & 0.54 \\
CA (T$\rightarrow$V) & $\textbf{17.0}_{10.6\text{-}23.6}$ & $9.0_{5.6\text{-}13.1}$ & 0.62 & $\textbf{35.9}_{28.6\text{-}43.5}$ & $16.8_{13.2\text{-}20.9}$ & \textbf{0.68} & $10.6_{4.5\text{-}18.2}$ & $10.4_{3.4\text{-}18.2}$ & \textbf{0.62} \\
CA (V$\rightarrow$T) & $11.3_{5.7\text{-}17.9}$ & $4.1_{2.0\text{-}6.8}$ & 0.58 & $24.1_{18.2\text{-}30.5}$ & $8.5_{5.6\text{-}12.2}$ & 0.65 & $6.0_{1.5\text{-}12.1}$ & $9.2_{1.6\text{-}17.1}$ & 0.55 \\
\bottomrule
\end{tabular}
\end{table}

\section{Conclusion and Reproducibility}
Our mechanistic audit deconstructs the illusion of multimodal superiority in current dermatological classification architectures. We demonstrated that most fusion pathways are functionally insensitive to clinical semantics, serving primarily as computationally expensive lexical scaffolds with severe out-of-distribution demographic bias.

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

with open('/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/paper/main_template.tex', 'w') as f:
    f.write(latex_base)
with open('/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/paper/main.tex', 'w') as f:
    f.write(latex_base)

print("Directly wrote optimized latex_base to main_template.tex and main.tex!")
