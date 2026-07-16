import re

with open('paper/main_template.tex', 'r') as f:
    c = f.read()

# Trim Intro
c = c.replace(
    "The integration of clinical patient metadata (e.g., age, sex, lesion location) with dermoscopic imagery via vision-language multimodal architectures has become a dominant paradigm in automated skin lesion analysis. Recent studies frequently report that models such as Cross-Attention Vision Transformers and Gated Multimodal Units (GMU) surpass unimodal image baselines.",
    "Integrating clinical metadata with dermoscopic imagery via vision-language multimodal architectures has become a dominant paradigm in skin lesion analysis. Recent studies frequently claim that models like Cross-Attention Vision Transformers and Gated Multimodal Units (GMU) surpass unimodal baselines."
)

c = c.replace(
    "In this work, we argue that superior accuracy on a multimodal test set does not necessarily indicate genuine semantic grounding. Models may achieve high performance by exploiting spurious correlations, such as biopsy markers in images \\cite{pacheco2020pad}, or by memorizing patient metadata due to lesion-level dataset overlap. To address this, we introduce a mechanistic auditing framework that isolates the text pathway to evaluate its true semantic contribution, shifting the evaluation focus from simple architectural leaderboards to rigorous mechanistic and fairness audits.",
    "We argue that superior multimodal accuracy does not inherently indicate genuine semantic grounding. Models may achieve high performance by exploiting spurious correlations, such as biopsy markers \\cite{pacheco2020pad}, or by memorizing metadata due to lesion-level dataset overlap. To address this, we introduce a mechanistic auditing framework that isolates the text pathway to quantify its true semantic contribution, shifting evaluation from architectural leaderboards to rigorous mechanistic and fairness audits."
)

# Trim Section 3.1
c = c.replace(
    "The standard practice for isolating the visual pathway in a multimodal model is to input an empty string (``''). However, we discovered this induces a tokenizer collapse. When an empty string is tokenized by ClinicalBERT \\cite{alsentzer2019publicly}, it yields only a [CLS] and [SEP] token. In Cross-Attention (V$\\rightarrow$T) layers, attending to this empty structure causes catastrophic numerical instability, crashing model accuracy to 24.72\\% (below the 36.77\\% majority baseline). We solved this by instituting a ``Neutral Text'' probe (e.g., ``No clinical history available''), which preserved token sequence stability (recovering accuracy to 40.63\\%) without injecting semantic priors.",
    "The standard practice for isolating the visual pathway is to input an empty string. However, we discovered this induces a tokenizer collapse. Tokenizing an empty string with ClinicalBERT \\cite{alsentzer2019publicly} yields only [CLS] and [SEP] tokens. In Cross-Attention (V$\\rightarrow$T) layers, attending to this empty structure causes catastrophic numerical instability, crashing accuracy to 24.72\\% (below the 36.77\\% majority baseline). We solved this using a ``Neutral Text'' probe (e.g., ``No clinical history available''), preserving sequence stability (recovering accuracy to 40.63\\%) without injecting semantic priors."
)

# Move Fig 4 into Section 3.2
# Wait, Fig 4 is currently between Section 3.2 and Section 3.3. It is fine.
# Move Fig 5 into Section 4. It's currently right under \section{Results and Discussion}. It is fine.

with open('paper/main_template.tex', 'w') as f:
    f.write(c)
with open('paper/main.tex', 'w') as f:
    f.write(c)

