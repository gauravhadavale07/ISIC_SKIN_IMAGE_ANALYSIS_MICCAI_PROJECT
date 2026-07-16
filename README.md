# ISIC Skin Image Analysis (MICCAI)

This repository contains the codebase for the ISIC Skin Image Analysis project, geared towards MICCAI submission. The project involves deep learning models, mechanistic audits, adversarial demographic debiasing, and statistical analysis pipelines for skin lesion classification and analysis.

## Features

- **Training and Evaluation Pipelines:** Comprehensive scripts to train, evaluate, and fine-tune models on ISIC datasets.
- **Mechanistic Audits:** Includes activation patching, contrastive steering, and feature knockout scripts (e.g., `task8_activation_patching.py`, `task12_contrastive_steering.py`).
- **Statistical Analysis:** Bootstrap analyzers, fairness assessments, and rigorous significance testing for robust evaluation.
- **Adversarial Robustness:** Tasks focused on demographic bias patches and adversarial demographic swaps.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gauravhadavale07/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT.git
   cd ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT
   ```

2. **Install dependencies:**
   Ensure you have PyTorch and other relevant dependencies installed according to your environment requirements.

3. **Data Preparation:**
   Note: Datasets are excluded from this repository. Ensure that the raw ISIC datasets and tabular metadata are placed correctly in the `data/` directory or as configured in `config.py`.

## Usage

Various tasks and experiments are separated into distinct scripts.
You can run an experiment using:
```bash
python run_experiment.py
```
Or run specific audit/evaluation tasks directly, for example:
```bash
python task14_feature_knockout.py
```

## Note

This repository contains code only. Papers, heavy datasets, model checkpoints, and AI coding agent configurations have been omitted to keep the repository lightweight and clean.
