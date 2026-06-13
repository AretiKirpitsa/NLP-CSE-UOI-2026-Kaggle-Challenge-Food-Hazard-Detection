# SemEval Food Hazard Detection Challenge

**Authors:** Nestor Moulias, Areti Kirpitsa

This repository contains our submission to the **SemEval Food Hazard Detection Challenge**, a text classification task that jointly predicts the **hazard category** and **product category** of food-incident reports collected from the web. Our final system reached **Top 8** on the leaderboard with a score of **0.85205**.

The repository documents the full research progression — from classical baselines to the final winning ensemble — so that each architectural decision can be reproduced and evaluated independently.

---

## Final Result

| System | Score |
|---|---|
| Final soft-voting ensemble (RoBERTa + AgriBERT + TF-IDF/LogReg) | **0.85205** |
| AgriBERT alone (OOF) | 0.7997 |
| RoBERTa-base alone (OOF) | 0.7246 |
| TF-IDF + Logistic Regression alone (OOF) | 0.6352 |

The decisive factor was the use of **AgriBERT** (`recobo/agriculture-bert-uncased`), a transformer pretrained on an agriculture-focused corpus whose vocabulary aligns closely with the lexical space of food-recall reports (crop names, agrochemicals, allergens, regulatory terminology).

---

## Repository Structure

```
.
├── notebooks/
│   ├── 01_tfidf_baseline.ipynb           # Classical TF-IDF + Logistic Regression
│   ├── 02_classifier_chains.ipynb        # Classifier Chains for multi-label
│   ├── 03_transformer_experiments.ipynb  # Single-backbone transformer trials
│   └── 04_final_ensemble.ipynb           # Final winning architecture
├── data/
│   ├── train.csv
│   ├── valid.csv
│   ├── test.csv
│   └── synthetic_samples.csv             # Synthetic augmentation set
├── outputs/                              # Predictions + cached OOF arrays
├── report/                               # Paper / report PDF
└── README.md
```

> **Note:** raw data files are not included in this repository due to challenge licensing. Place them under `data/` before running the notebooks.

---

## Architectures Explored

This project followed a progression of approaches. Each one is documented in its own notebook for transparency and reproducibility.

### 1. TF-IDF Baselines
Classical TF-IDF feature extraction (uni- and bi-grams, sublinear TF) combined with Logistic Regression. This served as the first baseline and informed the lexical member of the final ensemble.

### 2. Classifier Chains for Multi-label Classification
We treated the joint hazard/product prediction as a multi-label problem using Classifier Chains, which model dependencies between labels by feeding earlier predictions into later classifiers. While useful for exploring label dependencies, this approach was eventually superseded by independently trained classifiers combined through a soft-voting ensemble.

### 3. Transformer Fine-tuning
We fine-tuned several transformer backbones (RoBERTa-base, BERT variants) on the task. These experiments motivated the search for a domain-aligned model and led us to AgriBERT.

### 4. Final Architecture — Soft-Voting Ensemble
The winning system combines three decorrelated members:

- **RoBERTa-base** — general-purpose transformer baseline
- **AgriBERT** (`recobo/agriculture-bert-uncased`) — domain-pretrained encoder
- **TF-IDF + Logistic Regression** — captures lexical n-gram signals

Each member produces probability distributions over both hazard and product classes. Predictions are combined via weighted soft voting, with weights tuned on honest out-of-fold (OOF) predictions through a grid search over a discretized simplex.

Additional components:
- **5-fold stratified cross-validation** for honest OOF evaluation
- **Synthetic data augmentation** injected into training folds only (never validation), preserving the integrity of the OOF score
- **Deterministic post-processing rule** (`migration → food contact materials`) for a known systematic label dependency

---

## How to Reproduce

### Requirements

```bash
pip install -r requirements.txt
```

Main dependencies:
- `transformers`
- `torch`
- `scikit-learn`
- `pandas`, `numpy`
- `sentencepiece`

### Data Setup

Place the challenge files under `data/`:
- `train.csv`, `valid.csv`, `test.csv`
- `synthetic_samples.csv` (for the augmentation step)

### Running

Each notebook is self-contained. To reproduce the final result, run:

```
notebooks/04_final_ensemble.ipynb
```

The notebook caches OOF and test probability arrays per backbone (`oofH_*.npy`, `oofP_*.npy`, etc.), so you can run one backbone per session and resume.

**Hardware:** GPU recommended (Kaggle T4×2 or equivalent). One full backbone takes ~3–4 hours with `SEEDS=[42]`.

---

## Key Configuration

| Hyperparameter | Value |
|---|---|
| Max sequence length | 256 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Warmup ratio | 0.06 |
| Optimizer | AdamW (weight decay 0.01) |
| Folds | 5 (stratified on product class) |
| Seeds | [42] (extendable to [42, 123, 777]) |
| Input format | `[country] title text[:1000]` |

---

## Citation

If you use this work, please cite our paper:

```bibtex
@inproceedings{moulias_kirpitsa_2026,
  title     = {Your paper title here},
  author    = {Moulias, Nestor and Kirpitsa, Areti},
  booktitle = {Proceedings of SemEval},
  year      = {2026}
}
```

---

## Acknowledgments

- The **AgriBERT** model by `recobo` ([Hugging Face](https://huggingface.co/recobo/agriculture-bert-uncased)) was central to our system's performance.
- Challenge organizers for the dataset and evaluation framework.

---

## Contact

For questions or issues, please open an issue on this repository.

**Authors:**
- Nestor Moulias
- Areti Kirpitsa
