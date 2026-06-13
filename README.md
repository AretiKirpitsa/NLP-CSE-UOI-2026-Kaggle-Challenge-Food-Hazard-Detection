# SemEval-2025 Task 9 — The Food Hazard Detection Challenge

**Authors:** Nestor Moulias, Areti Kirpitsa

This repository contains our submission to **SemEval-2025 Task 9, Subtask 1: The Food Hazard Detection Challenge**, a text classification task that jointly predicts the **hazard category** and **product category** of food-incident reports collected from the web. Our final system reached **Top 8** on the kaggle leaderboard with a score of **0.85205**.

The repository documents the full research progression, from a tuned classical baseline to a domain-aware transformer ensemble, so that each architectural decision can be reproduced and inspected independently.

---

## Final Result

| System | Private Score | Public Score |
|---|---|---|
| Final soft-voting ensemble (RoBERTa + AgriBERT + TF-IDF/LogReg) | **0.85205 (Top 8)** | 0.75998 |
| Madrid pipeline (AgriBERT + joint decoding, valid) | 0.83521 | 0.77351 |
| DeBERTa multi-seed Ensemble| 0.81519 | **0.77363** |
| Tuned TF-IDF + Logistic Regression | 0.68891 | 0.61826 |

The decisive ingredient was the inclusion of **AgriBERT** (`recobo/agriculture-bert-uncased`), a transformer pretrained on an agriculture-focused corpus whose vocabulary aligns closely with the lexical space of food-recall reports (crop names, agrochemicals, allergens, regulatory terminology).

---

## Repository Structure

```
.
├── tfidf_hazard/
│   └── tfidfHazardFeaturesONLY.py        # Tuned TF-IDF + selective keywords
├── deberta_pipeline/
│   └── Full_DeBERTa_Hazard_LARGE_pipeline_PRODUCT_v3.ipynb
├── madrid/
│   └── madrid.py                         # AgriBERT + focal loss + joint decoding
├── final_ensemble/
│   └── testski4_5.ipynb                  # Final winning architecture
├── data/                                 # train.csv, valid.csv, test.csv, synthetic_samples.csv
├── outputs/                              # Predictions + cached features
├── report/                               # Paper PDF
└── README.md
```

> **Note:** raw challenge data is not included in this repository due to licensing. Place the CSV files under `data/` before running any of the scripts or notebooks.

---

## Architectures

This project followed a deliberate progression of architectures. Each one is documented below and lives in its own folder for transparency and reproducibility.

### 1. Tuned TF-IDF + Logistic Regression (`tfidf_hazard/`)

Our starting point and a foundational component of every subsequent system. Key elements:

- **Combined word and character TF-IDF features** (word n-grams + `char_wb` n-grams), stacked into a single sparse representation.
- **Grid search over the full feature pipeline**, jointly tuning word n-gram range, character n-gram range, word/character vocabulary size, and the Logistic Regression regularization strength `C`.
- **Selective keyword features for rare hazard classes**: hand-curated keyword lists (drawn from EU RASFF and FSIS taxonomies and validated on training data) are applied *only* to classes below a sample-count threshold, yielding two features per class (count + binary indicator). Frequent classes are left to the statistical TF-IDF signal alone, avoiding noise injection.
- **Artifact caching**: the fitted vectorizers, sparse feature matrices, keyword dictionary, and tuned model are persisted to disk so that downstream pipelines can consume them directly as a lightweight, decorrelated signal — without re-fitting.

### 2. DeBERTa-v3-Large Pipeline (`deberta_pipeline/`)

A heavier transformer pipeline that treats hazard and product separately and then ensembles within each subtask.

- **Hazard**: DeBERTa-v3-Large fine-tuned over multiple seeds, combined via multi-seed averaging, and softly ensembled with the tuned TF-IDF + LogReg member.
- **Product**: DeBERTa-v3-Large fine-tuned over three seeds, combined with an XGBoost classifier trained on the cached TF-IDF features, ensembled into a single prediction.
- **Per-seed feature caching**: probabilities, logits, and pooled embeddings are saved to disk so that any subset of seeds can be re-ensembled without re-training.

### 3. Madrid — AgriBERT + Focal Loss + Joint Decoding (`madrid/`)

A research pipeline that introduces AgriBERT to our system and explores the dependency structure between hazard and product labels.

- **Backbone**: `recobo/agriculture-bert-uncased`, our first domain-pretrained encoder.
- **Focal loss** (Lin et al., 2020) for class imbalance, with `gamma=2.0` and balanced class weights.
- **Classifier-chain inputs**: the product model is conditioned on the hazard prediction by prepending `hazard-category: <h> [SEP]` to its input, with **exposure-bias mixing** during training (gold hazard with probability 0.70, predicted hazard otherwise) to make the chain robust at inference time.
- **Marginalized joint decoding** over (hazard, product) pairs at inference: instead of committing to `argmax` hazard, the product chain is run once per hazard prefix and a joint score is computed as
  `score(h, p) = α · log P(h|x) + (1-α) · log P(p|x, h) + λ · log P(h, p) + I[compat(h, p)]`,
  where `log P(h, p)` is a smoothed bigram prior from training and `compat` is a hard co-occurrence mask. α and λ are tuned on validation by a 1-D line search.
- **Monte Carlo dropout passes** for both hazard and product to obtain better-calibrated probabilities.
- **Submission gate**: the Madrid prediction is submitted only if its validation score, agreement with the baseline, and absolute score all clear configured thresholds; otherwise a safer fallback is used.

This pipeline scored **0.76656** on validation and confirmed that AgriBERT was the right backbone to invest in.

### 4. Final Architecture — Soft-Voting Ensemble (`final_ensemble/`)

The winning system. Three deliberately decorrelated members:

- **RoBERTa-base** — general-purpose transformer baseline.
- **AgriBERT** (`recobo/agriculture-bert-uncased`) — domain-pretrained encoder, the dominant member.
- **TF-IDF + Logistic Regression** — the same lexical signal first built in stage 1.

Each member produces probability distributions over both hazard and product classes. Predictions are combined via **weighted soft voting**, with weights selected by grid search over a discretized simplex, optimizing the official SemEval metric on **honest out-of-fold (OOF) predictions** from 5-fold stratified cross-validation.

Additional components:
- **Synthetic data augmentation**, injected into training folds only (never validation), preserving the integrity of the OOF score.
- **Deterministic post-processing rule** (`migration → food contact materials`) for a known systematic label dependency.

---

## How to Reproduce

### Requirements

```bash
pip install -r requirements.txt
```

Main dependencies: `transformers`, `torch`, `scikit-learn`, `xgboost`, `pandas`, `numpy`, `scipy`, `sentencepiece`, `joblib`.

### Data Setup

Place the challenge files under `data/`:
- `train.csv`, `valid.csv`, `test.csv`
- `synthetic_samples.csv` (for the augmentation step in the final ensemble)

### Running

Each component is self-contained and can be run independently:

```bash
# 1. Tuned TF-IDF baseline (produces cached features for downstream use)
python tfidf_hazard/tfidfHazardFeaturesONLY.py

# 2. DeBERTa-v3-large pipeline
jupyter notebook deberta_pipeline/Full_DeBERTa_Hazard_LARGE_pipeline_PRODUCT_v3.ipynb

# 3. Madrid (AgriBERT + joint decoding)
python madrid/madrid.py

# 4. Final ensemble (reproduces the leaderboard submission)
jupyter notebook final_ensemble/testski4_5.ipynb
```

The final-ensemble notebook caches OOF and test probability arrays per backbone (`oofH_*.npy`, `oofP_*.npy`, etc.), so one backbone can be run per session and resumed later.

**Hardware:** GPU recommended (Kaggle T4×2 or equivalent). One full transformer backbone in the final ensemble takes ~3–4 hours with a single seed.

---

## Acknowledgments

- The **AgriBERT** model by `recobo` ([Hugging Face](https://huggingface.co/recobo/agriculture-bert-uncased)) was central to our system's performance.
- The challenge organizers for the dataset and evaluation framework.

---

## Contact

For questions or issues, please open an issue on this repository.

**Authors:**
- Nestor Moulias
- Areti Kirpitsa
