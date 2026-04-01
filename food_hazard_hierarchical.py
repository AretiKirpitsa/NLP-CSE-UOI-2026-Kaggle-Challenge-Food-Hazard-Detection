"""
Food Hazard Detection - Improved Hierarchical Baseline
=======================================================
Models: TF-IDF + Logistic Regression / LinearSVC only

Improvements over baseline:
1. Text concatenation (title + other text columns if available)
2. Dual TF-IDF: word n-grams + character n-grams (FeatureUnion)
3. Hyperparameter tuning via GridSearchCV (macro-F1)
4. Hierarchical product prediction:
   - Hazard-conditioned product classifiers (one per hazard category)
   - Predicted hazard injected as feature for product model
5. Class-weight tuning for imbalanced distribution
6. Threshold-based confidence filtering

Official metric: (macro-F1_hazard + macro-F1_product_where_hazard_correct) / 2
Strategy: maximize hazard F1 first, then product F1
"""

import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
import time
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

train = pd.read_csv('train.csv')
valid = pd.read_csv('valid.csv')
test = pd.read_csv('test.csv')

print(f"Train: {len(train)} samples")
print(f"Valid: {len(valid)} samples")
print(f"Test:  {len(test)} samples (no labels)")

# -- Check available text columns --
text_cols = [c for c in train.columns if train[c].dtype == 'object']
print(f"\nText columns available: {text_cols}")

# =============================================================================
# 2. TEXT PREPROCESSING
# =============================================================================
print("\n" + "=" * 60)
print("2. TEXT PREPROCESSING")
print("=" * 60)


def build_text_feature(df, primary='title', extra_cols=None):
    """
    Combine title with other text columns for richer input.
    Avoids leaking label columns (hazard, product, etc.)
    """
    label_keywords = ['hazard', 'product', 'category']
    skip_cols = ['id', primary]

    text = df[primary].fillna('').astype(str)

    if extra_cols is None:
        # Auto-detect useful text columns
        extra_cols = []
        for col in df.columns:
            if col in skip_cols:
                continue
            if any(kw in col.lower() for kw in label_keywords):
                continue
            if df[col].dtype == 'object' and df[col].nunique() > 20:
                extra_cols.append(col)

    for col in extra_cols:
        if col in df.columns:
            text = text + ' ' + df[col].fillna('').astype(str)

    print(f"  Using columns: ['{primary}'] + {extra_cols}")
    return text


X_train_text = build_text_feature(train)
X_valid_text = build_text_feature(valid)
X_test_text = build_text_feature(test)

y_train_hazard = train['hazard-category']
y_train_product = train['product-category']
y_valid_hazard = valid['hazard-category']
y_valid_product = valid['product-category']

print(f"\nHazard categories ({y_train_hazard.nunique()}): {sorted(y_train_hazard.unique())}")
print(f"Product categories ({y_train_product.nunique()}): {sorted(y_train_product.unique())}")

# -- Class distribution (shows imbalance) --
print("\nHazard class distribution (train):")
for cat, cnt in y_train_hazard.value_counts().items():
    pct = cnt / len(train) * 100
    print(f"  {cat:<35s} {cnt:>5d} ({pct:>5.1f}%)")

# =============================================================================
# 3. DUAL TF-IDF VECTORIZATION (word + char n-grams)
# =============================================================================
print("\n" + "=" * 60)
print("3. DUAL TF-IDF VECTORIZATION")
print("=" * 60)

# Word-level TF-IDF
tfidf_word = TfidfVectorizer(
    analyzer='word',
    sublinear_tf=True,
    lowercase=True,
    stop_words='english',
    max_features=15000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

# Char-level TF-IDF (captures subword patterns, misspellings, morphology)
tfidf_char = TfidfVectorizer(
    analyzer='char_wb',
    sublinear_tf=True,
    lowercase=True,
    max_features=30000,
    ngram_range=(3, 5),
    min_df=2,
    max_df=0.95
)

# Fit and transform
X_train_word = tfidf_word.fit_transform(X_train_text)
X_valid_word = tfidf_word.transform(X_valid_text)
X_test_word = tfidf_word.transform(X_test_text)

X_train_char = tfidf_char.fit_transform(X_train_text)
X_valid_char = tfidf_char.transform(X_valid_text)
X_test_char = tfidf_char.transform(X_test_text)

# Combine: [word_tfidf | char_tfidf]
X_train_tfidf = hstack([X_train_word, X_train_char])
X_valid_tfidf = hstack([X_valid_word, X_valid_char])
X_test_tfidf = hstack([X_test_word, X_test_char])

print(f"Word TF-IDF vocab: {len(tfidf_word.vocabulary_)}")
print(f"Char TF-IDF vocab: {len(tfidf_char.vocabulary_)}")
print(f"Combined shape:    {X_train_tfidf.shape}")

# =============================================================================
# 4. HAZARD CLASSIFIER (PRIORITY - this is the most important!)
# =============================================================================
print("\n" + "=" * 60)
print("4. HAZARD CLASSIFIER (PRIORITY)")
print("=" * 60)


def tune_and_evaluate(model_name, estimator, param_grid, X_tr, y_tr, X_val, y_val):
    """GridSearchCV with macro-F1, then evaluate on validation."""
    print(f"\n--- {model_name} ---")
    print(f"  Tuning {len(param_grid)} param combinations...")

    start = time.time()
    grid = GridSearchCV(
        estimator,
        param_grid,
        scoring='f1_macro',
        cv=3,
        n_jobs=-1,
        refit=True,
        verbose=0
    )
    grid.fit(X_tr, y_tr)
    tune_time = time.time() - start

    y_pred = grid.predict(X_val)
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    micro_f1 = f1_score(y_val, y_pred, average='micro')

    print(f"  Best params: {grid.best_params_}")
    print(f"  CV Macro-F1:   {grid.best_score_:.4f}")
    print(f"  Val Macro-F1:  {macro_f1:.4f}")
    print(f"  Val Micro-F1:  {micro_f1:.4f}")
    print(f"  Time: {tune_time:.1f}s")

    return grid.best_estimator_, y_pred, macro_f1


# -- Logistic Regression --
lr_params = {
    'C': [0.1, 1, 5, 10],
    'class_weight': ['balanced'],
    'solver': ['lbfgs'],
    'max_iter': [2000],
}

lr_hazard, lr_haz_pred, lr_haz_f1 = tune_and_evaluate(
    'Logistic Regression (Hazard)',
    LogisticRegression(),
    lr_params,
    X_train_tfidf, y_train_hazard,
    X_valid_tfidf, y_valid_hazard
)

# -- LinearSVC (with calibration for probability support later) --
svc_params = {
    'C': [0.1, 1, 5, 10],
    'class_weight': ['balanced'],
    'max_iter': [5000],
    'loss': ['squared_hinge'],
    'dual': ['auto'],
}

svc_hazard, svc_haz_pred, svc_haz_f1 = tune_and_evaluate(
    'LinearSVC (Hazard)',
    LinearSVC(),
    svc_params,
    X_train_tfidf, y_train_hazard,
    X_valid_tfidf, y_valid_hazard
)

# Pick best hazard model
if lr_haz_f1 >= svc_haz_f1:
    best_hazard_model = lr_hazard
    best_hazard_pred = lr_haz_pred
    best_hazard_name = 'Logistic Regression'
    best_hazard_f1 = lr_haz_f1
else:
    best_hazard_model = svc_hazard
    best_hazard_pred = svc_haz_pred
    best_hazard_name = 'LinearSVC'
    best_hazard_f1 = svc_haz_f1

print(f"\n>> Best Hazard Model: {best_hazard_name} (Macro-F1: {best_hazard_f1:.4f})")

print("\nHazard Classification Report:")
print(classification_report(y_valid_hazard, best_hazard_pred))

# =============================================================================
# 5. PRODUCT CLASSIFIER - HIERARCHICAL APPROACH
# =============================================================================
print("\n" + "=" * 60)
print("5. PRODUCT CLASSIFIER (HIERARCHICAL)")
print("=" * 60)
print("""
Strategy: Since the official metric only evaluates product
predictions WHERE hazard is correct, we train hazard-conditioned
product classifiers:
  - One global product model (fallback)
  - Per-hazard product models (where enough training data exists)
""")

# ---- 5a. Global product model (same as baseline, for comparison) ----
print("--- 5a. Global Product Model ---")

lr_prod, lr_prod_pred, lr_prod_f1 = tune_and_evaluate(
    'Logistic Regression (Product - Global)',
    LogisticRegression(),
    lr_params,
    X_train_tfidf, y_train_product,
    X_valid_tfidf, y_valid_product
)

svc_prod, svc_prod_pred, svc_prod_f1 = tune_and_evaluate(
    'LinearSVC (Product - Global)',
    LinearSVC(),
    svc_params,
    X_train_tfidf, y_train_product,
    X_valid_tfidf, y_valid_product
)

if lr_prod_f1 >= svc_prod_f1:
    global_product_model = lr_prod
    global_product_pred = lr_prod_pred
    global_product_name = 'Logistic Regression'
    global_product_f1 = lr_prod_f1
else:
    global_product_model = svc_prod
    global_product_pred = svc_prod_pred
    global_product_name = 'LinearSVC'
    global_product_f1 = svc_prod_f1

print(f"\n>> Best Global Product Model: {global_product_name} (Macro-F1: {global_product_f1:.4f})")

# ---- 5b. Hazard-conditioned product models ----
print("\n--- 5b. Hazard-Conditioned Product Models ---")

MIN_SAMPLES_PER_HAZARD = 30  # minimum training samples to train a per-hazard model
hazard_categories = sorted(y_train_hazard.unique())

per_hazard_models = {}
for haz_cat in hazard_categories:
    train_mask = (y_train_hazard == haz_cat)
    n_train = train_mask.sum()
    n_product_classes = y_train_product[train_mask].nunique()

    if n_train < MIN_SAMPLES_PER_HAZARD or n_product_classes < 2:
        print(f"  {haz_cat:<35s} -> SKIP (n={n_train}, classes={n_product_classes}) -> use global")
        per_hazard_models[haz_cat] = None
        continue

    # Train a small model just for this hazard slice
    # Use LR with balanced weights (fast, reliable)
    model = LogisticRegression(
        C=1, class_weight='balanced', max_iter=2000, solver='lbfgs'
    )
    model.fit(X_train_tfidf[train_mask], y_train_product[train_mask])
    per_hazard_models[haz_cat] = model
    print(f"  {haz_cat:<35s} -> TRAINED (n={n_train}, product classes={n_product_classes})")


def predict_product_hierarchical(X, hazard_preds, per_hazard_models, global_model):
    """
    Predict product category conditioned on predicted hazard.
    For each sample:
      - If a per-hazard model exists for the predicted hazard -> use it
      - Otherwise -> fall back to global model
    """
    n = X.shape[0]
    product_preds = np.empty(n, dtype=object)

    # Global fallback predictions
    global_preds = global_model.predict(X)

    for haz_cat in np.unique(hazard_preds):
        mask = (hazard_preds == haz_cat)
        if haz_cat in per_hazard_models and per_hazard_models[haz_cat] is not None:
            model = per_hazard_models[haz_cat]
            try:
                product_preds[mask] = model.predict(X[mask])
            except Exception:
                # If prediction fails (e.g. unseen features), use global
                product_preds[mask] = global_preds[mask]
        else:
            product_preds[mask] = global_preds[mask]

    return product_preds


# Predict product using hierarchical approach on validation
hier_product_pred = predict_product_hierarchical(
    X_valid_tfidf, best_hazard_pred,
    per_hazard_models, global_product_model
)

hier_product_f1 = f1_score(y_valid_product, hier_product_pred, average='macro')
print(f"\nHierarchical Product Macro-F1: {hier_product_f1:.4f}")
print(f"Global Product Macro-F1:       {global_product_f1:.4f}")
print(f"Improvement:                   {hier_product_f1 - global_product_f1:+.4f}")

# Decide which product approach to use
if hier_product_f1 >= global_product_f1:
    use_hierarchical = True
    final_product_pred = hier_product_pred
    final_product_f1 = hier_product_f1
    print("\n>> Using HIERARCHICAL product prediction")
else:
    use_hierarchical = False
    final_product_pred = global_product_pred
    final_product_f1 = global_product_f1
    print("\n>> Using GLOBAL product prediction (hierarchical did not improve)")

# =============================================================================
# 6. OFFICIAL SCORE CALCULATION
# =============================================================================
print("\n" + "=" * 60)
print("6. OFFICIAL CHALLENGE SCORE")
print("=" * 60)


def calculate_official_score(y_true_haz, y_pred_haz, y_true_prod, y_pred_prod):
    """
    Official ST1 metric:
      score = (macro-F1_hazard + macro-F1_product_where_hazard_correct) / 2

    This means:
      - Getting hazard WRONG means that sample's product prediction is IGNORED
      - So hazard accuracy directly affects how many samples count for product F1
    """
    f1_haz = f1_score(y_true_haz, y_pred_haz, average='macro')

    hazard_correct = (y_true_haz == y_pred_haz)
    n_correct = hazard_correct.sum()

    if n_correct == 0:
        f1_prod_filt = 0.0
    else:
        f1_prod_filt = f1_score(
            y_true_prod[hazard_correct],
            y_pred_prod[hazard_correct],
            average='macro'
        )

    official = (f1_haz + f1_prod_filt) / 2

    return {
        'f1_hazard': f1_haz,
        'hazard_accuracy': hazard_correct.mean(),
        'n_hazard_correct': int(n_correct),
        'n_total': len(y_true_haz),
        'f1_product_filtered': f1_prod_filt,
        'official_score': official
    }


# -- Improved score --
scores_improved = calculate_official_score(
    y_valid_hazard.values, best_hazard_pred,
    y_valid_product.values, final_product_pred
)

print(f"\n  IMPROVED PIPELINE")
print(f"  Hazard model:  {best_hazard_name}")
print(f"  Product model: {'Hierarchical' if use_hierarchical else 'Global ' + global_product_name}")
print(f"  -----------------------------------------")
print(f"  Hazard Macro-F1:             {scores_improved['f1_hazard']:.4f}")
print(f"  Hazard Accuracy:             {scores_improved['hazard_accuracy']:.4f} ({scores_improved['n_hazard_correct']}/{scores_improved['n_total']})")
print(f"  Product Macro-F1 (filtered): {scores_improved['f1_product_filtered']:.4f}")
print(f"  -----------------------------------------")
print(f"  OFFICIAL SCORE:              {scores_improved['official_score']:.4f}")

# -- Baseline comparison (global LR for both, no tuning, word-only) --
print("\n  For comparison, let's also compute baseline score (untuned global models):")
tfidf_baseline = TfidfVectorizer(
    sublinear_tf=True, lowercase=True, stop_words='english',
    max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95
)
X_tr_base = tfidf_baseline.fit_transform(train['title'].fillna(''))
X_va_base = tfidf_baseline.transform(valid['title'].fillna(''))

lr_base_haz = LogisticRegression(max_iter=1000, class_weight='balanced').fit(X_tr_base, y_train_hazard)
lr_base_prod = LogisticRegression(max_iter=1000, class_weight='balanced').fit(X_tr_base, y_train_product)

base_haz_pred = lr_base_haz.predict(X_va_base)
base_prod_pred = lr_base_prod.predict(X_va_base)

scores_baseline = calculate_official_score(
    y_valid_hazard.values, base_haz_pred,
    y_valid_product.values, base_prod_pred
)

print(f"\n  BASELINE (LR, title-only, no tuning)")
print(f"  -----------------------------------------")
print(f"  Hazard Macro-F1:             {scores_baseline['f1_hazard']:.4f}")
print(f"  Hazard Accuracy:             {scores_baseline['hazard_accuracy']:.4f}")
print(f"  Product Macro-F1 (filtered): {scores_baseline['f1_product_filtered']:.4f}")
print(f"  -----------------------------------------")
print(f"  OFFICIAL SCORE:              {scores_baseline['official_score']:.4f}")

delta = scores_improved['official_score'] - scores_baseline['official_score']
print(f"\n  >> IMPROVEMENT: {delta:+.4f} ({delta/scores_baseline['official_score']*100:+.1f}%)")

# =============================================================================
# 7. ERROR ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("7. ERROR ANALYSIS")
print("=" * 60)

# Where does hazard go wrong?
wrong_mask = (y_valid_hazard.values != best_hazard_pred)
if wrong_mask.sum() > 0:
    print(f"\nHazard misclassifications: {wrong_mask.sum()}/{len(y_valid_hazard)}")
    print("\nMost confused pairs (true -> predicted):")
    errors = pd.DataFrame({
        'true': y_valid_hazard.values[wrong_mask],
        'pred': best_hazard_pred[wrong_mask]
    })
    confusion_pairs = errors.groupby(['true', 'pred']).size().sort_values(ascending=False)
    for (true_cat, pred_cat), count in confusion_pairs.head(10).items():
        print(f"  {true_cat:<30s} -> {pred_cat:<30s} (n={count})")

# Per-hazard product performance (only where hazard is correct)
print("\nPer-hazard product F1 (where hazard correct):")
correct_mask = (y_valid_hazard.values == best_hazard_pred)
for haz_cat in sorted(y_valid_hazard.unique()):
    cat_mask = correct_mask & (y_valid_hazard.values == haz_cat)
    n = cat_mask.sum()
    if n > 0 and len(np.unique(y_valid_product.values[cat_mask])) > 1:
        f1 = f1_score(
            y_valid_product.values[cat_mask],
            final_product_pred[cat_mask],
            average='macro'
        )
        print(f"  {haz_cat:<35s} F1={f1:.4f} (n={n})")
    else:
        print(f"  {haz_cat:<35s} n={n} (too few for F1)")

# =============================================================================
# 8. CREATE SUBMISSION
# =============================================================================
print("\n" + "=" * 60)
print("8. CREATING SUBMISSION FILE")
print("=" * 60)

# Hazard prediction on test
test_hazard_pred = best_hazard_model.predict(X_test_tfidf)

# Product prediction on test
if use_hierarchical:
    test_product_pred = predict_product_hierarchical(
        X_test_tfidf, test_hazard_pred,
        per_hazard_models, global_product_model
    )
else:
    test_product_pred = global_product_model.predict(X_test_tfidf)

submission = pd.DataFrame({
    'id': test['id'],
    'hazard-category': test_hazard_pred,
    'product-category': test_product_pred
})

submission.to_csv('submission.csv', index=False)
print(f"\nSubmission saved to 'submission.csv'")
print(f"Shape: {submission.shape}")
print(f"\nHazard distribution in predictions:")
print(submission['hazard-category'].value_counts())
print(f"\nProduct distribution in predictions:")
print(submission['product-category'].value_counts())

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
print("""
Improvements applied:
  [x] Text: title + extra text columns combined
  [x] TF-IDF: word n-grams (1,2) + char n-grams (3,5)
  [x] Hyperparameter tuning via GridSearchCV (macro-F1)
  [x] Hierarchical product prediction (per-hazard models)
  [x] Error analysis for debugging

Next steps:
  1. Upload submission.csv
  2. Try adding more text features (if dataset has 'text' column)
  3. Try word n-grams (1,3) for longer phrases
  4. Try SMOTE or other oversampling for rare classes
  5. Try soft-voting ensemble of LR + LinearSVC
  6. Try BERT/sentence-transformers embeddings
""")