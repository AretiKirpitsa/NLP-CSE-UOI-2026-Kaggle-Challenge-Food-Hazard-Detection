"""
Food Hazard Detection - Baseline Models
========================================
Βήμα 1: TF-IDF + Logistic Regression (απλό baseline)

Preprocessing για TF-IDF:
- Lowercasing ✓
- Tokenization ✓
- Stop-words removal ✓ (θα δοκιμάσουμε και χωρίς)
- NO Stemming (θα δοκιμάσουμε)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
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

# Χρησιμοποιούμε το 'title' ως input (όπως στο official challenge)
X_train = train['title'].fillna('')
X_valid = valid['title'].fillna('')
X_test = test['title'].fillna('')

y_train_hazard = train['hazard-category']
y_train_product = train['product-category']
y_valid_hazard = valid['hazard-category']
y_valid_product = valid['product-category']

print(f"\nHazard categories: {y_train_hazard.nunique()}")
print(f"Product categories: {y_train_product.nunique()}")

# =============================================================================
# 2. TF-IDF VECTORIZATION
# =============================================================================
print("\n" + "=" * 60)
print("2. TF-IDF VECTORIZATION")
print("=" * 60)

# Απλό TF-IDF (θα πειραματιστούμε με παραμέτρους αργότερα)
tfidf = TfidfVectorizer(
    lowercase=True,           # Lowercasing
    stop_words='english',     # Remove stop-words
    max_features=5000,        # Limit vocabulary
    ngram_range=(1, 2),       # Unigrams + Bigrams
    min_df=2,                 # Ignore very rare words
    max_df=0.95               # Ignore very common words
)

# Fit στο train, transform σε όλα
X_train_tfidf = tfidf.fit_transform(X_train)
X_valid_tfidf = tfidf.transform(X_valid)
X_test_tfidf = tfidf.transform(X_test)

print(f"Vocabulary size: {len(tfidf.vocabulary_)}")
print(f"Train shape: {X_train_tfidf.shape}")
print(f"Valid shape: {X_valid_tfidf.shape}")

# =============================================================================
# 3. BASELINE MODELS
# =============================================================================
print("\n" + "=" * 60)
print("3. TRAINING BASELINE MODELS")
print("=" * 60)

def evaluate_model(model, X_train, y_train, X_valid, y_valid, name):
    """Train and evaluate a model"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    
    # Macro F1 (αυτό χρησιμοποιεί το challenge)
    macro_f1 = f1_score(y_valid, y_pred, average='macro')
    micro_f1 = f1_score(y_valid, y_pred, average='micro')
    
    print(f"\n{name}:")
    print(f"  Macro-F1: {macro_f1:.4f}")
    print(f"  Micro-F1: {micro_f1:.4f}")
    
    return model, y_pred, macro_f1

# Models to try
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Linear SVM': LinearSVC(max_iter=2000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1),
    'Naive Bayes': MultinomialNB(alpha=0.1)
}

# =============================================================================
# 3a. HAZARD CATEGORY (το πιο σημαντικό!)
# =============================================================================
print("\n--- HAZARD CATEGORY ---")

hazard_results = {}
for name, model in models.items():
    trained_model, y_pred, f1 = evaluate_model(
        model, X_train_tfidf, y_train_hazard, 
        X_valid_tfidf, y_valid_hazard, name
    )
    hazard_results[name] = {'model': trained_model, 'f1': f1, 'pred': y_pred}

best_hazard = max(hazard_results.items(), key=lambda x: x[1]['f1'])
print(f"\n✓ Best Hazard Model: {best_hazard[0]} (Macro-F1: {best_hazard[1]['f1']:.4f})")

# =============================================================================
# 3b. PRODUCT CATEGORY
# =============================================================================
print("\n--- PRODUCT CATEGORY ---")

product_results = {}
for name, model_class in [
    ('Logistic Regression', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ('Linear SVM', LinearSVC(max_iter=2000, class_weight='balanced')),
    ('Random Forest', RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1)),
    ('Naive Bayes', MultinomialNB(alpha=0.1))
]:
    trained_model, y_pred, f1 = evaluate_model(
        model_class, X_train_tfidf, y_train_product, 
        X_valid_tfidf, y_valid_product, name
    )
    product_results[name] = {'model': trained_model, 'f1': f1, 'pred': y_pred}

best_product = max(product_results.items(), key=lambda x: x[1]['f1'])
print(f"\n✓ Best Product Model: {best_product[0]} (Macro-F1: {best_product[1]['f1']:.4f})")

# =============================================================================
# 4. OFFICIAL SCORE CALCULATION
# =============================================================================
print("\n" + "=" * 60)
print("4. OFFICIAL CHALLENGE SCORE")
print("=" * 60)

def calculate_official_score(y_true_hazard, y_pred_hazard, y_true_product, y_pred_product):
    """
    Official score: (macro-F1_hazard + macro-F1_product_where_hazard_correct) / 2
    """
    # Macro-F1 for hazard
    f1_hazard = f1_score(y_true_hazard, y_pred_hazard, average='macro')
    
    # Mask: only where hazard prediction is correct
    hazard_correct_mask = (y_true_hazard == y_pred_hazard)
    
    if hazard_correct_mask.sum() == 0:
        f1_product_filtered = 0.0
    else:
        # F1 for product only on correct hazard predictions
        f1_product_filtered = f1_score(
            y_true_product[hazard_correct_mask], 
            y_pred_product[hazard_correct_mask], 
            average='macro'
        )
    
    official_score = (f1_hazard + f1_product_filtered) / 2
    
    return {
        'f1_hazard': f1_hazard,
        'f1_product_filtered': f1_product_filtered,
        'hazard_accuracy': hazard_correct_mask.mean(),
        'official_score': official_score
    }

# Calculate for best models
best_hazard_pred = hazard_results[best_hazard[0]]['pred']
best_product_pred = product_results[best_product[0]]['pred']

scores = calculate_official_score(
    y_valid_hazard.values, best_hazard_pred,
    y_valid_product.values, best_product_pred
)

print(f"\nUsing: {best_hazard[0]} (hazard) + {best_product[0]} (product)")
print(f"\n  Hazard Macro-F1:        {scores['f1_hazard']:.4f}")
print(f"  Hazard Accuracy:        {scores['hazard_accuracy']:.4f} ({int(scores['hazard_accuracy']*len(y_valid_hazard))}/{len(y_valid_hazard)} correct)")
print(f"  Product Macro-F1 (filtered): {scores['f1_product_filtered']:.4f}")
print(f"\n  ★ OFFICIAL SCORE: {scores['official_score']:.4f}")

# =============================================================================
# 5. PER-CLASS ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("5. PER-CLASS ANALYSIS (Hazard)")
print("=" * 60)

print("\nClassification Report:")
print(classification_report(y_valid_hazard, best_hazard_pred))

# =============================================================================
# 6. CREATE SUBMISSION
# =============================================================================
print("\n" + "=" * 60)
print("6. CREATING SUBMISSION FILE")
print("=" * 60)

# Train final models on train data and predict test
hazard_model = hazard_results[best_hazard[0]]['model']
product_model = product_results[best_product[0]]['model']

# Predict on test set
test_hazard_pred = hazard_model.predict(X_test_tfidf)
test_product_pred = product_model.predict(X_test_tfidf)

# Create submission dataframe
submission = pd.DataFrame({
    'id': test['id'],
    'hazard-category': test_hazard_pred,
    'product-category': test_product_pred
})

submission.to_csv('submission.csv', index=False)
print(f"\nSubmission saved to 'submission_baseline.csv'")
print(f"Shape: {submission.shape}")
print(submission.head())

print("\n" + "=" * 60)
print("DONE! Next steps:")
print("=" * 60)
print("""
1. Upload submission_baseline.csv to Kaggle
2. Try different TF-IDF parameters (ngrams, max_features)
3. Try class_weight adjustments
4. Try ensemble methods
5. Try BERT embeddings
""")
