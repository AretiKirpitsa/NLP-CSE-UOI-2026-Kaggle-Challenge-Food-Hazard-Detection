"""
MADRID — Ultimate + marginalized joint decoding over (hazard, product)
========================================================================
Inference-time upgrade only. Ultimate training is unchanged, so the
0.76656 anchor is preserved by the gate.

Core idea:
  Instead of committing to argmax hazard then predicting product, we
  compute P(p | x, h=k) for EVERY hazard class k by running the
  product chain once per hazard prefix, then jointly decode:

      score(h, p)  =   alpha   * log P(h | x)
                     + (1-alpha)* log P(p | x, h)
                     + lam     * log P(h, p)_train    (empirical bigram prior)
                     + I[compat(h, p)]                (hard co-occurrence mask)

      (h*, p*) = argmax_{h,p} score(h, p)

  When hazard is confident this reduces to ultimate. When hazard is
  split, the product chain gets multiple shots and the bigram prior
  breaks ties using hazard-product relationships the classifier never
  saw as an explicit signal.

  - alpha, lam tuned on valid by 1-D line search (single scalars, low
    overfit risk vs the -0.012..-0.030 val->Kaggle gap).
  - Gate: submit madrid only if val ST1 >= 0.795 AND agreement w/ ult
    >= 90% AND madrid >= ultimate. Else fallback to ultimate.

Cost: product inference runs NUM_HAZ times per chain with MC=5 (vs
MC=10 single branch in ultimate). Net ~5x product inference time;
hazard inference and all training unchanged.
"""

import os, gc, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG (ultimate-identical training; madrid-specific inference)
# =============================================================================
MODEL_NAME          = "recobo/agriculture-bert-uncased"
MAX_LENGTH_HAZARD   = 128
MAX_LENGTH_PRODUCT  = 160
BATCH_SIZE          = 8
LEARNING_RATE       = 2e-5
WEIGHT_DECAY        = 0.01
WARMUP_RATIO        = 0.1
EPOCHS_HAZARD       = 3
EPOCHS_PRODUCT      = 4
FOCAL_GAMMA         = 2.0
ECC_ENSEMBLE_K      = 5
SEED                = 42
EXPOSURE_GOLD_RATIO = 0.70
MC_HAZARD_PASSES    = 10
MC_PRODUCT_PASSES   = 5     # reduced because we now run NUM_HAZ branches

# Joint-decoding hyperparams (tuned on valid)
ALPHA_GRID          = [0.30, 0.40, 0.50, 0.60, 0.70]
LAMBDA_GRID         = [0.00, 0.10, 0.20, 0.30]
PRIOR_SMOOTHING     = 1.0   # add-alpha smoothing on training bigram counts

# Gate
GATE_VAL_FLOOR      = 0.795
GATE_MIN_AGREE      = 0.90

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
set_seed(SEED)

# =============================================================================
# DATA
# =============================================================================
train_df = pd.read_csv('train.csv')
valid_df = pd.read_csv('valid.csv')
test_df  = pd.read_csv('test.csv')
print(f"Train: {len(train_df)} | Valid: {len(valid_df)} | Test: {len(test_df)}")

def build_text(df):
    title = df['title'].fillna('').astype(str)
    if 'text' in df.columns:
        body = df['text'].fillna('').astype(str)
        return (title + ' [SEP] ' + body).tolist()
    return title.tolist()

X_train_text = build_text(train_df)
X_valid_text = build_text(valid_df)
X_test_text  = build_text(test_df)

y_train_hazard  = train_df['hazard-category'].values
y_train_product = train_df['product-category'].values
y_valid_hazard  = valid_df['hazard-category'].values
y_valid_product = valid_df['product-category'].values

hazard_enc  = LabelEncoder().fit(y_train_hazard)
product_enc = LabelEncoder().fit(y_train_product)
NUM_HAZ  = len(hazard_enc.classes_)
NUM_PROD = len(product_enc.classes_)
HAZ_CLASSES_STR = list(hazard_enc.classes_)
print(f"Hazard classes: {NUM_HAZ}  Product classes: {NUM_PROD}")

y_train_haz_int  = hazard_enc.transform(y_train_hazard)
y_train_prod_int = product_enc.transform(y_train_product)
y_valid_haz_int  = hazard_enc.transform(y_valid_hazard)
y_valid_prod_int = product_enc.transform(y_valid_product)

# =============================================================================
# BIGRAM PRIOR log P(h, p) from training (add-alpha smoothed)
# =============================================================================
bigram = np.full((NUM_HAZ, NUM_PROD), PRIOR_SMOOTHING, dtype=np.float64)
for h, p in zip(y_train_haz_int, y_train_prod_int):
    bigram[h, p] += 1.0
bigram /= bigram.sum()
log_bigram = np.log(bigram)  # (H, P)

# COMPAT mask (train + valid co-occurrence)
compat = np.zeros((NUM_HAZ, NUM_PROD), dtype=np.float32)
for h, p in zip(y_train_haz_int, y_train_prod_int): compat[h, p] = 1.0
for h, p in zip(y_valid_haz_int, y_valid_prod_int): compat[h, p] = 1.0
compat[compat.sum(1) == 0] = 1.0
print(f"Compat: avg {compat.sum(1).mean():.1f} products allowed per hazard")

# =============================================================================
# MODEL / LOSS / LOADERS (ultimate-identical)
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, w, gamma=2.0):
        super().__init__(); self.gamma = gamma
        self.register_buffer('w', torch.tensor(w, dtype=torch.float32))
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        pt = (probs * F.one_hot(targets, logits.size(-1)).float()).sum(-1).clamp(1e-8)
        return (-self.w[targets] * (1 - pt) ** self.gamma * torch.log(pt)).mean()

def cw_for(lbl, enc):
    li = enc.transform(lbl)
    return compute_class_weight('balanced', classes=np.arange(len(enc.classes_)), y=li).astype(np.float32)

hazard_w  = cw_for(y_train_hazard,  hazard_enc)
product_w = cw_for(y_train_product, product_enc)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class FoodHazardDataset(TorchDataset):
    def __init__(self, texts, labels, tok, ml):
        self.t, self.l, self.k, self.m = texts, labels, tok, ml
    def __len__(self): return len(self.t)
    def __getitem__(self, i):
        e = self.k(self.t[i], truncation=True, max_length=self.m,
                   padding='max_length', return_tensors='pt')
        item = {'input_ids': e['input_ids'].squeeze(0),
                'attention_mask': e['attention_mask'].squeeze(0)}
        if 'token_type_ids' in e: item['token_type_ids'] = e['token_type_ids'].squeeze(0)
        item['labels'] = torch.tensor(self.l[i], dtype=torch.long)
        return item

class AgricultureBERTClassifier(nn.Module):
    def __init__(self, name, num_classes, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
        h = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(h, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_classes))
    def forward(self, input_ids, attention_mask, token_type_ids=None, **kw):
        o = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids)
        return self.classifier(o.last_hidden_state[:, 0, :])

def train_model(m, tr, va, fl, ep, lr, wu, wd, dev, lab):
    m.to(dev)
    nd = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    params = [
        {'params': [p for n,p in m.named_parameters() if not any(x in n for x in nd)], 'weight_decay': wd},
        {'params': [p for n,p in m.named_parameters() if any(x in n for x in nd)], 'weight_decay': 0.0}]
    opt = torch.optim.AdamW(params, lr=lr)
    ts  = len(tr) * ep
    sch = get_cosine_schedule_with_warmup(opt, int(ts*wu), ts)
    best_f1, best_state = 0.0, None
    for e in range(ep):
        m.train(); tl = 0.0; t0 = time.time()
        for b in tr:
            b = {k: v.to(dev) for k,v in b.items()}
            y = b.pop('labels')
            loss = fl(m(**b), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sch.step(); tl += loss.item()
        f1, _, _ = eval_model(m, va, dev)
        print(f"  [{lab}] E{e+1}/{ep} Loss:{tl/len(tr):.4f} Val:{f1:.4f} ({time.time()-t0:.1f}s)")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k,v in m.state_dict().items()}
    if best_state is not None:
        m.load_state_dict(best_state); m.to(dev)
    return m, best_f1

def eval_model(m, loader, dev):
    m.eval(); pr, pb, lb = [], [], []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(dev) for k,v in b.items()}
            y = b.pop('labels'); lg = m(**b); p = F.softmax(lg, dim=-1)
            pr.append(lg.argmax(-1).cpu().numpy()); pb.append(p.cpu().numpy()); lb.append(y.cpu().numpy())
    pr = np.concatenate(pr); pb = np.concatenate(pb); lb = np.concatenate(lb)
    return f1_score(lb, pr, average='macro'), pr, pb

def predict_det(m, loader, dev):
    m.eval(); pr, pb = [], []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(dev) for k,v in b.items()}; b.pop('labels', None)
            lg = m(**b); p = F.softmax(lg, dim=-1)
            pr.append(lg.argmax(-1).cpu().numpy()); pb.append(p.cpu().numpy())
    return np.concatenate(pr), np.concatenate(pb)

def mc_predict(m, loader, dev, n):
    m.train(); passes = []
    with torch.no_grad():
        for _ in range(n):
            run = []
            for b in loader:
                b = {k: v.to(dev) for k,v in b.items()}; b.pop('labels', None)
                lg = m(**b); run.append(F.softmax(lg, dim=-1).cpu().numpy())
            passes.append(np.concatenate(run))
    m.eval()
    mc = np.mean(passes, axis=0)
    return mc.argmax(axis=1), mc

def chained_mixed(texts, gold, pred, ratio, rng):
    use = rng.random(len(texts)) < ratio
    return [f"hazard-category: {gold[i] if use[i] else pred[i]} [SEP] {t}"
            for i,t in enumerate(texts)]
def chained(texts, h):
    return [f"hazard-category: {a} [SEP] {t}" for t,a in zip(texts, h)]

# =============================================================================
# 1. HAZARD CHAINS (5, ultimate-identical)
#    + store chain 'train-prediction' for exposure mixing in Loop 2
# =============================================================================
print("\n" + "="*70); print("LOOP 1: HAZARD CHAINS"); print("="*70)

all_hv, all_ht = [], []
chain_f1s = []
chain_bootstrap_idx = []
chain_haz_train_preds = []  # hazard-string predictions on the chain's bootstrap
dummy = np.zeros(len(test_df), dtype=np.int64)

for ci in range(ECC_ENSEMBLE_K):
    cs = SEED + ci * 111
    set_seed(cs)
    print(f"\n--- HAZ CHAIN {ci+1}/{ECC_ENSEMBLE_K} (seed={cs}) ---")

    bi = np.random.choice(len(X_train_text), size=len(X_train_text), replace=True)
    chain_bootstrap_idx.append(bi)
    Xb = [X_train_text[i] for i in bi]
    yhb = hazard_enc.transform(y_train_hazard[bi])

    hzt = FoodHazardDataset(Xb, yhb, tokenizer, MAX_LENGTH_HAZARD)
    hzv = FoodHazardDataset(X_valid_text, y_valid_haz_int, tokenizer, MAX_LENGTH_HAZARD)
    hze = FoodHazardDataset(X_test_text, dummy, tokenizer, MAX_LENGTH_HAZARD)
    ltr = DataLoader(hzt, batch_size=BATCH_SIZE, shuffle=True)
    lva = DataLoader(hzv, batch_size=BATCH_SIZE, shuffle=False)
    lte = DataLoader(hze, batch_size=BATCH_SIZE, shuffle=False)

    hm = AgricultureBERTClassifier(MODEL_NAME, NUM_HAZ)
    hf = FocalLoss(hazard_w, FOCAL_GAMMA).to(DEVICE)
    hm, hb = train_model(hm, ltr, lva, hf, EPOCHS_HAZARD, LEARNING_RATE,
                         WARMUP_RATIO, WEIGHT_DECAY, DEVICE, f"Haz-C{ci+1}")
    chain_f1s.append(hb)

    _, hvp = mc_predict(hm, lva, DEVICE, MC_HAZARD_PASSES)
    _, htp = mc_predict(hm, lte, DEVICE, MC_HAZARD_PASSES)
    all_hv.append(hvp); all_ht.append(htp)

    # Train-pred for exposure mixing (in chain-2 loop below)
    lns = DataLoader(hzt, batch_size=BATCH_SIZE, shuffle=False)
    _, htrp = predict_det(hm, lns, DEVICE)
    chain_haz_train_preds.append(hazard_enc.inverse_transform(htrp.argmax(1)))

    del hm, hf, hzt, hzv, hze, ltr, lva, lte, lns
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# Ensemble hazard (F1-weighted) — this is the P(h|x) we use in joint decoding
cw_arr = np.array(chain_f1s); cw_arr = cw_arr / cw_arr.sum()
ult_hv = np.tensordot(cw_arr, all_hv, axes=([0],[0]))  # (N_val, H)
ult_ht = np.tensordot(cw_arr, all_ht, axes=([0],[0]))  # (N_te,  H)

# =============================================================================
# 2. PRODUCT CHAINS — NUM_HAZ-branch conditional inference
#    P_c(p | x, h=k)  for every k, so joint decoding can marginalize.
# =============================================================================
print("\n" + "="*70); print("LOOP 2: PRODUCT CHAINS (NUM_HAZ-branch inference)"); print("="*70)

# Allocate per-chain branched product posteriors
# shape per chain: (H, N, P) for valid and test
chain_prod_val = np.zeros((ECC_ENSEMBLE_K, NUM_HAZ, len(valid_df), NUM_PROD), dtype=np.float32)
chain_prod_te  = np.zeros((ECC_ENSEMBLE_K, NUM_HAZ, len(test_df),  NUM_PROD), dtype=np.float32)

for ci in range(ECC_ENSEMBLE_K):
    cs = SEED + ci * 111
    set_seed(cs)
    rng = np.random.RandomState(cs + 77)
    print(f"\n--- PROD CHAIN {ci+1}/{ECC_ENSEMBLE_K} (seed={cs}) ---")

    bi = chain_bootstrap_idx[ci]
    Xb = [X_train_text[i] for i in bi]
    ypb = product_enc.transform(y_train_product[bi])
    htrs = chain_haz_train_preds[ci]

    Xtrc = chained_mixed(Xb, y_train_hazard[bi], htrs, EXPOSURE_GOLD_RATIO, rng)
    # For product training we still use chain ci's OWN argmax hazard on valid (for best-ckpt selection)
    haz_val_this_chain = hazard_enc.inverse_transform(all_hv[ci].argmax(1))
    Xvac_train = chained(X_valid_text, haz_val_this_chain)

    pdt = FoodHazardDataset(Xtrc, ypb, tokenizer, MAX_LENGTH_PRODUCT)
    pdv = FoodHazardDataset(Xvac_train, y_valid_prod_int, tokenizer, MAX_LENGTH_PRODUCT)
    lptr = DataLoader(pdt, batch_size=BATCH_SIZE, shuffle=True)
    lpva = DataLoader(pdv, batch_size=BATCH_SIZE, shuffle=False)

    pm = AgricultureBERTClassifier(MODEL_NAME, NUM_PROD)
    pf = FocalLoss(product_w, FOCAL_GAMMA).to(DEVICE)
    pm, _ = train_model(pm, lptr, lpva, pf, EPOCHS_PRODUCT, LEARNING_RATE,
                        WARMUP_RATIO, WEIGHT_DECAY, DEVICE, f"Prod-C{ci+1}")

    # Conditional inference — ONE branch per hazard class
    print(f"  Branched inference: {NUM_HAZ} hazards x MC={MC_PRODUCT_PASSES}")
    for k, haz_str in enumerate(HAZ_CLASSES_STR):
        Xv_k = chained(X_valid_text, [haz_str] * len(X_valid_text))
        Xe_k = chained(X_test_text,  [haz_str] * len(X_test_text))
        dv_k = FoodHazardDataset(Xv_k, y_valid_prod_int, tokenizer, MAX_LENGTH_PRODUCT)
        de_k = FoodHazardDataset(Xe_k, dummy,            tokenizer, MAX_LENGTH_PRODUCT)
        lv_k = DataLoader(dv_k, batch_size=BATCH_SIZE, shuffle=False)
        le_k = DataLoader(de_k, batch_size=BATCH_SIZE, shuffle=False)
        _, pv_k = mc_predict(pm, lv_k, DEVICE, MC_PRODUCT_PASSES)
        _, pe_k = mc_predict(pm, le_k, DEVICE, MC_PRODUCT_PASSES)
        chain_prod_val[ci, k] = pv_k
        chain_prod_te[ci,  k] = pe_k
        print(f"    h={haz_str:<35s}  val_pred_dist_top3={np.bincount(pv_k.argmax(1), minlength=NUM_PROD).argsort()[-3:][::-1]}")

    del pm, pf, pdt, pdv, lptr, lpva
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# Ensemble across chains per hazard branch (F1-weighted on hazard F1)
# shape: (H, N, P)
prod_val_by_h = np.tensordot(cw_arr, chain_prod_val, axes=([0],[0]))  # (H, N_val, P)
prod_te_by_h  = np.tensordot(cw_arr, chain_prod_te,  axes=([0],[0]))  # (H, N_te,  P)

# =============================================================================
# 3. JOINT DECODING
#    score(h,p) = alpha*log P(h|x) + (1-alpha)*log P(p|x,h) + lam*log P(h,p) + compat
# =============================================================================
print("\n" + "="*70); print("JOINT DECODING"); print("="*70)

EPS = 1e-12
log_compat = np.where(compat > 0, 0.0, -1e9).astype(np.float32)  # hard mask

def joint_decode(haz_post, prod_post_by_h, alpha, lam):
    """
    haz_post:        (N, H)
    prod_post_by_h:  (H, N, P)
    returns h_idx, p_idx arrays
    """
    log_h = np.log(np.clip(haz_post, EPS, 1.0))                    # (N, H)
    log_p = np.log(np.clip(prod_post_by_h, EPS, 1.0))              # (H, N, P)
    log_p = np.transpose(log_p, (1, 0, 2))                         # (N, H, P)
    # joint score (N, H, P)
    score = (alpha * log_h[:, :, None]
             + (1 - alpha) * log_p
             + lam * log_bigram[None, :, :]
             + log_compat[None, :, :])
    flat = score.reshape(score.shape[0], -1)
    flat_idx = flat.argmax(axis=1)
    h_idx = flat_idx // NUM_PROD
    p_idx = flat_idx %  NUM_PROD
    return h_idx, p_idx

def st1_of(yh_str, ph_str, yp_str, pp_str):
    fh = f1_score(yh_str, ph_str, average='macro')
    m = (np.asarray(yh_str) == np.asarray(ph_str))
    fp = f1_score(np.asarray(yp_str)[m], np.asarray(pp_str)[m], average='macro') if m.any() else 0.0
    return (fh + fp) / 2, fh, fp

# Ultimate-only reference (compat-masked argmax)
uo_hv = ult_hv.argmax(1)
# Ultimate product = branch indexed by ult_hv argmax (this matches old ECC behaviour)
uo_pv_raw = prod_val_by_h[uo_hv, np.arange(len(valid_df)), :]
uo_pv_raw = uo_pv_raw * compat[uo_hv]
uo_pv_raw[uo_pv_raw.sum(1) == 0] = 1.0
uo_pv = uo_pv_raw.argmax(1)
uo_ht = ult_ht.argmax(1)
uo_pt_raw = prod_te_by_h[uo_ht, np.arange(len(test_df)), :]
uo_pt_raw = uo_pt_raw * compat[uo_ht]
uo_pt_raw[uo_pt_raw.sum(1) == 0] = 1.0
uo_pt = uo_pt_raw.argmax(1)

s_ult, fh_ult, fp_ult = st1_of(
    y_valid_hazard, hazard_enc.inverse_transform(uo_hv),
    y_valid_product, product_enc.inverse_transform(uo_pv))
print(f"  Ultimate reference: ST1={s_ult:.4f}  fh={fh_ult:.4f}  fp={fp_ult:.4f}")

# Grid search alpha, lam on val
print("\n  Grid search (alpha, lam) on val:")
best = {'st1': -1, 'alpha': None, 'lam': None}
for alpha in ALPHA_GRID:
    for lam in LAMBDA_GRID:
        h_idx, p_idx = joint_decode(ult_hv, prod_val_by_h, alpha, lam)
        s, fh, fp = st1_of(y_valid_hazard, hazard_enc.inverse_transform(h_idx),
                           y_valid_product, product_enc.inverse_transform(p_idx))
        marker = ""
        if s > best['st1']:
            best.update({'st1': s, 'alpha': alpha, 'lam': lam, 'fh': fh, 'fp': fp,
                         'h_idx': h_idx, 'p_idx': p_idx})
            marker = "  *"
        print(f"    alpha={alpha:.2f} lam={lam:.2f} -> ST1={s:.4f} fh={fh:.4f} fp={fp:.4f}{marker}")

print(f"\n  BEST: alpha={best['alpha']} lam={best['lam']} ST1={best['st1']:.4f} "
      f"(Δ vs ult {best['st1']-s_ult:+.4f})")

# =============================================================================
# 4. GATE + SUBMISSION
# =============================================================================
agree = ((best['h_idx'] == uo_hv) & (best['p_idx'] == uo_pv)).mean()
print(f"\n  Agreement with ultimate: {agree*100:.1f}%")

use = (best['st1'] >= GATE_VAL_FLOOR) and (agree >= GATE_MIN_AGREE) and (best['st1'] >= s_ult)
print(f"  Gate: val>={GATE_VAL_FLOOR} AND agree>={GATE_MIN_AGREE*100:.0f}% AND madrid>=ult")
print(f"  DECISION: {'SUBMIT MADRID' if use else 'FALLBACK TO ULTIMATE'}")

if use:
    ht_idx, pt_idx = joint_decode(ult_ht, prod_te_by_h, best['alpha'], best['lam'])
    fh_str = hazard_enc.inverse_transform(ht_idx)
    fp_str = product_enc.inverse_transform(pt_idx)
else:
    fh_str = hazard_enc.inverse_transform(uo_ht)
    fp_str = product_enc.inverse_transform(uo_pt)

sub = pd.DataFrame({'id': test_df['id'], 'hazard-category': fh_str, 'product-category': fp_str})
sub.to_csv('madrid.csv', index=False)
print(f"\nSaved: madrid.csv  shape={sub.shape}  content={'MADRID' if use else 'ULTIMATE FALLBACK'}")
try:
    import shutil
    shutil.copy('madrid.csv', '/content/drive/MyDrive/aretiiiii/madrid.csv')
    print("Saved to Google Drive")
except Exception:
    pass

print("\nHazard classification report (best madrid, val):")
print(classification_report(y_valid_hazard, hazard_enc.inverse_transform(best['h_idx'])))
print("\n" + "="*70); print("DONE — MADRID"); print("="*70)
