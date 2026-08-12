"""
STAPPIX - Fase 3: Baseline Models
Bangun model pembanding TF-IDF + Random Forest dan TF-IDF + XGBoost.

Cara pakai:
    pip install scikit-learn xgboost joblib --break-system-packages
    python train_baselines.py

Input:
    data/processed/train.csv, val.csv

Output:
    baselines/tfidf_vectorizer.joblib
    baselines/random_forest.joblib
    baselines/xgboost.joblib
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
BASELINE_DIR = Path(__file__).parent / "baselines"
BASELINE_DIR.mkdir(exist_ok=True)

TFIDF_MAX_FEATURES = 5000
RANDOM_SEED = 42


def main():
    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "val.csv"
    if not train_path.exists() or not val_path.exists():
        print("[ERROR] train.csv/val.csv tidak ditemukan. Jalankan Fase 2 dulu.")
        return

    train = pd.read_csv(train_path, dtype={"post_id": str})
    val = pd.read_csv(val_path, dtype={"post_id": str})

    print("Fit TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    X_train = vectorizer.fit_transform(train["text"].fillna(""))
    X_val = vectorizer.transform(val["text"].fillna(""))
    y_train = train["label"].astype(int)
    y_val = val["label"].astype(int)

    joblib.dump(vectorizer, BASELINE_DIR / "tfidf_vectorizer.joblib")

    # class_weight='balanced' di baseline sklearn/xgboost -- konsisten dengan
    # pendekatan imbalance handling yang sama seperti model utama (bukan resampling)
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_val_pred = rf.predict(X_val)
    rf_macro_f1 = f1_score(y_val, rf_val_pred, average="macro")
    print(f"  Random Forest val Macro F1: {rf_macro_f1:.4f}")
    joblib.dump(rf, BASELINE_DIR / "random_forest.joblib")

    print("\nTraining XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=200, scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED, eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)
    xgb_val_pred = xgb.predict(X_val)
    xgb_macro_f1 = f1_score(y_val, xgb_val_pred, average="macro")
    print(f"  XGBoost val Macro F1: {xgb_macro_f1:.4f}")
    joblib.dump(xgb, BASELINE_DIR / "xgboost.joblib")

    print(f"\nSelesai. Model tersimpan di {BASELINE_DIR}/")


if __name__ == "__main__":
    main()
