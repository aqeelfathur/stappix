"""
STAPPIX - Fase 3: Model Evaluation
Evaluasi lengkap model utama (IndoBERTweet) vs baseline (TF-IDF+RF,
TF-IDF+XGBoost) di testing set: Accuracy, Precision, Recall, Macro F1,
PR-AUC, Confusion Matrix.

Cara pakai:
    python evaluate_models.py

Input:
    data/processed/test.csv, test.pt
    classification_models/best_model_checkpoint/
    classification_models/baselines/*.joblib
    classification_models/threshold.json

Output:
    classification_models/evaluation_report.json
    classification_models/model_comparison.csv
    (cetak juga ke terminal)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CHECKPOINT_DIR = Path(__file__).parent / "best_model_checkpoint"
BASELINE_DIR = Path(__file__).parent / "baselines"
THRESHOLD_PATH = Path(__file__).parent / "threshold.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def eval_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_hoax": round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "recall_hoax": round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),  # [[TN,FP],[FN,TP]]
    }


def eval_indobertweet(threshold: float) -> dict:
    data = torch.load(PROCESSED_DIR / "test.pt")
    ds = TensorDataset(data["input_ids"], data["attention_mask"], data["labels"])
    loader = DataLoader(ds, batch_size=32)

    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR).to(DEVICE)
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= threshold).astype(int)
    return eval_metrics(all_labels, preds, all_probs)


def eval_baseline(name: str, test_df: pd.DataFrame) -> dict:
    vectorizer = joblib.load(BASELINE_DIR / "tfidf_vectorizer.joblib")
    model = joblib.load(BASELINE_DIR / f"{name}.joblib")

    X_test = vectorizer.transform(test_df["text"].fillna(""))
    y_test = test_df["label"].astype(int).values
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return eval_metrics(y_test, preds, probs)


def main():
    test_csv = PROCESSED_DIR / "test.csv"
    if not test_csv.exists():
        print("[ERROR] test.csv tidak ditemukan.")
        return
    test_df = pd.read_csv(test_csv, dtype={"post_id": str})

    threshold = 0.5
    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH) as f:
            threshold = json.load(f)["chosen_threshold"]
        print(f"Pakai threshold hasil tuning: {threshold}")
    else:
        print("[WARN] threshold.json tidak ditemukan, pakai default 0.5. "
              "Jalankan threshold_tuning.py dulu untuk hasil optimal.")

    results = {}

    if CHECKPOINT_DIR.exists():
        print("\nEvaluasi IndoBERTweet (model utama)...")
        results["IndoBERTweet"] = eval_indobertweet(threshold)
    else:
        print(f"[WARN] {CHECKPOINT_DIR} tidak ditemukan, skip model utama.")

    for name, label in [("random_forest", "TF-IDF + Random Forest"),
                         ("xgboost", "TF-IDF + XGBoost")]:
        model_path = BASELINE_DIR / f"{name}.joblib"
        if model_path.exists():
            print(f"Evaluasi {label}...")
            results[label] = eval_baseline(name, test_df)
        else:
            print(f"[WARN] {model_path} tidak ditemukan, skip {label}.")

    if not results:
        print("[ERROR] Tidak ada model yang berhasil dievaluasi.")
        return

    print("\n=== TABEL PERBANDINGAN PERFORMA ===")
    comparison_rows = []
    for model_name, metrics in results.items():
        row = {"model": model_name, **{k: v for k, v in metrics.items() if k != "confusion_matrix"}}
        comparison_rows.append(row)
        print(f"\n{model_name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(Path(__file__).parent / "model_comparison.csv", index=False)

    with open(Path(__file__).parent / "evaluation_report.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDisimpan -> model_comparison.csv, evaluation_report.json")


if __name__ == "__main__":
    main()
