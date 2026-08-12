"""
STAPPIX - Fase 3: Threshold Tuning
Cari threshold klasifikasi terbaik di validation set, prioritas Recall
kelas hoax (minimalkan false negative) - sesuai proposal.

Cara pakai:
    python threshold_tuning.py

Input:
    data/processed/val.pt
    classification_models/best_model_checkpoint/

Output:
    classification_models/threshold.json  (threshold terpilih + tabel sweep)
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CHECKPOINT_DIR = Path(__file__).parent / "best_model_checkpoint"
OUT_PATH = Path(__file__).parent / "threshold.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CANDIDATE_THRESHOLDS = np.arange(0.10, 0.55, 0.05)  # digeser rendah karena prioritas Recall


def get_val_probs():
    """Inferensi model best checkpoint di validation set -> P_hoax per baris."""
    data = torch.load(PROCESSED_DIR / "val.pt")
    ds = TensorDataset(data["input_ids"], data["attention_mask"], data["labels"])
    loader = DataLoader(ds, batch_size=32)

    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR).to(DEVICE)
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(label=1) = P_hoax
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def sweep_thresholds(probs: np.ndarray, labels: np.ndarray):
    rows = []
    for t in CANDIDATE_THRESHOLDS:
        preds = (probs >= t).astype(int)
        recall_hoax = recall_score(labels, preds, pos_label=1, zero_division=0)
        precision_hoax = precision_score(labels, preds, pos_label=1, zero_division=0)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        rows.append({
            "threshold": round(float(t), 2),
            "recall_hoax": round(recall_hoax, 4),
            "precision_hoax": round(precision_hoax, 4),
            "macro_f1": round(macro_f1, 4),
        })
    return rows


def choose_threshold(rows: list[dict]) -> dict:
    """
    Strategi pemilihan sesuai proposal: prioritaskan Recall kelas hoax,
    tapi tetap jaga Macro F1 tidak jatuh terlalu jauh dari yang terbaik
    (batas: Macro F1 >= 90% dari Macro F1 maksimum di seluruh threshold
    yang diuji). Di antara kandidat yang lolos syarat itu, pilih Recall
    tertinggi.
    """
    max_macro_f1 = max(r["macro_f1"] for r in rows)
    eligible = [r for r in rows if r["macro_f1"] >= 0.9 * max_macro_f1]
    if not eligible:
        eligible = rows
    best = max(eligible, key=lambda r: r["recall_hoax"])
    return best


def main():
    if not CHECKPOINT_DIR.exists():
        print(f"[ERROR] {CHECKPOINT_DIR} tidak ditemukan. Jalankan train_indobertweet.py dulu.")
        return

    probs, labels = get_val_probs()
    rows = sweep_thresholds(probs, labels)

    print(f"{'Threshold':<10}{'Recall(hoax)':<15}{'Precision(hoax)':<18}{'Macro F1':<10}")
    for r in rows:
        print(f"{r['threshold']:<10}{r['recall_hoax']:<15}{r['precision_hoax']:<18}{r['macro_f1']:<10}")

    best = choose_threshold(rows)
    print(f"\nThreshold terpilih: {best['threshold']} "
          f"(Recall hoax={best['recall_hoax']}, Macro F1={best['macro_f1']})")

    with open(OUT_PATH, "w") as f:
        json.dump({"chosen_threshold": best["threshold"], "sweep_table": rows}, f, indent=2)
    print(f"Disimpan -> {OUT_PATH}")


if __name__ == "__main__":
    main()
