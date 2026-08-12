"""
STAPPIX - Fase 2: Stratified Split + Imbalance Handling
Split 70% train / 15% val / 15% test (stratified), lalu hitung class
weight dari training set (untuk Weighted Cross-Entropy Loss di Fase 3 -
sesuai proposal, bukan oversampling/undersampling).

Cara pakai:
    python stratified_split.py

Input:
    data/processed/labeled_dataset.csv

Output:
    data/processed/train.csv, val.csv, test.csv
    data/processed/class_weights.json
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


def compute_class_weights(labels: pd.Series) -> dict:
    counts = labels.value_counts().sort_index()
    n_classes = len(counts)
    total = len(labels)
    return {int(cls): round(total / (n_classes * cnt), 4) for cls, cnt in counts.items()}


def assert_no_leakage(train, val, test):
    overlap = (set(train["post_id"]) & set(val["post_id"])) | \
              (set(train["post_id"]) & set(test["post_id"])) | \
              (set(val["post_id"]) & set(test["post_id"]))
    if overlap:
        raise RuntimeError(f"DATA LEAKAGE terdeteksi! {len(overlap)} post_id tumpang tindih.")
    print("Validasi data leakage: AMAN")


def main():
    path = PROCESSED_DIR / "labeled_dataset.csv"
    if not path.exists():
        print(f"[ERROR] {path} tidak ditemukan.")
        return

    df = pd.read_csv(path, dtype={"post_id": str})
    df["label"] = df["label"].astype(int)

    train_val, test = train_test_split(
        df, test_size=TEST_RATIO, stratify=df["label"], random_state=RANDOM_SEED
    )
    val_size_relative = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train, val = train_test_split(
        train_val, test_size=val_size_relative, stratify=train_val["label"], random_state=RANDOM_SEED
    )

    print(f"Total: {len(df)} | Train: {len(train)} ({len(train)/len(df)*100:.1f}%) | "
          f"Val: {len(val)} ({len(val)/len(df)*100:.1f}%) | "
          f"Test: {len(test)} ({len(test)/len(df)*100:.1f}%)")

    assert_no_leakage(train, val, test)

    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)

    weights = compute_class_weights(train["label"])
    with open(PROCESSED_DIR / "class_weights.json", "w") as f:
        json.dump(weights, f, indent=2)

    print(f"\nClass weights (dari training set): {weights}")
    print(f"Output -> {PROCESSED_DIR}/train.csv, val.csv, test.csv, class_weights.json")


if __name__ == "__main__":
    main()
