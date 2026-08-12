"""
STAPPIX - Gabungkan cleaned_dataset_simplified.csv + labeled_dataset.csv
Join berdasarkan post_id, ambil kolom 'label' dari labeled_dataset.

Cara pakai:
    python merge_labels.py

Input:
    data/processed/cleaned_dataset_simplified.csv
    data/annotation/labeled_dataset.csv   (cukup kolom post_id, text, label)

Output:
    data/processed/labeled_dataset.csv    (cleaned_dataset_simplified + kolom label)
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
ANNOTATION_DIR = Path(__file__).parent.parent / "data" / "annotation"

CLEANED_PATH = PROCESSED_DIR / "cleaned_dataset_simplified.csv"
LABELS_PATH = ANNOTATION_DIR / "labeled_dataset.csv"
OUT_PATH = PROCESSED_DIR / "labeled_dataset.csv"


def main():
    if not CLEANED_PATH.exists():
        print(f"[ERROR] {CLEANED_PATH} tidak ditemukan.")
        return
    if not LABELS_PATH.exists():
        print(f"[ERROR] {LABELS_PATH} tidak ditemukan.")
        return

    cleaned = pd.read_csv(CLEANED_PATH, dtype={"post_id": str})
    labels = pd.read_csv(LABELS_PATH, dtype={"post_id": str})[["post_id", "label"]]

    merged = cleaned.merge(labels, on="post_id", how="inner")

    n_unmatched = len(cleaned) - len(merged)
    if n_unmatched > 0:
        print(f"[WARN] {n_unmatched} post_id di cleaned_dataset_simplified "
              f"tidak ketemu pasangannya di labeled_dataset (dibuang).")

    merged["label"] = pd.to_numeric(merged["label"], errors="coerce").astype("Int64")
    merged.to_csv(OUT_PATH, index=False)

    print(f"Selesai. {len(merged)} baris -> {OUT_PATH}")
    print(f"Distribusi label:\n{merged['label'].value_counts()}")


if __name__ == "__main__":
    main()
