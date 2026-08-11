"""
STAPPIX - Fase 1: Agregasi Anotasi
Menggabungkan hasil 3 anotator -> majority voting -> hitung Fleiss' Kappa
-> gabungkan ke cleaned_dataset -> hasilkan labeled_dataset.csv final.

Cara pakai:
    python compute_kappa.py

Input:
    data/annotation/annotator_1.csv
    data/annotation/annotator_2.csv
    data/annotation/annotator_3.csv
    data/processed/cleaned_dataset.csv

Output:
    data/processed/labeled_dataset.csv
    (kolom tambahan: label, is_ambiguous, vote_hoax, vote_nonhoax)
"""

from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).parent
ANNOTATION_DIR = ROOT_DIR / "data" / "annotation"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

ANNOTATOR_FILES = ["annotator_1.csv", "annotator_2.csv", "annotator_3.csv"]


def fleiss_kappa(label_matrix: pd.DataFrame) -> float:
    """
    Hitung Fleiss' Kappa dari matrix (n_items x n_categories) berisi jumlah
    rater yang memilih tiap kategori per item.
    Rumus standar Fleiss (1971).
    """
    n_items, n_categories = label_matrix.shape
    n_raters = label_matrix.sum(axis=1).iloc[0]  # asumsi jumlah rater/item konstan

    # P_i: agreement per item
    p_i = ((label_matrix ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()

    # p_j: proporsi total tiap kategori
    p_j = label_matrix.sum(axis=0) / (n_items * n_raters)
    p_e_bar = (p_j ** 2).sum()

    if p_e_bar == 1:
        return 1.0  # hindari div by zero kalau semua sepakat total
    kappa = (p_bar - p_e_bar) / (1 - p_e_bar)
    return kappa


def main():
    dfs = {}
    for fname in ANNOTATOR_FILES:
        path = ANNOTATION_DIR / fname
        if not path.exists():
            print(f"[ERROR] {path} tidak ditemukan.")
            return
        df = pd.read_csv(path, dtype={"post_id": str})
        if df["label"].isna().any() or (df["label"].astype(str).str.strip() == "").any():
            n_missing = (df["label"].isna() | (df["label"].astype(str).str.strip() == "")).sum()
            print(f"[ERROR] {fname}: {n_missing} baris belum dilabeli. "
                  f"Selesaikan dulu semua label sebelum menjalankan script ini.")
            return
        df["label"] = df["label"].astype(int)
        dfs[fname] = df.set_index("post_id")["label"]

    # Gabung jadi 1 tabel: post_id x [label_annotator1, label_annotator2, label_annotator3]
    combined = pd.DataFrame(dfs)
    combined.columns = ["a1", "a2", "a3"]

    # Cek semua anotator melabeli post_id yang sama persis
    if combined.isna().any().any():
        print("[ERROR] Ada post_id yang tidak konsisten antar file anotator "
              "(tidak semua anotator melabeli baris yang sama). Cek ulang file.")
        return

    # Majority voting
    combined["vote_hoax"] = (combined[["a1", "a2", "a3"]] == 1).sum(axis=1)
    combined["vote_nonhoax"] = (combined[["a1", "a2", "a3"]] == 0).sum(axis=1)
    combined["label"] = (combined["vote_hoax"] >= 2).astype(int)
    combined["is_ambiguous"] = combined["vote_hoax"].isin([1, 2])  # split 2-1 di kedua arah

    # Fleiss' Kappa
    label_matrix = pd.DataFrame({
        0: combined["vote_nonhoax"],
        1: combined["vote_hoax"],
    })
    kappa = fleiss_kappa(label_matrix)

    print(f"Fleiss' Kappa: {kappa:.4f}")
    if kappa >= 0.60:
        print("-> Substantial agreement TERCAPAI (target >= 0.60). Lanjut.")
    else:
        print("-> Kappa DI BAWAH target 0.60. Sesuai checklist Fase 1: "
              "kalibrasi ulang annotation_guideline.md, lalu re-anotasi "
              "sampel dengan disagreement tinggi (lihat kolom is_ambiguous).")

    n_ambiguous = combined["is_ambiguous"].sum()
    print(f"Kasus ambigu (split 2-1): {n_ambiguous} dari {len(combined)} "
          f"({n_ambiguous/len(combined)*100:.1f}%)")

    # Gabungkan ke cleaned_dataset
    cleaned_path = PROCESSED_DIR / "cleaned_dataset.csv"
    cleaned = pd.read_csv(cleaned_path, dtype={"post_id": str}).set_index("post_id")
    final = cleaned.join(combined[["label", "is_ambiguous", "vote_hoax", "vote_nonhoax"]])

    final = final.reset_index()
    out_path = PROCESSED_DIR / "labeled_dataset.csv"
    final.to_csv(out_path, index=False)

    print(f"\nDistribusi label final:\n{final['label'].value_counts(normalize=True) * 100}")
    print(f"\nSelesai -> {out_path}")


if __name__ == "__main__":
    main()
