"""
STAPPIX - Fase 1: Persiapan Anotasi
Membuat 3 salinan dataset (satu per anotator) dengan kolom 'label' kosong
untuk diisi manual atau dipakai sebagai input auto-annotation Ollama.
Ketiga anotator melabeli DATASET YANG SAMA secara independen (bukan dibagi
per bagian) supaya bisa dihitung Fleiss' Kappa.

Cara pakai:
    python create_annotation_sheets.py

Input:
    data/processed/cleaned_dataset_simplified.csv
    (fallback: data/processed/cleaned_dataset.csv)

Output:
    data/annotation/annotator_1.csv
    data/annotation/annotator_2.csv
    data/annotation/annotator_3.csv
    (masing-masing berisi kolom: post_id, text, label)

Cara isi:
    Buka tiap file, isi kolom 'label' dengan 1 (hoax) atau 0 (non-hoax)
    untuk SETIAP baris. Kolom 'text' sudah berisi teks yang dibersihkan.
    Rujuk annotation_guideline.md untuk definisi & contoh kasus.
"""

from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
ANNOTATION_DIR = ROOT_DIR / "data" / "annotation"
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATOR_NAMES = ["annotator_1", "annotator_2", "annotator_3"]  # ganti sesuai nama tim: Ismail/Sebastian/Ananda dst
SOURCE_FILES = [
    PROCESSED_DIR / "cleaned_dataset_simplified.csv",
    PROCESSED_DIR / "cleaned_dataset.csv",
]


def load_source_frame() -> pd.DataFrame:
    for src in SOURCE_FILES:
        if src.exists():
            df = pd.read_csv(src, dtype={"post_id": str})
            if "text" in df.columns:
                return df[["post_id", "text"]].copy()
            if "text_clean" in df.columns:
                return df[["post_id", "text_clean"]].rename(columns={"text_clean": "text"}).copy()

    raise FileNotFoundError(
        "Tidak menemukan cleaned_dataset_simplified.csv atau cleaned_dataset.csv. "
        "Jalankan preprocessing/text_processing.py dulu."
    )


def main():
    try:
        sheet = load_source_frame()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return

    sheet["label"] = ""  # kosong, diisi manual: 0 (non-hoax) atau 1 (hoax)

    for name in ANNOTATOR_NAMES:
        out_path = ANNOTATION_DIR / f"{name}.csv"
        if out_path.exists():
            print(f"[SKIP] {out_path} sudah ada, tidak ditimpa "
                  f"(hapus manual dulu kalau memang mau reset).")
            continue
        sheet.to_csv(out_path, index=False)
        print(f"Dibuat: {out_path} ({len(sheet)} baris untuk dilabeli)")

    print("\nLangkah berikutnya: tiap anotator isi kolom 'label' secara "
          "independen, rujuk annotation_guideline.md. Setelah ketiganya "
          "selesai, jalankan compute_kappa.py.")


if __name__ == "__main__":
    main()
