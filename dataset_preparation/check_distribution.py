"""
STAPPIX - Fase 2: Class Distribution
Cek distribusi kelas hoax/non-hoax, bandingkan dengan target proposal
(30-40% hoax, 60-70% non-hoax).

Cara pakai:
    python check_distribution.py

Input:
    data/processed/labeled_dataset.csv
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
TARGET_HOAX_MIN, TARGET_HOAX_MAX = 0.30, 0.40


def main():
    path = PROCESSED_DIR / "labeled_dataset.csv"
    if not path.exists():
        print(f"[ERROR] {path} tidak ditemukan. Jalankan merge_labels.py dulu.")
        return

    df = pd.read_csv(path, dtype={"post_id": str})
    df["label"] = df["label"].astype(int)

    total = len(df)
    counts = df["label"].value_counts().sort_index()
    pct = (counts / total * 100).round(2)

    print(f"Total data: {total}")
    print(f"Non-hoax (0): {counts.get(0, 0)} ({pct.get(0, 0)}%)")
    print(f"Hoax (1):     {counts.get(1, 0)} ({pct.get(1, 0)}%)")

    hoax_ratio = counts.get(1, 0) / total
    print(f"\nTarget proposal: {TARGET_HOAX_MIN*100:.0f}-{TARGET_HOAX_MAX*100:.0f}% hoax")
    if TARGET_HOAX_MIN <= hoax_ratio <= TARGET_HOAX_MAX:
        print(f"-> {hoax_ratio*100:.1f}% SESUAI target.")
    else:
        print(f"-> {hoax_ratio*100:.1f}% DI LUAR target. Tetap lanjut - "
              f"ketimpangan ditangani lewat class weight (langkah berikutnya), "
              f"bukan resampling data.")


if __name__ == "__main__":
    main()
