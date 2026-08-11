"""
STAPPIX - Fase 1: Text Processing
Menggabungkan raw_targeted.csv + raw_random.csv, membersihkan teks,
menghitung Amplification Score & Velocity Score (Persamaan 1 & 2 di
proposal), lalu membersihkan duplikat/missing value/format.

Cara pakai:
    pip install pandas emoji --break-system-packages
    python text_processing.py

Input:
    data/raw/raw_targeted.csv
    data/raw/raw_random.csv

Output:
    data/processed/cleaned_dataset.csv
    -> siap dipakai untuk tahap anotasi (belum ada label)
"""

import re
from tempfile import NamedTemporaryFile
from pathlib import Path

import pandas as pd

try:
    import emoji
except ImportError:
    emoji = None
    print("[WARN] package 'emoji' belum terinstall, emoji akan dihapus "
          "pakai regex fallback (kurang akurat). Install: pip install emoji --break-system-packages")

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
# Emoji fallback regex (dipakai kalau package 'emoji' tidak ada)
EMOJI_FALLBACK_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U0001F300-\U0001F5FF"
    "\U00002600-\U000026FF"
    "]+", flags=re.UNICODE
)
# Karakter yang dipertahankan: huruf, angka, spasi, dan '#' (untuk tagar)
SPECIAL_CHAR_PATTERN = re.compile(r"[^a-zA-Z0-9\s#]")
MULTISPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Pipeline cleaning sesuai checklist Fase 1:
    hapus URL/mention/emoji/karakter khusus, case folding, pertahankan tagar.

    Catatan desain: tagar (#kata) dipertahankan sebagai teks biasa (simbol '#'
    dibuang, kata di belakangnya tetap ada) karena masih mengandung informasi
    semantik. Kalau kamu mau tagar tetap dengan simbol '#'-nya (dianggap
    token terpisah oleh tokenizer), tinggal hapus baris strip('#') di bawah.
    """
    if not isinstance(text, str):
        return ""

    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)

    if emoji:
        text = emoji.replace_emoji(text, replace=" ")
    else:
        text = EMOJI_FALLBACK_PATTERN.sub(" ", text)

    text = SPECIAL_CHAR_PATTERN.sub(" ", text)
    text = text.replace("#", "")  # keep the word, drop the symbol
    text = text.lower()
    text = MULTISPACE_PATTERN.sub(" ", text).strip()

    return text


def compute_hours_since_post(row) -> float:
    """Hitung jam sejak posting = scraped_at - timestamp_post."""
    try:
        posted = pd.to_datetime(row["timestamp_utc"], utc=True)
        scraped = pd.to_datetime(row["scraped_at_utc"], utc=True)
        delta_hours = (scraped - posted).total_seconds() / 3600
        return max(delta_hours, 0)  # jaga-jaga kalau ada anomaly timestamp
    except Exception:
        return None


def min_max_scale(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return series * 0  # semua nilai sama -> normalisasi jadi 0
    return (series - lo) / (hi - lo)


def main():
    targeted_path = RAW_DIR / "raw_targeted.csv"
    random_path = RAW_DIR / "raw_random.csv"

    dfs = []
    for p in [targeted_path, random_path]:
        if p.exists():
            dfs.append(pd.read_csv(p, dtype={"post_id": str}))
        else:
            print(f"[WARN] {p} tidak ditemukan, dilewati.")

    if not dfs:
        print("[ERROR] Tidak ada data raw ditemukan. Jalankan scraper.py dulu.")
        return

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total baris gabungan (sebelum cleaning): {len(df)}")

    # 1) Hapus duplikat berdasarkan post_id
    before = len(df)
    df = df.drop_duplicates(subset="post_id", keep="first")
    print(f"Hapus duplikat post_id: {before - len(df)} baris dihapus")

    # 2) Hapus missing value pada kolom krusial
    before = len(df)
    df = df.dropna(subset=["text", "timestamp_utc", "scraped_at_utc"])
    df = df[df["text"].str.strip() != ""]
    print(f"Hapus missing value (text/timestamp): {before - len(df)} baris dihapus")

    # 3) Cleaning teks
    df["text"] = df["text"].apply(clean_text)
    before = len(df)
    df = df[df["text"].str.strip() != ""]  # buang yang jadi kosong setelah cleaning
    print(f"Hapus baris yang kosong setelah text cleaning: {before - len(df)} baris dihapus")

    # 4) Hitung fitur numerik
    for col in ["likes", "reposts", "replies"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    df["hours_since_post"] = df.apply(compute_hours_since_post, axis=1)
    before = len(df)
    df = df.dropna(subset=["hours_since_post"])
    print(f"Hapus baris dengan timestamp tidak valid: {before - len(df)} baris dihapus")

    # Persamaan 1: Amplification Score
    df["amplification_raw"] = (
        1 + df["likes"] + 2 * df["reposts"] + 1.5 * df["replies"]
    ).apply(lambda x: pd.NA if x <= 0 else __import__("math").log(x))
    df = df.dropna(subset=["amplification_raw"])
    df["amplification_norm"] = min_max_scale(df["amplification_raw"])

    # Persamaan 2: Velocity Score
    total_engagement = df["likes"] + df["reposts"] + df["replies"]
    df["velocity_raw"] = total_engagement / (df["hours_since_post"] + 1)
    df["velocity_norm"] = min_max_scale(df["velocity_raw"])

    # 5) Susun kolom output final
    out_cols = [
        "post_id", "text",
        "likes", "reposts", "replies", "timestamp_utc", "hours_since_post",
        "amplification_raw", "amplification_norm",
        "velocity_raw", "velocity_norm",
        "author_username", "scraped_at_utc",
    ]
    df = df[out_cols].reset_index(drop=True)

    out_path = PROCESSED_DIR / "cleaned_dataset.csv"
    with NamedTemporaryFile("w", delete=False, suffix=".csv", dir=PROCESSED_DIR, newline="") as tmp_file:
        tmp_path = Path(tmp_file.name)
        df.to_csv(tmp_path, index=False)

    fallback_path = PROCESSED_DIR / "cleaned_dataset_simplified.csv"
    try:
        tmp_path.replace(out_path)
        final_out_path = out_path
    except PermissionError:
        tmp_path.replace(fallback_path)
        final_out_path = fallback_path

    print(f"\nSelesai. {len(df)} baris tersimpan -> {final_out_path}")
    print("\nLangkah berikutnya: siapkan annotation_guideline lalu jalankan "
          "create_annotation_sheets.py untuk membagi dataset ke 3 anotator.")


if __name__ == "__main__":
    main()
