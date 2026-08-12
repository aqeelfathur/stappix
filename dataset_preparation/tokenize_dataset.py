"""
STAPPIX - Fase 2: Tokenizer Loading + Text Tokenization
Load tokenizer IndoBERTweet, tokenisasi train/val/test jadi tensor .pt.

Cara pakai:
    pip install transformers torch --break-system-packages
    python tokenize_dataset.py

Input:
    data/processed/train.csv, val.csv, test.csv

Output:
    data/processed/train.pt, val.pt, test.pt
"""

from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_NAME = "indolem/indobertweet-base-uncased"
MAX_LENGTH = 128


def main():
    print(f"Loading tokenizer: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for split_name in ["train", "val", "test"]:
        csv_path = PROCESSED_DIR / f"{split_name}.csv"
        if not csv_path.exists():
            print(f"[ERROR] {csv_path} tidak ditemukan. Jalankan stratified_split.py dulu.")
            return

        df = pd.read_csv(csv_path, dtype={"post_id": str})
        df["label"] = df["label"].astype(int)

        encodings = tokenizer(
            df["text"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        result = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(df["label"].tolist()),
            "post_id": df["post_id"].tolist(),
        }
        out_path = PROCESSED_DIR / f"{split_name}.pt"
        torch.save(result, out_path)
        print(f"{split_name}: {len(df)} baris -> {out_path} "
              f"(input_ids shape: {result['input_ids'].shape})")

    print("\nSelesai. train.pt/val.pt/test.pt siap untuk Fase 3.")


if __name__ == "__main__":
    main()
