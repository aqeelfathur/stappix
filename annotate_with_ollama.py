"""
STAPPIX - Fase 1: Auto Annotation dengan Ollama
Mengisi kolom label pada annotator_1/2/3.csv menggunakan model lokal
qwen2.5:7b via Ollama API.

Cara pakai:
    python annotate_with_ollama.py

Input:
    data/annotation/annotator_1.csv
    data/annotation/annotator_2.csv
    data/annotation/annotator_3.csv

Output:
    file annotator yang sama, tetapi kolom label diisi otomatis.

Catatan:
    - Jalankan `ollama serve` jika API lokal belum aktif.
    - Pastikan model sudah terunduh: `ollama pull qwen2.5:7b`.
"""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

ROOT_DIR = Path(__file__).parent
ANNOTATION_DIR = ROOT_DIR / "data" / "annotation"
ANNOTATOR_FILES = ["annotator_1.csv", "annotator_2.csv", "annotator_3.csv"]
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = (
    "You are labeling Indonesian social media posts for the STAPPIX project. "
    "Return JSON only with keys label and reason. "
    "label must be 1 for hoax and 0 for non-hoax. "
    "Use label 1 if the post contains a false factual claim, misleading context, "
    "or an unannounced policy presented as official. Use label 0 if it is an "
    "opinion, satire, genuine question, or a claim consistent with verified sources."
)


def build_prompt(text: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Annotation guideline summary:\n"
        "- HOAX (1): false factual claim, misleading context, or fake policy claim.\n"
        "- NON-HOAX (0): opinion, satire, genuine question, or verified claim.\n"
        "- If unsure, choose the best label based on the strongest evidence in the post.\n\n"
        "Return this exact JSON shape only:\n"
        '{"label": 0, "reason": "short explanation"}\n\n'
        f"Post:\n{text}"
    )


def parse_json_payload(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def call_ollama(model: str, prompt: str, temperature: float, seed: int) -> tuple[int, str, str]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "seed": seed,
        },
    }
    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=600) as response:
        body = json.loads(response.read().decode("utf-8"))

    raw_response = body.get("response", "")
    parsed = parse_json_payload(raw_response)
    label = int(parsed["label"])
    if label not in (0, 1):
        raise ValueError(f"Label tidak valid: {label}")

    reason = str(parsed.get("reason", "")).strip()
    return label, reason, raw_response.strip()


def annotate_frame(df: pd.DataFrame, model: str, temperature: float, seed: int) -> pd.DataFrame:
    if "post_id" not in df.columns:
        raise ValueError("Kolom post_id tidak ditemukan.")

    if "text" not in df.columns:
        if "text_clean" in df.columns:
            df = df.rename(columns={"text_clean": "text"})
        else:
            raise ValueError("Kolom text tidak ditemukan.")

    if "label" not in df.columns:
        df["label"] = ""

    labels = []
    reasons = []

    for _, row in df.iterrows():
        current_label = str(row.get("label", "")).strip()
        if current_label in {"0", "1"}:
            labels.append(int(current_label))
            reasons.append("existing label")
            continue

        text = str(row["text"]).strip()
        label, reason, _ = call_ollama(model, build_prompt(text), temperature, seed)
        labels.append(label)
        reasons.append(reason)

    annotated = df.copy()
    annotated["label"] = labels
    annotated["auto_reason"] = reasons
    return annotated


def write_csv_safely(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, suffix=".csv", dir=out_path.parent, newline="") as tmp_file:
        tmp_path = Path(tmp_file.name)
        df.to_csv(tmp_path, index=False)

    try:
        tmp_path.replace(out_path)
        return out_path
    except PermissionError:
        fallback_path = out_path.with_name(f"{out_path.stem}_ollama.csv")
        tmp_path.replace(fallback_path)
        return fallback_path


def main():
    parser = argparse.ArgumentParser(description="Auto-annotate STAPPIX sheets with Ollama.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0, help="Only annotate the first N rows of each file.")
    parser.add_argument("--temperature-base", type=float, default=0.0)
    args = parser.parse_args()

    configs = [
        ("annotator_1.csv", args.temperature_base + 0.0, 101),
        ("annotator_2.csv", args.temperature_base + 0.2, 202),
        ("annotator_3.csv", args.temperature_base + 0.4, 303),
    ]

    for fname, temperature, seed in configs:
        path = ANNOTATION_DIR / fname
        if not path.exists():
            print(f"[ERROR] {path} tidak ditemukan. Jalankan create_annotation_sheets.py dulu.")
            return

        df = pd.read_csv(path, dtype={"post_id": str})
        if args.limit > 0:
            df = df.head(args.limit).copy()

        try:
            annotated = annotate_frame(df, args.model, temperature, seed)
        except (HTTPError, URLError) as exc:
            print(f"[ERROR] Tidak bisa menghubungi Ollama API di {OLLAMA_URL}: {exc}")
            return

        out_path = write_csv_safely(annotated, path)
        print(f"[OK] {fname} -> {out_path} ({len(annotated)} baris, model={args.model}, temp={temperature}, seed={seed})")


if __name__ == "__main__":
    main()