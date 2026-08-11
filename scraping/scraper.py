"""
STAPPIX - Fase 1: Data Collection
Scraping unggahan X terkait kebijakan publik (targeted + random collection).

Cara pakai:
    1. pip install git+https://github.com/PawiX25/twifork.git --break-system-packages
       pip install -r requirements.txt --break-system-packages
    2. Copy .env.example -> .env, isi kredensial akun X
    3. Isi TARGETED_TOPICS di config.py (hasil riset dari turnbackhoax.id / cekfakta.kominfo.go.id)
    4. python scraper.py

Output:
    data/raw/raw_targeted.csv
    data/raw/raw_random.csv
    (data disimpan INCREMENTAL per query, bukan di akhir — supaya kalau
     scraping terhenti di tengah, data yang sudah didapat tidak hilang)
"""

import asyncio
import csv
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from twikit import Client

import config

load_dotenv()

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
COOKIES_FILE = Path(__file__).parent / "cookies.json"

CSV_FIELDS = [
    "post_id", "query", "collection_type", "text",
    "likes", "reposts", "replies", "timestamp_utc",
    "author_username", "scraped_at_utc",
]


RATE_LIMIT_WAIT_SECONDS = 16 * 60  # X biasanya reset limit search per 15 menit
MAX_RETRIES_ON_429 = 3


def load_existing(path: Path) -> tuple[set, dict]:
    """Baca CSV yang sudah ada -> (set post_id yang sudah tersimpan,
    dict {query: jumlah post per query}) supaya run ulang bisa skip
    keyword yang sudah selesai dan tidak menyimpan post_id duplikat."""
    seen_ids = set()
    per_query_count = {}
    if not path.exists():
        return seen_ids, per_query_count
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seen_ids.add(row["post_id"])
            per_query_count[row["query"]] = per_query_count.get(row["query"], 0) + 1
    return seen_ids, per_query_count


def init_csv(path: Path):
    """Create CSV with header if it doesn't exist yet."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()


async def call_with_retry(coro_fn, *args, **kwargs):
    """Run an async call, retrying on rate-limit errors with a long backoff."""
    for attempt in range(1, MAX_RETRIES_ON_429 + 1):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "Rate limit" in str(e):
                if attempt == MAX_RETRIES_ON_429:
                    raise
                print(
                    f"  [RATE LIMIT] kena limit, tunggu {RATE_LIMIT_WAIT_SECONDS // 60} menit "
                    f"(percobaan {attempt}/{MAX_RETRIES_ON_429})..."
                )
                await asyncio.sleep(RATE_LIMIT_WAIT_SECONDS)
            else:
                raise


def append_rows(path: Path, rows: list[dict]):
    """Simpan segera ke disk (antisipasi post dihapus / akun disuspend)."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        for row in rows:
            writer.writerow(row)


async def login(client: Client):
    """
    PENTING (per 2026): X sudah mematikan login flow programatik
    (username+password lewat script tidak bisa lagi, bahkan di twifork).
    Satu-satunya cara: ambil cookies dari sesi browser yang sudah login
    manual, simpan ke scraping/cookies.json, lalu load di sini.

    Cara ambil cookies:
    1. Login ke x.com pakai browser biasa (akun sekunder kamu).
    2. Install extension "Cookie-Editor" (Chrome/Firefox).
    3. Buka x.com, klik extension -> Export -> Export as JSON.
    4. Simpan hasil export sebagai scraping/cookies.json
       (format: list of {"name":..., "value":..., ...} ATAU dict
       {"name": "value", ...} - lihat catatan di README).
    """
    if not COOKIES_FILE.exists():
        raise RuntimeError(
            f"Cookies belum ada di {COOKIES_FILE}.\n"
            "Login manual di browser -> export cookies pakai extension "
            "'Cookie-Editor' -> simpan sebagai scraping/cookies.json.\n"
            "Lihat README.md bagian 'Login via cookies' untuk detail lengkap."
        )
    # Try to load JSON cookie export and pass a dict to the client when possible.
    try:
        with open(COOKIES_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        # Fall back to passing the path to the client loader
        client.load_cookies(str(COOKIES_FILE))
        print("Login pakai cookies (passed file path to client.load_cookies).")
        return

    # Normalize list-of-cookie dicts into {name: value}
    if isinstance(raw, list):
        cookies = {c.get("name"): c.get("value") for c in raw if "name" in c}
    elif isinstance(raw, dict):
        cookies = raw
    else:
        cookies = {}

    try:
        client.load_cookies(cookies)
        print("Login pakai cookies tersimpan (loaded dict).")
    except TypeError:
        # Some client implementations accept a path only
        client.load_cookies(str(COOKIES_FILE))
        print("Login pakai cookies (fallback: passed file path to client.load_cookies).")


async def scrape_query(client: Client, query: str, collection_type: str, limit: int,
                        seen_ids: set) -> list[dict]:
    """Scrape satu keyword/topik, kembalikan list of dict siap ditulis ke CSV.
    Otomatis retry dengan jeda panjang kalau kena rate limit 429, dan skip
    post yang post_id-nya sudah pernah tersimpan sebelumnya (dedupe)."""
    rows = []
    try:
        results = await call_with_retry(client.search_tweet, query, product="Latest")
    except Exception as e:
        print(f"  [WARN] gagal search '{query}': {e}")
        return rows

    collected = 0
    while results and collected < limit:
        for tweet in results:
            if collected >= limit:
                break
            try:
                post_id = str(tweet.id)
                if post_id in seen_ids:
                    continue  # sudah pernah discrape sebelumnya, skip
                # Defensive attribute access because tweet objects may vary
                likes = getattr(tweet, "favorite_count", None)
                reposts = getattr(tweet, "retweet_count", None)
                replies = getattr(tweet, "reply_count", None)
                created_at = getattr(tweet, "created_at", None)
                if hasattr(created_at, "isoformat"):
                    timestamp = created_at.isoformat()
                else:
                    timestamp = str(created_at) if created_at is not None else ""
                author = getattr(getattr(tweet, "user", None), "screen_name", "")
                rows.append({
                    "post_id": post_id,
                    "query": query,
                    "collection_type": collection_type,
                    "text": getattr(tweet, "text", ""),
                    "likes": likes if likes is not None else 0,
                    "reposts": reposts if reposts is not None else 0,
                    "replies": replies if replies is not None else 0,
                    "timestamp_utc": timestamp,
                    "author_username": author,
                    "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
                })
                seen_ids.add(post_id)
                collected += 1
            except Exception as e:
                print(f"  [WARN] skip 1 tweet, error parsing: {e}")

        if collected >= limit:
            break

        # jeda antar-request halaman berikutnya
        await asyncio.sleep(random.uniform(*config.DELAY_BETWEEN_REQUESTS))
        try:
            results = await call_with_retry(results.next)
        except Exception as e:
            print(f"  [WARN] tidak bisa ambil halaman berikutnya: {e}")
            break

    return rows


async def run_collection(client: Client, keywords: list[str], collection_type: str,
                          total_target: int, out_path: Path):
    init_csv(out_path)
    seen_ids, per_query_count = load_existing(out_path)
    per_keyword_limit = max(1, total_target // max(1, len(keywords)))
    print(f"\n== {collection_type.upper()} COLLECTION == "
          f"target total {total_target}, {len(keywords)} keyword, "
          f"~{per_keyword_limit} post/keyword")

    total_collected = 0
    for kw in tqdm(keywords, desc=collection_type):
        already = per_query_count.get(kw, 0)
        if already >= per_keyword_limit:
            print(f"  '{kw}': sudah {already} post (>= target), SKIP")
            continue
        remaining = per_keyword_limit - already
        rows = await scrape_query(client, kw, collection_type, remaining, seen_ids)
        append_rows(out_path, rows)
        total_collected += len(rows)
        print(f"  '{kw}': +{len(rows)} post baru (total {already + len(rows)})")
        await asyncio.sleep(random.uniform(*config.DELAY_BETWEEN_QUERIES))

    print(f"Total {collection_type} (run ini): {total_collected} post baru -> {out_path}")


async def main():
    if not config.TARGETED_TOPICS:
        print("[PERINGATAN] config.TARGETED_TOPICS masih kosong!")
        print("Isi dulu daftar topik hoax dari turnbackhoax.id / cekfakta.kominfo.go.id")
        print("sebelum menjalankan targeted collection.\n")

    client = Client(language=config.LANG if hasattr(config, "LANG") else "id")
    await login(client)

    if config.TARGETED_TOPICS:
        await run_collection(
            client, config.TARGETED_TOPICS, "targeted",
            int(config.TOTAL_TARGET * config.TARGETED_RATIO),
            RAW_DIR / "raw_targeted.csv",
        )

    # For random collection we want a fixed per-keyword target (see config.RANDOM_PER_KEYWORD)
    await run_collection(
        client, config.RANDOM_KEYWORDS, "random",
        int(config.RANDOM_PER_KEYWORD * max(1, len(config.RANDOM_KEYWORDS))),
        RAW_DIR / "raw_random.csv",
    )

    print("\nSelesai. Cek data/raw/ untuk hasil scraping.")


if __name__ == "__main__":
    asyncio.run(main())