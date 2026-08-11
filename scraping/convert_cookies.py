"""
Convert hasil export cookies dari extension "Cookie-Editor" (format: list
of {"name":..., "value":..., ...}) menjadi format flat dict yang dipakai
twikit.load_cookies(), yaitu {"nama_cookie": "value", ...}.

Cara pakai:
    1. Login ke x.com di browser (akun sekunder).
    2. Install extension "Cookie-Editor" (Chrome Web Store / Firefox Add-ons).
    3. Buka x.com, klik icon Cookie-Editor -> Export -> Export as JSON.
    4. Paste hasilnya ke file "cookies_raw.json" di folder scraping/.
    5. Jalankan: python convert_cookies.py
       -> menghasilkan scraping/cookies.json (format siap pakai twikit)
"""

"""
Convert hasil export cookies ke format flat dict yang dipakai
twikit.load_cookies(), yaitu {"nama_cookie": "value", ...}.

Mendukung 2 sumber:
  A) Extension "Cookie-Editor" -> Export as JSON (TANPA enkripsi password!)
     format: list of {"name":..., "value":..., ...} ATAU dict langsung
  B) Extension "Get cookies.txt LOCALLY" -> download cookies.txt
     format: Netscape cookie file (plain text, tidak ada enkripsi)

Kalau Cookie-Editor versi kamu MEMAKSA pakai password saat export, JANGAN
dipakai — hasilnya terenkripsi dan script ini tidak bisa membacanya.
Pakai opsi B (Get cookies.txt LOCALLY) yang lebih simpel dan tanpa enkripsi.

Cara pakai:
    1. Login ke x.com di browser (akun sekunder).
    2. Export cookies pakai salah satu extension di atas.
    3a. Kalau JSON (Cookie-Editor): paste isinya ke scraping/cookies_raw.json
    3b. Kalau cookies.txt (Get cookies.txt LOCALLY): simpan file sebagai
        scraping/cookies_raw.txt
    4. Jalankan: python convert_cookies.py
       -> menghasilkan scraping/cookies.json (format siap pakai twikit)
"""

import json
from pathlib import Path

RAW_JSON_PATH = Path(__file__).parent / "cookies_raw.json"
RAW_TXT_PATH = Path(__file__).parent / "cookies_raw.txt"
OUT_PATH = Path(__file__).parent / "cookies.json"

# Cookie minimal yang WAJIB ada supaya twikit bisa autentikasi
REQUIRED_COOKIES = {"auth_token", "ct0"}


def parse_json_export(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        return {item["name"]: item["value"] for item in raw if "name" in item and "value" in item}
    raise ValueError("Format JSON tidak dikenali")


def parse_netscape_txt(text: str) -> dict:
    """Parse format cookies.txt (Netscape), dipakai extension
    'Get cookies.txt LOCALLY'. Format per baris (tab-separated):
    domain  include_subdomains  path  secure  expiry  name  value
    """
    flat = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 7:
            name, value = parts[5], parts[6]
            flat[name] = value
    return flat


def main():
    flat = {}

    if RAW_JSON_PATH.exists():
        with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
        try:
            raw = json.loads(content)
            flat = parse_json_export(raw)
        except json.JSONDecodeError:
            print("[ERROR] cookies_raw.json tidak bisa di-parse sebagai JSON.")
            print("Kemungkinan besar file ini masih TERENKRIPSI (password saat")
            print("export di Cookie-Editor). Matikan opsi enkripsi saat export,")
            print("atau pakai extension 'Get cookies.txt LOCALLY' sebagai gantinya.")
            return
    elif RAW_TXT_PATH.exists():
        with open(RAW_TXT_PATH, "r", encoding="utf-8") as f:
            flat = parse_netscape_txt(f.read())
    else:
        print(f"[ERROR] Tidak ditemukan {RAW_JSON_PATH} maupun {RAW_TXT_PATH}.")
        print("Export cookies dari browser dulu (lihat docstring di atas).")
        return

    missing = REQUIRED_COOKIES - set(flat.keys())
    if missing:
        print(f"[PERINGATAN] Cookie penting tidak ditemukan: {missing}")
        print("Pastikan kamu benar-benar sudah login di browser saat export,")
        print("dan export TIDAK dalam kondisi terenkripsi/password-protected.")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(flat, f, indent=2)

    print(f"OK -> {OUT_PATH} dibuat ({len(flat)} cookies).")
    print("Sekarang bisa jalankan: python scraper.py")


if __name__ == "__main__":
    main()