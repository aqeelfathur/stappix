# STAPPIX — Fase 1: Data Collection

## Setup (sekali saja)

```bash
cd scraping
pip install -r requirements.txt --break-system-packages
pip install git+https://github.com/PawiX25/twifork.git --break-system-packages
cp .env.example .env
# lalu edit .env, isi X_USERNAME / X_EMAIL / X_PASSWORD dengan akun X kamu
```

> Saran: pakai akun X "sekunder" (bukan akun utama kamu), karena scraping
> otomatis punya risiko kena rate limit atau flag dari X.

## Sebelum menjalankan: isi daftar topik hoax

Buka `config.py`, isi `TARGETED_TOPICS` dengan judul/isu spesifik hasil
riset manual dari:
- https://turnbackhoax.id (arsip pemeriksaan fakta Mafindo)
- https://cekfakta.kominfo.go.id

Fokus kategori: bansos, subsidi, pajak, pendidikan, kesehatan, energi,
regulasi pemerintah. Ambil ~15-25 topik/frasa spesifik supaya hasil
scraping benar-benar menyasar unggahan yang membahas hoax tsb (bukan cuma
kata umum yang bisa mengarah ke non-hoax).

`RANDOM_KEYWORDS` sudah ada default-nya, boleh ditambah/disesuaikan.

## Menjalankan

```bash
python scraper.py
```

Data tersimpan **incremental** per keyword ke:
- `data/raw/raw_targeted.csv`
- `data/raw/raw_random.csv`

Kalau proses terhenti di tengah jalan (misal kena rate limit), tinggal
jalankan ulang — data yang sudah masuk CSV sebelumnya tidak hilang
(tapi scraper saat ini belum skip keyword yang sudah selesai, jadi untuk
resume yang rapi, hapus keyword yang sudah kelar dari `config.py` sebelum
run ulang, atau tambahkan pengecekan duplikat `post_id` nanti di tahap
cleaning).

## Yang perlu diperhatikan

- **Rate limiting**: sudah diatur delay 3-7 detik antar-request dan
  15-30 detik antar-keyword di `config.py`. Jangan diperkecil drastis —
  risiko akun kena block naik.
- **Legalitas**: hanya untuk penelitian non-komersial, hanya data publik.
- **Ketidakstabilan library**: `twikit`/`twifork` reverse-engineer API
  internal X, bisa rusak lagi sewaktu-waktu kalau X ubah struktur. Kalau
  scraper tiba-tiba error total, cek dulu apakah ada versi terbaru
  `twifork` atau ganti ke `twscrape`/`Tweety` sebagai cadangan.
- Kolom yang disimpan sesuai kebutuhan Fase 1: `text`, `likes`,
  `reposts`, `replies`, `timestamp_utc` — cukup untuk hitung
  Amplification Score & Velocity Score di tahap berikutnya.

## Langkah setelah data terkumpul

Lanjut ke tahap **Text Processing** (masih Fase 1): cleaning teks +
hitung Amplification_norm & Velocity_norm. Aku buatkan script-nya
begitu data raw sudah ada / kamu siap lanjut.