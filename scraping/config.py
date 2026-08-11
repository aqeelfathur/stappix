"""
Konfigurasi keyword untuk data collection STAPPIX.

TARGETED_TOPICS  -> ~40% data, topik yang SUDAH diklarifikasi Kominfo/Mafindo
                     sebagai hoax terkait kebijakan publik.
                     Isi ini dari hasil browsing manual ke:
                     - https://turnbackhoax.id (arsip Mafindo)
                     - https://cekfakta.kominfo.go.id
                     Cari kasus dengan tag kategori: bansos, subsidi, pajak,
                     pendidikan, kesehatan, energi, regulasi pemerintah.
                     Ambil frasa kunci / nama isu spesifik dari judul artikel
                     debunk-nya (bukan kata umum), supaya hasil scraping
                     benar-benar menyasar unggahan tentang hoax tsb.

RANDOM_KEYWORDS  -> ~60% data, keyword umum kebijakan publik (tanpa embel-embel
                     "hoax"), supaya distribusi mendekati kondisi nyata
                     (campuran hoax & non-hoax).
"""

# --- Diisi dari hasil riset manual turnbackhoax.id (per Agustus 2026) ---
# Catatan: 2 item hoax soal "Iran menyerang fasilitas AS/Arab Saudi" dari list
# awal kamu SENGAJA TIDAK dimasukkan karena itu isu geopolitik/militer, bukan
# kebijakan publik domestik (di luar scope: bansos/subsidi/pajak/pendidikan/
# kesehatan/energi/regulasi). Kalau memang mau dipakai juga, tinggal tambahkan
# manual ke list di bawah.
TARGETED_TOPICS = [
    "pajak AC kulkas 2027",
    "Samsat pemutihan pajak TikTok",
    "larangan toko kelontong dekat Kopdes Merah Putih",
    "Pemprov Jabar SPP SMA SMK negeri",
    "razia pajak jalan tol",
    "Komdigi balik nama HP bekas",
    "SPBU Samsat tilang pajak",
    "beli Pertalite tunjukkan STNK pajak",
    "MK alihkan dana MBG pendidikan",
    "pajak sepeda pemerintah",
    "Menteri Pigai belanja koperasi Merah Putih",
]

RANDOM_KEYWORDS = [
    "bansos",
    "subsidi BBM",
    "subsidi listrik",
    "pajak PPN",
    "kebijakan pendidikan",
    "BPJS Kesehatan",
    "subsidi pupuk",
    "regulasi pemerintah",
    "kebijakan energi",
    "bantuan sosial",
]

# Target jumlah post
TOTAL_TARGET = 20  # reduced for quick dry-run; restore to 10000 for full run
TARGETED_RATIO = 0.4   # 40% -> ~4000 post
RANDOM_RATIO = 0.6     # 60% -> ~6000 post

# Per-keyword override for random collection (set to 600 tweets per keyword)
RANDOM_PER_KEYWORD = 600

# Rate limiting (detik) — jaga jarak antar-request supaya tidak kena block
# NOTE: dinaikkan setelah percobaan awal kena 429 (rate limit) dengan cepat.
DELAY_BETWEEN_REQUESTS = (0.5, 1.0)   # shortened for dry-run
DELAY_BETWEEN_QUERIES = (0.5, 1.0)   # shortened for dry-run

# Berapa post diambil per satu query pencarian (sebelum pindah ke query berikutnya)
POSTS_PER_QUERY_BATCH = 50

# Bahasa & lokasi (opsional filter search X)
LANG = "id"