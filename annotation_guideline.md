# Annotation Guideline — STAPPIX

## Tujuan
Melabeli setiap unggahan sebagai **hoax (1)** atau **non-hoax (0)** terkait
isu kebijakan publik (bansos, subsidi, pajak, pendidikan, kesehatan, energi,
regulasi pemerintah).

## Definisi Operasional

**HOAX (label = 1)** jika unggahan:
- Berisi klaim faktual yang **terbukti salah** menurut sumber pemeriksa fakta
  (Kominfo/Mafindo/Cek Fakta) atau sumber resmi pemerintah terkait.
- Berisi **misleading context**: fakta yang benar tapi disajikan dengan
  konteks yang menyesatkan (mis. foto/statistik asli tapi dinarasikan
  seolah-olah tentang kebijakan yang berbeda).
- Berisi klaim kebijakan yang **belum pernah ada/diumumkan resmi**, tapi
  dinarasikan seolah sudah menjadi kebijakan resmi.

**NON-HOAX (label = 0)** jika unggahan:
- Opini, kritik, atau sindiran terhadap kebijakan **tanpa klaim faktual palsu**.
- Satire/parodi yang jelas tidak dimaksudkan sebagai informasi faktual.
- Berita/klaim yang sesuai dengan sumber resmi/terverifikasi.
- Pertanyaan genuine tentang kebijakan (bukan pernyataan klaim).

## Penanganan Kasus Ambigu

- Jika ragu antara opini keras vs klaim faktual palsu → cek apakah ada
  **klaim spesifik yang bisa diverifikasi benar/salah** (angka, tanggal,
  nama kebijakan). Kalau ada dan itu salah → HOAX. Kalau murni opini
  tanpa klaim terukur → NON-HOAX.
- Jika konten adalah **repost/quote tweet** dari sumber hoax tanpa
  komentar tambahan yang mengklarifikasi → tetap HOAX (karena tetap
  menyebarkan klaim yang sama).
- Jika tidak yakin sama sekali setelah mempertimbangkan poin di atas →
  beri label sesuai penilaian terbaik kamu. Sistem majority voting +
  flag ambigu (2-1 split) akan menangkap kasus yang benar-benar tidak
  jelas ini secara otomatis di tahap agregasi.

## Contoh

**Positif (HOAX):**
> "bansos cair rp5jt cek link ini buat daftar" (link phishing, tidak ada
> program bansos dengan nominal/mekanisme tsb)
→ Klaim spesifik (nominal + mekanisme) yang tidak sesuai fakta resmi.

**Negatif (NON-HOAX):**
> "kebijakan PPN 12% ini menurut saya berat buat UMKM, semoga ditinjau
> ulang"
→ Opini terhadap kebijakan yang memang ada, tanpa klaim faktual palsu.

## Proses Anotasi

1. 3 anotator bekerja **independen**, tidak melihat hasil anotator lain.
2. Label akhir = majority voting (≥2 dari 3 suara sepakat).
3. Kasus dengan split 2-1 ditandai `is_ambiguous = True` untuk transparansi
   laporan (tetap dipakai untuk training, tapi dicatat sebagai catatan
   metodologis).
4. Fleiss' Kappa dihitung dari 3 set label untuk mengukur konsistensi
   antar-anotator. Target ≥ 0.60 (substantial agreement).
5. Jika Kappa < 0.60: kalibrasi ulang guideline ini (tambahkan contoh
   kasus yang jadi sumber ketidaksepakatan), lalu re-anotasi sampel
   dengan disagreement tinggi.

<!-- TODO: tambahkan contoh kasus tambahan dari hasil kalibrasi tim
     (Ismail sebagai PIC dokumentasi) begitu proses anotasi dimulai. -->
