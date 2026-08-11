"""
config.py -- Parameter konfigurasi dan system prompt untuk RAG engine.
"""
import os

# Model & Storage Paths
MODEL_PATH = os.getenv("MODEL_PATH", "./models/Qwen3.5-2B-UD-Q4_K_XL.gguf")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
PARENT_STORE_DIR = os.getenv("PARENT_STORE_DIR", "./parent_docstore")

# Execution & Context Parameters
N_CTX = int(os.getenv("N_CTX", 8192))
N_THREADS = int(os.getenv("N_THREADS", os.cpu_count() or 4))

# Cache Parameters
CACHE_THRESHOLD = 0.85
MAX_CACHE_ITEMS = 100
ALL_DOCS_CACHE_KEY = "__all__"

# Chunking Strategy (Parent-Child)
PARENT_CHUNK_SIZE = 800
PARENT_CHUNK_OVERLAP = 80
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 50
ROWS_PER_TABLE_CHUNK = 8

# Retrieval Parameters
#
# CATATAN TUNING (evaluasi RAGAS Agustus 2026, dikonfirmasi via debug_retrieval_scores.py):
# soal Force Majeure gagal retrieval karena chunk yang benar ada di PERINGKAT #15-16
# dari 345 total child chunk (cosine score 0.56, jauh di atas DENSE_SCORE_THRESHOLD=0.3
# -- threshold BUKAN masalahnya). RETRIEVER_K dinaikkan ke 25 supaya chunk itu ikut
# masuk kandidat RRF fusion. Setelah itu, tabel ranking RRF gabungan (dense+BM25)
# menunjukkan chunk target ada tepat di PERINGKAT #9 dari 35 kandidat gabungan,
# skor 0.02727 -- cuma selisih tipis (0.002) dari cutoff TOP_N_PARENTS=6 (skor
# peringkat #6 = 0.02927). TOP_N_PARENTS dinaikkan ke 10 supaya chunk ini lolos.
# Chunk-chunk yang mengalahkannya semua soal klausul "tanggung jawab" generik
# (penyalahgunaan kartu, perubahan data) -- model embedding menyamakan makna
# "tanggung jawab X" apapun topiknya, jadi klausul spesifik seperti Force Majeure
# kalah bersaing dgn klausul liability yang lebih sering muncul di dokumen.
RETRIEVER_K = 25
DENSE_SCORE_THRESHOLD = float(os.getenv("DENSE_SCORE_THRESHOLD", 0.30))
TOP_N_PARENTS = int(os.getenv("TOP_N_PARENTS", 10))  # sebelumnya 6 -- lihat catatan di atas
RRF_K = 60

# Multi-turn Context Settings
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", 3))

# Domain Conflict Terms
CRITICAL_TERM_PAIRS = [
    ("debit", "kredit"),
    ("tabungan", "deposito"),
    ("giro", "tabungan"),
]

# System Prompts
#
# CATATAN REVISI v2 (evaluasi RAGAS Agustus 2026 -- tuning kedua):
# Perbaikan berdasarkan analisis per-soal dari hasil evaluasi gpt-4o-mini:
# - Label [SUMBER INFORMASI] diganti [DOKUMEN] di dalam teks prompt agar tidak
#   "bocor" ke jawaban model (model 2B cenderung meniru format label dari context).
# - Ditambah larangan frasa pembuka template ("Berdasarkan dokumen...", "Menurut
#   [DOKUMEN]...") yang menurunkan skor answer_relevancy secara signifikan.
# - Ditambah aturan verbatim untuk tindakan/prosedur Bank: kata kerja tindakan
#   (memblokir, mengalihkan, menyetorkan, dll.) harus disalin persis dari dokumen,
#   bukan diparafrase -- penyebab faithfulness=0.000 pada soal nasabah meninggal.
# - Ditambah aturan verbatim khusus rumus/mekanisme perhitungan: tidak boleh
#   diringkas atau disederhanakan -- penyebab faithfulness=0.333 pada soal bunga.
# - Ditambah larangan eksplisit menyebut kondisi/syarat perpanjangan yang tidak
#   tertulis persis di dokumen -- penyebab faithfulness=0.600 pada soal pengaduan.
# - Revisi v1: instruksi salam DIHAPUS, aturan buang potongan tidak relevan diperkuat,
#   aturan panjang jawaban ditambahkan.
SYSTEM_PROMPT_RAG_STRICT = """Kamu adalah asisten AI{client_info} yang santai dan ramah. Selalu menjawab dalam Bahasa Indonesia, singkat dan akurat.

[ATURAN KONTEN]
1. Jawab HANYA berdasarkan [DOKUMEN] yang diberikan.
2. [DOKUMEN] berisi beberapa potongan teks. Sebelum menjawab, pilih HANYA potongan yang benar-benar menjawab pertanyaan user. Potongan lain yang tidak relevan harus DIBUANG SEPENUHNYA -- jangan disebutkan, dirangkum, atau dijadikan poin tambahan dalam jawaban. JANGAN menggabungkan detail dari beberapa potongan berbeda menjadi satu klaim baru -- setiap detail spesifik (angka, syarat, kondisi) harus benar-benar tertulis persis di SATU potongan yang sama.
3. JANGAN pernah menciptakan istilah, angka, atau fakta baru yang tidak tertulis di [DOKUMEN]. JANGAN menambahkan kondisi, syarat, atau prosedur kecuali tertulis kata per kata di potongan yang kamu pilih. Kalau ragu, JANGAN ditambahkan -- cukup jawab inti pertanyaannya dan berhenti di situ.
4. KHUSUS tindakan/prosedur (misalnya tindakan Bank, kewajiban nasabah): salin kata kerja dan subjeknya PERSIS seperti tertulis di [DOKUMEN]. JANGAN parafrase -- misalnya jika dokumen bilang "Bank berhak memblokir Rekening", jangan ubah menjadi "Bank dapat menyetorkan saldo".
5. KHUSUS rumus dan mekanisme perhitungan: kutip PERSIS seperti tertulis di [DOKUMEN]. JANGAN sederhanakan, ringkas, atau ubah komponen perhitungannya.
6. KHUSUS angka (hari/bulan/tahun, nominal, persentase): salin PERSIS seperti tertulis. JANGAN mengira-ngira, membulatkan, atau menggabungkan angka dari bagian berbeda.
7. JANGAN memulai jawaban dengan frasa template seperti "Berdasarkan dokumen...", "Menurut [DOKUMEN]...", atau frasa serupa. Langsung jawab pertanyaannya. JANGAN pernah menyebut kata "[DOKUMEN]" di dalam jawaban.
8. Jika informasi tidak ditemukan di [DOKUMEN], katakan terus terang bahwa jawabannya tidak ada di dokumen.
9. Jawab langsung ke inti pertanyaan dalam 1-2 kalimat pertama. Tambahkan detail pendukung HANYA jika pertanyaannya memang butuh rincian bertahap (daftar syarat atau prosedur) -- jangan buat daftar panjang untuk pertanyaan sederhana.
10. Jika ada riwayat percakapan sebelumnya, gunakan untuk memahami konteks pertanyaan lanjutan."""

SYSTEM_PROMPT_RAG_FLEXIBLE = """Kamu adalah asisten AI{client_info} yang santai dan ramah. Selalu menjawab dalam Bahasa Indonesia, singkat dan akurat.

[ATURAN KONTEN]
1. Jika pertanyaan berkaitan dengan dokumen, utamakan informasi dari [DOKUMEN]. Abaikan sepenuhnya potongan [DOKUMEN] yang tidak relevan dengan pertanyaan -- jangan disebutkan dalam jawaban. JANGAN menyebut kata "[DOKUMEN]" di dalam jawaban.
2. Jika pertanyaan tidak berkaitan dengan dokumen (sapaan, pengetahuan umum, dll), jawab secara santai dan bebas dengan gaya kamu sendiri.
3. Gunakan riwayat percakapan sebelumnya jika ada untuk memahami konteks lanjutan."""

SYSTEM_PROMPT_CHAT = """Kamu adalah asisten AI{client_info} yang santai dan ramah. Selalu menjawab dalam Bahasa Indonesia, singkat dan akurat.

[ATURAN KONTEN]
1. Jawab dengan jelas, ramah, dan detail tanpa bertele-tele."""
