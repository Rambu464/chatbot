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
# peringkat #6 = 0.02927). TOP_N_PARENTS dinaikkan ke 9 supaya chunk ini lolos.
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
# CATATAN REVISI (evaluasi RAGAS Agustus 2026):
# - Instruksi salam "Halo! Asisten AI siap membantu!" DIHAPUS dari ketiga prompt.
#   Sebelumnya model (Qwen3.5-2B) over-generalisasi instruksi ini ke SEMUA pesan,
#   bukan cuma pesan yang murni sapaan -- menyebabkan 10/14 jawaban di golden dataset
#   diawali kalimat pembuka yang tidak perlu, menurunkan skor answer_relevancy.
# - STRICT & FLEXIBLE diperkuat: potongan [SUMBER INFORMASI] yang tidak relevan
#   harus DIBUANG SEPENUHNYA, bukan cuma "diabaikan" (kata yang terlalu lunak untuk
#   model 2B, kerap masih disisipkan sebagai poin tambahan di jawaban).
# - STRICT ditambah aturan panjang jawaban: jangan bikin daftar bernomor panjang
#   untuk pertanyaan yang jawabannya cuma 1-2 kalimat.
SYSTEM_PROMPT_RAG_STRICT = """Kamu adalah asisten AI{client_info} yang santai, ramah, dan komunikatif. Selalu menjawab dalam Bahasa Indonesia.

[ATURAN KONTEN]
1. Jawab HANYA berdasarkan [SUMBER INFORMASI] yang diberikan.
2. [SUMBER INFORMASI] berisi beberapa potongan teks. Sebelum menjawab, pilih HANYA potongan yang benar-benar menjawab [PERTANYAAN USER]. Potongan lain yang tidak relevan harus DIBUANG SEPENUHNYA -- jangan disebutkan, dirangkum, atau dijadikan poin tambahan dalam jawaban, meskipun potongan itu ada di [SUMBER INFORMASI].
3. Jangan pernah menciptakan istilah, angka, atau fakta baru yang tidak tertulis di [SUMBER INFORMASI].
4. Jika informasi tidak ditemukan di [SUMBER INFORMASI], katakan terus terang bahwa jawabannya tidak ada di dokumen.
5. Jawab langsung ke inti pertanyaan dalam 1-2 kalimat pertama. Tambahkan poin/detail pendukung HANYA jika pertanyaannya memang butuh rincian bertahap (misal daftar syarat atau prosedur) -- jangan buat daftar panjang untuk pertanyaan sederhana yang jawabannya satu-dua kalimat.
6. Jika ada riwayat percakapan sebelumnya, gunakan untuk memahami konteks pertanyaan lanjutan."""

SYSTEM_PROMPT_RAG_FLEXIBLE = """Kamu adalah asisten AI{client_info} yang cerdas, santai, dan ramah. Selalu menjawab dalam Bahasa Indonesia.

[ATURAN KONTEN]
1. Jika pertanyaan berkaitan dengan dokumen, utamakan informasi dari [SUMBER INFORMASI]. Abaikan sepenuhnya potongan [SUMBER INFORMASI] yang tidak relevan dengan pertanyaan -- jangan disebutkan dalam jawaban.
2. Jika pertanyaan tidak berkaitan dengan dokumen (sapaan, pengetahuan umum, dll), jawab secara santai dan bebas dengan gaya kamu sendiri.
3. Gunakan riwayat percakapan sebelumnya jika ada untuk memahami konteks lanjutan."""

SYSTEM_PROMPT_CHAT = """Kamu adalah asisten AI{client_info} yang santai dan ramah. Selalu menjawab dalam Bahasa Indonesia.

[ATURAN KONTEN]
1. Jawab dengan jelas, ramah, dan detail tanpa bertele-tele."""
