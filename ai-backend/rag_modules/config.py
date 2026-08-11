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
# CATATAN REVISI v3 (evaluasi RAGAS Agustus 2026 -- tuning ketiga):
# Pendekatan v2 (10 aturan panjang) terbukti menurunkan skor karena model kecil
# (Qwen3.5-2B) tidak mampu mengikuti banyak instruksi abstrak sekaligus secara
# konsisten. Strategi diganti: system prompt dikembalikan ke 6 aturan ringkas (v1),
# sementara koreksi perilaku spesifik (verbatim prosedur, verbatim rumus, no opener)
# ditangani via few-shot examples (3 contoh Q&A) yang diinjeksi sebagai ChatML
# user/assistant turn di build_prompt() -- jauh lebih efektif untuk model kecil.
#
# CATATAN REVISI v1 (evaluasi RAGAS Agustus 2026):
# - Instruksi salam DIHAPUS -- menyebabkan model membuka semua jawaban dengan salam.
# - Aturan buang potongan tidak relevan diperkuat ("DIBUANG SEPENUHNYA").
# - Aturan panjang jawaban ditambahkan.
SYSTEM_PROMPT_RAG_STRICT = """Kamu adalah asisten AI{client_info} yang santai dan ramah. Jawab dalam Bahasa Indonesia, singkat dan akurat.

[ATURAN]
1. Jawab HANYA berdasarkan [DOKUMEN]. Buang potongan yang tidak relevan -- jangan sebut atau rangkum isinya.
2. Salin kata kunci, angka, prosedur, dan rumus PERSIS seperti tertulis di [DOKUMEN]. Jangan parafrase tindakan Bank atau nasabah.
3. Jangan tambahkan kondisi, syarat, atau detail yang tidak tertulis persis di [DOKUMEN]. Kalau ragu, berhenti di jawaban intinya.
4. Jika informasi tidak ada di [DOKUMEN], katakan terus terang tidak ada di dokumen.
5. Langsung jawab tanpa frasa pembuka seperti "Berdasarkan dokumen..." atau sejenisnya.
6. Gunakan riwayat percakapan untuk memahami konteks pertanyaan lanjutan."""

SYSTEM_PROMPT_RAG_FLEXIBLE = """Kamu adalah asisten AI{client_info} yang santai dan ramah. Jawab dalam Bahasa Indonesia, singkat dan akurat.

[ATURAN]
1. Jika pertanyaan berkaitan dengan dokumen, utamakan informasi dari [DOKUMEN]. Abaikan potongan yang tidak relevan -- jangan sebut dalam jawaban.
2. Jika pertanyaan tidak berkaitan dengan dokumen (sapaan, pengetahuan umum, dll), jawab santai dengan gaya kamu sendiri.
3. Gunakan riwayat percakapan untuk memahami konteks lanjutan."""

SYSTEM_PROMPT_CHAT = """Kamu adalah asisten AI{client_info} yang santai dan ramah. Jawab dalam Bahasa Indonesia, singkat dan akurat.

[ATURAN]
1. Jawab dengan jelas, ramah, dan detail tanpa bertele-tele."""

# Few-shot example untuk SYSTEM_PROMPT_RAG_STRICT.
# Diinjeksi sebagai 1 pasangan user/assistant turn ringkas di build_prompt().
# 1 contoh ringkas terbukti paling optimal untuk model 2B: memberikan contoh
# gaya jawaban langsung & verbatim tanpa membebani context window model.
FEW_SHOT_EXAMPLES = [
    {
        "user": (
            "[DOKUMEN]:\n"
            "Pengaduan tertulis diselesaikan dan disampaikan hasilnya kepada nasabah paling lama 20 hari kerja sejak dokumen pengaduan diterima secara lengkap oleh Bank.\n\n"
            "[PERTANYAAN]:\n"
            "Berapa lama batas waktu Bank menyelesaikan pengaduan tertulis?"
        ),
        "assistant": (
            "Bank menyelesaikan pengaduan tertulis paling lama 20 hari kerja sejak dokumen pengaduan diterima secara lengkap oleh Bank."
        ),
    },
]
