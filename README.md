
### RAGChat - AI Research Assistant

Proyek ini adalah implementasi sistem Retrieval-Augmented Generation (RAG) berbasis monorepo yang memadukan
antarmuka chatbot modern dengan backend AI lokal. Sistem ini dirancang untuk membaca dan menganalisis
dokumen PDF serta menjawab pertanyaan secara interaktif menggunakan Small Language Model (SLM), berjalan
sepenuhnya lokal (offline) baik di CPU maupun GPU — sekarang dilengkapi juga dengan **login, manajemen
peran pengguna, dan riwayat chat permanen**.

```markdown
# Struktur Direktori
chatbot-fork/
├── README.md
├── working.md
├── ai-backend/
│   ├── main.py          # FastAPI: RAG, auth, chat session, admin
│   ├── access.py        # Utilitas autentikasi dan otorisasi
│   ├── database.py      # SQLite: users, chat_sessions, messages
│   ├── rag.py           # Kompatibilitas/facade untuk pipeline RAG
│   ├── rag_modules/     # Modul cache, konfigurasi, ingestion, retrieval, dan generation
│   ├── debug_retrieval_scores.py
│   ├── eval_ragas.py
│   ├── stress_test_faithfulness.py
│   ├── golden_dataset.json
│   ├── manual_test_questions.json
│   ├── manual_cache_eval.py
│   ├── populate_via_api.py
│   ├── verify_cache_persistence.py
│   ├── requirements.txt
│   └── models/           # File model .gguf lokal, diabaikan Git
└── chatbot-frontend/
  ├── app/               # Halaman Next.js: chat, login, dan admin
  ├── components/        # Komponen layout, tema, sidebar, dan komponen UI
  ├── hooks/             # Custom React hooks
  ├── lib/               # Utilitas frontend
  ├── public/            # Widget chatbot dan aset publik
  ├── styles/            # Style tambahan
  ├── package.json
  ├── pnpm-lock.yaml
  ├── next.config.mjs
  └── tsconfig.json
```

## Arsitektur Singkat

**Sisi AI/RAG:**
- **Inference model**: langsung via `llama-cpp-python` (bukan Ollama) — model GGUF di-load in-process oleh
  backend FastAPI, tidak butuh server model terpisah.
- **Retrieval**: Parent Document Retriever (LangChain) — pencarian similarity jalan di level *child chunk*
  (kecil, presisi), tapi konteks yang dikirim ke LLM adalah *parent chunk* (lebih utuh) dan otomatis
  ter-dedup kalau beberapa child chunk mengarah ke parent yang sama.
- **Vector store**: ChromaDB, persist ke disk.
- **Multi-dokumen**: sistem mendukung banyak PDF ter-upload sekaligus, dengan opsi scoping pencarian ke
  1 dokumen aktif spesifik atau ke semua dokumen.
- **Semantic cache**: prompt yang mirip secara makna (bukan cuma exact match) dijawab dari cache tanpa
  generate ulang, di-scope per dokumen aktif **dan** per mode jawaban (lihat "Mode Jawaban" di bawah) supaya
  tidak salah campur.
- **Mode Jawaban (toggle)**: default **Mode Ketat** — AI hanya menjawab dari dokumen (anti-halusinasi).
  User bisa beralih ke **Mode Bebas** untuk obrolan umum di luar topik dokumen, tanpa mengubah perilaku
  default sistem.

**Sisi Aplikasi/UI:**
- **Autentikasi**: login wajib (username/password), token disimpan di `localStorage`, dikirim sebagai
  `Authorization: Bearer <token>` di setiap request ke backend.
- **Peran pengguna (role-based access)**: `superadmin`, `admin`, `user` — lihat tabel di bawah.
- **Riwayat chat permanen**: setiap sesi dan pesan tersimpan di SQLite (`chatbot.db`), tetap ada setelah
  logout/refresh, ditampilkan di sidebar.
- **Force Stop**: tombol berhenti saat AI sedang mengetik — generation di backend benar-benar dihentikan
  (bukan cuma berhenti tampil di layar), dan potongan jawaban yang sudah sempat digenerate tetap tersimpan
  ke riwayat.
- **Scroll bebas saat AI mengetik**: pengguna bisa scroll ke atas membaca pesan lama tanpa terus "ditarik"
  ke bawah selama AI masih streaming jawaban.

## Persyaratan Sistem
Pastikan sistem sudah terinstal perangkat lunak berikut:
1. **Node.js** (v18 atau lebih baru)
2. **Python** (v3.11 atau lebih baru)
3. **Git**

> Catatan: Ollama **tidak diperlukan** — model GGUF di-load langsung oleh backend lewat `llama-cpp-python`.

---

1. Spesifikasi Minimum (Running on CPU)
* **Prosesor:** Intel Core i5 Gen 10 / AMD Ryzen 5 3000 Series ke atas (teruji lancar di Intel Core i5-12450H).
* **RAM:** 8 GB (minimal sisa RAM bebas 4 GB untuk model & backend).
* **Penyimpanan:** ruang kosong minimal 5 GB.
* Waktu respons di CPU murni berkisar **8-15 detik** sampai token pertama muncul, tergantung panjang
  konteks dokumen yang di-retrieve.

2. Spesifikasi Rekomendasi (Running on GPU)
* **Prosesor:** Intel Core i7 / AMD Ryzen 7 Series terbaru.
* **RAM:** 16 GB (sangat disarankan untuk stabilitas *monorepo*).
* **GPU:** NVIDIA RTX 3050 / 4050 (atau lebih tinggi) dengan **VRAM 4 GB+** — cukup untuk full-offload
  model 2B ke GPU (`n_gpu_layers=-1` di `main.py`).
* **Penyimpanan:** ruang kosong minimal 5 GB.

## Panduan Instalasi & Menjalankan Sistem

### 1. Unduh Model GGUF
Proyek ini menggunakan model **Qwen3.5-2B** format GGUF agar bisa berjalan secara lokal via llama.cpp.
1. Buka tautan repositori: [unsloth/Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/blob/main/Qwen3.5-2B-UD-Q4_K_XL.gguf)
2. Unduh file `Qwen3.5-2B-UD-Q4_K_XL.gguf` (kuantisasi Unsloth Dynamic — keseimbangan terbaik antara
   ukuran dan kualitas untuk perangkat CPU/GPU terbatas), atau pakai hasil fine-tuning sendiri (format
   `q4_k_m`).
3. Buat folder `models/` di dalam `ai-backend/`, lalu pindahkan file `.gguf` tersebut ke sana:
   ```
   ai-backend/models/Qwen3.5-2B-UD-Q4_K_XL.gguf
   ```
   Kalau nama file atau lokasinya berbeda, sesuaikan lewat environment variable `MODEL_PATH` (lihat
   Bagian Konfigurasi di bawah), tidak perlu mengubah kode.

### 2. Menjalankan Backend (FastAPI)
Backend mengelola pemrosesan PDF, penyimpanan vektor, autentikasi, riwayat chat, dan langsung menjalankan
inference model (tanpa server model terpisah).
1. Masuk ke folder backend:
   ```bash
   cd ai-backend
   ```
2. Buat dan aktifkan *Virtual Environment*:
   ```bash
   python -m venv venv
   ```
   Di PowerShell:
      ```bash
      .\venv\Scripts\Activate.ps1
      ```
   Di Command Prompt (CMD):
      ```bash
      venv\Scripts\activate
      ```
   Di Git Bash / Terminal Linux:
      ```bash
      source venv/Scripts/activate
      ```
3. Instal dependensi Python:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan server backend:
   ```bash
   python main.py
   ```
   *(Backend berjalan di `http://localhost:8000`)*. Saat pertama kali jalan, file `chatbot.db` (SQLite)
   otomatis dibuat berikut 3 akun default (lihat bagian Autentikasi di bawah).

### 3. Menjalankan Frontend (Next.js)
1. Buka tab terminal baru (biarkan terminal backend tetap menyala) dan arahkan ke folder frontend:
   ```bash
   cd chatbot-frontend
   ```
2. Instal dependensi Node.js:
   ```bash
   npm install
   ```
3. Jalankan server frontend:
   ```bash
   npm run dev
   ```
   *(Frontend berjalan di `http://localhost:3000`)*. Buka browser, kamu akan diarahkan ke halaman login.

---

## 🔐 Autentikasi & Peran Pengguna

Login wajib sebelum mengakses chatbot. 3 akun default ter-seed otomatis saat `chatbot.db` pertama kali dibuat:

| Role | Username | Password | Bisa Apa |
|---|---|---|---|
| **Superadmin** | `superadmin` | `super123` | Semua akses Admin, + kelola akun user (tambah/hapus) lewat Admin Dashboard |
| **Admin** | `admin` | `admin123` | Chat, upload/hapus dokumen PDF ke basis RAG |
| **User** | `user` | `user123` | Chat saja, tidak bisa upload/hapus dokumen |

> ⚠️ **Ganti password default ini** kalau sistem akan diakses lebih dari sekadar demo lokal. Password
> saat ini disimpan **plaintext** di database (lihat catatan di `database.py`) — cukup aman untuk
> prototipe/demo skripsi, tapi sebaiknya di-hash (mis. bcrypt) sebelum dipakai dengan data pengguna nyata.

Token login disimpan di `localStorage` browser dan dikirim sebagai header `Authorization: Bearer <token>`
ke setiap request backend. Logout akan menghapus token dari `localStorage`.

## 💬 Riwayat Chat

Setiap pesan (user maupun AI) tersimpan permanen ke SQLite (`chatbot.db`), dikelompokkan per **sesi chat**.
- Sidebar menampilkan daftar sesi milik user yang sedang login, bisa diklik untuk membuka kembali,
  atau dihapus.
- Kalau AI dihentikan di tengah jalan (tombol Stop), potongan jawaban yang sudah sempat muncul tetap
  tersimpan ke riwayat — tidak hilang.
- Admin dan Superadmin bisa melihat sesi chat siapa pun (bukan hanya miliknya sendiri) lewat endpoint yang sama.

## ⚙️ Konfigurasi (Environment Variables)

Semua opsional — kalau tidak di-set, backend pakai nilai default yang sudah teruji jalan.

| Variabel | Default | Keterangan |
|---|---|---|
| `MODEL_PATH` | `./models/Qwen3.5-2B-UD-Q4_K_XL.gguf` | Lokasi file model GGUF |
| `CHROMA_DIR` | `./chroma_db` | Lokasi penyimpanan vector store |
| `PARENT_STORE_DIR` | `./parent_docstore` | Lokasi penyimpanan parent chunk (Parent Document Retriever) |
| `DOCUMENTS_REGISTRY_PATH` | `./documents_registry.json` | Daftar nama dokumen yang sudah di-upload |
| `N_CTX` | `8192` | Context window model |
| `N_THREADS` | jumlah core CPU | Jumlah thread untuk inference |
| `VERBOSE_LLM` | `false` | Set `true` untuk lihat log detail llama.cpp saat startup (mis. cek dukungan AVX2) |

## 🔗 Cara Kerja Integrasi (Alur Sistem)

1. **Frontend (Port 3000):** Pengguna login terlebih dahulu, token disimpan di `localStorage`. Setelah
   masuk, pengguna bisa memilih **dokumen aktif** (atau "semua dokumen") dan **mode jawaban** (Ketat/Bebas)
   lewat kontrol di atas kolom chat, lalu mengirim pesan atau (khusus Admin/Superadmin) mengunggah PDF.
   Setiap request menyertakan token di header `Authorization`.
2. **Backend (Port 8000):**
   - Memverifikasi token di setiap request; menolak (401/403) kalau token tidak valid atau role tidak
     punya izin (mis. `user` mencoba upload PDF).
   - Pesan user & jawaban AI disimpan ke SQLite sebelum/selama streaming berlangsung.
   - Saat PDF di-upload: teks diekstrak (PyMuPDF), dipecah jadi *parent* dan *child chunk* (Parent
     Document Retriever), lalu disimpan ke **ChromaDB** (child, untuk pencarian) dan **docstore lokal**
     (parent, untuk konteks yang dikirim ke model).
   - Saat ada pertanyaan chat: backend cek dulu apakah pertanyaan mirip secara makna dengan yang pernah
     ditanyakan sebelumnya **untuk dokumen dan mode jawaban yang sama** (semantic cache). Kalau tidak ada
     cache yang cocok, backend mencari potongan teks relevan dari ChromaDB (di-scope ke dokumen aktif kalau
     dipilih), menggabungkannya dengan pertanyaan pengguna, dan menjalankan generate lewat llama.cpp secara
     langsung di dalam proses backend (tidak ada server model terpisah).
   - Kalau pengguna menekan Stop, backend mendeteksi koneksi terputus dan benar-benar menghentikan proses
     generate (bukan cuma berhenti mengirim ke frontend), lalu menyimpan potongan jawaban terakhir.
3. **Model (llama.cpp, in-process):** Memproses prompt yang berisi konteks dari Backend dan mengembalikan
   jawaban secara streaming (token demi token) ke layar pengguna.

## 📄 Manajemen Dokumen (Admin & Superadmin)

- `GET /api/documents` — daftar semua dokumen yang sudah di-upload (bisa diakses semua role yang login).
- `POST /api/upload` — upload PDF baru ke basis RAG (**hanya Admin/Superadmin**).
- `DELETE /api/documents/{filename}` — hapus 1 dokumen beserta seluruh chunk dan cache-nya (**hanya
  Admin/Superadmin**).
- Upload ulang file dengan nama yang sama akan otomatis menggantikan (bukan menumpuk duplikat) chunk lama.
- Segala ID Dokumen yang diupload dapat dilihat melalui perintah berikut di terminal :
  ```bash
  python -c "from database import get_all_clients; import json; print(json.dumps(get_all_clients(), indent=2))"
  ```
  ```bash
  python -c "from database import get_documents_by_client; import json; print(json.dumps(get_documents_by_client(GANTI_DENGAN_ID_DANAMON), indent=2)"
  ```

## 👤 Manajemen User (Superadmin)

- `GET /api/admin/users` — daftar semua akun.
- `POST /api/admin/users` — tambah akun baru (`username`, `password`, `role`).
- `DELETE /api/admin/users/{user_id}` — hapus akun.

Semua endpoint di atas hanya bisa diakses oleh role `superadmin`, tersedia lewat halaman **Admin Dashboard**
di frontend.

## ⚠️ Catatan Penting
* File `.gguf`, folder `ai-backend/models/`, `chroma_db/`, `parent_docstore/`, serta `chatbot.db` diabaikan
  oleh Git (lewat `.gitignore`) karena ukurannya besar dan/atau bersifat data lokal. Jika melakukan *clone*
  proyek ini di komputer baru:
  - Model wajib diunduh ulang dari Hugging Face (lihat langkah 1).
  - Seluruh dokumen PDF perlu di-upload ulang lewat UI.
  - Akun default (superadmin/admin/user) akan ter-seed ulang otomatis saat backend pertama kali dijalankan.
* Sistem berjalan sepenuhnya lokal/offline setelah model ter-unduh — tidak ada panggilan API eksternal saat
  chat maupun upload dokumen.
* Password akun disimpan plaintext di database — lihat peringatan di bagian Autentikasi di atas.