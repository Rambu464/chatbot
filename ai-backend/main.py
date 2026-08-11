"""
main.py -- Aplikasi FastAPI: Routing HTTP, autentikasi JWT, manajemen client, dan sesi chat.

File ini berfungsi sebagai entry point server FastAPI dan orchestrator request HTTP.
Semua pemrosesan data RAG diposisikan di modul `rag` (terisolasi per-client).
"""
import os
import time
import asyncio
import threading
import traceback
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import database
import rag
from access import validate_upload_access


# =========================================================
# LIFESPAN -- inisialisasi resource GLOBAL RAG saat server startup
# (resource per-client dibuat lazy saat dipakai, lihat rag.get_client_state)
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    await rag.initialize()
    yield


app = FastAPI(lifespan=lifespan)

# Konfigurasi CORS Middleware (menggunakan Bearer token header)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# AUTENTIKASI
# =========================================================
async def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Header otentikasi tidak valid atau kosong.")
    token = authorization.split(" ")[1]
    user = database.get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Token tidak valid atau tidak dikenali.")
    return user


def require_role(user: dict, allowed_roles: list, action: str):
    if user["role"] not in allowed_roles:
        raise HTTPException(status_code=403, detail=f"Akses ditolak. {action} hanya untuk: {', '.join(allowed_roles)}.")


def require_client_access(user: dict, client_id: int):
    """admin_client cuma boleh akses client miliknya sendiri; superadmin/admin bebas."""
    if user["role"] == "admin_client":
        user_cid = user.get("client_id")
        if user_cid is None:
            raise HTTPException(status_code=403, detail="Akses ditolak. Anda belum memiliki Client yang ditugaskan.")
        try:
            if int(user_cid) != int(client_id):
                raise HTTPException(status_code=403, detail="Akses ditolak. Anda hanya dapat mengakses data Client Anda sendiri.")
        except (ValueError, TypeError):
            raise HTTPException(status_code=403, detail="Akses ditolak. ID client tidak valid.")



# =========================================================
# MULTI-TURN: ambil riwayat chat terakhir dari database
# =========================================================
def get_recent_history(session_id: str, max_turns: Optional[int] = None) -> list:
    """Ambil MAX_HISTORY_TURNS pasang (user+assistant) TERAKHIR dari database
    untuk 1 sesi, dipanggil SEBELUM pesan user yang baru disimpan -- jadi hasil
    fungsi ini TIDAK termasuk pesan yang lagi diproses sekarang."""
    max_turns = max_turns or rag.MAX_HISTORY_TURNS
    messages = database.get_chat_messages(session_id)
    trimmed = messages[-(max_turns * 2):] if messages else []
    return [
        {"role": m["role"], "content": m["content"]}
        for m in trimmed
        if m["role"] in ("user", "assistant") and m["content"].strip()
    ]


# =========================================================
# SCHEMAS
# =========================================================
class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str
    session_id: str
    document: Optional[str] = None
    general_mode: bool = False

class SavePartialRequest(BaseModel):
    session_id: str
    content: str

class WidgetChatRequest(BaseModel):
    message: str
    session_id: str
    document: Optional[str] = None
    general_mode: bool = False


class CreateSessionRequest(BaseModel):
    title: str
    client_id: int

class UserCreateRequest(BaseModel):
    username: str
    password: str
    role: str
    client_id: Optional[int] = None

class ClientCreateRequest(BaseModel):
    name: str
    type: str

class ClientInstansiCreateRequest(BaseModel):
    username: str
    instansi_name: str
    email: str
    password: str
    client_type: str

class ClientInstansiUpdateRequest(BaseModel):
    username: str
    instansi_name: str
    client_type: str
    password: Optional[str] = None


# =========================================================
# 1. AUTH
# =========================================================
@app.post("/api/auth/login")
async def login_endpoint(request: LoginRequest):
    user = database.get_user_by_credentials(request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Username atau password salah.")
    return {
        "username": user["username"],
        "role": user["role"],
        "token": user["token"],
        "client_id": user["client_id"],
        "client_name": user["client_name"],
        "password_changed": user["password_changed"],
    }


class ChangePasswordRequest(BaseModel):
    new_password: str


@app.post("/api/auth/change-password")
async def change_password_endpoint(request: ChangePasswordRequest, user=Depends(get_current_user)):
    try:
        database.update_user_password(user["id"], request.new_password)
        return {"status": "success", "message": "Password berhasil diperbarui."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



# =========================================================
# 2. MANAJEMEN CLIENT (superadmin/admin)
# =========================================================
@app.get("/api/clients")
async def list_clients(user=Depends(get_current_user)):
    return database.get_all_clients()

@app.post("/api/clients")
async def create_client(request: ClientCreateRequest, user=Depends(get_current_user)):
    require_role(user, ["superadmin", "admin"], "Menambah client")
    try:
        return database.add_client(request.name, request.type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/clients/{client_id}")
async def remove_client(client_id: int, user=Depends(get_current_user)):
    require_role(user, ["superadmin", "admin"], "Menghapus client")
    database.delete_client(client_id)
    rag.delete_client_data(client_id)
    return {"message": "Client berhasil dihapus."}


# =========================================================
# 3. CHAT SESSIONS
# =========================================================
@app.get("/api/chat/sessions")
async def get_sessions(user=Depends(get_current_user)):
    return database.get_chat_sessions(user["id"])

@app.post("/api/chat/sessions")
async def create_session(request: CreateSessionRequest, user=Depends(get_current_user)):
    require_client_access(user, request.client_id)
    return database.create_chat_session(user["id"], request.client_id, request.title)

@app.get("/api/chat/sessions/{session_id}")
async def get_session_messages(session_id: str, user=Depends(get_current_user)):
    sessions = database.get_chat_sessions(user["id"])
    session_ids = [s["id"] for s in sessions]
    if session_id not in session_ids and user["role"] not in ["admin", "superadmin"]:
        raise HTTPException(status_code=403, detail="Akses ditolak ke sesi chat ini.")
    return database.get_chat_messages(session_id)

@app.delete("/api/chat/sessions/{session_id}")
async def delete_session(session_id: str, user=Depends(get_current_user)):
    success = database.delete_chat_session(session_id, user["id"])
    if not success:
        raise HTTPException(status_code=400, detail="Gagal menghapus sesi. Pastikan Anda adalah pemiliknya.")
    return {"message": "Sesi chat berhasil dihapus."}


# =========================================================
# 4. CHAT (Hybrid RAG per-client + Semantic Cache + multi-turn + riwayat SQLite)
# =========================================================
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest, user=Depends(get_current_user)):
    start_time = time.time()
    user_input = request.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Pesan tidak boleh kosong.")

    sessions = database.get_chat_sessions(user["id"])
    session_ids = [s["id"] for s in sessions]
    if request.session_id not in session_ids and user["role"] not in ["admin", "superadmin"]:
        raise HTTPException(status_code=403, detail="Akses ditolak ke sesi chat ini.")

    conn = database.get_db_connection()
    session = conn.execute(
        "SELECT s.client_id, c.name as client_name FROM chat_sessions s JOIN clients c ON s.client_id = c.id WHERE s.id = ?",
        (request.session_id,)
    ).fetchone()
    conn.close()
    if not session:
        raise HTTPException(status_code=404, detail="Sesi chat tidak ditemukan.")
    client_id = session["client_id"]
    client_name = session["client_name"]

    if request.document:
        client_doc_names = [d["filename"] for d in database.get_documents_by_client(client_id)]
        if request.document not in client_doc_names:
            raise HTTPException(status_code=404, detail=f"Dokumen '{request.document}' tidak ditemukan untuk client ini.")

    # MULTI-TURN: ambil riwayat SEBELUM pesan baru disimpan, supaya tidak
    # dobel-hitung pesan yang sedang diproses sekarang.
    history = get_recent_history(request.session_id)

    database.save_message(request.session_id, "user", request.message)

    cache_key = rag.make_cache_key(client_id, request.document, request.general_mode)
    user_input_lower = user_input.lower()
    query_embedding = rag.state.embeddings.embed_query(user_input_lower)

    # 1. CEK SEMANTIC CACHE -- HANYA untuk pesan PERTAMA di sesi (history kosong).
    #    Kenapa: entry cache dibuat murni dari embedding teks query, TANPA
    #    memperhitungkan riwayat percakapan. Untuk pesan lanjutan yang maknanya
    #    bergantung konteks (mis. "boleh kasih tau stepnya"), cache lookup
    #    berisiko salah nyambungin ke jawaban dari sesi/topik lain yang
    #    kebetulan mirip secara embedding. Begitu ada histori, cache dilewati
    #    demi keamanan/akurasi jawaban (trade-off: sedikit mengurangi cache
    #    hit rate untuk percakapan panjang, tapi lebih aman).
    best_match, best_score = (None, 0.0)
    if not history:
        best_match, best_score = await rag.check_cache(cache_key, user_input_lower, query_embedding)

    if best_match:
        async def stream_cache():
            words = best_match["response"].split(" ")
            for word in words:
                yield word + " "
                await asyncio.sleep(0.01)
            yield f"\n\n*(⚡ Diambil dari Semantic Cache - {best_score*100:.1f}% mirip, {time.time() - start_time:.4f} detik)*"
            database.save_message(request.session_id, "assistant", best_match["response"])

        return StreamingResponse(stream_cache(), media_type="text/plain")

    # 2. HYBRID RAG RETRIEVAL (di-scope ke client_id + dokumen aktif kalau ada)
    retrieval_start = time.time()
    context, is_rag_mode = rag.get_context(client_id, user_input_lower, query_embedding, request.document)
    print(f"[TIMING] Retrieval: {time.time() - retrieval_start:.3f} detik")

    system_prompt = rag.select_system_prompt(is_rag_mode, request.general_mode, client_name=client_name)
    prompt = rag.build_prompt(system_prompt, user_input, context, history=history)

    stop_event = threading.Event()

    async def generate_stream():
        full_response = ""
        generation_start = time.time()
        first_token_time = None
        token_count = 0
        try:
            yield "\n"  # flush headers, trigger browser streaming
            async for chunk in rag.stream_llm(prompt, stop_event):
                if first_token_time is None:
                    first_token_time = time.time()
                    print(f"[TIMING] Waktu sampai token pertama: {first_token_time - generation_start:.3f} detik")
                full_response += chunk
                token_count += 1
                if chunk:
                    yield chunk
        except asyncio.CancelledError:
            print("Streaming dihentikan oleh user / koneksi terputus.")
            raise
        finally:
            elapsed = time.time() - generation_start
            if elapsed > 0 and token_count > 0:
                print(f"[TIMING] Total generate: {elapsed:.3f} detik untuk ~{token_count} chunk "
                      f"(~{token_count / elapsed:.2f} chunk/detik)")
            if full_response.strip():
                try:
                    database.save_message(request.session_id, "assistant", full_response)
                except Exception as db_err:
                    print(f"Gagal menyimpan pesan parsial: {db_err}")

                if not stop_event.is_set():
                    await rag.store_cache(cache_key, user_input_lower, query_embedding, full_response)

    return StreamingResponse(generate_stream(), media_type="text/plain")


@app.post("/api/chat/save_partial")
async def save_partial_endpoint(request: SavePartialRequest, user=Depends(get_current_user)):
    if request.content.strip():
        try:
            database.save_message(request.session_id, "assistant", request.content)
            return {"status": "saved"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gagal menyimpan pesan: {str(e)}")
    return {"status": "empty"}


@app.post("/api/widget/chat")
async def widget_chat_endpoint(
    request: WidgetChatRequest,
    x_api_key: str = Header(..., alias="X-API-KEY")
):
    # 1. Validasi API Key
    client = database.get_client_by_api_key(x_api_key)
    if not client:
        raise HTTPException(status_code=401, detail="API Key Widget tidak valid atau tidak ditemukan.")

    client_id = client["id"]
    client_name = client["name"]
    user_input = request.message.strip()

    if not user_input:
        raise HTTPException(status_code=400, detail="Pesan tidak boleh kosong.")

    # Ambil riwayat chat lokal widget jika ada
    # (Karena widget.js/livechat.tsx di-render secara stateless/tanpa login user,
    #  kita bypass saving message di DB jika session_id tidak terdaftar di chat_sessions,
    #  namun kita tetap proses RAG-nya agar responsenya keluar!)
    try:
        history = get_recent_history(request.session_id)
    except:
        history = []

    # Cek & simpan message jika session_id valid di db kita (opsional)
    try:
        database.save_message(request.session_id, "user", request.message)
    except:
        pass

    # 2. Proses RAG/LLM
    user_input_lower = user_input.lower()
    query_embedding = rag.state.embeddings.embed_query(user_input_lower)

    # Cek Cache
    cache_key = rag.make_cache_key(client_id, request.document, request.general_mode)
    best_match = None
    if not history:
        try:
            best_match, _ = await rag.check_cache(cache_key, user_input_lower, query_embedding)
        except:
            pass

    if best_match:
        # Simpan assistant message jika session_id valid
        try:
            database.save_message(request.session_id, "assistant", best_match["response"])
        except:
            pass
        return {"response": best_match["response"], "source": "cache"}

    # Jalankan Retrieval
    context, is_rag_mode = rag.get_context(client_id, user_input_lower, query_embedding, request.document)
    system_prompt = rag.select_system_prompt(is_rag_mode, request.general_mode, client_name=client_name)
    prompt = rag.build_prompt(system_prompt, user_input, context, history=history)

    # Generate response
    stop_event = threading.Event()
    full_response = ""
    try:
        async for chunk in rag.stream_llm(prompt, stop_event):
            full_response += chunk
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal generate LLM: {str(e)}")

    # Simpan assistant message ke DB jika session_id valid
    if full_response.strip():
        try:
            database.save_message(request.session_id, "assistant", full_response)
        except:
            pass
        try:
            await rag.store_cache(cache_key, user_input_lower, query_embedding, full_response)
        except:
            pass

    return {"response": full_response, "source": "llm"}



# =========================================================
# 5. DOKUMEN (RAG upload/list/delete, per client)
# =========================================================
@app.get("/api/documents/{client_id}")
async def list_documents_by_client(client_id: int, user=Depends(get_current_user)):
    require_client_access(user, client_id)
    return database.get_documents_by_client(client_id)


@app.get("/api/documents")
async def list_documents_all(user=Depends(get_current_user)):
    """Mengambil daftar seluruh dokumen (dikelompokkan per client untuk Admin/Superadmin)."""
    if user["role"] == "admin_client":
        if user.get("client_id") is None:
            return {"documents": []}
        return {"documents": database.get_documents_by_client(user["client_id"])}

    all_docs = []
    for client in database.get_all_clients():
        docs = database.get_documents_by_client(client["id"])
        for d in docs:
            d["client_name"] = client["name"]
        all_docs.extend(docs)
    return {"documents": all_docs}


@app.post("/api/upload")
async def upload_endpoint(
    file: UploadFile = File(...),
    client_id: Optional[int] = Query(None),
    user=Depends(get_current_user),
):
    require_role(user, ["admin", "superadmin", "admin_client"], "Upload dokumen")
    client_id = validate_upload_access(user, client_id)

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya file PDF yang didukung.")

    file_path = f"./temp_uploads/{file.filename}"
    try:
        os.makedirs("temp_uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Ekstraksi dan pengindeksan PDF per-client
        n_chunks = rag.ingest_pdf(client_id, file_path, file.filename)
        doc_metadata = database.add_document(client_id, file.filename, "PDF")

        return {
            "message": f"Sip! sudah dibaca '{file.filename}' ({n_chunks} halaman/potongan).",
            "doc": doc_metadata,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        print("=== ERROR SAAT UPLOAD PDF ===")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Gagal proses PDF: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.delete("/api/documents/{doc_id}")
async def remove_document(doc_id: int, user=Depends(get_current_user)):
    conn = database.get_db_connection()
    doc = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    conn.close()

    if not doc:
        raise HTTPException(status_code=404, detail="Dokumen tidak ditemukan.")

    client_id = doc["client_id"]
    filename = doc["filename"]

    require_role(user, ["superadmin", "admin", "admin_client"], "Hapus dokumen")
    require_client_access(user, client_id)

    database.delete_document(doc_id)
    rag.delete_document(client_id, filename)

    return {"message": f"Dokumen '{filename}' berhasil dihapus."}


# =========================================================
# 6. MANAJEMEN USER (superadmin/admin)
# =========================================================
@app.get("/api/admin/users")
async def list_users(user=Depends(get_current_user)):
    require_role(user, ["superadmin", "admin"], "Melihat daftar user")
    return database.get_all_users()

@app.post("/api/admin/users")
async def create_user(request: UserCreateRequest, user=Depends(get_current_user)):
    require_role(user, ["superadmin", "admin"], "Menambah user")
    try:
        return database.add_user(request.username, request.password, request.role, request.client_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/admin/users/{user_id}")
async def remove_user(user_id: int, user=Depends(get_current_user)):
    require_role(user, ["superadmin", "admin"], "Menghapus user")
    database.delete_user(user_id)
    return {"message": "User berhasil dihapus."}


@app.post("/api/admin/client-instansi")
async def create_client_instansi(request: ClientInstansiCreateRequest, user=Depends(get_current_user)):
    require_role(user, ["superadmin", "admin"], "Membuat client instansi")
    try:
        # 1. Buat client baru
        client = database.add_client(request.instansi_name, request.client_type)
        # 2. Buat user admin_client baru yang terikat ke client tersebut
        new_user = database.add_user(
            username=request.username,
            password=request.password,
            role="admin_client",
            client_id=client["id"],
            email=request.email,
            password_changed=0
        )
        return {"user": new_user, "client": client}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/admin/client-instansi/{user_id}")
async def update_client_instansi_endpoint(user_id: int, request: ClientInstansiUpdateRequest, user=Depends(get_current_user)):
    require_role(user, ["superadmin", "admin"], "Mengubah client instansi")
    try:
        database.update_client_instansi(
            user_id=user_id,
            username=request.username,
            instansi_name=request.instansi_name,
            client_type=request.client_type,
            password=request.password
        )
        return {"status": "success", "message": "Client instansi berhasil diperbarui."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/admin/clients/{client_id}/generate-api-key")
async def generate_api_key_endpoint(client_id: int, user=Depends(get_current_user)):
    require_role(user, ["superadmin", "admin", "admin_client"], "Generate API Key")
    # admin_client cuma boleh reset API key miliknya sendiri
    if user["role"] == "admin_client":
        require_client_access(user, client_id)
        
    try:
        new_key = database.generate_client_api_key(client_id)
        return {"api_key": new_key}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@app.get("/api/health")
async def health():
    total_cache_items = sum(len(v) for v in rag.state.prompt_cache.values())
    clients = database.get_all_clients()
    per_client_stats = []
    for c in clients:
        cid = c["id"]
        bm25 = rag.state.bm25_indexes.get(cid)
        per_client_stats.append({
            "client_id": cid,
            "client_name": c["name"],
            "loaded_in_memory": cid in rag.state.retrievers,
            "bm25_indexed_chunks": len(bm25.doc_ids) if bm25 else 0,
            "documents": len(database.get_documents_by_client(cid)),
        })
    return {
        "status": "ok",
        "rag_ready": os.path.exists(rag.CHROMA_DIR) and bool(os.listdir(rag.CHROMA_DIR)),
        "clients": per_client_stats,
        "cache_size": total_cache_items,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
