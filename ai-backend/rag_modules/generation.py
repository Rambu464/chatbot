"""
generation.py -- Inisialisasi model, penyusunan prompt multi-turn, dan generator streaming response.
"""
import os
import asyncio
import threading
from typing import List, Optional

from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings

from rag_modules.config import (
    MODEL_PATH,
    N_CTX,
    N_THREADS,
    SYSTEM_PROMPT_RAG_STRICT,
    SYSTEM_PROMPT_RAG_FLEXIBLE,
    SYSTEM_PROMPT_CHAT,
)
from rag_modules.state import state, RAGState


def select_system_prompt(is_rag_mode: bool, general_mode: bool, client_name: Optional[str] = None) -> str:
    """Memilih dan mendinamisasi system prompt yang sesuai berdasarkan mode RAG, general mode, dan nama client."""
    if is_rag_mode:
        base_template = SYSTEM_PROMPT_RAG_FLEXIBLE if general_mode else SYSTEM_PROMPT_RAG_STRICT
    else:
        base_template = SYSTEM_PROMPT_CHAT

    client_info = f" dari {client_name}" if client_name else ""
    client_intro = f" dari {client_name}" if client_name else ""

    return base_template.format(client_info=client_info, client_intro=client_intro)



def build_prompt(
    system_prompt: str,
    user_message: str,
    context: str = "",
    history: Optional[List[dict]] = None,
) -> str:
    """Menyusun struktur prompt berformat ChatML lengkap dengan riwayat percakapan multi-turn."""
    if context:
        user_block = f"[DOKUMEN]:\n{context}\n\n[PERTANYAAN]:\n{user_message}"
    else:
        user_block = user_message

    parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>\n"]
    for turn in (history or []):
        role = turn.get("role")
        content = turn.get("content", "")
        if role not in ("user", "assistant") or not content:
            continue
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{user_block}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")

    return "".join(parts)


async def stream_llm(prompt: str, stop_event: Optional[threading.Event] = None):
    """Menjalankan streaming token LLM secara non-blocking di thread terpisah."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    SENTINEL = object()

    def worker():
        with state.inference_lock:
            try:
                for token in state.llm.stream(prompt):
                    if stop_event is not None and stop_event.is_set():
                        break
                    asyncio.run_coroutine_threadsafe(queue.put(token), loop).result()
            except Exception as e:
                asyncio.run_coroutine_threadsafe(queue.put(e), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(SENTINEL), loop).result()

    threading.Thread(target=worker, daemon=True).start()

    while True:
        item = await queue.get()
        if item is SENTINEL:
            break
        if isinstance(item, Exception):
            raise item
        yield item


async def initialize() -> RAGState:
    """Inisialisasi model Embeddings HuggingFace & LLM LlamaCpp saat backend startup."""
    state.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model GGUF tidak ditemukan di '{MODEL_PATH}'.")

    state.llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=int(os.getenv("N_GPU_LAYERS", "-1")),
        n_batch=512,
        temperature=0.2,
        repeat_penalty=1.3,
        frequency_penalty=0.3,
        presence_penalty=0.1,
        last_n_tokens_size=512,
        max_tokens=1024,
        stop=["<|im_end|>", "<|im_start|>", "<think>"],
        streaming=True,
        verbose=os.getenv("VERBOSE_LLM", "false").lower() == "true",
    )

    state.retrievers = {}
    state.bm25_indexes = {}
    state.prompt_cache = {}
    state.cache_lock = asyncio.Lock()
    state.inference_lock = threading.Lock()

    return state
