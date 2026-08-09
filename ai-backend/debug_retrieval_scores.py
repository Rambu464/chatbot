"""
debug_retrieval_scores.py -- Diagnostik retrieval MURNI, tanpa LLM/RAGAS/Groq sama
sekali (jadi TIDAK makan kuota apapun). Menghitung skor cosine similarity & BM25
untuk SEMUA child chunk di ChromaDB (bukan cuma top-RETRIEVER_K), supaya kita bisa
lihat persis di peringkat berapa chunk yang relevan berada -- meski dia gagal masuk
kandidat akhir.

Cara pakai (dari folder ai-backend, venv aktif):
    python debug_retrieval_scores.py "Apakah Bank bertanggung jawab jika layanan terganggu karena bencana alam?"

Bisa juga cari manual pakai keyword tertentu untuk nemuin chunk_id target:
    python debug_retrieval_scores.py --find "FORCE MAJEURE"
"""
import sys
import asyncio
import math

sys.path.insert(0, ".")

from rag_modules import generation, retrieval
from rag_modules.state import state, get_client_state
from rag_modules.config import DENSE_SCORE_THRESHOLD, RETRIEVER_K, TOP_N_PARENTS

CLIENT_ID = 6  # ganti kalau client_id Danamon kamu beda


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


async def main():
    print("Inisialisasi embeddings (tanpa load LLM GGUF, biar cepat)...")
    from langchain_huggingface import HuggingFaceEmbeddings
    state.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    retriever, bm25_index = get_client_state(CLIENT_ID)

    if len(sys.argv) >= 3 and sys.argv[1] == "--find":
        keyword = sys.argv[2]
        print(f"\nMencari child chunk yang mengandung: {keyword!r}\n")
        collection = retriever.vectorstore._collection
        data = collection.get(include=["metadatas", "documents"])
        for doc, meta in zip(data["documents"], data["metadatas"]):
            if keyword.lower() in doc.lower():
                print(f"--- chunk_id={meta.get(retriever.id_key)} (source={meta.get('source')}) ---")
                print(doc[:300])
                print()
        return

    query = sys.argv[1] if len(sys.argv) > 1 else "Apakah Bank bertanggung jawab jika layanan terganggu karena bencana alam?"
    print(f"\nQuery: {query!r}\n")

    query_embedding = state.embeddings.embed_query(query.lower())

    # --- Replikasi PERSIS logika retrieve_parent_docs, tapi dengan skor RRF
    # ditampilkan untuk SEMUA kandidat gabungan -- supaya kelihatan persis di
    # peringkat berapa chunk target berada setelah fusion, dan seberapa jauh
    # selisihnya dari TOP_N_PARENTS. ---
    id_key = retriever.id_key
    collection = retriever.vectorstore._collection
    query_kwargs = {"query_embeddings": [query_embedding], "n_results": RETRIEVER_K, "include": ["metadatas", "embeddings"]}
    dense_results = collection.query(**query_kwargs)

    dense_ids = []
    if dense_results["metadatas"]:
        for meta, emb in zip(dense_results["metadatas"][0], dense_results["embeddings"][0]):
            score = cosine(query_embedding, emb)
            if score < DENSE_SCORE_THRESHOLD:
                continue
            doc_id = meta.get(id_key)
            if doc_id and doc_id not in dense_ids:
                dense_ids.append(doc_id)

    bm25_ids = bm25_index.search(query.lower(), k=RETRIEVER_K)

    print(f"=== dense_ids (setelah dedup per PARENT id): {len(dense_ids)} unique parent ===")
    for i, did in enumerate(dense_ids):
        print(f"  #{i}  parent_id={did}")
    print(f"\n=== bm25_ids: {len(bm25_ids)} ===")
    for i, did in enumerate(bm25_ids):
        print(f"  #{i}  parent_id={did}")

    rrf_scores = {}
    for rank, did in enumerate(dense_ids):
        rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (60 + rank + 1)
    for rank, did in enumerate(bm25_ids):
        rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (60 + rank + 1)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n=== RANKING RRF GABUNGAN (top {TOP_N_PARENTS} akan lolos ke context) ===")
    TARGET_ID = "6454d1a7-e78f-429b-b537-96d795044e64"
    for i, (did, score) in enumerate(ranked, start=1):
        marker = "  <-- TARGET CHUNK FORCE MAJEURE" if did == TARGET_ID else ""
        cutoff = " [LOLOS TOP_N_PARENTS]" if i <= TOP_N_PARENTS else ""
        print(f"#{i:2d}  rrf_score={score:.5f}  parent_id={did}{cutoff}{marker}")

    collection = retriever.vectorstore._collection
    all_data = collection.get(include=["metadatas", "documents", "embeddings"])

    print(f"Total child chunk di ChromaDB untuk client_id={CLIENT_ID}: {len(all_data['documents'])}\n")

    # Hitung cosine similarity utk SEMUA child chunk, bukan cuma top-K
    scored = []
    for doc, meta, emb in zip(all_data["documents"], all_data["metadatas"], all_data["embeddings"]):
        score = cosine(query_embedding, emb)
        scored.append((score, doc, meta))
    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"=== TOP 15 DENSE (dari total {len(scored)} chunk, threshold config saat ini: {DENSE_SCORE_THRESHOLD}) ===")
    for rank, (score, doc, meta) in enumerate(scored[:15], start=1):
        above = "LOLOS" if score >= DENSE_SCORE_THRESHOLD else "DIBUANG (di bawah threshold)"
        preview = doc[:80].replace("\n", " ")
        print(f"#{rank:2d}  score={score:.4f}  [{above}]  {preview}")

    # Cari spesifik di mana peringkat chunk yang mengandung "FORCE MAJEURE" / "bencana alam"
    print(f"\n=== POSISI CHUNK YANG MENGANDUNG 'FORCE MAJEURE' atau 'bencana alam' ===")
    found_any = False
    for rank, (score, doc, meta) in enumerate(scored, start=1):
        if "force majeure" in doc.lower() or "bencana alam" in doc.lower():
            found_any = True
            above = "LOLOS threshold" if score >= DENSE_SCORE_THRESHOLD else "DI BAWAH THRESHOLD"
            in_topk = "MASUK top-RETRIEVER_K" if rank <= RETRIEVER_K else "TIDAK masuk top-RETRIEVER_K"
            print(f"Peringkat #{rank} dari {len(scored)} total -- score={score:.4f} -- {above} -- {in_topk}")
            print(f"Isi: {doc[:200]}")
            print()
    if not found_any:
        print("[TIDAK ADA satupun chunk yang mengandung kata 'FORCE MAJEURE' atau 'bencana alam' -- kemungkinan besar teks ini tidak ter-index sama sekali di ChromaDB, bukan cuma kalah skor]")

    # Cek juga hasil BM25 utk query yang sama
    print(f"\n=== TOP 10 BM25 ===")
    bm25_results = bm25_index.search(query.lower(), k=10)
    for rank, doc_id in enumerate(bm25_results, start=1):
        print(f"#{rank}  chunk_id={doc_id}")


if __name__ == "__main__":
    asyncio.run(main())
