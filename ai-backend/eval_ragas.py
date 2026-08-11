import os
import sys
import json
import asyncio
import argparse

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from rag_modules import config, generation, retrieval
from rag_modules.state import state, get_client_state

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig


GOLDEN_DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
RESULTS_CSV_PATH = os.path.join(os.path.dirname(__file__), "ragas_results.csv")
METRIC_COLUMNS = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall"]


# ---------------------------------------------------------------------------
# 0. Progress tracking -- resume jika ada yang terputus di tengah jalan
# ---------------------------------------------------------------------------

def load_existing_results():
    if os.path.exists(RESULTS_CSV_PATH):
        return pd.read_csv(RESULTS_CSV_PATH)
    return None


def is_fully_valid(row) -> bool:
    return all(pd.notna(row.get(col)) for col in METRIC_COLUMNS)


def filter_pending_items(golden_items, existing_df, force: bool):
    if force or existing_df is None:
        return golden_items

    done_questions = {
        row["user_input"] for _, row in existing_df.iterrows() if is_fully_valid(row)
    }
    pending = [item for item in golden_items if item["question"] not in done_questions]

    n_skipped = len(golden_items) - len(pending)
    if n_skipped:
        print(f"[RESUME] {n_skipped} soal sudah punya skor valid dari run sebelumnya -- DI-SKIP.")
    return pending


def merge_and_save(new_df, existing_df):
    if existing_df is None:
        merged = new_df
    else:
        new_questions = set(new_df["user_input"])
        kept_old = existing_df[~existing_df["user_input"].isin(new_questions)]
        merged = pd.concat([kept_old, new_df], ignore_index=True)

    merged.to_csv(RESULTS_CSV_PATH, index=False)
    return merged


# ---------------------------------------------------------------------------
# 1. Jalankan pipeline RAG produksi untuk tiap pertanyaan golden dataset
# ---------------------------------------------------------------------------

def run_pipeline_for_question(client_id, question, document):
    question_lower = question.lower().strip()
    query_embedding = state.embeddings.embed_query(question_lower)

    parent_docs = retrieval.retrieve_parent_docs(
        client_id=client_id,
        query_text=question_lower,
        query_embedding=query_embedding,
        source_filter=document,
    )
    contexts = [d.page_content for d in parent_docs] or [""]
    context_text = retrieval.format_context(parent_docs) if parent_docs else ""
    is_rag_mode = bool(parent_docs)

    system_prompt = generation.select_system_prompt(is_rag_mode=is_rag_mode, general_mode=False)
    prompt = generation.build_prompt(system_prompt, question, context=context_text, use_few_shot=True)

    answer = state.llm.invoke(prompt)
    return contexts, answer.strip()


async def build_evaluation_dataset(golden_items):
    samples = []
    for item in golden_items:
        client_id = item["client_id"]
        get_client_state(client_id)

        contexts, answer = run_pipeline_for_question(
            client_id=client_id,
            question=item["question"],
            document=item.get("document"),
        )
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                retrieved_contexts=contexts,
                response=answer,
                reference=item["reference"],
            )
        )
        print(f"[OK] {item['question'][:60]!r} -> {len(contexts)} chunks, jawaban {len(answer)} char")

    return EvaluationDataset(samples=samples)


# ---------------------------------------------------------------------------
# 2. Judge LLM (Konfigurasi OpenAI / Google AI Studio)
# ---------------------------------------------------------------------------

def build_judge_llm(judge_model: str):
    if judge_model.startswith("gpt-"):
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
        llm = ChatOpenAI(model=judge_model, api_key=os.environ["OPENAI_API_KEY"], temperature=0)
        return LangchainLLMWrapper(llm)
    else:
        from google import genai as google_genai
        client = google_genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        return llm_factory(judge_model, provider="google", client=client, max_tokens=4096)


def build_judge_embeddings():
    return LangchainEmbeddingsWrapper(state.embeddings)


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser()
    # Menunjuk ke gpt-4o-mini yang super cepat, akurat, dan hemat untuk RAGAS
    parser.add_argument("--judge-model", default="gpt-4o-mini",
                         help="Model judge (misal: gpt-4o-mini, gpt-4o, gemini-1.5-flash)")
    parser.add_argument("--force", action="store_true",
                         help="Hitung ulang SEMUA soal")
    parser.add_argument("--limit", type=int, default=None,
                         help="Batasi jumlah soal yang diproses")
    args = parser.parse_args()

    print("Inisialisasi model embeddings & LLM backend...")
    await generation.initialize()

    from database import get_all_clients, get_documents_by_client

    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        golden_items = json.load(f)

    used_client_ids = {item["client_id"] for item in golden_items}
    all_clients = {c["id"]: c for c in get_all_clients()}

    print("\n=== CEK CLIENT SEBELUM EVAL ===")
    for cid in sorted(used_client_ids):
        client = all_clients.get(cid)
        if client is None:
            raise ValueError(f"client_id={cid} tidak ditemukan.")
        docs = get_documents_by_client(cid)
        doc_names = [d["filename"] for d in docs]
        print(f"client_id={cid} -> {client['name']} ({client['type']}), dokumen: {doc_names}")
    print("================================\n")

    existing_df = load_existing_results()
    pending_items = filter_pending_items(golden_items, existing_df, force=args.force)

    if args.limit is not None:
        pending_items = pending_items[: args.limit]

    if not pending_items:
        print("\nSemua soal sudah punya skor valid. Selesai!")
        return

    print(f"Judge model: {args.judge_model}")
    print(f"Menjalankan pipeline RAG produksi untuk {len(pending_items)} soal...")
    dataset = await build_evaluation_dataset(pending_items)

    print("Menjalankan evaluasi RAGAS (dengan proteksi Rate-Limit)...")
    judge_llm = build_judge_llm(args.judge_model)
    judge_embeddings = build_judge_embeddings()

    # KUNCI ANTI 429: max_workers=1 agar sekuensial + max_retries ditambah agar kuat menunggu jika terkena hit rate-limit
    run_config = RunConfig(
        max_workers=1, 
        timeout=300, 
        max_retries=20
    )

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]

    # Karena skema internal Ragas memanggil asinkronus, kita bungkus eksekusinya dengan kontrol jeda waktu manual
    # Jika versi Ragas Anda mendukung parameter internal sleep_time, objek run_config otomatis memanfaatkannya.
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )

    new_df = result.to_pandas()
    merged_df = merge_and_save(new_df, existing_df)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_lines = [
        f"=== RINGKASAN EVALUASI RAGAS -- {timestamp} ===",
        f"Judge model: {args.judge_model}",
        f"Total dataset: {len(golden_items)} | Diproses: {len(pending_items)}",
        "",
    ]
    for metric in METRIC_COLUMNS:
        if metric in merged_df.columns:
            n_valid = merged_df[metric].notna().sum()
            n_total = len(merged_df)
            mean_val = merged_df[metric].mean()
            summary_lines.append(f"{metric:45s}: {mean_val:.3f}  (valid: {n_valid}/{n_total})")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = os.path.join(os.path.dirname(__file__), f"ragas_summary_{timestamp}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    print(f"\nRingkasan tersimpan di : {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())