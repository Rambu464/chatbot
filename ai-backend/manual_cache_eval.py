"""
manual_cache_eval.py -- Evaluasi Semantic Prompt Cache menggunakan soal uji
BUATAN SENDIRI (manual_test_questions.json), bukan hasil generate LLM.

Kenapa ganti dari cache_eval.py (versi gpt-4o)?
- Soal uji jadi murni instrumen penelitian kamu sendiri, bukan produk LLM
  lain -- lebih independen dan gampang dipertanggungjawabkan saat sidang
  ("kenapa soal ini yang dipakai?" -> "saya yang tulis, ini alasannya...").
- Gratis, tidak butuh OPENAI_API_KEY sama sekali.
- Hasilnya 100% reprodusibel -- dijalankan berkali-kali hasilnya identik,
  tidak tergantung sampling LLM.

Cara kerja skor:
  Untuk tiap soal di manual_test_questions.json, sistem dites apakah
  keputusan cache-nya (hit ke sumber yang benar / miss) SESUAI dengan
  label "expected" yang kamu tulis sendiri. Skor per soal: 1 (benar) atau
  0 (salah). Skor akhir = total skor dibagi jumlah soal (rata-rata),
  dilaporkan sebagai persentase akurasi keseluruhan, plus breakdown per
  "category" kalau kamu isi field itu (mis. "paraphrase" vs "hard_negative").

Cara pakai (sejajar eval_ragas.py, di ai-backend/):
    1. Edit manual_test_questions.json -- isi soal-soal buatanmu sendiri.
    2. python manual_cache_eval.py
"""
import asyncio
import json

from dotenv import load_dotenv
load_dotenv()

from rag_modules import cache, generation
from rag_modules.state import state
import database

CLIENT_ID = 6
TEST_DOCUMENT_MARKER = "__manual_cache_eval_test__ syartum rek perbankan.pdf"  # terisolasi dari produksi
GOLDEN_DATASET_PATH = "golden_dataset.json"
MANUAL_TEST_PATH = "manual_test_questions.json"


async def main():
    print("Inisialisasi embeddings...")
    await generation.initialize()

    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        golden_items = json.load(f)

    with open(MANUAL_TEST_PATH, encoding="utf-8") as f:
        raw_cases = json.load(f)
    test_cases = [c for c in raw_cases if "test_question" in c]
    skipped = len(raw_cases) - len(test_cases)
    if skipped:
        print(f"({skipped} entri di manual_test_questions.json dilewati -- bukan soal uji, mis. blok _INSTRUKSI)")

    if not test_cases:
        print("Tidak ada soal uji ditemukan di manual_test_questions.json. Isi dulu filenya.")
        return

    cache_key = cache.make_cache_key(CLIENT_ID, TEST_DOCUMENT_MARKER, general_mode=False)
    print(f"Cache key (terisolasi dari produksi): {cache_key}\n")

    # Bersihkan dulu supaya tiap run mulai dari kondisi bersih
    state.prompt_cache.pop(cache_key, None)
    await asyncio.to_thread(database.delete_cache_entries_by_prefix, cache_key)

    # Isi cache dengan 14 soal golden dataset (simulasi: sudah pernah ditanya)
    for item in golden_items:
        q_lower = item["question"].lower().strip()
        emb = state.embeddings.embed_query(q_lower)
        await cache.store_cache(cache_key, q_lower, emb, response=item["reference"])
    print(f"Cache terisi dengan {len(golden_items)} soal golden dataset.\n")

    print("=" * 100)
    print("EVALUASI SOAL UJI BUATAN SENDIRI")
    print("=" * 100)

    results = []
    for i, case in enumerate(test_cases):
        test_q = case["test_question"]
        expected = case["expected"]  # "hit" atau "miss"
        src_idx = case.get("source_question_index")
        category = case.get("category", "-")

        q_lower = test_q.lower().strip()
        emb = state.embeddings.embed_query(q_lower)
        match, score = await cache.check_cache(cache_key, q_lower, emb)

        if match is None:
            actual = "miss"
            target_ok = True  # tidak relevan untuk kasus miss
        else:
            actual = "hit"
            expected_source_q = golden_items[src_idx]["question"].lower().strip() if src_idx is not None else None
            target_ok = (match["query"] == expected_source_q)

        correct = (actual == expected) and (expected != "hit" or target_ok)
        results.append({"case": case, "actual": actual, "score_sim": score, "correct": correct})

        mark = "BENAR" if correct else "SALAH"
        detail = ""
        if actual == "hit" and not target_ok:
            detail = " (hit tapi ke sumber yang KELIRU)"
        print(f"[{i+1:2d}] [{mark}] expected={expected:4s} actual={actual:4s} sim={score:.4f} "
              f"cat={category:15s} | {test_q}{detail}")

    total = len(results)
    total_correct = sum(r["correct"] for r in results)
    accuracy = total_correct / total

    print("\n" + "=" * 100)
    print("RINGKASAN")
    print("=" * 100)
    print(f"Total soal uji     : {total}")
    print(f"Jumlah benar       : {total_correct}")
    print(f"Skor akhir (rata-rata / akurasi) : {accuracy:.1%}")

    # Breakdown per category, kalau diisi
    categories = sorted(set(r["case"].get("category", "-") for r in results))
    if categories != ["-"]:
        print("\nBreakdown per kategori:")
        for cat in categories:
            in_cat = [r for r in results if r["case"].get("category", "-") == cat]
            cat_correct = sum(r["correct"] for r in in_cat)
            print(f"  {cat:20s} : {cat_correct}/{len(in_cat)} = {cat_correct/len(in_cat):.1%}")


if __name__ == "__main__":
    asyncio.run(main())
