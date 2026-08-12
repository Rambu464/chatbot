"""
stress_test_faithfulness.py -- Jalankan metrik Faithfulness N kali pada
response & context yang PERSIS SAMA (soal "pengadilan mana sengketa...",
yang dapat faithfulness=0.0), untuk membedakan dua kemungkinan:

  (a) Skor 0.0 KONSISTEN di semua run -> ini genuinely masalah generation
      (klaim response memang tidak cukup didukung konteks), bukan noise judge.
  (b) Skor BERVARIASI antar run -> ada ketidakstabilan judge gpt-4o untuk
      kasus abu-abu ini, perlu dicatat sebagai limitation di Bab 5.

response & retrieved_contexts diambil langsung dari ragas_results.csv kamu
(tidak perlu re-run pipeline RAG), jadi hanya membayar biaya Faithfulness
sebanyak N_RUNS kali untuk 1 baris -- murah.

Cara pakai (sejajar dengan eval_ragas.py):
    python stress_test_faithfulness.py
"""
import ast
import asyncio

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from ragas import SingleTurnSample
from ragas.metrics import Faithfulness

from eval_ragas import build_judge_llm, RESULTS_CSV_PATH


TARGET_QUESTION = "Ke pengadilan mana sengketa nasabah dan Bank Danamon diselesaikan jika penyelesaian sengketa lewat OJK tidak berhasil?"
N_RUNS = 5


async def main():
    df = pd.read_csv(RESULTS_CSV_PATH)
    matches = df[df["user_input"] == TARGET_QUESTION]
    if matches.empty:
        raise ValueError(f"Soal tidak ditemukan di {RESULTS_CSV_PATH}: {TARGET_QUESTION!r}")
    row = matches.iloc[0]

    contexts = ast.literal_eval(row["retrieved_contexts"])
    response = row["response"]

    print("Q       :", TARGET_QUESTION)
    print("Response:", response)
    print(f"\nMenjalankan Faithfulness {N_RUNS}x pada response & context yang PERSIS SAMA...\n")

    judge_llm = build_judge_llm("gpt-4o")
    metric = Faithfulness(llm=judge_llm)

    sample = SingleTurnSample(
        user_input=TARGET_QUESTION,
        response=response,
        retrieved_contexts=contexts,
    )

    scores = []
    for i in range(N_RUNS):
        score = await metric.single_turn_ascore(sample)
        scores.append(score)
        print(f"  Run {i + 1}: faithfulness = {score:.4f}")

    mean_s = sum(scores) / len(scores)
    print(f"\nMean : {mean_s:.4f}")
    print(f"Min  : {min(scores):.4f}")
    print(f"Max  : {max(scores):.4f}")

    if len(set(round(s, 4) for s in scores)) == 1:
        print("\n-> KONSISTEN di semua run: skor 0.0 ini genuine masalah generation")
        print("   (klaim response memang belum cukup didukung konteks), bukan noise judge.")
        print("   Perbaikan harus di sisi prompt generation, bukan di evaluasi.")
    else:
        print("\n-> BERVARIASI antar run: ada ketidakstabilan judge gpt-4o untuk kasus")
        print("   abu-abu seperti ini walau temperature=0. Catat sebagai limitation metode.")


if __name__ == "__main__":
    asyncio.run(main())
