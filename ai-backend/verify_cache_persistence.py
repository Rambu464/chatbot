"""
verify_cache_persistence.py -- Bukti empiris bahwa cache SURVIVE restart
proses, bukan cuma diklaim. Dijalankan sebagai 2 proses Python TERPISAH
(bukan 1 proses yang loop) -- ini penting, karena tujuannya membuktikan data
selamat walau proses lama sudah benar-benar mati (persis simulasi main.py
di-restart).

CARA PAKAI (2 langkah, di 2 pemanggilan `python` yang terpisah):

    # Langkah 1 -- proses A: isi 1 entri cache, lalu proses ini KELUAR
    python verify_cache_persistence.py populate

    # Langkah 2 -- proses B: proses BARU, tidak tahu-menahu soal proses A.
    # Hanya panggil generation.initialize() (yang sekarang hydrate dari
    # SQLite) lalu langsung check_cache() TANPA store_cache() lagi.
    python verify_cache_persistence.py check

Kalau langkah 2 melaporkan HIT, itu bukti valid: data cache ditulis proses A,
proses A mati total, proses B baru baca ulang dari SQLite dan berhasil
match -- bukan cuma "kebetulan masih di memori" karena memang python
process-nya sudah beda PID.
"""
import asyncio
import sys

from dotenv import load_dotenv
load_dotenv()

from rag_modules import cache, generation
from rag_modules.state import state

CACHE_KEY = cache.make_cache_key(client_id=999, document="verify_test.pdf", general_mode=False)
TEST_QUERY = "apa itu tes persistensi cache?"
TEST_RESPONSE = "Ini respons dummy untuk verifikasi persistensi cache lintas restart proses."


async def populate():
    print(f"[PID {__import__('os').getpid()}] Inisialisasi (proses A)...")
    await generation.initialize()

    emb = state.embeddings.embed_query(TEST_QUERY)
    await cache.store_cache(CACHE_KEY, TEST_QUERY, emb, TEST_RESPONSE)
    print(f"[PID {__import__('os').getpid()}] Entri cache disimpan ke SQLite untuk key={CACHE_KEY!r}.")
    print("Sekarang KELUAR dari proses ini (simulasi main.py mati), lalu jalankan:")
    print("    python verify_cache_persistence.py check")


async def check():
    print(f"[PID {__import__('os').getpid()}] Inisialisasi (proses B, TIDAK tahu soal proses A)...")
    await generation.initialize()  # <- ini yang hydrate dari SQLite kalau patch sudah diterapkan

    total_in_memory = sum(len(v) for v in state.prompt_cache.values())
    print(f"Total entri ter-load ke memori saat startup: {total_in_memory}")

    emb = state.embeddings.embed_query(TEST_QUERY)
    match, score = await cache.check_cache(CACHE_KEY, TEST_QUERY, emb)

    print("\n" + "=" * 70)
    if match is not None and match["response"] == TEST_RESPONSE:
        print("HASIL: HIT -- PERSISTENCE BEKERJA.")
        print(f"  score={score:.4f}, response cocok dengan yang disimpan proses A.")
    elif total_in_memory == 0:
        print("HASIL: MISS -- state.prompt_cache kosong saat startup.")
        print("  Kemungkinan besar patch generation.py (load_persisted_cache) belum diterapkan,")
        print("  atau proses 'populate' belum pernah dijalankan sebelum ini.")
    else:
        print("HASIL: MISS -- ada entri ter-load, tapi tidak match query test ini.")
        print("  Cek apakah CACHE_KEY/embedding cocok, atau threshold terlalu ketat.")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("populate", "check"):
        print("Pakai: python verify_cache_persistence.py [populate|check]")
        sys.exit(1)

    asyncio.run(populate() if sys.argv[1] == "populate" else check())
