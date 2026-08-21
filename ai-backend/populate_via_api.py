"""
populate_via_api.py -- Input 14 soal golden dataset lewat endpoint /api/chat
ASLI (bukan simulasi/pipeline langsung), biar cache ke-populate persis
seperti alur produksi sungguhan.

PENTING: cache di main.py cuma dicek kalau history sesi kosong (pesan
pertama). Makanya skrip ini bikin 1 SESI BARU per soal -- bukan 14 soal
dalam 1 sesi yang sama (kalau gitu cache nggak akan pernah aktif dari
soal ke-2 dst).

Dua mode:
  python populate_via_api.py populate
      -> kirim 14 soal asli, masing-masing sesi baru. Cache masih kosong,
         jadi semua MISS (mengisi cache untuk pertama kali).
  python populate_via_api.py recheck
      -> kirim 14 soal LAGI (sesi baru lagi per soal). Kalau cache masih
         hidup (baik proses sama atau setelah restart main.py), harusnya
         semua HIT -- ditandai teks "Diambil dari Semantic Cache" di respons.

Jalankan 'populate' -> restart main.py -> jalankan 'recheck' untuk bukti
paling meyakinkan bahwa persistence bekerja di jalur produksi asli.

Butuh: `pip install requests --break-system-packages` (kalau belum ada),
server main.py sudah jalan di BASE_URL.
"""
import json
import sys
import time

import requests

BASE_URL = "http://localhost:8000"
USERNAME = "danamon"
PASSWORD = "Danamon123*"
CLIENT_ID = 6
DOCUMENT = "syartum rek perbankan.pdf"
GOLDEN_DATASET_PATH = "golden_dataset.json"


def login() -> str:
    resp = requests.post(f"{BASE_URL}/api/auth/login", json={"username": USERNAME, "password": PASSWORD})
    resp.raise_for_status()
    token = resp.json()["token"]
    print(f"Login sukses sebagai {USERNAME}, token diperoleh.")
    return token


def new_session(token: str, title: str) -> str:
    resp = requests.post(
        f"{BASE_URL}/api/chat/sessions",
        headers={"Authorization": f"Bearer {token}"},
        json={"title": title, "client_id": CLIENT_ID},
    )
    resp.raise_for_status()
    return resp.json()["id"]


def send_message(token: str, session_id: str, message: str) -> str:
    resp = requests.post(
        f"{BASE_URL}/api/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={"message": message, "session_id": session_id, "document": DOCUMENT, "general_mode": False},
        stream=True,
    )
    resp.raise_for_status()
    return "".join(chunk.decode("utf-8", errors="ignore") for chunk in resp.iter_content(chunk_size=None))


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("populate", "recheck"):
        print("Pakai: python populate_via_api.py [populate|recheck]")
        sys.exit(1)

    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        golden_items = json.load(f)

    token = login()

    n_hit, n_miss = 0, 0
    for i, item in enumerate(golden_items):
        session_id = new_session(token, title=f"[TEST] Soal {i+1}")
        full_response = send_message(token, session_id, item["question"])

        is_cache_hit = "Diambil dari Semantic Cache" in full_response
        n_hit += is_cache_hit
        n_miss += not is_cache_hit
        tag = "HIT " if is_cache_hit else "MISS"
        print(f"[{i+1:2d}] {tag} | {item['question'][:70]}")
        time.sleep(0.3)  # jangan spam server terlalu cepat

    print(f"\nTotal: {n_hit} HIT, {n_miss} MISS dari {len(golden_items)} soal.")
    if sys.argv[1] == "populate":
        print("-> Wajar semua MISS (cache baru pertama kali diisi).")
        print("   Restart main.py sekarang, lalu jalankan: python populate_via_api.py recheck")
    else:
        if n_hit == len(golden_items):
            print("-> SEMUA HIT. Persistence lintas restart TERBUKTI di jalur produksi asli.")
        else:
            print("-> Ada yang MISS. Cek apakah main.py memang sudah di-restart dengan patch")
            print("   generation.py (load_persisted_cache) diterapkan.")


if __name__ == "__main__":
    main()
