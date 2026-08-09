"""
diagnose_pdf_extraction.py -- Bandingkan hasil ekstraksi teks PyMuPDF (dipakai di
ingestion.py) vs pdfplumber, khusus di halaman yang mengandung klausul soal #12
(pengaduan, kerugian materiil) dan #13 (pengadilan/yurisdiksi), untuk konfirmasi
apakah PyMuPDF salah decode font pada PDF ini.

Cara pakai:
    pip install pdfplumber --break-system-packages
    python diagnose_pdf_extraction.py "path/ke/syartum rek perbankan.pdf"
"""
import sys

import fitz  # PyMuPDF, sama seperti yang dipakai ingestion.py
import pdfplumber

KEYWORDS_TO_FIND = ["kerugian materiil", "Pengadilan Negeri", "klarifikasi", "yurisdiksi"]


def extract_pymupdf(path: str) -> list[str]:
    doc = fitz.open(path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return pages


def extract_pdfplumber(path: str) -> list[str]:
    with pdfplumber.open(path) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]


def find_pages_with_keyword_fragment(pages: list[str], fragment_lower: str) -> list[int]:
    """Cari halaman yang MUNGKIN mengandung keyword, dengan fuzzy match longgar
    (cek kemunculan sebagian huruf awal saja) karena PyMuPDF bisa jadi korup."""
    hits = []
    for i, text in enumerate(pages, start=1):
        if fragment_lower[:4] in text.lower():  # cocokkan 4 huruf pertama saja, longgar
            hits.append(i)
    return hits


def dump_page_range(pages: list[str], start: int, end: int, label: str):
    """Cetak mentah isi halaman start..end (1-indexed, inklusif) -- dipakai kalau
    pencarian keyword persis gagal, supaya bisa lihat wording asli dokumennya."""
    print(f"\n=== DUMP HALAMAN {start}-{end} ({label}) ===")
    for i in range(start, end + 1):
        if 1 <= i <= len(pages):
            print(f"\n--- Halaman {i} ---")
            print(pages[i - 1])
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_pdf_extraction.py <path_ke_pdf> [--dump START END]")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Membaca: {path}\n")

    pymupdf_pages = extract_pymupdf(path)
    plumber_pages = extract_pdfplumber(path)

    print(f"Jumlah halaman -- PyMuPDF: {len(pymupdf_pages)}, pdfplumber: {len(plumber_pages)}\n")

    # Mode dump manual: python diagnose_pdf_extraction.py file.pdf --dump 28 31
    if len(sys.argv) >= 4 and sys.argv[2] == "--dump":
        start, end = int(sys.argv[3]), int(sys.argv[4])
        dump_page_range(plumber_pages, start, end, "pdfplumber")
        return

    for keyword in ["Pengadilan Negeri", "kerugian materiil", "menolak menangani", "PENGAJUAN PENGADUAN"]:
        print(f"=== Mencari halaman yang mengandung: {keyword!r} ===")

        found = False
        # Cari via pdfplumber dulu (asumsi lebih akurat) untuk tahu halaman targetnya
        for i, text in enumerate(plumber_pages, start=1):
            if keyword.lower() in text.lower():
                found = True
                print(f"\n--- Halaman {i} (pdfplumber, BERSIH) ---")
                idx = text.lower().find(keyword.lower())
                print(text[max(0, idx - 150):idx + 150])

                print(f"\n--- Halaman {i} (PyMuPDF, dipakai ingestion.py) ---")
                pymupdf_text = pymupdf_pages[i - 1] if i - 1 < len(pymupdf_pages) else ""
                if keyword.lower() in pymupdf_text.lower():
                    idx2 = pymupdf_text.lower().find(keyword.lower())
                    print(pymupdf_text[max(0, idx2 - 150):idx2 + 150])
                    print("\n[HASIL: PyMuPDF berhasil menemukan teks ini dengan benar]")
                else:
                    print(pymupdf_text[:300] + " ...")
                    print(f"\n[HASIL: PyMuPDF GAGAL menemukan {keyword!r} secara utuh di halaman ini -- kemungkinan corrupt]")
                break

        if not found:
            print(f"[TIDAK DITEMUKAN sama sekali di pdfplumber -- kemungkinan wording beda atau klausul tidak ada]")
        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
