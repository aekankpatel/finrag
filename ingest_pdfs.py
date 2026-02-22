import os
import re
from pathlib import Path
import fitz  # PyMuPDF
from tqdm import tqdm

RAW_DIR = Path("finrag/data/raw")
OUT_DIR = Path("finrag/data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(t: str) -> str:
    # Normalize whitespace
    t = t.replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def extract_pdf_text(pdf_path: Path) -> tuple[str, int]:
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        # "text" works well for most filings/transcripts
        pages.append(page.get_text("text"))
    doc.close()
    full = "\n\n".join(pages)
    return clean_text(full), len(pages)

def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {RAW_DIR.resolve()}")

    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {RAW_DIR.resolve()}")

    manifest_lines = ["file,pages,chars,output_txt"]
    for pdf in tqdm(pdfs, desc="Extracting PDFs"):
        text, n_pages = extract_pdf_text(pdf)
        out_txt = OUT_DIR / f"{pdf.stem}.txt"
        out_txt.write_text(text, encoding="utf-8")

        manifest_lines.append(
            f"{pdf.name},{n_pages},{len(text)},{out_txt.name}"
        )

    (OUT_DIR / "manifest.csv").write_text("\n".join(manifest_lines), encoding="utf-8")
    print(f"\n✅ Done. Extracted {len(pdfs)} PDFs → {OUT_DIR.resolve()}")
    print(f"📄 Manifest: {(OUT_DIR / 'manifest.csv').resolve()}")

if __name__ == "__main__":
    main()