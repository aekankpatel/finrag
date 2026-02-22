import gdown
from pathlib import Path

FOLDER_ID = "1NiAUNTdyA6OX5V1ZVxoT1KrgOKk2kEQs"
INDEX_DIR = Path("finrag/index")

def ensure_index():
    if not INDEX_DIR.exists() or not (INDEX_DIR / "docstore.json").exists():
        print("📥 Downloading index from Google Drive...")
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        gdown.download_folder(
            id=FOLDER_ID,
            output=str(INDEX_DIR),
            quiet=False,
        )
        print("✅ Index downloaded.")
    else:
        print("✅ Index already exists locally.")

if __name__ == "__main__":
    ensure_index()
