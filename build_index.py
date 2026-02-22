from pathlib import Path
from tqdm import tqdm
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

PROCESSED_DIR = Path("finrag/data/processed")
INDEX_DIR = Path("finrag/index")

def load_docs():
    txt_files = sorted(PROCESSED_DIR.glob("*.txt"))
    if not txt_files:
        raise RuntimeError(f"No .txt files found in {PROCESSED_DIR.resolve()}")
    docs = []
    for f in tqdm(txt_files, desc="Loading text files"):
        text = f.read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(text=text, metadata={"source": f.name}))
    return docs

def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    Settings.llm = None

    splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=20)
    docs = load_docs()
    nodes = splitter.get_nodes_from_documents(docs)

    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))

    print(f"\n✅ Index built and saved to: {INDEX_DIR.resolve()}")
    print(f"📄 Nodes indexed: {len(nodes)}")

if __name__ == "__main__":
    main()
