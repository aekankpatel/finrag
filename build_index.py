from pathlib import Path
from pydantic import Field
from tqdm import tqdm
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.ollama import OllamaEmbedding

PROCESSED_DIR = Path("finrag/data/processed")
INDEX_DIR = Path("finrag/index")


class SafeOllamaEmbedding(OllamaEmbedding):
    max_chars: int = Field(default=3000)

    def _truncate(self, text: str) -> str:
        if text is None:
            return ""
        return text.strip()[: self.max_chars]

    def _get_text_embedding(self, text: str):
        return super()._get_text_embedding(self._truncate(text))

    def _get_text_embeddings(self, texts):
        return super()._get_text_embeddings([self._truncate(t) for t in texts])


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

    embed_model = SafeOllamaEmbedding(model_name="nomic-embed-text", max_chars=3000)
    Settings.embed_model = embed_model
    Settings.llm = None

    splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=20)

    docs = load_docs()
    nodes = splitter.get_nodes_from_documents(docs)

    storage_context = StorageContext.from_defaults()

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    index.storage_context.persist(persist_dir=str(INDEX_DIR))

    print(f"\n✅ Index built and saved to: {INDEX_DIR.resolve()}")
    print(f"📄 Nodes indexed: {len(nodes)}")


if __name__ == "__main__":
    main()
