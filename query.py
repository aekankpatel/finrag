from pathlib import Path
from pydantic import Field
from llama_index.core import load_index_from_storage, StorageContext, Settings
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

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


def load_index():
    embed_model = SafeOllamaEmbedding(model_name="nomic-embed-text", max_chars=3000)
    llm = Ollama(model="llama3.1:8b", request_timeout=120.0)
    Settings.embed_model = embed_model
    Settings.llm = llm

    storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
    index = load_index_from_storage(storage_context)
    return index


def query(index, question: str, top_k: int = 8, source_filter: str = None):
    if source_filter:
        filters = MetadataFilters(filters=[
            MetadataFilter(key="source", value=source_filter)
        ])
        query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="compact",
            filters=filters,
        )
    else:
        query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="compact",
        )

    response = query_engine.query(question)

    print("\n" + "=" * 60)
    print("🧠 ANSWER:")
    print("=" * 60)
    print(response.response)

    print("\n" + "=" * 60)
    print("📄 SOURCES:")
    print("=" * 60)
    for i, node in enumerate(response.source_nodes, 1):
        source = node.metadata.get("source", "unknown")
        score = round(node.score, 4) if node.score else "N/A"
        print(f"\n[{i}] File: {source}  |  Score: {score}")
        print(f"    {node.text[:300].strip()}...")


def main():
    print("📦 Loading index...")
    index = load_index()
    print("✅ Index loaded. Ready to query.")
    print("💡 Tip: prefix with @filename to filter by source, e.g: @meta_10K_2025.txt What are the risk factors?\n")

    while True:
        question = input("💬 Ask a question (or type 'exit' to quit):\n> ").strip()
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue

        # Parse optional @filename filter
        source_filter = None
        if question.startswith("@"):
            parts = question.split(" ", 1)
            source_filter = parts[0][1:]
            question = parts[1] if len(parts) > 1 else ""
            print(f"🔍 Filtering to: {source_filter}")

        if question:
            query(index, question, source_filter=source_filter)
        print()


if __name__ == "__main__":
    main()
