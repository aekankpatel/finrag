from pathlib import Path
from pydantic import Field
from llama_index.core import load_index_from_storage, StorageContext, Settings
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import json
import time

INDEX_DIR = Path("finrag/index")

# ---- Test questions with expected keywords ----
TEST_CASES = [
    {
        "question": "What regulatory risks does Meta face regarding privacy?",
        "source": "meta_10K_2025.txt",
        "expected_keywords": ["privacy", "GDPR", "regulation", "data"],
    },
    {
        "question": "How does Meta describe risks related to AI?",
        "source": "meta_10K_2025.txt",
        "expected_keywords": ["AI", "artificial intelligence", "regulation", "data"],
    },
    {
        "question": "What is Amazon's main source of revenue?",
        "source": "Amazon_10K_2025.txt",
        "expected_keywords": ["AWS", "cloud", "retail", "revenue", "services"],
    },
    {
        "question": "What risks does Amazon highlight in international operations?",
        "source": "Amazon_10K_2025.txt",
        "expected_keywords": ["international", "regulation", "currency", "political"],
    },
    {
        "question": "How did NVIDIA describe demand for its data center products?",
        "source": "NVIDIA_10Q_2025Q3.txt",
        "expected_keywords": ["data center", "demand", "GPU", "AI", "revenue"],
    },
    {
        "question": "What risks does NVIDIA mention regarding export controls?",
        "source": "NVIDIA_10Q_2025Q3.txt",
        "expected_keywords": ["export", "China", "control", "restrictions"],
    },
    {
        "question": "How does Bank of America describe interest rate risk?",
        "source": "BAC+2024+Annual+Report.txt",
        "expected_keywords": ["interest rate", "risk", "net interest income", "federal"],
    },
    {
        "question": "What are the key risk types Bank of America faces?",
        "source": "BAC+2024+Annual+Report.txt",
        "expected_keywords": ["credit", "market", "operational", "liquidity", "compliance"],
    },
    {
        "question": "What risks does Apple highlight related to supply chain?",
        "source": "aaple_10Q_2025Q1.txt",
        "expected_keywords": ["supply", "manufacturer", "component", "sourcing"],
    },
    {
        "question": "How does Apple describe competition risks?",
        "source": "aaple_10Q_2025Q1.txt",
        "expected_keywords": ["competition", "market", "product", "price"],
    },
]


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
    return load_index_from_storage(storage_context)


def score_answer(answer: str, expected_keywords: list) -> float:
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return round(hits / len(expected_keywords), 2)


def run_evaluation():
    print("📦 Loading index...")
    index = load_index()
    print("✅ Index loaded. Running evaluation...\n")

    results = []
    total_score = 0

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"[{i}/{len(TEST_CASES)}] {tc['question'][:60]}...")
        start = time.time()

        filters = MetadataFilters(filters=[
            MetadataFilter(key="source", value=tc["source"])
        ])
        engine = index.as_query_engine(
            similarity_top_k=8,
            response_mode="compact",
            filters=filters,
        )
        response = engine.query(tc["question"])
        elapsed = round(time.time() - start, 2)

        score = score_answer(response.response, tc["expected_keywords"])
        total_score += score

        result = {
            "question": tc["question"],
            "source": tc["source"],
            "score": score,
            "expected_keywords": tc["expected_keywords"],
            "keywords_found": [kw for kw in tc["expected_keywords"] if kw.lower() in response.response.lower()],
            "response_time_sec": elapsed,
            "answer_preview": response.response[:200],
        }
        results.append(result)

        status = "✅" if score >= 0.5 else "⚠️" if score > 0 else "❌"
        print(f"   {status} Score: {score} | Time: {elapsed}s | Keywords hit: {result['keywords_found']}\n")

    avg_score = round(total_score / len(TEST_CASES), 2)

    print("=" * 60)
    print(f"📊 EVALUATION COMPLETE")
    print(f"   Total questions : {len(TEST_CASES)}")
    print(f"   Average score   : {avg_score} / 1.0")
    print(f"   Pass rate (≥0.5): {sum(1 for r in results if r['score'] >= 0.5)}/{len(TEST_CASES)}")
    print("=" * 60)

    output_path = Path("finrag/eval_results.json")
    with open(output_path, "w") as f:
        json.dump({"avg_score": avg_score, "results": results}, f, indent=2)
    print(f"\n💾 Full results saved to: {output_path.resolve()}")


if __name__ == "__main__":
    run_evaluation()
