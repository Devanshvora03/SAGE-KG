import os
import json
import re
import traceback
from tqdm import tqdm
from typing import List, Dict, Any

from hipporag import HippoRAG

# ────────────────────────────────────────────────
#               CONFIGURATION
# ────────────────────────────────────────────────
CONFIG = {
    # Required: set via environment variable (recommended) or directly here
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    
    # Models
    "llm_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
    
    # Paths
    "documents_path": "data/documents.md",          # ← change to your file
    "save_directory": "knowledge_base",
    "results_path": "results/rag_results.json",
    
    # HippoRAG behaviour
    "index_batch_size": 128,                        # adjust based on memory
}


def load_api_key():
    """Ensure API key is available"""
    key = CONFIG["openai_api_key"]
    if not key or not key.startswith("sk-"):
        raise ValueError(
            "OPENAI_API_KEY is missing or invalid.\n"
            "Set it via environment variable:\n"
            "  export OPENAI_API_KEY='sk-...'\n"
            "or add it directly in CONFIG (not recommended for shared repos)."
        )
    os.environ["OPENAI_API_KEY"] = key


def extract_questions_and_passages(md_path: str) -> tuple[List[str], List[str]]:
    """
    Parse markdown file expecting sections separated by --- or similar,
    each containing **Question:** and **Context:** blocks.
    
    Returns: (list of questions, list of text passages/chunks)
    """
    if not os.path.isfile(md_path):
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    with open(md_path, encoding="utf-8") as f:
        content = f.read()

    # Split on horizontal rules (--- or longer)
    sections = [s.strip() for s in re.split(r'^-{3,}$', content, flags=re.MULTILINE) if s.strip()]

    questions = []
    passages = []

    for section in tqdm(sections, desc="Parsing markdown sections"):
        # Question
        q_match = re.search(r'\*\*Question:\*\*\s*(.+?)(?=\*\*Context:|$)', section, re.I | re.DOTALL)
        if q_match:
            q = q_match.group(1).strip()
            if q:
                questions.append(q)

        # Context → split into paragraphs
        ctx_match = re.search(r'\*\*Context:\*\*\s*(.+)', section, re.I | re.DOTALL)
        if ctx_match:
            text = ctx_match.group(1).strip()
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            passages.extend(paragraphs)

    print(f"→ Found {len(questions)} questions")
    print(f"→ Extracted {len(passages)} passages")

    return questions, passages


def run_evaluation():
    print("HippoRAG – Simple Indexing + QA Pipeline")
    print("─" * 60)

    # 1. API Key check
    load_api_key()

    # 2. Load data
    print("\n1. Loading documents and questions...")
    try:
        queries, passages = extract_questions_and_passages(CONFIG["documents_path"])
    except Exception as e:
        print(f"❌ Failed to load markdown: {e}")
        return

    if not passages:
        print("❌ No passages found → nothing to index.")
        return

    # 3. Initialize HippoRAG
    print("\n2. Initializing HippoRAG...")
    try:
        rag = HippoRAG(
            save_dir=CONFIG["save_directory"],
            llm_model_name=CONFIG["llm_model"],
            embedding_model_name=CONFIG["embedding_model"],
        )
    except Exception as e:
        print(f"❌ Failed to initialize HippoRAG: {e}")
        traceback.print_exc()
        return

    # 4. Indexing
    print(f"\n3. Indexing {len(passages)} passages...")
    try:
        rag.index(docs=passages, batch_size=CONFIG["index_batch_size"])
        print("✓ Indexing finished")
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        traceback.print_exc()
        return

    # 5. Run queries
    print(f"\n4. Answering {len(queries)} questions...")
    results: List[Dict[str, Any]] = []

    for q in tqdm(queries, desc="Answering"):
        try:
            output = rag.rag_qa(queries=[q])

            if not output or len(output) == 0:
                results.append({"query": q, "answer": ""})
                continue

            # Most common case: list of answers
            item = output[0]
            if isinstance(item, list) and len(item) > 0:
                item = item[0]

            # Try to get .answer attribute
            if hasattr(item, "answer"):
                answer = str(item.answer).strip()
            else:
                # Fallback: string representation heuristic
                s = str(item)
                m = re.search(r"answer\s*[:=]\s*['\"](.*?)['\"]", s, re.I | re.DOTALL)
                answer = m.group(1).strip() if m else ""

            results.append({"query": q, "answer": answer})

        except Exception as e:
            print(f"  Query failed: {q[:60]}... → {type(e).__name__}: {e}")
            results.append({"query": q, "answer": "", "error": str(e)[:180]})

    # 6. Save results
    os.makedirs(os.path.dirname(CONFIG["results_path"]), exist_ok=True)
    with open(CONFIG["results_path"], "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 7. Summary
    answered = sum(1 for r in results if r.get("answer"))
    failed = len(results) - answered

    print("\n" + "═" * 70)
    print("SUMMARY")
    print("═" * 70)
    print(f"  Questions total : {len(queries):3d}")
    print(f"  Answered        : {answered:3d}  ({answered/len(queries):.1%})")
    print(f"  Failed / empty  : {failed:3d}")
    print(f"  Results saved → {CONFIG['results_path']}")
    print("═" * 70)

    # Show a few examples
    if results:
        print("\nSample results:")
        print("─" * 70)
        for i, r in enumerate(results[:4], 1):
            ans = (r["answer"][:100] + "...") if len(r["answer"]) > 100 else r["answer"]
            print(f"[{i}] {r['query'][:68]}...")
            print(f"    → {ans or '<no answer>'}\n")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # Windows + torch/mp safety
    try:
        run_evaluation()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()