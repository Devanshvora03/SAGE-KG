import os
import json
import re
from tqdm import tqdm
import logging

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.misc_utils import compute_mdhash_id

# ────────────────────────────────────────────────
#  Configuration – change these according to your use-case
# ────────────────────────────────────────────────
CONFIG = {
    "openai_api_key": os.getenv("OPENAI_API_KEY"),          # or set it directly (but prefer env var)
    "llm_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
    
    "documents_md_path": "data/documents.md",               # markdown file with documents
    "custom_triplets_json": "data/extracted_triplets.json", # your pre-computed triples
    "save_directory": "knowledge_base",
    "results_path": "results/rag_results.json",
    
    "openie_mode": "online",
    "force_openie_from_scratch": False,
    "save_openie": True,
    "embedding_batch_size": 2048,
}

# ────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
os.environ["OPENAI_API_KEY"] = CONFIG["openai_api_key"] or ""

if not CONFIG["openai_api_key"]:
    raise ValueError("OPENAI_API_KEY is not set in environment or config")


def extract_documents_and_queries_from_md(file_path: str):
    """
    Simple markdown parser expecting format like:

    ---
    **Question:** ...
    **Context:** ...
    ---
    """
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    sections = [s.strip() for s in re.split(r'^-{3,}$', content, flags=re.MULTILINE) if s.strip()]

    queries = []
    passages = []

    for section in tqdm(sections, desc="Parsing markdown sections"):
        q_match = re.search(r'\*\*Question:\*\*\s*(.+)', section, re.I)
        if q_match:
            queries.append(q_match.group(1).strip())

        ctx_match = re.search(r'\*\*Context:\*\*\s*(.+)', section, re.I | re.DOTALL)
        if ctx_match:
            text = ctx_match.group(1).strip()
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            passages.extend(paragraphs)

    print(f"Extracted {len(queries)} queries and {len(passages)} passages")
    return queries, passages


def convert_custom_triplets_to_hipporag_format(passages: list[str], triplets_path: str):
    """
    Converts your custom triplet JSON into HippoRAG's expected OpenIE document format.
    Expects triplets JSON with entries like:
    [{"chunk_id": "...", "subject": "...", "predicate": "...", "object": "..."}, ...]
    """
    with open(triplets_path, encoding="utf-8") as f:
        raw_triplets = json.load(f)

    # Group by chunk index (assuming chunk_id pattern like md_context_5_0)
    pattern = re.compile(r'.*_(\d+)_\d+$')
    grouped = {}

    for t in raw_triplets:
        cid_raw = t.get("chunk_id", "")
        m = pattern.search(cid_raw)
        if not m:
            continue
        idx = int(m.group(1))
        grouped.setdefault(idx, {"entities": set(), "triples": []})

        sub = str(t.get("subject", "")).strip()
        pred = str(t.get("predicate", "")).strip()
        obj = str(t.get("object", "")).strip()

        if sub and pred and obj:
            grouped[idx]["entities"].update([sub, obj])
            grouped[idx]["triples"].append([sub, pred, obj])

    print(f"Triplets grouped for {len(grouped)} chunks")

    # Build final OpenIE documents
    openie_docs = []
    for i, passage in enumerate(passages):
        chunk_id = compute_mdhash_id(content=passage, prefix="chunk-")
        data = grouped.get(i, {"entities": set(), "triples": []})

        openie_docs.append({
            "idx": chunk_id,
            "passage": passage,
            "extracted_entities": sorted(data["entities"]),
            "extracted_triples": data["triples"]
        })

    total_triples = sum(len(d["extracted_triples"]) for d in openie_docs)
    total_ents = sum(len(d["extracted_entities"]) for d in openie_docs)

    print(f"Prepared {len(openie_docs)} documents | {total_triples} triples | {total_ents} entities")

    return {"docs": openie_docs}


def main():
    print("HippoRAG – Custom Triplets + Retrieval + QA")
    print("─" * 60)

    # 1. Load documents & queries
    print("1. Reading documents and queries...")
    queries, passages = extract_documents_and_queries_from_md(CONFIG["documents_md_path"])

    # 2. Prepare OpenIE data with custom triplets
    print("2. Converting custom triplets...")
    openie_data = convert_custom_triplets_to_hipporag_format(passages, CONFIG["custom_triplets_json"])

    os.makedirs(CONFIG["save_directory"], exist_ok=True)
    openie_save_path = os.path.join(CONFIG["save_directory"], "openie_custom.json")
    with open(openie_save_path, "w", encoding="utf-8") as f:
        json.dump(openie_data, f, indent=2, ensure_ascii=False)
    print(f"   → OpenIE data saved: {openie_save_path}")

    # 3. Initialize HippoRAG
    print("3. Initializing HippoRAG...")
    cfg = BaseConfig()
    cfg.openie_mode = CONFIG["openie_mode"]
    cfg.force_openie_from_scratch = CONFIG["force_openie_from_scratch"]
    cfg.save_openie = CONFIG["save_openie"]
    cfg.embedding_batch_size = CONFIG["embedding_batch_size"]

    rag = HippoRAG(
        global_config=cfg,
        save_dir=CONFIG["save_directory"],
        llm_model_name=CONFIG["llm_model"],
        embedding_model_name=CONFIG["embedding_model"],
    )

    # Speed-up trick (often 10–50× faster)
    print("4. Disabling synonymy edges...")
    rag.add_synonymy_edges = lambda: logging.info("Synonymy edges disabled")

    # 5. Index
    print(f"5. Indexing {len(passages)} passages...")
    rag.index(docs=passages)
    print("   Indexing finished")

    # 6. Run queries with fallback
    print(f"6. Answering {len(queries)} questions...")
    results = []
    fallback_count = 0

    for q in tqdm(queries, desc="Answering"):
        try:
            out = rag.rag_qa(queries=[q])
            ans = out[0][0].answer.strip()
            results.append({"query": q, "answer": ans, "method": "hipporag"})
        except AssertionError as e:
            if "No phrases found in the graph" in str(e):
                try:
                    out_dpr = rag.rag_qa_dpr(queries=[q])
                    ans = out_dpr[0][0].answer.strip()
                    results.append({"query": q, "answer": ans, "method": "dpr_fallback"})
                    fallback_count += 1
                except Exception as e2:
                    results.append({"query": q, "answer": "", "error": str(e2)[:200]})
                    fallback_count += 1
            else:
                results.append({"query": q, "answer": "", "error": str(e)[:200]})
                fallback_count += 1
        except Exception as e:
            results.append({"query": q, "answer": "", "error": str(e)[:200]})
            fallback_count += 1

    # 7. Save results
    os.makedirs(os.path.dirname(CONFIG["results_path"]), exist_ok=True)
    with open(CONFIG["results_path"], "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    answered = sum(1 for r in results if r.get("answer"))
    hippo_success = sum(1 for r in results if r.get("method") == "hipporag")
    print("\n" + "═" * 60)
    print("SUMMARY")
    print("═" * 60)
    print(f"Queries:      {len(queries)}")
    print(f"Answered:     {answered} ({answered / len(queries):.1%})")
    print(f"  • HippoRAG: {hippo_success}")
    print(f"  • DPR fallback: {fallback_count}")
    print(f"Failed:       {fallback_count}")
    print(f"Results → {CONFIG['results_path']}")
    print("═" * 60)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()