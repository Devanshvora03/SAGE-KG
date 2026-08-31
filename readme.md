# SAGE-KG

Sequential Uncertainty Resolution for knowledge-graph construction with small language models.

SAGE-KG turns a text corpus into a knowledge graph in three stages, then answers multi-hop questions from that graph. The method is **Sequential Uncertainty Resolution (SUR)**: semantic, structural, and representational uncertainty are resolved one after another, not in a single prompt.

```
documents  →  triples  →  graph + indexes  →  retrieved triples  →  answer
               SUR           NetworkX            hybrid KGI           LLM
```

Reported experiments use Qwen2.5 (3B / 7B / 14B) via Ollama, with GPT-4o-mini and Gemini 2.0 Flash as independent judges.

---

## Pipeline (paper to code)

| Stage | Uncertainty | Agent / module | Code |
| --- | --- | --- | --- |
| 1. Extract | Semantic, then structural, then representational | Fact Extractor, Schema Planner, Triplet Generator | [`sage_kg/extraction/agents.py`](sage_kg/extraction/agents.py) |
| 2. Construct | Union of per-chunk graphs | NetworkX MultiDiGraph + embeddings + TF-IDF | [`sage_kg/construction/create_kg.py`](sage_kg/construction/create_kg.py) |
| 3. Query | Hybrid retrieval | Chunk embedding, hybrid entity seeds, 3-hop expansion | [`sage_kg/querying/query_kg.py`](sage_kg/querying/query_kg.py) |

Only the last extraction stage is parsed as `(subject, predicate, object)`. Intermediate agent outputs stay free-form and are passed as CrewAI context (Intermediate Output Tolerance in the paper).

---

## Repository layout

```
sage_kg/
  extraction/agents.py        # SUR: three sequential CrewAI agents
  construction/create_kg.py   # triples → MultiDiGraph + indexes
  querying/query_kg.py        # hybrid retrieval + answering
  evaluation/
    intrinsic/                # triple quality (GPT / Gemini / MINE)
    extrinsic/                # QA quality (judge, EM, semantic relevance)
    human_extrinsic/          # manual QA judgments
    human_intrinsic_semeval/  # SemEval entity-recovery annotations
Baselines/                    # OpenIE, KGGen, GraphRAG, HippoRAG, vector RAG, zero-shot
Datasets/                     # HotpotQA, MuSiQue, 2WikiMultiHopQA, MINE, SemEval
Results/                      # released tables / per-run logs
Ablation/                     # stage-order and model-family ablations
```

`Results/` and `Ablation/` are experiment artifacts. You do not need them to run the pipeline.

---

## Setup

Python 3.10–3.13 and a running [Ollama](https://ollama.com) daemon.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:14b
```

For LLM-as-judge evaluation, copy `.env.example` to `.env` and set `OPENAI_API_KEY` and/or a Gemini key.

---

## Run the pipeline

Paper chunking: 200-token windows, 25-token overlap. This repo uses a word-window approximation with the same 200 / 25 defaults.

### 1. Extract triples

```bash
python sage_kg/extraction/agents.py qwen2.5:14b \
  --data Datasets/HotpotQA \
  --output output/hotpot \
  --chunk 200 \
  --overlap 25
```

Writes `output/hotpot/triples_*.json` with `subject`, `predicate`, `object`, `file_id`, `chunk_id`.

### 2. Build the graph

```bash
python sage_kg/construction/create_kg.py \
  --input-triplets output/hotpot/triples_<model>_<timestamp>.json \
  --graph-file output/hotpot/knowledge_graph.pickle \
  --chunk-file output/hotpot/chunk_data.pickle \
  --tfidf-file output/hotpot/tfidf_data.joblib
```

### 3. Query

QA files use this markdown shape:

```
**Question:** Who directed Inception?
**Answer:** Christopher Nolan
```

```bash
python sage_kg/querying/query_kg.py \
  --qa-file Datasets/HotpotQA/HotpotQA.md \
  --graph-file output/hotpot/knowledge_graph.pickle \
  --chunk-file output/hotpot/chunk_data.pickle \
  --tfidf-file output/hotpot/tfidf_data.joblib \
  --llm-model qwen2.5:14b \
  --output-file output/hotpot/answers.md
```

---

## Evaluation

| What | Judge / metric | Script |
| --- | --- | --- |
| Triple quality | GPT-4o-mini | `python sage_kg/evaluation/intrinsic/judge_gpt.py` |
| Triple quality | Gemini | `python sage_kg/evaluation/intrinsic/judge_gemini.py` |
| Relational completeness | MINE | `python sage_kg/evaluation/intrinsic/mine_evaluation.py` |
| QA quality | GPT / Gemini | `sage_kg/evaluation/extrinsic/judge_evaluation_*.py` |
| QA quality | Exact Match | `python sage_kg/evaluation/extrinsic/exact_match.py` |
| QA quality | Semantic relevance | `python sage_kg/evaluation/extrinsic/semantic_relevance.py` |

Human labels:

- Extrinsic QA: `sage_kg/evaluation/human_extrinsic/data.csv`
- SemEval entity recovery: `sage_kg/evaluation/human_intrinsic_semeval/{SAGE,KGGen,OpenIE,Zeroshot}/`

Released numbers for all methods and judges live under `Results/`.

---

## Baselines and ablations

`Baselines/` holds the comparison systems from the paper (OpenIE, KGGen, Microsoft GraphRAG, HippoRAG, standard vector RAG, zero-shot GraphRAG).

`Ablation/` holds the controlled variants:

- dataset-specific: Gemma vs Qwen vs simple vs non-agentic
- agent combinations: `1+3` (no planner) and `2+3` (no fact extractor)

---

## Datasets

| Dataset | Role |
| --- | --- |
| HotpotQA, MuSiQue, 2WikiMultiHopQA | Multi-hop QA (intrinsic + extrinsic) |
| MINE | Gold-fact relational completeness |
| SemEval-2010 Task 8 | Human entity-recovery check |

Dataset folders contain the manifests used for the reported runs.
