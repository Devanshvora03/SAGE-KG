# SAGE-KG  - Sequential Agentic Orchestration for Scalable Knowledge Graph Construction and Reasoning

SAGE-KG is a **research toolkit and evaluation suite** for **Graph Retrieval-Augmented Generation (Graph RAG)**.  
It provides an end-to-end pipeline for **triplet extraction, knowledge graph construction, hybrid graph + vector retrieval**, and **comprehensive intrinsic & extrinsic evaluation**, along with **strong baseline implementations**.

The repository is designed for **reproducible research** on multi-hop reasoning, graph-based retrieval, and agentic extraction pipelines using **open-source LLMs**.

---

## ✨ Key Features

- 🧩 **Agentic Triplet Extraction (SAGE)**  
  Sequential, role-specialized agents extract high-quality `(subject, relation, object)` triplets from raw text.

- 🕸️ **Schema-Free Knowledge Graph Construction**  
  Builds a global **NetworkX MultiDiGraph** with entity normalization, embeddings, and TF-IDF indexing.

- 🔍 **Hybrid Graph + Vector Retrieval**  
  Multi-hop graph traversal combined with embedding similarity and lexical ranking via **LlamaIndex KGI**.

- 📊 **Rigorous Evaluation Suite**  
  - *Intrinsic*: Context sufficiency & factual coverage  
  - *Extrinsic*: LLM-judged answer quality across multiple dimensions

- ⚖️ **Strong Baselines & Ablations**  
  Includes HippoRAG, Standard RAG, KGGen, GraphRAG variants, and systematic ablation studies.

---

## 🔗 Quick Navigation

### Core Pipeline
- **Triplet Extraction (Agentic SAGE)**  
  `SAGE-KG/Triplet Extraction - SAGE/agents.py`

- **Knowledge Graph Construction**  
  `SAGE-KG/Graph Construction/create_kg.py`

- **Graph Querying & Retrieval (KGI)**  
  `SAGE-KG/Graph Querying - KGI/query_kg.py`

### Baselines
- **HippoRAG**  
  `Baselines/Hipporag/sage_hipporag2.py`
- **Other Baselines**  
  Standard RAG, KGGen, Zero-Shot GraphRAG → see `Baselines/`

### Evaluation
- **Intrinsic Evaluation**  
  `SAGE-KG/Evaluation/Intrinsic/`
- **Extrinsic Evaluation**  
  `SAGE-KG/Evaluation/Extrinsic/`

### Outputs
- **Results**  
  `Results/`
- **Ablation Studies**  
  `Ablation/`

---

## 🧱 Repository Structure

```

SAGE-KG/
├── Triplet Extraction - SAGE/
│ └── agents.py
├── Graph Construction/
│ └── create_kg.py
├── Graph Querying - KGI/
│ └── query_kg.py
├── Evaluation/
│ ├── Intrinsic/
│ ├── Extrinsic/
│ ├── Human Extrinsic /
│ │ └── data.csv
│ └── Human Intrinsic - Semveval/
│ ├── SAGE/
│ ├── KGGen/
│ ├── OpenIE/
│ └── Zeroshot/
├── Baselines/
│ ├── Hipporag/
│ ├── StandardRAG/
│ ├── KGGen/
│ └── GraphRAG/
├── Datasets/
├── Results/ 
└── Ablation/
```

---

## 🧠 Pipeline Overview

### 1. Triplet Extraction (SAGE)
- Uses a **sequential agentic pipeline**:
  - Fact extraction
  - Entity–relation planning
  - Triplet materialization
- Produces clean, atomic `(s, r, o)` facts
- Supports multiple extraction variants (agentic, non-agentic, simple)

### 2. Knowledge Graph Construction
- Aggregates extracted triplets into a **global MultiGraph**
- Adds:
  - Entity embeddings
  - TF-IDF indices
  - Canonical entity alignment
- Designed for **hybrid retrieval and multi-hop traversal**

### 3. Graph Querying & RAG
- Uses **LlamaIndex Knowledge Graph Index (KGI)**
- Retrieval combines:
  - Graph neighborhood expansion
  - Embedding similarity
  - Lexical relevance
- Retrieved subgraphs are passed to an LLM for answer generation

---

## 📊 Evaluation Framework

## 👥 Human Evaluation

SAGE-KG includes **human-annotated evaluations** to complement LLM-based metrics and provide grounded validation.  
Human **extrinsic evaluation** (`Evaluation/Human - Extrinsic Evaluation/data.csv`) contains manual judgments of final QA answers for correctness and completeness.  
Human **intrinsic evaluation** (`Evaluation/Human - Intrinsic Evaluation - Semveval/`) measures extraction fidelity against SemEval-style annotations.  
This setup enables direct comparison between **SAGE, KGGen, OpenIE, and Zero-shot** triplet extractors.  
Human evaluations isolate **true extraction quality and reasoning utility**, independent of automated judges.

---

## 🔬 Ablation Studies

The `Ablation/` directory contains controlled experiments analyzing individual components:

### Dataset-Specific Ablations
- `Ablation-Hotpot`
- `Ablation-Musique`
- `Ablation-Wiki`

Compare:
- Triplet extraction variants (Gemma, Qwen, Simple, Non-Agentic)
- Impact on retrieval quality and context sufficiency

### Agentic Component Ablations
- `Agents/1+3`
- `Agents/2+3`

Analyze:
- Which agent combinations matter most
- Effect of multi-hop expansion strategies

Each ablation includes:
- Validation markdown files (`*_validation.md`)
- Per-query judgments
- Triplet counts and retrieval statistics

---

## 📚 Supported Datasets

Located in `Datasets/`:
- **HotpotQA**
- **MuSiQue**
- **2WikiMultiHopQA**

Includes dataset manifests and example inputs used across all experiments.
