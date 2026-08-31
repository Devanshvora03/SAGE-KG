"""
SAGE-KG Stage 3: hybrid graph retrieval and answer generation.

Paper mapping
-------------
This is the KGI-style querying used for triplet-based methods in the paper.
Given a question q:

  1. Select the most similar triple-chunks by embedding (narrow the corpus).
  2. Hybrid seed entities inside those chunks:
       dense cosine  +  TF-IDF cosine
  3. Expand each seed up to max_hop_depth along the MultiDiGraph
     (multi-hop neighbourhood = the retrieved subgraph).
  4. Pass the collected (s, r, o) facts to an LLM for a short answer.

The same graph files are produced by sage_kg/construction/create_kg.py.
Hop depth 3 is the default used in the main KGI experiments.
"""

import argparse
import re
import os
import json
import numpy as np
import pickle
import joblib
import networkx as nx
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from sklearn.metrics.pairwise import cosine_similarity


def read_sample_file(file_path):
    """Parse a markdown QA dump: **Question:** / **Answer:** pairs."""
    qa_pairs = []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    current_question = None
    current_answer = None

    for line in lines:
        line = line.strip()

        if line.startswith('**Question:**'):
            if current_question and current_answer:
                qa_pairs.append((current_question, current_answer))
            current_question = line.replace('**Question:**', '').strip()
            current_answer = None

        elif line.startswith('**Answer:**'):
            current_answer = line.replace('**Answer:**', '').strip()

    if current_question and current_answer:
        qa_pairs.append((current_question, current_answer))

    return qa_pairs


def save_to_markdown(queries, ground_truths, results, output_file):
    """Write the format consumed by sage_kg/evaluation/extrinsic/exact_match.py."""
    output_content = "# Retrieval Results\n\n"
    output_content += "---\n\n"

    for i, (query, truth) in enumerate(zip(queries, ground_truths), 1):
        output_content += f"\n### Pair {i}\n"
        output_content += f"**Question:** {query}\n"
        output_content += f"**Ground Truth:** {truth}\n"
        output_content += f"**Retrieved Answer:** {results.get(query, 'No answer available')}\n\n"
        output_content += "---\n"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_content)


class NetworkXRetriever:
    """Hybrid chunk → entity seed → multi-hop expansion retriever."""

    def __init__(self, max_hop_depth=3):
        self.max_hop_depth = max_hop_depth
        self.chunk_embeddings = {}
        self.chunk_ids = []
        self.embedding_matrix = None
        self.chunk_triplet_mapping = {}

        self.G = nx.MultiDiGraph()
        self.vectorizer = None
        self.tfidf_matrix = None
        self.tfidf_entity_list = []

    def load_graph_data(self, graph_file):
        with open(graph_file, "rb") as f:
            self.G = pickle.load(f)

    def load_chunk_data(self, chunk_file):
        with open(chunk_file, "rb") as f:
            data = pickle.load(f)
            self.chunk_triplet_mapping = data["chunk_triplet_mapping"]
            self.chunk_embeddings = data["chunk_embeddings"]
            self.chunk_ids = list(self.chunk_embeddings.keys())
            embeddings_list = [self.chunk_embeddings[cid] for cid in self.chunk_ids]
            self.embedding_matrix = np.vstack(embeddings_list)

    def load_tfidf_data(self, tfidf_file):
        tfidf_data = joblib.load(tfidf_file)
        self.vectorizer = tfidf_data["vectorizer"]
        self.tfidf_matrix = tfidf_data["tfidf_matrix"]
        self.tfidf_entity_list = tfidf_data["entity_list"]

    def _select_chunks(self, query_text, embedding_model, top_k=5):
        """First-stage filter: top-k triple-chunks by dense similarity to q."""
        query_emb = embedding_model.encode([query_text])[0].reshape(1, -1)
        similarities = cosine_similarity(query_emb, self.embedding_matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.chunk_ids[idx] for idx in top_indices]

    def _vector_search_entities(self, query_text, selected_chunks, embedding_model, top_k=5):
        """Dense seed entities that appear in the selected chunks."""
        query_emb = embedding_model.encode([query_text])[0].reshape(1, -1)
        scores = []
        for node in self.G.nodes:
            data = self.G.nodes[node]
            if set(data.get('chunk_ids', [])) & set(selected_chunks):
                emb = data.get('embedding')
                if emb is not None:
                    sim = cosine_similarity(query_emb, emb.reshape(1, -1))[0][0]
                    scores.append((node, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [n for n, s in scores[:top_k]]

    def _tfidf_search_entities(self, query_text, selected_chunks, top_k=5):
        """Lexical seed entities (names, rare tokens) inside the same chunks."""
        query_vec = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        scores = []
        for i, entity in enumerate(self.tfidf_entity_list):
            if entity in self.G.nodes:
                data = self.G.nodes[entity]
                if set(data.get('chunk_ids', [])) & set(selected_chunks):
                    scores.append((entity, similarities[i]))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [n for n, s in scores[:top_k]]

    def _get_hybrid_seeds(self, query_text, embedding_model):
        """Union of dense and TF-IDF seeds so neither signal can drop a key entity."""
        chunks = self._select_chunks(query_text, embedding_model)
        vec_entities = self._vector_search_entities(query_text, chunks, embedding_model, 5)
        tfidf_entities = self._tfidf_search_entities(query_text, chunks, 5)
        seeds = list(set(vec_entities + tfidf_entities))
        return seeds

    def _collect_multihop_triplets(self, seeds):
        """BFS from each seed up to max_hop_depth; collect readable (s, r, o) strings."""
        triplets = set()
        for seed in seeds:
            if seed not in self.G.nodes:
                continue
            current = {seed}
            for depth in range(1, self.max_hop_depth + 1):
                next_level = set()
                for node in current:
                    for neighbor in self.G.neighbors(node):
                        for key in self.G[node][neighbor]:
                            data = self.G[node][neighbor][key]
                            p = data['original_predicate']
                            triplet = f"({node}, {p}, {neighbor})"
                            triplets.add(triplet)
                            next_level.add(neighbor)
                current = next_level
        return list(triplets)

    def retrieve_triplets(self, query_text, embedding_model):
        seeds = self._get_hybrid_seeds(query_text, embedding_model)
        triplets = self._collect_multihop_triplets(seeds)
        return triplets


def answer_question(query, triplets, llm):
    """Short factual answer from retrieved triples only (no extra corpus text)."""
    context = "\n".join(triplets)
    prompt = f"""
You are an expert analyst given a set of factual triplets extracted from reliable sources.
Your task is to carefully analyze these facts and provide a clear, concise, short to the point answer to the question.
Answer the question as factual type, just the fact, with no description.

Factual context:
{context}

Question: {query}
Answer: """
    response = llm.complete(prompt)
    return response.text.strip()


def main():
    parser = argparse.ArgumentParser(description="SAGE-KG: hybrid graph retrieval + LLM answering")
    parser.add_argument("--qa-file", default="questions.md", help="Input QA Markdown file")
    parser.add_argument("--output-file", default="output.md", help="Output results Markdown file")
    parser.add_argument("--graph-file", default="knowledge_graph.pickle", help="Input graph pickle file")
    parser.add_argument("--chunk-file", default="chunk_data.pickle", help="Input chunk data pickle file")
    parser.add_argument("--tfidf-file", default="tfidf_data.joblib", help="Input TF-IDF joblib file")
    parser.add_argument("--embedding-model", default="all-mpnet-base-v2", help="Must match create_kg.py")
    parser.add_argument("--llm-model", default="qwen2.5:14b", help="Ollama LLM used to write answers")
    parser.add_argument("--max-hops", type=int, default=3, help="Graph expansion depth")
    args = parser.parse_args()

    embedding_model = SentenceTransformer(args.embedding_model)
    llm = Ollama(model=args.llm_model, temperature=0)

    qa_pairs = read_sample_file(args.qa_file)

    if not qa_pairs:
        print(f"No Question/Answer pairs found in {args.qa_file}")
        return

    queries = [q for q, _ in qa_pairs]
    ground_truths = [a for _, a in qa_pairs]

    retriever = NetworkXRetriever(max_hop_depth=args.max_hops)
    retriever.load_graph_data(args.graph_file)
    retriever.load_chunk_data(args.chunk_file)
    retriever.load_tfidf_data(args.tfidf_file)

    results = {}

    for query in queries:
        triplets = retriever.retrieve_triplets(query, embedding_model)
        answer = "No answer available"
        if triplets:
            answer = answer_question(query, triplets, llm)
        results[query] = answer

    save_to_markdown(queries, ground_truths, results, args.output_file)
    print(f"Wrote {len(results)} answers to {args.output_file}")


if __name__ == "__main__":
    main()
