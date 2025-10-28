import argparse
import json
import re
import numpy as np
import pickle
from collections import defaultdict
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import networkx as nx

def sanitize_relationship_name(rel_name):
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', rel_name)
    if sanitized and sanitized[0].isdigit():
        sanitized = f"REL_{sanitized}"
    sanitized = sanitized.upper()
    if not sanitized or sanitized == "_":
        sanitized = "UNKNOWN_RELATION"
    return sanitized

def preprocess_chunk_data(chunk_data):
    unique_entities = set()
    all_relationships = []
    entity_to_chunks = defaultdict(set)
    chunk_triplet_mapping = {}
    all_triplets = []
    
    for chunk in chunk_data:
        chunk_id = chunk["chunk_id"]
        file_id = chunk["file_id"]
        triplets = chunk["triplets"]
        
        chunk_triplet_mapping[chunk_id] = {
            "file_id": file_id,
            "triplets": triplets,
            "triplet_count": len(triplets)
        }
        
        for idx, triplet in enumerate(triplets):
            subj = triplet["subject"]
            rel = triplet["predicate"]
            obj = triplet["object"]
            
            triplet_with_chunk = triplet.copy()
            triplet_with_chunk["chunk_id"] = chunk_id
            triplet_with_chunk["global_idx"] = len(all_triplets)
            all_triplets.append(triplet_with_chunk)
            
            unique_entities.add(subj)
            unique_entities.add(obj)
            
            entity_to_chunks[subj].add(chunk_id)
            entity_to_chunks[obj].add(chunk_id)
            
            rel_sanitized = sanitize_relationship_name(rel)
            all_relationships.append({
                "subject": subj,
                "object": obj,
                "predicate": rel_sanitized,
                "original_predicate": rel,
                "chunk_id": chunk_id,
                "global_idx": len(all_triplets) - 1
            })
    
    return (unique_entities, all_relationships, entity_to_chunks, 
            chunk_triplet_mapping, all_triplets)

def generate_entity_embeddings(entities, embedding_model, batch_size=2048):
    entity_embeddings = {}
    entities_list = list(entities)
    
    for i in tqdm(range(0, len(entities_list), batch_size), desc="Generating entity embeddings"):
        batch = entities_list[i:i + batch_size]
        embeddings = embedding_model.get_text_embedding_batch(batch, show_progress_bar=False)
        for entity, embedding in zip(batch, embeddings):
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            entity_embeddings[entity] = embedding
    
    return entity_embeddings

def generate_chunk_embeddings(chunk_triplet_mapping, embedding_model, batch_size=256):
    chunk_embeddings = {}
    chunks_list = list(chunk_triplet_mapping.keys())
    
    chunk_texts = []
    for chunk_id in chunks_list:
        triplets = chunk_triplet_mapping[chunk_id]["triplets"]
        triplet_texts = [
            f"{t['subject']} {t['predicate']} {t['object']}"
            for t in triplets
        ]
        chunk_text = " | ".join(triplet_texts)
        chunk_texts.append(chunk_text)
    
    for i in tqdm(range(0, len(chunks_list), batch_size), desc="Generating chunk embeddings"):
        batch_chunks = chunks_list[i:i + batch_size]
        batch_texts = chunk_texts[i:i + batch_size]
        
        embeddings = embedding_model.get_text_embedding_batch(batch_texts, show_progress_bar=False)
        for chunk_id, embedding in zip(batch_chunks, embeddings):
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            chunk_embeddings[chunk_id] = embedding
    
    return chunk_embeddings

def create_networkx_graph(entities, relationships, entity_to_chunks, entity_embeddings):
    G = nx.MultiDiGraph()
    
    for entity in tqdm(entities, desc="Adding entities"):
        G.add_node(entity, 
                  node_type='entity',
                  embedding=entity_embeddings[entity],
                  chunk_ids=list(entity_to_chunks[entity]))
    
    for rel in tqdm(relationships, desc="Adding relationships"):
        G.add_edge(rel["subject"], 
                  rel["object"],
                  predicate=rel["predicate"],
                  original_predicate=rel["original_predicate"],
                  chunk_id=rel["chunk_id"],
                  global_idx=rel["global_idx"])
    
    return G

def create_tfidf_index(entities, entity_to_chunks, chunk_triplet_mapping):
    entity_documents = {}
    
    for entity in tqdm(entities, desc="Creating entity documents for TF-IDF"):
        entity_text_parts = [entity]
        
        for chunk_id in entity_to_chunks[entity]:
            chunk_info = chunk_triplet_mapping[chunk_id]
            for triplet in chunk_info["triplets"]:
                if triplet["subject"] == entity or triplet["object"] == entity:
                    entity_text_parts.append(f"{triplet['subject']} {triplet['predicate']} {triplet['object']}")
        
        entity_documents[entity] = " ".join(entity_text_parts)
    
    entity_list = list(entities)
    documents = [entity_documents[entity] for entity in entity_list]
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    return vectorizer, tfidf_matrix, entity_list

def save_graph_data(G, chunk_triplet_mapping, chunk_embeddings, vectorizer, tfidf_matrix, entity_list,
                    graph_file, chunk_file, tfidf_file):
    with open(graph_file, "wb") as f:
        pickle.dump(G, f)
    
    with open(chunk_file, "wb") as f:
        pickle.dump({
            "chunk_triplet_mapping": chunk_triplet_mapping,
            "chunk_embeddings": chunk_embeddings
        }, f)
    
    joblib.dump({
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "entity_list": entity_list
    }, tfidf_file)

def create_chunks_from_triplets(triplets_data, chunk_size=3):
    chunks = []
    current_triplets = []
    chunk_id_counter = 0

    for triplet in triplets_data:
        current_triplets.append(triplet)
        if len(current_triplets) == chunk_size:
            chunk = {
                "chunk_id": f"chunk_{chunk_id_counter}",
                "file_id": triplet.get("file_id"),
                "triplets": current_triplets.copy()
            }
            chunks.append(chunk)
            current_triplets = []
            chunk_id_counter += 1
    
    if current_triplets:
        chunk = {
            "chunk_id": f"chunk_{chunk_id_counter}",
            "file_id": current_triplets[0].get("file_id"),
            "triplets": current_triplets
        }
        chunks.append(chunk)

    return chunks

def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Builder")
    parser.add_argument("--input-triplets", default="triplets.json", help="Input triplets JSON file")
    parser.add_argument("--graph-file", default="knowledge_graph.pickle", help="Output graph pickle file")
    parser.add_argument("--chunk-file", default="chunk_data.pickle", help="Output chunk data pickle file")
    parser.add_argument("--tfidf-file", default="tfidf_data.joblib", help="Output TF-IDF joblib file")
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5", help="Embedding model name")
    args = parser.parse_args()
    
    embedding_model = HuggingFaceEmbedding(model_name=args.embedding_model)
    
    try:
        with open(args.input_triplets, "r", encoding="utf-8") as f:
            triplets_data = json.load(f)

        chunk_data = create_chunks_from_triplets(triplets_data)
        
        (unique_entities, all_relationships, entity_to_chunks, 
         chunk_triplet_mapping, all_triplets) = preprocess_chunk_data(chunk_data)
        
        entity_embeddings = generate_entity_embeddings(unique_entities, embedding_model)
        
        chunk_embeddings = generate_chunk_embeddings(chunk_triplet_mapping, embedding_model)
        
        G = create_networkx_graph(unique_entities, all_relationships, entity_to_chunks, entity_embeddings)
        
        vectorizer, tfidf_matrix, entity_list = create_tfidf_index(unique_entities, entity_to_chunks, chunk_triplet_mapping)
        
        save_graph_data(G, chunk_triplet_mapping, chunk_embeddings, vectorizer, tfidf_matrix, entity_list,
                        args.graph_file, args.chunk_file, args.tfidf_file)
        
    except Exception as e:
        print(f"Script failed: {e}")

if __name__ == "__main__":
    main()