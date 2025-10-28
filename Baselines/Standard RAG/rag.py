import argparse
import os
import glob
import re
from typing import List, Dict
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
import faiss
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required.")

def load_and_chunk_documents(data_folder: str = "data") -> tuple[List[Document], List[Dict]]:
    md_files = glob.glob(os.path.join(data_folder, '*.md'))
    if not md_files:
        raise ValueError(f"No markdown files found in {data_folder}")
    
    all_documents = []
    question_answer_pairs = []
    splitter = SentenceSplitter(chunk_size=400, chunk_overlap=50)
    
    for md_file in md_files:
        with open(md_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        sections = re.split(r'\*\*Question:\*\*', text.strip(), flags=re.IGNORECASE)
        file_name = os.path.basename(md_file)
        
        section_idx = 0
        for section in sections[1:]:
            question_match = re.match(r'^\s*(.*?)\s*(?=\*\*Answer:\*\*|$)', section, re.DOTALL | re.IGNORECASE)
            if not question_match:
                section_idx += 1
                continue
            question = question_match.group(1).strip()
            
            answer_match = re.search(r'\*\*Answer:\*\*\s*(.*?)\s*(?=\*\*Context:\*\*|$)', section, re.DOTALL | re.IGNORECASE)
            answer = answer_match.group(1).strip() if answer_match else ""
            
            context_match = re.search(r'\*\*Context:\*\*\s*(.*?)(?=\*\*Question:\*\*|$)', section, re.DOTALL | re.IGNORECASE)
            if not context_match:
                section_idx += 1
                continue
            context_text = context_match.group(1).strip()
            if not context_text:
                section_idx += 1
                continue
            
            if question and answer:
                question_answer_pairs.append({
                    "question": question,
                    "ground_truth": answer,
                    "file_id": file_name,
                    "section_id": f"{file_name}_context_{section_idx}"
                })
            
            doc = Document(
                text=context_text,
                metadata={
                    "file_id": file_name,
                    "section_id": f"{file_name}_context_{section_idx}"
                }
            )
            
            nodes = splitter.get_nodes_from_documents([doc])
            all_documents.extend(nodes)
            
            section_idx += 1
    
    return all_documents, question_answer_pairs

def setup_embeddings(embed_model_name: str = "BAAI/bge-large-en-v1.5"):
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    Settings.embed_model = embed_model
    return embed_model

def create_faiss_index(dimension: int = 1024):
    faiss_index = faiss.IndexFlatL2(dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context

def build_index(chunks, storage_context, embed_model, batch_size: int = 64):
    index = VectorStoreIndex(
        nodes=chunks,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
        batch_size=batch_size
    )
    index.storage_context.persist(persist_dir="./faiss_index")
    return index

def setup_openai_client():
    client = OpenAI(api_key=OPENAI_API_KEY)
    return client

def generate_responses(question_answer_pairs, client, model_name: str = "gpt-4o-mini"):
    prompt_template = (
        "You are a factual assistant. Provide a concise, accurate, single-sentence answer to the following query. "
        "Include only information explicitly supported by the context, avoid speculation, and do not provide explanations.\n\n"
        "Query: {query}\n\n"
        "Answer:"
    )
    
    responses = []
    for i, qa_pair in enumerate(tqdm(question_answer_pairs, desc="Processing Queries"), 1):
        query = qa_pair["question"]
        ground_truth = qa_pair["ground_truth"]

        if not query.strip():
            continue

        full_prompt = prompt_template.format(query=query)

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0
        )

        answer_text = response.choices[0].message.content.strip()

        responses.append(
            f"### Query {i}: {query}\n"
            f"**LLM Answer**: {answer_text}\n"
            f"**Ground Truth**: {ground_truth}\n"
        )

    return responses

def save_responses(responses, output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# LLM Responses with Ground Truth\n\n")
        f.write(f"**Total Context Questions and Answers**: {len(responses)}\n\n")
        for r in responses:
            f.write(r + "\n")

def main():
    parser = argparse.ArgumentParser(description="Vector Search Evaluation Tool")
    parser.add_argument("--data-folder", default="data", help="Folder containing input Markdown files")
    parser.add_argument("--output-file", default="llm_responses.md", help="Output Markdown file for responses")
    parser.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5", help="Embedding model name")
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embeddings")
    args = parser.parse_args()

    chunks, question_answer_pairs = load_and_chunk_documents(args.data_folder)
    if not chunks:
        raise ValueError("No chunks loaded, cannot create index.")
    
    embed_model = setup_embeddings(args.embed_model)
    storage_context = create_faiss_index()
    index = build_index(chunks, storage_context, embed_model, args.batch_size)
    
    client = setup_openai_client()
    responses = generate_responses(question_answer_pairs, client, args.openai_model)
    save_responses(responses, args.output_file)

if __name__ == "__main__":
    main()