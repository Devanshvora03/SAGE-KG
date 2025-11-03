import os
import re
import json
import logging
import unicodedata
from typing import List, Dict, Any
from openie import StanfordOpenIE
from charset_normalizer import detect
from llama_index.core.node_parser import SentenceSplitter
import spacy
import tiktoken
import argparse
import yaml

def setup_logging(log_file: str = "openie_triplet_extractor.log") -> logging.Logger:
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()


def detect_encoding(file_path: str) -> str:
    """Detect file encoding using charset_normalizer."""
    try:
        with open(file_path, "rb") as f:
            result = detect(f.read())
        return result.get("encoding", "utf-8") or "utf-8"
    except Exception as e:
        logger.warning(f"Encoding detection failed for {file_path}: {e}")
        return "utf-8"


def clean_chunk(chunk: str) -> str:
    """Normalize, clean, and sanitize text chunks."""
    if not isinstance(chunk, str):
        return ""
    chunk = unicodedata.normalize("NFKD", chunk)
    chunk = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f]", "", chunk)
    chunk = chunk.replace("�", "")
    chunk = chunk.replace("{", "{{").replace("}", "}}")
    chunk = chunk.replace(r"\$", "$")
    return re.sub(r"\s+", " ", chunk).strip()


def normalize_relation(relation: str) -> str:
    """Normalize OpenIE relations for consistency."""
    return relation.strip().replace(" ", "_").lower()


class TokenCounter:
    """Utility class for counting tokens using tiktoken."""

    def __init__(self):
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoder = None
            logger.warning("tiktoken not available, using character-based estimation")

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.encoder:
            return len(self.encoder.encode(text))
        return len(text) // 4


def split_text(text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> List[str]:
    """Split text into smaller chunks using SentenceSplitter."""
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    try:
        sections = re.split(r"\*\*Question:\*\*", text.strip(), flags=re.IGNORECASE)
        all_chunks = []
        for section in sections[1:]:  
            context_match = re.search(
                r"Context:\s*(.*?)(?=\*\*Question:\*\*|$)",
                section,
                re.DOTALL | re.IGNORECASE
            )
            if context_match:
                facts_text = context_match.group(1).strip()
                if facts_text:
                    chunks = splitter.split_text(facts_text)
                    all_chunks.extend([clean_chunk(c) for c in chunks if c])
        return all_chunks
    except Exception as e:
        logger.error(f"Error splitting text: {e}")
        return []



def extract_triplets_from_markdown(
    folder_path: str,
    output_file: str,
    max_chunk_chars: int = 100000,
    log_every: int = 50
):
    """Extract subject-predicate-object triplets from markdown files."""
    nlp = spacy.load("en_core_web_sm")
    token_counter = TokenCounter()
    all_triplets: List[Dict[str, Any]] = []
    oversized_chunks: List[Dict[str, Any]] = []
    total_chunks = 0

    md_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".md")]
    if not md_files:
        logger.error(f"No markdown files found in {folder_path}.")
        return

    with StanfordOpenIE(timeout=180000) as client:
        for md_file in md_files:
            file_path = os.path.join(folder_path, md_file)
            try:
                encoding = detect_encoding(file_path)
                with open(file_path, "r", encoding=encoding) as f:
                    text = f.read()

                sections = [s.strip() for s in re.split(r"---+", text) if s.strip()]
                logger.info(f"Processing {md_file} with {len(sections)} sections.")

                for section_idx, section in enumerate(sections):
                    chunks = split_text(section)
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_name = f"{md_file}_context_{section_idx}_{chunk_idx}"
                        total_chunks += 1
                        token_count = token_counter.count_tokens(chunk)

                        if len(chunk) > max_chunk_chars:
                            oversized_chunks.append({
                                "chunk_id": chunk_name,
                                "file_id": md_file,
                                "size": len(chunk),
                                "tokens": token_count
                            })
                            logger.warning(f"Oversized chunk {chunk_name}: {len(chunk)} chars")

                        try:
                            triplets = client.annotate(chunk)
                            for triple in triplets:
                                all_triplets.append({
                                    "subject": triple["subject"].strip().lower(),
                                    "predicate": normalize_relation(triple["relation"]),
                                    "object": triple["object"].strip().lower(),
                                    "file_id": md_file,
                                    "chunk_id": chunk_name
                                })
                        except Exception as e:
                            logger.error(f"Error processing {chunk_name}: {e}")

                        if total_chunks % log_every == 0:
                            logger.info(f"Processed {total_chunks} chunks so far...")

            except Exception as e:
                logger.error(f"Error reading or processing {md_file}: {e}")

    
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_triplets, f, indent=2, ensure_ascii=False)

    logger.info(f"Extraction complete: {len(all_triplets)} triplets saved to {output_file}")
    logger.info(f"Total chunks processed: {total_chunks}")
    if oversized_chunks:
        logger.warning(f"{len(oversized_chunks)} oversized chunks detected.")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML if provided."""
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="OpenIE-based Triplet Extractor")
    parser.add_argument("--data_dir", required=True, help="Path to folder containing markdown files")
    parser.add_argument("--output_file", required=True, help="Output JSON file path")
    parser.add_argument("--config", default=None, help="Optional config.yaml path")
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = config.get("data_dir", args.data_dir)
    output_file = config.get("output_file", args.output_file)
    max_chunk_chars = config.get("max_chunk_chars", 100000)

    extract_triplets_from_markdown(
        folder_path=data_dir,
        output_file=output_file,
        max_chunk_chars=max_chunk_chars
    )


if __name__ == "__main__":
    main()
