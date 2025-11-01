"""
Triple Processor
----------------
Extracts (subject, predicate, object) triples from text documents using the KGGen API.

Usage:
    python triple_processor.py --api_key <OPENAI_API_KEY> --data_dir data --output_dir outputs

Environment Variables (optional):
    OPENAI_API_KEY     If not provided via CLI, read from environment
"""

import os
import re
import sys
import json
import time
import glob
import logging
import warnings
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from tqdm import tqdm
from charset_normalizer import detect
from llama_index.core.node_parser import SentenceSplitter
from kg_gen import KGGen  
import litellm

litellm.cache = None

def setup_logger(log_path: str = "logs/triple_processor.log") -> logging.Logger:
    """Configure logging for both console and file output."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("openai").setLevel(logging.CRITICAL)
    return logger

@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    file_id: Optional[str] = None
    chunk_id: Optional[str] = None

    def __str__(self) -> str:
        return f"({self.subject}, {self.predicate}, {self.object})"

    def key(self) -> Tuple[str, str, str]:
        return (self.subject.lower(), self.predicate.lower(), self.object.lower())


@dataclass
class ProcessingMetrics:
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_chunks_processed: int = 0
    total_triplets_extracted: int = 0
    total_unique_triplets: int = 0
    chunks_failed: int = 0
    chunk_processing_times: List[float] = field(default_factory=list)
    average_chunk_time: float = 0.0
    errors: List[str] = field(default_factory=list)

    def add_error(self, error_msg: str):
        self.errors.append(error_msg)

    def finalize(self):
        self.end_time = time.time()
        if self.chunk_processing_times:
            self.average_chunk_time = sum(self.chunk_processing_times) / len(self.chunk_processing_times)

    @property
    def total_time(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def to_dict(self) -> Dict:
        success_rate = round((self.total_chunks_processed - self.chunks_failed) / max(self.total_chunks_processed, 1) * 100, 2)
        return {
            "processing_summary": {
                "total_time_seconds": round(self.total_time, 2),
                "total_chunks_processed": self.total_chunks_processed,
                "chunks_failed": self.chunks_failed,
                "success_rate": success_rate,
                "triplets_extracted": self.total_triplets_extracted,
                "unique_triplets_saved": self.total_unique_triplets,
                "avg_triplets_per_chunk": round(self.total_triplets_extracted / max(self.total_chunks_processed, 1), 2),
                "avg_chunk_time": round(self.average_chunk_time, 2),
            },
            "performance_metrics": {
                "chunks_per_minute": round((self.total_chunks_processed / max(self.total_time / 60, 0.01)), 2),
                "triplets_per_minute": round((self.total_triplets_extracted / max(self.total_time / 60, 0.01)), 2),
            },
            "error_summary": {
                "total_errors": len(self.errors),
                "recent_errors": self.errors[-10:]
            }
        }

def detect_encoding(file_path: str) -> str:
    """Detect the encoding of a text file."""
    try:
        with open(file_path, "rb") as f:
            result = detect(f.read())
        return result.get("encoding", "utf-8") or "utf-8"
    except Exception:
        return "utf-8"


def clean_chunk(chunk: str) -> str:
    """Normalize and sanitize text chunk."""
    if not isinstance(chunk, str):
        return ""
    chunk = unicodedata.normalize("NFKD", chunk)
    chunk = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f]", "", chunk)
    chunk = chunk.replace("�", "")
    chunk = chunk.replace("{", "{{").replace("}", "}}")
    chunk = re.sub(r"\s+", " ", chunk).strip()
    return chunk


def split_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Split large text into manageable chunks."""
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    try:
        chunks = splitter.split_text(text)
        return [clean_chunk(c) for c in chunks]
    except Exception:
        return []


class TripleProcessor:
    def __init__(self, api_key: str, data_folder: str, output_dir: str, logger: logging.Logger, max_retries: int = 3):
        self.api_key = api_key
        self.data_folder = data_folder
        self.output_dir = output_dir
        self.max_retries = max_retries
        self.logger = logger
        self.metrics = ProcessingMetrics()

        
        os.makedirs(output_dir, exist_ok=True)
        self.txt_file = os.path.join(output_dir, "triples.txt")
        self.json_file = os.path.join(output_dir, "triples.json")
        self.metrics_file = os.path.join(output_dir, "metrics.json")

        
        self.all_triplets: List[Triple] = []

    def load_documents(self) -> List[Dict]:
        """Load and split markdown files from the data directory."""
        md_files = glob.glob(os.path.join(self.data_folder, "*.md"))
        all_chunks = []

        for file_path in md_files:
            file_name = os.path.basename(file_path)
            encoding = detect_encoding(file_path)

            try:
                with open(file_path, "r", encoding=encoding, errors="replace") as f:
                    text = f.read()
            except Exception as e:
                self.logger.error(f"Failed to read {file_name}: {e}")
                self.metrics.add_error(str(e))
                continue

            sections = re.split(r"\*\*Question:\*\*", text.strip(), flags=re.IGNORECASE)
            if sections and not sections[0]:
                sections.pop(0)

            for section_idx, section in enumerate(sections):
                context_match = re.search(r"Context:\s*(.*?)(?=\*\*Question:\*\*|$)", section, re.DOTALL | re.IGNORECASE)
                context_text = context_match.group(1).strip() if context_match else section.strip()

                for idx, chunk in enumerate(split_text(context_text)):
                    all_chunks.append({
                        "file_id": file_name,
                        "chunk_id": f"{file_name}_{section_idx}_{idx}",
                        "text": chunk,
                    })

        self.metrics.total_chunks_processed = len(all_chunks)
        self.logger.info(f"Loaded {len(all_chunks)} chunks.")
        return all_chunks

    def initialize_kggen(self):
        """Initialize KGGen with OpenAI key."""
        os.environ["OPENAI_API_KEY"] = self.api_key
        return KGGen(model="openai/gpt-4o", temperature=0.0)

    def process_chunk(self, kg: KGGen, chunk_data: Dict) -> List[Triple]:
        """Generate and deduplicate triples for a text chunk."""
        start_time = time.time()
        file_id, chunk_id, text = chunk_data.values()
        if not text:
            return []

        try:
            graph = kg.generate(input_data=text, context=f"Content from {file_id}")
            raw_triplets = [Triple(*[str(r).lower() for r in rel], file_id=file_id, chunk_id=chunk_id)
                            for rel in getattr(graph, "relations", []) if len(rel) == 3]

            self.metrics.total_triplets_extracted += len(raw_triplets)
            self.metrics.chunk_processing_times.append(time.time() - start_time)

            unique_triplets = self.deduplicate_triplets(raw_triplets)
            self.all_triplets.extend(unique_triplets)
            self.save_intermediate_results(unique_triplets)
            return unique_triplets

        except Exception as e:
            self.metrics.chunks_failed += 1
            self.metrics.add_error(f"Chunk {chunk_id} failed: {e}")
            return []

    def deduplicate_triplets(self, triplets: List[Triple]) -> List[Triple]:
        seen = {t.key() for t in self.all_triplets}
        return [t for t in triplets if t.key() not in seen]

    def save_intermediate_results(self, triplets: List[Triple]):
        """Save triples incrementally to files."""
        try:
            with open(self.txt_file, "a", encoding="utf-8") as f:
                for t in triplets:
                    f.write(f"{str(t)}\n")

            existing = []
            if os.path.exists(self.json_file):
                with open(self.json_file, "r", encoding="utf-8") as f:
                    try:
                        existing = json.load(f)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Corrupted JSON file: {self.json_file}, restarting.")

            existing.extend([t.__dict__ for t in triplets])
            with open(self.json_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)

            
            self.metrics.total_unique_triplets = len(self.deduplicate_triplets(self.all_triplets))
            self.metrics.finalize()
            with open(self.metrics_file, "w", encoding="utf-8") as f:
                json.dump(self.metrics.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save intermediate results: {e}")
            self.metrics.add_error(f"Save failed: {e}")

    def run(self):
        """Main execution pipeline."""
        self.logger.info("Starting Triple Processor...")
        chunks = self.load_documents()
        if not chunks:
            self.logger.error("No text chunks found in the data directory.")
            return

        kg = self.initialize_kggen()
        for chunk in tqdm(chunks, desc="Processing Chunks"):
            self.process_chunk(kg, chunk)

        self.metrics.finalize()
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics.to_dict(), f, indent=2, ensure_ascii=False)

        self.logger.info("Processing complete.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Triple Extraction Pipeline")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--data_dir", type=str, default="data", help="Input data directory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: Please provide an OpenAI API key using --api_key or set OPENAI_API_KEY in the environment.")
        sys.exit(1)

    logger = setup_logger()
    processor = TripleProcessor(api_key=args.api_key, data_folder=args.data_dir, output_dir=args.output_dir, logger=logger)
    processor.run()