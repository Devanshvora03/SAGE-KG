import argparse
import json
import re
import os
import glob
import unicodedata
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from charset_normalizer import detect
import ollama

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
        return {
            "processing_summary": {
                "total_time_hours": round(self.total_time / 3600, 2),
                "chunks_processed": self.total_chunks_processed,
                "chunks_failed": self.chunks_failed,
                "success_rate": round((self.total_chunks_processed - self.chunks_failed) / max(self.total_chunks_processed, 1) * 100, 2),
                "triplets_extracted": self.total_triplets_extracted,
                "unique_triplets_saved": self.total_unique_triplets,
                "average_triplets_per_chunk": round(self.total_triplets_extracted / max(self.total_chunks_processed - self.chunks_failed, 1), 2),
                "average_unique_triplets_per_chunk": round(self.total_unique_triplets / max(self.total_chunks_processed - self.chunks_failed, 1), 2),
                "average_chunk_processing_time": round(self.average_chunk_time, 2)
            },
            "performance_metrics": {
                "chunks_per_hour": round((self.total_chunks_processed - self.chunks_failed) / max(self.total_time / 3600, 0.01), 2),
                "triplets_per_hour": round(self.total_triplets_extracted / max(self.total_time / 3600, 0.01), 2),
                "unique_triplets_per_hour": round(self.total_unique_triplets / max(self.total_time / 3600, 0.01), 2)
            },
            "error_summary": {
                "total_errors": len(self.errors),
                "error_rate": round(len(self.errors) / max(self.total_chunks_processed, 1) * 100, 2),
                "errors": self.errors[-10:]
            }
        }

def detect_encoding(file_path: str) -> str:
    try:
        with open(file_path, 'rb') as f:
            result = detect(f.read())
        return result.get('encoding', 'utf-8') or 'utf-8'
    except Exception as e:
        logging.warning(f"Encoding detection failed for {file_path}: {e}")
        return 'utf-8'

def clean_chunk(chunk):
    if not isinstance(chunk, str):
        return ""
    
    chunk = unicodedata.normalize('NFKD', chunk)
    chunk = re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f]', '', chunk)
    chunk = chunk.replace('�', '')
    chunk = chunk.replace('{', '{{').replace('}', '}}')
    chunk = chunk.replace(r'\$', '$')
    chunk = re.sub(r'\s+', ' ', chunk).strip()
    return chunk

def split_text(text):
    from llama_index.core.node_parser import SentenceSplitter
    splitter = SentenceSplitter(chunk_size=400, chunk_overlap=50)
    try:
        chunks = splitter.split_text(text)
        return [clean_chunk(chunk) for chunk in chunks]
    except Exception as e:
        logging.error(f"Error splitting text: {e}")
        with open("error_chunks.txt", 'a', encoding='utf-8') as f:
            f.write(f"Text splitting error: {e}\n{'-'*50}\n")
        return []

class TripleProcessor:
    def __init__(self, model: str = "qwen2.5:3b", data_folder: str = "data", max_retries: int = 3):
        self.metrics = ProcessingMetrics()
        self.model = model
        self.data_folder = data_folder
        self.max_retries = max_retries
    
    def load_documents(self) -> List[Dict]:
        load_start = time.time()
        md_files = glob.glob(os.path.join(self.data_folder, '*.md'))
        
        if not md_files:
            logging.error(f"No markdown files found in {self.data_folder}")
            return []
        
        all_chunks = []
        for md_file in md_files:
            try:
                encoding = detect_encoding(md_file)
                with open(md_file, 'r', encoding=encoding) as f:
                    text = f.read()
                
                sections = re.split(r'\*\*Question:\*\*', text.strip(), flags=re.IGNORECASE)
                file_name = os.path.basename(md_file)
                
                section_idx = 0
                for section in sections[1:]:
                    context_match = re.search(r'\*\*Context:\*\*\s*(.*?)(?=\*\*Question:\*\*|$)', section, re.DOTALL | re.IGNORECASE)
                    if not context_match:
                        logging.warning(f"No context found in section {section_idx} of {file_name}")
                        section_idx += 1
                        continue
                    
                    context_text = context_match.group(1).strip()
                    if not context_text:
                        logging.warning(f"Empty context in section {section_idx} of {file_name}")
                        section_idx += 1
                        continue
                    
                    chunks = split_text(context_text)
                    for chunk_idx, chunk in enumerate(chunks):
                        if chunk:
                            chunk_data = {
                                "file_id": file_name,
                                "chunk_id": f"{file_name}_question_{section_idx}_chunk_{chunk_idx}",
                                "text": chunk
                            }
                            all_chunks.append(chunk_data)
                    section_idx += 1
                    
                    logging.info(f"Loaded {len(chunks)} chunks from {file_name} for question {section_idx}")
                    
            except Exception as e:
                logging.error(f"Error reading {md_file}: {e}")
                self.metrics.add_error(f"File loading error: {md_file} - {str(e)}")
                continue
        
        load_time = time.time() - load_start
        logging.info(f"Total loaded chunks: {len(all_chunks)} in {load_time/3600:.2f} hours")
        return all_chunks
    
    def parse_triple(self, triple_str: str) -> Triple:
        triple_str = triple_str.strip()
        triple_str = re.sub(r'^[-•*]\s*', '', triple_str)
        triple_str = re.sub(r'^\d+\.\s*', '', triple_str)
        triple_str = re.sub(r'\(\$(\d+),\s*(\d+),\s*(\d+)\)', r'($\1\2\3)', triple_str)
        triple_str = re.sub(r'\(\$(\d+\.\d+)\)', r'($\1)', triple_str)
        triple_str = re.sub(r'\$\s*(\d+)\s*([mb]illion)', r'$\1 \2', triple_str, flags=re.IGNORECASE)
        triple_str = re.sub(r'\$\s*(\d+)\s*(thousand)', r'$\1 \2', triple_str, flags=re.IGNORECASE)
        
        match = re.search(r'\((.*)\)', triple_str)
        if match:
            triple_content = match.group(1)
        else:
            triple_content = triple_str
        
        pattern = r',\s*(?![^()]*\))(?!\d)(?=\s*[a-zA-Z])'
        parts = re.split(pattern, triple_content, maxsplit=2)
        
        if len(parts) != 3:
            parts = []
            current_part = ""
            i = 0
            paren_count = 0
            
            while i < len(triple_content):
                char = triple_content[i]
                
                if char == '(':
                    paren_count += 1
                    current_part += char
                elif char == ')':
                    paren_count -= 1
                    current_part += char
                elif char == ',' and paren_count == 0:
                    if (i > 0 and i < len(triple_content) - 1 and 
                        triple_content[i-1].isdigit() and triple_content[i+1].isdigit()):
                        current_part += char
                    else:
                        if len(parts) < 2:
                            parts.append(current_part.strip())
                            current_part = ""
                            while i + 1 < len(triple_content) and triple_content[i + 1].isspace():
                                i += 1
                        else:
                            current_part += char
                else:
                    current_part += char
                
                i += 1
            
            if current_part.strip():
                parts.append(current_part.strip())
        
        parts = [part.strip().lower() for part in parts if part.strip()]
        
        if len(parts) != 3:
            logging.warning(f"Invalid triple format: {triple_str} -> {parts}")
            return None
            
        triple = Triple(*parts)
        return triple if self.is_valid_triple(triple) else None
    
    def extract_triplets_from_output(self, output: str) -> List[Triple]:
        triplets = []
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line = re.sub(r'\(\$(\d+),\s*(\d+),\s*(\d+)\)', r'($\1\2\3)', line)
            line = re.sub(r'\(\$(\d+\.\d+)\)', r'($\1)', line)
            line = re.sub(r'\$\s*(\d+)\s*([mb]illion)', r'$\1 \2', line, flags=re.IGNORECASE)
            line = re.sub(r'\$\s*(\d+)\s*(thousand)', r'$\1 \2', line, flags=re.IGNORECASE)
            
            match = re.match(r'^\s*[-•*]?\s*\d*\.?\s*\((.*)\)\s*$', line)
            if match:
                triple_content = match.group(1)
                triple = self.parse_triple(f"({triple_content})")
                if triple:
                    triplets.append(triple)
                    logging.info(f"Extracted triplet: {triple}")
        
        return triplets
    
    def is_valid_triple(self, triple: Triple) -> bool:
        return (triple and
                triple.subject.strip() and
                triple.predicate.strip() and
                triple.object.strip() and
                triple.subject != triple.object and
                triple.subject.lower() not in ["none", "n/a"] and
                triple.predicate.lower() not in ["none", "n/a"] and
                triple.object.lower() not in ["none", "n/a"])
    
    def deduplicate_triplets(self, triplets: List[Triple]) -> List[Triple]:
        seen = set()
        deduplicated = []
        for triple in triplets:
            if triple.key() not in seen:
                seen.add(triple.key())
                deduplicated.append(triple)
        return deduplicated
    
    def process_chunk(self, chunk_data: Dict) -> List[Triple]:
        chunk_start_time = time.time()
        chunk_id = chunk_data["chunk_id"]
        file_id = chunk_data["file_id"]
        text = chunk_data["text"]
        
        if not text:
            logging.info(f"Chunk {chunk_id} has no text to process.")
            return []
        
        try:
            prompt = f"""
            Extract factual triplets from the following text in the format (subject, predicate, object).
            
            Rules:
            - Strict format: (subject, predicate, object)
            - Output each triplet on a new line, enclosed in parentheses, with exactly two commas, no trailing commas or spaces.
            
            TEXT: {text}
            """
            
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0}
            )
            final_output = response['message']['content']
            
            logging.info(f"Raw output for chunk {chunk_id}: {final_output}")
            
            standardized_triplets = self.extract_triplets_from_output(final_output)
            
            if not standardized_triplets:
                logging.warning(f"No triplets extracted from final output for chunk {chunk_id}.")
                return []
            
            for triple in standardized_triplets:
                triple.file_id = file_id
                triple.chunk_id = chunk_id
            
            chunk_time = time.time() - chunk_start_time
            self.metrics.chunk_processing_times.append(chunk_time)
            
            logging.info(f"Extracted {len(standardized_triplets)} triplets from chunk {chunk_id} in {chunk_time/3600:.2f} hours")
            return self.deduplicate_triplets(standardized_triplets)
            
        except Exception as e:
            chunk_time = time.time() - chunk_start_time
            self.metrics.chunk_processing_times.append(chunk_time)
            self.metrics.chunks_failed += 1
            error_msg = f"Error processing chunk {chunk_id}: {str(e)}"
            self.metrics.add_error(error_msg)
            logging.error(error_msg, exc_info=True)
            return []
    
    def process_chunks(self, chunks: List[Dict]) -> List[Triple]:
        logging.info("Starting chunk processing workflow...")
        all_processed_triplets = []
        
        self.metrics.total_chunks_processed = len(chunks)
        
        for chunk in tqdm(chunks, desc="Processing Chunks"):
            try:
                processed_triplets = self.process_chunk(chunk)
                all_processed_triplets.extend(processed_triplets)
                self.metrics.total_triplets_extracted += len(processed_triplets)
                logging.info(f"Successfully processed chunk {chunk['chunk_id']}, resulting in {len(processed_triplets)} triplets.")
            except Exception as e:
                error_msg = f"Critical error processing chunk {chunk['chunk_id']}: {str(e)}"
                self.metrics.add_error(error_msg)
                logging.error(error_msg, exc_info=True)
                continue
        
        return all_processed_triplets
    
    def save_results(self, all_triplets: List[Triple], txt_file: str = None, json_file: str = None, metrics_file: str = None) -> None:
        unique_triplets = self.deduplicate_triplets(all_triplets)
        self.metrics.total_unique_triplets = len(unique_triplets)
        logging.info(f"Deduplicated triplets: {len(all_triplets)} total -> {len(unique_triplets)} unique (removed {len(all_triplets) - len(unique_triplets)} duplicates)")
        
        cleaned_triplets = []
        for triple in unique_triplets:
            cleaned_subject = triple.subject.replace("_", " ")
            cleaned_object = triple.object.replace("_", " ")
            cleaned_triple = Triple(
                subject=cleaned_subject,
                predicate=triple.predicate,
                object=cleaned_object,
                file_id=triple.file_id,
                chunk_id=triple.chunk_id
            )
            cleaned_triplets.append(cleaned_triple)
        
        model_suffix = self.model.split(':')[-1] if ':' in self.model else "unknown"
        txt_file = txt_file or f"extracted_{model_suffix}.txt"
        json_file = json_file or f"extracted_{model_suffix}.json"
        metrics_file = metrics_file or f"metrics_{model_suffix}.json"
        
        with open(txt_file, "w", encoding="utf-8") as f:
            for triple in cleaned_triplets:
                f.write(f"{str(triple)}\n")
        
        serializable_triplets = [
            {
                "subject": t.subject.lower(),
                "predicate": t.predicate.lower(),
                "object": t.object.lower(),
                "file_id": t.file_id,
                "chunk_id": t.chunk_id
            }
            for t in cleaned_triplets
        ]
        
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(serializable_triplets, f, indent=2, ensure_ascii=False)
        
        self.metrics.finalize()
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics.to_dict(), f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved {len(cleaned_triplets)} unique triplets to {txt_file}")
        logging.info(f"Saved {len(cleaned_triplets)} unique triplets with provenance to {json_file}")
        logging.info(f"Saved metrics to {metrics_file}")
    
    def print_metrics_summary(self):
        print("\n" + "="*80)
        print("PROCESSING METRICS SUMMARY")
        print("="*80)
        
        metrics_dict = self.metrics.to_dict()
        
        summary = metrics_dict["processing_summary"]
        print(f"\n📊 PROCESSING OVERVIEW:")
        print(f"   Total Time: {summary['total_time_hours']:.2f} hours")
        print(f"   Chunks Processed: {summary['chunks_processed']}")
        print(f"   Chunks Failed: {summary['chunks_failed']}")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        print(f"   Total Triplets Extracted: {summary['triplets_extracted']}")
        print(f"   Unique Triplets Saved: {summary['unique_triplets_saved']}")
        print(f"   Avg Triplets/Chunk: {summary['average_triplets_per_chunk']:.1f}")
        print(f"   Avg Unique Triplets/Chunk: {summary['average_unique_triplets_per_chunk']:.1f}")
        print(f"   Avg Time/Chunk: {summary['average_chunk_processing_time']:.2f} hours")
        
        performance = metrics_dict["performance_metrics"]
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   Chunks/Hour: {performance['chunks_per_hour']:.2f}")
        print(f"   Total Triplets/Hour: {performance['triplets_per_hour']:.2f}")
        print(f"   Unique Triplets/Hour: {performance['unique_triplets_per_hour']:.2f}")
        
        error_summary = metrics_dict["error_summary"]
        print(f"\n❌ ERROR SUMMARY:")
        print(f"   Total Errors: {error_summary['total_errors']}")
        print(f"   Error Rate: {error_summary['error_rate']:.1f}%")
        if error_summary['errors']:
            print(f"   Recent Errors:")
            for i, error in enumerate(error_summary['errors'][-5:], 1):
                print(f"     {i}. {error}")
        
        print("="*80)
    
    def run(self, txt_file: str = None, json_file: str = None, metrics_file: str = None) -> None:
        try:
            logging.info("Starting triple processing pipeline...")
            
            chunks = self.load_documents()
            if not chunks:
                logging.error("No chunks to process")
                return
            
            all_triplets = self.process_chunks(chunks)
            
            self.save_results(all_triplets, txt_file, json_file, metrics_file)
            
            self.print_metrics_summary()
            
        except Exception as e:
            error_msg = f"A critical error occurred in the main execution run: {str(e)}"
            self.metrics.add_error(error_msg)
            logging.error(error_msg, exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="Triple Extraction Processor")
    parser.add_argument("--data-folder", default="data", help="Folder containing input Markdown files")
    parser.add_argument("--model", default="qwen2.5:14b", help="Ollama model name")
    parser.add_argument("--output-txt", default=None, help="Output TXT file for triplets (default: extracted_<model>.txt)")
    parser.add_argument("--output-json", default=None, help="Output JSON file for triplets (default: extracted_<model>.json)")
    parser.add_argument("--metrics-file", default=None, help="Output metrics JSON file (default: metrics_<model>.json)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("triple_processor.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    print("Starting Triple Processor with Enhanced Metrics...")
    print("="*60)
    
    processor = TripleProcessor(model=args.model, data_folder=args.data_folder)
    
    start_time = time.time()
    print(f"Initialization completed in {time.time() - start_time:.2f} seconds")
    print("-" * 60)
    
    processor.run(args.output_txt, args.output_json, args.metrics_file)
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    print("Processing completed! Check the metrics files for detailed analysis.")

if __name__ == "__main__":
    main()