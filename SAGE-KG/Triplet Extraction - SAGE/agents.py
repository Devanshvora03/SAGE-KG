import argparse
import json
import re
import os
import glob
import unicodedata
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from charset_normalizer import detect
from llama_index.core.node_parser import SentenceSplitter
from crewai import Agent, Task, Crew, Process
from langchain_community.chat_models import ChatOllama
import sys

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

def detect_encoding(file_path: str) -> str:
    try:
        with open(file_path, 'rb') as f:
            result = detect(f.read())
        return result.get('encoding', 'utf-8') or 'utf-8'
    except Exception:
        return 'utf-8'

def clean_chunk(chunk):
    if not isinstance(chunk, str):
        return ""
    
    chunk = unicodedata.normalize('NFKD', chunk)
    chunk = re.sub(r'\s+', ' ', chunk).strip()
    return chunk

def split_text(text):
    splitter = SentenceSplitter(chunk_size=400, chunk_overlap=50)
    try:
        chunks = splitter.split_text(text)
        return [clean_chunk(chunk) for chunk in chunks]
    except Exception:
        return []

class TripleProcessor:
    def __init__(self, llm, data_folder: str = "data"):
        self.llm = llm
        self.data_folder = data_folder
        
        self.fact_extraction_agent = Agent(
            role='Fact Extractor',
            goal='Extract complete factual statements as short lines preserving all details',
            backstory='Expert at identifying and extracting complete factual information without loss',
            llm=self.llm
        )
        
        self.hierarchy_agent = Agent(
            role='Entity Planner',
            goal='Identify entities and plan triplet structure',
            backstory='Expert in entity recognition and triplet planning',
            llm=self.llm
        )
        
        self.decomposition_agent = Agent(
            role='Triplet Creator',
            goal='Convert fact lines into connected atomic triplets',
            backstory='Expert at breaking facts into connected atomic triplets',
            llm=self.llm
        )
    
    def load_documents(self) -> List[Dict]:
        md_files = glob.glob(os.path.join(self.data_folder, '*.md'))
        
        if not md_files:
            return []
        
        all_chunks = []
        for md_file in md_files:
            try:
                encoding = detect_encoding(md_file)
                with open(md_file, 'r', encoding=encoding) as f:
                    text = f.read()
                
                file_name = os.path.basename(md_file)
                chunks = split_text(text)
                
                for chunk_idx, chunk in enumerate(chunks):
                    if chunk:
                        chunk_data = {
                            "file_id": file_name,
                            "chunk_id": f"{file_name}_{chunk_idx}",
                            "text": chunk
                        }
                        all_chunks.append(chunk_data)
                
            except Exception:
                continue
        
        return all_chunks
    
    def parse_triple(self, triple_str: str) -> Triple:
        triple_str = triple_str.strip()
        triple_str = re.sub(r'^[-•*]\s*', '', triple_str)
        triple_str = re.sub(r'^\d+\.\s*', '', triple_str)
        
        match = re.search(r'\((.*)\)', triple_str)
        if match:
            triple_content = match.group(1)
        else:
            triple_content = triple_str
        
        pattern = r',\s*(?![^()]*\))(?=\s*[a-zA-Z])'
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
            
            match = re.match(r'^\s*[-•*]?\s*\d*\.?\s*\((.*)\)\s*$', line)
            if match:
                triple_content = match.group(1)
                triple = self.parse_triple(f"({triple_content})")
                if triple:
                    triplets.append(triple)
        
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
        chunk_id = chunk_data["chunk_id"]
        file_id = chunk_data["file_id"]
        text = chunk_data["text"]
        
        if not text:
            return []
        
        try:
            fact_extraction_task = Task(
                description=f"""
                Extract all factual statements from this text as short, complete lines.
                
                Rules:
                - One fact per line containing all its support information. 
                - The facts should be able to summarize or reason for events/facts in it.
                - Keep all numbers, dates, names, amounts exactly as written.
                - Include supporting details (context, conditions, specifications).
                - No inference - only stated facts.
                
                TEXT: {text}
                
                Output each fact as a separate line.
                """,
                agent=self.fact_extraction_agent,
                expected_output="List of factual statements as short lines"
            )
            
            hierarchy_task = Task(
                description=f"""
                Analyze fact lines and plan triplet structure, no intermediate triplets required.
                
                Tasks:
                1. List all entity names (use exact names from source).
                    - All numbers, names, dates, should be treated as separate entities with context identifiers etc.
                    - No short forms, use exact names.
                2. Identify which facts are compound (multiple relationships).
                3. Plan how compound facts should be broken down and connect with atomic facts.
                    - Use 1–2 step bridges between facts to preserve context.
                    - Ensure connections across facts through shared entities.
                    - The planning should be done in such a way that it is able to summarize or reason for events/facts in it.
                4. Entity names must hold the descriptive details (phases, categories, types, levels, rounds, etc).
                5. Predicates must stay **simple, generic verbs** (e.g., supports, awards, includes, requires, uses).
                6. Ensure no contextual information (numbers, dates, monetary values) is lost in the plan.
                7. Just give the plan, no triplets needed.
                
                Input facts: {{previous_task_output}}
                
                Output planning analysis only - no triplets yet.
                """,
                agent=self.hierarchy_agent,
                expected_output="Entity analysis and triplet planning"
            )
            
            decomposition_task = Task(
                description=f"""
                Convert fact lines into atomic triplets using the planning analysis while preserving all contextual details and connections.
                
                Rules:
                - Strict format: (subject, predicate, object)
                - Subjects/objects: carry the descriptive and contextual info (phases, rounds, categories, levels, amounts).
                - Predicates: only simple linking verbs (supports, awards, includes, requires, uses).
                - Every numerical or monetary value must be linked to its specific context.
                - Ensure each fact from the plan is represented, no information omitted.
                - Do not use underscores for subject, object.
                
                **Pattern:**
                Input: (entity, action, amount X for target Y in context Z)
                Output:
                - (entity, has_program, context Z program)
                - (context Z program, has_amount, amount X)
                - (context Z program, targets, target Y)
                
                **IMPORTANT:**
                - Output your final triplets in this EXACT format: (subject, predicate, object) with commas separating subject, predicate, and object.
                - Ensure each triplet has exactly two commas, one after the subject and one after the predicate.
                - Each triplet must be on a new line, enclosed in parentheses, with no trailing commas or spaces.
                
                **Example Output:**
                - (subject, has_program, context Z program)
                - (context Z program, has_amount, amount X)
                - (context Z program, targets, target Y)
                """,
                agent=self.decomposition_agent,
                context=[hierarchy_task],
                expected_output="Connected atomic triplets in (subject, predicate, object) format"
            )
            
            crew = Crew(
                agents=[self.fact_extraction_agent, self.hierarchy_agent, self.decomposition_agent],
                tasks=[fact_extraction_task, hierarchy_task, decomposition_task],
                process=Process.sequential
            )
            
            result = crew.kickoff()
            final_output = result.raw if hasattr(result, 'raw') else str(result)
            
            standardized_triplets = self.extract_triplets_from_output(final_output)
            
            if not standardized_triplets:
                return []
            
            for triple in standardized_triplets:
                triple.file_id = file_id
                triple.chunk_id = chunk_id
            
            return self.deduplicate_triplets(standardized_triplets)
            
        except Exception:
            return []
    
    def process_chunks(self, chunks: List[Dict]) -> List[Triple]:
        all_processed_triplets = []
        
        for chunk in tqdm(chunks, desc="Processing Chunks"):
            try:
                processed_triplets = self.process_chunk(chunk)
                all_processed_triplets.extend(processed_triplets)
            except Exception:
                continue
        
        return all_processed_triplets
    
    def save_results(self, all_triplets: List[Triple], txt_file: str = None, json_file: str = None) -> None:
        unique_triplets = self.deduplicate_triplets(all_triplets)
        
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
        
        model_suffix = self.llm.model.split(':')[-1] if hasattr(self.llm, 'model') else "unknown"
        txt_file = txt_file or f"extracted_{model_suffix}.txt"
        json_file = json_file or f"extracted_{model_suffix}.json"
        
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
    
    def run(self) -> None:
        try:
            chunks = self.load_documents()
            if not chunks:
                return
            
            all_triplets = self.process_chunks(chunks)
            
            self.save_results(all_triplets)
            
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser(description="Triple Extraction Processor")
    parser.add_argument("--data-folder", default="data", help="Folder containing input Markdown files")
    parser.add_argument("--model", default="qwen2.5:14b", help="Ollama model name")
    parser.add_argument("--output-json", default="triplets.json", help="Output JSON file for triplets")
    parser.add_argument("--output-txt", default="triplets.txt", help="Output TXT file for triplets")
    args = parser.parse_args()
    
    llm = ChatOllama(model=f"ollama/{args.model}", temperature=0)
    processor = TripleProcessor(llm=llm, data_folder=args.data_folder)
    processor.run()
    
    print("Processing completed!")

if __name__ == "__main__":
    main()