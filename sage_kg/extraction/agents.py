"""
SAGE-KG Stage 1: Sequential Uncertainty Resolution (SUR) triplet extraction.

Paper mapping
-------------
Corpus chunk c is processed by three CrewAI agents in a fixed order:

    c  --A_sem-->  F  --A_str-->  H  --A_rep-->  G

  * Fact Extractor   (A_sem): resolve semantic uncertainty.
      Output F is free-form factual lines. No entities or schema yet.
  * Entity Planner   (A_str): resolve structural uncertainty.
      Output H is an entity--relation plan (also free-form).
  * Triplet Creator  (A_rep): resolve representational uncertainty.
      Output G is parseable (subject, predicate, object) triples.

Only the last stage is parsed programmatically. Intermediate outputs are
passed as CrewAI task context so small models can focus on reasoning
rather than format compliance (see the paper, Intermediate Output Tolerance).

Input:  a folder of .md / .txt documents.
Output: triples_{model}_{timestamp}.json with keys
        subject, predicate, object, file_id, chunk_id
        (this is the schema consumed by sage_kg/construction/create_kg.py).
"""

import json
import re
import os
import glob
import unicodedata
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from charset_normalizer import detect
from crewai import Agent, Task, Crew, Process
from langchain_community.chat_models import ChatOllama
import sys
from argparse import ArgumentParser, Namespace

logger = logging.getLogger(__name__)


@dataclass
class Triple:
    """One (subject, predicate, object) fact, tagged with its source chunk."""
    subject: str
    predicate: str
    object: str
    file_id: Optional[str] = None
    chunk_id: Optional[str] = None

    def __str__(self) -> str:
        return f"({self.subject}, {self.predicate}, {self.object})"

    def key(self) -> Tuple[str, str, str]:
        """Case-insensitive identity used for de-duplication."""
        return (self.subject.lower(), self.predicate.lower(), self.object.lower())


class TripleProcessor:
    """End-to-end SUR extractor: load docs → chunk → three agents → save triples."""

    def __init__(self, llm, data_folder: str, chunk_size: int = 200, overlap: int = 25,
                 max_retries: int = 3, output_dir: str = "output"):
        self.llm = llm
        self.data_folder = Path(data_folder).expanduser().resolve()
        # Paper experiments used 200-token windows / 25-token overlap.
        # This standalone script approximates that with word windows.
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_retries = max_retries
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.total_files = 0
        self.total_chunks = 0
        self.total_triplets = 0
        self.total_unique = 0
        self.errors: List[str] = []
        self.start_time = time.time()

        self._init_agents()

    def _init_agents(self):
        """Create the three SUR agents. Roles match Figure 1 in the paper."""
        # A_sem: what the text states, without committing to a schema.
        self.fact_extraction_agent = Agent(
            role='Fact Extractor',
            goal='Extract complete factual statements as short lines preserving all details',
            backstory='Expert at identifying and extracting complete factual information without loss',
            llm=self.llm,
            verbose=True
        )

        # A_str: entities, bridges, and relation plan over the extracted facts.
        self.planner_agent = Agent(
            role='Entity Planner',
            goal='Identify entities and plan triplet structure covering all the facts',
            backstory='Expert in entity recognition and triplet planning with perfect recall and accuracy',
            llm=self.llm,
            verbose=True
        )

        # A_rep: serialize only the planned structure as atomic triples.
        self.triplet_creator_agent = Agent(
            role='Triplet Creator',
            goal='Convert fact lines into connected atomic triplets covering all the information and entities given in the plan',
            backstory='Expert at breaking facts into connected atomic triplets with perfect recall',
            llm=self.llm,
            verbose=True
        )

    @staticmethod
    def detect_encoding(file_path: Path) -> str:
        """Guess file encoding so mixed corpus dumps do not fail on open()."""
        try:
            with open(file_path, 'rb') as f:
                result = detect(f.read(8192 * 4))
            return result.get('encoding', 'utf-8') or 'utf-8'
        except Exception:
            return 'utf-8'

    @staticmethod
    def clean_chunk(chunk: str) -> str:
        """Normalize unicode and escape braces (CrewAI / LangChain treat {} as templates)."""
        if not isinstance(chunk, str):
            return ""
        chunk = unicodedata.normalize('NFKD', chunk)
        chunk = re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f]', '', chunk)
        chunk = chunk.replace('�', '')
        chunk = chunk.replace('{', '{{').replace('}', '}}')
        chunk = chunk.replace(r'\$', '$')
        chunk = re.sub(r'\s+', ' ', chunk).strip()
        return chunk

    def split_text(self, text: str) -> List[str]:
        """Sliding word window with overlap so facts that straddle a cut are not lost."""
        words = text.split()
        if not words:
            return []

        chunks = []
        start = 0
        n = len(words)

        while start < n:
            end = min(start + self.chunk_size, n)
            chunk = " ".join(words[start:end])
            chunk = self.clean_chunk(chunk)
            if chunk:
                chunks.append(chunk)
            if end == n:
                break
            start = max(0, end - self.overlap)

        return chunks

    def load_documents(self, patterns: List[str]) -> List[Dict]:
        """Read matching files and emit {file_id, chunk_id, text} records."""
        all_files = []
        for pat in patterns:
            all_files.extend(self.data_folder.glob(pat))

        if not all_files:
            logger.error(f"No files found matching patterns in {self.data_folder}")
            return []

        all_files = sorted(set(all_files))
        self.total_files = len(all_files)
        logger.info(f"Found {self.total_files} files")

        chunks_list = []

        for fpath in all_files:
            try:
                enc = self.detect_encoding(fpath)
                with open(fpath, 'r', encoding=enc, errors='replace') as f:
                    content = f.read()

                chunks = self.split_text(content)

                for i, chunk_text in enumerate(chunks):
                    if not chunk_text.strip():
                        continue
                    chunks_list.append({
                        "file_id": fpath.name,
                        "chunk_id": f"{fpath.stem}_chunk_{i:03d}",
                        "text": chunk_text
                    })

                logger.info(f"{fpath.name}: {len(chunks)} chunks")

            except Exception as e:
                msg = f"Failed to load {fpath.name}: {e}"
                self.errors.append(msg)
                logger.error(msg)

        self.total_chunks = len(chunks_list)
        logger.info(f"Total chunks to process: {self.total_chunks}")
        return chunks_list

    def parse_triple(self, line: str) -> Optional[Triple]:
        """Parse one (s, p, o) line. Commas inside nested parentheses are kept."""
        line = line.strip()
        line = re.sub(r'^[-•*]\s*', '', line)
        line = re.sub(r'^\d+\.\s*', '', line)

        match = re.search(r'\((.*)\)', line)
        if match:
            content = match.group(1)
        else:
            content = line

        # Split on top-level commas only, so "(a, has (b, c), d)" still works.
        parts = []
        current = ""
        paren = 0
        i = 0
        while i < len(content):
            c = content[i]
            if c == '(':
                paren += 1
            elif c == ')':
                paren -= 1
            if c == ',' and paren == 0:
                parts.append(current.strip())
                current = ""
                i += 1
                while i < len(content) and content[i].isspace():
                    i += 1
                continue
            current += c
            i += 1

        if current.strip():
            parts.append(current.strip())

        if len(parts) != 3:
            logger.debug(f"Could not parse triple: {line}")
            return None

        s, p, o = [x.strip().lower() for x in parts]
        triple = Triple(s, p, o)

        if self.is_valid_triple(triple):
            return triple
        return None

    @staticmethod
    def is_valid_triple(t: Triple) -> bool:
        """Drop empty, placeholder, or degenerate (subject == object) triples."""
        if not (t.subject and t.predicate and t.object):
            return False
        bad = {"none", "n/a", ""}
        return not (t.subject in bad or t.predicate in bad or t.object in bad or t.subject == t.object)

    def extract_triplets_from_output(self, output: str) -> List[Triple]:
        """Keep only lines that parse as valid triples from the Triplet Creator."""
        triplets = []
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            t = self.parse_triple(line)
            if t:
                triplets.append(t)
        return triplets

    def deduplicate_triplets(self, triplets: List[Triple]) -> List[Triple]:
        seen = set()
        unique = []
        for t in triplets:
            k = t.key()
            if k not in seen:
                seen.add(k)
                unique.append(t)
        return unique

    def process_chunk(self, chunk: Dict) -> List[Triple]:
        """Run SUR on one chunk: Fact Extractor → Planner → Triplet Creator."""
        cid = chunk["chunk_id"]
        fid = chunk["file_id"]
        text = chunk["text"].strip()

        if len(text) < 20:
            return []

        try:
            # Stage 1 (semantic): facts only. Not parsed; consumed as context.
            fact_task = Task(
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

            # Stage 2 (structural): plan entities and bridges. Still not parsed.
            planner_task = Task(
                description=f"""
                Analyze fact lines and plan triplet structure, no intermediate triplets required.
                
                Tasks:
                1. List all entity names (use exact names from source).
                    - All numbers, names, dates, should be treated as separate entities with context identifiers etc.
                    - No short forms, use exact names.
                2. Plan how compound facts should be broken down and connect with atomic facts.
                    - Use 1–2 step bridges between facts to preserve context.
                    - Make bridge triplets connected with consistent logical entities, not long triplets.
                    - Ensure connections across facts through shared entities with consistent names.
                    - The planning should be done in such a way that it is able to summarize or reason for events/facts in it.
                3. Ensure consitent entity names in different triplets for better graph connection
                4. Predicates must stay **simple, generic verbs** (e.g., supports, awards, includes, requires, uses).
                5. Just give the plan, no triplets needed.
                
                Output planning analysis only - no triplets yet.
                """,
                agent=self.planner_agent,
                context=[fact_task],
                expected_output="Entity analysis and triplet planning"
            )

            # Stage 3 (representational): the only stage whose output is parsed.
            triplet_task = Task(
                description=f"""
                Convert fact lines into atomic triplets using the planning analysis.
                
                Rules:
                - **Format: (subject, predicate, object) - one per line, no underscores**.
                - **All the entities should be used in the triplets which are mentioned in the plan, no loss**.
                - **Ensure consitent entity names in different triplets for better graph connection**.
                
                Example pattern - if fact is "entity does action with amount X for target Y in context Z":
                - (entity, has program, context Z program)
                - (context Z program, has amount, amount X)
                - (context Z program, targets, target Y)
                
                Output format: Each line must be exactly (subject, predicate, object) with two commas.
                """,
                agent=self.triplet_creator_agent,
                context=[planner_task, fact_task],
                expected_output="Connected atomic triplets in (subject, predicate, object) format"
            )

            # Sequential process = paper's S0 → S1 → S2 → S3 state evolution.
            crew = Crew(
                agents=[self.fact_extraction_agent, self.planner_agent, self.triplet_creator_agent],
                tasks=[fact_task, planner_task, triplet_task],
                process=Process.sequential,
                verbose=True
            )

            result = crew.kickoff()
            raw = result.raw if hasattr(result, 'raw') else str(result)

            triplets = self.extract_triplets_from_output(raw)

            for t in triplets:
                t.file_id = fid
                t.chunk_id = cid

            logger.info(f"{cid} → {len(triplets)} triplets")
            return self.deduplicate_triplets(triplets)

        except Exception as e:
            msg = f"Chunk {cid} failed: {str(e)}"
            self.errors.append(msg)
            logger.exception(msg)
            return []

    def run(self, file_patterns: List[str]):
        logger.info("Starting triple extraction pipeline")
        logger.info(f"Input directory:  {self.data_folder}")
        logger.info(f"Model:            {getattr(self.llm, 'model', 'unknown')}")
        logger.info(f"Chunk size:       {self.chunk_size} words ± {self.overlap} overlap")

        chunks = self.load_documents(file_patterns)
        if not chunks:
            print("No chunks to process. Exiting.")
            return [], None

        all_triplets: List[Triple] = []

        for chunk in tqdm(chunks, desc="Processing chunks"):
            triples = self.process_chunk(chunk)
            all_triplets.extend(triples)
            self.total_triplets += len(triples)

        unique_triplets = self.deduplicate_triplets(all_triplets)
        self.total_unique = len(unique_triplets)

        json_path = self._save_results(unique_triplets)
        self._print_summary()
        return unique_triplets, json_path

    def extract_text(self, text: str, file_id: str = "inline") -> List[Triple]:
        """Run SUR on a single string. Useful as a library call without a data folder."""
        chunks = self.split_text(text)
        all_triplets: List[Triple] = []
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < 20:
                continue
            triples = self.process_chunk({
                "file_id": file_id,
                "chunk_id": f"{file_id}_chunk_{i:03d}",
                "text": chunk_text,
            })
            all_triplets.extend(triples)
        return self.deduplicate_triplets(all_triplets)

    def _save_results(self, triplets: List[Triple]):
        """Write .txt (readable) and .json (input to create_kg.py)."""
        model_part = getattr(self.llm, 'model', 'model').replace(':', '_').replace('/', '_')
        ts = time.strftime("%Y%m%d_%H%M%S")

        stem = f"triples_{model_part}_{ts}"
        txt_path = self.output_dir / f"{stem}.txt"
        json_path = self.output_dir / f"{stem}.json"

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"# {len(triplets)} unique triples  |  model: {model_part}  |  {time.ctime()}\n\n")
            for t in triplets:
                f.write(f"{t}\n")

        # Keys match Results/Triplets/*.json and sage_kg/construction/create_kg.py.
        data = [{
            "subject": t.subject.lower(),
            "predicate": t.predicate.lower(),
            "object": t.object.lower(),
            "file_id": t.file_id,
            "chunk_id": t.chunk_id
        } for t in triplets]

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(triplets)} unique triples")
        logger.info(f"  → {txt_path.name}")
        logger.info(f"  → {json_path.name}")
        return json_path

    def _print_summary(self):
        duration = time.time() - self.start_time
        print("\n" + "═" * 70)
        print("EXTRACTION SUMMARY")
        print("═" * 70)
        print(f"Files           : {self.total_files:4d}")
        print(f"Chunks          : {self.total_chunks:4d}")
        print(f"Triples (total) : {self.total_triplets:5d}")
        print(f"Unique triples  : {self.total_unique:5d}")
        print(f"Duration        : {duration:.1f} s  ({duration/60:.1f} min)")
        print(f"Triples / chunk : {self.total_triplets / max(1, self.total_chunks):.1f}")
        print(f"Errors          : {len(self.errors)}")
        print("═" * 70)
        if self.errors:
            print("Last few errors:")
            for e in self.errors[-3:]:
                print(f"  • {e}")
        print()


def setup_logging(model_name: str, log_dir: Path):
    log_dir.mkdir(exist_ok=True)
    safe = re.sub(r'[:/\\]', '_', model_name)
    logfile = log_dir / f"extract_{safe}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        handlers=[
            logging.FileHandler(logfile, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="SAGE-KG SUR extraction: Fact Extractor → Planner → Triplet Creator"
    )
    parser.add_argument("models", nargs="+", help="Ollama model names, e.g. qwen2.5:14b")
    parser.add_argument("--data", "-d", default="./data", help="Input folder of .md / .txt files")
    parser.add_argument("--output", "-o", default="./output", help="Output folder for triples JSON")
    parser.add_argument("--patterns", nargs="+", default=["*.md", "*.txt"],
                        help="File glob patterns")
    parser.add_argument("--chunk", type=int, default=200,
                        help="Max words per chunk (paper: 200)")
    parser.add_argument("--overlap", type=int, default=25,
                        help="Word overlap between chunks (paper: 25)")
    return parser.parse_args()


def main():
    args = parse_args()

    for model_name in args.models:
        print(f"\n{'═'*65}")
        print(f" MODEL: {model_name}")
        print(f"  Data → {args.data}")
        print(f"Output → {args.output}")
        print('═'*65)

        log_dir = Path(args.output) / "logs"
        setup_logging(model_name, log_dir)

        # Temperature 0: extraction should be deterministic, not creative.
        llm = ChatOllama(model=model_name, temperature=0)

        processor = TripleProcessor(
            llm=llm,
            data_folder=args.data,
            chunk_size=args.chunk,
            overlap=args.overlap,
            output_dir=args.output
        )

        processor.run(args.patterns)

        print(f"Finished {model_name}\n")


if __name__ == "__main__":
    main()
