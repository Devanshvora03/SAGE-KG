import argparse
import json
import os
import time
from datetime import datetime
from typing import List, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required.")
client = OpenAI(api_key=api_key)

SCORING_RUBRICS = {
    "completeness": {
        "description": "whether the answer includes ALL important facts and distinct points from the ground truth, allowing consistent factual additions",
        "rubric": (
            "Scoring Guide (0-10):\n"
            "- 10: Fully captures all Ground Truth facts with possibly helpful relevant detail.\n"
            "- 8-9: Covers most facts clearly with minor omissions or some additional context that does not contradict.\n"
            "- 6-7: Captures some key facts but misses several points or adds moderately extraneous/non-contradictory info.\n"
            "- 4-5: Partial coverage with many omissions or questionable additional info.\n"
            "- 1-3: Contains little of the Ground Truth facts.\n"
            "- 0: No relevant facts are present or answer is misleading."
        )
    },
    "accuracy": {
        "description": "whether the answer is factually correct compared to ground truth, tolerating consistent elaborations",
        "rubric": (
            "Scoring Guide (0-10):\n"
            "- 10: Fully accurate; no factual errors.\n"
            "- 8-9: Mostly accurate with minor trivial errors or consistent additions.\n"
            "- 6-7: Some factual inaccuracies or minor misinterpretations.\n"
            "- 4-5: Several incorrect points.\n"
            "- 1-3: Largely incorrect.\n"
            "- 0: Completely false or unrelated."
        )
    },
    "knowledgeability": {
        "description": "whether the answer shows accurate domain knowledge consistent with the ground truth, allowing relevant expansions",
        "rubric": (
            "Scoring Guide (0-10):\n"
            "- 10: Fully matches domain knowledge with clarity.\n"
            "- 8-9: Mostly aligns with minor gaps or some relevant added detail.\n"
            "- 6-7: Exhibits some understanding but also gaps.\n"
            "- 4-5: Limited knowledge shown.\n"
            "- 1-3: Minimal or incorrect domain knowledge.\n"
            "- 0: No relevant domain knowledge."
        )
    },
    "relevance": {
        "description": "whether the answer stays on-topic using only ground truth facts or consistent relevant information",
        "rubric": (
            "Scoring Guide (0-10):\n"
            "- 10: Entirely relevant and on-topic.\n"
            "- 8-9: Mostly relevant; minimal off-topic content.\n"
            "- 6-7: Some minor digressions.\n"
            "- 4-5: Noticeable off-topic content.\n"
            "- 1-3: Barely related.\n"
            "- 0: Completely irrelevant."
        )
    },
    "logical_coherence": {
        "description": "whether the answer presents the ground truth facts clearly and logically, with possible well-integrated expansions",
        "rubric": (
            "Scoring Guide (0-10):\n"
            "- 10: Clear, well-structured, logically coherent.\n"
            "- 8-9: Mostly clear with minor flow issues.\n"
            "- 6-7: Some structure but less clear.\n"
            "- 4-5: Poorly organized.\n"
            "- 1-3: Very hard to follow.\n"
            "- 0: Completely incoherent."
        )
    }
}


def parse_qa_pairs_from_md(filepath: str) -> List[Tuple[str, str, str]]:
    """
    Parse markdown in the exact format:
    **Question:** ...
    **Ground Truth:** ...
    **Retrieved Answer:** ...
    ---

    Returns list of tuples: (question, ground_truth, retrieved_answer)
    """
    pairs = []
    current = {"q": None, "gt": None, "pred": None}

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            stripped = line.strip()

            if stripped.startswith("**Question:**"):
                # Save previous complete block if exists
                if current["q"] and current["gt"] and current["pred"]:
                    pairs.append((current["q"], current["gt"], current["pred"]))
                current = {
                    "q": stripped.replace("**Question:**", "", 1).strip(),
                    "gt": None,
                    "pred": None
                }

            elif stripped.startswith("**Ground Truth:**"):
                current["gt"] = stripped.replace("**Ground Truth:**", "", 1).strip()

            elif stripped.startswith("**Retrieved Answer:**"):
                current["pred"] = stripped.replace("**Retrieved Answer:**", "", 1).strip()

            elif stripped in ("---", ""):
                if current["q"] and current["gt"] and current["pred"]:
                    pairs.append((current["q"], current["gt"], current["pred"]))
                current = {"q": None, "gt": None, "pred": None}

    # Last block
    if current["q"] and current["gt"] and current["pred"]:
        pairs.append((current["q"], current["gt"], current["pred"]))

    return pairs


def build_scoring_prompt(question: str, ground_truth: str, retrieved_answer: str,
                         criterion: str, description: str, rubric: str) -> str:
    return f"""You are an impartial evaluation judge.

You are given:

Question:
\"\"\"{question}\"\"\"

Ground Truth Answer:
\"\"\"{ground_truth}\"\"\"

Retrieved Answer:
\"\"\"{retrieved_answer}\"\"\"

Your task:
Evaluate how well the retrieved answer captures ALL relevant factual information in the Ground Truth Answer, considering the context of the Question.

- The retrieved answer should fully include every important fact from the Ground Truth Answer.
- Relevant facts present in the Question but not explicitly in the Ground Truth Answer may be included without penalty.
- The retrieved answer should not omit key facts from the Ground Truth Answer.
- The retrieved answer should not contain incorrect facts or contradictions relative to both the Ground Truth and the Question.

Your evaluation must be based on the criterion: {criterion} — {description}

Scoring Rubric:
{rubric}

Provide output ONLY in this exact JSON format (no extra text, no markdown):
{{
  "retrieved": {{"score": <integer 0 to 10>}}
}}
""".strip()


def create_batch_requests(pairs: List[Tuple[str, str, str]]) -> List[dict]:
    requests = []
    request_id = 0

    for pair_idx, (question, gt, retrieved) in enumerate(pairs):
        for criterion, details in SCORING_RUBRICS.items():
            prompt = build_scoring_prompt(
                question, gt, retrieved,
                criterion, details["description"], details["rubric"]
            )

            requests.append({
                "custom_id": f"pair_{pair_idx}_{criterion}_{request_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are an impartial evaluation judge."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0,
                    "max_tokens": 80
                }
            })
            request_id += 1

    return requests


def save_batch_file(requests: List[dict], filename: str = "batch_requests.jsonl"):
    with open(filename, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    print(f"Saved {len(requests)} batch requests to {filename}")


def submit_batch_job(filename: str = "batch_requests.jsonl"):
    try:
        print("Uploading file...")
        with open(filename, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")

        print("Creating batch...")
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "RAG / Retrieval Evaluation - 2025"}
        )

        print(f"Batch created successfully")
        print(f"Batch ID:     {batch.id}")
        print(f"Status:       {batch.status}")
        print(f"Input file:   {file_obj.id}")

        batch_info = {
            "batch_id": batch.id,
            "input_file_id": file_obj.id,
            "created_at": datetime.now().isoformat(),
            "status": batch.status
        }
        with open("last_batch_info.json", "w", encoding="utf-8") as f:
            json.dump(batch_info, f, indent=2)

        return batch.id

    except Exception as e:
        print(f"Error submitting batch: {e}")
        return None


def monitor_batch(batch_id: str):
    try:
        batch = client.batches.retrieve(batch_id)
        print("\nBatch status:")
        print(f"  ID:              {batch.id}")
        print(f"  Status:          {batch.status}")
        if batch.request_counts:
            print(f"  Total:           {batch.request_counts.total}")
            print(f"  Completed:       {batch.request_counts.completed}")
            print(f"  Failed:          {batch.request_counts.failed}")
        if batch.status == "completed":
            print(f"  Output file ID:  {batch.output_file_id}")
            print(f"  Error file ID:   {batch.error_file_id or 'none'}")
        return batch
    except Exception as e:
        print(f"Error checking batch: {e}")
        return None


def download_results(batch_id: str, output_file: str = "batch_results.jsonl"):
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        print(f"Batch not completed (status: {batch.status})")
        return None

    print("Downloading results...")
    try:
        content = client.files.content(batch.output_file_id)
        with open(output_file, "wb") as f:
            f.write(content.content)
        print(f"Results saved → {output_file}")

        if batch.error_file_id:
            error_content = client.files.content(batch.error_file_id)
            error_path = "batch_errors.jsonl"
            with open(error_path, "wb") as f:
                f.write(error_content.content)
            print(f"Errors saved → {error_path}")

        return output_file
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def parse_custom_id(custom_id: str) -> Tuple[int, str, int]:
    parts = custom_id.split("_")
    pair_idx = int(parts[1])
    req_id = int(parts[-1])
    criterion = "_".join(parts[2:-1])
    return pair_idx, criterion, req_id


def process_batch_results(results_file: str, pairs: List[Tuple[str, str, str]]):
    results = []
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))

    organized = {}
    errors = []

    for item in results:
        if item.get("error"):
            print(f"API error for {item['custom_id']}: {item['error']}")
            continue

        if item["response"]["status_code"] != 200:
            continue

        try:
            pair_idx, criterion, _ = parse_custom_id(item["custom_id"])
            content = item["response"]["body"]["choices"][0]["message"]["content"].strip()

            # Clean possible code block
            if content.startswith("```json"):
                content = content.split("```json", 1)[1].rsplit("```", 1)[0].strip()
            elif content.startswith("```"):
                content = content.strip("```").strip()

            parsed = json.loads(content)
            score = int(float(parsed["retrieved"]["score"]))

            if pair_idx not in organized:
                organized[pair_idx] = {}
            organized[pair_idx][criterion] = score

        except Exception as e:
            errors.append((item.get("custom_id", "?"), str(e)))

    if errors:
        print(f"Found {len(errors)} parsing errors")

    eval_results = []
    for idx in sorted(organized.keys()):
        if idx < len(pairs):
            q, gt, pred = pairs[idx]
            scores_dict = {}
            for crit in SCORING_RUBRICS:
                scores_dict[crit] = organized[idx].get(crit, None)

            eval_results.append({
                "pair_index": idx + 1,
                "question": q,
                "ground_truth": gt,
                "retrieved_answer": pred,
                "scores": scores_dict
            })

    return eval_results


def calculate_averages(eval_results):
    avgs = {}
    for crit in SCORING_RUBRICS:
        values = [r["scores"][crit] for r in eval_results if r["scores"][crit] is not None]
        avgs[crit] = sum(values) / len(values) if values else None
    return avgs


def save_markdown_report(eval_results, averages, output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Retrieval Evaluation – OpenAI Batch\n\n")
        f.write("## Average Scores\n\n")
        for crit, val in averages.items():
            v = f"{val:.2f}" if val is not None else "N/A"
            f.write(f"- **{crit.capitalize()}**: {v}\n")
        f.write("\n---\n\n## Individual Evaluations\n\n")

        for r in eval_results:
            f.write(f"### Pair {r['pair_index']}\n\n")
            f.write(f"**Question:**\n{r['question']}\n\n")
            f.write(f"**Ground Truth:**\n{r['ground_truth']}\n\n")
            f.write(f"**Retrieved Answer:**\n{r['retrieved_answer']}\n\n")
            f.write("**Scores:**\n")
            for crit, score in r["scores"].items():
                s = f"{score:.1f}" if score is not None else "N/A"
                f.write(f"- {crit.capitalize():14} {s}\n")
            f.write("\n---\n")

    print(f"Report saved → {output_file}")


def main():
    parser = argparse.ArgumentParser(description="OpenAI Batch Evaluation – strict markdown format")
    parser.add_argument("--input", "-i", default="output.md", help="input markdown file")
    parser.add_argument("--output", "-o", default="openai_evaluation.md", help="output report")
    parser.add_argument("--batch-file", default="batch_requests.jsonl")
    parser.add_argument("--results-file", default="batch_results.jsonl")
    args = parser.parse_args()

    print("Loading evaluation data...")
    pairs = parse_qa_pairs_from_md(args.input)
    if not pairs:
        print("No valid QA pairs found.")
        return

    print(f"→ Found {len(pairs)} QA pairs")

    print("Generating batch requests...")
    requests = create_batch_requests(pairs)
    print(f"→ {len(requests)} evaluation prompts created")

    save_batch_file(requests, args.batch_file)

    print("\nSubmit batch now? (y/n)")
    if input().strip().lower() != 'y':
        print("Exiting. You can submit later manually.")
        return

    batch_id = submit_batch_job(args.batch_file)
    if not batch_id:
        return

    print("\nMonitoring batch... (you can also check https://platform.openai.com/batches)")
    print("Press Ctrl+C to stop monitoring (batch continues running)")

    try:
        while True:
            batch = monitor_batch(batch_id)
            if batch and batch.status in ("completed", "failed", "expired", "cancelled"):
                break
            time.sleep(45)
    except KeyboardInterrupt:
        print("\nMonitoring stopped. Batch continues in background.")

    if batch and batch.status == "completed":
        print("\nDownloading results...")
        results_path = download_results(batch_id, args.results_file)
        if results_path:
            print("Processing results...")
            eval_results = process_batch_results(results_path, pairs)
            if eval_results:
                averages = calculate_averages(eval_results)
                save_markdown_report(eval_results, averages, args.output)

                print("\nFinal averages:")
                for k, v in averages.items():
                    print(f"  {k:16} : {v:.2f}" if v is not None else f"  {k:16} : N/A")


if __name__ == "__main__":
    main()