import argparse
import os
import json
import time
import re
import string
from typing import List, Tuple
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is required.")
genai.configure(api_key=api_key)

MODEL_NAME = "gemini-1.5-flash"          # ← most common stable name in 2025; change if needed
MAX_WORKERS = 10

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


def normalize(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_qa_pairs_from_md(filepath: str) -> List[Tuple[str, str, str]]:
    """
    Parse markdown in format:
    **Question:** ...
    **Ground Truth:** ...
    **Retrieved Answer:** ...
    ---
    Returns list of (question, ground_truth, retrieved_answer)
    """
    pairs = []
    current = {"q": None, "gt": None, "pred": None}

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            stripped = line.strip()

            if stripped.startswith("**Question:**"):
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

    # Don't forget the last block
    if current["q"] and current["gt"] and current["pred"]:
        pairs.append((current["q"], current["gt"], current["pred"]))

    return pairs


def build_scoring_prompt(question, ground_truth, retrieved_answer, criterion, description, rubric):
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

Provide output ONLY in this JSON format:
{{
  "retrieved": {{"score": <integer from 0 to 10>}}
}}
""".strip()


def create_requests(pairs: List[Tuple[str, str, str]]):
    requests = []
    request_id = 0

    for pair_idx, (q, gt, pred) in enumerate(pairs):
        for criterion, details in SCORING_RUBRICS.items():
            prompt = build_scoring_prompt(
                q, gt, pred,
                criterion, details["description"], details["rubric"]
            )
            requests.append({
                "custom_id": f"pair_{pair_idx}_{criterion}_{request_id}",
                "prompt": prompt
            })
            request_id += 1
    return requests


def process_single_request(req, model):
    try:
        response = model.generate_content(req["prompt"])
        content = response.text.strip()
        return {
            "custom_id": req["custom_id"],
            "response": {"body": {"choices": [{"message": {"content": content}}]}}
        }
    except Exception as e:
        return {
            "custom_id": req["custom_id"],
            "response": {"error": str(e)}
        }


def process_requests_parallel(requests, output_file="results.jsonl", max_workers=MAX_WORKERS):
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=genai.types.GenerationConfig(temperature=0)
    )

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_req = {executor.submit(process_single_request, req, model): req for req in requests}

        with tqdm(total=len(requests), desc="Processing Gemini requests") as pbar:
            for future in as_completed(future_to_req):
                result = future.result()
                results.append(result)
                pbar.update(1)

                if len(results) % 50 == 0:
                    with open(output_file, "w", encoding="utf-8") as f:
                        for r in results:
                            f.write(json.dumps(r) + "\n")

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Saved {len(results)} results to {output_file}")
    return output_file


def parse_custom_id(custom_id: str):
    parts = custom_id.split("_")
    pair_idx = int(parts[1])
    request_id = int(parts[-1])
    criterion = "_".join(parts[2:-1])
    return pair_idx, criterion, request_id


def process_results(results_file="results.jsonl", pairs=None):
    results = []
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))

    print(f"Processing {len(results)} results...")

    organized = {}
    errors = []

    for result in results:
        try:
            pair_idx, criterion, _ = parse_custom_id(result["custom_id"])

            if pair_idx not in organized:
                organized[pair_idx] = {}

            if "error" in result.get("response", {}):
                organized[pair_idx][criterion] = {"score": None}
                continue

            content = result["response"]["body"]["choices"][0]["message"]["content"]

            # Clean possible markdown code block
            if content.startswith("```json"):
                content = content.split("```json", 1)[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.strip("```").strip()

            parsed = json.loads(content)
            score = float(parsed["retrieved"]["score"])

            organized[pair_idx][criterion] = {"score": score}

        except Exception as e:
            errors.append((result.get("custom_id", "unknown"), str(e)))
            if pair_idx in organized:
                organized[pair_idx][criterion] = {"score": None}

    if errors:
        print(f"Found {len(errors)} parsing errors")

    evaluation_results = []
    for pair_idx in sorted(organized.keys()):
        if pairs and pair_idx < len(pairs):
            q, gt, pred = pairs[pair_idx]
            res = {
                "pair_index": pair_idx + 1,
                "question": q,
                "ground_truth": gt,
                "retrieved_answer": pred,
                "scores": {"retrieved": {}}
            }
            for crit in SCORING_RUBRICS:
                res["scores"]["retrieved"][crit] = {
                    "score": organized[pair_idx].get(crit, {"score": None})["score"]
                }
            evaluation_results.append(res)

    return evaluation_results


def calculate_average_scores(evaluation_results):
    avgs = {"retrieved": {}}
    for crit in SCORING_RUBRICS:
        values = [
            r["scores"]["retrieved"][crit]["score"]
            for r in evaluation_results
            if r["scores"]["retrieved"][crit]["score"] is not None
        ]
        avgs["retrieved"][crit] = sum(values) / len(values) if values else None
    return avgs


def save_results_markdown(results, avg_scores, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Retrieval Evaluation Results (Gemini)\n\n")
        f.write("## Average Scores\n\n")
        for crit, avg in avg_scores["retrieved"].items():
            val = f"{avg:.2f}" if avg is not None else "N/A"
            f.write(f"- **{crit.capitalize()}**: {val}\n")
        f.write("\n---\n\n## Individual Results\n\n")

        for r in results:
            f.write(f"### Pair {r['pair_index']}\n")
            f.write(f"**Question:** {r['question']}\n\n")
            f.write(f"**Ground Truth:**\n{r['ground_truth']}\n\n")
            f.write(f"**Retrieved Answer:**\n{r['retrieved_answer']}\n\n")
            f.write("**Scores:**\n")
            for crit, s in r["scores"]["retrieved"].items():
                val = f"{s['score']:.1f}" if s["score"] is not None else "N/A"
                f.write(f"- {crit.capitalize()}: {val}\n")
            f.write("\n---\n")


def main():
    parser = argparse.ArgumentParser(description="Gemini LLM-as-a-Judge - strict markdown parsing")
    parser.add_argument("--input", "-i", default="output.md", help="input markdown file")
    parser.add_argument("--output", "-o", default="gemini_evaluation.md")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--results-jsonl", default="gemini_results.jsonl")
    args = parser.parse_args()

    print("Loading test data...")
    pairs = parse_qa_pairs_from_md(args.input)
    if not pairs:
        print("No valid QA pairs found in file.")
        return

    print(f"Found {len(pairs)} QA pairs.")

    print("Building evaluation requests...")
    requests = create_requests(pairs)

    print(f"Processing {len(requests)} evaluation calls...")
    process_requests_parallel(requests, args.results_jsonl, args.max_workers)

    print("Parsing results...")
    eval_results = process_results(args.results_jsonl, pairs)
    if not eval_results:
        print("No evaluation results could be processed.")
        return

    avgs = calculate_average_scores(eval_results)
    save_results_markdown(eval_results, avgs, args.output)

    print("\nFinal averages:")
    for crit, v in avgs["retrieved"].items():
        print(f"  {crit:16} : {v:.2f}" if v is not None else f"  {crit:16} : N/A")


if __name__ == "__main__":
    main()