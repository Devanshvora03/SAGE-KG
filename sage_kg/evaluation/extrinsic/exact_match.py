"""Exact Match (EM) on query_kg.py markdown: Ground Truth vs Retrieved Answer.
"""

import argparse
import string
import re
from typing import List, Tuple

def normalize(text: str) -> str:
    """Very strict normalization for exact match"""
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def exact_match_score(pairs: List[Tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    matches = sum(1 for gt, pred in pairs if normalize(gt) == normalize(pred))
    return (matches / len(pairs)) * 100

def parse_qa_pairs_from_md(filepath: str) -> List[Tuple[str, str]]:
    """
    Parse markdown in format:
    **Question:** ...
    **Ground Truth:** ...
    **Retrieved Answer:** ...
    ---
    """
    pairs = []
    current = {"q": None, "gt": None, "pred": None}

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            stripped = line.strip()

            if stripped.startswith("**Question:**"):
                if current["gt"] and current["pred"]:
                    pairs.append((current["gt"], current["pred"]))
                current = {"q": stripped.replace("**Question:**", "", 1).strip(), "gt": None, "pred": None}

            elif stripped.startswith("**Ground Truth:**"):
                current["gt"] = stripped.replace("**Ground Truth:**", "", 1).strip()

            elif stripped.startswith("**Retrieved Answer:**"):
                current["pred"] = stripped.replace("**Retrieved Answer:**", "", 1).strip()

            elif stripped == "---" or stripped == "":
                if current["gt"] and current["pred"]:
                    pairs.append((current["gt"], current["pred"]))
                    current = {"q": None, "gt": None, "pred": None}

    # last one
    if current["gt"] and current["pred"]:
        pairs.append((current["gt"], current["pred"]))

    return pairs

def main():
    parser = argparse.ArgumentParser(description="Exact Match Evaluation (strict normalization)")
    parser.add_argument("--input", "-i", default="output.md", help="Markdown file with evaluation results")
    parser.add_argument("--output", "-o", default=None, help="Optional: save score to file")
    args = parser.parse_args()

    pairs = parse_qa_pairs_from_md(args.input)
    if not pairs:
        print("No valid (Ground Truth, Retrieved Answer) pairs found.")
        return

    score = exact_match_score(pairs)

    print(f"Number of pairs evaluated : {len(pairs):3d}")
    print(f"Exact Match Score         : {score:6.2f} %")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(f"exact_match_score: {score:.4f}\n")
            f.write(f"num_pairs: {len(pairs)}\n")
        print(f"Score saved to: {args.output}")

if __name__ == "__main__":
    main()