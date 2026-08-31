"""Semantic Relevance (S-R): MiniLM cosine between ground-truth and predicted answers.
"""

import argparse
from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def parse_ground_truth_and_retrieved(md_path: str) -> List[Tuple[str, str]]:
    pairs = []
    gt = pred = None

    with open(md_path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("**Ground Truth:**"):
                gt = s.replace("**Ground Truth:**", "", 1).strip()
            elif s.startswith("**Retrieved Answer:**"):
                pred = s.replace("**Retrieved Answer:**", "", 1).strip()
                if gt is not None and pred is not None:
                    pairs.append((gt, pred))
                    gt = pred = None
            elif s == "---":
                # optional reset
                gt = pred = None

    # last pair if any
    if gt is not None and pred is not None:
        pairs.append((gt, pred))

    return pairs

def compute_mean_cosine_similarity(gts: List[str], preds: List[str], model_name="all-MiniLM-L6-v2") -> float:
    if not gts:
        return 0.0

    model = SentenceTransformer(model_name)
    gt_emb = model.encode(gts, convert_to_numpy=True, show_progress_bar=True)
    pred_emb = model.encode(preds, convert_to_numpy=True, show_progress_bar=False)

    sims = cosine_similarity(gt_emb, pred_emb).diagonal()
    return float(np.mean(sims)) * 100.0

def main():
    parser = argparse.ArgumentParser(description="Semantic Similarity (cosine) between GT and Retrieved answers")
    parser.add_argument("--input", "-i", default="output.md", help="Evaluation markdown file")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="sentence-transformers model")
    args = parser.parse_args()

    pairs = parse_ground_truth_and_retrieved(args.input)
    if not pairs:
        print("No evaluation pairs found in file.")
        return

    gts = [gt for gt, _ in pairs]
    preds = [pred for _, pred in pairs]

    score = compute_mean_cosine_similarity(gts, preds, args.model)

    print(f"Number of pairs          : {len(pairs):3d}")
    print(f"Model                    : {args.model}")
    print(f"Average Cosine Similarity: {score:6.2f} %")

if __name__ == "__main__":
    main()