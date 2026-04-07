# examples/run_agent.py — end-to-end example of the CXR SpatialAgent pipeline
"""
Demonstrates both simulated and real execution modes.

Usage:
    # Simulated mode (Claude acts as each tool — no model weights needed)
    python examples/run_agent.py

    # With a real CXR image in simulated mode
    python examples/run_agent.py --image path/to/cxr.jpg

    # Real mode (uses TorchXRayVision, SAM2, YOLOv8 — requires model weights)
    python examples/run_agent.py --image cxr.jpg --mode real

    # Batch: answer all questions from the spatial taxonomy for a set of entities
    python examples/run_agent.py --image cxr.jpg --batch

    # Save full results to JSON
    python examples/run_agent.py --image cxr.jpg --output results.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import SpatialAgent
from utils.question_router import parse_question, generate_questions

# Representative questions covering all spatial taxonomy types
SAMPLE_QUESTIONS = [
    # Presence
    "Is pleural effusion present in the chest X-ray?",
    # Location
    "Where is the consolidation located?",
    # Distribution
    "What is the distribution pattern of the consolidation: localized, focal, diffuse, "
    "unilateral, bilateral, central, peripheral, or multilobular?",
    # Laterality
    "Is the pleural effusion more extensive in the right lung or in the left lung?",
    # Containment
    "Is the pleural effusion contained entirely within the lung field without extending "
    "beyond the pleural surface?",
    # Device tip (safety-critical)
    "Where is the tip of the endotracheal tube relative to the carina?",
    # Relative position
    "What is the relative position of the consolidation with respect to the hilum?",
    # Zone presence
    "Is the atelectasis located predominantly in the right lower zone of the lung?",
    # Border / silhouette
    "Does the consolidation involve or obscure the left hemidiaphragm?",
    # General
    "Is there a pneumothorax? If so, which side and how large?",
    "Please assess cardiac size and comment on signs of heart failure.",
    "Evaluate this post-intubation CXR — is the ET tube correctly positioned?",
]


def main():
    parser = argparse.ArgumentParser(description="CXR SpatialAgent example")
    parser.add_argument("--image",   default=None, help="Path to CXR image (optional)")
    parser.add_argument("--question", default=None, help="Single clinical question")
    parser.add_argument("--mode",    choices=["simulated", "real"], default="simulated",
                        help="simulated = Claude acts as tools (default); real = uses DL models")
    parser.add_argument("--batch",   action="store_true",
                        help="Generate and answer all spatial questions for a fixed entity set")
    parser.add_argument("--output",  default=None, help="Save results as JSON to this path")
    args = parser.parse_args()

    agent = SpatialAgent(mode=args.mode)

    if args.batch:
        # ── Generate all spatial questions for a representative entity set ────
        questions = generate_questions(
            findings_all={"consolidation", "pleural effusion", "atelectasis"},
            devices={"endotracheal tube", "nasogastric tube"},
            refs={"carina", "hilum", "diaphragm"},
            borders={"left heart border", "left hemidiaphragm"},
            zones=True,
        )
        questions = sorted(questions)
        print(f"\nGenerated {len(questions)} spatial questions. Running first 10...\n")
        all_results = []
        for q in list(questions)[:10]:
            pq = parse_question(q)
            print(f"[{pq.q_type.value}] {q}")
            result = agent.run(question=q, image_path=args.image, verbose=False)
            print(f"  → {result.report[:120]}...\n")
            all_results.append(result)

        if args.output:
            _save_batch(all_results, args.output)
        return

    if args.question:
        # ── Single question ────────────────────────────────────────────────────
        result = agent.run(question=args.question, image_path=args.image)
        if args.output:
            _save_single(result, args.output)
        return

    # ── Default: run all sample questions ─────────────────────────────────────
    print("\nAvailable sample questions:")
    for i, q in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"  [{i:2d}] {q}")

    print(f"\nRunning first 3 questions in {args.mode!r} mode...\n")
    for q in SAMPLE_QUESTIONS[:3]:
        result = agent.run(question=q, image_path=args.image)
        if args.output:
            _save_single(result, args.output)


def _save_single(result, path: str):
    data = {
        "question": result.question,
        "image_path": result.image_path,
        "plan_reasoning": result.plan.reasoning,
        "steps": [
            {
                "step": s.step,
                "tool": s.tool_id,
                "purpose": s.purpose,
                "output": next(
                    (r.output for r in result.tool_results if r.step == s.step), {}
                ),
            }
            for s in result.plan
        ],
        "report": result.report,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


def _save_batch(results, path: str):
    data = [
        {"question": r.question, "report": r.report}
        for r in results
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nBatch results saved to {path}")


if __name__ == "__main__":
    main()
