#!/usr/bin/env python3
"""
Thin Command-Line Launcher for HRR Project (Hierarchical Retrieval & Reranking)
-------------------------------------------------------------------------------
Orchestrates Study 1 and Study 2 experiments by calling existing standalone scripts
in the exact required execution order using the current Python interpreter.

Usage:
  python main.py --help
  python main.py --study 1 [--dry-run]
  python main.py --study 2 [--dry-run]
  python main.py --study 1 --step bm25
  python main.py --study 2 --step stepA
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Study 1 steps in exact execution order
STUDY_1_STEPS = [
    ("build_subset", "step1b_build_subset.py", "Build 100k-passage / 1k-query subset from BeIR/nq"),
    ("bm25", "run_bm25.py", "Evaluate BM25 baseline on Dev and Test"),
    ("dense", "run_dense.py", "Embed passages with multi-qa-MiniLM and run flat dense search"),
    ("hierarchical", "run_hierarchical.py", "Run two-stage hard hierarchical retrieval (N=20)"),
    ("fusion", "run_fusion.py", "Evaluate soft fusion (lambda=0.0 and lambda=0.25)"),
    ("cross_encoder", "step6b_7a_rerank.py", "Run cross-encoder reranking (top 30) & failure analysis"),
    ("aggregation", "run_step8_aggregation.py", "Tune article-evidence aggregation on Dev and evaluate on Test"),
    ("gat", "step7_gat_reranker.py", "Train and evaluate GAT reranker and no-edges control"),
    ("analysis", "run_analysis.py", "Run paired bootstrap significance tests and route miss diagnosis"),
]

# Study 2 steps in exact execution order
STUDY_2_STEPS = [
    ("stepA", os.path.join("scale_up", "stepA_build_subset.py"), "Build scaled 200k-passage / 3,452-query dataset"),
    ("stepB", os.path.join("scale_up", "stepB_dense_and_rerank.py"), "Embed scaled corpus, run flat dense and CE top-30 reranking"),
    ("stepC", os.path.join("scale_up", "stepC_cross_validated_gat.py"), "Run 5-fold cross-validated GAT, GAT-no-edges, and aggregation"),
    ("depth", os.path.join("scale_up", "step_extra_depth_sensitivity.py"), "Run cross-encoder rerank-depth sensitivity (D in {30, 50, 100})"),
]

# Aliases for flexible step naming
STEP_ALIASES = {
    "1": {
        "build": "build_subset",
        "subset": "build_subset",
        "ce": "cross_encoder",
        "rerank": "cross_encoder",
        "agg": "aggregation",
        "bootstrap": "analysis",
    },
    "2": {
        "step_a": "stepA",
        "step_b": "stepB",
        "step_c": "stepC",
        "build": "stepA",
        "dense": "stepB",
        "gat": "stepC",
        "depth_sensitivity": "depth",
    }
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified launcher for Study 1 and Study 2 retrieval pipelines."
    )
    parser.add_argument(
        "--study",
        type=int,
        choices=[1, 2],
        help="Select study pipeline: 1 (diagnostic, 1,000 queries) or 2 (scale-up, 3,452 queries).",
    )
    parser.add_argument(
        "--step",
        type=str,
        default=None,
        help="Run only a specific step by name (e.g., 'bm25', 'dense', 'stepA', 'stepC', 'depth').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands that would be executed without running them.",
    )
    return parser.parse_args()


def resolve_step(study: int, step_name: str, step_list):
    step_name_norm = step_name.lower().replace("-", "_")
    # Check aliases
    aliases = STEP_ALIASES.get(str(study), {})
    canonical = aliases.get(step_name_norm, step_name_norm)
    for name, script_rel, desc in step_list:
        if name.lower() == canonical:
            return [(name, script_rel, desc)]
    return None


def run_pipeline(steps, dry_run=False):
    for idx, (name, script_rel, desc) in enumerate(steps, start=1):
        script_path = PROJECT_ROOT / script_rel
        if not script_path.exists():
            print(f"Error: Script file not found: {script_path}", file=sys.stderr)
            sys.exit(1)

        cmd = [sys.executable, str(script_path)]
        cmd_str = f'"{sys.executable}" "{script_path}"'

        print(f"\n{'='*70}")
        print(f"[{idx}/{len(steps)}] Step: {name.upper()}")
        print(f"Description: {desc}")
        print(f"Script:      {script_rel}")
        print(f"{'='*70}")

        if dry_run:
            print(f"[DRY-RUN] Command: {cmd_str}")
        else:
            print(f"[EXECUTING] {cmd_str}\n")
            res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if res.returncode != 0:
                print(f"\nExecution failed at step '{name}' with exit code {res.returncode}.", file=sys.stderr)
                sys.exit(res.returncode)


def main():
    args = parse_args()

    if args.study is None and args.step is None:
        print("Please specify --study {1, 2} or a specific --step. Use --help for usage details.")
        sys.exit(1)

    # If study is specified
    if args.study == 1:
        steps = STUDY_1_STEPS
        if args.step:
            resolved = resolve_step(1, args.step, steps)
            if not resolved:
                valid = [s[0] for s in steps]
                print(f"Unknown step '{args.step}' for Study 1. Valid steps are: {valid}", file=sys.stderr)
                sys.exit(1)
            steps = resolved
        run_pipeline(steps, dry_run=args.dry_run)

    elif args.study == 2:
        steps = STUDY_2_STEPS
        if args.step:
            resolved = resolve_step(2, args.step, steps)
            if not resolved:
                valid = [s[0] for s in steps]
                print(f"Unknown step '{args.step}' for Study 2. Valid steps are: {valid}", file=sys.stderr)
                sys.exit(1)
            steps = resolved
        run_pipeline(steps, dry_run=args.dry_run)

    else:
        # Step was given without study, try to infer
        resolved_1 = resolve_step(1, args.step, STUDY_1_STEPS)
        resolved_2 = resolve_step(2, args.step, STUDY_2_STEPS)
        if resolved_1 and not resolved_2:
            run_pipeline(resolved_1, dry_run=args.dry_run)
        elif resolved_2 and not resolved_1:
            run_pipeline(resolved_2, dry_run=args.dry_run)
        elif resolved_1 and resolved_2:
            print(f"Step '{args.step}' exists in both Study 1 and Study 2. Please specify --study 1 or --study 2.", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"Unknown step '{args.step}'. Valid Study 1 steps: {[s[0] for s in STUDY_1_STEPS]}. Valid Study 2 steps: {[s[0] for s in STUDY_2_STEPS]}.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
