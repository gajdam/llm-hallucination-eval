#!/usr/bin/env python3
"""LLM Hallucination Evaluation — entry point.

Usage examples:
  python main.py                              # run all LLMs in config.yaml
  python main.py --llm claude-opus-4-6        # single LLM
  python main.py --samples 50                 # quick test with 50 samples
  python main.py --config my_config.yaml
  python main.py --output results/run1
"""

import argparse
import sys

import yaml
from rich.console import Console

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM hallucination rates on the FEVER dataset."
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config file (default: config.yaml)"
    )
    parser.add_argument(
        "--llm",
        help="Evaluate only this model (must match a 'name' in config llms list)"
    )
    parser.add_argument(
        "--samples", type=int,
        help="Override max_samples from config"
    )
    parser.add_argument(
        "--output",
        help="Override output directory from config"
    )
    parser.add_argument(
        "--split", default=None,
        help="FEVER split to use (e.g. labelled_dev, paper_test)"
    )
    args = parser.parse_args()

    # Load config
    try:
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        console.print(f"[red]Config file not found: {args.config}[/red]")
        sys.exit(1)

    # Apply CLI overrides
    if args.samples is not None:
        config["fever"]["max_samples"] = args.samples
    if args.output:
        config.setdefault("evaluation", {})["output_dir"] = args.output
    if args.split:
        config["fever"]["split"] = args.split

    # Run
    from src.pipeline import EvaluationPipeline
    pipeline = EvaluationPipeline(config)
    pipeline.run(llm_filter=args.llm)


if __name__ == "__main__":
    main()
