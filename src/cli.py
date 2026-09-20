"""
Unified command-line interface for the project.

    hotel pipeline    # clean -> features -> train LightGBM -> evaluate
    hotel analysis    # regenerate all report figures + output/ tables
    hotel train       # fit and compare all four candidate models

Subcommand imports are lazy so that ``hotel analysis`` does not require the
gradient-boosting libraries to be importable.
"""

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser and its subcommands."""
    parser = argparse.ArgumentParser(
        prog="hotel",
        description="Hotel booking cancellation analysis and prediction.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "pipeline",
        help="clean the raw data, engineer features, train and evaluate the model",
    )
    subparsers.add_parser(
        "analysis",
        help="regenerate every report figure and output/ summary table",
    )
    subparsers.add_parser(
        "train",
        help="fit and compare all four candidate models",
    )
    return parser


def main(argv=None) -> int:
    """Dispatch to the requested subcommand and return an exit code."""
    args = build_parser().parse_args(argv)

    if args.command == "pipeline":
        from src.run_pipeline import run

        run()
    elif args.command == "analysis":
        from src.run_analysis import run

        run()
    elif args.command == "train":
        from src.train import main as train_main

        train_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
