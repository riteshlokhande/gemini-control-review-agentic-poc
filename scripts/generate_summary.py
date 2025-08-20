#!/usr/bin/env python3
"""
generate_summary.py

Modular pass/fail summary generator with Not Applicable counts:
  - Discovers project root (data/ + logs/) automatically
  - Loads config.yaml for paths and metadata settings
  - Reads pivoted report CSV
  - Computes per-question pass/fail/not_applicable counts
  - Writes summary CSV under data/processed/
  - Logs to both console and logs/generate_summary.log
"""

import argparse
import logging
from logging import StreamHandler, FileHandler, Formatter
from pathlib import Path
import yaml
import pandas as pd


def setup_logging(log_path: Path):
    """Configure logging to console and file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = Formatter("%(asctime)s %(levelname)s %(name)s • %(message)s")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    sh = StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)


def get_project_root() -> Path:
    """Find the project root by locating data/ and logs/ directories."""
    here = Path(__file__).resolve().parent
    for p in (here, *here.parents):
        if (p / "data").is_dir() and (p / "logs").is_dir():
            return p
    return here.parent


def load_config(cfg_path: Path) -> dict:
    """Load YAML configuration."""
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def load_dataframe(path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame."""
    logging.info("Loading report from %s", path)
    return pd.read_csv(path, encoding="utf-8")


def summarize(df: pd.DataFrame, metadata_fields: list) -> pd.DataFrame:
    """
    Compute pass/fail/not_applicable counts for each question column.
    Returns a DataFrame with columns:
      question, passed_count, failed_count, not_applicable_count
    """
    logging.info("Computing summary on DataFrame with shape %s", df.shape)
    question_cols = [c for c in df.columns if c not in metadata_fields]
    summary_rows = []

    for q in question_cols:
        passed = int((df[q] == "Yes").sum())
        failed = int((df[q] == "No").sum())
        not_applicable = int((df[q] == "Not Applicable").sum())

        summary_rows.append({
            "question": q,
            "passed_count": passed,
            "failed_count": failed,
            "not_applicable_count": not_applicable
        })

        logging.info(
            "Question '%s': passed=%d, failed=%d, not_applicable=%d",
            q, passed, failed, not_applicable
        )

    return pd.DataFrame(summary_rows)


def export_csv(df: pd.DataFrame, out_path: Path):
    """Write DataFrame to CSV, ensuring directory exists."""
    logging.info("Writing summary CSV to %s", out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    logging.info("Summary export complete")


def parse_args():
    """Parse CLI arguments for config file path."""
    parser = argparse.ArgumentParser(
        description="Generate pass/fail summary from pivoted QA report"
    )
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to YAML config file (relative to project root)"
    )
    return parser.parse_args()


def main():
    # 1) Resolve project root and config
    args = parse_args()
    project_root = get_project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path

    # 2) Load configuration
    cfg = load_config(cfg_path)

    # 3) Setup logging
    log_path = project_root / cfg.get("paths", {}).get(
        "summary_log", "logs/generate_summary.log"
    )
    setup_logging(log_path)
    logging.info("Project root: %s", project_root)

    # 4) Resolve input/output paths from config
    formatted_report = Path(cfg.get("formatted_report", "data/output/formatted_report.csv"))
    summary_report   = Path(cfg.get("summary_report", "data/output/summary_report.csv"))
    if not formatted_report.is_absolute():
        formatted_report = project_root / formatted_report
    if not summary_report.is_absolute():
        summary_report = project_root / summary_report

    metadata_fields = cfg.get(
        "metadata_fields",
        ["payment_id", "payment_date", "payment_amount", "payment_currency"]
    )

    # 5) Load pivoted table
    df = load_dataframe(formatted_report)

    # 6) Compute summary (now includes Not Applicable)
    summary_df = summarize(df, metadata_fields)

    # 7) Export summary
    export_csv(summary_df, summary_report)

    logging.info("generate_summary.py completed successfully")


if __name__ == "__main__":
    main()
