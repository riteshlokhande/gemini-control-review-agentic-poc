#!/usr/bin/env python3
"""
format_predictions.py

Transforms QA predictions and original payment attributes into
a pivoted CSV report.
Features:
  - Loads raw payment records from qa_inference_input.json
  - Loads QA predictions from JSONL
  - Merges on payment_id
  - Pivots answers into columns
  - Exports report to CSV under data/processed/
  - Writes logs to logs/format_predictions.log and stdout
  - Fully config-driven and modular
"""

import argparse
import json
import logging
from logging import StreamHandler, FileHandler, Formatter
from pathlib import Path
import yaml
import pandas as pd


def setup_logging(log_file: Path):
    """Configure root logger with console and file handlers."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = Formatter("%(asctime)s %(levelname)s %(name)s • %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    ch = StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)


def get_project_root() -> Path:
    """Detect project root by looking for data/ and logs/ directories."""
    here = Path(__file__).resolve().parent
    for p in (here, *here.parents):
        if (p / "data").is_dir() and (p / "logs").is_dir():
            return p
    # fallback: assume parent of script is root
    return here.parent


def load_config(path: Path) -> dict:
    """Load YAML config from a file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> list:
    """Load a JSON array from a file."""
    logging.info("Loading JSON records from %s", path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list:
    """Load one JSON object per line."""
    logging.info("Loading JSONL predictions from %s", path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logging.info("Loaded %d predictions", len(records))
    return records


def merge_records(
    raw: list, preds: list, key: str = "payment_id"
) -> list:
    """
    Merge raw payment records with QA predictions on payment_id.
    Returns list of dicts with all raw fields plus 'answers'.
    """
    logging.info("Merging %d raw records with %d predictions", len(raw), len(preds))
    pred_map = {r[key]: r.get("answers", []) for r in preds}
    merged = []
    for rec in raw:
        pid = rec.get(key)
        rec_copy = dict(rec)  # copy raw attributes
        rec_copy["answers"] = pred_map.get(pid, [])
        merged.append(rec_copy)
        if pid not in pred_map:
            logging.warning("No predictions found for %s", pid)
    return merged


def pivot_to_dataframe(records: list, metadata: list) -> pd.DataFrame:
    """
    Given merged records with raw attributes and answers,
    pivot into a flat DataFrame: one row per payment_id, columns for metadata + each question.
    """
    logging.info("Pivoting %d merged records into DataFrame", len(records))
    rows = []
    for rec in records:
        row = {field: rec.get(field, "") for field in metadata}
        for qa in rec.get("answers", []):
            q = qa.get("question", "").strip()
            a = qa.get("answer", "").strip()
            row[q] = a
        rows.append(row)

    # Determine column order: metadata first, then sorted questions
    all_cols = set().union(*rows) if rows else set(metadata)
    question_cols = sorted(all_cols - set(metadata))
    final_cols = metadata + question_cols

    df = pd.DataFrame(rows, columns=final_cols)
    logging.info("DataFrame shape: %s", df.shape)
    return df


def export_csv(df: pd.DataFrame, out_path: Path):
    """Write DataFrame to CSV, creating parent dirs if needed."""
    logging.info("Exporting DataFrame to CSV: %s", out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    logging.info("CSV export complete")


def parse_args():
    """Parse CLI args for config path."""
    parser = argparse.ArgumentParser(
        description="Merge payment attributes with QA predictions and pivot to CSV"
    )
    parser.add_argument(
        "-c", "--config", default="config.yaml",
        help="YAML config file (relative to project root)"
    )
    return parser.parse_args()


def main():
    # 1) Parse args & locate project root
    args = parse_args()
    root = get_project_root()

    # 2) Load config
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_config(cfg_path)

    # 3) Setup logging
    log_file = root / cfg.get("paths", {}).get("log_file", "logs/format_predictions.log")
    setup_logging(log_file)
    logging.info("Project root: %s", root)

    # 4) Resolve data directories & files
    raw_dir       = root / cfg.get("paths", {}).get("raw_data_dir", "data")
    proc_dir      = root / cfg.get("paths", {}).get("processed_data_dir", "data")
    input_file    = raw_dir / cfg["paths"]["input_file"]
    pred_file     = proc_dir / cfg["paths"]["predictions_file"]
    report_file   = proc_dir / cfg["paths"]["formatted_report"]

    # 5) Load inputs
    raw_records  = load_json(input_file)
    preds_records = load_jsonl(pred_file)

    # 6) Merge & pivot
    merged = merge_records(raw_records, preds_records, key=cfg.get("merge_key", "payment_id"))
    df = pivot_to_dataframe(merged, cfg.get("metadata_fields"))

    # 7) Export CSV
    export_csv(df, report_file)

    logging.info("format_predictions.py completed successfully")


if __name__ == "__main__":
    main()
