#!/usr/bin/env python3
"""
transform_inputs.py

Enriches synthetic payment records with:
  - A static list of control-assessment questions
  - All required payment attributes (filling None if missing)
  - Control definitions from config.yaml
Resolves project root dynamically so you can run it from / or /scripts.
"""

import json
import logging
from pathlib import Path

import yaml


def get_project_root(markers=("data", "logs")) -> Path:
    """
    Walk up from this script’s dir until we find 'data' and 'logs'.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if all((candidate / m).exists() for m in markers):
            return candidate
    raise RuntimeError(f"Cannot locate project root from {here}")


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main():
    # -------------------------------------------------------------------------
    # 1. Resolve project structure & logging
    # -------------------------------------------------------------------------
    project_root = get_project_root()
    data_dir     = project_root / "data"
    logs_dir     = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=logs_dir / "transform_inputs.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting transform_inputs; project root = %s", project_root)

    # -------------------------------------------------------------------------
    # 2. Load paths from data/config.yaml
    # -------------------------------------------------------------------------
    cfg_path = project_root / "config.yaml"
    if not cfg_path.exists():
        logger.error("Missing data/config.yaml at %s", cfg_path)
        raise FileNotFoundError(cfg_path)

    cfg = load_yaml(cfg_path)
    payments_file   = data_dir / cfg.get("payments_file",      "synthetic_payments.json")
    questions_file  = data_dir / cfg.get("questions_file",     "questions.json")
    output_file     = data_dir / cfg.get("input_file",         "qa_inference_inputs.json")

    # -------------------------------------------------------------------------
    # 3. Load control definitions from root config.yaml
    # -------------------------------------------------------------------------
    root_cfg_path = project_root / "config.yaml"
    if not root_cfg_path.exists():
        logger.error("Missing root config.yaml at %s", root_cfg_path)
        raise FileNotFoundError(root_cfg_path)

    root_cfg = load_yaml(root_cfg_path)
    controls = root_cfg.get("controls", {})
    # e.g. controls["vendor_whitelist"], controls["allowed_currencies"], etc.

    # -------------------------------------------------------------------------
    # 4. Load questions and payments
    # -------------------------------------------------------------------------
    questions = json.loads((data_dir / questions_file).read_text(encoding="utf-8"))["questions"]
    with (payments_file).open(encoding="utf-8") as f:
        payments = json.load(f)

    logger.info("Loaded %d payments, %d questions, and controls: %s",
                len(payments), len(questions), list(controls.keys()))

    # -------------------------------------------------------------------------
    # 5. Required payment fields (ensure presence)
    # -------------------------------------------------------------------------
    required_fields = [
        "payment_id", "amount", "currency", "vendor",
        "payment_method", "initiated_by", "approved_by",
        "approval_threshold", "timestamp", "status",
        "approval_type", "approval_outcome", "approval_date"
    ]

    # -------------------------------------------------------------------------
    # 6. Enrich records
    # -------------------------------------------------------------------------
    enriched = []
    for rec in payments:
        # 6.a Fill missing required fields with None
        for field in required_fields:
            rec.setdefault(field, None)

        # 6.b Attach questions
        rec["questions"] = questions

        # 6.c Attach control lists for downstream evaluation
        rec["vendor_whitelist"]     = controls.get("vendor_whitelist", {})
        rec["allowed_currencies"]   = controls.get("allowed_currencies", [])
        rec["valid_statuses"]       = controls.get("valid_statuses", [])
        rec["authorized_approvers"] = controls.get("authorized_approvers", [])

        enriched.append(rec)

    # -------------------------------------------------------------------------
    # 7. Write enriched JSON array
    # -------------------------------------------------------------------------
    output_path = output_file if output_file.is_absolute() else data_dir / output_file
    with output_path.open("w", encoding="utf-8") as out_f:
        json.dump(enriched, out_f, indent=2, ensure_ascii=False)

    logger.info("Wrote %d enriched records to %s", len(enriched), output_path)


if __name__ == "__main__":
    main()
