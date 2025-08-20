#!/usr/bin/env python3

import json
import logging
import sys
from pathlib import Path
import yaml

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def find_project_root(markers=("data", "logs")) -> Path:
    here = Path(__file__).resolve().parent
    for p in (here, *here.parents):
        if all((p / m).exists() for m in markers):
            return p
    raise RuntimeError(f"Cannot locate project root from {here}")

def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def render_json(obj: dict) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)

def build_example_block(example: dict, controls: dict) -> str:
    ctl = example.get("controls", controls)
    return (
        "Payment record:\n" +
        render_json(example["payment"]) + "\n\n" +
        "Controls:\n" +
        render_json(ctl) + "\n\n" +
        "QA pairs:\n" +
        render_json(example["qa_pairs"]) + "\n\n" +
        "---\n\n"
    )

def build_prompt(record: dict, few_shots: list, controls: dict) -> str:
    controls_str = render_json(controls)
    questions    = record.get("questions", [])
    payment_str  = render_json(record)

    prompt = (
        "Control definitions:\n" +
        f"{controls_str}\n\n" +
        "You are an auditor evaluating the effectiveness of payment controls.\n"
        "Return only a JSON array of objects with:\n"
        "- question: string\n"
        "- answer: one of \"Yes\", \"No\", \"Not Applicable\"\n\n"
    )

    for ex in few_shots:
        prompt += build_example_block(ex, controls)

    prompt += (
        "Now evaluate this payment:\n"
        "Payment record:\n" +
        f"{payment_str}\n\n" +
        "Controls:\n" +
        f"{controls_str}\n\n" +
        "Questions:\n" +
        f"{render_json(questions)}\n\n" +
        "Return a JSON array of QA pairs (no fences, no extra text):\n"
        "[\n"
        "  {\"question\": \"...\", \"answer\": \"Yes\"},\n"
        "  …\n"
        "]\n"
    )
    return prompt

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    root = find_project_root()
    data_dir = root / "data"
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Load configs
    cfg_root = load_yaml(root / "config.yaml")

    # Paths
    input_path  = data_dir / cfg_root["paths"]["historic_data"]
    output_path = data_dir / cfg_root["paths"]["few_shot_prompts"]
    log_path    = logs_dir / cfg_root["paths"].get("few_shot_log", "few_shot_generation.log")

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Project root: %s", root)
    logger.info("Input records: %s", input_path)
    logger.info("Output prompts: %s", output_path)

    # Load data
    try:
        records = json.loads(input_path.read_text(encoding="utf-8"))
        logger.info("Loaded %d payment records", len(records))
    except Exception as e:
        logger.error("Failed to load payment records: %s", e)
        sys.exit(1)

    few_shots = cfg_root.get("few_shot", {}).get("examples", [])
    controls  = cfg_root.get("controls", {})

    # Write prompts
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_f:
        for idx, record in enumerate(records, start=1):
            pid = record.get("payment_id", f"record-{idx}")
            logger.info("Building prompt %d/%d (%s)", idx, len(records), pid)
            try:
                prompt = build_prompt(record, few_shots, controls)
                out_f.write(json.dumps({"payment_id": pid, "prompt": prompt}, ensure_ascii=False) + "\n")
                logger.info("Wrote prompt for %s", pid)
            except Exception as ex:
                logger.exception("Error building prompt for %s: %s", pid, ex)

    logger.info("Few-shot prompt generation complete → %s", output_path)

if __name__ == "__main__":
    main()
