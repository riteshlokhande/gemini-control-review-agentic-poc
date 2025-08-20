#!/usr/bin/env python3
"""
generate_answers.py

1) Discovers project root (data/ + logs/).
2) Loads configs (root/config.yaml and data/config.yaml).
3) Reads enriched payment records with questions.
4) Loads control definitions and few-shot examples (inline or JSONL).
5) Builds prompts in batches, calls Gemini via generate_batch or parallel generate_content.
6) Normalizes answers and writes out data/qa_predictions.jsonl.
"""

import json
import logging
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
import google.generativeai as genai
from google.generativeai import GenerativeModel


def find_project_root(markers=("data", "logs")) -> Path:
    here = Path(__file__).resolve().parent
    for p in (here, *here.parents):
        if all((p / m).exists() for m in markers):
            return p
    raise RuntimeError(f"Cannot locate project root from {here}")


def normalize_answer(raw: str) -> str:
    key = raw.strip().strip('"').lower()
    return {"yes": "Yes", "no": "No", "not applicable": "Not Applicable"}.get(key, "Not Applicable")


def extract_json_array(text: str) -> list:
    clean = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start = clean.find("[")
    end   = clean.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    return json.loads(clean[start:end])


def load_jsonl(path: Path) -> list:
    examples = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                examples.append(json.loads(line))
        logging.info("Loaded %d few-shot examples from %s", len(examples), path)
    except FileNotFoundError:
        logging.warning("Few-shot file not found at %s", path)
    except json.JSONDecodeError as e:
        logging.error("Failed to parse few-shot JSONL %s: %s", path, e)
    return examples


def build_prompt(record: dict, few_shots: list, controls: dict) -> str:
    # 1) Global control definitions
    prompt = "Control definitions:\n" + json.dumps(controls, indent=2, ensure_ascii=False) + "\n\n"
    # 2) Schema & instructions
    prompt += (
        "Output schema:\n"
        "[\n"
        '  { "question": "string", "answer": "Yes|No|Not Applicable" }\n'
        "]\n\n"
        "You are an auditor assessing payment controls. "
        "Return ONLY a JSON array matching the schema above. No extra text.\n\n"
    )
    # 3) Few-shot examples
    if few_shots:
        for ex in few_shots:
            ex_controls = ex.get("controls", controls)
            prompt += "Payment record (example):\n"
            prompt += json.dumps(ex["payment"], indent=2, ensure_ascii=False) + "\n\n"
            prompt += "Controls (example):\n"
            prompt += json.dumps(ex_controls, indent=2, ensure_ascii=False) + "\n\n"
            prompt += "QA pairs (example):\n"
            prompt += json.dumps(ex["qa_pairs"], indent=2, ensure_ascii=False) + "\n\n"
            prompt += "---\n\n"
    else:
        prompt += "(No few-shot examples configured)\n\n"
    # 4) Target record + controls + questions
    prompt += "Payment record:\n"
    prompt += json.dumps(record, indent=2, ensure_ascii=False) + "\n\n"
    prompt += "Controls:\n"
    prompt += json.dumps(controls, indent=2, ensure_ascii=False) + "\n\n"
    prompt += "Questions:\n"
    prompt += json.dumps(record["questions"], indent=2, ensure_ascii=False) + "\n\n"
    # 5) Final ask
    prompt += (
        "Return a JSON array of QA pairs:\n"
        "[\n"
        '  { "question": "...", "answer": "Yes" },\n'
        "  …\n"
        "]\n"
    )
    return prompt


def main():
    # 1) Resolve project root
    root = find_project_root()
    data_dir = root / "data"
    logs_dir = root / "logs"
    logs_dir.mkdir(exist_ok=True)

    # 2) Load configs
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))

    # 3) Setup logging
    logging.basicConfig(
        filename=logs_dir / "generate_answers.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.info("Project root: %s", root)

    # 4) Configure Gemini
    genai.configure(api_key=cfg["gemini"]["api-key"])
    model = GenerativeModel(cfg["model"]["name"])
    gen_cfg = {
        "temperature":       cfg["model"]["temperature"],
        "max_output_tokens": cfg["model"]["max_tokens"]
    }

    # 5) Load few-shot examples (inline → fallback JSONL)
    few_shots = cfg.get("few_shot", {}).get("examples", [])
    few_shot_file = cfg.get("few_shot", {}).get("file")
    if not few_shots and few_shot_file:
        few_shots = load_jsonl(data_dir / few_shot_file)
    if not few_shots:
        logger.warning("Proceeding zero-shot (no few-shot examples)")

    # 6) Load controls
    controls = cfg.get("controls", {})
    if not controls:
        logger.warning("No controls defined in config.yaml")

    # 7) Paths for input & output
    input_path       = data_dir / cfg["paths"]["input_file"]
    predictions_path = data_dir / cfg["paths"]["predictions_file"]
    if not input_path.exists():
        logger.error("Missing input file: %s", input_path)
        sys.exit(1)

    # 8) Load enriched records
    records = json.loads(input_path.read_text(encoding="utf-8"))
    logger.info("Loaded %d records", len(records))

    # 9) Batch settings
    batch_size = cfg.get("generation", {}).get("batch_size", len(records))

    # 10) Process in batches
    with predictions_path.open("w", encoding="utf-8") as out_f:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            prompts = [build_prompt(r, few_shots, controls) for r in batch]

            # 10a) Try true batch endpoint
            try:
                if hasattr(model, "generate_batch"):
                    responses = model.generate_batch(
                        prompts=prompts,
                        generation_config=gen_cfg
                    )
                    texts = [resp.text for resp in responses]
                    logger.info("Batch-requested %d prompts via generate_batch", len(prompts))
                else:
                    raise AttributeError
            except AttributeError:
                # 10b) Fallback to parallel single calls
                texts = [None] * len(prompts)
                with ThreadPoolExecutor(max_workers=batch_size) as exe:
                    future_to_idx = {
                        exe.submit(model.generate_content, p, generation_config=gen_cfg): idx
                        for idx, p in enumerate(prompts)
                    }
                    for fut in as_completed(future_to_idx):
                        idx = future_to_idx[fut]
                        try:
                            texts[idx] = fut.result().text
                        except Exception as e:
                            logger.error("Prompt-%d failed: %s", idx, e)
                            texts[idx] = ""
                logger.info("Processed %d prompts in parallel", len(prompts))

            # 11) Parse and write results
            for rec, txt in zip(batch, texts):
                pid = rec.get("payment_id", "<no-id>")
                arr = extract_json_array(txt)
                answers = [
                    {"question": qa.get("question", "").strip(),
                     "answer":   normalize_answer(qa.get("answer", ""))}
                    for qa in arr
                ]
                out_f.write(json.dumps({"payment_id": pid, "answers": answers}, ensure_ascii=False) + "\n")
                logger.info("Wrote %d answers for %s", len(answers), pid)

    logger.info("All predictions written to %s", predictions_path)


if __name__ == "__main__":
    main()
