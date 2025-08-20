#!/usr/bin/env python3
"""
generate_payment_and_qa.py

1) Generate synthetic payment records with full attribute coverage
2) For each record, inject control definitions, questions, and few-shot examples
3) Build prompts in parallel, call Gemini to get Yes/No/Not Applicable answers
4) Normalize, log, and write out:
   - data/synthetic_payments.json
   - data/qa_predictions.jsonl
"""

import json
import logging
import random
import re
import sys
import time

from uuid import uuid4
from datetime import datetime, timedelta, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
import google.generativeai as genai
from google.generativeai import GenerativeModel

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

def build_prompt(record: dict,
                 questions: list,
                 controls: dict,
                 few_shots: list) -> str:
    # 1) Global control definitions
    prompt  = "Control definitions:\n"
    prompt += json.dumps(controls, indent=2, ensure_ascii=False) + "\n\n"
    # 2) Output schema & instructions
    prompt += (
        "Output schema:\n"
        "[\n"
        '  { "question": "string", "answer": "Yes|No|Not Applicable" }\n'
        "]\n\n"
        "You are an auditor assessing payment controls. "
        "Return ONLY a JSON array matching the schema above. No extra text.\n\n"
    )
    # 3) Few-shot examples
    for ex in few_shots:
        ex_controls = ex.get("controls", controls)
        prompt += "Payment record (example):\n"
        prompt += json.dumps(ex["payment"], indent=2, ensure_ascii=False) + "\n\n"
        prompt += "Controls (example):\n"
        prompt += json.dumps(ex_controls, indent=2, ensure_ascii=False) + "\n\n"
        prompt += "QA pairs (example):\n"
        prompt += json.dumps(ex["qa_pairs"], indent=2, ensure_ascii=False) + "\n\n"
        prompt += "---\n\n"
    # 4) Target record + controls + questions
    prompt += "Payment record:\n"
    prompt += json.dumps(record, indent=2, ensure_ascii=False) + "\n\n"
    prompt += "Controls:\n"
    prompt += json.dumps(controls, indent=2, ensure_ascii=False) + "\n\n"
    prompt += "Questions:\n"
    prompt += json.dumps(questions, indent=2, ensure_ascii=False) + "\n\n"
    # 5) Final ask
    prompt += (
        "Return a JSON array of QA pairs:\n"
        "[\n"
        '  { "question": "...", "answer": "Yes" },\n'
        "  …\n"
        "]\n"
    )
    return prompt

def generate_for_record(record: dict,
                        questions: list,
                        controls: dict,
                        few_shots: list,
                        model: GenerativeModel,
                        gen_cfg: dict,
                        logger: logging.Logger) -> dict:
    pid = record["payment_id"]
    prompt = build_prompt(record, questions, controls, few_shots)
    start = time.time()
    resp = model.generate_content(prompt, generation_config=gen_cfg)
    duration = time.time() - start
    logger.info("Gemini call for %s took %.2f sec", pid, duration)
    arr = extract_json_array(resp.text)
    answers = [
        {
            "question": qa.get("question", "").strip(),
            "answer":   normalize_answer(qa.get("answer", ""))
        }
        for qa in arr
    ]
    return {
        "payment_id": pid,
        "payment":    record,
        "questions":  questions,
        "answers":    answers
    }

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # 1) Resolve paths & logging
    root = find_project_root()
    data = root / "data"
    logs = root / "logs"
    logs.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logs / "synthetic_data_generation.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting generate_payment_and_qa; project_root=%s", root)

    # 2) Load configs
    cfg_root = load_yaml(root / "config.yaml")

    # 3) Gemini & model setup
    genai.configure(api_key=cfg_root["gemini"]["api-key"])
    model   = GenerativeModel(cfg_root["model"]["name"])
    gen_cfg = {
        "temperature":       cfg_root["model"]["temperature"],
        "max_output_tokens": cfg_root["model"]["max_tokens"]
    }

    # 4) Control defs, few-shots, questions
    controls  = cfg_root.get("controls", {})
    few_shots = cfg_root.get("few_shot", {}).get("examples", [])
    questions = json.loads((data / cfg_root["paths"]["questions_file"]).read_text())

    # 5) Synthetic data params
    num_records   = cfg_root["generation"]["payment_count"]
    vendor_map    = controls.get("vendor_whitelist", {})
    approvers     = controls.get("authorized_approvers", [])
    currencies    = controls.get("allowed_currencies", [])
    statuses      = controls.get("valid_statuses", [])
    initiators    = cfg_root.get("initiators", [])
    batch_size    = cfg_root.get("generation", {}).get("batch_size", 5)

    # 6) Generate synthetic payments
    logger.info("Generating %d synthetic payment records", num_records)
    synthetic = []
    for _ in range(num_records):
        pid = str(uuid4())
        now = datetime.now(timezone.utc)
        ts  = now - timedelta(days=random.randint(0,30), hours=random.randint(0,23))
        adt = ts + timedelta(hours=random.randint(0,48))

        method   = random.choice(list(vendor_map.keys())) if vendor_map else "UPI"
        vendor   = random.choice(vendor_map.get(method, ["Unknown"]))
        currency = random.choice(currencies) if currencies else "INR"
        status   = random.choice(statuses)   if statuses   else "Processed"
        amt      = round(random.uniform(1_000, 500_000), 2)
        thr      = round(random.uniform(50_000, 300_000), 2)

        if amt > thr:
            atype, outcome = "Dual", random.choice(["Approved", "Rejected"])
        else:
            atype, outcome = "Single", "Approved"

        rec = {
            "payment_id":        pid,
            "amount":            f"{amt:.2f}",
            "currency":          currency,
            "vendor":            vendor,
            "payment_method":    method,
            "initiated_by":      random.choice(initiators) if initiators else "user001",
            "approved_by":       random.choice(approvers)  if approvers  else "manager001",
            "approval_threshold":f"{thr:.2f}",
            "timestamp":         ts.isoformat(),
            "approval_date":     adt.isoformat(),
            "status":            status,
            "approval_type":     atype,
            "approval_outcome":  outcome
        }
        synthetic.append(rec)

    # 7) Persist synthetic data
    syn_path = data / cfg_root["paths"]["historic_data"]
    data.mkdir(exist_ok=True)
    with syn_path.open("w", encoding="utf-8") as f:
        json.dump(synthetic, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d synthetic records → %s", len(synthetic), syn_path)

    # 8) Batch‐process QA calls
    hist_qa_path = data / cfg_root["paths"]["historic_qa_data"]
    with ThreadPoolExecutor(max_workers=batch_size) as exec, \
         hist_qa_path.open("w", encoding="utf-8")    as out_f:

        futures = {
            exec.submit(
                generate_for_record,
                rec, questions, controls, few_shots, model, gen_cfg, logger
            ): rec["payment_id"] for rec in synthetic
        }

        for fut in as_completed(futures):
            result = fut.result()
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            logger.info(
                "Recorded %d answers for %s",
                len(result["answers"]), result["payment_id"]
            )

    logger.info("All QA predictions written → %s", hist_qa_path)


if __name__ == "__main__":
    main()
