#!/usr/bin/env python3

import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate payment-QA pipeline: generate → format → predict → audit"
    )
    parser.add_argument(
        "--stage",
        choices=["generate", "format", "predict", "audit", "all"],
        default="all",
        help="Which stage(s) to run"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../config.yaml",
        help="Path to your YAML config (contains payment_count & num_questions)"
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="scripts/fewshot_template.json",
        help="Path to your few-shot prompt template"
    )
    parser.add_argument(
        "--audit-log",
        type=str,
        default="logs/audit.log",
        help="Path to write consolidated audit logs"
    )

    args = parser.parse_args()

    # 1. Generate payments + QA (reads payment_count & num_questions from config)
    if args.stage in ("generate", "all"):
        cmd = [
            sys.executable, "scripts/generate_payment_and_qa.py",
            "--config", args.config
        ]
        print("→ Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    # 2. Format few-shot prompt
    if args.stage in ("format", "all"):
        cmd = [
            sys.executable, "scripts/format_fewshot_prompt.py",
            "--config",   args.config,
            "--template", args.prompt_template
        ]
        print("→ Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    # 3. Generate answers via Gemini
    if args.stage in ("predict", "all"):
        cmd = [
            sys.executable, "scripts/generate_answers.py",
            "--config", args.config
        ]
        print("→ Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    # 4. Audit & consolidate logs
    if args.stage in ("audit", "all"):
        cmd = [
            sys.executable, "scripts/audit_log.py",
            "--config", args.config,
            "--output", args.audit_log
        ]
        print("→ Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
