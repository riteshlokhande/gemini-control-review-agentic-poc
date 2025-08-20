import json, hashlib, datetime

with open("predictions.jsonl") as f:
    predictions = [json.loads(line) for line in f]

log_entries = []
for p in predictions:
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "input_hash": hashlib.md5(json.dumps(p["input"]).encode()).hexdigest(),
        "question": p["input"]["question"],
        "predicted_answer": p["answer"],
        "prompt_excerpt": p["prompt"][:300] + "..."
    }
    log_entries.append(entry)

with open("logs/audit_log.jsonl", "w") as f:
    for e in log_entries:
        f.write(json.dumps(e) + "\n")
