import yaml, json
import google.generativeai as genai

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Configure Gemini
genai.configure(api_key=config["gemini"]["api-key"])
model = genai.GenerativeModel(config["model"]["name"])

# Payment system topics
topics = [
    "Payment authorization controls",
    "Reconciliation between payment ledger and bank statements",
    "Monitoring duplicate payments",
    "Validation of vendor bank details",
    "Segregation of duties in payment processing",
    "Threshold-based approval workflows",
    "Audit trail integrity for payment transactions",
    "Exception handling in payment failures",
    "Sensitive data protection in payment files",
    "Automated alerts for unusual payment patterns"
]

# Prompt template
def build_prompt(topic):
    return f"""
Generate 30 synthetic QA pairs for the topic: "{topic}".
Each pair should include:
- Context: 2–3 sentences describing a realistic payment scenario
- Question: Relevant to control monitoring or testing
- Answer: Concise, technically sound, and control-aware

Format each QA pair as a JSON object with keys: context, question, answer.
Output as JSONL entries.
"""

# Generate and save
with open("data/synthetic_qa_payments.jsonl", "w") as f:
    for topic in topics:
        response = model.generate_content(
            build_prompt(topic),
            generation_config={
                "temperature": config["model"]["temperature"],
                "max_output_tokens": config["model"]["max_tokens"]
            }
        )
        for line in response.text.strip().split("\n"):
            if line.strip():
                f.write(line + "\n")
