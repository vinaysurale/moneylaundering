import json

log_file = r"C:\Users\vinay\.gemini\antigravity-ide\brain\6984f22f-89a6-42cb-bd23-625f1a285d9e\.system_generated\logs\transcript.jsonl"
with open(log_file, "r", encoding="utf-8") as f:
    line = f.readline()
    if line:
        data = json.loads(line)
        print(json.dumps(data, indent=2))
