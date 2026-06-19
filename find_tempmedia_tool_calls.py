import os
import json

log_file = r"C:\Users\vinay\.gemini\antigravity-ide\brain\40073139-7d1f-4b93-b28e-a501188d7616\.system_generated\logs\transcript.jsonl"
with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        if ".tempmediaStorage" in line or ".png" in line:
            data = json.loads(line)
            print(f"Step: {data.get('step_index')}, Type: {data.get('type')}")
            # If there's a tool_calls key, print it
            if "tool_calls" in data:
                for tc in data["tool_calls"]:
                    print(f"  Tool: {tc.get('name')}")
                    # If it is browser subagent, print the task or media paths
                    if tc.get("name") == "browser_subagent":
                        print(f"  Browser Task: {tc.get('args', {}).get('TaskName')}")
                        print(f"  Browser MediaPaths: {tc.get('args', {}).get('MediaPaths')}")
            print("-" * 50)
