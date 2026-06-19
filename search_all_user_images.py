import os
import json

brain_dir = r"C:\Users\vinay\.gemini\antigravity-ide\brain"
for folder in os.listdir(brain_dir):
    folder_path = os.path.join(brain_dir, folder)
    if os.path.isdir(folder_path):
        log_file = os.path.join(folder_path, ".system_generated", "logs", "transcript.jsonl")
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        if data.get("source") == "USER_EXPLICIT" or data.get("type") == "USER_INPUT":
                            # Look inside data for any media or image attachment reference
                            content = data.get("content", "")
                            # Check if the content contains standard markdown image references or files
                            if ".png" in content or ".jpg" in content or ".jpeg" in content or ".webp" in content or "media" in content:
                                print(f"=== Conv: {folder} ===")
                                print(f"Step: {data.get('step_index')}")
                                print(content)
                                print("-" * 80)
            except Exception as e:
                pass
