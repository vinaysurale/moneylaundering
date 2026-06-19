import os
import time

start_dir = r"C:\Users\vinay"
now = time.time()
three_days_ago = now - 3 * 24 * 3600

print("Scanning for recent images...")
count = 0
for root, dirs, files in os.walk(start_dir):
    # skip AppData except .gemini, skip node_modules, skip virtual environments
    if "AppData" in root and ".gemini" not in root:
        continue
    if "node_modules" in root or ".venv" in root or "venv" in root or ".git" in root:
        continue
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".svg")):
            filepath = os.path.join(root, file)
            try:
                mtime = os.path.getmtime(filepath)
                if mtime > three_days_ago:
                    print(f"File: {filepath}")
                    print(f"  Modified: {time.ctime(mtime)}")
                    count += 1
                    if count >= 100:
                        print("Reached cap of 100 files")
                        break
            except Exception:
                pass
    if count >= 100:
        break
print(f"Scan complete. Found {count} files.")
