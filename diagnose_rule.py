import os
import glob
import json

print("🔍 Scanning for raw experiment logs...")

# Find all JSON files and sort by most recently modified
files = glob.glob("**/*.json", recursive=True)
files.sort(key=os.path.getmtime, reverse=True)

if not files:
    print("⚠️ No JSON log files found. The arrays might only exist in your terminal history.")
else:
    for f in files[:3]: # Check the 3 most recent files
        print(f"\n📄 Inspecting: {f}")
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                # Pretty print the dictionary so we can read the raw arrays
                print(json.dumps(data, indent=2))
        except Exception as e:
            print(f"Could not read {f}: {e}")