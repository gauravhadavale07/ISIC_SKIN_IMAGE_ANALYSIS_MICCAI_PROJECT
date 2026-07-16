import os

files_to_concat = [
    "PROJECT_REPORT.md",
    "THESIS_DRAFT.md",
    "verification_report.md"
]

content = ""
for f in files_to_concat:
    if os.path.exists(f):
        with open(f, 'r') as file:
            content += f"\n\n# {f}\n\n"
            content += file.read()

with open("All_Experiments.md", "w") as out:
    out.write(content)

print(f"Combined into All_Experiments.md, length {len(content)}")
