import os
import glob

output_file = '/home/ec2-user/.gemini/antigravity-ide/brain/10305c78-e8c4-4e69-b47b-bf067636a3c8/20000_word_forensic_master_dossier.md'
project_dir = '/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT'

def get_file_content(filepath):
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except Exception as e:
        return f"[Error reading file: {e}]"

with open(output_file, 'w') as out:
    out.write("# THE 20,000 WORD FORENSIC MASTER DOSSIER\n\n")
    out.write("This document is the ultimate, exhaustive, code-level documentation of the entire MICCAI project. It includes the full logic, logs, and raw mathematical data outputs for every task.\n\n")
    
    # 1. Existing Monumental Breakdown
    monumental_path = '/home/ec2-user/.gemini/antigravity-ide/brain/10305c78-e8c4-4e69-b47b-bf067636a3c8/monumental_task_breakdown.md'
    if os.path.exists(monumental_path):
        out.write(get_file_content(monumental_path))
        out.write("\n\n---\n\n")
        
    # 2. Add Project Report and Verification Logs
    out.write("# PART II: VERIFICATION LOGS AND PROJECT STATE\n\n")
    out.write("## 1. Project Report\n")
    out.write(get_file_content(os.path.join(project_dir, 'PROJECT_REPORT.md')))
    out.write("\n\n## 2. Verification Report\n")
    out.write(get_file_content(os.path.join(project_dir, 'verification_report.txt')))
    out.write("\n\n---\n\n")
    
    # 3. Add Master Draft
    out.write("# PART III: THE MASTER LATEX MANUSCRIPT\n\n")
    out.write("```latex\n")
    out.write(get_file_content(os.path.join(project_dir, 'paper/reading_between_lesions_master_draft.tex')))
    out.write("\n```\n\n---\n\n")
    
    # 4. Add Every Single Task Script
    out.write("# PART IV: EXACT SOURCE CODE FOR EVERY MECHANISTIC TASK\n\n")
    task_files = sorted(glob.glob(os.path.join(project_dir, 'task*.py')))
    for task_file in task_files:
        basename = os.path.basename(task_file)
        out.write(f"## {basename}\n")
        out.write(f"```python\n{get_file_content(task_file)}\n```\n\n")
        
    # 5. Add All Result CSVs
    out.write("# PART V: RAW EMPIRICAL DATA (CSV OUTPUTS)\n\n")
    csv_files = sorted(glob.glob(os.path.join(project_dir, 'results', '*.csv')))
    # Also add task16_results.csv from root
    csv_files.append(os.path.join(project_dir, 'task16_results.csv'))
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            basename = os.path.basename(csv_file)
            out.write(f"## {basename}\n")
            out.write(f"```csv\n{get_file_content(csv_file)}\n```\n\n")
            
    # 6. Add Model Architecture code
    out.write("# PART VI: MODEL ARCHITECTURE SOURCE CODE\n\n")
    model_files = sorted(glob.glob(os.path.join(project_dir, 'models', '*.py')))
    for model_file in model_files:
        basename = os.path.basename(model_file)
        out.write(f"## {basename}\n")
        out.write(f"```python\n{get_file_content(model_file)}\n```\n\n")

print(f"Generated massive dossier at {output_file}")
