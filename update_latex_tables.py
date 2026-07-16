import re

# Read stats
with open('table12_stats.txt', 'r') as f:
    stats_lines = f.readlines()

t1_stats = {}
t2_stats = {}
current_table = 0
current_arch = None
for raw_line in stats_lines:
    line = raw_line.strip()
    if 'TABLE 1' in line:
        current_table = 1
        continue
    elif 'TABLE 2' in line:
        current_table = 2
        continue
    
    if current_table == 1 and line:
        if ':' in line:
            arch, rest = line.split(':', 1)
            arch = arch.strip()
            if arch == "Image-Only": arch = "Image-Only Baseline"
            if arch == "Text-Only": arch = "Text-Only Baseline"
            if arch == "Cross-Attention T->V": arch = "Cross-Attention (T$\\rightarrow$V)"
            if arch == "Cross-Attention (V->T)": arch = "Cross-Attention (V$\\rightarrow$T)"
            parts = rest.split('|')
            acc = parts[0].split('=')[1].strip()
            f1 = parts[1].split('=')[1].strip()
            auc = parts[2].split('=')[1].strip()
            t1_stats[arch] = (acc, f1, auc)
    elif current_table == 2 and line:
        if ':' in line and not raw_line.startswith('  '):
            arch = line.split(':')[0].strip()
            if arch == "Cross-Attention T->V": arch = "Cross-Attention (T$\\rightarrow$V)"
            if arch == "Cross-Attention (V->T)": arch = "Cross-Attention (V$\\rightarrow$T)"
            current_arch = arch
            t2_stats[current_arch] = {}
        elif current_arch and raw_line.startswith('  '):
            key, val = line.split(':', 1)
            t2_stats[current_arch][key.strip()] = val.strip()

# Also read task1_output.txt if available
t3_stats = {}
try:
    with open('task1_output.txt', 'r') as f:
        t1_out = f.read()
    for block in t1_out.split('Evaluating: '):
        if 'Holm-corrected p-value:' in block:
            arch = block.split('\n')[0].strip()
            if arch == "Cross-Attention V->T": arch = "Cross-Attention (V$\\rightarrow$T)"
            if arch == "Cross-Attention T->V": arch = "Cross-Attention (T$\\rightarrow$V)"
            real_acc = re.search(r'Mean Real Accuracy:\s+([0-9\.\\]+)', block).group(1)
            shuf_acc = re.search(r'Mean Shuffle Accuracy:\s+([0-9\.\\]+)', block).group(1)
            delta = re.search(r'Mean Delta \(Real - Shuffle\):\s+([\+\-0-9\. ]+)', block).group(1)
            pval = re.search(r'Holm-corrected p-value:\s+([^\n]+)', block).group(1).strip()
            t3_stats[arch] = (real_acc, shuf_acc, delta.strip(), pval.replace('<', '$<$'))
except Exception as e:
    print(f"Skipping Table 3: {e}")

# Apply to main.tex and main_template.tex
for filename in ['paper/main_template.tex', 'paper/main.tex']:
    with open(filename, 'r') as f:
        content = f.read()

    # Replace Table 1
    for arch, (acc, f1, auc) in t1_stats.items():
        pattern = re.compile(re.escape(arch) + r' & [^&\n]*\\pm[^&\n]* & [^&\n]*\\pm[^&\n]* & [^\\\&\n]*\\pm[^\\\&\n]* \\\\')
        a = acc.replace('\\pm', '$\\pm$')
        f = f1.replace('\\pm', '$\\pm$')
        u = auc.replace('\\pm', '$\\pm$')
        replacement = f"{arch} & {a}\\% & {f}\\% & {u} \\\\"
        content = pattern.sub(replacement.replace('\\', '\\\\'), content)

    # Replace Table 2
    for arch, st in t2_stats.items():
        dagger1 = "$^\\dagger$" if arch in ["Image-Only", "Text-Only"] else ""
        dagger2 = "$^\\dagger$" if arch in ["Image-Only", "Text-Only"] else ""
        star = "*" if "V$\\rightarrow$T" in arch else ""
        
        pattern = re.compile(re.escape(arch) + r' & [^&\n]*\\pm[^&\n]* & [^&\n]*\\pm[^&\n]* & [^&\n]*\\pm[^&\n]* & [^&\n]*\\pm[^&\n]* & [^&\n]*\\pm[^&\n]* & [^\\\&\n]*\\pm[^\\\&\n]* \\\\')
        a = st['Acc'].replace('\\pm', '$\\pm$')
        au = st['AUC'].replace('\\pm', '$\\pm$')
        f = st['F1'].replace('\\pm', '$\\pm$')
        c = st['CFR'].replace('\\pm', '$\\pm$')
        d = st['DelP'].replace('\\pm', '$\\pm$')
        ck = st['CKA'].replace('\\pm', '$\\pm$')
        replacement = f"{arch} & {a}\\% & {au} & {f}\\% & {c}\\%{dagger1}{star} & {d} pp{star} & {ck}{dagger2} \\\\"
        content = pattern.sub(replacement.replace('\\', '\\\\'), content)
        
    # Replace Table 3
    for arch, (real, shuf, delta, pval) in t3_stats.items():
        # Late Fusion & 41.21$\pm$0.00\% & 40.96$\pm$0.21\% & +0.25 pp & 0.22569 \\
        pattern = re.compile(re.escape(arch) + r' & [0-9\.\\]+\%? & [0-9\.\\]+\%? & [\+\-0-9\. ]+pp & [0-9\.\$\\< ]+ \\\\')
        
        if '<' in pval:
            pval = f"\\textbf{{{pval}}}"
            
        # Ensure delta has + if positive
        if not delta.startswith('-') and not delta.startswith('+'):
            delta = '+' + delta
            
        replacement = f"{arch} & {real}\\% & {shuf}\\% & {delta} pp & {pval} \\\\"
        content = pattern.sub(replacement.replace('\\', '\\\\'), content)

    with open(filename, 'w') as f:
        f.write(content)
    print(f"Updated {filename}")
