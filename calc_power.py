import json
import numpy as np
from scipy import stats

with open('results/experiment_progress.json', 'r') as f:
    data = json.load(f)

img_only = np.array(data['results']['Image-Only']['Accuracy'])
t2v = np.array(data['results']['Cross-Attention T\u2192V']['Accuracy'])

diff = t2v - img_only
mean_diff = np.mean(diff)
std_diff = np.std(diff, ddof=1)
effect_size = mean_diff / std_diff

print("Diffs:", diff)
print("Mean Diff:", mean_diff)
print("Std Diff:", std_diff)
print("Effect Size (Cohen's d):", effect_size)

alpha = 0.05 / 98
power_target = 0.80

def get_power(n, d, alpha):
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha/2, df)
    nc = d * np.sqrt(n)
    p = 1 - stats.nct.cdf(t_crit, df, nc) + stats.nct.cdf(-t_crit, df, nc)
    return p

for n in range(4, 1000):
    p = get_power(n, effect_size, alpha)
    if p >= power_target:
        print("Required N:", n)
        break

import re
with open('experiment_run.log', 'r') as f:
    log_text = f.read()

seed_42_blocks = log_text.split('COMMENCING RUN WITH SEED:')[1]
times = re.findall(r'Epoch \d+/\d+ \[.*?\]: 100%\|.*?\[(\d+):(\d+)<', seed_42_blocks)
total_seconds = sum(int(m)*60 + int(s) for m, s in times)
print("Total seconds for 1 seed:", total_seconds)
