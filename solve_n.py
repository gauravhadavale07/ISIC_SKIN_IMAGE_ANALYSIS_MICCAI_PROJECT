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
d = mean_diff / std_diff

print(f"Effect Size (Cohen's d): {d}")

alpha = 0.05 / 98

for n in range(2, 1000):
    df = n - 1
    t_required = stats.t.ppf(1 - alpha, df)
    t_stat = d * np.sqrt(n)
    if t_stat > t_required:
        print(f"Required N: {n}")
        additional_seeds = n - 3
        print(f"Additional Seeds: {additional_seeds}")
        total_time_seconds = additional_seeds * 92
        mins, secs = divmod(total_time_seconds, 60)
        hours, mins = divmod(mins, 60)
        print(f"Estimated wall-clock time: {hours:02d}:{mins:02d}:{secs:02d} ({total_time_seconds} seconds)")
        break
