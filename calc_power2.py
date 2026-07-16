import numpy as np
from scipy import stats

effect_size = 12.344995224941389
alpha = 0.05 / 98

def get_power(n, d, alpha):
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha/2, df)
    nc = d * np.sqrt(n)
    p = 1 - stats.nct.cdf(t_crit, df, nc) + stats.nct.cdf(-t_crit, df, nc)
    return p, t_crit, nc

for n in range(3, 10):
    p, tc, nc = get_power(n, effect_size, alpha)
    print(f"n={n}: power={p:.4f}, t_crit={tc:.2f}, nc={nc:.2f}")

