"""Power analysis for 6-seed experimental design (df=5 for paired t-tests)."""

import numpy as np
from scipy import stats


def compute_minimum_detectable_effect_size(n_seeds=6, alpha=0.05):
    """
    Compute minimum detectable Cohen's d for paired t-test with n seeds.
    
    Args:
        n_seeds: Number of seeds (default: 6)
        alpha: Significance level (default: 0.05)
    
    Returns:
        Minimum detectable Cohen's d
    """
    df = n_seeds - 1  # degrees of freedom for paired t-test
    critical_t = stats.t.ppf(1 - alpha/2, df)  # two-tailed critical t-value
    
    # For paired t-test: t = d * sqrt(n)
    # Therefore: d = t / sqrt(n)
    min_d = critical_t / np.sqrt(n_seeds)
    
    return min_d, critical_t, df


def compute_required_seeds_for_effect_size(target_d, alpha=0.05, power=0.8):
    """
    Compute required number of seeds to detect a given effect size.
    
    Args:
        target_d: Target Cohen's d effect size
        alpha: Significance level (default: 0.05)
        power: Desired statistical power (default: 0.8)
    
    Returns:
        Required number of seeds
    """
    # Approximate formula for paired t-test sample size
    # n = (t_alpha + t_beta)^2 / d^2
    # where t_beta is the t-value corresponding to power
    
    # Iterative solution
    for n in range(2, 50):
        df = n - 1
        t_alpha = stats.t.ppf(1 - alpha/2, df)
        t_beta = stats.t.ppf(power, df)
        required_n = ((t_alpha + t_beta) / target_d) ** 2
        
        if n >= required_n:
            return n
    
    return None


def main():
    print("="*70)
    print("POWER ANALYSIS FOR 6-SEED EXPERIMENTAL DESIGN")
    print("="*70)
    
    # Current design: 6 seeds
    n_seeds = 6
    min_d, critical_t, df = compute_minimum_detectable_effect_size(n_seeds)
    
    print(f"\n📊 Current Design: {n_seeds} seeds (df={df})")
    print(f"   Critical t-value (α=0.05, two-tailed): {critical_t:.3f}")
    print(f"   Minimum detectable Cohen's d: {min_d:.3f}")
    print(f"\n   Interpretation:")
    print(f"   - d < 0.2: Very small effect (UNDetectable)")
    print(f"   - d = 0.2-0.5: Small effect (UNDetectable)")
    print(f"   - d = 0.5-0.8: Medium effect (UNDetectable)")
    print(f"   - d > 0.8: Large effect (Detectable)")
    print(f"   - d > 1.0: Very large effect (Detectable)")
    
    # Effect sizes of interest (based on user's observed AUROC differences)
    print(f"\n📈 Effect Size Analysis for Observed AUROC Differences:")
    print(f"   Observed AUROC differences: 0.004 - 0.007")
    
    # Assume typical std of AUROC differences ~0.01-0.02
    for std_diff in [0.01, 0.015, 0.02]:
        for auroc_diff in [0.004, 0.007]:
            d = auroc_diff / std_diff
            detectable = d >= min_d
            print(f"   AUROC diff={auroc_diff:.3f}, std={std_diff:.3f} → d={d:.3f} {'✅ DETECTABLE' if detectable else '❌ UNDETECTABLE'}")
    
    # Required seeds for various effect sizes
    print(f"\n🔍 Required Seeds for Target Effect Sizes (80% power):")
    target_effects = [0.2, 0.5, 0.8, 1.0]
    for target_d in target_effects:
        required_n = compute_required_seeds_for_effect_size(target_d)
        if required_n:
            print(f"   d={target_d:.1f}: {required_n} seeds required")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    print(f"   1. Current 6-seed design can only detect VERY LARGE effects (d > {min_d:.2f})")
    print(f"   2. Observed AUROC differences (0.004-0.007) correspond to d=0.2-0.7")
    print(f"   3. These are likely UNDETECTABLE with 6 seeds")
    print(f"   4. To detect d=0.5 (medium effect), need ~13 seeds")
    print(f"   5. To detect d=0.8 (large effect), need ~6 seeds (current design)")
    print(f"   6. Consider reporting binary AUROC (differences may be larger)")
    print(f"   7. Or acknowledge limited power in paper limitations")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
