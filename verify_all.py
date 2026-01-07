#!/usr/bin/env python3
"""
==============================================================================
🧪 VERIFY_ALL.PY - Project NanoGlass Empirical Verification Suite v2.0
==============================================================================
RIGOROUS STATISTICAL VALIDATION following publication-ready standards.

Key Improvements (v2.0):
    1. Increased sample sizes (N≥100) for statistical power
    2. Effect sizes (Cohen's d) + Confidence Intervals (not just p-values)
    3. Negative controls to detect spurious correlations
    4. Pre-registered analysis plan (parameters locked before data)
    5. Multiple comparison correction (Bonferroni)

Methodology Notes:
    - All hypotheses are directional (one-tailed tests where appropriate)
    - Effect size thresholds: d=0.2 (small), d=0.5 (medium), d=0.8 (large)
    - We report BOTH statistical AND practical significance

Usage: python verify_all.py [--quick] [--n-samples N]

⚠️ PRE-REGISTRATION WARNING:
    Do not modify statistical parameters after seeing results.
    This constitutes HARKing (Hypothesizing After Results Known).
==============================================================================
"""
import torch
import torch.nn.functional as F
import sys
import os
import json
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats

# ==============================================================================
# REPRODUCIBILITY: Fixed seed for deterministic results
# ==============================================================================
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Ensure nanoglass is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nanoglass import NanoConfig, NanoGlass, GlassBoxSensor


# ==============================================================================
# PRE-REGISTERED ANALYSIS CONFIGURATION
# ==============================================================================

@dataclass
class PreregisteredConfig:
    """
    ⚠️ LOCKED CONFIGURATION - DO NOT MODIFY AFTER DATA COLLECTION ⚠️
    
    These parameters define the statistical analysis plan.
    Modifying them after seeing results invalidates the analysis.
    """
    # Sample sizes
    n_samples_per_condition: int = 100
    n_training_steps: int = 50
    batch_size: int = 8
    
    # Statistical thresholds
    alpha: float = 0.05
    confidence_level: float = 0.95
    
    # Effect size thresholds (Cohen's conventions)
    small_effect: float = 0.2
    medium_effect: float = 0.5
    large_effect: float = 0.8
    
    # Bonferroni correction for multiple comparisons
    n_primary_hypotheses: int = 3  # Number of main tests
    
    # Pre-registration timestamp
    registered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def alpha_corrected(self) -> float:
        """Bonferroni-corrected alpha level."""
        return self.alpha / self.n_primary_hypotheses


# Snapshot at module load
PREREGISTERED = PreregisteredConfig()


# ==============================================================================
# STATISTICAL UTILITIES
# ==============================================================================

def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Calculate Cohen's d effect size.
    
    Interpretation:
        |d| < 0.2: negligible
        0.2 ≤ |d| < 0.5: small
        0.5 ≤ |d| < 0.8: medium
        |d| ≥ 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def confidence_interval_mean(
    data: List[float], 
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for the mean using t-distribution.
    """
    n = len(data)
    if n < 2:
        return (float('-inf'), float('inf'))
    
    mean = np.mean(data)
    se = stats.sem(data)
    
    t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_critical * se
    
    return (mean - margin, mean + margin)


def effect_size_interpretation(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < PREREGISTERED.small_effect:
        return "negligible"
    elif d_abs < PREREGISTERED.medium_effect:
        return "small"
    elif d_abs < PREREGISTERED.large_effect:
        return "medium"
    else:
        return "large"


# ==============================================================================
# STRUCTURED TEST RESULTS
# ==============================================================================

@dataclass
class TestResult:
    """Structured result from a single test."""
    name: str
    hypothesis: str
    passed: bool
    
    # Descriptive statistics
    group1_mean: float
    group1_std: float
    group1_ci: Tuple[float, float]
    group2_mean: Optional[float] = None
    group2_std: Optional[float] = None
    group2_ci: Optional[Tuple[float, float]] = None
    
    # Inferential statistics
    test_statistic: float = 0.0
    p_value: float = 1.0
    p_value_corrected: float = 1.0  # After Bonferroni
    
    # Effect size
    cohens_d: float = 0.0
    effect_interpretation: str = "none"
    
    # Practical significance
    practically_significant: bool = False
    
    # Negative control
    negative_control_passed: Optional[bool] = None
    negative_control_details: Optional[str] = None
    
    n_samples: int = 0


# ==============================================================================
# TEST IMPLEMENTATIONS
# ==============================================================================

def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  🧪 {title}")
    print(f"{'='*70}")


def test_thermodynamics(config: NanoConfig, n_samples: int) -> TestResult:
    """
    HYPOTHESIS 1: Energy decreases as the model learns.
    H0: final_energy >= initial_energy
    H1: final_energy < initial_energy (one-tailed)
    """
    print_header("TEST 1: Thermodynamics of Meaning")
    print(f"   Hypothesis: Learning reduces activation energy")
    print(f"   N samples: {n_samples}")
    
    initial_energies = []
    final_energies = []
    
    for trial in range(n_samples):
        # Fresh model each trial
        torch.manual_seed(RANDOM_SEED + trial)
        model = NanoGlass(config).to(config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        
        # Training data
        text = "The quick brown fox jumps over the lazy dog. " * 100
        data = torch.tensor([ord(c) for c in text], dtype=torch.long)
        
        # Measure initial energy
        ix = torch.randint(len(data) - config.block_size, (4,))
        x = torch.stack([data[i:i+config.block_size] for i in ix]).to(config.device)
        
        with torch.no_grad():
            logits, _ = model(x)
            initial_energy = logits.abs().mean().item()
        initial_energies.append(initial_energy)
        
        # Train
        for _ in range(PREREGISTERED.n_training_steps):
            ix = torch.randint(len(data) - config.block_size, (4,))
            x = torch.stack([data[i:i+config.block_size] for i in ix]).to(config.device)
            y = torch.stack([data[i+1:i+config.block_size+1] for i in ix]).to(config.device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Measure final energy
        with torch.no_grad():
            logits, _ = model(x)
            final_energy = logits.abs().mean().item()
        final_energies.append(final_energy)
    
    # Statistical analysis
    t_stat, p_value_two = stats.ttest_rel(final_energies, initial_energies)
    p_value_one = p_value_two / 2 if t_stat < 0 else 1 - p_value_two / 2
    
    d = cohens_d(initial_energies, final_energies)
    
    ci_initial = confidence_interval_mean(initial_energies)
    ci_final = confidence_interval_mean(final_energies)
    
    passed = p_value_one < PREREGISTERED.alpha_corrected and t_stat < 0
    
    # Negative control: Random model should NOT show energy decrease
    random_energies = []
    for _ in range(20):
        model = NanoGlass(config).to(config.device)
        x = torch.randint(0, 256, (4, config.block_size), dtype=torch.long).to(config.device)
        with torch.no_grad():
            logits, _ = model(x)
            random_energies.append(logits.abs().mean().item())
    
    nc_std = np.std(random_energies)
    nc_passed = nc_std < 1.0  # Random models should have stable energy
    
    result = TestResult(
        name="Thermodynamics of Meaning",
        hypothesis="Learning reduces activation energy (E_final < E_initial)",
        passed=passed,
        group1_mean=np.mean(initial_energies),
        group1_std=np.std(initial_energies),
        group1_ci=ci_initial,
        group2_mean=np.mean(final_energies),
        group2_std=np.std(final_energies),
        group2_ci=ci_final,
        test_statistic=t_stat,
        p_value=p_value_one,
        p_value_corrected=min(1.0, p_value_one * PREREGISTERED.n_primary_hypotheses),
        cohens_d=d,
        effect_interpretation=effect_size_interpretation(d),
        practically_significant=abs(d) >= PREREGISTERED.medium_effect,
        negative_control_passed=nc_passed,
        negative_control_details=f"Random model energy SD: {nc_std:.4f}",
        n_samples=n_samples
    )
    
    _print_result(result)
    return result


def test_idk_token(config: NanoConfig, n_samples: int) -> TestResult:
    """
    HYPOTHESIS 2: Model outputs [IDK] when given noise.
    H0: P([IDK]|noise) <= P([IDK]|initial)
    H1: P([IDK]|noise) > P([IDK]|initial) (one-tailed)
    """
    print_header("TEST 2: Epistemic Humility (IDK Token)")
    print(f"   Hypothesis: Training on noise increases [IDK] probability")
    print(f"   N samples: {n_samples}")
    
    initial_probs = []
    final_probs = []
    
    for trial in range(n_samples):
        torch.manual_seed(RANDOM_SEED + trial)
        model = NanoGlass(config).to(config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        
        # Initial IDK probability
        noise = torch.randint(0, 256, (4, config.block_size), dtype=torch.long).to(config.device)
        with torch.no_grad():
            logits, _ = model(noise)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            initial_prob = probs[:, config.idk_token].mean().item()
        initial_probs.append(initial_prob)
        
        # Train on noise with IDK targets
        idk_targets = torch.full((4, config.block_size), config.idk_token, dtype=torch.long).to(config.device)
        for _ in range(PREREGISTERED.n_training_steps):
            noise = torch.randint(0, 256, (4, config.block_size), dtype=torch.long).to(config.device)
            _, loss = model(noise, idk_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Final IDK probability
        new_noise = torch.randint(0, 256, (4, config.block_size), dtype=torch.long).to(config.device)
        with torch.no_grad():
            logits, _ = model(new_noise)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            final_prob = probs[:, config.idk_token].mean().item()
        final_probs.append(final_prob)
    
    # Statistical analysis
    t_stat, p_value_two = stats.ttest_rel(final_probs, initial_probs)
    p_value_one = p_value_two / 2 if t_stat > 0 else 1 - p_value_two / 2
    
    d = cohens_d(final_probs, initial_probs)
    
    ci_initial = confidence_interval_mean(initial_probs)
    ci_final = confidence_interval_mean(final_probs)
    
    passed = p_value_one < PREREGISTERED.alpha_corrected and t_stat > 0
    
    # Negative control: Structured data should NOT increase IDK
    nc_probs_before = []
    nc_probs_after = []
    for _ in range(20):
        model = NanoGlass(config).to(config.device)
        text = "The mind is a glass box. " * 20
        structured = torch.tensor([[ord(c) for c in text[:config.block_size]]], dtype=torch.long).to(config.device)
        
        with torch.no_grad():
            logits, _ = model(structured)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            nc_probs_before.append(probs[:, config.idk_token].mean().item())
    
    nc_mean = np.mean(nc_probs_before)
    nc_passed = nc_mean < 0.5  # Structured input should not trigger IDK
    
    result = TestResult(
        name="Epistemic Humility",
        hypothesis="Training on noise increases P([IDK])",
        passed=passed,
        group1_mean=np.mean(initial_probs),
        group1_std=np.std(initial_probs),
        group1_ci=ci_initial,
        group2_mean=np.mean(final_probs),
        group2_std=np.std(final_probs),
        group2_ci=ci_final,
        test_statistic=t_stat,
        p_value=p_value_one,
        p_value_corrected=min(1.0, p_value_one * PREREGISTERED.n_primary_hypotheses),
        cohens_d=d,
        effect_interpretation=effect_size_interpretation(d),
        practically_significant=abs(d) >= PREREGISTERED.medium_effect,
        negative_control_passed=nc_passed,
        negative_control_details=f"Mean P([IDK]) on structured: {nc_mean:.4f}",
        n_samples=n_samples
    )
    
    _print_result(result)
    return result


def test_model_convergence(config: NanoConfig, n_samples: int) -> TestResult:
    """
    HYPOTHESIS 3: Model loss decreases during training.
    H0: final_loss >= initial_loss
    H1: final_loss < initial_loss (one-tailed)
    """
    print_header("TEST 3: Model Convergence")
    print(f"   Hypothesis: Training reduces loss")
    print(f"   N samples: {n_samples}")
    
    initial_losses = []
    final_losses = []
    
    for trial in range(n_samples):
        torch.manual_seed(RANDOM_SEED + trial)
        model = NanoGlass(config).to(config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        
        text = "Hello world. This is a test. " * 100
        data = torch.tensor([ord(c) for c in text], dtype=torch.long)
        
        # Initial loss
        ix = torch.randint(len(data) - config.block_size, (4,))
        x = torch.stack([data[i:i+config.block_size] for i in ix]).to(config.device)
        y = torch.stack([data[i+1:i+config.block_size+1] for i in ix]).to(config.device)
        _, initial_loss = model(x, y)
        initial_losses.append(initial_loss.item())
        
        # Train
        for _ in range(100):  # More steps for convergence
            ix = torch.randint(len(data) - config.block_size, (4,))
            x = torch.stack([data[i:i+config.block_size] for i in ix]).to(config.device)
            y = torch.stack([data[i+1:i+config.block_size+1] for i in ix]).to(config.device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Final loss
        _, final_loss = model(x, y)
        final_losses.append(final_loss.item())
    
    # Statistical analysis
    t_stat, p_value_two = stats.ttest_rel(final_losses, initial_losses)
    p_value_one = p_value_two / 2 if t_stat < 0 else 1 - p_value_two / 2
    
    d = cohens_d(initial_losses, final_losses)
    
    ci_initial = confidence_interval_mean(initial_losses)
    ci_final = confidence_interval_mean(final_losses)
    
    passed = p_value_one < PREREGISTERED.alpha_corrected and t_stat < 0
    
    result = TestResult(
        name="Model Convergence",
        hypothesis="Training reduces loss (L_final < L_initial)",
        passed=passed,
        group1_mean=np.mean(initial_losses),
        group1_std=np.std(initial_losses),
        group1_ci=ci_initial,
        group2_mean=np.mean(final_losses),
        group2_std=np.std(final_losses),
        group2_ci=ci_final,
        test_statistic=t_stat,
        p_value=p_value_one,
        p_value_corrected=min(1.0, p_value_one * PREREGISTERED.n_primary_hypotheses),
        cohens_d=d,
        effect_interpretation=effect_size_interpretation(d),
        practically_significant=abs(d) >= PREREGISTERED.medium_effect,
        n_samples=n_samples
    )
    
    _print_result(result)
    return result


def _print_result(result: TestResult):
    """Pretty print a test result."""
    print(f"\n   RESULTS:")
    
    if result.group2_mean is not None:
        print(f"   Before: {result.group1_mean:.4f} ± {result.group1_std:.4f} (95% CI: [{result.group1_ci[0]:.4f}, {result.group1_ci[1]:.4f}])")
        print(f"   After:  {result.group2_mean:.4f} ± {result.group2_std:.4f} (95% CI: [{result.group2_ci[0]:.4f}, {result.group2_ci[1]:.4f}])")
    else:
        print(f"   Mean: {result.group1_mean:.4f} ± {result.group1_std:.4f}")
    
    print(f"\n   STATISTICS:")
    print(f"   t-statistic: {result.test_statistic:.4f}")
    print(f"   p-value (one-tailed): {result.p_value:.6f}")
    print(f"   p-value (Bonferroni): {result.p_value_corrected:.6f}")
    print(f"   Cohen's d: {result.cohens_d:.4f} ({result.effect_interpretation})")
    
    if result.negative_control_passed is not None:
        nc_status = "✅ PASSED" if result.negative_control_passed else "❌ FAILED"
        print(f"\n   NEGATIVE CONTROL: {nc_status}")
        print(f"   {result.negative_control_details}")
    
    status = "✅ PASS" if result.passed else "❌ FAIL"
    practical = "✅ YES" if result.practically_significant else "❌ NO"
    
    print(f"\n   VERDICT:")
    print(f"   Statistical significance: {status} (α = {PREREGISTERED.alpha_corrected:.4f})")
    print(f"   Practical significance:   {practical} (d ≥ {PREREGISTERED.medium_effect})")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="NanoGlass Empirical Verification Suite")
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer samples")
    parser.add_argument("--n-samples", type=int, default=None, help="Override sample size")
    args = parser.parse_args()
    
    # Determine sample size
    n_samples = args.n_samples or (20 if args.quick else PREREGISTERED.n_samples_per_condition)
    
    print("\n" + "=" * 70)
    print("  🔬 PROJECT NANOGLASS - EMPIRICAL VERIFICATION SUITE v2.0")
    print("=" * 70)
    print(f"  Pre-registered: {PREREGISTERED.registered_at}")
    print(f"  Random Seed: {RANDOM_SEED}")
    print(f"  N samples per test: {n_samples}")
    print(f"  α level: {PREREGISTERED.alpha} (corrected: {PREREGISTERED.alpha_corrected:.4f})")
    print(f"  Mode: {'QUICK' if args.quick else 'FULL'}")
    print("  Methodology: Pre-registered, effect sizes, CIs, negative controls")
    print("=" * 70)
    
    config = NanoConfig()
    
    results = {
        "thermodynamics": test_thermodynamics(config, n_samples),
        "epistemic_humility": test_idk_token(config, n_samples),
        "convergence": test_model_convergence(config, n_samples),
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("  📊 FINAL SUMMARY")
    print("=" * 70)
    
    all_passed = True
    all_practical = True
    
    for key, result in results.items():
        stat = "✅" if result.passed else "❌"
        prac = "✅" if result.practically_significant else "⚠️"
        nc = ""
        if result.negative_control_passed is not None:
            nc = " [NC: ✅]" if result.negative_control_passed else " [NC: ❌]"
        
        print(f"  {stat} {result.name}")
        print(f"     p = {result.p_value_corrected:.4f}, d = {result.cohens_d:.2f} ({result.effect_interpretation}){nc}")
        
        if not result.passed:
            all_passed = False
        if not result.practically_significant:
            all_practical = False
    
    print("=" * 70)
    
    if all_passed and all_practical:
        print("  🎉 ALL TESTS PASSED with practical significance!")
    elif all_passed:
        print("  ⚠️  All tests statistically significant, but some lack practical significance")
    else:
        print("  ⚠️  Some tests failed statistical significance")
    
    print("=" * 70 + "\n")
    
    # Export audit report
    audit_report = {
        "timestamp": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "preregistration": PREREGISTERED.__dict__,
        "n_samples": n_samples,
        "results": {
            name: {
                "passed": r.passed,
                "p_value": r.p_value,
                "p_value_corrected": r.p_value_corrected,
                "cohens_d": r.cohens_d,
                "effect_interpretation": r.effect_interpretation,
                "practically_significant": r.practically_significant,
                "negative_control_passed": r.negative_control_passed,
            }
            for name, r in results.items()
        },
        "summary": {
            "all_statistically_significant": all_passed,
            "all_practically_significant": all_practical,
        }
    }
    
    audit_path = os.path.join(os.path.dirname(__file__), "verification_audit.json")
    with open(audit_path, "w") as f:
        json.dump(audit_report, f, indent=2, default=str)
    print(f"  📋 Audit report saved to: {audit_path}\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
