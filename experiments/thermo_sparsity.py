#!/usr/bin/env python3
"""
==============================================================================
🔥 PROJECT 9: THERMODYNAMICS OF MEANING - EMPIRICAL VALIDATION
==============================================================================
Measures L1 sparsity of real model activations to test the hypothesis:
    E(truth) < E(confusion)
    
This script replaces the previous simulation with actual model measurements.

Key Changes from v1:
    - Uses real NanoGlass model activations (not synthetic vectors)
    - Statistical rigor: N >= 100 samples, t-test with p-value
    - Modular design for future PINN integration

TODO (Future Work):
    - Integrate NIST-JANAF thermochemical database for physical validation
    - Implement Physics-Informed Neural Network (PINN) loss functions
    - Connect activation "energy" to Gibbs free energy formalism
==============================================================================
"""
import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
from typing import Tuple, Dict, List
from dataclasses import dataclass
import sys
import os

# Ensure nanoglass is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanoglass import NanoConfig, NanoGlass

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RANDOM_SEED = 42
N_SAMPLES = 100  # Minimum for statistical significance
ALPHA = 0.05     # Significance level
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


@dataclass
class EnergyMeasurement:
    """Structured result from energy measurement."""
    mean: float
    std: float
    samples: List[float]


def create_factual_batch(batch_size: int, seq_len: int) -> torch.Tensor:
    """
    Creates a batch of 'factual' (structured) data.
    Represents coherent, predictable patterns the model can learn.
    """
    # Structured patterns: repeating sequences that the model can predict
    patterns = [
        "The quick brown fox jumps over the lazy dog. ",
        "Paris is the capital of France. ",
        "Water freezes at zero degrees Celsius. ",
        "The sun rises in the east and sets in the west. ",
    ]
    
    samples = []
    for i in range(batch_size):
        pattern = patterns[i % len(patterns)]
        text = (pattern * ((seq_len // len(pattern)) + 1))[:seq_len]
        tensor = torch.tensor([ord(c) for c in text], dtype=torch.long)
        samples.append(tensor)
    
    return torch.stack(samples)


def create_noise_batch(batch_size: int, seq_len: int) -> torch.Tensor:
    """
    Creates a batch of 'noise' (unstructured) data.
    Represents confusion/entropy that should trigger hallucination-like states.
    """
    return torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long)


def measure_activation_energy(
    model: NanoGlass, 
    input_batch: torch.Tensor
) -> EnergyMeasurement:
    """
    Measures the 'energy' (L1 norm of activations) for a batch.
    
    Theory:
        - Lower L1 = sparser, more efficient representation = "truth"
        - Higher L1 = denser, interfering activations = "confusion"
    
    Returns:
        EnergyMeasurement with mean, std, and per-sample values
    """
    model.eval()
    energies = []
    
    with torch.no_grad():
        input_batch = input_batch.to(DEVICE)
        logits, _ = model(input_batch)
        
        # Access the sensor's energy history from the last forward pass
        # Each sample in the batch contributes to the energy
        for i in range(input_batch.size(0)):
            # We measure after the final layer norm (most refined representation)
            # Using the logits as proxy for final activation state
            sample_logits = logits[i]
            l1_energy = sample_logits.abs().mean().item()
            energies.append(l1_energy)
    
    return EnergyMeasurement(
        mean=np.mean(energies),
        std=np.std(energies),
        samples=energies
    )


def run_thermodynamics_experiment() -> Dict:
    """
    Main experiment: Compare activation energy for factual vs noise inputs.
    
    Hypothesis:
        H0: E(truth) >= E(confusion)  [Null: No difference or truth is higher]
        H1: E(truth) < E(confusion)   [Alternative: Truth is lower energy]
    """
    print("\n" + "=" * 70)
    print("🔥 PROJECT 9: THERMODYNAMICS OF MEANING (Empirical Validation)")
    print("=" * 70)
    print(f"   Device: {DEVICE}")
    print(f"   N samples per condition: {N_SAMPLES}")
    print(f"   Significance level (α): {ALPHA}")
    print(f"   Random seed: {RANDOM_SEED}")
    print("-" * 70)
    
    # Initialize model
    config = NanoConfig()
    model = NanoGlass(config).to(DEVICE)
    
    # Brief training to establish representations
    print("   Training model briefly to establish representations...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Training data: factual patterns
    train_text = "The quick brown fox jumps over the lazy dog. " * 500
    train_data = torch.tensor([ord(c) for c in train_text], dtype=torch.long)
    
    for step in range(50):
        ix = torch.randint(len(train_data) - config.block_size, (8,))
        x = torch.stack([train_data[i:i+config.block_size] for i in ix]).to(DEVICE)
        y = torch.stack([train_data[i+1:i+config.block_size+1] for i in ix]).to(DEVICE)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"   Training complete. Final loss: {loss.item():.4f}")
    print("-" * 70)
    
    # Measure energy on factual data
    print("   Measuring energy on FACTUAL data...")
    factual_batch = create_factual_batch(N_SAMPLES, config.block_size)
    energy_truth = measure_activation_energy(model, factual_batch)
    
    # Measure energy on noise data
    print("   Measuring energy on NOISE data...")
    noise_batch = create_noise_batch(N_SAMPLES, config.block_size)
    energy_confusion = measure_activation_energy(model, noise_batch)
    
    # Statistical test (one-tailed t-test: truth < confusion)
    t_stat, p_value_two_tailed = stats.ttest_ind(
        energy_truth.samples, 
        energy_confusion.samples,
        equal_var=False  # Welch's t-test (more robust)
    )
    
    # Convert to one-tailed (we expect truth < confusion, so t_stat should be negative)
    p_value_one_tailed = p_value_two_tailed / 2 if t_stat < 0 else 1 - (p_value_two_tailed / 2)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((energy_truth.std**2 + energy_confusion.std**2) / 2)
    cohens_d = (energy_confusion.mean - energy_truth.mean) / pooled_std if pooled_std > 0 else 0
    
    # Results
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print(f"   Energy of TRUTH (Factual):    {energy_truth.mean:.4f} ± {energy_truth.std:.4f}")
    print(f"   Energy of CONFUSION (Noise):  {energy_confusion.mean:.4f} ± {energy_confusion.std:.4f}")
    print("-" * 70)
    print(f"   Δ Energy = {energy_confusion.mean - energy_truth.mean:.4f}")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value (one-tailed): {p_value_one_tailed:.6f}")
    print(f"   Cohen's d: {cohens_d:.4f}")
    print("-" * 70)
    
    # Interpretation
    if p_value_one_tailed < ALPHA and t_stat < 0:
        print("   ✅ HYPOTHESIS SUPPORTED: E(truth) < E(confusion)")
        print(f"      Statistically significant at α = {ALPHA}")
        if cohens_d > 0.8:
            print("      Effect size: LARGE")
        elif cohens_d > 0.5:
            print("      Effect size: MEDIUM")
        else:
            print("      Effect size: SMALL")
        hypothesis_supported = True
    else:
        print("   ❌ HYPOTHESIS NOT SUPPORTED")
        print(f"      p-value ({p_value_one_tailed:.4f}) >= α ({ALPHA})")
        hypothesis_supported = False
    
    print("=" * 70)
    
    # Return structured results
    return {
        "energy_truth_mean": energy_truth.mean,
        "energy_truth_std": energy_truth.std,
        "energy_confusion_mean": energy_confusion.mean,
        "energy_confusion_std": energy_confusion.std,
        "delta_energy": energy_confusion.mean - energy_truth.mean,
        "t_statistic": t_stat,
        "p_value_one_tailed": p_value_one_tailed,
        "cohens_d": cohens_d,
        "hypothesis_supported": hypothesis_supported,
        "n_samples": N_SAMPLES,
        "alpha": ALPHA
    }


if __name__ == "__main__":
    results = run_thermodynamics_experiment()
    
    # Save results to JSON
    import json
    output_path = os.path.join(os.path.dirname(__file__), "thermo_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n   📋 Results saved to: {output_path}")
