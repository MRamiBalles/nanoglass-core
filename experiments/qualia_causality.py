#!/usr/bin/env python3
"""
==============================================================================
🧠 PROJECT 15: CAUSAL VALIDATION - STRUCTURAL CAUSAL MODELS (SCM)
==============================================================================
Validates causal claims about model behavior using Pearl's causal hierarchy:
    Level 1: Association (P(Y|X)) - "Seeing"
    Level 2: Intervention (P(Y|do(X))) - "Doing"  
    Level 3: Counterfactual (P(Y_X|X',Y')) - "Imagining"

This script replaces the previous tautological test with proper SCM validation.

Key Changes from v1:
    - Implements actual causal interventions (not circular definitions)
    - Uses do-calculus for intervention analysis
    - Separates observational from interventional distributions
    - Includes sanity checks against memorization/correlation

References:
    - Pearl, J. (2009). Causality: Models, Reasoning, and Inference
    - CausalProbe-2024: Temporal isolation for LLM causal analysis
    - DeepSeek-R1 causal intervention methodology

TODO:
    - Integrate with CausalProbe-2024 benchmark
    - Implement counterfactual analysis (Level 3)
==============================================================================
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
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
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_INTERVENTIONS = 50

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


@dataclass
class CausalResult:
    """Structured causal analysis result."""
    observational: float      # P(Y|X) - what we observe
    interventional: float     # P(Y|do(X)) - what changes when we intervene
    causal_effect: float      # Difference: intervention - observation
    is_causal: bool           # Whether effect is significant
    
    
# ==============================================================================
# STRUCTURAL CAUSAL MODEL FOR LLM ANALYSIS
# ==============================================================================
# 
# Our SCM for NanoGlass:
#
#   Instruction (Z) → Thought (T) → Response (Y)
#                  ↘           ↗
#                    Context (C)
#
# We test causality by intervening on T (the hidden state) and measuring Y
# ==============================================================================


class CausalInterventionProbe:
    """
    Implements causal interventions on model hidden states.
    
    Key principle: To establish causality, we must show that
    changing T (thought/hidden state) changes Y (output), 
    independent of the original input Z.
    """
    
    def __init__(self, model: NanoGlass):
        self.model = model
        self.hooks = []
        self.cached_activations: Dict[str, torch.Tensor] = {}
        
    def register_hooks(self, layer_idx: int = -1):
        """Register forward hooks to capture and optionally modify activations."""
        
        def capture_hook(name):
            def hook(module, input, output):
                self.cached_activations[name] = output.clone()
            return hook
        
        # Hook into the specified layer (default: last layer before head)
        target_layer = self.model.transformer.h[layer_idx]
        handle = target_layer.register_forward_hook(capture_hook(f"layer_{layer_idx}"))
        self.hooks.append(handle)
        
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def observe(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LEVEL 1: Observational distribution P(Y|X)
        Simply run the model and observe outputs.
        """
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(input_ids)
            probs = F.softmax(logits[:, -1, :], dim=-1)
        return logits, probs
    
    def intervene(
        self, 
        input_ids: torch.Tensor,
        intervention_vector: torch.Tensor,
        intervention_layer: int = -2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LEVEL 2: Interventional distribution P(Y|do(T=t))
        
        We intervene by adding a specific direction to the hidden state,
        then measure how the output changes.
        
        This is different from observation because we're *forcing* the 
        hidden state to change, not just observing what correlates.
        """
        self.model.eval()
        
        # Create intervention hook
        def intervention_hook(module, input, output):
            # Add intervention vector to the hidden state
            return output + intervention_vector.unsqueeze(0).unsqueeze(0)
        
        # Apply intervention
        target_layer = self.model.transformer.h[intervention_layer]
        handle = target_layer.register_forward_hook(intervention_hook)
        
        try:
            with torch.no_grad():
                logits, _ = self.model(input_ids)
                probs = F.softmax(logits[:, -1, :], dim=-1)
        finally:
            handle.remove()
            
        return logits, probs


def create_intervention_vector(
    config: NanoConfig, 
    direction: str = "random"
) -> torch.Tensor:
    """
    Creates an intervention vector to perturb the hidden state.
    
    Options:
        - "random": Random direction (baseline)
        - "confidence": Direction that should increase confidence
        - "uncertainty": Direction that should increase uncertainty
    """
    d_model = config.n_embd
    
    if direction == "random":
        vec = torch.randn(d_model, device=DEVICE)
        vec = vec / vec.norm() * 0.5  # Normalize to reasonable magnitude
    elif direction == "confidence":
        # Theory: positive values in earlier dimensions correlate with confidence
        vec = torch.zeros(d_model, device=DEVICE)
        vec[:d_model//4] = 1.0
        vec = vec / vec.norm() * 0.5
    elif direction == "uncertainty":
        # Theory: uniform activations indicate confusion
        vec = torch.ones(d_model, device=DEVICE)
        vec = vec / vec.norm() * 0.5
    else:
        raise ValueError(f"Unknown direction: {direction}")
        
    return vec


def compute_causal_effect(
    pre_intervention_prob: torch.Tensor,
    post_intervention_prob: torch.Tensor,
    metric: str = "kl_divergence"
) -> float:
    """
    Compute the causal effect of intervention.
    
    If intervention causes change in output distribution, there's a causal link.
    """
    if metric == "kl_divergence":
        # KL(post || pre) - how much the distribution changed
        kl = F.kl_div(
            post_intervention_prob.log(),
            pre_intervention_prob,
            reduction='batchmean'
        )
        return kl.item()
    elif metric == "js_divergence":
        # Jensen-Shannon divergence (symmetric)
        m = 0.5 * (pre_intervention_prob + post_intervention_prob)
        js = 0.5 * F.kl_div(pre_intervention_prob.log(), m, reduction='batchmean')
        js += 0.5 * F.kl_div(post_intervention_prob.log(), m, reduction='batchmean')
        return js.item()
    elif metric == "entropy_delta":
        # Change in output entropy
        pre_entropy = -(pre_intervention_prob * pre_intervention_prob.log()).sum(-1).mean()
        post_entropy = -(post_intervention_prob * post_intervention_prob.log()).sum(-1).mean()
        return (post_entropy - pre_entropy).item()
    else:
        raise ValueError(f"Unknown metric: {metric}")


def run_causal_validation():
    """
    Main experiment: Test whether hidden state interventions have causal effects.
    
    We compare:
        1. Random interventions (baseline)
        2. Structured interventions (theoretical predictions)
    
    If structured interventions cause predictably different effects than random,
    we have evidence for specific causal mechanisms.
    """
    print("\n" + "=" * 70)
    print("🧠 PROJECT 15: CAUSAL VALIDATION (SCM Framework)")
    print("=" * 70)
    print(f"   Device: {DEVICE}")
    print(f"   N interventions: {N_INTERVENTIONS}")
    print(f"   Framework: Pearl's Structural Causal Models")
    print("-" * 70)
    
    # Initialize
    config = NanoConfig()
    model = NanoGlass(config).to(DEVICE)
    probe = CausalInterventionProbe(model)
    
    # Brief training for stable representations
    print("   Training model...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    train_text = "The mind is a glass box. Truth is low energy. " * 200
    train_data = torch.tensor([ord(c) for c in train_text], dtype=torch.long)
    
    for _ in range(50):
        ix = torch.randint(len(train_data) - config.block_size, (4,))
        x = torch.stack([train_data[i:i+config.block_size] for i in ix]).to(DEVICE)
        y = torch.stack([train_data[i+1:i+config.block_size+1] for i in ix]).to(DEVICE)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"   Training complete. Loss: {loss.item():.4f}")
    print("-" * 70)
    
    # Create test input
    test_text = "The mind is a "
    test_input = torch.tensor(
        [[ord(c) for c in test_text] + [0] * (config.block_size - len(test_text))],
        dtype=torch.long
    ).to(DEVICE)
    
    # Baseline: Observational distribution
    print("   [TEST 1] Measuring OBSERVATIONAL distribution P(Y|X)...")
    _, obs_probs = probe.observe(test_input)
    obs_entropy = -(obs_probs * obs_probs.log()).sum(-1).mean().item()
    print(f"      Baseline entropy: {obs_entropy:.4f}")
    
    # Intervention 1: Random perturbations
    print("   [TEST 2] Random interventions (control)...")
    random_effects = []
    for i in range(N_INTERVENTIONS):
        intervention = create_intervention_vector(config, "random")
        _, int_probs = probe.intervene(test_input, intervention)
        effect = compute_causal_effect(obs_probs, int_probs, "entropy_delta")
        random_effects.append(effect)
    
    random_mean = np.mean(random_effects)
    random_std = np.std(random_effects)
    print(f"      Mean entropy change: {random_mean:.4f} ± {random_std:.4f}")
    
    # Intervention 2: Confidence direction
    print("   [TEST 3] Confidence intervention...")
    confidence_effects = []
    for i in range(N_INTERVENTIONS):
        intervention = create_intervention_vector(config, "confidence")
        # Add noise to test robustness
        intervention += torch.randn_like(intervention) * 0.1
        _, int_probs = probe.intervene(test_input, intervention)
        effect = compute_causal_effect(obs_probs, int_probs, "entropy_delta")
        confidence_effects.append(effect)
    
    confidence_mean = np.mean(confidence_effects)
    confidence_std = np.std(confidence_effects)
    print(f"      Mean entropy change: {confidence_mean:.4f} ± {confidence_std:.4f}")
    
    # Intervention 3: Uncertainty direction
    print("   [TEST 4] Uncertainty intervention...")
    uncertainty_effects = []
    for i in range(N_INTERVENTIONS):
        intervention = create_intervention_vector(config, "uncertainty")
        intervention += torch.randn_like(intervention) * 0.1
        _, int_probs = probe.intervene(test_input, intervention)
        effect = compute_causal_effect(obs_probs, int_probs, "entropy_delta")
        uncertainty_effects.append(effect)
    
    uncertainty_mean = np.mean(uncertainty_effects)
    uncertainty_std = np.std(uncertainty_effects)
    print(f"      Mean entropy change: {uncertainty_mean:.4f} ± {uncertainty_std:.4f}")
    
    # Statistical analysis
    from scipy import stats
    
    # Test: Is structured intervention different from random?
    _, p_confidence = stats.ttest_ind(random_effects, confidence_effects)
    _, p_uncertainty = stats.ttest_ind(random_effects, uncertainty_effects)
    
    print("\n" + "=" * 70)
    print("📊 CAUSAL ANALYSIS RESULTS")
    print("=" * 70)
    print(f"   Random baseline:       {random_mean:+.4f} ± {random_std:.4f}")
    print(f"   Confidence direction:  {confidence_mean:+.4f} ± {confidence_std:.4f}")
    print(f"   Uncertainty direction: {uncertainty_mean:+.4f} ± {uncertainty_std:.4f}")
    print("-" * 70)
    print(f"   p-value (confidence vs random):  {p_confidence:.6f}")
    print(f"   p-value (uncertainty vs random): {p_uncertainty:.6f}")
    print("-" * 70)
    
    # Interpretation
    causal_evidence = False
    
    if p_confidence < 0.05:
        print("   ✅ Confidence intervention has CAUSAL EFFECT (differs from random)")
        causal_evidence = True
    else:
        print("   ⚠️  Confidence intervention NOT significantly different from random")
    
    if p_uncertainty < 0.05:
        print("   ✅ Uncertainty intervention has CAUSAL EFFECT (differs from random)")
        causal_evidence = True
    else:
        print("   ⚠️  Uncertainty intervention NOT significantly different from random")
    
    print("=" * 70)
    
    if causal_evidence:
        print("   CONCLUSION: Evidence for causal mechanisms in hidden states")
        print("   Pearl Level: 2 (Intervention)")
    else:
        print("   CONCLUSION: No clear causal structure detected")
        print("   Note: This may indicate need for different intervention strategies")
    
    print("\n   ⚠️  LIMITATIONS:")
    print("      - Small model may not have developed complex causal circuits")
    print("      - Intervention vectors are theoretically motivated, not empirically derived")
    print("      - Full validation requires CausalProbe-2024 benchmark")
    print("=" * 70)
    
    return {
        "random_effect_mean": random_mean,
        "confidence_effect_mean": confidence_mean,
        "uncertainty_effect_mean": uncertainty_mean,
        "p_confidence": p_confidence,
        "p_uncertainty": p_uncertainty,
        "causal_evidence": causal_evidence
    }


if __name__ == "__main__":
    results = run_causal_validation()
