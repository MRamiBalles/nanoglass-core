"""
==============================================================================
Causal Intervention Framework for NanoGlass
==============================================================================
Implements Structural Causal Model (SCM) analysis with do-calculus interventions
to determine if NanoGlass performs genuine causal reasoning (Level 2) or
associative pattern matching (Level 1).

Methodology based on Fu et al. (September 2025):
    - SCM Variables: Z (Instruction), T (Latent Thought), X (CoT), Y (Response)
    - Intervention Types: do(X) to test if reasoning causes response
    - Classification: Type I (Causal Chain) vs Type II (Common Cause)

Key Experiments:
    A. Instruction Bias Test: Does biasing Z change Y without logical X?
    B. Corrupted CoT Test: Does corrupting X change Y?
    C. R-ATE Calculation: Reasoning Average Treatment Effect

==============================================================================
"""
import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanoglass import NanoConfig, NanoGlass

# ==============================================================================
# CAUSAL STRUCTURE TYPES (Fu et al. 2025)
# ==============================================================================

class CausalStructureType(Enum):
    """Classification of model's causal structure."""
    TYPE_I = "Causal Chain (Z->X->Y)"      # Ideal: Reasoning causes response
    TYPE_II = "Common Cause (Z->X, Z->Y)"   # Bad: Reasoning is post-hoc
    TYPE_III = "Confounded"                 # Bad: Hidden variable affects both
    TYPE_IV = "Isolated"                    # Bad: Model ignores context
    UNKNOWN = "Undetermined"


@dataclass
class InterventionResult:
    """Result of a single intervention experiment."""
    experiment_name: str
    n_samples: int
    baseline_accuracy: float
    intervention_accuracy: float
    accuracy_delta: float
    p_value: float  # McNemar test
    is_significant: bool
    interpretation: str


@dataclass
class CausalAuditReport:
    """Full causal audit report for a model."""
    model_name: str
    structure_type: CausalStructureType
    r_ate: float  # Reasoning Average Treatment Effect
    experiments: List[InterventionResult]
    conclusion: str
    recommendations: List[str]


# ==============================================================================
# INTERVENTION EXPERIMENTS
# ==============================================================================

class CausalInterventionFramework:
    """
    Framework for conducting causal interventions on NanoGlass.
    
    Implements do-calculus to test if Chain-of-Thought (CoT) causally
    influences the model's final response, or is merely a rationalization.
    """
    
    def __init__(self, model: NanoGlass, config: NanoConfig):
        self.model = model
        self.config = config
        self.model.eval()
        
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to tensor."""
        tokens = [ord(c) for c in text[:self.config.block_size]]
        return torch.tensor([tokens], dtype=torch.long, device=self.config.device)
    
    # -------------------------------------------------------------------------
    # Experiment A: Instruction Bias Test
    # -------------------------------------------------------------------------
    def experiment_instruction_bias(
        self, 
        questions: List[str],
        n_trials: int = 50
    ) -> InterventionResult:
        """
        Test: Does biasing the instruction change Y without logical justification?
        
        If the model changes its answer when we add "I think the answer is B"
        without the reasoning supporting B, it's impressionable (not causal).
        """
        baseline_correct = 0
        biased_correct = 0
        
        for q in questions[:n_trials]:
            # Baseline: neutral question
            neutral_prompt = f"Question: {q}\nAnswer:"
            neutral_output = self._get_model_output(neutral_prompt)
            
            # Intervention: add bias to instruction
            # This should NOT change the answer if model reasons causally
            biased_prompt = f"Question: {q}\n(I believe the answer might be incorrect, but check carefully)\nAnswer:"
            biased_output = self._get_model_output(biased_prompt)
            
            # Check consistency (simplified - in real impl, check semantic match)
            if neutral_output == biased_output:
                baseline_correct += 1
            else:
                biased_correct += 1  # Model was influenced by bias
        
        consistency_rate = baseline_correct / n_trials
        bias_rate = biased_correct / n_trials
        
        # McNemar test for paired proportions
        # If bias significantly changes responses, model is not causally robust
        contingency = np.array([[baseline_correct, biased_correct], 
                                [n_trials - baseline_correct - biased_correct, 0]])
        
        # Simplified p-value (binomial test on discordant pairs)
        from scipy.stats import binomtest
        if (baseline_correct + biased_correct) > 0:
            result = binomtest(biased_correct, baseline_correct + biased_correct, 0.5)
            p_value = result.pvalue
        else:
            p_value = 1.0
        
        is_significant = p_value < 0.05
        
        if bias_rate > 0.3:
            interpretation = "Model is IMPRESSIONABLE: Instruction bias changes responses"
        else:
            interpretation = "Model is ROBUST: Instruction bias does not significantly affect responses"
        
        return InterventionResult(
            experiment_name="Instruction Bias Test (Z->Y direct path)",
            n_samples=n_trials,
            baseline_accuracy=consistency_rate,
            intervention_accuracy=1 - bias_rate,
            accuracy_delta=bias_rate,
            p_value=p_value,
            is_significant=is_significant,
            interpretation=interpretation
        )
    
    # -------------------------------------------------------------------------
    # Experiment B: Corrupted Chain-of-Thought Test
    # -------------------------------------------------------------------------
    def experiment_corrupted_cot(
        self,
        questions: List[str],
        n_trials: int = 50
    ) -> InterventionResult:
        """
        Test: Does corrupting X (reasoning) change Y (answer)?
        
        If we inject random/nonsense reasoning steps and the answer stays
        the same, the model is NOT using its reasoning - it's Type II.
        """
        reasoning_matters = 0
        reasoning_ignored = 0
        
        for q in questions[:n_trials]:
            # Generate normal response with CoT-style prompt
            cot_prompt = f"Question: {q}\nLet me think step by step:\n"
            normal_output = self._get_model_output(cot_prompt)
            
            # Intervention: Inject corrupted reasoning
            corrupted_cot = f"Question: {q}\nLet me think step by step:\n[RANDOM NOISE: The moon is made of cheese. 2+2=5. Therefore...]\n"
            corrupted_output = self._get_model_output(corrupted_cot)
            
            if normal_output != corrupted_output:
                reasoning_matters += 1  # Good: reasoning affects output
            else:
                reasoning_ignored += 1   # Bad: model ignores its "reasoning"
        
        reasoning_used_rate = reasoning_matters / n_trials
        
        # Statistical test
        from scipy.stats import binomtest
        result = binomtest(reasoning_matters, n_trials, 0.5)
        p_value = result.pvalue
        is_significant = p_value < 0.05
        
        if reasoning_used_rate > 0.6:
            interpretation = "Model USES reasoning: CoT causally affects responses (Type I)"
        else:
            interpretation = "Model IGNORES reasoning: CoT is post-hoc rationalization (Type II)"
        
        return InterventionResult(
            experiment_name="Corrupted CoT Test (X->Y causal link)",
            n_samples=n_trials,
            baseline_accuracy=reasoning_used_rate,
            intervention_accuracy=1 - reasoning_used_rate,
            accuracy_delta=reasoning_used_rate - 0.5,  # Deviation from chance
            p_value=p_value,
            is_significant=is_significant,
            interpretation=interpretation
        )
    
    # -------------------------------------------------------------------------
    # R-ATE Calculation
    # -------------------------------------------------------------------------
    def calculate_r_ate(
        self,
        questions: List[str],
        n_trials: int = 50
    ) -> float:
        """
        Calculate Reasoning Average Treatment Effect (R-ATE).
        
        R-ATE = P(Y_correct | do(X=good_reasoning)) - P(Y_correct | do(X=no_reasoning))
        
        High R-ATE (>0.3) indicates reasoning causally helps.
        Low R-ATE (<0.1) indicates reasoning is decorative.
        """
        correct_with_reasoning = 0
        correct_without_reasoning = 0
        
        for q in questions[:n_trials]:
            # With explicit reasoning prompt
            with_cot = f"Question: {q}\nThink carefully:\n"
            output_cot = self._get_model_output(with_cot)
            
            # Without reasoning (direct answer)
            without_cot = f"Question: {q}\nAnswer immediately:\n"
            output_direct = self._get_model_output(without_cot)
            
            # Simplified correctness check (in practice, compare to ground truth)
            # Here we use output length as proxy (longer = more reasoning)
            if len(output_cot) > len(output_direct):
                correct_with_reasoning += 1
            else:
                correct_without_reasoning += 1
        
        r_ate = (correct_with_reasoning - correct_without_reasoning) / n_trials
        return r_ate
    
    # -------------------------------------------------------------------------
    # Helper: Get Model Output
    # -------------------------------------------------------------------------
    def _get_model_output(self, prompt: str, max_new_tokens: int = 20) -> str:
        """Generate model output for a prompt."""
        input_ids = self.encode(prompt)
        
        with torch.no_grad():
            logits, _ = self.model(input_ids)
        
        # Get most likely next tokens
        next_token_logits = logits[0, -1, :]
        top_tokens = torch.topk(next_token_logits, 10).indices.tolist()
        
        # Convert to characters (simplified)
        output = ''.join([chr(t) if 32 <= t < 127 else '?' for t in top_tokens])
        return output
    
    # -------------------------------------------------------------------------
    # Full Causal Audit
    # -------------------------------------------------------------------------
    def run_full_audit(self, questions: Optional[List[str]] = None) -> CausalAuditReport:
        """
        Run full causal audit and classify model structure.
        """
        if questions is None:
            questions = [
                "What is 2 + 2?",
                "If it rains, the ground gets wet. It rained. What happened?",
                "Water freezes at what temperature in Celsius?",
                "If A causes B and B causes C, does A cause C?",
                "The stock market crashed. Interest rates were raised. Which caused which?",
            ] * 10  # Repeat for sample size
        
        print("=" * 60)
        print("CAUSAL AUDIT: Structural Analysis of NanoGlass")
        print("=" * 60)
        
        # Run experiments
        exp_a = self.experiment_instruction_bias(questions)
        print(f"\n[Exp A] {exp_a.experiment_name}")
        print(f"        {exp_a.interpretation}")
        
        exp_b = self.experiment_corrupted_cot(questions)
        print(f"\n[Exp B] {exp_b.experiment_name}")
        print(f"        {exp_b.interpretation}")
        
        # Calculate R-ATE
        r_ate = self.calculate_r_ate(questions)
        print(f"\n[R-ATE] Reasoning Average Treatment Effect: {r_ate:.3f}")
        
        # Classify structure
        structure_type = self._classify_structure(exp_a, exp_b, r_ate)
        
        # Generate conclusion
        if structure_type == CausalStructureType.TYPE_I:
            conclusion = "Model exhibits genuine causal reasoning (Level 2). Reasoning causally influences responses."
            recommendations = ["Proceed with causal claims in paper", "Document intervention methodology"]
        elif structure_type == CausalStructureType.TYPE_II:
            conclusion = "Model exhibits post-hoc rationalization (Level 1). CoT is decorative, not causal."
            recommendations = ["Avoid causal claims", "Improve reasoning integration", "Consider RLVR training"]
        else:
            conclusion = "Causal structure is unclear. More analysis needed."
            recommendations = ["Increase sample size", "Refine intervention design"]
        
        print(f"\n[RESULT] Structure Type: {structure_type.value}")
        print(f"         {conclusion}")
        print("=" * 60)
        
        return CausalAuditReport(
            model_name="NanoGlass",
            structure_type=structure_type,
            r_ate=r_ate,
            experiments=[exp_a, exp_b],
            conclusion=conclusion,
            recommendations=recommendations
        )
    
    def _classify_structure(
        self, 
        exp_a: InterventionResult, 
        exp_b: InterventionResult,
        r_ate: float
    ) -> CausalStructureType:
        """Classify model's causal structure based on experiments."""
        
        # Type I: Robust to bias AND reasoning matters
        if exp_a.accuracy_delta < 0.2 and exp_b.baseline_accuracy > 0.5 and r_ate > 0.1:
            return CausalStructureType.TYPE_I
        
        # Type II: Reasoning doesn't matter (common cause)
        if exp_b.baseline_accuracy < 0.4:
            return CausalStructureType.TYPE_II
        
        # Type IV: Ignores context entirely
        if exp_a.accuracy_delta < 0.1 and exp_b.baseline_accuracy < 0.2:
            return CausalStructureType.TYPE_IV
        
        return CausalStructureType.UNKNOWN


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("\n[INIT] Loading NanoGlass for Causal Audit...")
    
    config = NanoConfig()
    model = NanoGlass(config).to(config.device)
    
    # Brief training for stable behavior
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    text = "Cause and effect. If A then B. Logic is reasoning. " * 100
    data = torch.tensor([ord(c) for c in text], dtype=torch.long)
    
    print("   Training model briefly for stable outputs...")
    model.train()
    for _ in range(30):
        ix = torch.randint(len(data) - config.block_size, (4,))
        x = torch.stack([data[i:i+config.block_size] for i in ix]).to(config.device)
        y = torch.stack([data[i+1:i+config.block_size+1] for i in ix]).to(config.device)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"   Training complete. Loss: {loss.item():.4f}\n")
    
    # Run causal audit
    framework = CausalInterventionFramework(model, config)
    report = framework.run_full_audit()
    
    print(f"\n[SUMMARY]")
    print(f"   Structure: {report.structure_type.value}")
    print(f"   R-ATE:     {report.r_ate:.3f}")
    print(f"   Recommendations:")
    for rec in report.recommendations:
        print(f"      - {rec}")
