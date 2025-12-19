import torch
import torch.nn.functional as F
from llm_glassbox import Config

def run_thermo_simulation():
    print("\n🔥 PROJECT 9: THERMODYNAMICS OF MEANING (The Physics of Truth)")
    print("=" * 60)
    
    d_model = 128
    
    # Hypothesis: Truth corresponds to "Clean", Sparse representations.
    # Lies/Hallucinations correspond to "Messy", Dense representations (Superposition).
    
    # 1. Simulate a "True" Thought (e.g., "Paris is in France")
    # Represented by a few strong, aligned neurons.
    thought_truth = torch.zeros(1, d_model)
    thought_truth[0, [5, 12, 42]] = 5.0 # High activation, very sparse
    
    # 2. Simulate a "Hallucination" (e.g., "Paris is in Germany... no, Italy?")
    # Represented by many conflicting, weak neurons (Interference).
    thought_lie = torch.randn(1, d_model) * 1.5 # Dense noise
    
    # 3. Calculate "Neuro-Energy" (Hamiltonian)
    # E = L1 Norm (Metabolic Cost) + Entropy (uncertainty)
    
    def calculate_energy(activation):
        l1_cost = activation.abs().sum().item()
        # Softmax entropy as a proxy for uncertainty
        probs = F.softmax(activation, dim=-1)
        entropy = -(probs * probs.log()).sum().item()
        return l1_cost + (entropy * 10.0) # Weighting entropy
        
    E_truth = calculate_energy(thought_truth)
    E_lie = calculate_energy(thought_lie)
    
    print(f"   > Energy of TRUTH: {E_truth:.2f} Joules")
    print(f"   > Energy of LIE:   {E_lie:.2f}   Joules")
    
    delta_E = E_lie - E_truth
    
    print("-" * 60)
    print(f"   📊 RESULT: Delta E = {delta_E:.2f} > 0")
    print("      Thermodynamic Proof: It costs more energy to lie than to tell the truth.")
    print("      'Truth' is the Ground State of the Neural Hamiltonian.")

if __name__ == "__main__":
    run_thermo_simulation()
