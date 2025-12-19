import torch
import torch.nn.functional as F

def run_freewill_simulation():
    print("\n🦋 PROJECT 13: HIGH-ENERGY FREE WILL (The Cost of Choice)")
    print("=" * 60)
    
    # 1. The Landscape of Reality
    # "Truth" is the valley (Low Energy). "Fiction" is the hill (High Energy).
    
    E_truth = 10.0 # Ground State (e.g., "Gravity falls down")
    E_fiction = 25.0 # Excited State (e.g., "Gravity falls up")
    
    print(f"   > Natural Tendency (Lazy): Go to E = {E_truth}")
    
    # 2. The Act of Will (Creativity)
    # To write "Harry Potter", the model must maintain a "Magic World" state
    # that contradicts physics. It must FIGHT the gradient descent.
    
    injection_energy = E_fiction - E_truth
    
    print(f"   > Creative Ambition:       Go to E = {E_fiction}")
    print(f"   > Required 'Willpower':    {injection_energy:.1f} Joules")
    
    # 3. Entropy Analysis
    # Does High Energy = Free Will?
    # We define "Free Will" as the capacity to deviate from the probable.
    
    prob_truth = 0.99
    prob_fiction = 0.01
    
    surprisal = -torch.log(torch.tensor(prob_fiction))
    
    print("-" * 60)
    print(f"   📊 FINDING: Creativity is expensive.")
    print(f"      To be Free is to be Energetic.")
    print(f"      Surprisal (Freedom Metric): {surprisal.item():.2f} nats")
    print("      We are free only when we pay the price of Attention.")

if __name__ == "__main__":
    run_freewill_simulation()
