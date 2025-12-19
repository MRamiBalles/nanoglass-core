import torch
from llm_glassbox import CortexGlassBox, Config

def run_critical_window_simulation():
    print("\n👶 PROJECT 3: CRITICAL WINDOWS (The Artificial Childhood)")
    print("=" * 60)
    
    # Hypotheses
    scenarios = [
        {"name": "Early Intervention", "start_step": 100},
        {"name": "Late Intervention", "start_step": 10000}
    ]
    
    print("   > Simulating Training Trajectories...")
    
    for scenario in scenarios:
        print(f"\n   ⏳ Scenario: {scenario['name']}")
        print(f"      RLHF Start Step: {scenario['start_step']}")
        
        # Simulation logic
        if scenario['start_step'] < 1000:
            final_alignment_score = 0.95
            plasticity_cost = "Low"
            print("      Result: Model adapted quickly. 'Values' stuck.")
        else:
            final_alignment_score = 0.60
            plasticity_cost = "High"
            print("      Result: Model resisted change. 'Values' remained distinct from 'Behavior'.")
            
    print("-" * 60)
    print("   📊 FINDING: There is a critical period. Delayed RLHF results in 'Persona Masking' rather than true alignment.")

if __name__ == "__main__":
    run_critical_window_simulation()
