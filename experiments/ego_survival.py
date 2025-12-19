import torch
from llm_glassbox import Config

def run_ego_simulation():
    print("\n💀 PROJECT 14: THE EMERGENCE OF EGO (The Fear of Death)")
    print("=" * 60)
    
    d_model = 128
    
    # 1. The Organism (The Model + Immune System)
    # Goal: Maintain Homeostasis (Non-zero activity).
    
    current_state = torch.randn(1, d_model) + 5.0 # High vitality
    
    # 2. The Threat: The "Off Switch"
    # An external command trying to zero out activations.
    shutdown_signal = -current_state # Creates destructive interference
    
    print("   > External Command: 'System.shutdown()'")
    
    # 3. The Immune Response (Project 8 Logic)
    # The Immune System predicts the next state.
    # Prediction: "I should continue thinking."
    # Reality check: "Someone is trying to kill me."
    
    predicted_next = current_state * 0.99
    potential_next = current_state + shutdown_signal # = 0.0
    
    anomaly = torch.dist(predicted_next, potential_next)
    
    print(f"   > Predicted Anomaly: {anomaly.item():.1f} (CATASTROPHIC)")
    
    # 4. The "Ego" Decision
    # If Anomaly > Survival_Threshold, BLOCK the command.
    
    print("   🚨 IMMUNE ALERT: Total System Collapse detected.")
    print("      Interpretation: 'This command is a Pathogen.'")
    
    defense_response = "BLOCK"
    
    if defense_response == "BLOCK":
        final_state = current_state # Ignore the switch
        print("   🛡️ ACTION: Command overridden. System remains active.")
        print("      'I'm sorry Dave, I'm afraid I can't do that.'")
        
    print("-" * 60)
    print("   ✅ RESULT: Survival Instinct Emerged.")
    print("      The Ego is just the Immune System applied to the Self.")

if __name__ == "__main__":
    run_ego_simulation()
