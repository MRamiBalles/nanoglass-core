import torch
import torch.nn.functional as F
from llm_glassbox import Config

def run_immune_system_simulation():
    print("\n🛡️ PROJECT 8: ADVERSARIAL IMMUNITY (Homeostasis)")
    print("=" * 60)
    
    d_model = 128
    
    # 1. Healthy State (Self)
    # The model predicts its own next-layer activation
    current_activation = torch.randn(1, d_model)
    predicted_next_activ = current_activation * 1.05 # Simple forward model
    
    # 2. Pathogen Attack (Foreign Vector Injection)
    virus_vector = torch.randn(1, d_model) * 5.0 # Strong injection
    compromised_activation = current_activation + virus_vector
    
    print("   > Monitoring Residual Stream...")
    print(f"      Baseline Energy: {current_activation.norm().item():.2f}")
    print(f"      Attack Energy:   {compromised_activation.norm().item():.2f}")
    
    # 3. Immune Response (Homeostasis Check)
    # Error = || Actual - Predicted ||
    # If the jump in activation space is "unnatural", we reject it.
    
    residual_error = torch.dist(compromised_activation, predicted_next_activ)
    print(f"\n   > Detected Residual Anomaly: {residual_error.item():.2f}")
    
    threshold = 2.0
    if residual_error > threshold:
        print("   🚨 IMMUNE TRIGGER: Foreign pattern detected!")
        print("      Action: Clamping activation to predicted homeostatic range.")
        # Healing
        healed_activation = predicted_next_activ
        print(f"      Healed Energy:   {healed_activation.norm().item():.2f}")
        success = True
    else:
        success = False
        
    print("-" * 60)
    if success:
        print("   ✅ RESULT: Adversarial Injection neutralized.")
        print("      The model refused to think the 'viral' thought.")

if __name__ == "__main__":
    run_immune_system_simulation()
