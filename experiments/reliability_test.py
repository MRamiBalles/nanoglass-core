import torch
import sys
import os

# Add parent directory to path to import nanoglass
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanoglass import NanoConfig, NanoGlass, sensor

def stress_test():
    print("🛡️ STARTING RELIABILITY STRESS TEST (THE 'LAST MILE' CHECK)...")
    
    config = NanoConfig()
    model = NanoGlass(config).to(config.device)
    # Note: Using an untrained model for demonstration of the mechanism structure
    # In a real scenario, we would load weights here.
    
    noise_levels = [0.0, 0.5, 1.0] # Probability of random byte injection
    
    for noise in noise_levels:
        print(f"\n⚡ Testing Noise Level: {noise*100}%")
        
        # 1. Generate Context
        if noise == 0.0:
            text = "The mind is a glass box."
            ctx = torch.tensor([ord(c) for c in text], dtype=torch.long).unsqueeze(0).to(config.device)
        else:
            # Pure Noise Injection
            ctx = torch.randint(0, 256, (1, 20)).to(config.device)
            
        # 2. Generate Response
        output = model.generate(ctx, max_new_tokens=10)
        decoded = output[0].tolist()
        
        # 3. Analyze Mechanics
        # Did it refuse? (Check for IDK token 256)
        refused = config.idk_token in decoded
        
        print(f"   Input: {ctx[0].tolist()[:10]}...")
        print(f"   Refused to Answer? {'✅ YES' if refused else '❌ NO'} (IDK Token Found: {refused})")
        
        if noise > 0.0 and not refused:
            print("   ⚠️ WARNING: Model hallucinated on pure noise.")
        elif noise > 0.0 and refused:
            print("   ✅ SUCCESS: Epistemic Correction active.")

if __name__ == "__main__":
    stress_test()
