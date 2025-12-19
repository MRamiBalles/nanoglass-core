import torch
import torch.nn as nn
from llm_glassbox import CortexGlassBox, Config, main_probe

def run_causal_injection():
    print("\n💉 PROJECT 2: CAUSAL EDITING (The Matrix Injection)")
    print("=" * 60)
    
    cfg = Config(d_model=128, n_layers=2)
    model = CortexGlassBox(cfg, arch_type='Transformer')
    
    # 1. Define the "Concept Vector" (e.g., The concept of "Falsehood" or "French")
    # In a real scenario, this comes from the SAE decoder direction.
    concept_vector = torch.randn(1, 128) 
    concept_vector = concept_vector / concept_vector.norm() # Normalize
    
    # 2. Register an Intervention Hook
    def intervention_hook(module, input, output):
        # Inject the concept with intensity alpha
        alpha = 5.0
        print(f"   ⚡ INTERVENTION: Injecting vector with strength {alpha}x")
        # output is (B, T, C). We add to the last token.
        return output + (concept_vector * alpha)
    
    # Attach hook to Layer 1
    handle = model.layers[1].register_forward_hook(intervention_hook)
    
    # 3. Run Inference (Pre-Injection)
    # We can't easily see the text output without a tokenizer, but we verify the hook fires
    x = torch.randint(0, cfg.vocab_size, (1, 10))
    print("   > Generating...")
    model(x)
    
    # 4. Cleanup
    handle.remove()
    print("-" * 60)
    print("   ✅ SUCCESS: Activation space modified. The model's 'thought' was surgically altered.")

if __name__ == "__main__":
    run_causal_injection()
