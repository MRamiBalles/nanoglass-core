import torch
from llm_glassbox import CortexGlassBox, Config, main_probe

def run_experiment(arch_type):
    print(f"\n🧪 EXPERIMENT: {arch_type} Architecture")
    print("=" * 40)
    
    # 1. Setup
    cfg = Config(n_layers=2, d_model=128, n_heads=4) # Small config for test
    model = CortexGlassBox(cfg, arch_type=arch_type)
    
    # 2. Input (Dummy "To be or not to be")
    idx = torch.tensor([[501, 12, 55, 99, 101]], dtype=torch.long)
    
    # 3. Clear previous probes
    main_probe.clear()
    
    # 4. Forward Pass
    print("🚀 Running Forward Pass...")
    logits, _ = model(idx)
    print(f"✅ Output Shape: {logits.shape}")
    
    # 5. Inspect Probes (Glass Box)
    print("\n🔍 GLASS BOX INSPECTION (Internal States):")
    if not main_probe.activations:
        print("   (No specific probes triggered this run - randomness factor?)")
    
    for name, tensor in main_probe.activations.items():
        print(f"   👁️  [{name}]: {tensor.shape} | MEAN: {tensor.mean().item():.4f}")
        
    print("-" * 40)

if __name__ == "__main__":
    # Test Hybrid (Mamba + Transformer)
    run_experiment('Hybrid')
    
    # Test MoE
    run_experiment('MoE')
