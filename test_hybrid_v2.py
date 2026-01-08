
import torch
import torch.nn as nn
from llm_glassbox import CortexGlassBox, Config, NanoGlassHybridBlock
from mamba2_ssd import Mamba2Block
from layers.moe_granular import GranularMoE

def test_jamba_architecture():
    print("\n[TEST] Jamba Hybrid Architecture Validation")
    print("=" * 60)
    
    # Init config
    cfg = Config()
    cfg.n_layers = 16 
    model = CortexGlassBox(cfg)
    
    print(f"   Model created with {len(model.layers)} layers.")
    
    # 1. Verify 1:7 Attention Ratio
    print("\n   [1] Validating Attention/Mamba Ratio (1:7)...")
    attn_count = 0
    mamba_count = 0
    
    for i, layer in enumerate(model.layers):
        is_attn = layer.is_attention
        is_moe = layer.is_moe
        
        type_str = "ATTENTION" if is_attn else "MAMBA-2  "
        mlp_str = "MoE" if is_moe else "MLP"
        
        if is_attn:
            attn_count += 1
            # Check RoPE is disabled
            assert layer.mixer.rope is None, f"Layer {i}: RoPE should be disabled for Attention!"
        else:
            mamba_count += 1
            assert isinstance(layer.mixer, Mamba2Block), f"Layer {i}: Should be Mamba2Block"
            
        print(f"      Layer {i:02d}: {type_str} | {mlp_str}")
        
    print(f"      Stats: {attn_count} Attention / {mamba_count} Mamba layers")
    
    # Expectation: Layer 7 and 15 are Attention. Total 2.
    expected_attn = 2
    if attn_count == expected_attn and mamba_count == 14:
        print("      [OK] Ratio Correct!")
    else:
        print(f"      [FAIL] Ratio Mismatch! Expected {expected_attn} Attn tests.")

    # 2. Verify MoE Interleaving
    print("\n   [2] Validating MoE Graph...")
    moe_layers = [l for l in model.layers if l.is_moe]
    mlp_layers = [l for l in model.layers if not l.is_moe]
    
    print(f"      MoE Layers: {len(moe_layers)}")
    if len(moe_layers) == len(mlp_layers):
        print("      [OK] Interleaving Correct (50/50)!")
    else:
        print("      [FAIL] Interleaving Mismatch!")
        
    # 3. Parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n   [3] Total Parameters: {total_params:,}")

    # 4. Forward Pass
    print("\n   [4] Running Forward Pass...")
    x = torch.randint(0, cfg.vocab_size, (1, 64))
    try:
        logits, _ = model(x)
        print(f"      Forward Pass Successful. Output: {logits.shape}")
        print("      [OK] End-to-End Validated!")
    except Exception as e:
        print(f"      [FAIL] Forward Pass Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_jamba_architecture()
