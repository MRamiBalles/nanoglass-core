"""
==============================================================================
SFT WARM-UP: ATOMIC DISTILLATION (CPU OPTIMIZED)
==============================================================================
Initializes the NanoGlass v2 (Jamba Hybrid) model via Supervised Fine-Tuning.
Optimized for CPU execution to avoid OOM via "MiniPuzzle" technique.

Optimizations:
    - Reduced Architecture (Mini-Jamba): 6 Layers, 16 Experts.
    - Gradient Accumulation: Simulate Batch Size 32 with Batch Size 1 RAM.
    - Context Clipping: 64 tokens (sufficient for atomic logic).

Output:
    - Saves 'nanoglass_sft_v2.pth'
==============================================================================
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import v2 Model
try:
    from llm_glassbox import CortexGlassBox, Config
except ImportError:
    from nanoglass import NanoConfig as Config, NanoGlass as CortexGlassBox

from experiments.synthetic_data_gen import SyntheticThermoDataset

def train_sft_cpu_optimized():
    print("\n[INIT] Atomic SFT Warm-up (CPU Protocol)")
    print("=" * 60)
    
    # 1. Setup Architecture (MiniPuzzle Reduction)
    device = "cpu"
    print(f"   Device: {device}")
    
    cfg = Config()
    cfg.n_layers = 6       # Reduced from 16 to 6
    cfg.moe_experts = 16   # Reduced from 64 to 16
    cfg.moe_top_k = 2      # Reduced from 6 to 2
    cfg.block_size = 64    # Atomic Context Window
    cfg.d_model = 384      # Keep width for compatibility (or reduce if needed)
    cfg.dropout = 0.1
    
    # Explicitly set batch/accum params
    real_batch_size = 1
    grad_accum_steps = 32
    effective_batch_size = real_batch_size * grad_accum_steps
    
    print(f"   Config: L={cfg.n_layers}, Experts={cfg.moe_experts}, Ctx={cfg.block_size}")
    
    model = CortexGlassBox(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model Params: {total_params/1e6:.1f}M (Reduced)")
    
    # 2. Data
    # 1000 samples high quality
    ds = SyntheticThermoDataset(cfg.vocab_size, cfg.block_size, sample_count=1000)
    dl = DataLoader(ds, batch_size=real_batch_size)
    
    # 3. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    
    # 4. Training Loop (Accumulated)
    epochs = 3
    model.train()
    
    print(f"   Training for {epochs} epochs with Grad Accumulation ({grad_accum_steps} steps)")
    
    for epoch in range(epochs):
        phase_name = get_phase_name(epoch)
        print(f"\n   [Epoch {epoch+1}/{epochs}] Phase: {phase_name}")
        
        optimizer.zero_grad()
        epoch_loss = 0
        micro_steps = 0
        updates = 0
        accum_loss = 0
        
        for i, (x, y) in enumerate(dl):
            x, y = x.to(device), y.to(device)
            
            # Forward
            logits, loss = model(x, y)
            
            # Scale loss for accumulation
            loss = loss / grad_accum_steps
            loss.backward()
            
            accum_loss += loss.item()
            micro_steps += 1
            
            # Optimization Step
            if micro_steps % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # Reporting
                updates += 1
                valid_loss = accum_loss * grad_accum_steps # Recover actual loss scale
                epoch_loss += valid_loss
                msg = f"Update {updates:03d} | Loss: {valid_loss:.4f}"
                print(f"      {msg} ", end="\r")
                
                # [LOG] Write to file for user to follow independently
                with open("sft_training.log", "a") as f:
                    f.write(msg + "\n")
                
                accum_loss = 0
                
        # Handle trailing gradients
        if micro_steps % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        avg_loss = epoch_loss / max(1, updates)
        print(f"\n      >> Epoch Completed. Avg Loss: {avg_loss:.4f}")
        
    # 5. Save Weights
    save_path = "nanoglass_sft_v2.pth"
    torch.save(model.state_dict(), save_path)
    print("=" * 60)
    print(f"[SUCCESS] SFT Complete. Weights saved to {save_path}")

def get_phase_name(epoch):
    if epoch == 0: return "Structural Alignment"
    if epoch == 1: return "Expert Specialization"
    return "Causal Logic"

if __name__ == "__main__":
    train_sft_cpu_optimized()
