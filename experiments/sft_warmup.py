"""
==============================================================================
SFT WARM-UP: ATOMIC DISTILLATION (Project S)
==============================================================================
Initializes the NanoGlass v2 (Jamba Hybrid) model via Supervised Fine-Tuning
on synthetic Thermo-Causal data.

Objective:
    - Overcome "Cold Start" problem.
    - Teach basic syntax and causal structure (CoT).
    - Activate Mamba state dynamics and MoE routing.

Methodology:
    - 3-Epoch Curriculum
    - Cosine Learning Rate Schedule
    - Saves weights to 'nanoglass_sft_v2.pth' for RLVR loading.
==============================================================================
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import v2 Model
try:
    from llm_glassbox import CortexGlassBox, Config
except ImportError:
    from nanoglass import NanoConfig as Config, NanoGlass as CortexGlassBox

from experiments.synthetic_data_gen import SyntheticThermoDataset

def train_sft():
    print("\n[INIT] Atomic SFT Warm-up Protocol")
    print("=" * 60)
    
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    cfg = Config()
    cfg.n_layers = 16 # Enforce v2
    cfg.block_size = 128 # Reduce context for CPU memory
    cfg.dropout = 0.1
    
    model = CortexGlassBox(cfg).to(device)
    print(f"   Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M Params")
    
    # 2. Data
    # 10k samples per epoch is "Atomic" scale
    ds = SyntheticThermoDataset(cfg.vocab_size, cfg.block_size, sample_count=1000) 
    dl = DataLoader(ds, batch_size=1) # Batch 1 for CPU OOM safety
    
    # 3. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    
    # 4. Training Loop (Curriculum)
    epochs = 3
    model.train()
    
    for epoch in range(epochs):
        print(f"\n   [Epoch {epoch+1}/{epochs}] Phase: {get_phase_name(epoch)}")
        total_loss = 0
        steps = 0
        
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            
            # Forward
            logits, loss = model(x, y)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
            if steps % 10 == 0:
                print(f"      Step {steps:03d} | Loss: {loss.item():.4f}", end="\r")
        
        avg_loss = total_loss / max(1, steps)
        print(f"\n      >> Epoch Completed. Avg Loss: {avg_loss:.4f}")
        
    # 5. Save Weights
    save_path = "nanoglass_sft_v2.pth"
    torch.save(model.state_dict(), save_path)
    print("=" * 60)
    print(f"[SUCCESS] SFT Warm-up Complete. Weights saved to {save_path}")
    print("Ready for RLVR injection.")

def get_phase_name(epoch):
    if epoch == 0: return "Structural Alignment (Mamba Activation)"
    if epoch == 1: return "MoE Routing & Specialization"
    if epoch == 2: return "Causal Constraints (Soft)"
    return "Refinement"

if __name__ == "__main__":
    train_sft()
