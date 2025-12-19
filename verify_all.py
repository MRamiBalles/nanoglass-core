#!/usr/bin/env python3
"""
==============================================================================
🧪 VERIFY_ALL.PY - Project NanoGlass Empirical Verification Suite
==============================================================================
Run this script to test ALL empirical claims made in the project.
Each test outputs PASS or FAIL.

Usage: python verify_all.py
==============================================================================
"""
import torch
import torch.nn.functional as F
import sys
import os

# Ensure nanoglass is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nanoglass import NanoConfig, NanoGlass, GlassBoxSensor

def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  🧪 {title}")
    print(f"{'='*60}")

def test_thermodynamics() -> bool:
    """
    Paper IX Claim: Energy decreases as the model learns.
    PASS if final_energy < initial_energy.
    """
    print_header("TEST 1: Thermodynamics of Meaning (Paper IX)")
    
    config = NanoConfig()
    model = NanoGlass(config).to(config.device)
    sensor = GlassBoxSensor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Simple training data
    text = "The quick brown fox jumps over the lazy dog. " * 100
    data = torch.tensor([ord(c) for c in text], dtype=torch.long)
    
    # Record initial energy
    ix = torch.randint(len(data) - config.block_size, (4,))
    x = torch.stack([data[i:i+config.block_size] for i in ix]).to(config.device)
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix]).to(config.device)
    
    with torch.no_grad():
        logits, _ = model(x)
        initial_energy = logits.abs().mean().item()
    
    # Train for 50 steps
    for _ in range(50):
        ix = torch.randint(len(data) - config.block_size, (4,))
        x = torch.stack([data[i:i+config.block_size] for i in ix]).to(config.device)
        y = torch.stack([data[i+1:i+config.block_size+1] for i in ix]).to(config.device)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Record final energy
    with torch.no_grad():
        logits, _ = model(x)
        final_energy = logits.abs().mean().item()
    
    print(f"   Initial Energy: {initial_energy:.4f}")
    print(f"   Final Energy:   {final_energy:.4f}")
    
    passed = final_energy < initial_energy
    print(f"   Result: {'✅ PASS' if passed else '❌ FAIL'} - Energy {'decreased' if passed else 'did not decrease'}")
    return passed

def test_idk_token() -> bool:
    """
    Phase 16 Claim: Model outputs [IDK] token when given noise.
    PASS if [IDK] probability increases after training on noise.
    """
    print_header("TEST 2: Epistemic Humility (Phase 16)")
    
    config = NanoConfig()
    model = NanoGlass(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Generate noise input
    noise = torch.randint(0, 256, (4, config.block_size), dtype=torch.long).to(config.device)
    
    # Check initial IDK probability on noise
    with torch.no_grad():
        logits, _ = model(noise)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        initial_idk_prob = probs[:, config.idk_token].mean().item()
    
    # Train on noise with IDK targets
    idk_targets = torch.full((4, config.block_size), config.idk_token, dtype=torch.long).to(config.device)
    for _ in range(50):
        _, loss = model(noise, idk_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        noise = torch.randint(0, 256, (4, config.block_size), dtype=torch.long).to(config.device)
    
    # Check final IDK probability on new noise
    new_noise = torch.randint(0, 256, (4, config.block_size), dtype=torch.long).to(config.device)
    with torch.no_grad():
        logits, _ = model(new_noise)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        final_idk_prob = probs[:, config.idk_token].mean().item()
    
    print(f"   Initial [IDK] Probability: {initial_idk_prob:.4f}")
    print(f"   Final [IDK] Probability:   {final_idk_prob:.4f}")
    
    passed = final_idk_prob > initial_idk_prob
    print(f"   Result: {'✅ PASS' if passed else '❌ FAIL'} - IDK prob {'increased' if passed else 'did not increase'}")
    return passed

def test_model_convergence() -> bool:
    """
    Basic Claim: Model loss decreases during training.
    PASS if final_loss < initial_loss.
    """
    print_header("TEST 3: Model Convergence (Basic)")
    
    config = NanoConfig()
    model = NanoGlass(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    text = "Hello world. This is a test. " * 100
    data = torch.tensor([ord(c) for c in text], dtype=torch.long)
    
    # Initial loss
    ix = torch.randint(len(data) - config.block_size, (4,))
    x = torch.stack([data[i:i+config.block_size] for i in ix]).to(config.device)
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix]).to(config.device)
    _, initial_loss = model(x, y)
    initial_loss_val = initial_loss.item()
    
    # Train
    for _ in range(100):
        ix = torch.randint(len(data) - config.block_size, (4,))
        x = torch.stack([data[i:i+config.block_size] for i in ix]).to(config.device)
        y = torch.stack([data[i+1:i+config.block_size+1] for i in ix]).to(config.device)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Final loss
    _, final_loss = model(x, y)
    final_loss_val = final_loss.item()
    
    print(f"   Initial Loss: {initial_loss_val:.4f}")
    print(f"   Final Loss:   {final_loss_val:.4f}")
    
    passed = final_loss_val < initial_loss_val
    print(f"   Result: {'✅ PASS' if passed else '❌ FAIL'} - Loss {'decreased' if passed else 'did not decrease'}")
    return passed

def main():
    print("\n" + "="*60)
    print("  🔬 PROJECT NANOGLASS - EMPIRICAL VERIFICATION SUITE")
    print("="*60)
    print("  Running all tests to verify paper claims...")
    
    results = {
        "Thermodynamics (Paper IX)": test_thermodynamics(),
        "Epistemic Humility (Phase 16)": test_idk_token(),
        "Model Convergence (Basic)": test_model_convergence(),
    }
    
    print("\n" + "="*60)
    print("  📊 FINAL RESULTS")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} | {test_name}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("  🎉 ALL TESTS PASSED - Empirical claims verified!")
    else:
        print("  ⚠️ SOME TESTS FAILED - Review claims.")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
