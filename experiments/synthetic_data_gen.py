"""
==============================================================================
SYNTHETIC DATA GENERATOR: THERMO-CAUSAL DATASET
==============================================================================
Generates high-quality synthetic training data for NanoGlass SFT Warm-up.
Combines:
1. ThermoLearn: Valid physics equations (Gibbs Free Energy)
2. Causal Reasoning: Step-by-step CoT derivation
3. Molecular Logic: Simple structural representations

Output format: PyTorch IterableDataset
==============================================================================
"""
import torch
from torch.utils.data import IterableDataset
import random
import numpy as np
from dataclasses import dataclass

@dataclass
class Material:
    name: str
    enthalpy: float # H (kJ/mol)
    entropy: float  # S (J/mol*K)
    
    def gibbs(self, T: float) -> float:
        # G = H - TS (Convert S to kJ for consistency if needed, but keeping simple scalar here)
        # Assuming units are consistent for the model's abstract "physics"
        return self.enthalpy - T * self.entropy

class SyntheticThermoDataset(IterableDataset):
    def __init__(self, vocab_size, block_size, sample_count=10000):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.sample_count = sample_count
        
        # Knowledge Base
        self.elements = ["Li", "Co", "O", "Si", "Fe", "Ni", "Mn", "Al"]
        self.structures = ["Cubic", "Hexagonal", "Monoclinic", "Amorphous"]
        
    def _generate_causal_chain(self):
        """
        Generates a reasoning chain:
        Question -> Law -> Calculation -> Conclusion
        """
        # 1. Setup Problem
        elem = random.choice(self.elements) + "2" + random.choice(self.elements) + "O4"
        struct = random.choice(self.structures)
        T = random.randint(200, 1000)
        
        # Hidden Physics "Ground Truth"
        H = random.uniform(-500.0, -100.0)
        S = random.uniform(0.1, 0.5)
        G = H - T * S
        stability = "Stable" if G < -300 else "Unstable" # Arbitrary threshold for toy physics
        
        # 2. Build Text
        # Format: "Question: [Q] \n Reasoning: [CoT] \n Answer: [A]"
        
        prompt = f"Analyze stability of {elem} ({struct}) at {T}K."
        
        cot = [
            f"1. Identify parameters: Enthalpy(H) ~ {H:.1f}, Entropy(S) ~ {S:.3f}.",
            f"2. Apply Gibbs Law: G = H - T*S.",
            f"3. Calculate: {H:.1f} - {T} * {S:.3f} = {G:.1f}.",
            f"4. Threshold check: {G:.1f} vs -300.0."
        ]
        
        conclusion = f"{stability} (G={G:.1f})"
        
        full_text = f"Question: {prompt}\nReasoning: {' '.join(cot)}\nAnswer: {conclusion}<|endoftext|>"
        return full_text

    def _generate_general_knowledge(self):
        """Simple syntax filler to prevent overfitting to equations."""
        templates = [
            "The mitochondrion is the powerhouse of the cell.",
            "Deep learning requires gradient descent optimization.",
            "NanoGlass is a hybrid architecture utilizing Mamba and MoE.",
            "Causal reasoning requires identifying Z -> T -> Y relationships.",
            "Entropy is a measure of disorder in a closed system."
        ]
        return random.choice(templates) + "<|endoftext|>"

    def __iter__(self):
        for _ in range(self.sample_count):
            # Ratio: 80% Causal Thermo, 20% General
            if random.random() < 0.8:
                text = self._generate_causal_chain()
            else:
                text = self._generate_general_knowledge()
                
            # Simple Byte-Pair Encode simulation (ASCII)
            # In production, use the real tokenizer. Here we use ASCII for the "Cold Start".
            # The model will learn *character-level* or *byte-level* patterns first.
            tokens = [ord(c) for c in text]
            tokens = tokens[:self.block_size] # Truncate
            
            # Pad if too short
            if len(tokens) < self.block_size:
                tokens += [0] * (self.block_size - len(tokens))
                
            x = torch.tensor(tokens[:-1], dtype=torch.long)
            y = torch.tensor(tokens[1:], dtype=torch.long) # Next token prediction
            
            yield x, y

if __name__ == "__main__":
    # Test generation
    ds = SyntheticThermoDataset(50304, 256, 5)
    print("Testing Synthetic Data Generation...")
    for x, y in ds:
        print(f"Sample X (len {len(x)}): {x[:10]}...")
        decoded = "".join([chr(i) for i in x if i != 0])
        print(f"Text: {decoded[:100]}...")
        break
