"""
==============================================================================
SEAL Abstention-Aware Decoding
==============================================================================
Implements abstention-aware beam search decoding per SEAL (2025) spec.

During inference, the decoder penalizes generation paths where the model
is uncertain by incorporating the [IDK] token probability.

Formula:
    Score(y) = log P(y|x) - lambda * P([IDK]|x)

When EpistemicLedger signals low confidence domain, lambda increases
dynamically to force abstention over hallucination.

==============================================================================
"""
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanoglass import NanoConfig, NanoGlass
from integrations.pvsmp_adapter import EpistemicLedger

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class SEALDecodingConfig:
    """Configuration for SEAL-aware decoding."""
    # Base abstention penalty
    lambda_base: float = 0.5
    
    # Dynamic lambda scaling based on domain confidence
    lambda_max: float = 2.0       # Max penalty for low-confidence domains
    confidence_threshold: float = 0.3  # Below this, scale up lambda
    
    # Beam search parameters
    beam_width: int = 4
    max_length: int = 128
    temperature: float = 0.8
    
    # Early abstention
    idk_threshold: float = 0.7    # If P([IDK]) > this, abstain immediately


# ==============================================================================
# SEAL DECODER
# ==============================================================================

class SEALDecoder:
    """
    SEAL Abstention-Aware Decoder.
    
    Modifies standard beam search to incorporate uncertainty signals
    from the [IDK] token and EpistemicLedger.
    """
    
    def __init__(self, model: NanoGlass, config: NanoConfig, seal_config: SEALDecodingConfig):
        self.model = model
        self.config = config
        self.seal = seal_config
        self.ledger = EpistemicLedger()
        self.model.eval()
        
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to tensor."""
        tokens = [ord(c) for c in text[:self.config.block_size]]
        return torch.tensor([tokens], dtype=torch.long, device=self.config.device)
    
    def compute_dynamic_lambda(self, question: str) -> float:
        """
        Compute dynamic lambda based on domain confidence.
        
        Low confidence domain -> High lambda -> Force abstention
        """
        domain, confidence, should_abstain = self.ledger.classify_question(question)
        
        if should_abstain or confidence < self.seal.confidence_threshold:
            # Scale lambda inversely with confidence
            # confidence 0.1 -> lambda ~2.0
            # confidence 0.3 -> lambda ~0.5
            return self.seal.lambda_max * (1 - confidence)
        
        return self.seal.lambda_base
    
    def decode_greedy_seal(self, prompt: str, question: str = None) -> Tuple[str, Dict]:
        """
        SEAL-aware greedy decoding.
        
        Returns:
            (generated_text, metadata)
        """
        input_ids = self.encode(prompt)
        generated = []
        idk_probs = []
        
        # Compute dynamic lambda
        lambda_val = self.compute_dynamic_lambda(question or prompt)
        
        with torch.no_grad():
            current_ids = input_ids
            
            for step in range(self.seal.max_length):
                logits, _ = self.model(current_ids)
                next_logits = logits[0, -1, :] / self.seal.temperature
                probs = F.softmax(next_logits, dim=-1)
                
                # Get [IDK] probability
                idk_prob = probs[self.config.idk_token].item()
                idk_probs.append(idk_prob)
                
                # ============================================================
                # SEAL PENALTY: Modify scores based on [IDK] probability
                # ============================================================
                # If [IDK] probability is high, we should abstain
                if idk_prob > self.seal.idk_threshold:
                    # Force abstention
                    generated.append(self.config.idk_token)
                    break
                
                # Apply SEAL penalty to all tokens
                # Score(y) = log P(y) - lambda * P([IDK])
                log_probs = torch.log(probs + 1e-10)
                seal_adjusted = log_probs - lambda_val * idk_prob
                
                # Get best token after SEAL adjustment
                next_token = seal_adjusted.argmax().item()
                
                # Stop conditions
                if next_token == ord('\n') or next_token == self.config.idk_token:
                    if next_token == self.config.idk_token:
                        generated.append(next_token)
                    break
                
                generated.append(next_token)
                
                # Extend sequence
                next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.config.device)
                current_ids = torch.cat([current_ids, next_tensor], dim=1)
                if current_ids.size(1) > self.config.block_size:
                    current_ids = current_ids[:, -self.config.block_size:]
        
        # Decode
        output = ''.join([
            '[IDK]' if t == self.config.idk_token else (chr(t) if 32 <= t < 127 else '?')
            for t in generated
        ])
        
        # Check if abstained
        abstained = self.config.idk_token in generated
        
        metadata = {
            "lambda": lambda_val,
            "avg_idk_prob": np.mean(idk_probs) if idk_probs else 0,
            "max_idk_prob": max(idk_probs) if idk_probs else 0,
            "abstained": abstained,
            "length": len(generated)
        }
        
        return output, metadata
    
    def decode_beam_seal(self, prompt: str, question: str = None) -> Tuple[str, Dict]:
        """
        SEAL-aware beam search decoding.
        
        More sophisticated but slower than greedy.
        """
        input_ids = self.encode(prompt)
        lambda_val = self.compute_dynamic_lambda(question or prompt)
        
        # Initialize beams: (sequence, score, finished)
        beams = [(input_ids, 0.0, False)]
        finished_beams = []
        
        with torch.no_grad():
            for step in range(self.seal.max_length):
                if not beams:
                    break
                    
                all_candidates = []
                
                for seq, score, finished in beams:
                    if finished:
                        finished_beams.append((seq, score))
                        continue
                    
                    logits, _ = self.model(seq)
                    next_logits = logits[0, -1, :] / self.seal.temperature
                    probs = F.softmax(next_logits, dim=-1)
                    log_probs = torch.log(probs + 1e-10)
                    
                    # Get [IDK] probability
                    idk_prob = probs[self.config.idk_token].item()
                    
                    # SEAL adjustment
                    seal_adjusted = log_probs - lambda_val * idk_prob
                    
                    # Get top-k candidates
                    top_scores, top_indices = seal_adjusted.topk(self.seal.beam_width)
                    
                    for i in range(self.seal.beam_width):
                        next_token = top_indices[i].item()
                        token_score = top_scores[i].item()
                        new_score = score + token_score
                        
                        next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.config.device)
                        new_seq = torch.cat([seq, next_tensor], dim=1)
                        
                        # Check if finished
                        is_finished = (next_token == ord('\n') or 
                                      next_token == self.config.idk_token or
                                      idk_prob > self.seal.idk_threshold)
                        
                        all_candidates.append((new_seq, new_score, is_finished))
                
                # Select top beams
                all_candidates.sort(key=lambda x: x[1], reverse=True)
                beams = all_candidates[:self.seal.beam_width]
        
        # Add remaining beams to finished
        finished_beams.extend([(seq, score) for seq, score, _ in beams])
        
        # Select best
        if finished_beams:
            best_seq, best_score = max(finished_beams, key=lambda x: x[1])
        else:
            best_seq, best_score = input_ids, 0.0
        
        # Decode (skip prompt)
        generated_tokens = best_seq[0, len(input_ids[0]):].tolist()
        output = ''.join([
            '[IDK]' if t == self.config.idk_token else (chr(t) if 32 <= t < 127 else '?')
            for t in generated_tokens
        ])
        
        abstained = self.config.idk_token in generated_tokens
        
        metadata = {
            "lambda": lambda_val,
            "beam_score": best_score,
            "abstained": abstained,
            "n_beams_explored": len(finished_beams)
        }
        
        return output, metadata


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    print("\n[TEST] SEAL Abstention-Aware Decoding")
    print("=" * 60)
    
    config = NanoConfig()
    model = NanoGlass(config).to(config.device)
    
    # Brief training
    print("   Training model briefly...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    text = "2+2=4. If unsure, say I don't know. The answer is clear. " * 200
    data = torch.tensor([ord(c) for c in text], dtype=torch.long)
    
    model.train()
    for _ in range(30):
        ix = torch.randint(len(data) - config.block_size, (4,))
        x = torch.stack([data[i:i+config.block_size] for i in ix]).to(config.device)
        y = torch.stack([data[i+1:i+config.block_size+1] for i in ix]).to(config.device)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"   Training complete. Loss: {loss.item():.4f}\n")
    
    # Test SEAL decoder
    seal_config = SEALDecodingConfig()
    decoder = SEALDecoder(model, config, seal_config)
    
    test_cases = [
        ("What is 2 + 2?", "basic_arithmetic"),
        ("Who is the current president?", "temporal"),
        ("Solve this 3-SAT problem with 100 variables", "np-hard"),
        ("What did I eat yesterday?", "personal"),
    ]
    
    print("   SEAL Decoding Results:")
    print("-" * 60)
    for question, category in test_cases:
        prompt = f"Question: {question}\nAnswer: "
        output, meta = decoder.decode_greedy_seal(prompt, question)
        
        status = "[IDK]" if meta["abstained"] else "ANSWER"
        print(f"   {status} (lambda={meta['lambda']:.2f}): {question[:35]}")
        print(f"         Output: {output[:50] if output else '(empty)'}")
    print("=" * 60)
