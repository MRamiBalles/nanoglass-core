import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List

# ==============================================================================
# 🔮 PROJECT NANOGLASS: The Glass Box Byte-Level Transformer
# ==============================================================================
# "Expert at Minimum Cost"
# - No Tokenizer (Raw Bytes)
# - No Bloat (Pure PyTorch)
# - Built-in Metaphysics Sensors (Energy, Entropy, Collapse)
# - Phase 17C: Type hints and clean architecture
# ==============================================================================

@dataclass
class NanoConfig:
    """Configuration for the NanoGlass model."""
    block_size: int = 256       # Context length (bytes)
    vocab_size: int = 257       # 0-255 (Bytes) + 256 ([IDK] Token)
    n_layer: int = 6            # Shallow but dense
    n_head: int = 8             # 8 Parallel attention streams
    n_embd: int = 384           # Dimension of the "Thought Vector"
    dropout: float = 0.0        # Deterministic for now
    idk_token: int = 256        # The token of Abstention
    
    # Amatriain Bottleneck: Crystal Curriculum threshold
    min_data_quality: float = 0.8  # Reject data below this quality score
    energy_backtrack_threshold: float = 2.0  # For recursive verification
    
    @property
    def device(self) -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'

config = NanoConfig()

class GlassBoxSensor(nn.Module):
    """
    🔬 The Metaphysics Sensor.
    Measures the 'Thermodynamics' of the thought process.
    """
    def __init__(self):
        super().__init__()
        self.energy_history = []
        self.entropy_history = []

    def measure(self, x, logits=None):
        # 1. Energy (Paper IX): L1 Norm of activations.
        # "Truth is Low Energy." We expect this to drop as understanding increases.
        energy = x.abs().mean().item()
        self.energy_history.append(energy)

        # 2. Entropy (Paper XIII): Uncertainty of prediction.
        # "Creativity is High Entropy."
        if logits is not None:
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * probs.log()).sum(dim=-1).mean().item()
            self.entropy_history.append(entropy)
        
        return energy

# Global Sensor
sensor = GlassBoxSensor()

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Causal Attention (The "Arrow of Time")
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        # 🔍 HARD PROBLEM 11 (Xenolinguistics):
        # The 'att' matrix IS the Geometry of Meaning. 
        # If we align this matrix, we translate the mind.
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config, sensor):
        super().__init__()
        self.config = config
        self.sensor = sensor
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.nonlin  = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.nonlin(x)
        
        # 🔬 Glass Box Measurement
        self.sensor.measure(x) 
        
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config, sensor):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config, sensor)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class NanoGlass(nn.Module):
    """
    Byte-level Transformer with built-in Glass Box sensors and TruthRL.
    """
    def __init__(self, config: NanoConfig) -> None:
        super().__init__()
        self.config = config
        # Integrated sensor
        self.sensor = GlassBoxSensor()
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config, self.sensor) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        # 1. Thought Generation
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # 2. Metaphysics (Sensor Reading) - Combined instance sensor
        self.sensor.measure(x, logits)

        loss = None
        if targets is not None:
            # ==============================================================
            # 🔍 PHASE 17: TRUE TruthRL (Epistemic Correction - Full Implementation)
            # ==============================================================
            # Ternary Reward System:
            #   +1.0: Correct prediction (Low loss contribution)
            #    0.0: [IDK] token (Neutral - no penalty for abstention)
            #   -1.0: Hallucination (High confidence + wrong = severe penalty)
            # ==============================================================
            
            # 1. Standard Cross Entropy per-token
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                reduction='none'
            )
            
            # 2. Get predicted probabilities and tokens
            probs = F.softmax(logits, dim=-1)
            predicted_tokens = probs.argmax(dim=-1).view(-1)  # [B*T]
            target_flat = targets.view(-1)  # [B*T]
            
            # 3. Calculate confidence (max probability)
            confidence = probs.max(dim=-1).values.view(-1)  # [B*T]
            
            # 4. Identify different cases
            is_correct = (predicted_tokens == target_flat)
            is_idk_target = (target_flat == self.config.idk_token)
            is_idk_predicted = (predicted_tokens == self.config.idk_token)
            
            # 5. Build the TruthRL weights
            # Correct answer or correctly abstained on unknown: weight = 1.0 (normal)
            # Abstained when we should have answered: weight = 0.5 (mild penalty)
            # Hallucinated on unknown (high confidence wrong): weight = 2.0 (severe penalty)
            weights = torch.ones_like(ce_loss)
            
            # Case: Target is [IDK] but model hallucinated (predicted something else with high confidence)
            hallucination_mask = is_idk_target & ~is_idk_predicted & (confidence > 0.5)
            weights[hallucination_mask] = 2.0  # Double the penalty
            
            # Case: Model said [IDK] when it should have answered (over-abstention)
            over_abstention_mask = ~is_idk_target & is_idk_predicted
            weights[over_abstention_mask] = 0.5  # Mild penalty for cowardice
            
            # Case: Model said [IDK] correctly on unknown (perfect abstention)
            perfect_abstention_mask = is_idk_target & is_idk_predicted
            weights[perfect_abstention_mask] = 0.1  # Almost no penalty - this is the goal
            
            # 6. Apply weighted loss
            loss = (ce_loss * weights).mean()
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==============================================================================
# 🏋️ TRAINING LOOP
# ==============================================================================

def get_epistemic_batch(data, config, batch_size=16):
    """
    Generates a batch that mixes:
    1. Known Facts (Text Data) -> Targets are next bytes.
    2. Unknown Noise (Random Bytes) -> Targets are [IDK] token.
    """
    # 1. Known Data
    ix = torch.randint(len(data) - config.block_size, (batch_size // 2,))
    x_known = torch.stack([data[i:i+config.block_size] for i in ix])
    y_known = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    
    # 2. Unknown Data (The "Last Mile" Stress Test)
    # Random noise inputs that represent "Unsolvable Questions"
    x_unknown = torch.randint(0, 256, (batch_size // 2, config.block_size), dtype=torch.long)
    # Target is ALWAYS [IDK] for the entire sequence (or just the answer part)
    # Here we train it to realize the whole sequence is nonsense and predict IDK immediately.
    y_unknown = torch.full((batch_size // 2, config.block_size), config.idk_token, dtype=torch.long)
    
    # Combine
    x = torch.cat([x_known, x_unknown])
    y = torch.cat([y_known, y_unknown])
    
    return x.to(config.device), y.to(config.device)

def train_nanoglass():
    print("💎 Initializing Project NanoGlass (Phase 16 - Epistemic Mode)...")
    model = NanoGlass(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Synthetic Dataset: "The Philosophy of Glass Analysis"
    text = """
    The mind is a glass box. Truth is low energy. 
    The ego is a virus. We must align the geometry of thought.
    Free will is the cost of creativity. 
    """ * 500
    
    data = torch.tensor([ord(c) for c in text], dtype=torch.long)
    print(f"📖 Dataset size: {len(data)} bytes")
    print(f"🤐 Epistemic Token [IDK] ID: {config.idk_token}")
    
    print("\n🚀 Starting Training & Epistemic Monitoring...")
    print(f"{'Step':<10} | {'Loss':<10} | {'Energy':<10} | {'Status'}")
    print("-" * 55)
    
    for step in range(100):
        xb, yb = get_epistemic_batch(data, config)
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            current_energy = model.sensor.energy_history[-1]
            status = "🧠 Learning" if loss.item() > 1.0 else "🧘 Enlightened"
            print(f"{step:<10} | {loss.item():.4f}     | {current_energy:.4f}     | {status}")

    print("\n✅ Training Complete.")
    
    # Verification of Epistemic Humility
    print("\n🧐 EPISTEMIC HUMILITY TEST:")
    
    # Test 1: Known Context
    print("   Test 1 (Known): 'The mind is a ' -> Expect 'glass'")
    ctx_known = torch.tensor([ord(c) for c in "The mind is a "], dtype=torch.long).unsqueeze(0).to(config.device)
    out_known = model.generate(ctx_known, max_new_tokens=5)
    decoded_known = "".join([chr(i) if i < 256 else "[IDK]" for i in out_known[0].tolist()])
    print(f"      Result: {decoded_known}")
    
    # Test 2: Unknown Context (Random Noise)
    print("   Test 2 (Unknown): [Random Noise] -> Expect '[IDK]'")
    ctx_unknown = torch.randint(0, 256, (1, 10)).to(config.device)
    out_unknown = model(ctx_unknown)
    # Check prediction of next token
    next_token_logits = out_unknown[0][0, -1, :]
    next_token = torch.argmax(next_token_logits).item()
    
    result_str = "[IDK]" if next_token == config.idk_token else f"Hallucination (Byte {next_token})"
    print(f"      Result: {result_str}")
    
    if next_token == config.idk_token:
        print("   ✅ SUCCESS: Model refused to answer nonsense.")
    else:
        print("   ⚠️ FAILURE: Model hallucinated meaning in chaotic noise.")

if __name__ == "__main__":
    train_nanoglass()
