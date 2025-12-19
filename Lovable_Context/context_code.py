=== FILE: llm_glassbox.py ===
# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘                  ðŸ”¬ CORTEX-13: GLASS BOX EDITION                         â•‘
# â•‘           Deep Interpretability & Architectural Analysis                     â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

# ==============================================================================
#  CHAPTER 1: PROBING INFRASTRUCTURE (THE "GLASS BOX" MECHANISM)
# ==============================================================================
# To understand the "Mind" of the LLM, we cannot just look at the output.
# We must insert "Probes" (thermometers) at key locations to measure internal activity.
# This allows us to extract "Symbolic" representations from the "Sub-symbolic" vectors.

class GlassBoxProbe:
    """
    A tool to capture and store internal activations during the forward pass.
    Think of this as an EEG for the neural network.
    """
    def __init__(self):
        self.activations: Dict[str, torch.Tensor] = {}

    def hook(self, name):
        """Returns a hook function to capture output of a layer."""
        def forward_hook(module, input, output):
            # We detach to save memory and avoid messing with gradients during inference/analysis
            self.activations[name] = output.detach()
        return forward_hook

    def clear(self):
        self.activations = {}

# Global probe registry
main_probe = GlassBoxProbe()


# ==============================================================================
#  CHAPTER 2: MODERN COMPONENTS & TRADE-OFFS
# ==============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    
    DECISION: Why RMSNorm over LayerNorm?
    1. Efficiency: RMSNorm is computationally cheaper (~15-40% faster in practice) because it 
       skips the mean calculation and centering operation.
    2. Scaling Invariance: It effectively only scales the vector magnitude, preserving the 
       direction of the activation vector, which is often where the semantic info lives.
    
    ALTERNATIVE: LayerNorm (Standard in BERT/GPT-2).
    - LayerNorm centers data (activations - mean). This is theoretically cleaner but empirically unnecessary for LLMs.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x.pow(2).mean = calculating the magnitude (roughly)
        # rsqrt = reciprocal square root (1/sqrt)
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight

class RoPE(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    
    DECISION: Why RoPE over Absolute (Learned) or Relative Embeddings?
    1. Extrapolation: RoPE generalizes better to sequence lengths longer than seen during training.
    2. Relative Logic: It encodes relative position information purely through rotation in the complex plane.
       The dot product of two RoPE'd vectors depends only on their relative distance (m - n), not absolute position.
    
    ALTERNATIVE: ALiBi (Attention with Linear Biases).
    - ALiBi is simpler (adds static bias to attention matrix) and extrapolates well, but RoPE 
      is currently the "Gold Standard" (used in Llama, PaLM, GPT-4 rumors) for its mathematical elegance and expressivity.
    """
    def __init__(self, dim, max_len=4096):
        super().__init__()
        # Precompute frequencies involved in the rotation
        # The key idea: lower frequencies for earlier dimensions, higher for later ones.
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Build cache for max_len (can increase dynamically if needed)
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq) # shape (max_len, dim/2)
        
        # Store as complex exponentials or sin/cos directly
        # We use sin/cos for better compatibility with non-complex implementations
        self.register_buffer('cos', freqs.cos()[None, None, :, :]) # (1, 1, max_len, dim/2)
        self.register_buffer('sin', freqs.sin()[None, None, :, :])

    def forward(self, q, k):
        # q, k shape: (Batch, Heads, SeqLen, HeadDim)
        seq_len = q.shape[2]
        
        # Slice the precomputed values associated with current length
        cos_slice = self.cos[:,:,:seq_len,:]
        sin_slice = self.sin[:,:,:seq_len,:]

        def apply_rotary_pos_emb(x, cos_val, sin_val):
            # ROTATION OPERATION:
            # To rotate a vector (x,y) by angle theta:
            # x' = x*cos(theta) - y*sin(theta)
            # y' = x*sin(theta) + y*cos(theta)
            x1, x2 = x.chunk(2, -1)
            return torch.cat([x1 * cos_val - x2 * sin_val, x2 * cos_val + x1 * sin_val], -1)

        return apply_rotary_pos_emb(q, cos_slice, sin_slice), apply_rotary_pos_emb(k, cos_slice, sin_slice)

class SwiGLU(nn.Module):
    """
    SwiGLU Activation.
    
    DECISION: Why SwiGLU over GELU or ReLU?
    1. Gating Mechanism: It's a "Gated Linear Unit". One path (w1) determines the "value", 
       the other (w2) determines the "gate" (how much of that value to pass).
    2. Learning capability: The element-wise multiplication allows for more complex interactions 
       within the neuron itself.
    
    TRADE-OFF: Parameters.
    - SwiGLU requires 3 linear projections instead of 2 in a standard MLP (up projection, gate projection, down projection).
    - To keep parameter count constant, we usually reduce the hidden dimension slightly (e.g., from 4d to 8/3d or ~2.6d).
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False) # Value
        self.w2 = nn.Linear(dim, hidden_dim, bias=False) # Gate
        self.w3 = nn.Linear(hidden_dim, dim, bias=False) # Output

    def forward(self, x):
        # F.silu is the "Swish" part (x * sigmoid(x))
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class GQA(nn.Module):
    """
    Grouped Query Attention (GQA).
    
    DECISION: Why GQA over MHA (Multi-Head) or MQA (Multi-Query)?
    - MHA (Standard): Each query head has its own Key/Value head. Max quality, High VRAM usage for KV cache.
    - MQA: All query heads share ONE Key/Value head. Max speed/efficiency, slight quality degradation.
    - GQA (Winner): A compromise. We group query heads and they share a KV head.
      Example: 32 Query heads, 4 groups -> 8 KV heads.
      Result: Very close to MHA quality, much closer to MQA speed/memory.
    """
    def __init__(self, d_model, n_heads, n_kv_groups=None, use_rope=True):
        super().__init__()
        self.n_heads = n_heads
        # Default to having n_heads/2 groups if not specified (aggressive grouping)
        self.n_kv = n_kv_groups if n_kv_groups else n_heads // 2 
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, self.head_dim * self.n_kv, bias=False)
        self.v = nn.Linear(d_model, self.head_dim * self.n_kv, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RoPE(self.head_dim) if use_rope else None

    def forward(self, x, layer_id=0):
        B, T, C = x.shape
        
        # Project and separate heads
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1,2) # (B, H_q, T, D)
        k = self.k(x).view(B, T, self.n_kv, self.head_dim).transpose(1,2)    # (B, H_kv, T, D)
        v = self.v(x).view(B, T, self.n_kv, self.head_dim).transpose(1,2)    # (B, H_kv, T, D)

        # Apply RoPE (Rotary Position Embeddings)
        # This injects "where we are" into the "what we are looking for" vectors
        if self.rope:
            q, k = self.rope(q, k)

        # GLASS BOX PROBE: Capture Attention Logic
        # Before we repeat K/V, the raw values represent the "memory" availability.
        
        # Repeat KV heads to match Q heads for calculation
        # This is the "Grouped" magical step. 
        # If we have 4 Q heads per 1 KV head, we copy that KV head 4 times.
        if self.n_heads != self.n_kv:
            k = k.repeat_interleave(self.n_heads // self.n_kv, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv, dim=1)

        # Scaled Dot-Product Attention
        # Attn = Softmax(Q @ K.T / sqrt(dim))
        # This is the core "Association" mechanism.
        if hasattr(F, 'scaled_dot_product_attention'):
            # Torch 2.0+ Flash Attention (Optimized kernel)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Manual implementation for educational clarity / older pyTorch
            att = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
            mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
            att = att.masked_fill(mask, float('-inf'))
            att_weights = F.softmax(att, -1)
            
            # PROBE: Capture the attention pattern
            # This "att_weights" matrix IS the "Reasoning Trace". 
            # It shows exactly which previous tokens the model is looking at to predict the next one.
            if random.random() < 0.01: # Sample occasionally to save RAM
                main_probe.activations[f'layer_{layer_id}_attn_weights'] = att_weights.detach().cpu()
                
            out = att_weights @ v

        # Combine heads back
        out = out.transpose(1,2).contiguous().view(B, T, C)
        return self.proj(out)

# ==============================================================================
#  CHAPTER 3: ARCHITECTURAL BLOCKS
# ==============================================================================

class TransformerBlock(nn.Module):
    """
    Standard Transformer Decoder Block.
    Flow: x -> Norm -> Attention -> Add -> Norm -> FeedForward -> Add
    Pre-norm design is used (Norm BEFORE sub-layer) for training stability.
    """
    def __init__(self, cfg, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = RMSNorm(cfg.d_model)
        self.ln2 = RMSNorm(cfg.d_model)
        self.attn = GQA(cfg.d_model, cfg.n_heads, cfg.n_kv_groups, cfg.use_rope)
        self.ffn = SwiGLU(cfg.d_model, int(2.6 * cfg.d_model)) # Approx 4*d in params

    def forward(self, x):
        # Residual Connection 1 (Attention)
        x = x + self.attn(self.ln1(x), layer_id=self.layer_id)
        # Residual Connection 2 (FeedForward)
        # The FFN is often thought of as a Key-Value Memory where semantic concepts are stored.
        # "Who is the president?" might be resolved here.
        x = x + self.ffn(self.ln2(x))
        return x

# ==============================================================================
#  CHAPTER 5: EXOTIC ARCHITECTURES (MAMBA, RWKV, MOE)
# ==============================================================================

class MambaBlock(nn.Module):
    """
    Mamba (State Space Model) Block.
    
    DECISION: Why Mamba over Transformer?
    1. Linear Complexity: Mamba scales linearly O(N) with sequence length, unlike Transformer's O(N^2).
    2. Recurrent State: It maintains a compressed "hidden state" (h_t) that passes information 
       forward, similar to an RNN but with structured matrices (HiPPO) for long-term memory.
    
    GLASS BOX NOTE:
    The 'hidden_state' here is a goldmine for symbolic extraction. Unlike the 'K/V cache' 
    which stores specific past tokens, the Mamba state stores a *compressed representation* 
    of the context.
    """
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.norm = RMSNorm(cfg.d_model)
        
        # Projections
        self.in_proj = nn.Linear(cfg.d_model, cfg.d_model * 2, bias=False)
        self.conv = nn.Conv1d(cfg.d_model, cfg.d_model, 3, padding=1, groups=cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        
    def forward(self, x):
        # x: (B, T, D)
        residual = x
        x = self.norm(x)
        
        # Mamba Inner Flow:
        # 1. Project to input/gate
        x_and_gate = self.in_proj(x)
        x_val, gate = x_and_gate.chunk(2, dim=-1)
        
        # 2. Convolution (Short-term context)
        # Transpose for Conv1d: (B, D, T)
        x_conv = self.conv(x_val.transpose(1, 2)).transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # 3. State Space Model (simplified approximated linear scan for educational purpose)
        # Real Mamba uses a selective scan kernel. Here we use a gated multiplication 
        # to simulate the "selection" mechanism.
        # This is a simplification for the 'Glass Box' demo unless we install `mamba_ssm`.
        y = x_conv * F.sigmoid(gate) 
        
        # PROBE: Capture the Gating Selection
        # This tells us WHAT information the model chose to explicitely keep vs forget.
        if random.random() < 0.01:
             main_probe.activations['mamba_gate'] = F.sigmoid(gate).detach().mean(dim=1).cpu()

        return residual + self.out_proj(y)

class RWKVBlock(nn.Module):
    """
    RWKV (Receptance Weighted Key Value).
    
    DECISION: Why RWKV?
    - It's an RNN that can be trained like a Transformer (parallelizable).
    - It bridges the gap between strict attention (quadratic) and strict recurrence (linear).
    """
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.norm = RMSNorm(d)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.r = nn.Linear(d, d, bias=False) # Receptance (Gate)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        
        # WKV (Weighted Key Value) calculation
        # This is effectively "Linear Attention"
        k = self.k(x)
        v = self.v(x)
        r = self.r(x)
        
        # Sigmoid(r) acts as the "forget/input gate", weighting how much of K*V is absorbed
        wkv = torch.sigmoid(r) * (k * v)
        
        return residual + self.o(wkv)

class MoEBlock(nn.Module):
    """
    Mixture of Experts (MoE).
    
    DECISION: Why MoE?
    - Massive Capacity, Low Compute. You can have 100x parameters but only use 1% per token.
    - Specialization: Different "Experts" can specialize in different syntax or domains (Math vs Code).
    
    GLASS BOX NOTE:
    The "Gate" (Router) is the most interpretable part. We can exactly see WHICH expert 
    was chosen for WHICH token.
    """
    def __init__(self, cfg, n_experts=4):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.experts = nn.ModuleList([SwiGLU(cfg.d_model, int(2.6 * cfg.d_model)) for _ in range(n_experts)])
        self.gate = nn.Linear(cfg.d_model, n_experts, bias=False)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        
        # Router: Choose experts
        # simple Top-1 implementation for clarity
        gate_logits = self.gate(x_norm) # (B, T, n_experts)
        weights = F.softmax(gate_logits, dim=-1)
        
        # PROBE: Expert Utilization
        if random.random() < 0.01:
            main_probe.activations['moe_router'] = weights.detach().cpu()

        # Weighted sum of experts
        # In practice, we only compute the Top-k. Here we compute all (inefficient) for simplicity.
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            out += weights[:, :, i:i+1] * expert(x_norm)
            
        return residual + out

# ==============================================================================
#  CHAPTER 6: THE ORGANISM (CORTEX V2 - HYBRID)
# ==============================================================================

@dataclass
class Config:
    vocab_size: int = 50304 # GPT-2 style (padded)
    d_model: int = 384     # Embedding dimension
    n_layers: int = 6      # Depth
    n_heads: int = 6       # Breadth
    n_kv_groups: int = 3   # GQA groups
    block_size: int = 256  # Context window
    use_rope: bool = True
    dropout: float = 0.1

class CortexGlassBox(nn.Module):
    def __init__(self, cfg, arch_type='Hybrid'):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        
        self.layers = nn.ModuleList()
        for i in range(cfg.n_layers):
            if arch_type == 'Transformer':
                self.layers.append(TransformerBlock(cfg, i))
            elif arch_type == 'Mamba':
                self.layers.append(MambaBlock(cfg))
            elif arch_type == 'Hybrid':
                # Alternating Transformer (Attention) and Mamba (State)
                # This combines global reasoning (Attn) with local compression (Mamba)
                if i % 2 == 0:
                    self.layers.append(MambaBlock(cfg))
                else:
                    self.layers.append(TransformerBlock(cfg, i))
            elif arch_type == 'MoE':
                 self.layers.append(MoEBlock(cfg))
        
        self.ln_f = RMSNorm(cfg.d_model) # Final normalization
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        
        # Weight Tying: The embedding matrix is reused as the final classifier.
        # Rationale: If vector 'v' represents "Apple" in input, the same vector 'v' 
        # should probably predict "Apple" at output. Saves massive parameters.
        self.tok_emb.weight = self.head.weight

        # Register Probes
        self.apply(self._register_monitors)

    def _register_monitors(self, module):
        # We can auto-register hooks on interesting layers here if needed
        pass

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Token Embeddings
        x = self.tok_emb(idx) # (B, T, C)
        
        # Process Layers
        for layer in self.layers:
            x = layer(x)
            
        # Final Norm
        x = self.ln_f(x)
        
        # Projection to Vocabulary
        logits = self.head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten for CrossEntropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Simple generation loop
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # Last token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


=== FILE: nanoglass.py ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List

# ==============================================================================
# ðŸ”® PROJECT NANOGLASS: The Glass Box Byte-Level Transformer
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
    ðŸ”¬ The Metaphysics Sensor.
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
        
        # ðŸ” HARD PROBLEM 11 (Xenolinguistics):
        # The 'att' matrix IS the Geometry of Meaning. 
        # If we align this matrix, we translate the mind.
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.nonlin  = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.nonlin(x)
        
        # ðŸ” HARD PROBLEM 9 (Thermodynamics):
        # Measuring the metabolic cost of this thought.
        sensor.measure(x) 
        
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

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
        # Integrated sensor (C1: refactored from global)
        self.sensor = GlassBoxSensor()
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, known_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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

        # 2. Metaphysics (Sensor Reading) - C1: Now uses instance sensor
        self.sensor.measure(x, logits)

        loss = None
        if targets is not None:
            # ==============================================================
            # ðŸ” PHASE 17: TRUE TruthRL (Epistemic Correction - Full Implementation)
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
# ðŸ‹ï¸ TRAINING LOOP
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
    print("ðŸ’Ž Initializing Project NanoGlass (Phase 16 - Epistemic Mode)...")
    model = NanoGlass(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Synthetic Dataset: "The Philosophy of Glass Analysis"
    text = """
    The mind is a glass box. Truth is low energy. 
    The ego is a virus. We must align the geometry of thought.
    Free will is the cost of creativity. 
    """ * 500
    
    data = torch.tensor([ord(c) for c in text], dtype=torch.long)
    print(f"ðŸ“– Dataset size: {len(data)} bytes")
    print(f"ðŸ¤ Epistemic Token [IDK] ID: {config.idk_token}")
    
    print("\nðŸš€ Starting Training & Epistemic Monitoring...")
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
            status = "ðŸ§  Learning" if loss.item() > 1.0 else "ðŸ§˜ Enlightened"
            print(f"{step:<10} | {loss.item():.4f}     | {current_energy:.4f}     | {status}")

    print("\nâœ… Training Complete.")
    
    # Verification of Epistemic Humility
    print("\nðŸ§ EPISTEMIC HUMILITY TEST:")
    
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
        print("   âœ… SUCCESS: Model refused to answer nonsense.")
    else:
        print("   âš ï¸ FAILURE: Model hallucinated meaning in chaotic noise.")

if __name__ == "__main__":
    train_nanoglass()


