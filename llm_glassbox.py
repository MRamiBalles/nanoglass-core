# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                  🔬 CORTEX-13: GLASS BOX EDITION                         ║
# ║           Deep Interpretability & Architectural Analysis                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

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

from nanoglass_probes import main_probe
from mamba2_ssd import Mamba2Block, Mamba2Config
from nanoglass import GlassBoxSensor


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
    
    ⚠️ CRITICAL WARNING: SIMPLIFIED IMPLEMENTATION ⚠️
    ═══════════════════════════════════════════════════════════════════════════
    This is a PEDAGOGICAL APPROXIMATION of Mamba, NOT the real architecture.
    
    WHAT THIS IMPLEMENTATION DOES:
        - Gated convolution with sigmoid selection (simulates selectivity)
        - Linear complexity O(N) for sequence length
        
    WHAT IT LACKS (vs Real Mamba/Mamba-2):
        - Selective State Space Model (S6) with input-dependent A, B, C matrices
        - Discretization via zero-order hold
        - Hardware-aware parallel scan algorithm
        - Tensor Core optimizations (2-8x speedup in Mamba-2)
    
    FOR PRODUCTION USE:
        pip install mamba-ssm
        from mamba_ssm import Mamba2
        self.mamba = Mamba2(d_model=cfg.d_model, d_state=16, d_conv=4)
    
    References:
        - Mamba: Gu & Dao (2023) arXiv:2312.00752
        - Mamba-2 SSD: Dao & Gu (2024) arXiv:2405.21060
        - Jamba (AI21): Production hybrid architecture
    ═══════════════════════════════════════════════════════════════════════════
    
    EDUCATIONAL NOTES:
    1. Linear Complexity: Mamba scales O(N), unlike Transformer's O(N^2).
    2. Recurrent State: Compressed "hidden state" passes information forward,
       using structured matrices (HiPPO) for long-term memory.
    
    GLASS BOX NOTE:
    The 'hidden_state' is valuable for symbolic extraction. Unlike K/V cache
    which stores specific past tokens, the Mamba state stores a *compressed
    representation* of the entire context.
    """
    
    # Flag to indicate this is a real implementation
    IS_SIMPLIFIED = False
    
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        
        # [INTEGRATION] Use the faithful Mamba-2 SSD implementation
        m2_cfg = Mamba2Config(
            d_model=cfg.d_model,
            d_state=64,      # Optimized for CPU/GlassBox
            d_conv=4,
            expand=2
        )
        self.mamba = Mamba2Block(m2_cfg)
        
    def forward(self, x):
        # x: (B, T, D)
        # Mamba2Block handles normalization and residual internally if needed,
        # but here we follow the llm_glassbox convention.
        y = self.mamba(x)
        
        # PROBE: Capture activations for interpretability
        if random.random() < 0.01:
            main_probe.activations['mamba_is_real'] = True
            main_probe.activations['mamba_type'] = "Mamba-2 SSD"

        return y

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

# ==============================================================================
#  CHAPTER 6: THE ORGANISM (CORTEX V2 - HYBRID JAMBA ARCHITECTURE)
# ==============================================================================

try:
    from layers.moe_granular import GranularMoE, MoEConfig
    from mamba2_ssd import Mamba2Block, Mamba2Config
except ImportError:
    # Use mocks/placeholders if dependencies missing during dev
    pass

@dataclass
class Config:
    vocab_size: int = 50304 # GPT-2 style (padded)
    d_model: int = 384     # Embedding dimension
    n_layers: int = 16     # Depth (Increased for hybrid capacity)
    n_heads: int = 6       # Breadth
    n_kv_groups: int = 3   # GQA groups
    block_size: int = 1024 # Increased context for Mamba efficiency
    use_rope: bool = False # Jamba does not use explicit PE in attention
    dropout: float = 0.1
    # MoE Settings
    moe_experts: int = 64
    moe_top_k: int = 6
    moe_shared: int = 2
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # SEAL / [IDK] token settings
    idk_token: int = 50303 # Last token in GPT-2 style vocab

class NanoGlassHybridBlock(nn.Module):
    """
    Unified Hybrid Block implementing Jamba 1.5 Architecture.
    
    Structure:
    1. Mixer Layer (Ratio 1:7)
       - Layer 0-6: Mamba-2 SSD (Compression)
       - Layer 7: Attention (Global Sync) - No RoPE
       - Pre-norm: RMSNorm
       
    2. Processing Layer (Ratio 1:1)
       - Even Layers: Dense MLP
       - Odd Layers: Granular MoE
       - Pre-norm: RMSNorm
    """
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        # 1. Determine Mixer Type (1:7 Ratio)
        # Layers 7, 15, 23... are Attention. All others Mamba.
        # This ensures we start with Mamba compression.
        self.is_attention = (layer_idx + 1) % 8 == 0
        
        self.mixer_norm = RMSNorm(cfg.d_model)
        
        if self.is_attention:
            # Attention Layer (Global Reasoning)
            # CAUTION: Jamba disables RoPE/PE in attention layers
            self.mixer = GQA(cfg.d_model, cfg.n_heads, cfg.n_kv_groups, use_rope=False)
        else:
            # Mamba-2 SSD Layer (Context Compression)
            mamba_cfg = Mamba2Config(
                d_model=cfg.d_model,
                d_state=128, # Standard state size
                expand=2
            )
            self.mixer = Mamba2Block(mamba_cfg)
            
        # 2. Determine MLP Type (MoE every 2 layers)
        # Even: MLP, Odd: MoE (Nemotron-3 pattern)
        # e.g., L0=MLP, L1=MoE, L2=MLP...
        self.is_moe = (layer_idx % 2 != 0)
        
        self.mlp_norm = RMSNorm(cfg.d_model)
        
        if self.is_moe:
            # Granular MoE (Specialized Processing)
            moe_cfg = MoEConfig(
                embed_dim=cfg.d_model,
                num_experts=cfg.moe_experts,
                num_shared=cfg.moe_shared,
                top_k=cfg.moe_top_k
            )
            self.mlp = GranularMoE(moe_cfg)
        else:
            # Dense MLP (General Knowledge Highway)
            self.mlp = SwiGLU(cfg.d_model, int(2.6 * cfg.d_model))

    def forward(self, x):
        # 1. Mixer Path (Mamba or Attention)
        residual = x
        x = self.mixer_norm(x)
        
        # Check if Attention requires layer_id (GQA impl specific)
        if self.is_attention:
            x = self.mixer(x, layer_id=self.layer_idx)
        else:
            x = self.mixer(x)
            
        x = residual + x
        
        # 2. MLP/MoE Path
        residual = x
        x = self.mlp_norm(x)
        
        if self.is_moe:
            # MoE returns (output, aux_loss). We discard aux_loss for inference/simple fwd
            # In training loop, we'd capture it.
            # Wrapper to handle tuple return if needed, but GranularMoE returns tuple.
            x_out, _ = self.mlp(x, return_aux_loss=False)
            x = x_out
        else:
            x = self.mlp(x)
            
        x = residual + x
        return x

class CortexGlassBox(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        
        # Hybrid Layers
        self.layers = nn.ModuleList()
        for i in range(cfg.n_layers):
            self.layers.append(NanoGlassHybridBlock(cfg, i))
        
        self.ln_f = RMSNorm(cfg.d_model) # Final normalization
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        
        # Weight Tying
        self.tok_emb.weight = self.head.weight
        
        # [STABILITY] Initialize head with lower variance
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        
        # [IDK TOKEN] Semantic initialization to prevent numerical collapse
        # Use mean of common "uncertainty" words in ASCII range
        # Characters: 'u','n','c','e','r','t','a','i','n' = [117,110,99,101,114,116,97,105,110]
        if hasattr(cfg, 'idk_token'):
            with torch.no_grad():
                uncertainty_chars = [117, 110, 99, 101, 114, 116, 97, 105, 110]  # "uncertain"
                mean_emb = self.tok_emb.weight[uncertainty_chars].mean(dim=0)
                self.tok_emb.weight[cfg.idk_token] = mean_emb
        
        # [ACTIVATION TRACKING] For auxiliary loss
        self.activation_accumulator = []
        
        # [INTEGRATION] Unified GlassBox Sensor
        self.sensor = GlassBoxSensor()
        
    def _register_monitors(self, module):
        # We can auto-register hooks on interesting layers here if needed
        pass

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Token Embeddings
        x = self.tok_emb(idx) # (B, T, C)
        
        # Process Layers with activation tracking
        for layer in self.layers:
            x = layer(x)
            # Track activations for auxiliary loss
            self.activation_accumulator.append(x.detach().clone())
            
        # Final Norm
        x = self.ln_f(x)
        
        # Projection to Vocabulary
        logits = self.head(x) # (B, T, vocab_size)

        # [INTEGRATION] Unified Sensor Reading
        self.sensor.measure(x, logits)

        loss = None
        activation_loss = None
        if targets is not None:
            # Flatten for CrossEntropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # [ACTIVATION LOSS] Auxiliary loss for Mamba stability (Jamba 1.5 technique)
            # Penalize large activations to prevent numerical instability
            if self.activation_accumulator:
                act_tensor = torch.cat([a.flatten() for a in self.activation_accumulator])
                activation_loss = 1e-5 * (act_tensor ** 2).mean()
                loss = loss + activation_loss
                self.activation_accumulator = []  # Reset

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
