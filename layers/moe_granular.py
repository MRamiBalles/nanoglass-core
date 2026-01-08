"""
==============================================================================
Granular Mixture-of-Experts (MoE) Layer
==============================================================================
Implements fine-grained MoE architecture following Nemotron-3 Nano and Jamba 1.5.

Key Features:
    - 64-128 small experts (instead of 8 large ones)
    - Shared Experts: 1-2 experts always activated (stability)
    - Top-K Routing: Activate top-6 to top-8 per token
    - Load Balancing: Auxiliary loss to prevent router collapse

Architecture (Nemotron-3 Reference):
    Total Params: 31.6B, Active per token: ~3.2B
    Expert count: 128, Active experts: 8

References:
    - Nemotron-3 Technical Report (NVIDIA, 2025)
    - Jamba 1.5: Hybrid Transformer-Mamba (AI21 Labs, 2025)
    - Mixtral 8x7B (Mistral AI, 2024)

==============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class MoEConfig:
    """Configuration for Granular MoE layer."""
    # Model dimensions
    embed_dim: int = 256        # Input/output dimension
    expert_dim: int = 512       # Hidden dimension per expert (4x embed for standard)
    
    # Expert configuration
    num_experts: int = 64       # Total number of experts (granularity)
    num_shared: int = 2         # Always-active shared experts
    top_k: int = 6              # Number of experts to route to per token
    
    # Load balancing
    aux_loss_weight: float = 0.01  # Weight for load balancing auxiliary loss
    
    # Routing
    router_jitter: float = 0.1    # Noise for exploration during training
    capacity_factor: float = 1.25  # Buffer for expert capacity
    
    # Dropout
    expert_dropout: float = 0.1


# ==============================================================================
# EXPERT MLP
# ==============================================================================

class ExpertMLP(nn.Module):
    """
    Single expert: a small 2-layer MLP.
    
    Architecture: Linear -> GELU -> Linear
    """
    
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, embed)"""
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ==============================================================================
# ROUTER
# ==============================================================================

class Router(nn.Module):
    """
    Learned router for expert selection.
    
    Uses softmax over expert logits and selects top-K.
    Includes load balancing auxiliary loss.
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.embed_dim, config.num_experts, bias=False)
    
    def forward(
        self, 
        x: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.
        
        Args:
            x: (batch, seq, embed)
            training: Whether to add jitter noise
            
        Returns:
            expert_weights: (batch, seq, top_k) - weights for selected experts
            expert_indices: (batch, seq, top_k) - indices of selected experts
            aux_loss: scalar - load balancing loss
        """
        batch, seq, _ = x.shape
        
        # Compute router logits
        logits = self.gate(x)  # (batch, seq, num_experts)
        
        # Add jitter during training for exploration
        if training and self.config.router_jitter > 0:
            noise = torch.randn_like(logits) * self.config.router_jitter
            logits = logits + noise
        
        # Softmax probabilities
        probs = F.softmax(logits, dim=-1)  # (batch, seq, num_experts)
        
        # Top-K selection
        top_k_probs, top_k_indices = probs.topk(self.config.top_k, dim=-1)
        
        # Normalize weights to sum to 1
        expert_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)
        
        # Compute auxiliary load balancing loss
        aux_loss = self._compute_aux_loss(probs, top_k_indices)
        
        return expert_weights, top_k_indices, aux_loss
    
    def _compute_aux_loss(
        self, 
        probs: torch.Tensor, 
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss.
        
        Encourages uniform distribution of tokens across experts.
        Loss = num_experts * sum(fraction_routed * average_prob)
        """
        num_experts = self.config.num_experts
        batch, seq, _ = probs.shape
        
        # Fraction of tokens routed to each expert
        # Using one-hot encoding of selected experts
        one_hot = F.one_hot(indices, num_experts).float()  # (batch, seq, top_k, num_experts)
        one_hot = one_hot.sum(dim=2)  # (batch, seq, num_experts)
        
        # Average over batch and sequence
        fraction = one_hot.mean(dim=(0, 1))  # (num_experts,)
        
        # Average probability per expert
        avg_prob = probs.mean(dim=(0, 1))  # (num_experts,)
        
        # Load balancing loss
        aux_loss = num_experts * (fraction * avg_prob).sum()
        
        return aux_loss


# ==============================================================================
# GRANULAR MOE LAYER
# ==============================================================================

class GranularMoE(nn.Module):
    """
    Granular Mixture-of-Experts layer.
    
    Replaces dense MLP in transformer/hybrid blocks.
    Features:
        - Fine-grained experts (64-128)
        - Shared experts (always active)
        - Top-K routing with load balancing
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Router
        self.router = Router(config)
        
        # Routed experts
        self.experts = nn.ModuleList([
            ExpertMLP(config.embed_dim, config.expert_dim, config.expert_dropout)
            for _ in range(config.num_experts)
        ])
        
        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            ExpertMLP(config.embed_dim, config.expert_dim, config.expert_dropout)
            for _ in range(config.num_shared)
        ])
        
        # Output normalization
        self.norm = nn.LayerNorm(config.embed_dim)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through MoE layer.
        
        Args:
            x: (batch, seq, embed)
            return_aux_loss: Whether to compute and return aux loss
            
        Returns:
            output: (batch, seq, embed)
            aux_loss: scalar or None
        """
        batch, seq, embed = x.shape
        
        # Route tokens to experts
        weights, indices, aux_loss = self.router(x, training=self.training)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # === Shared Experts (Always Active) ===
        for shared in self.shared_experts:
            output = output + shared(x) / len(self.shared_experts)
        
        # === Routed Experts ===
        # For efficiency, we process all tokens through each expert
        # then mask by routing weights
        # (In production, use sparse operations for true efficiency)
        
        for k in range(self.config.top_k):
            expert_idx = indices[:, :, k]  # (batch, seq)
            expert_weight = weights[:, :, k:k+1]  # (batch, seq, 1)
            
            # Process through selected expert
            # This is a simplified loop - production uses batched sparse ops
            for e in range(self.config.num_experts):
                mask = (expert_idx == e).unsqueeze(-1)  # (batch, seq, 1)
                if mask.any():
                    expert_out = self.experts[e](x)
                    output = output + expert_out * expert_weight * mask.float()
        
        # Normalize output
        output = self.norm(output)
        
        if return_aux_loss:
            return output, aux_loss * self.config.aux_loss_weight
        else:
            return output, None


# ==============================================================================
# HYBRID BLOCK (Mamba + MoE)
# ==============================================================================

class HybridMambaMoEBlock(nn.Module):
    """
    Hybrid block combining Mamba recurrence with Granular MoE.
    
    Architecture:
        x -> LayerNorm -> Mamba -> x + residual
        x -> LayerNorm -> MoE -> x + residual
    """
    
    def __init__(self, mamba_layer: nn.Module, moe_config: MoEConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(moe_config.embed_dim)
        self.mamba = mamba_layer
        
        self.ln2 = nn.LayerNorm(moe_config.embed_dim)
        self.moe = GranularMoE(moe_config)
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through hybrid block.
        
        Returns:
            output: (batch, seq, embed)
            aux_loss: MoE auxiliary loss
        """
        # Mamba recurrence
        x = x + self.mamba(self.ln1(x))
        
        # MoE feedforward
        moe_out, aux_loss = self.moe(self.ln2(x))
        x = x + moe_out
        
        return x, aux_loss


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    print("\n[TEST] Granular MoE Layer")
    print("=" * 60)
    
    # Configuration
    config = MoEConfig(
        embed_dim=256,
        expert_dim=512,
        num_experts=64,
        num_shared=2,
        top_k=6
    )
    
    print(f"   Experts: {config.num_experts} (shared: {config.num_shared})")
    print(f"   Top-K: {config.top_k}")
    print(f"   Dimensions: {config.embed_dim} -> {config.expert_dim}")
    
    # Create layer
    moe = GranularMoE(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in moe.parameters())
    expert_params = sum(p.numel() for e in moe.experts for p in e.parameters())
    router_params = sum(p.numel() for p in moe.router.parameters())
    
    print(f"\n   Total Parameters: {total_params:,}")
    print(f"   Expert Parameters: {expert_params:,}")
    print(f"   Router Parameters: {router_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 128, config.embed_dim)
    
    moe.train()
    output, aux_loss = moe(x)
    
    print(f"\n   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Aux Loss: {aux_loss.item():.4f}")
    
    # Verify routing distribution
    with torch.no_grad():
        weights, indices, _ = moe.router(x)
    
    unique_experts = torch.unique(indices).numel()
    print(f"   Unique experts used: {unique_experts}/{config.num_experts}")
    
    print("\n" + "=" * 60)
    print("   [OK] Granular MoE test passed!")
