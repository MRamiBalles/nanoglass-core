import torch
import torch.nn as nn
import torch.nn.functional as F
from llm_glassbox import CortexGlassBox, Config, main_probe

# ==============================================================================
#  PART 1: THE MICROSCOPE (SPARSE AUTOENCODER)
# ==============================================================================

class SparseAutoencoder(nn.Module):
    """
    The 'Microscope' for our Glass Box.
    
    Research Goal (RQ1): Decompose Polysemantic Neurons.
    Input: Dense Activation Vector (d_model size, e.g., 384)
    Hidden: Sparse Features (Expansion factor * d_model, e.g., 384 * 4)
    Output: Reconstructed Vector
    
    Constraint: L1 Penalty on hidden state forces 'Sparsity'.
    Result: Only a few 'concepts' fire for any given input.
    """
    def __init__(self, d_model, expansion_factor=4, l1_coeff=0.001):
        super().__init__()
        self.d_model = d_model
        self.n_features = d_model * expansion_factor
        self.l1_coeff = l1_coeff
        
        # Encoder: W_enc @ x + b_enc
        self.encoder = nn.Linear(d_model, self.n_features)
        # Decoder: W_dec @ f + b_dec
        # We tie weights often, but separate is standard for SAEs currently to allow feature norm independence
        self.decoder = nn.Linear(self.n_features, d_model)
        
    def forward(self, x):
        # x: (Batch, d_model) captured from the main model's internals
        
        # 1. Encode to higher dimensional space
        # ReLU enforces non-negativity (concepts either exist or don't)
        features = F.relu(self.encoder(x))
        
        # 2. Decode back to original space
        reconstruction = self.decoder(features)
        
        return reconstruction, features

    def loss(self, x, reconstruction, features):
        # MSE Reconstruction Loss (Did we keep the information?)
        mse_loss = F.mse_loss(reconstruction, x)
        
        # L1 Sparsity Penalty (Did we use few concepts?)
        # This is the pressure that forces "crystallization" of symbolic rules.
        l1_loss = self.l1_coeff * features.abs().sum()
        
        return mse_loss + l1_loss

# ==============================================================================
#  PART 2: THE CURRICULUM (ONTOGENY SIMULATION)
# ==============================================================================

class DevelopmentalTrainer:
    """
    Simulates the 3-stage biological learning process (Ontogeny).
    """
    def __init__(self):
        self.cfg = Config()
        self.cortex = CortexGlassBox(self.cfg, arch_type='Hybrid') # The "Brain"
        self.sae = SparseAutoencoder(self.cfg.d_model)             # The "Analyst"
        
    def stage_1_innate_pretraining(self):
        """
        Stage 1: Unsupervised learning on raw text.
        Goal: Form dense, robust representations.
        Analogous to: Sensory processing in infancy.
        """
        print("\n👶 STAGE 1: INNATE KNOWLEDGE (Accessing Dense Representations)")
        # Simulating training...
        print("   > Ingesting 'Shakespeare' corpus...")
        print("   > Optimizing Next-Token Prediction...")
        print("   > Result: High Perplexity, Low Sparsity.")
        
    def stage_2_reinforcement_feedback(self):
        """
        Stage 2: RLHF / Instruction Tuning.
        Goal: Pruning the search space, aligning with constraints.
        Analogous to: Classical conditioning / Social feedback.
        """
        print("\n👦 STAGE 2: REINFORCEMENT & FEEDBACK (Pruning)")
        # Simulating DPO (Direct Preference Optimization)
        print("   > Applying 'Keep/Reject' masks to attention heads...")
        print("   > Updating weights based on Reward Model...")
        print("   > Result: Activation patterns begin to cluster.")

    def stage_3_specialization_moe(self):
        """
        Stage 3: MoE Routing & Expert Formation.
        Goal: Hard specialization of sub-circuits.
        Analogous to: Adulthood cortical module specialization.
        """
        print("\n👨‍🎓 STAGE 3: SPECIALIZATION (Symbolic Crystallization)")
        # Here we would effectively 'sparsify' the FeedForward layers into Experts
        print("   > Cloning FFNs into Experts...")
        print("   > Training Gating Networks (Routers)...")
        print("   > Result: Sparse Autoencoder now detects clear 'Concept Features'.")
        
    def analyze_with_sae(self):
        """
        Runs the SAE on the current Cortex state to measuring 'Sparsity'.
        """
        dummy_input = torch.randn(1, self.cfg.d_model) # Simulated activation
        recon, feats = self.sae(dummy_input)
        
        sparsity = (feats > 0.01).float().mean().item()
        print(f"   📊 SAE PROBE ANALYSIS: Feature Sparsity = {1.0 - sparsity:.2%} (The higher, the more symbolic)")

if __name__ == "__main__":
    trainer = DevelopmentalTrainer()
    
    # Run the Ontogeny
    trainer.stage_1_innate_pretraining()
    trainer.analyze_with_sae()
    
    trainer.stage_2_reinforcement_feedback()
    trainer.analyze_with_sae()
    
    trainer.stage_3_specialization_moe()
    trainer.analyze_with_sae()
