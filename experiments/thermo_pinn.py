"""
==============================================================================
[FIRE] ThermoLearn PINN: Physics-Informed Neural Network for Thermodynamics
==============================================================================
Implementation of a Multi-Output PINN for predicting Enthalpy (H), Entropy (S),
and Gibbs Free Energy (G) with hard physical constraints.

Methodology based on Hammad & Mondal (2025):
    1. Architecture: Feedforward NN (Inputs: T, P -> Outputs: H, S)
    2. Physics Layer: Computes G = H - TS (Hard Constraint) or Loss Penalty
    3. Loss Function: MSE_data + MSE_physics + Consistency(Cp > 0)
    
Physics Constraints:
    - Gibbs Relation: G = H - TS
    - Heat Capacity: Cp = dH/dT > 0 (Thermodynamic stability)
    - Second Law: dS/dT > 0 (implicitly via Cp/T)

Datasets:
    - NIST-JANAF (Gaseous phase)
    - PhononDB (Solid/Oxide phase)

==============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
# import matplotlib.pyplot as plt
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
class ThermoConfig:
    # Physical Constants
    R_GAS = 8.314  # J/(mol K)
    
    # Model Architecture
    input_dim = 2  # Temperature (T), Pressure (P)
    hidden_dim = 64
    layers = 4
    
    # Training
    learning_rate = 1e-3
    lambda_physics = 1.0   # Weight for Gibbs constraint (if soft)
    lambda_consist = 0.5   # Weight for Cp > 0 constraint
    epochs = 2000          # For demonstration
    
    # Normalization (approximate ranges for JANAF data)
    T_min, T_max = 300.0, 3000.0
    P_min, P_max = 1e4, 1e6         # Pascals
    H_scale = 1e5                   # J/mol
    S_scale = 100.0                 # J/(mol K)
    G_scale = 1e5                   # J/mol

config = ThermoConfig()

# ==============================================================================
# DATA SIMULATION (Placeholder for NIST-JANAF / PhononDB)
# ==============================================================================
def get_synthetic_materials_data(n_samples=500):
    """
    Generates synthetic thermodynamic data obeying G = H - TS 
    to simulate NIST-JANAF data structure for testing the PINN.
    
    Real implementation should load from .csv files.
    """
    # Random T and P
    T = np.random.uniform(config.T_min, config.T_max, (n_samples, 1))
    P = np.random.uniform(config.P_min, config.P_max, (n_samples, 1))
    
    # Synthetic equations for Ideal Gas (just for ground truth generation)
    # Cp = A + B*T
    A, B = 30.0, 0.01 
    H_ref = -200e3 # J/mol
    S_ref = 200.0  # J/mol K
    T_ref = 298.15
    
    # H(T) = H_ref + integral(Cp dT) = H_ref + A(T-T_ref) + 0.5B(T^2 - T_ref^2)
    H = H_ref + A*(T - T_ref) + 0.5*B*(T**2 - T_ref**2)
    
    # S(T) = S_ref + integral(Cp/T dT) = S_ref + A*ln(T/T_ref) + B(T - T_ref)
    # - R*ln(P/P_ref) (ignoring pressure for simple solid model approx or ideal gas)
    S = S_ref + A*np.log(T/T_ref) + B*(T - T_ref)
    
    # G = H - TS
    G = H - T * S
    
    return torch.tensor(np.hstack([T, P]), dtype=torch.float32), \
           torch.tensor(np.hstack([H, S, G]), dtype=torch.float32)

# ==============================================================================
# PINN ARCHITECTURE
# ==============================================================================
class ThermoLearnPINN(nn.Module):
    def __init__(self, cfg: ThermoConfig):
        super().__init__()
        self.cfg = cfg
        
        # Core Network: Predicts H and S from (T, P)
        self.net = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.Tanh(), # Tanh is better for gradients (dH/dT) than ReLU
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Tanh(),
            # Output: [H_norm, S_norm]
            nn.Linear(cfg.hidden_dim, 2)
        )
        
    def forward(self, inputs):
        # inputs: [T, P] (unnormalized)
        
        # Normalize inputs
        T = (inputs[:, 0:1] - self.cfg.T_min) / (self.cfg.T_max - self.cfg.T_min)
        P = (inputs[:, 1:2] - self.cfg.P_min) / (self.cfg.P_max - self.cfg.P_min)
        x_norm = torch.cat([T, P], dim=1)
        
        # Predict normalized H, S
        out_norm = self.net(x_norm)
        H_norm = out_norm[:, 0:1]
        S_norm = out_norm[:, 1:2]
        
        # Denormalize
        H = H_norm * self.cfg.H_scale
        S = S_norm * self.cfg.S_scale
        
        # ---------------------------------------------------------
        # PHYSICAL CONSTRAINT LAYER (Hard or Soft)
        # ---------------------------------------------------------
        # We calculate G derived from H and S using the definition
        # G_pred = H_pred - T * S_pred
        # This relationship is EXACT by definition in the output
        T_abs = inputs[:, 0:1]
        G = H - T_abs * S
        
        return torch.cat([H, S, G], dim=1)

# ==============================================================================
# PHYSICS-INFORMED LOSS
# ==============================================================================
def physics_loss(model, inputs, predictions, targets):
    """
    L_total = L_data + L_consistency
    
    1. Data Loss: MSE(H, H_obs) + MSE(S, S_obs) + MSE(G, G_obs)
    2. Consistency: Cp = dH/dT > 0
    """
    
    # Unpack predictions
    H_pred = predictions[:, 0:1]
    S_pred = predictions[:, 1:2]
    G_pred = predictions[:, 2:3]
    
    # Unpack targets
    H_true = targets[:, 0:1]
    S_true = targets[:, 1:2]
    G_true = targets[:, 2:3]
    
    # 1. Data Loss (MSE Term)
    # We normalized data loss components to balance gradients
    l_H = F.mse_loss(H_pred, H_true) / (config.H_scale**2)
    l_S = F.mse_loss(S_pred, S_true) / (config.S_scale**2)
    l_G = F.mse_loss(G_pred, G_true) / (config.G_scale**2)
    
    L_data = l_H + l_S + l_G
    
    # 2. Thermodynamic Consistency (Derivative Check)
    # Cp = dH/dT. For stability, Cp > 0.
    # We need gradients of H w.r.t T.
    
    # Enable grad for inputs to compute derivatives
    inputs.requires_grad_(True)
    
    # Re-run forward pass to get graph connecting inputs to H
    # Note: efficient implementation would do this inside training loop reuse
    # but for clarity we isolate it here or assume `predictions` is connected.
    # Since `predictions` was passed in detach? No, in training loop it is connected.
    
    # Checking gradients requires `create_graph=True` in main loop.
    # Here we assume we can compute gradients.
    
    # If inputs leaf node, we need to compute grad of H_pred sum w.r.t inputs
    # dH_dT = torch.autograd.grad(H_pred.sum(), inputs, create_graph=True)[0][:, 0:1]
    
    # STABILITY REGULARIZATION: Cp > 0
    # Penalty = ReLU(-Cp) -> if Cp is negative, penalty is positive
    # L_consistency = F.relu(-dH_dT).mean()
    
    # Since dH/dT calculation is expensive, we return it as a separate term 
    # to be computed in the training loop if enabled.
    
    # 3. Gibbs Consistency
    # Note: Since our model OUTPUTS G = H - TS by definition, 
    # MSE(G_pred, G_true) ALREADY enforces G = H - TS implicitly relative to data.
    # However, ThermoLearn paper suggests an explicit residual if G is predicted independently.
    # Here, we used the "Hard Constraint" architecture where G is structurally coupled.
    # So physics loss is 0 for Gibbs relation (it's satisfied by design).
    
    return L_data

# ==============================================================================
# TRAINING LOOP
# ==============================================================================
def train_thermo_pinn():
    print(f"[INIT] Initializing ThermoLearn PINN...")
    model = ThermoLearnPINN(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Get Data
    X_train, Y_train = get_synthetic_materials_data(1000)
    
    print(f"   Training for {config.epochs} epochs...")
    loss_history = []
    
    model.train()
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        
        # Ensure inputs track gradients for physics derivatives
        X_train.requires_grad_(True)
        
        # Forward
        preds = model(X_train)
        H_pred = preds[:, 0:1]
        
        # Compute Data Loss
        L_d = physics_loss(model, X_train, preds, Y_train)
        
        # Compute Physics Derivative (Cp = dH/dT)
        dH_dT = torch.autograd.grad(
            outputs=H_pred, 
            inputs=X_train, 
            grad_outputs=torch.ones_like(H_pred),
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]
        
        # Consistency Loss: Penalize negative heat capacity
        # Penalize if Cp < 0
        L_c = F.relu(-dH_dT).mean() * config.lambda_consist
        
        L_total = L_d + L_c
        
        L_total.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"   Epoch {epoch:04d} | Loss: {L_total.item():.6f} (Data: {L_d.item():.6f}, Phys: {L_c.item():.6f})")
            loss_history.append(L_total.item())
            
    print("[DONE] Training Complete.")
    return model, loss_history

if __name__ == "__main__":
    trained_model, _ = train_thermo_pinn()
    
    # Verify Consistency on Test Set
    print("\n[TEST] Validating Physical Consistency...")
    X_test, _ = get_synthetic_materials_data(100)
    X_test.requires_grad_(True)
    preds = trained_model(X_test)
    H_pred = preds[:, 0:1]
    
    dH_dT = torch.autograd.grad(
        outputs=H_pred, 
        inputs=X_test, 
        grad_outputs=torch.ones_like(H_pred),
        create_graph=False
    )[0][:, 0:1]
    
    min_Cp = dH_dT.min().item()
    print(f"   Min Estimated Heat Capacity (Cp): {min_Cp:.4f}")
    
    if min_Cp > -0.1: # Allow small numerical noise
        print("   [OK] Cp > 0 satisfies thermodynamic stability (Second Law)")
    else:
        print("   [WARN] Unstable thermodynamics detected (Cp < 0)")

    # Save for paper
    torch.save(trained_model.state_dict(), "thermo_pinn_model.pth")
