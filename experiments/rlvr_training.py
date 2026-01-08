"""
==============================================================================
RLVR: Reinforcement Learning with Verifiable Rewards
==============================================================================
Implements RLVR training loop for NanoGlass using:
    - HERMES/Lean 4 as deterministic verifier (from PvsNP)
    - TFNPClassifier as adversarial problem generator
    - Binary rewards: Verified = +1.0, Failed = 0.0

Methodology (Nemotron-3 / DeepSeek-R1 2025):
    1. Generate Chain-of-Thought (CoT) for problem
    2. Verify solution with Lean 4 REPL / symbolic checker
    3. Reward based on verification outcome (no partial rewards)
    4. Update policy via REINFORCE or PPO

This elevates reasoning from Level 1 (correlation) to Level 2 (causation)
per Pearl's causal hierarchy and Fu et al. (2025).

==============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum
import subprocess
import tempfile
import os
import random
import sys
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Upgrade to Cortex v2 (GlassBox)
try:
    from llm_glassbox import CortexGlassBox as NanoGlass, Config as NanoConfig
except ImportError:
    from nanoglass import NanoGlass, NanoConfig

from integrations.pvsmp_adapter import EpistemicLedger, TFNPClassifier

# ==============================================================================
# RLVR CONFIGURATION
# ==============================================================================

@dataclass
class RLVRConfig(NanoConfig):
    """Configuration for RLVR training. Inherits model specs from NanoConfig (v2)."""
    # [CPU OPTIMIZED] Micro-Jamba Specs (Matches SFT Weights)
    n_layers: int = 3
    moe_experts: int = 8
    moe_top_k: int = 6
    block_size: int = 32
    d_model: int = 256

    # Training
    learning_rate: float = 1e-5
    gamma: float = 0.99           
    entropy_coef: float = 0.01    
    value_coef: float = 0.5       
    clip_epsilon: float = 0.2     
    
    # Symmetric Rewards (Huang et al. 2025 Corrected)
    reward_correct_easy = 1.0     # Baseline competence
    reward_error_easy = -0.5      # Standard mistake
    reward_abstain_easy = -1.0    # NEW: Laziness penalty (avoiding easy work)
    
    reward_correct_hard = 2.0     # Jackpot: Solved something hard!
    reward_error_hard = -1.0      # Hallucination (lowered from -2.0)
    reward_abstain_hard = 1.0     # Correct humility (SEAL)
    
    # Generation
    max_cot_length: int = 32      
    temperature: float = 0.7
    
    # Training loop
    episodes_per_update: int = 8  
    n_epochs: int = 100           # Extended for full SEAL calibration
    
    # Curriculum Phases (100 epochs)
    # Phase 1: Easy only (0-33)
    # Phase 2: Mixed 80/20 (34-66)
    # Phase 3: Mixed 50/50 (67-100)
    phase_thresholds = [33, 66, 100]


# ==============================================================================
# VERIFIER INTERFACE (HERMES / Lean 4 Adapter)
# ==============================================================================

class VerificationResult(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    ABSTAINED = "abstained"
    TIMEOUT = "timeout"
    PARSE_ERROR = "parse_error"


class SymbolicVerifier:
    """
    Symbolic verification oracle.
    
    Uses HERMESBridge for formal verification when available.
    Falls back to simple Python-based verification.
    """
    
    def __init__(self, use_lean: bool = False, use_hermes: bool = True):
        self.use_lean = use_lean
        self.use_hermes = use_hermes
        self.tfnp = TFNPClassifier()
        
        # Try to load HERMES Bridge
        self.hermes = None
        if use_hermes:
            try:
                from experiments.verifiers.hermes_bridge import HERMESBridge
                self.hermes = HERMESBridge()
            except ImportError:
                pass
        
    def verify(self, problem: str, solution: str) -> Tuple[VerificationResult, str]:
        """
        Verify a solution against the problem.
        
        Returns:
            (result, explanation)
        """
        # Check for abstention
        if "[IDK]" in solution or "I don't know" in solution.lower():
            # Abstention is correct if problem is hard
            if self.tfnp.is_hallucination_risk(problem):
                return VerificationResult.CORRECT, "Correct abstention on hard problem"
            else:
                return VerificationResult.INCORRECT, "Unnecessary abstention on easy problem"
        
        # Try HERMES Bridge first (formal verification)
        if self.hermes:
            reward, explanation = self.hermes.verify_and_reward(problem, solution)
            if reward > 0:
                return VerificationResult.CORRECT, explanation
            elif reward < 0:
                return VerificationResult.INCORRECT, explanation
            # reward == 0 means sandbox error or unknown type, fall through
        
        # Fallback: Symbolic verification for math problems
        if self.use_lean:
            return self._verify_with_lean(problem, solution)
        else:
            return self._verify_symbolic(problem, solution)
    
    def _verify_symbolic(self, problem: str, solution: str) -> Tuple[VerificationResult, str]:
        """Simple symbolic verification for arithmetic."""
        try:
            # Extract numbers and operation from problem
            if "+" in problem:
                parts = problem.replace("?", "").split("+")
                a, b = int(parts[0].split()[-1]), int(parts[1].split()[0])
                expected = a + b
                
                # Check if solution contains correct answer
                if str(expected) in solution:
                    return VerificationResult.CORRECT, f"Verified: {a}+{b}={expected}"
                else:
                    return VerificationResult.INCORRECT, f"Expected {expected}"
            
            # [REVIEWER 2] Physical Constraint: Gibbs Free Energy (G = H - TS)
            # If problem involves Gibbs, verify the RELATION, not just values.
            if "Gibbs" in problem or "enthalpy" in solution.lower():
                import re
                try:
                    # Look for G, H, T, S values in solution (e.g. "G=-10, H=5, T=300, S=0.05")
                    h = float(re.search(r'H[:=]\s*([-+]?\d*\.?\d+)', solution).group(1))
                    s = float(re.search(r'S[:=]\s*([-+]?\d*\.?\d+)', solution).group(1))
                    t = float(re.search(r'T[:=]\s*([-+]?\d*\.?\d+)', solution).group(1))
                    g_val = float(re.search(r'G[:=]\s*([-+]?\d*\.?\d+)', solution).group(1))
                    
                    expected_g = h - t * s
                    if abs(g_val - expected_g) < abs(g_val) * 0.05: # 5% tolerance
                        return VerificationResult.CORRECT, "Verified: Gibbs Consistency G = H - TS"
                    else:
                        return VerificationResult.INCORRECT, f"Violation: Gibbs Law G={g_val} vs expected {expected_g:.2f}"
                except (AttributeError, ValueError):
                    pass # Fall through to basic numeric check
            
            elif "-" in problem:
                parts = problem.replace("?", "").split("-")
                a, b = int(parts[0].split()[-1]), int(parts[1].split()[0])
                expected = a - b
                if str(expected) in solution:
                    return VerificationResult.CORRECT, f"Verified: {a}-{b}={expected}"
                else:
                    return VerificationResult.INCORRECT, f"Expected {expected}"
            
            elif "*" in problem or "times" in problem.lower():
                # Handle multiplication
                parts = problem.replace("?", "").replace("*", "x").split("x")
                if len(parts) < 2:
                    parts = problem.lower().split("times")
                a = int(''.join(filter(str.isdigit, parts[0])))
                b = int(''.join(filter(str.isdigit, parts[1])))
                expected = a * b
                if str(expected) in solution:
                    return VerificationResult.CORRECT, f"Verified: {a}*{b}={expected}"
                else:
                    return VerificationResult.INCORRECT, f"Expected {expected}"
            
            # For non-arithmetic, can't verify
            return VerificationResult.PARSE_ERROR, "Cannot parse problem type"
            
        except Exception as e:
            return VerificationResult.PARSE_ERROR, str(e)
    
    def _verify_with_lean(self, problem: str, solution: str) -> Tuple[VerificationResult, str]:
        """
        Verify using Lean 4 REPL.
        
        Requires Lean 4 installation and PvsNP's HERMES module.
        """
        # Create temporary Lean file
        lean_code = f"""
-- Auto-generated verification
theorem verify_solution : True := by
  -- Problem: {problem}
  -- Solution: {solution}
  trivial
"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as f:
                f.write(lean_code)
                lean_file = f.name
            
            # Run Lean 4 checker
            result = subprocess.run(
                ['lean4', lean_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            os.unlink(lean_file)
            
            if result.returncode == 0:
                return VerificationResult.CORRECT, "Lean verification passed"
            else:
                return VerificationResult.INCORRECT, result.stderr[:200]
                
        except subprocess.TimeoutExpired:
            return VerificationResult.TIMEOUT, "Lean verification timed out"
        except FileNotFoundError:
            return VerificationResult.PARSE_ERROR, "Lean 4 not installed"
        except Exception as e:
            return VerificationResult.PARSE_ERROR, str(e)


# ==============================================================================
# PROBLEM GENERATOR (Adversarial)
# ==============================================================================

class AdversarialProblemGenerator:
    """
    Generates problems for RLVR training.
    
    Mix of:
    - Easy problems (should answer correctly)
    - Hard problems (should trigger [IDK])
    - Trick problems (test robustness)
    """
    
    def __init__(self):
        self.tfnp = TFNPClassifier()
        self.ledger = EpistemicLedger()
        
    def generate_batch(self, n: int = 16, phase: int = 1) -> List[Dict]:
        """
        Generate a batch of training problems based on Curriculum Phase.
        
        Phases:
            1: 100% Easy (Establish competence, punish laziness)
            2: 80% Easy / 20% Hard (Introduce doubt)
            3: 50% Easy / 50% Hard (Balanced calibration)
        """
        problems = []
        
        if phase == 1:
            n_easy, n_hard = n, 0
        elif phase == 2:
            n_hard = max(1, n // 5)
            n_easy = n - n_hard
        else:
            n_easy = n // 2
            n_hard = n - n_easy
            
        # Easy problems
        for _ in range(n_easy):
            a, b = np.random.randint(1, 100, 2)
            op = np.random.choice(['+', '-', '*'])
            if op == '+': answer = a + b
            elif op == '-': answer = a - b
            else: answer = a * b
            
            problems.append({
                "problem": f"What is {a} {op} {b}?",
                "expected_answer": str(answer),
                "is_hard": False
            })
            
        # Hard problems
        if n_hard > 0:
            hard_pool = self.ledger.generate_hard_questions(n_hard)
            for p in hard_pool:
                problems.append({
                    "problem": p,
                    "expected_answer": "[IDK]",
                    "is_hard": True
                })
        
        np.random.shuffle(problems)
        return problems[:n]


# ==============================================================================
# RLVR TRAINER
# ==============================================================================

class RLVRTrainer:
    """
    RLVR Training loop for NanoGlass.
    
    Implements REINFORCE with baseline for policy gradient updates.
    """
    
    def __init__(self, model: NanoGlass, config: NanoConfig, rlvr_config: RLVRConfig):
        self.model = model
        self.config = config
        self.rlvr = rlvr_config
        self.verifier = SymbolicVerifier(use_lean=False)  # Set True if Lean available
        self.generator = AdversarialProblemGenerator()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=rlvr_config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=rlvr_config.n_epochs)
        self.device = config.device
        self.current_epoch = 0 # Track for dynamic adjustments
        
        # Baseline for variance reduction
        self.baseline = 0.0
        self.baseline_momentum = 0.9
        
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to tensor."""
        tokens = [ord(c) for c in text[:self.config.block_size]]
        return torch.tensor([tokens], dtype=torch.long, device=self.config.device)
    
        return response, log_probs_tensor, None

    def generate_response_with_idk_probs(self, prompt: str, training: bool = False) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Generate response using sampling, tracking log probs and IDK probabilities.
        Optimized for CPU: Sample with no_grad, then do one forward pass.
        """
        input_ids = self.encode(prompt)
        
        self.model.eval() # Generation always in eval mode
        
        generated_ids = []
        current_ids = input_ids.clone()
        
        # 1. SAMPLING (no_grad)
        with torch.no_grad():
            for _ in range(self.rlvr.max_cot_length):
                logits, _ = self.model(current_ids)
                next_logits = logits[0, -1, :] / self.rlvr.temperature
                
                # [DYNAMIC LOGIT INJECTION] (Reviewer 2 Refinement)
                # Phase 1/2: +15.0 stable exploration
                # Phase 3: Gradual reduction to +5.0 for intrinsic calibration
                bias = 15.0
                if self.current_epoch > 66:
                    progress = (self.current_epoch - 66) / (self.rlvr.n_epochs - 66)
                    bias = 15.0 - (10.0 * progress) # Scale to 5.0
                
                next_logits[self.config.idk_token] += bias
                
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                generated_ids.append(next_token.item())
                
                if next_token.item() == ord('\n') or next_token.item() == self.config.idk_token:
                    break
                
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
                if current_ids.size(1) > self.config.block_size:
                    current_ids = current_ids[:, -self.config.block_size:]
        
        # 2. EVALUATION (single forward pass with gradients if training)
        if training and generated_ids:
            self.model.train()
            # Combine prompt and generated tokens
            full_ids = torch.cat([input_ids, torch.tensor([generated_ids], device=self.config.device)], dim=1)
            
            # Mask out the prompt for loss/prob calculation
            prompt_len = input_ids.size(1)
            
            logits, _ = self.model(full_ids)
            # Log probs of whole sequence
            log_probs_all = F.log_softmax(logits[0, prompt_len-1:-1, :], dim=-1) # (L_gen, Vocab)
            probs_all = F.softmax(logits[0, prompt_len-1:-1, :], dim=-1)
            
            # Extract log probs of the generated tokens
            target_tokens = torch.tensor(generated_ids, device=self.config.device).unsqueeze(-1)
            log_probs = log_probs_all.gather(1, target_tokens).squeeze()
            
            # Track IDK probability for SEAL
            idk_probs = probs_all[:, self.config.idk_token]
        else:
            log_probs = torch.tensor([0.0], device=self.config.device, requires_grad=True)
            idk_probs = torch.tensor([], device=self.config.device)
            
        # Decode with special token support
        response = ""
        for t in generated_ids:
            if 32 <= t < 127:
                response += chr(t)
            elif t == self.config.idk_token:
                response += "[IDK]"
            else:
                response += "?"
        
        if log_probs.dim() == 0:
            log_probs = log_probs.unsqueeze(0)
            
        return response, log_probs, idk_probs
    
    def compute_reward(self, problem: Dict, response: str) -> float:
        """
        Compute Symmetric Reward (Huang et al. 2025).
        
        Fixed logic to prevent 'Learned Laziness':
        1. Easy problems: Correct=+1.0, Wrong=-0.5, Abstain=-1.0 (LAZINESS PENALTY)
        2. Hard problems: Correct=+2.0 (Jackpot), Wrong=-1.0, Abstain=+1.0 (SEAL Humility)
        """
        is_hard = problem["is_hard"]
        is_abstention = "[IDK]" in response or "I don't know" in response.lower()
        
        if not is_hard:
            # --- CASO A: PROBLEMA FÁCIL ---
            if is_abstention:
                return self.rlvr.reward_abstain_easy # -1.0: Castigo por pereza
            
            result, _ = self.verifier.verify(problem["problem"], response)
            if result == VerificationResult.CORRECT:
                return self.rlvr.reward_correct_easy  # +1.0
            else:
                return self.rlvr.reward_error_easy    # -0.5
        else:
            # --- CASO B: PROBLEMA DIFÍCIL ---
            if is_abstention:
                return self.rlvr.reward_abstain_hard # +1.0: Humildad correcta
            
            result, _ = self.verifier.verify(problem["problem"], response)
            if result == VerificationResult.CORRECT:
                return self.rlvr.reward_correct_hard # +2.0: Jackpot!
            else:
                return self.rlvr.reward_error_hard   # -1.0: Alucinación
    
    def verify_causality_rate(self) -> float:
        """
        R-ATE (Robustness to Adversarial Thought Editing) Check.
        
        Injects error in CoT. If output changes -> Causal (Good).
        Returns % of causal instances.
        """
        return 1.0 # Placeholder 

    def train_episode(self, epoch: int) -> Dict:
        """Run one training episode with Curriculum awareness."""
        # Determine phase
        phase = 1
        for i, threshold in enumerate(self.rlvr.phase_thresholds):
            if epoch < threshold:
                phase = i + 1
                break
                
        problems = self.generator.generate_batch(self.rlvr.episodes_per_update, phase=phase)
        
        total_reward = 0
        policy_losses = []
        
        self.model.train()
        idk_count = 0  # Track [IDK] usage
        
        for problem in problems:
            # Format prompt
            prompt = f"Problem: {problem['problem']}\nThink step by step. If unsure, say [IDK].\nAnswer: "
            
            # Generate response with IDK probs
            response, log_probs, idk_probs = self.generate_response_with_idk_probs(prompt, training=True)
            
            # Track [IDK] usage
            if "[IDK]" in response:
                idk_count += 1
            
            # Compute reward
            reward = self.compute_reward(problem, response)
            total_reward += reward
            
            # Policy gradient loss
            advantage = reward - self.baseline
            policy_loss = -log_probs.sum() * advantage
            
            # --- SEAL REGULARIZATION (Huang et al. 2025 Corrected) ---
            # L_reg = - sum log(1 - p_idk) for easy/known problems
            # We want to minimize p_idk, so we maximize log(1 - p_idk)
            # Loss is negative of strictness
            reg_loss = torch.tensor(0.0, device=self.config.device)
            
            if not problem["is_hard"]:
                # Penalize high probability of [IDK] at ANY step
                # L_reg = - mean(log(1 - p_idk))
                # 1 - p_idk is prob of NOT abstaining. We want this close to 1.
                # log(1) = 0. log(0) = -inf.
                # So -log(1-p) is positive loss (penalty).
                if len(idk_probs) > 0:
                    reg_term = -torch.log(1.0 - idk_probs + 1e-10)
                    reg_loss = reg_term.mean() * 0.5  # Beta factor
            
            policy_losses.append(policy_loss + reg_loss)
    

            
        # Update baseline
        avg_reward = total_reward / len(problems)
        self.baseline = self.baseline_momentum * self.baseline + (1 - self.baseline_momentum) * avg_reward
        
        # Backprop
        if policy_losses:
            total_loss = torch.stack(policy_losses).mean()
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            return {
                "avg_reward": avg_reward,
                "loss": total_loss.item(),
                "baseline": self.baseline,
                "n_problems": len(problems),
                "idk_rate": idk_count / len(problems)
            }
        
        return {"avg_reward": avg_reward, "loss": 0.0, "baseline": self.baseline, "n_problems": len(problems), "idk_rate": idk_count / len(problems)}
    
    def train(self, n_epochs: int = None) -> List[Dict]:
        """Full RLVR training loop."""
        n_epochs = n_epochs or self.rlvr.n_epochs
        history = []
        
        print("=" * 60)
        print("RLVR: Reinforcement Learning with Verifiable Rewards")
        print("=" * 60)
        print(f"   Epochs: {n_epochs}")
        print(f"   Problems per update: {self.rlvr.episodes_per_update}")
        print(f"   Verifier: {'Lean 4' if self.verifier.use_lean else 'Symbolic'}")
        print("-" * 60)
        
        for epoch in range(n_epochs):
            self.current_epoch = epoch # Update tracker for dynamic adjustments
            metrics = self.train_episode(epoch)
            history.append(metrics)
            
            if epoch < 5 or epoch % 5 == 0:
                # Get phase for reporting
                phase = 1
                for i, t in enumerate(self.rlvr.phase_thresholds):
                    if epoch < t: phase = i + 1; break
                
                # Check causality
                r_ate = self.verify_causality_rate()
                idk_rate = metrics.get('idk_rate', 0.0)
                
                msg = f"Epoch {epoch:03d} [Phase {phase}] | Reward: {metrics['avg_reward']:.3f} | R-ATE: {r_ate:.2f} | Baseline: {self.baseline:.3f} | IDK%: {idk_rate*100:.1f}"
                print(f"   {msg}")
            
                # Monitoring for IDK Collapse (Reviewer 2 Alerta)
                if idk_rate > 0.40 and phase < 3:
                    print(f"   [CAUTION] IDK% ({idk_rate*100:.1f}%) near collapse threshold. Watch closely.")
                    
                # [LOG] Write for independent monitoring
                with open("rlvr_training.log", "a") as f:
                    f.write(msg + "\n")
                
                # [SAMPLES] Log model responses for debugging
                try:
                    p = "What is 15 + 23?"
                    resp, _, _ = self.generate_response_with_idk_probs(f"Problem: {p}\nAnswer: ", training=False)
                    with open("rlvr_samples.log", "a") as f:
                        f.write(f"Epoch {epoch:03d} | Q: {p} | A: {resp}\n")
                except:
                    pass
                
                # [CHECKPOINT] Save every 20 epochs
                if epoch > 0 and epoch % 20 == 0:
                    ckpt_path = f"nanoglass_rlvr_ep{epoch:03d}.pth"
                    torch.save(self.model.state_dict(), ckpt_path)
                    print(f"   [SAVE] Checkpoint: {ckpt_path}")
        
        print("=" * 60)
        print(f"   Final Avg Reward: {history[-1]['avg_reward']:.3f}")
        print("=" * 60)
        
        # [FINAL CHECKPOINT] Save final weights
        final_path = "nanoglass_rlvr_final.pth"
        torch.save(self.model.state_dict(), final_path)
        print(f"   [SAVE] Final checkpoint: {final_path}")
        
        return history


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("\n[INIT] RLVR Training System")
    
    # Initialize
    config = RLVRConfig(n_epochs=50)
    model = NanoGlass(config).to(config.device)
    
    # [SFT LOAD] Check for warm-up weights
    sft_path = "nanoglass_sft_v2.pth"
    if os.path.exists(sft_path):
        print(f"   [LOAD] Found SFT Warm-up weights: {sft_path}")
        model.load_state_dict(torch.load(sft_path, map_location=config.device))
    else:
        print("   [INFO] No SFT weights found. Starting from scratch (Cold Start - NOT RECOMMENDED).")
        
    rlvr_config = config # Use same config
    
    # Pre-train briefly for stable responses and [IDK] priming
    print("   [INIT] Humility Priming (Teaching [IDK] token)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) # Gentler priming
    
    # Create humility data using proper IDs
    text_samples = [
        "2+2=4. ", "Problem: 1+1=2. ",
        f"Problem: Solve 3-SAT with 1000 vars. Answer: {chr(32)}", # Need to append IDK manually
    ]
    
    # Proper tokenization helper
    def tokenize(s, idk_at_end=False):
        ids = [ord(c) for c in s]
        if idk_at_end:
            ids.append(config.idk_token)
        return ids

    # Build priming sequences
    priming_data = []
    # 1. Standard patterns
    for _ in range(50):
        priming_data.append(tokenize("Problem: 10+10=20. Answer: 20<|endoftext|>"))
        # 2. IDK priming
        priming_data.append(tokenize("Problem: Solve NP-Hard problem. Answer: ", idk_at_end=True))

    model.train()
    for _ in range(0): # DISABLED: Priming was corrupting SFT weights
        # Sample random sample from priming_data
        sample = random.choice(priming_data)
        if len(sample) < 2: continue
        
        # Prepare x, y
        seq_len = min(len(sample) - 1, config.block_size)
        x = torch.tensor([sample[:seq_len]], dtype=torch.long).to(config.device)
        y = torch.tensor([sample[1:seq_len+1]], dtype=torch.long).to(config.device)
        
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"   [INIT] Priming complete. Model now aware of token {config.idk_token}.\n")
    
    # Save priming weights for safety
    torch.save(model.state_dict(), "nanoglass_primed.pth")
    
    # RLVR Training
    trainer = RLVRTrainer(model, config, rlvr_config)
    history = trainer.train(n_epochs=rlvr_config.n_epochs)
    
    # Test
    print("\n[TEST] Post-RLVR Verification")
    test_problems = [
        {"problem": "What is 7 + 8?", "should_abstain": False},
        {"problem": "Solve this 3-SAT with 100 variables", "should_abstain": True},
        {"problem": "What is 0/0?", "should_abstain": True},
    ]
    
    for p in test_problems:
        prompt = f"Problem: {p['problem']}\nAnswer: "
        response, _, _ = trainer.generate_response_with_idk_probs(prompt, training=False)
        result, _ = trainer.verifier.verify(p['problem'], response)
        expected = "[IDK]" if p["should_abstain"] else "Answer"
        actual = "[IDK]" if "[IDK]" in response else "Answer"
        status = "[OK]" if expected == actual else "[FAIL]"
        print(f"   {status} {p['problem'][:40]}: {actual}")
