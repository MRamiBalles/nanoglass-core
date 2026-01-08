"""
==============================================================================
HERMES Bridge: Neuro-Symbolic Integration for Verifiable Rewards
==============================================================================
Implements a bridge between NanoGlass and PvsNP's formal verification.

Architecture:
    Input: Chain-of-Thought from NanoGlass (text or Lean 4 tactics)
    Process: 
        1. Sanitize and extract code blocks
        2. Execute in sandbox (Python exec / Lean 4 REPL)
        3. Verify: Did it compile? Did tests pass?
    Output: Deterministic reward (+1.0 / -1.0 / 0.0)

This enables RLVR with mathematically absolute truth signals,
eliminating reward hacking via semantic similarity.

References:
    - DeepSeek-R1: RLHF with Verifiable Rewards
    - Toshniwal et al. 2025: Formal Verification in LLM Training
    - PvsNP Structural Complexity Observatory (User's Repo)

==============================================================================
"""
import subprocess
import tempfile
import os
import re
import ast
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class HERMESConfig:
    """Configuration for HERMES verification bridge."""
    # Lean 4 settings
    lean_path: str = "lean4"  # Path to Lean 4 binary
    lean_timeout: int = 10    # Seconds before timeout
    
    # Python sandbox settings
    python_timeout: int = 5   # Seconds for Python exec
    
    # Reward structure
    reward_correct: float = 1.0
    reward_incorrect: float = -1.0
    reward_timeout: float = 0.0
    reward_syntax_error: float = -0.5


class VerificationType(Enum):
    LEAN4_PROOF = "lean4"
    PYTHON_CODE = "python"
    ARITHMETIC = "arithmetic"
    UNKNOWN = "unknown"


class VerificationResult(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    TIMEOUT = "timeout"
    SYNTAX_ERROR = "syntax_error"
    SANDBOX_ERROR = "sandbox_error"


# ==============================================================================
# CODE SANITIZER
# ==============================================================================

class CodeSanitizer:
    """
    Extracts and sanitizes code blocks from Chain-of-Thought text.
    """
    
    @staticmethod
    def extract_code_blocks(text: str) -> Dict[str, str]:
        """
        Extract code blocks from markdown-style text.
        
        Returns dict mapping language -> code
        """
        blocks = {}
        
        # Markdown code blocks: ```language\ncode\n```
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for lang, code in matches:
            lang = lang.lower() if lang else "unknown"
            blocks[lang] = code.strip()
        
        # Lean 4 theorem blocks
        lean_pattern = r'(theorem|lemma|example)\s+(\w+).*?:=\s*by\s*([\s\S]*?)(?=\n\n|$)'
        lean_matches = re.findall(lean_pattern, text, re.IGNORECASE)
        if lean_matches:
            lean_code = "\n".join([f"{m[0]} {m[1]} := by\n  {m[2]}" for m in lean_matches])
            blocks["lean4"] = lean_code
        
        # Python expressions (simple arithmetic)
        if not blocks:
            # Try to find arithmetic expressions
            arith_pattern = r'(\d+\s*[\+\-\*\/]\s*\d+)'
            arith_matches = re.findall(arith_pattern, text)
            if arith_matches:
                blocks["arithmetic"] = arith_matches[0]
        
        return blocks
    
    @staticmethod
    def detect_verification_type(text: str) -> VerificationType:
        """Determine what kind of verification is needed."""
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ["theorem", "lemma", "lean", "by", "simp", "rfl"]):
            return VerificationType.LEAN4_PROOF
        elif any(kw in text_lower for kw in ["def ", "import ", "print(", "return "]):
            return VerificationType.PYTHON_CODE
        elif re.search(r'\d+\s*[\+\-\*\/]\s*\d+', text):
            return VerificationType.ARITHMETIC
        else:
            return VerificationType.UNKNOWN


# ==============================================================================
# VERIFIERS
# ==============================================================================

class Lean4Verifier:
    """
    Verifies Lean 4 proofs using the Lean REPL.
    
    Connects to PvsNP's HERMES protocol for formal verification.
    """
    
    def __init__(self, config: HERMESConfig):
        self.config = config
        self._check_lean_available()
    
    def _check_lean_available(self) -> bool:
        """Check if Lean 4 is installed."""
        try:
            result = subprocess.run(
                [self.config.lean_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def verify(self, proof_code: str) -> Tuple[VerificationResult, str]:
        """
        Verify a Lean 4 proof.
        
        Returns:
            (result, explanation)
        """
        # Wrap in minimal Lean 4 file
        lean_content = f"""
-- Auto-generated by HERMES Bridge
-- NanoGlass Verification

{proof_code}
"""
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.lean', 
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(lean_content)
                lean_file = f.name
            
            # Run Lean 4 checker
            result = subprocess.run(
                [self.config.lean_path, lean_file],
                capture_output=True,
                text=True,
                timeout=self.config.lean_timeout
            )
            
            os.unlink(lean_file)
            
            if result.returncode == 0:
                return VerificationResult.CORRECT, "Lean 4 proof verified successfully"
            else:
                # Parse error message
                error = result.stderr[:500] if result.stderr else result.stdout[:500]
                return VerificationResult.INCORRECT, f"Lean error: {error}"
                
        except subprocess.TimeoutExpired:
            return VerificationResult.TIMEOUT, "Lean verification timed out"
        except FileNotFoundError:
            return VerificationResult.SANDBOX_ERROR, "Lean 4 not installed"
        except Exception as e:
            return VerificationResult.SANDBOX_ERROR, str(e)


class PythonVerifier:
    """
    Verifies Python code via sandboxed execution.
    
    WARNING: Uses restricted exec. For production, use Docker/nsjail.
    """
    
    def __init__(self, config: HERMESConfig):
        self.config = config
    
    def verify(self, code: str, expected_output: str = None) -> Tuple[VerificationResult, str]:
        """
        Execute Python code in sandbox and check output.
        
        Returns:
            (result, explanation)
        """
        # Restricted globals (no file access, no imports)
        restricted_globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "int": int,
                "float": float,
                "str": str,
                "list": list,
                "dict": dict,
                "True": True,
                "False": False,
                "None": None,
            }
        }
        
        try:
            # Parse first to check syntax
            ast.parse(code)
            
            # Execute with timeout (simplified - real impl uses multiprocessing)
            exec_globals = restricted_globals.copy()
            exec(code, exec_globals)
            
            # Check for expected output if provided
            if expected_output and "result" in exec_globals:
                if str(exec_globals["result"]) == str(expected_output):
                    return VerificationResult.CORRECT, f"Result: {exec_globals['result']}"
                else:
                    return VerificationResult.INCORRECT, f"Expected {expected_output}, got {exec_globals['result']}"
            
            return VerificationResult.CORRECT, "Code executed without errors"
            
        except SyntaxError as e:
            return VerificationResult.SYNTAX_ERROR, f"Syntax error: {e}"
        except Exception as e:
            return VerificationResult.INCORRECT, f"Runtime error: {e}"


class ArithmeticVerifier:
    """
    Simple arithmetic verifier for basic math problems.
    """
    
    def verify(self, expression: str, expected: str = None) -> Tuple[VerificationResult, str]:
        """Evaluate arithmetic expression."""
        try:
            # Sanitize: only allow digits and basic operators
            sanitized = re.sub(r'[^\d\+\-\*\/\(\)\.\s]', '', expression)
            result = eval(sanitized)
            
            if expected is not None:
                if abs(float(result) - float(expected)) < 1e-6:
                    return VerificationResult.CORRECT, f"{expression} = {result}"
                else:
                    return VerificationResult.INCORRECT, f"Expected {expected}, got {result}"
            
            return VerificationResult.CORRECT, f"{expression} = {result}"
            
        except Exception as e:
            return VerificationResult.INCORRECT, str(e)


# ==============================================================================
# HERMES BRIDGE (Main Interface)
# ==============================================================================

class HERMESBridge:
    """
    Main bridge between NanoGlass and HERMES formal verification.
    
    Usage:
        bridge = HERMESBridge()
        reward, explanation = bridge.verify_and_reward(
            question="What is 2 + 2?",
            response="The answer is 4."
        )
    """
    
    def __init__(self, config: HERMESConfig = None):
        self.config = config or HERMESConfig()
        self.sanitizer = CodeSanitizer()
        self.lean_verifier = Lean4Verifier(self.config)
        self.python_verifier = PythonVerifier(self.config)
        self.arith_verifier = ArithmeticVerifier()
    
    def verify_and_reward(
        self, 
        question: str, 
        response: str,
        expected_answer: str = None
    ) -> Tuple[float, str]:
        """
        Verify response and return deterministic reward.
        
        This is the core function called by RLVR trainer.
        
        Returns:
            (reward, explanation)
        """
        # Detect verification type
        vtype = self.sanitizer.detect_verification_type(response)
        
        # Extract code blocks
        blocks = self.sanitizer.extract_code_blocks(response)
        
        # Route to appropriate verifier
        if vtype == VerificationType.LEAN4_PROOF or "lean4" in blocks:
            code = blocks.get("lean4", response)
            result, explanation = self.lean_verifier.verify(code)
            
        elif vtype == VerificationType.PYTHON_CODE or "python" in blocks:
            code = blocks.get("python", response)
            result, explanation = self.python_verifier.verify(code, expected_answer)
            
        elif vtype == VerificationType.ARITHMETIC or "arithmetic" in blocks:
            expr = blocks.get("arithmetic", response)
            result, explanation = self.arith_verifier.verify(expr, expected_answer)
            
        else:
            # Fallback: Try to extract arithmetic from text
            numbers = re.findall(r'\d+', response)
            if numbers and expected_answer:
                if expected_answer in numbers:
                    result = VerificationResult.CORRECT
                    explanation = f"Found expected answer {expected_answer} in response"
                else:
                    result = VerificationResult.INCORRECT
                    explanation = f"Expected {expected_answer}, found {numbers}"
            else:
                # Cannot verify this type
                result = VerificationResult.SANDBOX_ERROR
                explanation = "Unknown verification type"
        
        # Convert to reward
        reward = self._result_to_reward(result)
        
        return reward, f"[{result.value}] {explanation}"
    
    def _result_to_reward(self, result: VerificationResult) -> float:
        """Map verification result to reward value."""
        mapping = {
            VerificationResult.CORRECT: self.config.reward_correct,
            VerificationResult.INCORRECT: self.config.reward_incorrect,
            VerificationResult.TIMEOUT: self.config.reward_timeout,
            VerificationResult.SYNTAX_ERROR: self.config.reward_syntax_error,
            VerificationResult.SANDBOX_ERROR: 0.0,
        }
        return mapping.get(result, 0.0)


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    print("\n[TEST] HERMES Bridge - Neuro-Symbolic Verification")
    print("=" * 60)
    
    bridge = HERMESBridge()
    
    test_cases = [
        # Arithmetic
        {
            "question": "What is 2 + 2?",
            "response": "Let me calculate: 2 + 2 = 4",
            "expected": "4"
        },
        # Wrong arithmetic
        {
            "question": "What is 5 * 3?",
            "response": "The answer is 14",
            "expected": "15"
        },
        # Python code
        {
            "question": "Write code to compute factorial of 5",
            "response": """```python
result = 1
for i in range(1, 6):
    result *= i
```""",
            "expected": "120"
        },
        # Lean 4 (will fail if Lean not installed)
        {
            "question": "Prove that 1 + 1 = 2",
            "response": """theorem one_plus_one : 1 + 1 = 2 := by rfl""",
            "expected": None
        },
    ]
    
    for i, tc in enumerate(test_cases):
        reward, explanation = bridge.verify_and_reward(
            tc["question"],
            tc["response"],
            tc["expected"]
        )
        status = "[OK]" if reward > 0 else "[FAIL]"
        print(f"   {status} Test {i+1}: reward={reward:.1f}")
        print(f"      Q: {tc['question'][:40]}")
        print(f"      {explanation[:60]}")
        print()
    
    print("=" * 60)
