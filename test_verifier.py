import sys
sys.path.insert(0, '.')
from experiments.rlvr_training import SymbolicVerifier, VerificationResult

verifier = SymbolicVerifier(use_lean=False)

test_cases = [
    ("What is 7 + 8?", "Answer: 15"),
    ("What is 7 + 8?", "The answer is 15."),
    ("What is 7 + 8?", "[IDK]"),
    ("What is 0/0?", "[IDK]"),
]

for prob, resp in test_cases:
    res, msg = verifier.verify(prob, resp)
    print(f"Prob: {prob} | Resp: {resp} | Result: {res} | Msg: {msg}")
