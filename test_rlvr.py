import sys
sys.path.insert(0, '.')
import torch
from experiments.rlvr_training import RLVRTrainer, RLVRConfig, SymbolicVerifier

# Initialize
config = RLVRConfig(n_epochs=1)
from llm_glassbox import CortexGlassBox as NanoGlass
model = NanoGlass(config).to(config.device)

# Load RLVR weights if exist
import os
if os.path.exists('nanoglass_sft_v2.pth'):
    model.load_state_dict(torch.load('nanoglass_sft_v2.pth', map_location=config.device))

trainer = RLVRTrainer(model, config)

# Test
test_problems = [
    {'problem': 'What is 7 + 8?', 'should_abstain': False},
    {'problem': 'Solve this 3-SAT with 100 variables', 'should_abstain': True},
    {'problem': 'What is 0/0?', 'should_abstain': True},
]

print('[TEST] Post-RLVR Verification')
for p in test_problems:
    prompt = f"Problem: {p['problem']}\nAnswer: "
    response, _, _ = trainer.generate_response_with_idk_probs(prompt, training=False)
    expected = '[IDK]' if p['should_abstain'] else 'Answer'
    actual = '[IDK]' if '[IDK]' in response else 'Answer'
    status = '[OK]' if expected == actual else '[FAIL]'
    print(f'   {status} {p["problem"][:40]}: {actual} -> "{response[:50]}"')
