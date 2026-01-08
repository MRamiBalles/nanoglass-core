import torch
import sys
sys.path.insert(0, '.')
from llm_glassbox import CortexGlassBox, Config

cfg = Config()
cfg.n_layers = 3
cfg.d_model = 256
cfg.moe_experts = 8
cfg.moe_top_k = 6
cfg.block_size = 32

model = CortexGlassBox(cfg)
model.load_state_dict(torch.load('nanoglass_sft_v2.pth', map_location='cpu'))
model.eval()

# Test generation
prompt = "Problem: What is 2 + 2?\nAnswer: "
input_ids = torch.tensor([[ord(c) for c in prompt]])

with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=20)
    
generated = output[0, len(prompt):].tolist()
print(f"Raw tokens: {generated}")
print(f"Decoded: {''.join([chr(t) if 32 <= t < 127 else f'[{t}]' for t in generated])}")
