import torch
import torch.nn.functional as F
from llm_glassbox import CortexGlassBox, Config

cfg = Config()
cfg.n_layers = 3
cfg.d_model = 256
cfg.moe_experts = 8

model = CortexGlassBox(cfg)
x = torch.randint(0, 128, (1, 32))
y = torch.randint(0, 128, (1, 32))

logits, loss = model(x, y)
print(f"Initial Loss (B=1, T=32): {loss.item():.4f}")
print(f"Logits Mean: {logits.mean().item():.4f}, Std: {logits.std().item():.4f}")
print(f"Log(Vocab Size): {torch.log(torch.tensor(cfg.vocab_size, dtype=torch.float)).item():.4f}")
