"""
==============================================================================
NanoGlass Probes (Glass Box Infrastructure)
==============================================================================
Centralized probing mechanism for inspecting internal model states.
Used by Mamba-2, MoE, and Main Model to report telemetry.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class GlassBoxProbe:
    """
    Observer for neural internal states.
    Stores activations, entropies, and energies without affecting gradients.
    """
    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.enabled = True
    
    def log(self, key: str, value: torch.Tensor):
        """Log a tensor value (detached and moved to CPU)."""
        if self.enabled:
            # Aggregate or store last? For now, store last.
            # In a real run, might want moving averages.
            if value.numel() == 1:
                self.data[key] = value.item()
            else:
                self.data[key] = value.detach().float().cpu()
    
    def clear(self):
        self.data = {}

# Singleton instance
main_probe = GlassBoxProbe()
