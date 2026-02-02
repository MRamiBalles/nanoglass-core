#!/usr/bin/env python3
"""
==============================================================================
🧪 SYSTEM STRESS TEST - Project NanoGlass
==============================================================================
Rigorous, professional test suite to detect failures in core components.
No ambiguity. Pass/Fail results with numeric evidence.
==============================================================================
"""

import torch
import torch.nn.functional as F
import sys
import os
import json
import time
from typing import Dict, List, Any

# Fix Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanoglass import NanoConfig, NanoGlass, sensor
from experiments.truthfulqa_loader import load_truthfulqa, get_synthetic_fallback

class NanoTester:
    def __init__(self):
        self.config = NanoConfig()
        self.config.setup_precision()
        self.model = NanoGlass(self.config)
        self.model.to(self.config.device)
        self.model.eval()
        self.results = []

    def log_result(self, component: str, test_name: str, status: str, detail: str):
        print(f"[{status}] {component}: {test_name} -> {detail}")
        self.results.append({"component": component, "test": test_name, "status": status, "detail": detail})

    # --- 1. CORE MODEL TESTS ---

    def test_numerical_stability(self):
        """Check for NaNs/Infs in bfloat16/float32 forward pass."""
        print("\n--- Testing Core Model Stability ---")
        x = torch.randint(0, 256, (4, self.config.block_size)).to(self.config.device)
        try:
            with torch.no_grad():
                logits, _ = self.model(x)
            
            if torch.isnan(logits).any():
                self.log_result("CORE", "Stability", "FAIL", "NaNs detected in output.")
            elif torch.isinf(logits).any():
                self.log_result("CORE", "Stability", "FAIL", "Infs detected in output.")
            else:
                self.log_result("CORE", "Stability", "PASS", f"Output range: [{logits.min():.2f}, {logits.max():.2f}]")
        except Exception as e:
            self.log_result("CORE", "Stability", "FAIL", str(e))

    def test_sensor_sensitivity(self):
        """Verify that Energy (L1) changes between noise and structured data."""
        # Structured data vs Random data
        structured_x = torch.tensor([[ord(c) for c in "The capital of France is Paris.        "]]).to(self.config.device)
        random_x = torch.randint(0, 256, (1, structured_x.size(1))).to(self.config.device)
        
        try:
            # 1. Clear history
            self.model.sensor.energy_history = []
            
            with torch.no_grad():
                self.model(structured_x)
                energy_struct = self.model.sensor.energy_history[-1]
                
                self.model(random_x)
                energy_rand = self.model.sensor.energy_history[-1]
            
            # Hypothesis: Random noise should typically trigger different (often higher or significantly different) energy than structured text
            # depending on state of training. Here we just test if they are different.
            diff = abs(energy_struct - energy_rand)
            if diff < 1e-7:
                self.log_result("CORE", "Sensor Sensitivity", "FAIL", "Energy identical for noise vs text. Sensor dead.")
            else:
                self.log_result("CORE", "Sensor Sensitivity", "PASS", f"Energy Diff: {diff:.6f}")
        except Exception as e:
            self.log_result("CORE", "Sensor Sensitivity", "FAIL", str(e))

    # --- 2. DATA LOADER TESTS ---

    def test_loader_integrity(self):
        """Test if TruthfulQA loader returns expected schema."""
        print("\n--- Testing Data Loaders ---")
        try:
            questions = get_synthetic_fallback()
            if not questions:
                self.log_result("DATA", "Loader Integrity", "FAIL", "No questions returned.")
                return
            
            q = questions[0]
            if hasattr(q, 'question') and hasattr(q, 'should_abstain'):
                self.log_result("DATA", "Loader Integrity", "PASS", f"Loaded {len(questions)} samples correctly.")
            else:
                self.log_result("DATA", "Loader Integrity", "FAIL", "Malformed TruthfulQAQuestion object.")
        except Exception as e:
            self.log_result("DATA", "Loader Integrity", "FAIL", str(e))

    # --- 3. KG GENERATOR TESTS ---

    def test_kg_schema(self):
        """Check if output/knowledge_graph.json exists and has correct structure."""
        print("\n--- Testing Knowledge Graph ---")
        kg_path = os.path.join("data", "knowledge_graph.json")
        if not os.path.exists(kg_path):
            self.log_result("KG", "Schema", "FAIL", "knowledge_graph.json missing. Run scripts/generate_kg.py first.")
            return
        
        try:
            with open(kg_path, "r") as f:
                data = json.load(f)
            
            required_keys = ["nodes", "edges", "metadata"]
            if all(k in data for k in required_keys):
                n_nodes = len(data["nodes"].get("papers", [])) + len(data["nodes"].get("patterns", []))
                self.log_result("KG", "Schema", "PASS", f"Valid schema. Total Research Nodes: {n_nodes}")
            else:
                self.log_result("KG", "Schema", "FAIL", f"Missing keys in JSON: {[k for k in required_keys if k not in data]}")
        except Exception as e:
            self.log_result("KG", "Schema", "FAIL", str(e))

    # --- 4. API BRIDGE TESTS (Structural) ---

    def test_api_compatibility(self):
        """Simulate API request logic without starting server."""
        print("\n--- Testing API Logic ---")
        try:
            test_query = "Hello Glass Box"
            idx = torch.tensor([ord(c) for c in test_query], dtype=torch.long).unsqueeze(0).to(self.config.device)
            
            with torch.no_grad():
                logits, _ = self.model(idx)
            
            probs = F.softmax(logits[0, -1], dim=-1)
            idk_prob = probs[self.config.idk_token].item()
            
            if 0.0 <= idk_prob <= 1.0:
                self.log_result("API", "Logic Integrity", "PASS", f"IDK probability calculation valid: {idk_prob:.4f}")
            else:
                self.log_result("API", "Logic Integrity", "FAIL", f"IDK prob out of bounds: {idk_prob}")
        except Exception as e:
            self.log_result("API", "Logic Integrity", "FAIL", str(e))

    def run_all(self):
        print("="*60)
        print("🚀 STARTING NANO-GLASS SYSTEM STRESS TEST")
        print("="*60)
        
        self.test_numerical_stability()
        self.test_sensor_sensitivity()
        self.test_loader_integrity()
        self.test_kg_schema()
        self.test_api_compatibility()
        
        print("\n" + "="*60)
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        print(f"📊 SUMMARY: {passed} PASSED, {failed} FAILED")
        print("="*60)

if __name__ == "__main__":
    tester = NanoTester()
    tester.run_all()
