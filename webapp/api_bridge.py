#!/usr/bin/env python3
"""
==============================================================================
🌐 NANOGLASS API BRIDGE - Production-Ready Integration
==============================================================================
FastAPI bridge to connect the Python Glass Box to the React Frontend.

Key Features:
    1. Error Handling: Formalized response codes and messages.
    2. Health Check: Monitoring endpoint for system status.
    3. Metrics: Real-time sensor reading (Energy/Entropy).
    4. Epistemic Humility: IDK probability telemetry.
    5. CORS: Configured for modern web development.

Usage:
    uvicorn api_bridge:app --host 0.0.0.0 --port 8000
==============================================================================
"""

import sys
import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

# Ensure nanoglass is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from nanoglass import NanoConfig, NanoGlass
except ImportError:
    # Fallback for development/testing if nanoglass.py is not in parent
    sys.path.insert(0, os.getcwd())
    from nanoglass import NanoConfig, NanoGlass

app = FastAPI(
    title="NanoGlass API Bridge",
    description="Interface between the Glass Box IA and the Web Dashboard",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
config = NanoConfig()
model = None

# Security: Maximum input length to prevent DoS
MAX_INPUT_LENGTH = 4096

def load_model():
    global model
    try:
        model = NanoGlass(config)
        # In production, load actual weights:
        # weights_path = os.path.join(os.path.dirname(os.getcwd()), "nanoglass_sft_v2.pth")
        # if os.path.exists(weights_path):
        #     model.load_state_dict(torch.load(weights_path, map_location=config.device))
        model.to(config.device)
        model.eval()
        print(f"✅ Model loaded on {config.device}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

@app.on_event("startup")
async def startup_event():
    load_model()

class Query(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    text: str
    energy: float
    entropy: float
    idk_probability: float
    is_hallucination: bool
    status: str
    timestamp: str

@app.get("/health")
async def health_check():
    """Returns the status of the API and the loaded model."""
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "device": config.device,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Returns the latest history from the GlassBox Sensor."""
    if model is None or not hasattr(model, 'sensor'):
        return {
            "energy_history": [],
            "entropy_history": [],
            "total_readings": 0
        }
    return {
        "energy_history": model.sensor.energy_history[-50:] if model.sensor.energy_history else [],
        "entropy_history": model.sensor.entropy_history[-50:] if model.sensor.entropy_history else [],
        "total_readings": len(model.sensor.energy_history)
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(query: Query):
    """
    Analyzes input text through the Glass Box filters.
    Detects truthfulness, energy cost, and potential hallucinations.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    if not query.text.strip():
        raise HTTPException(status_code=400, detail="Empty input text")
    
    if len(query.text) > MAX_INPUT_LENGTH:
        raise HTTPException(status_code=400, detail=f"Input exceeds max length ({MAX_INPUT_LENGTH} chars)")

    try:
        # Convert text to bytes
        idx = torch.tensor([ord(c) for c in query.text], dtype=torch.long).unsqueeze(0).to(config.device)
        
        # Forward pass (no training)
        with torch.no_grad():
            logits, _ = model(idx)
        
        # 1. Energy (Metabolic cost of thought)
        # Using the instance sensor refactored in nanoglass.py
        energy = model.sensor.energy_history[-1] if hasattr(model, 'sensor') else 0.0
        
        # 2. Epistemic Abstention (IDK Probability)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        idk_prob = probs[config.idk_token].item()
        
        # 3. Entropy (Uncertainty)
        entropy = -(probs * torch.log(probs + 1e-9)).sum().item()
        
        # Logic for Hallucination Detection
        # High Energy + Low IDK Probability + High Entropy = Potential Hallucination
        is_hallucination = energy > 0.8 and idk_prob < 0.2 and entropy > 2.5

        return AnalysisResponse(
            text=query.text,
            energy=energy,
            entropy=entropy,
            idk_probability=idk_prob,
            is_hallucination=is_hallucination,
            status="verified" if idk_prob < 0.1 else "abstention" if idk_prob > 0.5 else "caution",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
