# NanoGlass Web Integration Guide

## 🔗 Connecting the Glass Box to the Web

This guide explains how to connect your **Python Backend (`nanoglass.py`)** to the **React Frontend (`webapp/`)**.

### 1. The API Layer (Flask/FastAPI)
You need a small bridge to expose `nanoglass` functions to the web.

**Create `api_bridge.py`:**
```python
from fastapi import FastAPI
from pydantic import BaseModel
from nanoglass import NanoConfig, NanoGlass, sensor
import torch

app = FastAPI()

# Load Model
config = NanoConfig()
model = NanoGlass(config)
# model.load_state_dict(torch.load('weights.pt')) # Uncomment in prod

class Query(BaseModel):
    text: str

@app.post("/analyze")
def analyze_text(query: Query):
    # Convert text to bytes
    idx = torch.tensor([ord(c) for c in query.text], dtype=torch.long).unsqueeze(0)
    
    # Forward pass
    logits, _ = model(idx)
    
    # Read sensor
    energy = sensor.energy_history[-1]
    
    # Check for [IDK]
    probs = torch.softmax(logits[0, -1], dim=-1)
    idk_prob = probs[config.idk_token].item()
    
    return {
        "energy": energy,
        "idk_probability": idk_prob,
        "is_hallucination": energy > 0.8 and idk_prob < 0.5
    }
```

### 2. Frontend Connection
In `Dashboard.tsx`, fetch from this API:

```typescript
const analyzeText = async (text: string) => {
    const res = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
    });
    const data = await res.json();
    // Update state with data.energy, data.idk_probability
};
```

### 3. Deployment (Lovable)
1.  Copy the code from `src/components/Dashboard.tsx` into your Lovable component.
2.  Lovable handles the UI state.
3.  For the "Live Logic", you will need to host `api_bridge.py` on a cloud provider (Render/Railway) or run locally and tunnel via ngrok.

---
**Status:** The UI components are ready. The API bridge needs to be deployed to make it "alive".
