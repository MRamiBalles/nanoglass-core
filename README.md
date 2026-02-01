# 🔮 Project NanoGlass

> **"Expert at Minimum Cost"**

A from-scratch, byte-level Transformer designed for maximum transparency and epistemic humility.

---

## 🚀 Quick Start

```bash
# Run the main training script with built-in verification
python nanoglass.py

# Run the empirical verification suite (PASS/FAIL for all claims)
python verify_all.py
```

---

## 📁 Project Structure

```
electric-gravity-nanoglass/
├── nanoglass.py           # Core byte-level Transformer with TruthRL
├── verify_all.py          # Empirical verification suite
├── llm_glassbox.py        # Cortex-13 hybrid implementation (Mamba-2 + GQA)
├── webapp/                # React Dashboard Frontend
│   └── api_bridge.py      # FastAPI bridge for web integration
├── experiments/           # Research & Benchmarking
│   ├── truthfulqa_loader.py    # REAL TruthfulQA integration
│   ├── truthfulness_benchmark.py # Truthfulness & Calibration
│   ├── anchored_review.py      # Claim validation vs baselines
│   └── ... (20+ scripts)
├── scripts/               
│   └── generate_kg.py     # Knowledge Graph Generator (Idea2Story)
├── data/                  # Knowledge Graph & Cache
└── research_*.tex         # LaTeX papers for each finding
```

---

## 🧠 Core Features

### 1. Byte-Level Architecture
- **Vocabulary:** 257 tokens (0-255 bytes + [IDK] token)
- **No tokenizer dependency** - pure transparency

### 2. TruthRL (Epistemic Correction)
The model is trained with a ternary reward system:
- **+1.0** Correct answer → Normal loss
- **0.0** Abstention ([IDK]) → Minimal penalty
- **-1.0** Hallucination → 2x penalty

### 3. Glass Box Sensors
Real-time measurement of:
- **Energy** (L1 Norm) - Lower = More confident/truthful
- **Entropy** - Higher = More creative/uncertain

---

## ⚠️ Epistemic Caution

> **This project uses analogies for didactic purposes.**
> - "The model fears death" means: "the loss function incentivizes continuity."
> - "Softmax is quantum collapse" means: "structural analogy, not physical equivalence."
> - **These are models of AI, not claims about consciousness.**

---

## 📊 Key Findings

| Paper | Finding | Status |
|-------|---------|--------|
| IX | Truth minimizes energy | ⏳ Verify via `verify_all.py` |
| XVI | [IDK] token enables abstention | ⏳ Verify via `verify_all.py` |
| XVII | Gödel: Hallucination ↔ Completeness trade-off | ✅ Theoretical |

---

## 📚 Documentation

- **Walkthrough:** See `brain/*/walkthrough.md` for full project narrative
- **Audit Plan:** See `brain/*/audit_plan.md` for corrections and roadmap
- **Papers:** All `research_*.tex` files compile in Overleaf

---

## 🔬 Research Philosophy

This project explores the hypothesis that:
> **A system that never hallucinates is a Formal System (Consistent but Incomplete, per Gödel). A system that can hallucinate may access "Unprovable Truths."**

The balance between hallucination and truth is the central trade-off of AI alignment.

---

## 📜 License

Research/Educational Use. See individual papers for academic attribution.
