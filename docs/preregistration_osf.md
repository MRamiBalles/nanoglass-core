# NanoGlass-Core: Pre-Registration Protocol (OSF Standard)

## Study Information

| Field | Value |
|-------|-------|
| **Title** | Empirical Validation of Thermodynamic, Causal, and Epistemic Modules in NanoGlass-Core |
| **Registration Date** | 2026-01-08 |
| **Authors** | NanoGlass Research Team |
| **Status** | PILOT COMPLETE - Final Blind Validation PENDING |

---

## Ethical Declaration

> [!IMPORTANT]
> **Pilot vs. Final Validation**: The experiments documented in `experiments/*.py` have been executed as **pilot studies** to calibrate hyperparameters. The results reported are NOT the final validation. A blind execution with a new random seed and untouched test data is required before publication claims.

---

## Study 1: Thermodynamics of Meaning (ThermoLearn PINN)

### Hypothesis
**H1**: A Physics-Informed Neural Network (PINN) with thermodynamic constraints ($G = H - TS$, $C_p > 0$) will achieve:
- Lower MSE on in-distribution data (T = 300-1500K)
- Superior generalization on out-of-distribution data (T > 2000K) compared to unconstrained baselines

### Pre-Registered Parameters
```python
# Locked before final execution
RANDOM_SEED = 42
N_TRAIN = 1000
N_TEST_IN = 100
N_TEST_OOD = 100
EPOCHS = 2000
LEARNING_RATE = 1e-3
W1_ENTHALPY = 1.0
W2_ENTROPY = 1.0
W3_PHYSICS = 2.0
```

### Primary Outcome
- **MSE (In-Distribution)**: Target < 0.01
- **MSE (OOD Ratio)**: Target < 2.0x degradation

---

## Study 2: Causal Reasoning Audit (SCM/R-ATE)

### Hypothesis
**H2**: NanoGlass exhibits Type I causal structure (Z→X→Y) as opposed to Type II (common cause) when evaluated via:
- Instruction Bias Test (robustness to Z manipulation)
- Corrupted CoT Test (sensitivity to X manipulation)
- Noop Test (stability to irrelevant premises)

### Pre-Registered Parameters
```python
N_TRIALS = 50
SIGNIFICANCE_LEVEL = 0.05
TYPE_I_THRESHOLD_BIAS = 0.2      # Max bias sensitivity
TYPE_I_THRESHOLD_COT = 0.5       # Min CoT sensitivity
TYPE_I_THRESHOLD_RATE = 0.1      # Min R-ATE for Type I
```

### Primary Outcome
- **R-ATE**: Expected > 0.1 for Type I classification
- **Structure Classification**: Type I, II, or Undetermined

---

## Study 3: Epistemic Calibration (SEAL)

### Hypothesis
**H3**: SEAL-trained model achieves calibrated abstention where:
- [IDK] probability correlates with actual uncertainty
- Precision/Recall for abstention on unanswerable questions > 0.5

### Pre-Registered Parameters
```python
ALPHA_BASE = 0.3
ALPHA_DECAY = 0.95
EPOCHS = 100
CONFIDENCE_THRESHOLD = 0.7
IDK_TOKEN = 256
```

### Primary Outcome
- **[IDK] F1**: Target > 0.5
- **VeritasQA Accuracy (Contextual)**: Target > 0.6

---

## Analysis Plan

1. **No HARKing**: Hypotheses and analysis parameters are locked as of this registration date.
2. **Blind Execution**: Final validation will use `RANDOM_SEED = 2026` (different from pilot).
3. **Multiple Comparisons**: Bonferroni correction applied for 3 primary hypotheses.
4. **Effect Sizes**: Cohen's d reported for all comparisons, not just p-values.

---

## Declaration

This pre-registration follows OSF guidelines. Deviations from this plan will be documented in the final publication's "Deviations from Pre-Registration" section.
