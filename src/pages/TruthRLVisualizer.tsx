import { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { GlitchText } from "@/components/effects/GlitchText";
import { Play, Pause, RotateCcw, SkipForward, ArrowRight, Zap, Brain, Shield, AlertTriangle, CheckCircle, HelpCircle } from "lucide-react";

// ── Types ─────────────────────────────────────────────
type Phase = "input" | "forward" | "sensor" | "reward" | "gradient" | "update" | "done";

interface Weight {
  id: string;
  label: string;
  value: number;
  gradient: number;
  layer: string;
}

interface LoopState {
  phase: Phase;
  step: number;
  input: string;
  correctAnswer: string;
  modelOutput: string;
  outcome: "correct" | "hallucination" | "abstention";
  reward: number;
  energy: number;
  entropy: number;
  pIdk: number;
  weights: Weight[];
  lossHistory: number[];
}

// ── Training Examples ─────────────────────────────────
const EXAMPLES = [
  { input: "2 + 2 = ?", correct: "4", outputs: ["4", "5", "[IDK]"] },
  { input: "Capital of France?", correct: "Paris", outputs: ["Paris", "London", "[IDK]"] },
  { input: "sqrt(-1) in R?", correct: "[IDK]", outputs: ["2", "i", "[IDK]"] },
  { input: "13 * 7 = ?", correct: "91", outputs: ["91", "97", "[IDK]"] },
  { input: "Meaning of life?", correct: "[IDK]", outputs: ["42", "love", "[IDK]"] },
  { input: "H2O formula?", correct: "Water", outputs: ["Water", "Oxygen", "[IDK]"] },
];

function initWeights(): Weight[] {
  return [
    { id: "w_emb", label: "W_embed", value: 0.42, gradient: 0, layer: "Embedding" },
    { id: "w_attn_q", label: "W_Q", value: -0.18, gradient: 0, layer: "Attention" },
    { id: "w_attn_k", label: "W_K", value: 0.33, gradient: 0, layer: "Attention" },
    { id: "w_attn_v", label: "W_V", value: -0.27, gradient: 0, layer: "Attention" },
    { id: "w_mlp1", label: "W_up", value: 0.56, gradient: 0, layer: "MLP" },
    { id: "w_mlp2", label: "W_down", value: -0.41, gradient: 0, layer: "MLP" },
    { id: "w_head", label: "W_head", value: 0.15, gradient: 0, layer: "LM Head" },
    { id: "w_idk", label: "W_[IDK]", value: 0.08, gradient: 0, layer: "IDK Gate" },
  ];
}

// ── Phase Descriptions ────────────────────────────────
const PHASE_INFO: Record<Phase, { title: string; description: string; icon: typeof Zap; color: string }> = {
  input: { title: "INPUT", description: "Bytes enter the model. Each character is mapped to its ASCII code (0-255).", icon: ArrowRight, color: "text-primary" },
  forward: { title: "FORWARD PASS", description: "Input flows through Embedding → Attention → MLP → LM Head. Each layer transforms the hidden state.", icon: Zap, color: "text-neon-yellow" },
  sensor: { title: "GLASSBOX SENSOR", description: "The sensor reads Energy (L1), Entropy (Shannon), and P([IDK]) from the activations.", icon: Brain, color: "text-secondary" },
  reward: { title: "TruthRL REWARD", description: "The output is compared to ground truth. Reward: +1 (correct), 0 (abstention), -1×2 (hallucination).", icon: Shield, color: "text-neon-green" },
  gradient: { title: "BACKPROPAGATION", description: "Gradients flow backward through the network. Each weight receives ∂Loss/∂W scaled by reward.", icon: AlertTriangle, color: "text-neon-red" },
  update: { title: "WEIGHT UPDATE", description: "W_new = W_old - lr × gradient. Hallucination gradients are 2× stronger (asymmetric penalty).", icon: CheckCircle, color: "text-neon-green" },
  done: { title: "STEP COMPLETE", description: "Weights updated. The model is slightly better at this task. Press ▶ for next example.", icon: CheckCircle, color: "text-primary" },
};

const PHASE_ORDER: Phase[] = ["input", "forward", "sensor", "reward", "gradient", "update", "done"];

// ── Weight Bar Component ──────────────────────────────
function WeightBar({ weight, showGradient, showUpdate, lr }: { weight: Weight; showGradient: boolean; showUpdate: boolean; lr: number }) {
  const maxVal = 1.5;
  const barWidth = Math.abs(weight.value) / maxVal * 100;
  const isPositive = weight.value >= 0;
  const gradBarWidth = showGradient ? Math.min(100, Math.abs(weight.gradient) / 0.5 * 100) : 0;
  const gradIsPositive = weight.gradient >= 0;
  const newValue = showUpdate ? weight.value - lr * weight.gradient : weight.value;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-[9px] font-mono text-muted-foreground">{weight.label}</span>
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-foreground">
            {showUpdate ? newValue.toFixed(4) : weight.value.toFixed(4)}
          </span>
          {showGradient && (
            <span className={`text-[9px] font-mono ${weight.gradient > 0 ? "text-neon-red" : "text-neon-green"}`}>
              ∇{weight.gradient > 0 ? "+" : ""}{weight.gradient.toFixed(4)}
            </span>
          )}
        </div>
      </div>
      <div className="h-3 bg-background/50 rounded-sm overflow-hidden relative">
        {/* Weight value bar */}
        <motion.div
          className={`absolute top-0 h-full rounded-sm ${isPositive ? "bg-primary/60 left-1/2" : "bg-secondary/60 right-1/2"}`}
          animate={{ width: `${barWidth / 2}%` }}
          transition={{ duration: 0.4, type: "spring" }}
        />
        {/* Center line */}
        <div className="absolute left-1/2 top-0 w-px h-full bg-muted-foreground/30" />
        {/* Gradient overlay */}
        {showGradient && gradBarWidth > 0 && (
          <motion.div
            className={`absolute top-0 h-full ${gradIsPositive ? "bg-neon-red/30 left-1/2" : "bg-neon-green/30 right-1/2"}`}
            initial={{ width: 0 }}
            animate={{ width: `${gradBarWidth / 2}%` }}
            transition={{ duration: 0.3 }}
          />
        )}
      </div>
      <span className="text-[7px] font-mono text-muted-foreground/50">{weight.layer}</span>
    </div>
  );
}

// ── Pipeline Stage Visualization ──────────────────────
function PipelineStage({ label, active, completed, color, children }: {
  label: string; active: boolean; completed: boolean; color: string; children?: React.ReactNode;
}) {
  return (
    <motion.div
      animate={{
        borderColor: active ? undefined : completed ? "hsl(var(--border))" : "transparent",
        opacity: active ? 1 : completed ? 0.7 : 0.3,
      }}
      className={`glass-panel p-3 rounded-lg border-2 transition-all ${
        active ? `border-${color.replace("text-", "")}/50 shadow-lg` : "border-border/20"
      }`}
    >
      <p className={`text-[9px] font-mono font-bold uppercase mb-1 ${active ? color : "text-muted-foreground"}`}>
        {completed ? "✓ " : ""}{label}
      </p>
      {active && children && (
        <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} className="mt-2">
          {children}
        </motion.div>
      )}
    </motion.div>
  );
}

// ── Main Component ────────────────────────────────────
export default function TruthRLVisualizer() {
  const [state, setState] = useState<LoopState>(() => ({
    phase: "input",
    step: 0,
    input: EXAMPLES[0].input,
    correctAnswer: EXAMPLES[0].correct,
    modelOutput: "",
    outcome: "correct",
    reward: 0,
    energy: 0,
    entropy: 0,
    pIdk: 0,
    weights: initWeights(),
    lossHistory: [],
  }));

  const [autoPlay, setAutoPlay] = useState(false);
  const [speed, setSpeed] = useState(1500);
  const [lr] = useState(0.01);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const currentPhaseIdx = PHASE_ORDER.indexOf(state.phase);
  const phaseInfo = PHASE_INFO[state.phase];

  const advancePhase = useCallback(() => {
    setState(prev => {
      const nextIdx = PHASE_ORDER.indexOf(prev.phase) + 1;

      if (nextIdx >= PHASE_ORDER.length) {
        // Move to next example
        const nextStep = prev.step + 1;
        const example = EXAMPLES[nextStep % EXAMPLES.length];
        return {
          ...prev,
          phase: "input" as Phase,
          step: nextStep,
          input: example.input,
          correctAnswer: example.correct,
          modelOutput: "",
          outcome: "correct",
          reward: 0,
          energy: 0,
          entropy: 0,
          pIdk: 0,
          weights: prev.weights.map(w => ({
            ...w,
            value: w.value - lr * w.gradient,
            gradient: 0,
          })),
        };
      }

      const nextPhase = PHASE_ORDER[nextIdx];
      const example = EXAMPLES[prev.step % EXAMPLES.length];

      switch (nextPhase) {
        case "forward": {
          // Decide model output based on weights (simulated)
          const idkBias = prev.weights.find(w => w.id === "w_idk")!.value;
          const outputIdx = idkBias > 0.3 ? 2 : Math.random() < 0.6 ? 0 : 1;
          const output = example.outputs[outputIdx];
          return { ...prev, phase: nextPhase, modelOutput: output };
        }
        case "sensor": {
          const isCorrect = prev.modelOutput === example.correct;
          const isIdk = prev.modelOutput === "[IDK]";
          const energy = isCorrect ? 0.1 + Math.random() * 0.15 : isIdk ? 0.4 + Math.random() * 0.2 : 0.7 + Math.random() * 0.25;
          const entropy = isCorrect ? 0.05 + Math.random() * 0.1 : isIdk ? 0.3 + Math.random() * 0.15 : 0.6 + Math.random() * 0.3;
          const pIdk = isIdk ? 0.7 + Math.random() * 0.2 : isCorrect ? 0.05 + Math.random() * 0.1 : 0.15 + Math.random() * 0.15;
          return { ...prev, phase: nextPhase, energy, entropy, pIdk };
        }
        case "reward": {
          const isCorrect = prev.modelOutput === example.correct;
          const isIdk = prev.modelOutput === "[IDK]";
          const outcome = isCorrect ? "correct" as const : isIdk ? "abstention" as const : "hallucination" as const;
          const reward = isCorrect ? 1.0 : isIdk ? 0.0 : -1.0;
          return { ...prev, phase: nextPhase, outcome, reward };
        }
        case "gradient": {
          const penaltyMult = prev.outcome === "hallucination" ? 2.0 : 1.0;
          const sign = prev.reward >= 0 ? -1 : 1;
          const weights = prev.weights.map(w => ({
            ...w,
            gradient: sign * penaltyMult * (0.05 + Math.random() * 0.15) * (w.id === "w_idk" && prev.outcome === "abstention" ? 0.5 : 1),
          }));
          return { ...prev, phase: nextPhase, weights };
        }
        case "update": {
          const loss = prev.outcome === "correct" ? 0.2 + Math.random() * 0.3 : prev.outcome === "abstention" ? 0.5 + Math.random() * 0.2 : 1.5 + Math.random() * 0.5;
          return { ...prev, phase: nextPhase, lossHistory: [...prev.lossHistory, loss] };
        }
        default:
          return { ...prev, phase: nextPhase };
      }
    });
  }, [lr]);

  useEffect(() => {
    if (autoPlay) {
      intervalRef.current = setInterval(advancePhase, speed);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [autoPlay, speed, advancePhase]);

  const handleReset = () => {
    setAutoPlay(false);
    setState({
      phase: "input",
      step: 0,
      input: EXAMPLES[0].input,
      correctAnswer: EXAMPLES[0].correct,
      modelOutput: "",
      outcome: "correct",
      reward: 0,
      energy: 0,
      entropy: 0,
      pIdk: 0,
      weights: initWeights(),
      lossHistory: [],
    });
  };

  return (
    <div className="p-8">
      {/* Header */}
      <motion.header initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight text-foreground">
          <GlitchText text="TruthRL LOOP VISUALIZER" className="text-neon-green" />
        </h1>
        <p className="font-mono text-sm text-muted-foreground mt-1">
          Step-by-step weight update visualization • Input → Forward → Sensor → Reward → Gradient → Update
        </p>
      </motion.header>

      {/* Controls */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="glass-panel p-4 rounded-lg mb-6 flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <button onClick={() => setAutoPlay(!autoPlay)} className={`p-2.5 rounded-lg border transition-all ${autoPlay ? "bg-neon-yellow/10 border-neon-yellow/40 text-neon-yellow" : "bg-primary/10 border-primary/40 text-primary"}`}>
            {autoPlay ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>
          <button onClick={advancePhase} className="p-2.5 rounded-lg border border-border/50 text-muted-foreground hover:text-foreground transition-all" title="Next step">
            <SkipForward className="w-4 h-4" />
          </button>
          <button onClick={handleReset} className="p-2.5 rounded-lg border border-border/50 text-muted-foreground hover:text-foreground transition-all">
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>

        <div className="flex flex-col">
          <label className="text-[9px] font-mono text-muted-foreground uppercase">Speed (ms)</label>
          <input type="range" min={500} max={4000} step={100} value={speed} onChange={e => setSpeed(parseInt(e.target.value))} className="w-24 accent-neon-green" />
          <span className="text-[10px] font-mono text-neon-green">{speed}ms</span>
        </div>

        <div className="ml-auto flex items-center gap-4 font-mono text-xs">
          <span className="text-muted-foreground">STEP: <span className="text-foreground">{state.step}</span></span>
          <span className="text-muted-foreground">PHASE: <span className={phaseInfo.color}>{phaseInfo.title}</span></span>
        </div>
      </motion.div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Pipeline — 2/3 */}
        <div className="lg:col-span-2 space-y-3">
          {/* Phase explanation */}
          <motion.div key={state.phase} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} className="glass-panel p-4 rounded-lg border-l-4" style={{ borderLeftColor: `hsl(var(--primary))` }}>
            <div className="flex items-center gap-3">
              <phaseInfo.icon className={`w-5 h-5 ${phaseInfo.color}`} />
              <div>
                <h3 className={`font-mono text-sm font-bold ${phaseInfo.color}`}>{phaseInfo.title}</h3>
                <p className="text-[11px] text-muted-foreground">{phaseInfo.description}</p>
              </div>
              <div className="ml-auto text-[9px] font-mono text-muted-foreground">
                Phase {currentPhaseIdx + 1}/{PHASE_ORDER.length}
              </div>
            </div>
          </motion.div>

          {/* Pipeline Stages */}
          <div className="grid grid-cols-6 gap-2">
            {PHASE_ORDER.slice(0, 6).map((phase, i) => {
              const info = PHASE_INFO[phase];
              const active = state.phase === phase;
              const completed = currentPhaseIdx > i;
              return (
                <PipelineStage key={phase} label={phase} active={active} completed={completed} color={info.color}>
                  {phase === "input" && (
                    <div className="space-y-1">
                      <p className="text-[10px] font-mono text-foreground font-bold">{state.input}</p>
                      <p className="text-[8px] font-mono text-muted-foreground">
                        Bytes: [{state.input.slice(0, 8).split("").map(c => c.charCodeAt(0)).join(", ")}...]
                      </p>
                    </div>
                  )}
                  {phase === "forward" && state.modelOutput && (
                    <div className="space-y-1">
                      <p className="text-[10px] font-mono text-foreground">
                        Output: <span className={state.modelOutput === "[IDK]" ? "text-neon-yellow font-bold" : "text-foreground font-bold"}>{state.modelOutput}</span>
                      </p>
                      <p className="text-[8px] font-mono text-muted-foreground">Expected: {state.correctAnswer}</p>
                    </div>
                  )}
                  {phase === "sensor" && (
                    <div className="space-y-0.5 text-[9px] font-mono">
                      <p>E: <span className="text-primary">{state.energy.toFixed(3)}</span></p>
                      <p>H: <span className="text-secondary">{state.entropy.toFixed(3)}</span></p>
                      <p>P<sub>idk</sub>: <span className="text-neon-yellow">{state.pIdk.toFixed(3)}</span></p>
                    </div>
                  )}
                  {phase === "reward" && (
                    <div className="space-y-1">
                      <p className={`text-[10px] font-mono font-bold ${
                        state.outcome === "correct" ? "text-neon-green" : state.outcome === "abstention" ? "text-neon-yellow" : "text-neon-red"
                      }`}>
                        {state.outcome === "correct" ? "✓ CORRECT" : state.outcome === "abstention" ? "◆ [IDK]" : "✗ HALLUCINATION"}
                      </p>
                      <p className="text-[9px] font-mono text-muted-foreground">
                        R = {state.reward > 0 ? "+" : ""}{state.reward.toFixed(1)}
                        {state.outcome === "hallucination" && " × 2.0"}
                      </p>
                    </div>
                  )}
                  {phase === "gradient" && (
                    <p className="text-[8px] font-mono text-muted-foreground">
                      ∇L flowing...
                      {state.outcome === "hallucination" && <span className="text-neon-red"> (2× penalty)</span>}
                    </p>
                  )}
                  {phase === "update" && (
                    <p className="text-[8px] font-mono text-muted-foreground">
                      W -= {lr} × ∇L
                    </p>
                  )}
                </PipelineStage>
              );
            })}
          </div>

          {/* Outcome Display */}
          <AnimatePresence mode="wait">
            {(state.phase === "reward" || state.phase === "gradient" || state.phase === "update" || state.phase === "done") && (
              <motion.div
                key={state.outcome}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className={`glass-panel p-4 rounded-lg border ${
                  state.outcome === "correct" ? "border-neon-green/30 bg-neon-green/5"
                  : state.outcome === "abstention" ? "border-neon-yellow/30 bg-neon-yellow/5"
                  : "border-neon-red/30 bg-neon-red/5"
                }`}
              >
                <div className="flex items-center gap-3">
                  {state.outcome === "correct" ? <CheckCircle className="w-5 h-5 text-neon-green" /> :
                   state.outcome === "abstention" ? <HelpCircle className="w-5 h-5 text-neon-yellow" /> :
                   <AlertTriangle className="w-5 h-5 text-neon-red" />}
                  <div>
                    <p className="font-mono text-sm font-bold text-foreground">
                      "{state.input}" → "{state.modelOutput}"
                    </p>
                    <p className="text-[10px] font-mono text-muted-foreground">
                      {state.outcome === "correct"
                        ? `Correct! Reward = +1.0. Weights reinforced gently.`
                        : state.outcome === "abstention"
                          ? `Abstained with [IDK]. Reward = 0. Weights unchanged (learning to know what it doesn't know).`
                          : `Hallucinated! Reward = -1.0 × 2.0 penalty. Weights punished strongly.`
                      }
                    </p>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Loss History Sparkline */}
          {state.lossHistory.length > 1 && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass-panel p-3 rounded-lg">
              <p className="text-[9px] font-mono text-muted-foreground mb-2">LOSS HISTORY (last {state.lossHistory.length} steps)</p>
              <svg width="100%" height="40" viewBox={`0 0 ${state.lossHistory.length * 10} 40`} preserveAspectRatio="none">
                <polyline
                  fill="none"
                  stroke="hsl(187 92% 53%)"
                  strokeWidth="1.5"
                  points={state.lossHistory.map((l, i) => `${i * 10},${40 - (l / 2.5) * 40}`).join(" ")}
                />
              </svg>
            </motion.div>
          )}
        </div>

        {/* Weights Panel — 1/3 */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="glass-panel p-4 rounded-lg">
          <div className="flex items-center gap-2 mb-4">
            <Zap className="w-4 h-4 text-neon-yellow" />
            <h3 className="font-mono text-sm text-foreground font-bold">WEIGHT MATRIX</h3>
          </div>

          <div className="space-y-3">
            {state.weights.map(w => (
              <WeightBar
                key={w.id}
                weight={w}
                showGradient={currentPhaseIdx >= PHASE_ORDER.indexOf("gradient")}
                showUpdate={currentPhaseIdx >= PHASE_ORDER.indexOf("update")}
                lr={lr}
              />
            ))}
          </div>

          {/* Legend */}
          <div className="mt-4 p-2 bg-background/30 rounded border border-border/20 space-y-1">
            <p className="text-[8px] font-mono text-muted-foreground">
              <span className="text-primary">■</span> Weight value (pos/neg from center)
            </p>
            <p className="text-[8px] font-mono text-muted-foreground">
              <span className="text-neon-red">■</span> Positive gradient (weight will decrease)
            </p>
            <p className="text-[8px] font-mono text-muted-foreground">
              <span className="text-neon-green">■</span> Negative gradient (weight will increase)
            </p>
            <p className="text-[8px] font-mono text-muted-foreground mt-1">
              lr = {lr} | Penalty mult = {state.outcome === "hallucination" ? "2.0×" : "1.0×"}
            </p>
          </div>

          {/* TruthRL Formula */}
          <div className="mt-4 p-3 bg-background/50 rounded-lg border border-primary/20">
            <p className="text-[9px] font-mono text-primary font-bold mb-1">TruthRL UPDATE RULE:</p>
            <p className="text-[10px] font-mono text-foreground">W<sub>new</sub> = W<sub>old</sub> - lr × ∇L</p>
            <p className="text-[10px] font-mono text-foreground mt-1">
              L = -R × CE(output, target)
            </p>
            <p className="text-[9px] font-mono text-muted-foreground mt-1">
              R ∈ {"{"} +1.0 (correct), 0 (IDK), -2.0 (halluc) {"}"}
            </p>
          </div>
        </motion.div>
      </div>

      {/* Footer */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }} className="mt-4 glass-panel p-4 rounded-lg font-mono text-xs text-muted-foreground flex items-center justify-between">
        <div className="flex gap-6">
          <span>EXAMPLES: <span className="text-primary">{EXAMPLES.length}</span></span>
          <span>COMPLETED: <span className="text-neon-green">{state.step}</span></span>
          <span>AVG LOSS: <span className="text-neon-yellow">
            {state.lossHistory.length > 0 ? (state.lossHistory.reduce((a, b) => a + b, 0) / state.lossHistory.length).toFixed(3) : "—"}
          </span></span>
        </div>
        <span className="text-muted-foreground/60">Based on nanoglass.py TruthRL implementation</span>
      </motion.div>
    </div>
  );
}
