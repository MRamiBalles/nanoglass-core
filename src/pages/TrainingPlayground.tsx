import { useState, useEffect, useRef, useCallback } from "react";
import { motion } from "framer-motion";
import { Play, Pause, RotateCcw, Cpu, TrendingDown, Grid3X3, BarChart3, GitCompareArrows } from "lucide-react";
import { GlitchText } from "@/components/effects/GlitchText";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart, Legend,
} from "recharts";

// ── Types ──────────────────────────────────────────────
interface TrainingConfig {
  lr: number;
  batchSize: number;
  nLayers: number;
  nHeads: number;
  dModel: number;
  architecture: "transformer" | "mamba" | "hybrid" | "moe";
  optimizer: "adamw" | "sgd" | "lion";
  warmupSteps: number;
}

interface TrainingStep {
  step: number;
  loss: number;
  valLoss: number;
  lr: number;
  gradNorm: number;
  energy: number;
  entropy: number;
  perplexity: number;
  truthrlPenalty: number;
}

const DEFAULT_CONFIG: TrainingConfig = {
  lr: 3e-4, batchSize: 64, nLayers: 6, nHeads: 6, dModel: 384,
  architecture: "transformer", optimizer: "adamw", warmupSteps: 50,
};

// ── Simulation Engine ──────────────────────────────────
function simulateStep(step: number, config: TrainingConfig): TrainingStep {
  const t = step;
  const noise = () => (Math.random() - 0.5) * 0.02;
  const warmup = Math.min(1, t / Math.max(1, config.warmupSteps));
  const cosineDecay = 0.5 * (1 + Math.cos(Math.PI * Math.min(t / 500, 1)));
  const currentLr = config.lr * warmup * (0.1 + 0.9 * cosineDecay);
  const archFactor = { transformer: 1.0, mamba: 0.85, hybrid: 0.75, moe: 0.9 }[config.architecture];
  const depthFactor = 1 - 0.02 * (config.nLayers - 4);
  const baseLoss = 4.5 * Math.exp(-t * 0.008 * (1 / archFactor)) + 0.3 * archFactor + depthFactor * 0.1;
  const loss = Math.max(0.15, baseLoss + noise() * (1 + 2 / (1 + t * 0.01)));
  const valSpike = Math.random() < 0.03 ? 0.3 : 0;
  const valLoss = loss * (1.05 + 0.1 * Math.sin(t * 0.05)) + valSpike + Math.abs(noise());
  const gradNorm = 5 * Math.exp(-t * 0.015) + 0.5 + noise() * 2;
  const energy = 0.9 * Math.exp(-t * 0.005) + 0.08 + noise();
  const entropy = 0.8 * Math.exp(-t * 0.003) + 0.15 + noise() * 0.5;
  const truthrlPenalty = Math.max(0, 0.4 * Math.exp(-t * 0.012) + noise() * 0.1);
  const perplexity = Math.exp(loss);
  return { step: t, loss, valLoss, lr: currentLr, gradNorm, energy, entropy, perplexity, truthrlPenalty };
}

// ── Attention Heatmap ──────────────────────────────────
function AttentionHeatmap({ step, nHeads }: { step: number; nHeads: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const seqLen = 16;
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const cellW = canvas.width / seqLen;
    const cellH = canvas.height / seqLen;
    const sharpness = 1 + step * 0.05;
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < seqLen; j++) {
        if (j > i) {
          ctx.fillStyle = "hsl(222 47% 4%)";
        } else {
          const dist = Math.abs(i - j);
          const rawWeight = Math.exp(-dist * 0.3 * sharpness) + (Math.random() * 0.15 / sharpness);
          const weight = Math.min(1, rawWeight);
          const hue = 187 + (1 - weight) * 83;
          const lightness = 20 + weight * 40;
          ctx.fillStyle = `hsl(${hue} 85% ${lightness}%)`;
        }
        ctx.fillRect(j * cellW, i * cellH, cellW - 0.5, cellH - 0.5);
      }
    }
  }, [step, nHeads]);
  return <canvas ref={canvasRef} width={256} height={256} className="w-full h-full rounded-md" style={{ imageRendering: "pixelated" }} />;
}

// ── Token Distribution ────────────────────────────────
function TokenDistribution({ step }: { step: number }) {
  const tokens = ["the", "of", "to", "and", "a", "in", "[IDK]", "is", "for", "that"];
  const sharpness = 1 + step * 0.02;
  const rawProbs = tokens.map((_, i) => Math.pow(1 / (i + 1), sharpness) + Math.random() * 0.02);
  const sum = rawProbs.reduce((a, b) => a + b, 0);
  const probs = rawProbs.map(p => p / sum);
  return (
    <div className="space-y-1.5">
      {tokens.map((token, i) => {
        const prob = probs[i];
        const isIdk = token === "[IDK]";
        return (
          <div key={token} className="flex items-center gap-2 font-mono text-[10px]">
            <span className={`w-10 text-right ${isIdk ? "text-neon-yellow" : "text-muted-foreground"}`}>{token}</span>
            <div className="flex-1 h-3 bg-background/50 rounded-sm overflow-hidden">
              <motion.div className={`h-full rounded-sm ${isIdk ? "bg-neon-yellow/80" : "bg-primary/60"}`} initial={{ width: 0 }} animate={{ width: `${prob * 100}%` }} transition={{ duration: 0.3 }} />
            </div>
            <span className="w-12 text-right text-muted-foreground">{(prob * 100).toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}

// ── Gradient Flow ─────────────────────────────────────
function GradientFlow({ step, nLayers }: { step: number; nLayers: number }) {
  const layers = Array.from({ length: nLayers }, (_, i) => {
    const depth = i / nLayers;
    const vanishing = Math.exp(-depth * 3 / (1 + step * 0.02));
    const magnitude = vanishing * (0.5 + Math.random() * 0.3);
    return { layer: i, magnitude };
  });
  const maxMag = Math.max(...layers.map(l => l.magnitude));
  return (
    <div className="flex items-end gap-1 h-24">
      {layers.map(({ layer, magnitude }) => {
        const normalized = magnitude / (maxMag || 1);
        const hue = normalized > 0.3 ? 142 : normalized > 0.1 ? 45 : 0;
        return (
          <div key={layer} className="flex-1 flex flex-col items-center gap-1">
            <motion.div className="w-full rounded-t-sm" style={{ backgroundColor: `hsl(${hue} 76% 45%)` }} initial={{ height: 0 }} animate={{ height: `${normalized * 100}%` }} transition={{ duration: 0.3 }} />
            <span className="text-[8px] font-mono text-muted-foreground">{layer}</span>
          </div>
        );
      })}
    </div>
  );
}

// ── Config Controls (reusable for A/B) ────────────────
function ConfigControls({
  config, onChange, label, color,
}: {
  config: TrainingConfig;
  onChange: (key: keyof TrainingConfig, value: number | string) => void;
  label: string;
  color: string;
}) {
  return (
    <div className="glass-panel p-3 rounded-lg space-y-2">
      <div className="flex items-center gap-2 mb-1">
        <div className={`w-3 h-3 rounded-full ${color}`} />
        <span className="font-mono text-xs font-bold text-foreground">{label}</span>
      </div>
      <div className="flex flex-wrap gap-3">
        <div className="flex flex-col">
          <label className="text-[8px] font-mono text-muted-foreground uppercase">Arch</label>
          <select value={config.architecture} onChange={e => onChange("architecture", e.target.value)} className="bg-background/50 border border-border/50 rounded px-1.5 py-0.5 text-[10px] font-mono text-foreground">
            <option value="transformer">Transformer</option>
            <option value="mamba">Mamba-2</option>
            <option value="hybrid">Hybrid</option>
            <option value="moe">MoE</option>
          </select>
        </div>
        <div className="flex flex-col">
          <label className="text-[8px] font-mono text-muted-foreground uppercase">LR</label>
          <input type="range" min={-5} max={-2} step={0.1} value={Math.log10(config.lr)} onChange={e => onChange("lr", Math.pow(10, parseFloat(e.target.value)))} className="w-16 accent-current" />
          <span className="text-[9px] font-mono text-muted-foreground">{config.lr.toExponential(1)}</span>
        </div>
        <div className="flex flex-col">
          <label className="text-[8px] font-mono text-muted-foreground uppercase">Layers</label>
          <input type="range" min={2} max={12} step={1} value={config.nLayers} onChange={e => onChange("nLayers", parseInt(e.target.value))} className="w-14 accent-current" />
          <span className="text-[9px] font-mono text-muted-foreground">{config.nLayers}</span>
        </div>
        <div className="flex flex-col">
          <label className="text-[8px] font-mono text-muted-foreground uppercase">Heads</label>
          <input type="range" min={1} max={12} step={1} value={config.nHeads} onChange={e => onChange("nHeads", parseInt(e.target.value))} className="w-14 accent-current" />
          <span className="text-[9px] font-mono text-muted-foreground">{config.nHeads}</span>
        </div>
      </div>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────
export default function TrainingPlayground() {
  const [mode, setMode] = useState<"single" | "ab">("single");

  // Config A (single mode uses this)
  const [configA, setConfigA] = useState<TrainingConfig>({ ...DEFAULT_CONFIG });
  const [historyA, setHistoryA] = useState<TrainingStep[]>([]);
  const stepRefA = useRef(0);

  // Config B (A/B mode only)
  const [configB, setConfigB] = useState<TrainingConfig>({ ...DEFAULT_CONFIG, architecture: "mamba", lr: 1e-3 });
  const [historyB, setHistoryB] = useState<TrainingStep[]>([]);
  const stepRefB = useRef(0);

  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(100);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const currentA = historyA.length > 0 ? historyA[historyA.length - 1] : null;
  const currentB = historyB.length > 0 ? historyB[historyB.length - 1] : null;

  const tick = useCallback(() => {
    setHistoryA(prev => {
      const next = simulateStep(stepRefA.current, configA);
      stepRefA.current += 1;
      return [...prev, next];
    });
    if (mode === "ab") {
      setHistoryB(prev => {
        const next = simulateStep(stepRefB.current, configB);
        stepRefB.current += 1;
        return [...prev, next];
      });
    }
  }, [configA, configB, mode]);

  useEffect(() => {
    if (running) {
      intervalRef.current = setInterval(tick, speed);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [running, speed, tick]);

  const handleReset = () => {
    setRunning(false);
    stepRefA.current = 0;
    stepRefB.current = 0;
    setHistoryA([]);
    setHistoryB([]);
  };

  const updateConfigA = (key: keyof TrainingConfig, value: number | string) => setConfigA(prev => ({ ...prev, [key]: value }));
  const updateConfigB = (key: keyof TrainingConfig, value: number | string) => setConfigB(prev => ({ ...prev, [key]: value }));

  // Merge chart data for A/B overlay
  const maxSteps = Math.max(historyA.length, historyB.length);
  const chartData = Array.from({ length: Math.min(maxSteps, 200) }, (_, i) => {
    const idx = Math.max(0, maxSteps - 200) + i;
    const a = historyA[idx];
    const b = historyB[idx];
    return {
      step: a?.step ?? b?.step ?? idx,
      lossA: a?.loss,
      valLossA: a?.valLoss,
      lossB: b?.loss,
      valLossB: b?.valLoss,
      truthrlA: a?.truthrlPenalty,
      truthrlB: b?.truthrlPenalty,
    };
  });

  // Single mode chart data
  const singleChartData = historyA.slice(-200);

  return (
    <div className="p-8">
      {/* Header */}
      <motion.header initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight text-foreground">
          <GlitchText text="TRAINING PLAYGROUND" className="text-primary" />
        </h1>
        <p className="font-mono text-sm text-muted-foreground mt-1">
          nanoGPT-style training loop • {mode === "ab" ? "A/B Comparator Mode" : "Real-time loss, attention & gradient visualization"}
        </p>
      </motion.header>

      {/* Controls Bar */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="glass-panel p-4 rounded-lg mb-6 flex flex-wrap items-center gap-4">
        {/* Play / Pause / Reset */}
        <div className="flex items-center gap-2">
          <button onClick={() => setRunning(!running)} className={`p-2.5 rounded-lg border transition-all ${running ? "bg-neon-yellow/10 border-neon-yellow/40 text-neon-yellow" : "bg-primary/10 border-primary/40 text-primary"}`}>
            {running ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>
          <button onClick={handleReset} className="p-2.5 rounded-lg border border-border/50 text-muted-foreground hover:text-foreground transition-all">
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>

        {/* Mode Toggle */}
        <button
          onClick={() => { setMode(m => m === "single" ? "ab" : "single"); handleReset(); }}
          className={`flex items-center gap-2 px-3 py-2 rounded-lg border transition-all font-mono text-xs ${
            mode === "ab" ? "bg-neon-purple/10 border-neon-purple/40 text-neon-purple" : "border-border/50 text-muted-foreground hover:text-foreground"
          }`}
        >
          <GitCompareArrows className="w-4 h-4" />
          {mode === "ab" ? "A/B MODE" : "SINGLE"}
        </button>

        {/* Single mode controls */}
        {mode === "single" && (
          <>
            <div className="flex flex-col">
              <label className="text-[9px] font-mono text-muted-foreground uppercase">Architecture</label>
              <select value={configA.architecture} onChange={e => updateConfigA("architecture", e.target.value)} className="bg-background/50 border border-border/50 rounded px-2 py-1 text-xs font-mono text-foreground">
                <option value="transformer">Transformer (GPT-2)</option>
                <option value="mamba">Mamba-2 (SSM)</option>
                <option value="hybrid">Hybrid (GQA+Mamba)</option>
                <option value="moe">MoE (8 Experts)</option>
              </select>
            </div>
            <div className="flex flex-col">
              <label className="text-[9px] font-mono text-muted-foreground uppercase">Learning Rate</label>
              <input type="range" min={-5} max={-2} step={0.1} value={Math.log10(configA.lr)} onChange={e => updateConfigA("lr", Math.pow(10, parseFloat(e.target.value)))} className="w-24 accent-primary" />
              <span className="text-[10px] font-mono text-primary">{configA.lr.toExponential(1)}</span>
            </div>
            <div className="flex flex-col">
              <label className="text-[9px] font-mono text-muted-foreground uppercase">Layers</label>
              <input type="range" min={2} max={12} step={1} value={configA.nLayers} onChange={e => updateConfigA("nLayers", parseInt(e.target.value))} className="w-20 accent-secondary" />
              <span className="text-[10px] font-mono text-secondary">{configA.nLayers}</span>
            </div>
            <div className="flex flex-col">
              <label className="text-[9px] font-mono text-muted-foreground uppercase">Heads</label>
              <input type="range" min={1} max={12} step={1} value={configA.nHeads} onChange={e => updateConfigA("nHeads", parseInt(e.target.value))} className="w-20 accent-secondary" />
              <span className="text-[10px] font-mono text-secondary">{configA.nHeads}</span>
            </div>
          </>
        )}

        {/* Speed */}
        <div className="flex flex-col">
          <label className="text-[9px] font-mono text-muted-foreground uppercase">Speed (ms)</label>
          <input type="range" min={20} max={500} step={10} value={speed} onChange={e => setSpeed(parseInt(e.target.value))} className="w-20 accent-neon-green" />
          <span className="text-[10px] font-mono text-neon-green">{speed}ms</span>
        </div>

        {/* Step Counter */}
        <div className="ml-auto flex items-center gap-4 font-mono text-xs">
          <span className="text-muted-foreground">STEP: <span className="text-foreground">{stepRefA.current}</span></span>
          {currentA && <span className="text-muted-foreground">LOSS{mode === "ab" ? " A" : ""}: <span className="text-primary">{currentA.loss.toFixed(4)}</span></span>}
          {mode === "ab" && currentB && <span className="text-muted-foreground">LOSS B: <span className="text-neon-purple">{currentB.loss.toFixed(4)}</span></span>}
        </div>
      </motion.div>

      {/* A/B Config Panels */}
      {mode === "ab" && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <ConfigControls config={configA} onChange={updateConfigA} label="CONFIG A" color="bg-primary" />
          <ConfigControls config={configB} onChange={updateConfigB} label="CONFIG B" color="bg-neon-purple" />
        </div>
      )}

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Loss Curves — 2/3 */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="lg:col-span-2 glass-panel p-4 rounded-lg">
          <div className="flex items-center gap-2 mb-3">
            <TrendingDown className="w-4 h-4 text-primary" />
            <h3 className="font-mono text-sm text-foreground font-bold">
              {mode === "ab" ? "A/B LOSS COMPARISON" : "LOSS CURVES"}
            </h3>
            <span className="text-[9px] font-mono text-muted-foreground ml-auto">
              {mode === "ab"
                ? "A train (cyan) • A val (cyan dashed) • B train (purple) • B val (purple dashed)"
                : "train (cyan) • val (purple) • truthRL penalty (yellow)"
              }
            </span>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              {mode === "ab" ? (
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 33% 15%)" />
                  <XAxis dataKey="step" stroke="hsl(215 20% 35%)" tick={{ fontSize: 10, fontFamily: "monospace" }} />
                  <YAxis stroke="hsl(215 20% 35%)" tick={{ fontSize: 10, fontFamily: "monospace" }} domain={[0, "auto"]} />
                  <Tooltip contentStyle={{ backgroundColor: "hsl(217 33% 8%)", border: "1px solid hsl(187 92% 53% / 0.3)", borderRadius: "8px", fontFamily: "monospace", fontSize: "11px" }} />
                  <Legend wrapperStyle={{ fontFamily: "monospace", fontSize: "10px" }} />
                  <Line type="monotone" dataKey="lossA" stroke="hsl(187 92% 53%)" strokeWidth={2} dot={false} name={`A: ${configA.architecture}`} />
                  <Line type="monotone" dataKey="valLossA" stroke="hsl(187 92% 53%)" strokeWidth={1} strokeDasharray="5 3" dot={false} name="A: val" />
                  <Line type="monotone" dataKey="lossB" stroke="hsl(270 91% 75%)" strokeWidth={2} dot={false} name={`B: ${configB.architecture}`} />
                  <Line type="monotone" dataKey="valLossB" stroke="hsl(270 91% 75%)" strokeWidth={1} strokeDasharray="5 3" dot={false} name="B: val" />
                </LineChart>
              ) : (
                <AreaChart data={singleChartData}>
                  <defs>
                    <linearGradient id="trainGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="hsl(187 92% 53%)" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="hsl(187 92% 53%)" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 33% 15%)" />
                  <XAxis dataKey="step" stroke="hsl(215 20% 35%)" tick={{ fontSize: 10, fontFamily: "monospace" }} />
                  <YAxis stroke="hsl(215 20% 35%)" tick={{ fontSize: 10, fontFamily: "monospace" }} domain={[0, "auto"]} />
                  <Tooltip contentStyle={{ backgroundColor: "hsl(217 33% 8%)", border: "1px solid hsl(187 92% 53% / 0.3)", borderRadius: "8px", fontFamily: "monospace", fontSize: "11px" }} />
                  <Area type="monotone" dataKey="loss" stroke="hsl(187 92% 53%)" fill="url(#trainGrad)" strokeWidth={2} dot={false} name="Train Loss" />
                  <Line type="monotone" dataKey="valLoss" stroke="hsl(270 91% 75%)" strokeWidth={1.5} strokeDasharray="5 3" dot={false} name="Val Loss" />
                  <Line type="monotone" dataKey="truthrlPenalty" stroke="hsl(45 93% 47%)" strokeWidth={1} dot={false} name="TruthRL Penalty" />
                </AreaChart>
              )}
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Right column: Metrics */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="space-y-4">
          {mode === "ab" ? (
            <>
              {/* A/B Metrics Comparison */}
              <div className="glass-panel p-4 rounded-lg">
                <div className="flex items-center gap-2 mb-3">
                  <GitCompareArrows className="w-4 h-4 text-neon-purple" />
                  <h3 className="font-mono text-sm text-foreground font-bold">A vs B</h3>
                </div>
                <div className="space-y-2">
                  {[
                    { label: "Loss", a: currentA?.loss, b: currentB?.loss },
                    { label: "Val Loss", a: currentA?.valLoss, b: currentB?.valLoss },
                    { label: "Perplexity", a: currentA?.perplexity, b: currentB?.perplexity },
                    { label: "Energy", a: currentA?.energy, b: currentB?.energy },
                    { label: "Entropy", a: currentA?.entropy, b: currentB?.entropy },
                    { label: "Grad Norm", a: currentA?.gradNorm, b: currentB?.gradNorm },
                    { label: "TruthRL", a: currentA?.truthrlPenalty, b: currentB?.truthrlPenalty },
                  ].map(({ label, a, b }) => {
                    const aVal = a ?? 0;
                    const bVal = b ?? 0;
                    const winner = aVal < bVal ? "A" : bVal < aVal ? "B" : "=";
                    return (
                      <div key={label} className="grid grid-cols-[60px_1fr_20px_1fr] gap-1 items-center">
                        <span className="text-[8px] font-mono text-muted-foreground uppercase">{label}</span>
                        <span className={`text-[10px] font-mono text-right ${winner === "A" ? "text-primary font-bold" : "text-muted-foreground"}`}>
                          {a?.toFixed(4) ?? "—"}
                        </span>
                        <span className={`text-[9px] font-mono text-center ${winner === "A" ? "text-primary" : winner === "B" ? "text-neon-purple" : "text-muted-foreground"}`}>
                          {winner === "A" ? "◀" : winner === "B" ? "▶" : "="}
                        </span>
                        <span className={`text-[10px] font-mono ${winner === "B" ? "text-neon-purple font-bold" : "text-muted-foreground"}`}>
                          {b?.toFixed(4) ?? "—"}
                        </span>
                      </div>
                    );
                  })}
                </div>
                {currentA && currentB && (
                  <div className="mt-3 p-2 rounded bg-background/30 border border-border/20">
                    <p className="text-[9px] font-mono text-muted-foreground">
                      Δ Loss: <span className={currentA.loss < currentB.loss ? "text-primary" : "text-neon-purple"}>
                        {Math.abs(currentA.loss - currentB.loss).toFixed(4)}
                      </span>
                      {" "}({currentA.loss < currentB.loss ? "A wins" : "B wins"})
                    </p>
                    <p className="text-[9px] font-mono text-muted-foreground">
                      Δ PPL: <span className={currentA.perplexity < currentB.perplexity ? "text-primary" : "text-neon-purple"}>
                        {Math.abs(currentA.perplexity - currentB.perplexity).toFixed(1)}
                      </span>
                    </p>
                  </div>
                )}
              </div>
            </>
          ) : (
            /* Single mode metrics */
            <div className="glass-panel p-4 rounded-lg">
              <div className="flex items-center gap-2 mb-3">
                <Cpu className="w-4 h-4 text-neon-green" />
                <h3 className="font-mono text-sm text-foreground font-bold">LIVE METRICS</h3>
              </div>
              {currentA ? (
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { label: "Loss", value: currentA.loss.toFixed(4), color: "text-primary" },
                    { label: "Val Loss", value: currentA.valLoss.toFixed(4), color: "text-secondary" },
                    { label: "Perplexity", value: currentA.perplexity.toFixed(1), color: "text-neon-yellow" },
                    { label: "Grad Norm", value: currentA.gradNorm.toFixed(3), color: "text-neon-green" },
                    { label: "Energy (L1)", value: currentA.energy.toFixed(4), color: "text-primary" },
                    { label: "Entropy", value: currentA.entropy.toFixed(4), color: "text-secondary" },
                    { label: "LR", value: currentA.lr.toExponential(2), color: "text-muted-foreground" },
                    { label: "TruthRL", value: currentA.truthrlPenalty.toFixed(4), color: "text-neon-yellow" },
                  ].map(({ label, value, color }) => (
                    <div key={label} className="bg-background/30 rounded p-2">
                      <p className="text-[9px] font-mono text-muted-foreground uppercase">{label}</p>
                      <p className={`text-sm font-mono font-bold ${color}`}>{value}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs font-mono text-muted-foreground text-center py-6">Press ▶ to start training</p>
              )}
            </div>
          )}
        </motion.div>
      </div>

      {/* Bottom Row: Attention, Gradient Flow, Token Distribution */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }} className="glass-panel p-4 rounded-lg">
          <div className="flex items-center gap-2 mb-3">
            <Grid3X3 className="w-4 h-4 text-primary" />
            <h3 className="font-mono text-sm text-foreground font-bold">ATTENTION MAP {mode === "ab" ? "(A)" : ""}</h3>
          </div>
          <div className="aspect-square">
            <AttentionHeatmap step={stepRefA.current} nHeads={configA.nHeads} />
          </div>
          <div className="flex justify-between mt-2 text-[9px] font-mono text-muted-foreground">
            <span>Query →</span>
            <span>Head 0 / {configA.nHeads}</span>
            <span>Key ↓</span>
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }} className="glass-panel p-4 rounded-lg">
          <div className="flex items-center gap-2 mb-3">
            <BarChart3 className="w-4 h-4 text-neon-green" />
            <h3 className="font-mono text-sm text-foreground font-bold">GRADIENT FLOW {mode === "ab" ? "(A)" : ""}</h3>
          </div>
          <GradientFlow step={stepRefA.current} nLayers={configA.nLayers} />
          <p className="text-[9px] font-mono text-muted-foreground text-center mt-2">
            Layer index → | ■ healthy ■ weak ■ vanishing
          </p>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }} className="glass-panel p-4 rounded-lg">
          <div className="flex items-center gap-2 mb-3">
            <BarChart3 className="w-4 h-4 text-neon-yellow" />
            <h3 className="font-mono text-sm text-foreground font-bold">TOKEN PROBS</h3>
          </div>
          <TokenDistribution step={stepRefA.current} />
        </motion.div>
      </div>

      {/* A/B Bottom Row: Config B visualizations */}
      {mode === "ab" && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-panel p-4 rounded-lg">
            <div className="flex items-center gap-2 mb-3">
              <Grid3X3 className="w-4 h-4 text-neon-purple" />
              <h3 className="font-mono text-sm text-foreground font-bold">ATTENTION MAP (B)</h3>
            </div>
            <div className="aspect-square max-h-48">
              <AttentionHeatmap step={stepRefB.current} nHeads={configB.nHeads} />
            </div>
          </motion.div>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-panel p-4 rounded-lg">
            <div className="flex items-center gap-2 mb-3">
              <BarChart3 className="w-4 h-4 text-neon-purple" />
              <h3 className="font-mono text-sm text-foreground font-bold">GRADIENT FLOW (B)</h3>
            </div>
            <GradientFlow step={stepRefB.current} nLayers={configB.nLayers} />
          </motion.div>
        </div>
      )}

      {/* Footer */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.7 }} className="mt-4 glass-panel p-4 rounded-lg">
        <div className="flex items-center justify-between font-mono text-xs text-muted-foreground">
          <div className="flex items-center gap-6">
            {mode === "ab" ? (
              <>
                <span>A: <span className="text-primary uppercase">{configA.architecture}</span> ({((configA.nLayers * configA.dModel * configA.dModel * 4 * 2 + configA.dModel * 257) / 1e6).toFixed(1)}M)</span>
                <span>B: <span className="text-neon-purple uppercase">{configB.architecture}</span> ({((configB.nLayers * configB.dModel * configB.dModel * 4 * 2 + configB.dModel * 257) / 1e6).toFixed(1)}M)</span>
              </>
            ) : (
              <>
                <span>ARCH: <span className="text-primary uppercase">{configA.architecture}</span></span>
                <span>PARAMS: <span className="text-secondary">{((configA.nLayers * configA.dModel * configA.dModel * 4 * 2 + configA.dModel * 257) / 1e6).toFixed(1)}M</span></span>
                <span>d_model: <span className="text-foreground">{configA.dModel}</span></span>
              </>
            )}
          </div>
          <span className="text-muted-foreground/60">
            Ref: <a href="https://github.com/karpathy/nanoGPT" target="_blank" rel="noopener" className="text-primary/60 hover:text-primary">karpathy/nanoGPT</a>
          </span>
        </div>
      </motion.div>
    </div>
  );
}
