import { useState, useEffect, useRef, useCallback } from "react";
import { motion } from "framer-motion";
import { Play, Pause, RotateCcw, Cpu, TrendingDown, Grid3X3, BarChart3 } from "lucide-react";
import { GlitchText } from "@/components/effects/GlitchText";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart,
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

// ── Simulation Engine ──────────────────────────────────
function simulateStep(step: number, config: TrainingConfig, prev: TrainingStep | null): TrainingStep {
  const t = step;
  const noise = () => (Math.random() - 0.5) * 0.02;

  // Cosine schedule with warmup
  const warmup = Math.min(1, t / Math.max(1, config.warmupSteps));
  const cosineDecay = 0.5 * (1 + Math.cos(Math.PI * Math.min(t / 500, 1)));
  const currentLr = config.lr * warmup * (0.1 + 0.9 * cosineDecay);

  // Architecture-specific convergence rates
  const archFactor = { transformer: 1.0, mamba: 0.85, hybrid: 0.75, moe: 0.9 }[config.architecture];
  const depthFactor = 1 - 0.02 * (config.nLayers - 4); // deeper = slower start, lower floor

  // Loss: exponential decay with noise, architecture modulates floor
  const baseLoss = 4.5 * Math.exp(-t * 0.008 * (1 / archFactor)) + 0.3 * archFactor + depthFactor * 0.1;
  const loss = Math.max(0.15, baseLoss + noise() * (1 + 2 / (1 + t * 0.01)));

  // Val loss: slightly higher, occasional spikes
  const valSpike = Math.random() < 0.03 ? 0.3 : 0;
  const valLoss = loss * (1.05 + 0.1 * Math.sin(t * 0.05)) + valSpike + Math.abs(noise());

  // Gradient norm: high at start, stabilizes
  const gradNorm = 5 * Math.exp(-t * 0.015) + 0.5 + noise() * 2;

  // Energy (L1 activation norm): drops with training
  const energy = 0.9 * Math.exp(-t * 0.005) + 0.08 + noise();

  // Entropy: starts high, compresses
  const entropy = 0.8 * Math.exp(-t * 0.003) + 0.15 + noise() * 0.5;

  // TruthRL penalty: spiky early, near-zero late
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

    // Generate causal attention pattern with training-dependent sharpness
    const sharpness = 1 + step * 0.05; // attention sharpens over training

    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < seqLen; j++) {
        if (j > i) {
          // Causal mask: future tokens are black
          ctx.fillStyle = "hsl(222 47% 4%)";
        } else {
          // Attention weight: diagonal-biased + positional decay
          const dist = Math.abs(i - j);
          const rawWeight = Math.exp(-dist * 0.3 * sharpness) + (Math.random() * 0.15 / sharpness);
          const weight = Math.min(1, rawWeight);

          // Cyan-to-purple gradient based on weight
          const hue = 187 + (1 - weight) * 83; // 187 (cyan) → 270 (purple)
          const lightness = 20 + weight * 40;
          ctx.fillStyle = `hsl(${hue} 85% ${lightness}%)`;
        }
        ctx.fillRect(j * cellW, i * cellH, cellW - 0.5, cellH - 0.5);
      }
    }
  }, [step, nHeads]);

  return (
    <canvas
      ref={canvasRef}
      width={256}
      height={256}
      className="w-full h-full rounded-md"
      style={{ imageRendering: "pixelated" }}
    />
  );
}

// ── Token Distribution Bar ────────────────────────────
function TokenDistribution({ step }: { step: number }) {
  const tokens = ["the", "of", "to", "and", "a", "in", "[IDK]", "is", "for", "that"];
  const sharpness = 1 + step * 0.02;

  // Zipf-like distribution that sharpens with training
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
            <span className={`w-10 text-right ${isIdk ? "text-neon-yellow" : "text-muted-foreground"}`}>
              {token}
            </span>
            <div className="flex-1 h-3 bg-background/50 rounded-sm overflow-hidden">
              <motion.div
                className={`h-full rounded-sm ${isIdk ? "bg-neon-yellow/80" : "bg-primary/60"}`}
                initial={{ width: 0 }}
                animate={{ width: `${prob * 100}%` }}
                transition={{ duration: 0.3 }}
              />
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
    // Gradient magnitude: vanishes in deep layers early, stabilizes later
    const vanishing = Math.exp(-depth * 3 / (1 + step * 0.02));
    const magnitude = vanishing * (0.5 + Math.random() * 0.3);
    return { layer: i, magnitude };
  });

  const maxMag = Math.max(...layers.map(l => l.magnitude));

  return (
    <div className="flex items-end gap-1 h-24">
      {layers.map(({ layer, magnitude }) => {
        const normalized = magnitude / (maxMag || 1);
        const hue = normalized > 0.3 ? 142 : normalized > 0.1 ? 45 : 0; // green/yellow/red
        return (
          <div key={layer} className="flex-1 flex flex-col items-center gap-1">
            <motion.div
              className="w-full rounded-t-sm"
              style={{ backgroundColor: `hsl(${hue} 76% 45%)` }}
              initial={{ height: 0 }}
              animate={{ height: `${normalized * 100}%` }}
              transition={{ duration: 0.3 }}
            />
            <span className="text-[8px] font-mono text-muted-foreground">{layer}</span>
          </div>
        );
      })}
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────
export default function TrainingPlayground() {
  const [config, setConfig] = useState<TrainingConfig>({
    lr: 3e-4,
    batchSize: 64,
    nLayers: 6,
    nHeads: 6,
    dModel: 384,
    architecture: "transformer",
    optimizer: "adamw",
    warmupSteps: 50,
  });

  const [history, setHistory] = useState<TrainingStep[]>([]);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(100); // ms per step
  const stepRef = useRef(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const currentStep = history.length > 0 ? history[history.length - 1] : null;

  const tick = useCallback(() => {
    setHistory(prev => {
      const last = prev.length > 0 ? prev[prev.length - 1] : null;
      const next = simulateStep(stepRef.current, config, last);
      stepRef.current += 1;
      return [...prev, next];
    });
  }, [config]);

  useEffect(() => {
    if (running) {
      intervalRef.current = setInterval(tick, speed);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [running, speed, tick]);

  const handleReset = () => {
    setRunning(false);
    stepRef.current = 0;
    setHistory([]);
  };

  const updateConfig = (key: keyof TrainingConfig, value: number | string) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  // Chart data: last 200 steps
  const chartData = history.slice(-200);

  return (
    <div className="p-8">
      {/* Header */}
      <motion.header initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight text-foreground">
          <GlitchText text="TRAINING PLAYGROUND" className="text-primary" />
        </h1>
        <p className="font-mono text-sm text-muted-foreground mt-1">
          nanoGPT-style training loop • Real-time loss, attention & gradient visualization
        </p>
      </motion.header>

      {/* Controls Bar */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="glass-panel p-4 rounded-lg mb-6 flex flex-wrap items-center gap-4"
      >
        {/* Play / Pause / Reset */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setRunning(!running)}
            className={`p-2.5 rounded-lg border transition-all ${
              running
                ? "bg-neon-yellow/10 border-neon-yellow/40 text-neon-yellow"
                : "bg-primary/10 border-primary/40 text-primary"
            }`}
          >
            {running ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>
          <button
            onClick={handleReset}
            className="p-2.5 rounded-lg border border-border/50 text-muted-foreground hover:text-foreground transition-all"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>

        {/* Architecture */}
        <div className="flex flex-col">
          <label className="text-[9px] font-mono text-muted-foreground uppercase">Architecture</label>
          <select
            value={config.architecture}
            onChange={e => updateConfig("architecture", e.target.value)}
            className="bg-background/50 border border-border/50 rounded px-2 py-1 text-xs font-mono text-foreground"
          >
            <option value="transformer">Transformer (GPT-2)</option>
            <option value="mamba">Mamba-2 (SSM)</option>
            <option value="hybrid">Hybrid (GQA+Mamba)</option>
            <option value="moe">MoE (8 Experts)</option>
          </select>
        </div>

        {/* LR */}
        <div className="flex flex-col">
          <label className="text-[9px] font-mono text-muted-foreground uppercase">Learning Rate</label>
          <input
            type="range"
            min={-5}
            max={-2}
            step={0.1}
            value={Math.log10(config.lr)}
            onChange={e => updateConfig("lr", Math.pow(10, parseFloat(e.target.value)))}
            className="w-24 accent-primary"
          />
          <span className="text-[10px] font-mono text-primary">{config.lr.toExponential(1)}</span>
        </div>

        {/* Layers */}
        <div className="flex flex-col">
          <label className="text-[9px] font-mono text-muted-foreground uppercase">Layers</label>
          <input
            type="range"
            min={2}
            max={12}
            step={1}
            value={config.nLayers}
            onChange={e => updateConfig("nLayers", parseInt(e.target.value))}
            className="w-20 accent-secondary"
          />
          <span className="text-[10px] font-mono text-secondary">{config.nLayers}</span>
        </div>

        {/* Heads */}
        <div className="flex flex-col">
          <label className="text-[9px] font-mono text-muted-foreground uppercase">Heads</label>
          <input
            type="range"
            min={1}
            max={12}
            step={1}
            value={config.nHeads}
            onChange={e => updateConfig("nHeads", parseInt(e.target.value))}
            className="w-20 accent-secondary"
          />
          <span className="text-[10px] font-mono text-secondary">{config.nHeads}</span>
        </div>

        {/* Speed */}
        <div className="flex flex-col">
          <label className="text-[9px] font-mono text-muted-foreground uppercase">Speed (ms)</label>
          <input
            type="range"
            min={20}
            max={500}
            step={10}
            value={speed}
            onChange={e => setSpeed(parseInt(e.target.value))}
            className="w-20 accent-neon-green"
          />
          <span className="text-[10px] font-mono text-neon-green">{speed}ms</span>
        </div>

        {/* Step Counter */}
        <div className="ml-auto flex items-center gap-4 font-mono text-xs">
          <span className="text-muted-foreground">STEP: <span className="text-foreground">{stepRef.current}</span></span>
          {currentStep && (
            <>
              <span className="text-muted-foreground">LOSS: <span className="text-primary">{currentStep.loss.toFixed(4)}</span></span>
              <span className="text-muted-foreground">PPL: <span className="text-neon-yellow">{currentStep.perplexity.toFixed(1)}</span></span>
            </>
          )}
        </div>
      </motion.div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Loss Curves — 2/3 */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="lg:col-span-2 glass-panel p-4 rounded-lg"
        >
          <div className="flex items-center gap-2 mb-3">
            <TrendingDown className="w-4 h-4 text-primary" />
            <h3 className="font-mono text-sm text-foreground font-bold">LOSS CURVES</h3>
            <span className="text-[9px] font-mono text-muted-foreground ml-auto">
              train (cyan) • val (purple) • truthRL penalty (yellow)
            </span>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="trainGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="hsl(187 92% 53%)" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="hsl(187 92% 53%)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 33% 15%)" />
                <XAxis dataKey="step" stroke="hsl(215 20% 35%)" tick={{ fontSize: 10, fontFamily: "monospace" }} />
                <YAxis stroke="hsl(215 20% 35%)" tick={{ fontSize: 10, fontFamily: "monospace" }} domain={[0, "auto"]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(217 33% 8%)",
                    border: "1px solid hsl(187 92% 53% / 0.3)",
                    borderRadius: "8px",
                    fontFamily: "monospace",
                    fontSize: "11px",
                  }}
                />
                <Area type="monotone" dataKey="loss" stroke="hsl(187 92% 53%)" fill="url(#trainGrad)" strokeWidth={2} dot={false} name="Train Loss" />
                <Line type="monotone" dataKey="valLoss" stroke="hsl(270 91% 75%)" strokeWidth={1.5} strokeDasharray="5 3" dot={false} name="Val Loss" />
                <Line type="monotone" dataKey="truthrlPenalty" stroke="hsl(45 93% 47%)" strokeWidth={1} dot={false} name="TruthRL Penalty" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Right column: Metrics */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="space-y-4"
        >
          {/* Live Metrics */}
          <div className="glass-panel p-4 rounded-lg">
            <div className="flex items-center gap-2 mb-3">
              <Cpu className="w-4 h-4 text-neon-green" />
              <h3 className="font-mono text-sm text-foreground font-bold">LIVE METRICS</h3>
            </div>
            {currentStep ? (
              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: "Loss", value: currentStep.loss.toFixed(4), color: "text-primary" },
                  { label: "Val Loss", value: currentStep.valLoss.toFixed(4), color: "text-secondary" },
                  { label: "Perplexity", value: currentStep.perplexity.toFixed(1), color: "text-neon-yellow" },
                  { label: "Grad Norm", value: currentStep.gradNorm.toFixed(3), color: "text-neon-green" },
                  { label: "Energy (L1)", value: currentStep.energy.toFixed(4), color: "text-primary" },
                  { label: "Entropy", value: currentStep.entropy.toFixed(4), color: "text-secondary" },
                  { label: "LR", value: currentStep.lr.toExponential(2), color: "text-muted-foreground" },
                  { label: "TruthRL", value: currentStep.truthrlPenalty.toFixed(4), color: "text-neon-yellow" },
                ].map(({ label, value, color }) => (
                  <div key={label} className="bg-background/30 rounded p-2">
                    <p className="text-[9px] font-mono text-muted-foreground uppercase">{label}</p>
                    <p className={`text-sm font-mono font-bold ${color}`}>{value}</p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs font-mono text-muted-foreground text-center py-6">
                Press ▶ to start training
              </p>
            )}
          </div>
        </motion.div>
      </div>

      {/* Bottom Row: Attention, Gradient Flow, Token Distribution */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
        {/* Attention Heatmap */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="glass-panel p-4 rounded-lg"
        >
          <div className="flex items-center gap-2 mb-3">
            <Grid3X3 className="w-4 h-4 text-primary" />
            <h3 className="font-mono text-sm text-foreground font-bold">ATTENTION MAP</h3>
          </div>
          <div className="aspect-square">
            <AttentionHeatmap step={stepRef.current} nHeads={config.nHeads} />
          </div>
          <div className="flex justify-between mt-2 text-[9px] font-mono text-muted-foreground">
            <span>Query →</span>
            <span>Head 0 / {config.nHeads}</span>
            <span>Key ↓</span>
          </div>
        </motion.div>

        {/* Gradient Flow */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="glass-panel p-4 rounded-lg"
        >
          <div className="flex items-center gap-2 mb-3">
            <BarChart3 className="w-4 h-4 text-neon-green" />
            <h3 className="font-mono text-sm text-foreground font-bold">GRADIENT FLOW</h3>
          </div>
          <GradientFlow step={stepRef.current} nLayers={config.nLayers} />
          <p className="text-[9px] font-mono text-muted-foreground text-center mt-2">
            Layer index → | ■ healthy ■ weak ■ vanishing
          </p>
        </motion.div>

        {/* Token Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="glass-panel p-4 rounded-lg"
        >
          <div className="flex items-center gap-2 mb-3">
            <BarChart3 className="w-4 h-4 text-neon-yellow" />
            <h3 className="font-mono text-sm text-foreground font-bold">TOKEN PROBS</h3>
          </div>
          <TokenDistribution step={stepRef.current} />
        </motion.div>
      </div>

      {/* Architecture Info Footer */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.7 }}
        className="mt-4 glass-panel p-4 rounded-lg"
      >
        <div className="flex items-center justify-between font-mono text-xs text-muted-foreground">
          <div className="flex items-center gap-6">
            <span>ARCH: <span className="text-primary uppercase">{config.architecture}</span></span>
            <span>PARAMS: <span className="text-secondary">
              {((config.nLayers * config.dModel * config.dModel * 4 * 2 + config.dModel * 257) / 1e6).toFixed(1)}M
            </span></span>
            <span>d_model: <span className="text-foreground">{config.dModel}</span></span>
            <span>OPTIMIZER: <span className="text-neon-green uppercase">{config.optimizer}</span></span>
          </div>
          <span className="text-muted-foreground/60">
            Ref: <a href="https://github.com/karpathy/nanoGPT" target="_blank" rel="noopener" className="text-primary/60 hover:text-primary">karpathy/nanoGPT</a>
          </span>
        </div>
      </motion.div>
    </div>
  );
}
