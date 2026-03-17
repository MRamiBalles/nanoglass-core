import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, Thermometer, AlertTriangle, ShieldCheck } from "lucide-react";
import { GlitchText } from "@/components/effects/GlitchText";

// ── Vocabulary (byte-level + [IDK]) ───────────────────
const VOCAB = Array.from({ length: 256 }, (_, i) => String.fromCharCode(i));
VOCAB.push("[IDK]");
const IDK_TOKEN = 256;

// ── Simulated corpus fragments ────────────────────────
const CORPUS = [
  "The glass box reveals truth through sparsity. Energy flows downward as the model compresses meaning.",
  "Neural networks obey thermodynamic laws. Free energy minimization drives convergence to symbolic basins.",
  "Attention sharpens with training. Causal masks enforce autoregressive factorization of the joint distribution.",
  "When uncertainty exceeds the threshold, the [IDK] token fires. This is epistemic humility encoded as architecture.",
  "Mamba processes sequences in O(n) time using selective state spaces. The hybrid architecture combines this with GQA.",
  "Sparse autoencoders decompose polysemantic neurons into monosemantic features for interpretable circuits.",
];

interface GeneratedToken {
  char: string;
  energy: number;
  entropy: number;
  topProbs: { token: string; prob: number }[];
  isIdk: boolean;
  isHallucination: boolean;
  step: number;
}

// ── Simulated inference engine ────────────────────────
function generateNextToken(
  context: string,
  step: number,
  temperature: number,
  corpus: string
): GeneratedToken {
  // Find best matching position in corpus
  const contextLen = Math.min(context.length, 40);
  const tail = context.slice(-contextLen);
  let bestMatch = -1;
  let bestLen = 0;

  for (let i = 0; i < corpus.length - 1; i++) {
    let matchLen = 0;
    for (let j = 0; j < tail.length && i + j < corpus.length; j++) {
      if (corpus[i + j] === tail[tail.length - 1 - (tail.length - 1 - j)]) matchLen++;
      else break;
    }
    if (matchLen > bestLen) {
      bestLen = matchLen;
      bestMatch = i + matchLen;
    }
  }

  // Pick next char from corpus with temperature-based randomness
  let nextChar: string;
  const idkThreshold = 0.15 + temperature * 0.3;
  const randomVal = Math.random();

  if (bestMatch >= 0 && bestMatch < corpus.length && randomVal > temperature * 0.4) {
    nextChar = corpus[bestMatch];
  } else if (randomVal < idkThreshold * 0.1) {
    nextChar = "[IDK]";
  } else {
    // Pick from corpus randomly
    const idx = Math.floor(Math.random() * corpus.length);
    nextChar = corpus[idx];
  }

  const isIdk = nextChar === "[IDK]";

  // Energy: lower when confident, higher when uncertain
  const confidence = bestLen / Math.max(contextLen, 1);
  const energy = Math.max(0.01, (1 - confidence) * 0.8 + Math.random() * 0.15 * temperature);

  // Entropy: related to temperature and confidence
  const entropy = Math.max(0.01, temperature * 0.5 * (1 - confidence * 0.7) + Math.random() * 0.1);

  // Hallucination: high energy + high entropy + not IDK
  const isHallucination = energy > 0.6 && entropy > 0.4 && !isIdk;

  // Top token probabilities (simulated softmax)
  const topTokens = ["e", "t", "a", "o", "n", " ", ".", nextChar, "[IDK]"];
  const unique = [...new Set(topTokens)];
  const rawProbs = unique.map((t, i) => {
    if (t === nextChar) return 3 + (1 - temperature) * 5;
    if (t === "[IDK]") return isIdk ? 5 : 0.3;
    return Math.random() * 2 / (i + 1);
  });
  const sum = rawProbs.reduce((a, b) => a + b, 0);
  const topProbs = unique.map((t, i) => ({
    token: t === " " ? "▁" : t,
    prob: rawProbs[i] / sum,
  })).sort((a, b) => b.prob - a.prob).slice(0, 6);

  return { char: nextChar, energy, entropy, topProbs, isIdk, isHallucination, step };
}

// ── Energy Sparkline ──────────────────────────────────
function Sparkline({ data, color, height = 40 }: { data: number[]; color: string; height?: number }) {
  if (data.length < 2) return null;
  const max = Math.max(...data, 1);
  const w = 200;
  const points = data.slice(-60).map((v, i, arr) => {
    const x = (i / (arr.length - 1)) * w;
    const y = height - (v / max) * (height - 4);
    return `${x},${y}`;
  }).join(" ");

  return (
    <svg width={w} height={height} className="w-full">
      <polyline fill="none" stroke={color} strokeWidth="1.5" points={points} />
    </svg>
  );
}

// ── Main Component ────────────────────────────────────
export default function InferenceDemo() {
  const [tokens, setTokens] = useState<GeneratedToken[]>([]);
  const [running, setRunning] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [corpusIdx, setCorpusIdx] = useState(0);
  const [speed, setSpeed] = useState(120);
  const outputRef = useRef<HTMLDivElement>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const stepRef = useRef(0);

  const currentCorpus = CORPUS[corpusIdx % CORPUS.length];
  const lastToken = tokens.length > 0 ? tokens[tokens.length - 1] : null;

  const tick = useCallback(() => {
    setTokens(prev => {
      const context = prev.map(t => t.char).join("");
      const next = generateNextToken(context, stepRef.current, temperature, currentCorpus);
      stepRef.current++;
      return [...prev, next];
    });
  }, [temperature, currentCorpus]);

  useEffect(() => {
    if (running) {
      intervalRef.current = setInterval(tick, speed);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [running, speed, tick]);

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [tokens]);

  const handleReset = () => {
    setRunning(false);
    stepRef.current = 0;
    setTokens([]);
  };

  const energyHistory = tokens.map(t => t.energy);
  const entropyHistory = tokens.map(t => t.entropy);
  const hallucinationCount = tokens.filter(t => t.isHallucination).length;
  const idkCount = tokens.filter(t => t.isIdk).length;

  return (
    <div className="p-8">
      {/* Header */}
      <motion.header initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight text-foreground">
          <GlitchText text="INFERENCE DEMO" className="text-neon-yellow" />
        </h1>
        <p className="font-mono text-sm text-muted-foreground mt-1">
          Byte-level autoregressive generation • llama2.c-style token stream
        </p>
      </motion.header>

      {/* Controls */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="glass-panel p-4 rounded-lg mb-6 flex flex-wrap items-center gap-4"
      >
        <div className="flex items-center gap-2">
          <button
            onClick={() => setRunning(!running)}
            className={`p-2.5 rounded-lg border transition-all ${
              running ? "bg-neon-yellow/10 border-neon-yellow/40 text-neon-yellow" : "bg-primary/10 border-primary/40 text-primary"
            }`}
          >
            {running ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>
          <button onClick={handleReset} className="p-2.5 rounded-lg border border-border/50 text-muted-foreground hover:text-foreground transition-all">
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>

        <div className="flex flex-col">
          <label className="text-[9px] font-mono text-muted-foreground uppercase">Temperature</label>
          <input type="range" min={0.1} max={2.0} step={0.05} value={temperature} onChange={e => setTemperature(parseFloat(e.target.value))} className="w-28 accent-neon-yellow" />
          <span className="text-[10px] font-mono text-neon-yellow">{temperature.toFixed(2)}</span>
        </div>

        <div className="flex flex-col">
          <label className="text-[9px] font-mono text-muted-foreground uppercase">Speed (ms)</label>
          <input type="range" min={30} max={500} step={10} value={speed} onChange={e => setSpeed(parseInt(e.target.value))} className="w-20 accent-neon-green" />
          <span className="text-[10px] font-mono text-neon-green">{speed}ms</span>
        </div>

        <div className="flex flex-col">
          <label className="text-[9px] font-mono text-muted-foreground uppercase">Corpus</label>
          <select
            value={corpusIdx}
            onChange={e => { setCorpusIdx(parseInt(e.target.value)); handleReset(); }}
            className="bg-background/50 border border-border/50 rounded px-2 py-1 text-xs font-mono text-foreground"
          >
            {CORPUS.map((c, i) => (
              <option key={i} value={i}>{c.slice(0, 35)}…</option>
            ))}
          </select>
        </div>

        <div className="ml-auto flex items-center gap-4 font-mono text-xs">
          <span className="text-muted-foreground">TOKENS: <span className="text-foreground">{tokens.length}</span></span>
          <span className="text-muted-foreground">BLOCKED: <span className="text-neon-red">{hallucinationCount}</span></span>
          <span className="text-muted-foreground">[IDK]: <span className="text-neon-yellow">{idkCount}</span></span>
        </div>
      </motion.div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Token Stream — 2/3 */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="lg:col-span-2 glass-panel rounded-lg overflow-hidden"
        >
          <div className="p-3 border-b border-border/30 flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-neon-green heartbeat" />
            <span className="font-mono text-xs text-muted-foreground">TOKEN STREAM • byte-level autoregressive</span>
          </div>
          <div ref={outputRef} className="p-4 h-80 overflow-y-auto bg-background/90 font-mono text-sm leading-relaxed">
            {tokens.length === 0 && (
              <span className="text-muted-foreground/50">Press ▶ to begin generation...</span>
            )}
            <AnimatePresence>
              {tokens.map((t, i) => (
                <motion.span
                  key={i}
                  initial={{ opacity: 0, scale: 1.5 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.15 }}
                  className={
                    t.isIdk
                      ? "text-neon-yellow bg-neon-yellow/10 px-1 rounded font-bold"
                      : t.isHallucination
                        ? "text-neon-red line-through opacity-50"
                        : "text-foreground"
                  }
                  title={`E:${t.energy.toFixed(3)} H:${t.entropy.toFixed(3)}`}
                >
                  {t.char}
                </motion.span>
              ))}
            </AnimatePresence>
            {running && (
              <motion.span
                animate={{ opacity: [1, 0] }}
                transition={{ repeat: Infinity, duration: 0.6 }}
                className="text-primary font-bold"
              >
                ▌
              </motion.span>
            )}
          </div>
        </motion.div>

        {/* Right Panel */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="space-y-4"
        >
          {/* Token Probabilities */}
          <div className="glass-panel p-4 rounded-lg">
            <h3 className="font-mono text-sm text-foreground font-bold mb-3 flex items-center gap-2">
              <Thermometer className="w-4 h-4 text-neon-yellow" />
              TOP-K PROBABILITIES
            </h3>
            {lastToken ? (
              <div className="space-y-1.5">
                {lastToken.topProbs.map((p, i) => (
                  <div key={i} className="flex items-center gap-2 font-mono text-[10px]">
                    <span className={`w-10 text-right ${p.token === "[IDK]" ? "text-neon-yellow" : "text-muted-foreground"}`}>
                      {p.token}
                    </span>
                    <div className="flex-1 h-3 bg-background/50 rounded-sm overflow-hidden">
                      <motion.div
                        className={`h-full rounded-sm ${p.token === "[IDK]" ? "bg-neon-yellow/80" : "bg-primary/60"}`}
                        animate={{ width: `${p.prob * 100}%` }}
                        transition={{ duration: 0.2 }}
                      />
                    </div>
                    <span className="w-12 text-right text-muted-foreground">{(p.prob * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs font-mono text-muted-foreground text-center py-4">Waiting…</p>
            )}
          </div>

          {/* Status */}
          <div className="glass-panel p-4 rounded-lg">
            <h3 className="font-mono text-sm text-foreground font-bold mb-3">GLASBOX SENSOR</h3>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-[9px] font-mono text-muted-foreground mb-1">
                  <span>ENERGY (L1)</span>
                  <span className="text-primary">{lastToken?.energy.toFixed(4) ?? "—"}</span>
                </div>
                <Sparkline data={energyHistory} color="hsl(187 92% 53%)" />
              </div>
              <div>
                <div className="flex justify-between text-[9px] font-mono text-muted-foreground mb-1">
                  <span>ENTROPY</span>
                  <span className="text-secondary">{lastToken?.entropy.toFixed(4) ?? "—"}</span>
                </div>
                <Sparkline data={entropyHistory} color="hsl(270 91% 75%)" />
              </div>
            </div>
          </div>

          {/* TruthRL Shield */}
          <div className="glass-panel p-4 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              {lastToken?.isHallucination ? (
                <AlertTriangle className="w-4 h-4 text-neon-red" />
              ) : (
                <ShieldCheck className="w-4 h-4 text-neon-green" />
              )}
              <h3 className="font-mono text-sm font-bold text-foreground">TruthRL SHIELD</h3>
            </div>
            <p className={`font-mono text-xs ${lastToken?.isHallucination ? "text-neon-red" : lastToken?.isIdk ? "text-neon-yellow" : "text-neon-green"}`}>
              {lastToken?.isHallucination
                ? "⚠ HALLUCINATION DETECTED — Token struck through"
                : lastToken?.isIdk
                  ? "◆ ABSTENTION — [IDK] token fired (epistemic humility)"
                  : "✓ NOMINAL — Low energy, high confidence"
              }
            </p>
          </div>
        </motion.div>
      </div>

      {/* Footer */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="mt-4 glass-panel p-4 rounded-lg font-mono text-xs text-muted-foreground flex items-center justify-between"
      >
        <div className="flex gap-6">
          <span>VOCAB: <span className="text-primary">257</span> (0-255 + [IDK])</span>
          <span>TEMP: <span className="text-neon-yellow">{temperature.toFixed(2)}</span></span>
          <span>CONTEXT: <span className="text-foreground">{Math.min(tokens.length, 40)} bytes</span></span>
        </div>
        <span className="text-muted-foreground/60">
          Ref: <a href="https://github.com/karpathy/llama2.c" target="_blank" rel="noopener" className="text-primary/60 hover:text-primary">karpathy/llama2.c</a>
        </span>
      </motion.div>
    </div>
  );
}
