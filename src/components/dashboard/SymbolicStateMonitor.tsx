import { useEffect, useState, useRef } from "react";
import { motion } from "framer-motion";
import { Hexagon, TrendingDown, TrendingUp, Minus } from "lucide-react";

interface SymbolicState {
  E: number;   // Energy (L1)
  H: number;   // Entropy (Shannon)
  P_idk: number; // P([IDK])
  Sparsity: number; // % active features
  phase: "distributed" | "transition" | "crystallized";
  timestamp: number;
}

function determinePhase(sparsity: number, energy: number): SymbolicState["phase"] {
  if (sparsity > 90 && energy < 0.2) return "crystallized";
  if (sparsity > 60 || energy < 0.4) return "transition";
  return "distributed";
}

function Gauge({ value, max, label, color, unit }: {
  value: number; max: number; label: string; color: string; unit?: string;
}) {
  const pct = Math.min(100, (value / max) * 100);
  const trend = useRef(value);
  const direction = value > trend.current ? "up" : value < trend.current ? "down" : "stable";
  useEffect(() => { trend.current = value; }, [value]);

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-[9px] font-mono text-muted-foreground uppercase">{label}</span>
        <div className="flex items-center gap-1">
          {direction === "up" ? <TrendingUp className="w-3 h-3 text-neon-red" /> :
           direction === "down" ? <TrendingDown className="w-3 h-3 text-neon-green" /> :
           <Minus className="w-3 h-3 text-muted-foreground" />}
          <span className={`text-sm font-mono font-bold ${color}`}>
            {value.toFixed(4)}{unit}
          </span>
        </div>
      </div>
      <div className="h-2 bg-background/50 rounded-full overflow-hidden">
        <motion.div
          className={`h-full rounded-full`}
          style={{ backgroundColor: `hsl(var(--${color.replace("text-", "")}))` }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.5, type: "spring" }}
        />
      </div>
    </div>
  );
}

function PhaseIndicator({ phase }: { phase: SymbolicState["phase"] }) {
  const config = {
    distributed: { label: "DISTRIBUTED", color: "text-neon-red", bg: "bg-neon-red/10 border-neon-red/30", desc: "Weights spread. High entropy. Pre-learning." },
    transition: { label: "PHASE TRANSITION", color: "text-neon-yellow", bg: "bg-neon-yellow/10 border-neon-yellow/30", desc: "Sparsity increasing. Features emerging." },
    crystallized: { label: "CRYSTALLIZED", color: "text-neon-green", bg: "bg-neon-green/10 border-neon-green/30", desc: "Symbolic basins formed. Low energy." },
  };
  const c = config[phase];

  return (
    <motion.div
      key={phase}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`p-3 rounded-lg border ${c.bg}`}
    >
      <div className="flex items-center gap-2">
        <Hexagon className={`w-4 h-4 ${c.color}`} />
        <div>
          <p className={`text-[10px] font-mono font-bold ${c.color}`}>{c.label}</p>
          <p className="text-[8px] font-mono text-muted-foreground">{c.desc}</p>
        </div>
      </div>
    </motion.div>
  );
}

export function SymbolicStateMonitor() {
  const [history, setHistory] = useState<SymbolicState[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Simulate evolving state
  useEffect(() => {
    let step = 0;
    const interval = setInterval(() => {
      step++;
      const t = step;
      const energy = Math.max(0.05, 0.9 * Math.exp(-t * 0.015) + (Math.random() - 0.5) * 0.04);
      const entropy = Math.max(0.02, 0.8 * Math.exp(-t * 0.01) + (Math.random() - 0.5) * 0.03);
      const pIdk = Math.min(0.95, 0.05 + 0.4 * (1 - Math.exp(-t * 0.008)) + (Math.random() - 0.5) * 0.05);
      const sparsity = Math.min(98, 20 + 70 * (1 - Math.exp(-t * 0.012)) + (Math.random() - 0.5) * 3);
      const phase = determinePhase(sparsity, energy);

      const newState: SymbolicState = { E: energy, H: entropy, P_idk: pIdk, Sparsity: sparsity, phase, timestamp: Date.now() };
      setHistory(prev => [...prev.slice(-60), newState]);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  // Draw phase space canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || history.length < 2) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.fillStyle = "rgba(0,0,0,0.08)";
    ctx.fillRect(0, 0, w, h);

    // Draw trajectory in E-H phase space
    const last = history.slice(-30);
    for (let i = 1; i < last.length; i++) {
      const prev = last[i - 1];
      const curr = last[i];
      const x1 = prev.E * w;
      const y1 = (1 - prev.H) * h;
      const x2 = curr.E * w;
      const y2 = (1 - curr.H) * h;

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      const alpha = i / last.length;
      ctx.strokeStyle = curr.phase === "crystallized"
        ? `hsla(142, 76%, 45%, ${alpha})`
        : curr.phase === "transition"
          ? `hsla(45, 93%, 47%, ${alpha})`
          : `hsla(0, 84%, 60%, ${alpha})`;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Draw point
      if (i === last.length - 1) {
        ctx.beginPath();
        ctx.arc(x2, y2, 4, 0, Math.PI * 2);
        ctx.fillStyle = curr.phase === "crystallized" ? "hsl(142 76% 45%)" : curr.phase === "transition" ? "hsl(45 93% 47%)" : "hsl(0 84% 60%)";
        ctx.fill();
      }
    }

    // Axes labels
    ctx.font = "8px monospace";
    ctx.fillStyle = "hsla(215, 20%, 55%, 0.6)";
    ctx.fillText("E →", w - 25, h - 4);
    ctx.fillText("H ↑", 2, 10);

    // Phase regions
    ctx.strokeStyle = "hsla(142, 76%, 45%, 0.15)";
    ctx.setLineDash([3, 3]);
    ctx.strokeRect(0, h * 0.8, w * 0.2, h * 0.2); // Crystallized zone
    ctx.setLineDash([]);
    ctx.fillStyle = "hsla(142, 76%, 45%, 0.1)";
    ctx.font = "7px monospace";
    ctx.fillText("Crystal", 2, h - 4);
  }, [history]);

  const current = history.length > 0 ? history[history.length - 1] : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.35 }}
      className="glass-panel p-4 rounded-lg"
    >
      <div className="flex items-center gap-2 mb-3">
        <Hexagon className="w-4 h-4 text-primary" />
        <h3 className="font-mono text-sm text-foreground font-bold">SYMBOLIC STATE S(t)</h3>
        <span className="ml-auto text-[8px] font-mono text-muted-foreground">t = {history.length}</span>
      </div>

      {current ? (
        <div className="space-y-3">
          {/* Gauges */}
          <Gauge value={current.E} max={1} label="Energy E(t)" color="text-primary" />
          <Gauge value={current.H} max={1} label="Entropy H(t)" color="text-secondary" />
          <Gauge value={current.P_idk} max={1} label="P_idk(t)" color="text-neon-yellow" />
          <Gauge value={current.Sparsity} max={100} label="Sparsity(t)" color="text-neon-green" unit="%" />

          {/* Phase */}
          <PhaseIndicator phase={current.phase} />

          {/* Phase Space Canvas */}
          <div className="mt-2">
            <p className="text-[8px] font-mono text-muted-foreground mb-1">E-H PHASE SPACE TRAJECTORY</p>
            <canvas
              ref={canvasRef}
              width={200}
              height={120}
              className="w-full h-24 bg-background/50 rounded border border-border/20"
            />
          </div>

          {/* Vector Notation */}
          <div className="p-2 bg-background/30 rounded border border-border/20">
            <p className="text-[9px] font-mono text-primary">
              S({history.length}) = {"{"} {current.E.toFixed(3)}, {current.H.toFixed(3)}, {current.P_idk.toFixed(3)}, {current.Sparsity.toFixed(1)}% {"}"}
            </p>
            <p className="text-[7px] font-mono text-muted-foreground mt-0.5">
              {current.Sparsity > 90 && current.E < 0.2
                ? "⚡ Sparsity > 90% ∧ E < 0.2 → SYMBOLIC CRYSTALLIZATION"
                : current.Sparsity > 60
                  ? "↗ Approaching phase transition boundary..."
                  : "◌ Distributed representation regime"
              }
            </p>
          </div>
        </div>
      ) : (
        <p className="text-xs font-mono text-muted-foreground text-center py-6">Initializing sensor...</p>
      )}
    </motion.div>
  );
}
