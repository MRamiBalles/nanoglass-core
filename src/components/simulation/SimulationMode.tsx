import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, Activity, Zap, Brain, TrendingDown, TrendingUp } from "lucide-react";
import { useSettings } from "@/contexts/SettingsContext";

interface SimulationState {
  energy: number;
  entropy: number;
  sparsity: number;
  temperature: number;
  activeNeurons: number;
  totalNeurons: number;
  cycles: number;
  status: "idle" | "running" | "converged" | "diverged";
}

const initialState: SimulationState = {
  energy: 1.0,
  entropy: 0.8,
  sparsity: 0.3,
  temperature: 1.0,
  activeNeurons: 847,
  totalNeurons: 1024,
  cycles: 0,
  status: "idle",
};

export function SimulationMode() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [state, setState] = useState<SimulationState>(initialState);
  const [history, setHistory] = useState<{ energy: number; entropy: number }[]>([]);
  const { getSpeedMultiplier } = useSettings();

  const runSimulation = useCallback(() => {
    if (state.status !== "running") return;

    setState((prev) => {
      // Simulated physics-based update
      const noise = (Math.random() - 0.5) * 0.05;
      const learningRate = 0.02 / prev.temperature;

      // Energy minimization with noise
      let newEnergy = prev.energy - learningRate * (prev.energy - 0.1) + noise * 0.1;
      newEnergy = Math.max(0.05, Math.min(1.5, newEnergy));

      // Entropy decay
      let newEntropy = prev.entropy * 0.995 + noise * 0.02;
      newEntropy = Math.max(0.05, Math.min(1.0, newEntropy));

      // Sparsity increases as energy decreases
      let newSparsity = 1 - newEnergy * 0.7 + noise * 0.05;
      newSparsity = Math.max(0.2, Math.min(0.98, newSparsity));

      // Temperature annealing
      const newTemp = Math.max(0.1, prev.temperature * 0.998);

      // Active neurons follow sparsity
      const newActiveNeurons = Math.floor(prev.totalNeurons * newSparsity);

      const newCycles = prev.cycles + 1;

      // Check convergence
      let newStatus = prev.status;
      if (newEnergy < 0.12 && newEntropy < 0.15) {
        newStatus = "converged";
      } else if (newEnergy > 1.3) {
        newStatus = "diverged";
      }

      return {
        ...prev,
        energy: newEnergy,
        entropy: newEntropy,
        sparsity: newSparsity,
        temperature: newTemp,
        activeNeurons: newActiveNeurons,
        cycles: newCycles,
        status: newStatus,
      };
    });

    setHistory((prev) => {
      const newHistory = [...prev, { energy: state.energy, entropy: state.entropy }];
      return newHistory.slice(-50); // Keep last 50 points
    });
  }, [state.status, state.energy, state.entropy]);

  useEffect(() => {
    if (state.status !== "running") return;

    const speedMultiplier = getSpeedMultiplier();
    const interval = setInterval(runSimulation, 100 * speedMultiplier);

    return () => clearInterval(interval);
  }, [state.status, runSimulation, getSpeedMultiplier]);

  const handleStart = () => setState((s) => ({ ...s, status: "running" }));
  const handlePause = () => setState((s) => ({ ...s, status: "idle" }));
  const handleReset = () => {
    setState(initialState);
    setHistory([]);
  };

  const getStatusColor = () => {
    switch (state.status) {
      case "running":
        return "text-neon-cyan";
      case "converged":
        return "text-neon-green";
      case "diverged":
        return "text-neon-red";
      default:
        return "text-muted-foreground";
    }
  };

  return (
    <div className="relative">
      {/* Toggle Button */}
      <motion.button
        onClick={() => setIsExpanded(!isExpanded)}
        className={`glass-panel px-4 py-2 rounded-lg flex items-center gap-2 hover-glow ${
          state.status === "running" ? "border-neon-cyan/50" : ""
        }`}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <Activity className={`w-4 h-4 ${state.status === "running" ? "text-neon-cyan animate-pulse" : "text-muted-foreground"}`} />
        <span className="font-mono text-sm text-foreground">SIMULATION</span>
        {state.status === "running" && (
          <span className="font-mono text-xs text-neon-cyan">#{state.cycles}</span>
        )}
      </motion.button>

      {/* Expanded Panel */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            className="absolute top-full left-0 mt-2 w-80 glass-panel rounded-xl p-4 z-50"
          >
            {/* Status Header */}
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <div
                  className={`w-2 h-2 rounded-full ${
                    state.status === "running" ? "bg-neon-cyan heartbeat" : "bg-muted-foreground"
                  }`}
                />
                <span className={`font-mono text-xs uppercase ${getStatusColor()}`}>
                  {state.status}
                </span>
              </div>
              <div className="flex gap-2">
                {state.status === "running" ? (
                  <button
                    onClick={handlePause}
                    className="w-8 h-8 rounded-lg bg-neon-yellow/10 border border-neon-yellow/30 flex items-center justify-center hover:bg-neon-yellow/20 transition-colors"
                  >
                    <Pause className="w-4 h-4 text-neon-yellow" />
                  </button>
                ) : (
                  <button
                    onClick={handleStart}
                    className="w-8 h-8 rounded-lg bg-neon-green/10 border border-neon-green/30 flex items-center justify-center hover:bg-neon-green/20 transition-colors"
                  >
                    <Play className="w-4 h-4 text-neon-green" />
                  </button>
                )}
                <button
                  onClick={handleReset}
                  className="w-8 h-8 rounded-lg bg-background/50 border border-border/50 flex items-center justify-center hover:bg-background transition-colors"
                >
                  <RotateCcw className="w-4 h-4 text-muted-foreground" />
                </button>
              </div>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 gap-3 mb-4">
              <MetricBox
                label="Free Energy"
                value={state.energy.toFixed(4)}
                icon={Zap}
                color="yellow"
                trend={state.energy < 0.5 ? "down" : "up"}
              />
              <MetricBox
                label="Entropy"
                value={state.entropy.toFixed(4)}
                icon={Activity}
                color="purple"
                trend={state.entropy < 0.3 ? "down" : "up"}
              />
              <MetricBox
                label="Sparsity"
                value={`${(state.sparsity * 100).toFixed(1)}%`}
                icon={Brain}
                color="cyan"
                trend={state.sparsity > 0.7 ? "up" : "down"}
              />
              <MetricBox
                label="Active"
                value={`${state.activeNeurons}/${state.totalNeurons}`}
                icon={Brain}
                color="green"
              />
            </div>

            {/* Mini Chart */}
            <div className="h-16 bg-background/50 rounded-lg overflow-hidden relative">
              <svg className="w-full h-full" viewBox="0 0 200 60" preserveAspectRatio="none">
                {history.length > 1 && (
                  <>
                    {/* Energy line */}
                    <polyline
                      fill="none"
                      stroke="hsl(var(--neon-cyan))"
                      strokeWidth="1.5"
                      points={history
                        .map((h, i) => `${(i / (history.length - 1)) * 200},${(1 - h.energy) * 50 + 5}`)
                        .join(" ")}
                    />
                    {/* Entropy line */}
                    <polyline
                      fill="none"
                      stroke="hsl(var(--neon-purple))"
                      strokeWidth="1.5"
                      strokeDasharray="4 2"
                      points={history
                        .map((h, i) => `${(i / (history.length - 1)) * 200},${(1 - h.entropy) * 50 + 5}`)
                        .join(" ")}
                    />
                  </>
                )}
              </svg>
              <div className="absolute bottom-1 right-2 flex gap-3 font-mono text-[9px]">
                <span className="text-neon-cyan">— Energy</span>
                <span className="text-neon-purple">╍ Entropy</span>
              </div>
            </div>

            {/* Temperature Bar */}
            <div className="mt-4">
              <div className="flex justify-between mb-1">
                <span className="font-mono text-[10px] text-muted-foreground">TEMPERATURE</span>
                <span className="font-mono text-[10px] text-neon-yellow">{state.temperature.toFixed(3)}</span>
              </div>
              <div className="h-1 bg-background/50 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-neon-cyan via-neon-yellow to-neon-red"
                  style={{ width: `${state.temperature * 100}%` }}
                  layout
                />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

interface MetricBoxProps {
  label: string;
  value: string;
  icon: React.ComponentType<{ className?: string }>;
  color: "cyan" | "purple" | "yellow" | "green";
  trend?: "up" | "down";
}

function MetricBox({ label, value, icon: Icon, color, trend }: MetricBoxProps) {
  const colorClasses = {
    cyan: "text-neon-cyan border-neon-cyan/20",
    purple: "text-neon-purple border-neon-purple/20",
    yellow: "text-neon-yellow border-neon-yellow/20",
    green: "text-neon-green border-neon-green/20",
  };

  const TrendIcon = trend === "down" ? TrendingDown : TrendingUp;

  return (
    <div className={`p-2 rounded-lg bg-background/30 border ${colorClasses[color].split(" ")[1]}`}>
      <div className="flex items-center gap-1 mb-1">
        <Icon className={`w-3 h-3 ${colorClasses[color].split(" ")[0]}`} />
        <span className="font-mono text-[9px] text-muted-foreground uppercase">{label}</span>
        {trend && <TrendIcon className={`w-3 h-3 ml-auto ${trend === "down" ? "text-neon-green" : "text-neon-red"}`} />}
      </div>
      <p className={`font-mono text-sm font-bold ${colorClasses[color].split(" ")[0]}`}>{value}</p>
    </div>
  );
}
