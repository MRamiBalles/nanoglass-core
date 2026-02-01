import { motion } from "framer-motion";
import { Zap, Activity, Shield, Brain } from "lucide-react";
import { useEffect, useState } from "react";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { SparsityChart } from "@/components/dashboard/SparsityChart";
import { LogTerminal } from "@/components/dashboard/LogTerminal";
import { GlitchText } from "@/components/effects/GlitchText";
import { SimulationMode } from "@/components/simulation/SimulationMode";

export default function Dashboard() {
  const [metrics, setMetrics] = useState({
    energy: 0.1542,
    entropy: 0.12,
    sparsity: 32.4,
    status: "ACTIVE",
    isHallucinating: false
  });

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch("http://localhost:8000/health");
        const health = await response.json();
        
        if (health.status === "ok") {
          const res = await fetch("http://localhost:8000/metrics");
          const data = await res.json();
          if (data.energy_history.length > 0) {
            setMetrics(prev => ({
              ...prev,
              energy: data.energy_history[data.energy_history.length - 1],
              entropy: data.entropy_history[data.entropy_history.length - 1] || prev.entropy,
            }));
          }
        }
      } catch (e) {
        console.log("API Bridge not reachable. Running in simulation mode.");
      }
    };

    const interval = setInterval(fetchMetrics, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-8">
      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-end gap-4 flex-wrap">
          <div>
            <h1 className="text-4xl font-bold tracking-tight text-foreground">
              PROJECT{" "}
              <GlitchText text="NANOGLASS" className="neon-cyan" />
            </h1>
            <p className="font-mono text-sm text-muted-foreground mt-1">
              <GlitchText text="GLASS BOX INTERPRETER V1.0 • CORTEX-13 ARCHITECTURE" />
            </p>
          </div>
          <div className="ml-auto flex items-center gap-3">
            <SimulationMode />
            <div className="flex items-center gap-2 glass-panel px-4 py-2 rounded-lg">
              <div className="w-2 h-2 rounded-full bg-neon-green heartbeat" />
              <span className="font-mono text-xs text-neon-green">SYSTEM ONLINE</span>
            </div>
          </div>
        </div>
      </motion.header>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <MetricCard
          title="Free Energy"
          value={metrics.energy.toFixed(4)}
          label="Minimizing (Optimal)"
          icon={Zap}
          iconColor="yellow"
          delay={0.1}
        />
        <MetricCard
          title="L0 Sparsity"
          value={metrics.sparsity.toFixed(1)}
          label="Crystallized (Boolean)"
          icon={Activity}
          iconColor="purple"
          delay={0.2}
        />
        <MetricCard
          title="Entropy"
          value={metrics.entropy.toFixed(3)}
          label="Routing Diversity"
          icon={Brain}
          iconColor="cyan"
          delay={0.25}
        />
        <MetricCard
          title="TruthRL Status"
          value={metrics.status}
          label={metrics.isHallucinating ? "HALLUCINATION DETECTED" : "Hallucination Shield: ON"}
          icon={Shield}
          iconColor={metrics.isHallucinating ? "red" : "green"}
          delay={0.3}
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Sparsity Chart - 2/3 width */}
        <div className="lg:col-span-2">
          <SparsityChart />
        </div>

        {/* Log Terminal - 1/3 width */}
        <div className="lg:col-span-1 min-h-[400px]">
          <LogTerminal />
        </div>
      </div>

      {/* Footer Stats */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        className="mt-6 glass-panel p-4 rounded-lg"
      >
        <div className="flex items-center justify-between font-mono text-xs text-muted-foreground">
          <div className="flex items-center gap-6">
            <span>FEATURES ACTIVE: <span className="text-neon-cyan">847</span></span>
            <span>EXPERTS LOADED: <span className="text-neon-purple">8/8</span></span>
            <span>SAE SPARSITY: <span className="text-neon-green">94.2%</span></span>
          </div>
          <div className="flex items-center gap-2">
            <span>UPTIME:</span>
            <span className="text-foreground">04:32:17</span>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
