import { motion } from "framer-motion";
import { Zap, Activity, Shield, Brain } from "lucide-react";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { SparsityChart } from "@/components/dashboard/SparsityChart"; // Replaced EnergyChart
import { LogTerminal } from "@/components/dashboard/LogTerminal";

export default function Dashboard() {
  return (
    <div className="p-8">
      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-end gap-4">
          <div>
            <h1 className="text-4xl font-bold tracking-tight text-foreground">
              PROJECT <span className="neon-cyan">NANOGLASS</span>
            </h1>
            <p className="font-mono text-sm text-muted-foreground mt-1">
              GLASS BOX INTERPRETER V1.0 • CORTEX-13 ARCHITECTURE
            </p>
          </div>
          <div className="ml-auto flex items-center gap-2 glass-panel px-4 py-2 rounded-lg">
            <div className="w-2 h-2 rounded-full bg-neon-green heartbeat" />
            <span className="font-mono text-xs text-neon-green">SYSTEM ONLINE</span>
          </div>
        </div>
      </motion.header>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <MetricCard
          title="Free Energy"
          value="0.1542"
          label="Minimizing (Optimal)"
          icon={Zap}
          iconColor="yellow"
          delay={0.1}
        />
        <MetricCard
          title="L0 Sparsity"
          value="32.4"
          label="Crystallized (Boolean)"
          icon={Activity}
          iconColor="purple"
          delay={0.2}
        />
        <MetricCard
          title="MoE Experts"
          value="8/8"
          label="Routing Entropy: 0.12"
          icon={Brain}
          iconColor="cyan"
          delay={0.25}
        />
        <MetricCard
          title="TruthRL Status"
          value="ACTIVE"
          label="Hallucination Shield: ON"
          icon={Shield}
          iconColor="green"
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
