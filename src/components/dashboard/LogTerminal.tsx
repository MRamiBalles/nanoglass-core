import { motion } from "framer-motion";
import { Terminal } from "lucide-react";
import { useEffect, useState } from "react";

interface LogEntry {
  id: number;
  timestamp: string;
  message: string;
  type: "info" | "success" | "warning" | "error";
}

const initialLogs: LogEntry[] = [
  { id: 1, timestamp: "10:41:58", message: "Cortex-13 Architecture initialized. Hybrid Mode: ON", type: "info" },
  { id: 2, timestamp: "10:42:01", message: 'Loading "Electric Gravity" modules... [OK]', type: "success" },
  { id: 3, timestamp: "10:42:05", message: 'Mamba State Compression: 94.2% (Linear Scan Active)', type: "info" },
  { id: 4, timestamp: "10:42:15", message: "HALLUCINATION DETECTED. TruthRL Intervention applied.", type: "error" },
  { id: 5, timestamp: "10:42:22", message: 'Concept "Freedom" crystallized at Layer 4 (L0=32)', type: "success" },
  { id: 6, timestamp: "10:42:30", message: "Neurogenesis: Pruning senescent weights in Block 2...", type: "warning" },
  { id: 7, timestamp: "10:42:45", message: 'Glass Box Probe: Attention Heatmap captured.', type: "info" },
];

const newLogMessages = [
  { message: "Thermodynamic Free Energy minimized (dH < 0.01)", type: "success" as const },
  { message: "MoE Router: Expert #2 (Logic) selected for token 'therefore'", type: "info" as const },
  { message: "Plasticity Critical Window closing... targeted noise injection.", type: "warning" as const },
  { message: "Sparsity Transition: L0 drops to 28. Logic gate formed.", type: "success" as const },
  { message: "Recursive Self-Model updated. Identity verified.", type: "info" as const },
  { message: "Entropic Decay detected in long-term memory. Replaying dreams...", type: "warning" as const },
];

const typeColors = {
  info: "text-muted-foreground",
  success: "text-neon-green",
  warning: "text-neon-yellow",
  error: "text-neon-red",
};

const typePrefixes = {
  info: "[CTX-13]",
  success: "[OK]",
  warning: "[WARN]",
  error: "[ALERT]",
};

export function LogTerminal() {
  const [logs, setLogs] = useState<LogEntry[]>(initialLogs);
  const [currentLogIndex, setCurrentLogIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      const now = new Date();
      const timestamp = `${now.getHours().toString().padStart(2, "0")}:${now.getMinutes().toString().padStart(2, "0")}:${now.getSeconds().toString().padStart(2, "0")}`;

      const newLog = newLogMessages[currentLogIndex % newLogMessages.length];

      setLogs((prev) => [
        ...prev.slice(-10),
        {
          id: Date.now(),
          timestamp,
          ...newLog,
        },
      ]);

      setCurrentLogIndex((prev) => prev + 1);
    }, 3500);

    return () => clearInterval(interval);
  }, [currentLogIndex]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4, duration: 0.5 }}
      className="glass-panel p-4 h-full flex flex-col"
    >
      <div className="flex items-center gap-2 mb-4 pb-3 border-b border-border/30">
        <Terminal className="w-4 h-4 text-neon-cyan" />
        <h3 className="font-mono text-sm text-muted-foreground uppercase tracking-wider">
          Cortex-13 Kernel
        </h3>
        <div className="ml-auto flex gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-neon-red/60" />
          <div className="w-2.5 h-2.5 rounded-full bg-neon-yellow/60" />
          <div className="w-2.5 h-2.5 rounded-full bg-neon-green/60" />
        </div>
      </div>

      <div className="flex-1 overflow-hidden bg-background/90 rounded-lg p-3 font-mono text-xs">
        <div className="space-y-1.5 overflow-y-auto max-h-full">
          {logs.map((log, index) => (
            <motion.div
              key={log.id}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.2 }}
              className="flex gap-2"
            >
              <span className="text-muted-foreground/60 shrink-0">[{log.timestamp}]</span>
              <span className={`${typeColors[log.type]} shrink-0`}>{typePrefixes[log.type]}</span>
              <span className="text-foreground/80 break-all">{log.message}</span>
            </motion.div>
          ))}
          <div className="flex items-center gap-1 text-neon-cyan">
            <span>&gt;</span>
            <span className="w-2 h-4 bg-neon-cyan cursor-blink" />
          </div>
        </div>
      </div>
    </motion.div>
  );
}
