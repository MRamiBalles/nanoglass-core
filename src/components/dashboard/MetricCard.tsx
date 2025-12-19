import { LucideIcon } from "lucide-react";
import { motion } from "framer-motion";

interface MetricCardProps {
  title: string;
  value: string;
  label: string;
  icon: LucideIcon;
  iconColor: "cyan" | "purple" | "green" | "yellow" | "red";
  delay?: number;
}

const colorClasses = {
  cyan: {
    icon: "text-neon-cyan",
    glow: "shadow-[0_0_15px_hsl(var(--neon-cyan)/0.3)]",
    bg: "bg-neon-cyan/10",
    border: "border-neon-cyan/20",
  },
  purple: {
    icon: "text-neon-purple",
    glow: "shadow-[0_0_15px_hsl(var(--neon-purple)/0.3)]",
    bg: "bg-neon-purple/10",
    border: "border-neon-purple/20",
  },
  green: {
    icon: "text-neon-green",
    glow: "shadow-[0_0_15px_hsl(var(--neon-green)/0.3)]",
    bg: "bg-neon-green/10",
    border: "border-neon-green/20",
  },
  yellow: {
    icon: "text-neon-yellow",
    glow: "shadow-[0_0_15px_hsl(var(--neon-yellow)/0.3)]",
    bg: "bg-neon-yellow/10",
    border: "border-neon-yellow/20",
  },
  red: {
    icon: "text-neon-red",
    glow: "shadow-[0_0_15px_hsl(var(--neon-red)/0.3)]",
    bg: "bg-neon-red/10",
    border: "border-neon-red/20",
  },
};

export function MetricCard({ title, value, label, icon: Icon, iconColor, delay = 0 }: MetricCardProps) {
  const colors = colorClasses[iconColor];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5 }}
      className="glass-panel p-6 hover-glow"
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-2">
            {title}
          </p>
          <motion.p
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: delay + 0.2, duration: 0.3 }}
            className="text-3xl font-mono font-bold text-foreground mb-1"
          >
            {value}
          </motion.p>
          <p className={`text-xs font-mono ${colors.icon}`}>
            {label}
          </p>
        </div>
        <div className={`p-3 rounded-lg ${colors.bg} ${colors.border} border ${colors.glow}`}>
          <Icon className={`w-5 h-5 ${colors.icon}`} />
        </div>
      </div>
    </motion.div>
  );
}
