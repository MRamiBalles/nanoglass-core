import { motion } from "framer-motion";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

const generateData = () => {
  const data = [];
  let energy = 0.85;
  let entropy = 0.5;

  for (let i = 0; i <= 60; i += 5) {
    // Energy decreases over time (learning)
    energy = Math.max(0.1, energy - (Math.random() * 0.08 + 0.02));
    // Entropy fluctuates
    entropy = Math.min(0.8, Math.max(0.2, entropy + (Math.random() - 0.5) * 0.15));

    data.push({
      time: `T+${i}s`,
      energy: parseFloat(energy.toFixed(4)),
      entropy: parseFloat(entropy.toFixed(4)),
    });
  }

  return data;
};

const chartData = generateData();

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-panel p-3 rounded-lg border border-border/50">
        <p className="font-mono text-xs text-muted-foreground mb-2">{label}</p>
        {payload.map((entry: any, index: number) => (
          <p key={index} className="font-mono text-sm" style={{ color: entry.color }}>
            {entry.name}: {entry.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export function EnergyChart() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3, duration: 0.5 }}
      className="glass-panel p-6 hover-glow relative scanline overflow-hidden"
    >
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="font-mono text-sm text-muted-foreground uppercase tracking-wider">
            Energy State Monitor
          </h3>
          <p className="text-xs text-muted-foreground/60 mt-1 font-mono">
            Real-time thermodynamic analysis
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-neon-cyan shadow-[0_0_8px_hsl(var(--neon-cyan))]" />
            <span className="text-xs font-mono text-muted-foreground">Total Energy</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-neon-purple shadow-[0_0_8px_hsl(var(--neon-purple))]" />
            <span className="text-xs font-mono text-muted-foreground">Entropy</span>
          </div>
        </div>
      </div>

      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="energyGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="hsl(187, 92%, 53%)" stopOpacity={0.4} />
                <stop offset="100%" stopColor="hsl(187, 92%, 53%)" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="entropyGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="hsl(270, 91%, 75%)" stopOpacity={0.3} />
                <stop offset="100%" stopColor="hsl(270, 91%, 75%)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="hsl(217, 33%, 18%)"
              vertical={false}
            />
            <XAxis
              dataKey="time"
              axisLine={false}
              tickLine={false}
              tick={{ fill: "hsl(215, 20%, 55%)", fontSize: 10, fontFamily: "JetBrains Mono" }}
            />
            <YAxis
              axisLine={false}
              tickLine={false}
              tick={{ fill: "hsl(215, 20%, 55%)", fontSize: 10, fontFamily: "JetBrains Mono" }}
              domain={[0, 1]}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine
              y={0.15}
              stroke="hsl(142, 76%, 45%)"
              strokeDasharray="5 5"
              strokeOpacity={0.5}
              label={{
                value: "OPTIMAL",
                position: "right",
                fill: "hsl(142, 76%, 45%)",
                fontSize: 10,
                fontFamily: "JetBrains Mono",
              }}
            />
            <Area
              type="monotone"
              dataKey="energy"
              stroke="hsl(187, 92%, 53%)"
              strokeWidth={2}
              fill="url(#energyGradient)"
              name="Energy"
            />
            <Area
              type="monotone"
              dataKey="entropy"
              stroke="hsl(270, 91%, 75%)"
              strokeWidth={2}
              strokeDasharray="5 5"
              fill="url(#entropyGradient)"
              name="Entropy"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}
