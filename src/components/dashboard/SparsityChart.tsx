import { motion } from "framer-motion";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
} from "recharts";

const generateSparsityData = () => {
    const data = [];
    // Simulate the "Phase Transition" described in the paper
    // Stage 1: Innate (High Entropy, High L0)
    // Stage 2: Adaptive (RLHF Pressure, Sharp Drop)
    // Stage 3: Specialized (MoE, Low L0 Stability)

    for (let i = 0; i <= 100; i++) {
        let l0;
        if (i < 30) {
            // Stage 1: High noise
            l0 = 400 + (Math.random() * 50);
        } else if (i < 50) {
            // Stage 2: Collapse/Transition
            const progress = (i - 30) / 20;
            l0 = 400 - (360 * progress) + (Math.random() * 20);
        } else {
            // Stage 3: Crystallized Symbolism
            l0 = 35 + (Math.random() * 5);
        }

        data.push({
            step: i * 100, // Training steps
            l0: Math.floor(l0),
            stage: i < 30 ? "Innate" : i < 50 ? "Adaptive" : "Symbolic",
        });
    }

    return data;
};

const chartData = generateSparsityData();

const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
        const data = payload[0].payload;
        return (
            <div className="glass-panel p-3 rounded-lg border border-border/50">
                <p className="font-mono text-xs text-muted-foreground mb-1">Step: {label}</p>
                <p className="font-mono text-xs text-neon-purple mb-2">Stage: {data.stage}</p>
                <p className="font-mono text-sm text-neon-cyan">
                    L0 Norm: {data.l0} active
                </p>
            </div>
        );
    }
    return null;
};

export function SparsityChart() {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.5 }}
            className="glass-panel p-6 hover-glow relative scanline overflow-hidden h-full"
        >
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h3 className="font-mono text-sm text-muted-foreground uppercase tracking-wider">
                        Sparsity Transition (L0)
                    </h3>
                    <p className="text-xs text-muted-foreground/60 mt-1 font-mono">
                        Evidence of Symbolic Crystallization
                    </p>
                </div>
                <div className="flex items-center gap-2">
                    <span className="text-xs font-mono text-neon-cyan">Target: &lt; 40</span>
                </div>
            </div>

            <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                        <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="hsl(217, 33%, 18%)"
                            vertical={false}
                        />
                        <XAxis
                            dataKey="step"
                            axisLine={false}
                            tickLine={false}
                            tick={{ fill: "hsl(215, 20%, 55%)", fontSize: 10, fontFamily: "JetBrains Mono" }}
                        />
                        <YAxis
                            axisLine={false}
                            tickLine={false}
                            tick={{ fill: "hsl(215, 20%, 55%)", fontSize: 10, fontFamily: "JetBrains Mono" }}
                            domain={[0, 500]}
                        />
                        <Tooltip content={<CustomTooltip />} />
                        <ReferenceLine
                            x={3000}
                            stroke="hsl(270, 91%, 75%)"
                            strokeDasharray="3 3"
                            label={{
                                value: "RLHF ONSET",
                                position: "insideTopRight",
                                fill: "hsl(270, 91%, 75%)",
                                fontSize: 9,
                                angle: -90,
                                dx: 10,
                                fontFamily: "JetBrains Mono",
                            }}
                        />
                        <ReferenceLine
                            y={35}
                            stroke="hsl(142, 76%, 45%)"
                            strokeDasharray="5 5"
                            label={{
                                value: "SYMBOLIC BOUNDARY",
                                position: "right",
                                fill: "hsl(142, 76%, 45%)",
                                fontSize: 10,
                                fontFamily: "JetBrains Mono",
                            }}
                        />
                        <Line
                            type="monotone"
                            dataKey="l0"
                            stroke="hsl(187, 92%, 53%)"
                            strokeWidth={2}
                            dot={false}
                            name="L0 Norm"
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </motion.div>
    );
}
