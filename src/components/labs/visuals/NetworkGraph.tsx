import { motion } from "framer-motion";
import { useEffect, useState, useMemo } from "react";

interface NetworkGraphProps {
    parameterValue: number; // e.g., "Noise Scale" (0.0 - 2.0)
    active: boolean;
}

export function NetworkGraph({ parameterValue, active }: NetworkGraphProps) {
    // Logic: 
    // High Noise (> 1.2) -> Chaotic blinking, nodes breaking.
    // Optimal Noise (0.5) -> Rhythmic pulsing, new connections forming.
    // Low Noise (< 0.2) -> Static, frozen graph.

    const nodeCount = 12;
    const radius = 80;

    // Create nodes in a circle/random layout
    const nodes = useMemo(() => {
        return Array.from({ length: nodeCount }).map((_, i) => ({
            id: i,
            x: 150 + Math.cos((i / nodeCount) * Math.PI * 2) * radius * (0.8 + Math.random() * 0.4),
            y: 150 + Math.sin((i / nodeCount) * Math.PI * 2) * radius * (0.8 + Math.random() * 0.4),
        }));
    }, []);

    // Generate random edges
    const edges = useMemo(() => {
        const e = [];
        for (let i = 0; i < nodeCount; i++) {
            const target = (i + Math.floor(Math.random() * 3) + 1) % nodeCount;
            e.push({ source: i, target: target, id: `${i}-${target}` });
            // Add minimal connectivity
            if (Math.random() > 0.5) {
                e.push({ source: i, target: (target + 1) % nodeCount, id: `${i}-${target}-2` });
            }
        }
        return e;
    }, []);

    return (
        <div className="w-full h-64 bg-black/40 rounded-lg overflow-hidden relative border border-white/5">
            <svg className="w-full h-full" viewBox="0 0 300 300">
                {/* Edges */}
                {edges.map((edge) => {
                    const source = nodes[edge.source];
                    const target = nodes[edge.target];
                    return (
                        <motion.line
                            key={edge.id}
                            x1={source.x}
                            y1={source.y}
                            x2={target.x}
                            y2={target.y}
                            stroke={parameterValue > 1.2 ? "rgba(255, 0, 0, 0.5)" : "rgba(0, 255, 255, 0.3)"}
                            strokeWidth="1"
                            animate={{
                                opacity: active ? [0.2, 0.8, 0.2] : 0.2,
                                strokeDasharray: parameterValue > 1.2 ? ["5 5", "10 10", "5 5"] : "0 0"
                            }}
                            transition={{
                                duration: parameterValue > 1.2 ? 0.1 : (2.0 / (parameterValue + 0.1)), // Faster blink with high noise
                                repeat: Infinity,
                                repeatType: "reverse"
                            }}
                        />
                    );
                })}

                {/* Nodes */}
                {nodes.map((node) => (
                    <motion.circle
                        key={node.id}
                        cx={node.x}
                        cy={node.y}
                        r={parameterValue > 1.2 ? 2 + Math.random() * 3 : 4} // Unstable size if high noise
                        className={parameterValue > 1.2 ? "fill-neon-red" : "fill-neon-green"}
                        animate={{
                            scale: active ? [1, 1.2, 1] : 1,
                            opacity: active ? (parameterValue < 0.2 ? 0.3 : 1) : 0.5
                        }}
                        transition={{
                            duration: active ? (Math.random() * 1 + 0.5) / (parameterValue + 1) : 0,
                            repeat: Infinity
                        }}
                    />
                ))}
            </svg>

            {/* Overlay Stats */}
            <div className="absolute bottom-2 left-2 text-[10px] font-mono text-muted-foreground bg-black/50 p-1 rounded">
                <div>Nodes: {nodeCount}</div>
                <div>Synapses: {edges.length}</div>
                <div>Plasticity: {parameterValue.toFixed(2)}σ</div>
            </div>
        </div>
    );
}
