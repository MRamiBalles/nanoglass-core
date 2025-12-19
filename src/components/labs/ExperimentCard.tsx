import { motion } from "framer-motion";
import { Play, RotateCcw, Terminal, CheckCircle2, AlertTriangle, Lightbulb, Microscope, Construction, Sliders } from "lucide-react";
import { useState } from "react";
import { Experiment, SimulationResult } from "@/data/experiments";
import { useResearchProgress } from "@/hooks/useResearchProgress";
import { NetworkGraph } from "@/components/labs/visuals/NetworkGraph";
import { EnergyLandscape } from "@/components/labs/visuals/EnergyLandscape";
import { QualiaField } from "@/components/labs/visuals/QualiaField";

interface ExperimentCardProps {
    experiment: Experiment;
    onRun: (id: string) => void;
}

export function ExperimentCard({ experiment, onRun }: ExperimentCardProps) {
    const [isRunning, setIsRunning] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const [progress, setProgress] = useState(0);

    const { completeExperiment, completedIds } = useResearchProgress();
    const isCompleted = completedIds.includes(experiment.id);

    // Parameter State
    const [params, setParams] = useState<Record<string, number>>(() => {
        const initial: Record<string, number> = {};
        experiment.parameters?.forEach(p => initial[p.id] = p.defaultValue);
        return initial;
    });

    const [simResult, setSimResult] = useState<SimulationResult | null>(null);

    const handleRun = () => {
        setIsRunning(true);
        setProgress(0);
        setSimResult(null);
        setLogs([`Initializing ${experiment.scriptName}...`]);

        // Simulate execution
        let step = 0;
        const interval = setInterval(() => {
            step++;
            setProgress(p => Math.min(p + 20, 100));

            if (step === 1) setLogs(prev => [...prev, "Loading weights...", "Connecting to Cortex-13 Kernel..."]);
            if (step === 2) {
                if (experiment.parameters) {
                    setLogs(prev => [...prev, `Applying parameters: ${Object.entries(params).map(([k, v]) => `${k}=${v}`).join(", ")}`]);
                } else {
                    setLogs(prev => [...prev, "Running experimental protocol...", experiment.metrics[0] ? `Metric update: ${experiment.metrics[0].label} adjusted.` : "Collecting data..."]);
                }
            }
            if (step === 3) setLogs(prev => [...prev, "Verifying results...", "Saving state checkpoint."]);

            if (step >= 4) {
                clearInterval(interval);
                setIsRunning(false);

                // Calculate Result logic
                if (experiment.runSimulation && experiment.parameters) {
                    const result = experiment.runSimulation(params);
                    setSimResult(result);
                    setLogs(prev => [...prev, ...result.logs]);

                    // Gamification Logic: Only complete on success
                    if (result.success) {
                        completeExperiment(experiment.id);
                    }
                } else {
                    // For simple experiments without logic, assume success
                    setLogs(prev => [...prev, "Protocol completed successfully."]);
                    completeExperiment(experiment.id);
                }

                onRun(experiment.id);
            }
        }, 600);
    };

    const isPlanned = experiment.status === "planned";

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`glass-panel p-5 relative overflow-hidden group hover:border-neon-cyan/30 transition-colors ${isPlanned ? "opacity-70 border-dashed" :
                isCompleted ? "border-neon-green/30 bg-neon-green/5" : ""
                }`}
        >
            <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                <experiment.icon className="w-24 h-24" />
            </div>

            <div className="relative z-10 flex flex-col h-full">
                {/* Header */}
                <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-background/50 border border-border">
                            <experiment.icon className={`w-6 h-6 ${experiment.category === "Bio-Mimetic" ? "text-neon-green" :
                                experiment.category === "Physics" ? "text-neon-purple" : "text-neon-yellow"
                                }`} />
                        </div>
                        <div>
                            <h3 className="font-bold text-foreground flex items-center gap-2">
                                {experiment.name}
                                {isPlanned && <span className="text-[9px] bg-neon-yellow/10 text-neon-yellow px-1 py-0.5 rounded border border-neon-yellow/20">PLANNED</span>}
                                {isCompleted && !isPlanned && <span className="text-[9px] bg-neon-green/10 text-neon-green px-1 py-0.5 rounded border border-neon-green/20 flex items-center gap-1"><CheckCircle2 className="w-3 h-3" /> COMPLETED</span>}
                            </h3>
                            <p className="font-mono text-[10px] text-muted-foreground">{experiment.scriptName}</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        {isRunning && <span className="animate-spin"><RotateCcw className="w-4 h-4 text-neon-cyan" /></span>}
                    </div>
                </div>

                {/* Description */}
                <p className="text-sm text-muted-foreground mb-4 border-b border-border/10 pb-4">
                    {experiment.description}
                </p>

                {/* PARAMETERS SECTION (Interactive) */}
                {!isPlanned && experiment.parameters && (
                    <div className="mb-4 space-y-3 bg-accent/20 p-3 rounded-lg border border-accent/30">
                        <div className="flex items-center gap-2 mb-2">
                            <Sliders className="w-3 h-3 text-neon-cyan" />
                            <span className="font-mono text-[10px] uppercase font-bold text-neon-cyan">Simulation Parameters</span>
                        </div>
                        {experiment.parameters.map(p => (
                            <div key={p.id}>
                                <div className="flex justify-between text-[10px] mb-1 font-mono">
                                    <span className="text-muted-foreground">{p.label}</span>
                                    <span className="text-foreground">{params[p.id]} {p.unit}</span>
                                </div>
                                <input
                                    type="range"
                                    min={p.min}
                                    max={p.max}
                                    step={p.step}
                                    value={params[p.id]}
                                    onChange={(e) => setParams(prev => ({ ...prev, [p.id]: parseFloat(e.target.value) }))}
                                    className="w-full h-1 bg-muted rounded-lg appearance-none cursor-pointer accent-neon-cyan"
                                    disabled={isRunning}
                                />
                                <p className="text-[9px] text-muted-foreground/60 italic mt-0.5">{p.description}</p>
                            </div>
                        ))}
                    </div>
                )}

                {/* SIMULATION RESULT (Dynamic) */}
                {simResult && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className={`mb-4 p-3 rounded-lg border ${simResult.success ? "bg-neon-green/10 border-neon-green/30 text-neon-green" : "bg-neon-red/10 border-neon-red/30 text-neon-red"} font-mono text-xs`}
                    >
                        <div className="flex items-center gap-2 font-bold mb-1">
                            {simResult.success ? <CheckCircle2 className="w-4 h-4" /> : <AlertTriangle className="w-4 h-4" />}
                            {simResult.success ? "SIMULATION SUCCESS" : "SIMULATION FAILED"}
                        </div>
                        <p className="opacity-90">{simResult.message}</p>
                        {/* Visualizing dynamic metrics */}
                        <div className="flex gap-2 mt-2">
                            {simResult.metrics.map((m, i) => (
                                <div key={i} className={`text-[10px] px-2 py-1 rounded bg-background/50 border ${simResult.success ? "border-neon-green/30" : "border-neon-red/30"}`}>
                                    <span className="opacity-70 mr-1">{m.label}:</span>
                                    <span className="font-bold">{m.value}</span>
                                </div>
                            ))}
                        </div>
                    </motion.div>
                )}

                {/* VISUALIZERS (Phase 4 Injection) */}
                {!isPlanned && (
                    <div className="mb-4">
                        {experiment.id === "bio-neurogenesis" && (
                            <NetworkGraph
                                active={isRunning || !!simResult}
                                parameterValue={params["noise_scale"] || 0.5}
                            />
                        )}
                        {experiment.id === "phys-thermo" && (
                            <EnergyLandscape
                                active={isRunning || !!simResult}
                                temperature={params["temperature"] || 0.8}
                            />
                        )}
                        {experiment.id === "xeno-qualia" && (
                            <QualiaField
                                active={isRunning || !!simResult}
                                dimensions={params["dim_reduction"] || 3}
                            />
                        )}
                    </div>
                )}

                {/* DEFAULT METRICS (Static fallback) */}
                {!simResult && !isPlanned && experiment.metrics.length > 0 && !["bio-neurogenesis", "phys-thermo", "xeno-qualia"].includes(experiment.id) && (
                    <div className="grid grid-cols-2 gap-2 mb-4">
                        {experiment.metrics.map((metric, i) => (
                            <div key={i} className="bg-background/40 p-2 rounded border border-border/50">
                                <p className="font-mono text-[10px] text-muted-foreground uppercase">{metric.label}</p>
                                <p className="font-mono text-sm font-bold text-foreground">{metric.value}</p>
                            </div>
                        ))}
                    </div>
                )}

                {/* RESEARCH Q&A SECTION (Static Context) */}
                {!simResult && (
                    <div className="gap-3 flex flex-col mb-4 bg-background/20 p-3 rounded-lg border border-border/30">
                        {/* 1. The Hypothesis (Always visible) */}
                        <div className="flex gap-2">
                            <Lightbulb className="w-4 h-4 text-neon-yellow shrink-0 mt-0.5" />
                            <div>
                                <span className="font-mono text-[10px] uppercase text-muted-foreground mb-1 block">Hypothesis</span>
                                <p className="text-xs text-foreground/90 italic">"{experiment.hypothesis}"</p>
                            </div>
                        </div>

                        {/* 2. The Finding */}
                        {!isPlanned && experiment.finding && (
                            <div className="flex gap-2 border-t border-border/20 pt-2 mt-1">
                                <CheckCircle2 className="w-4 h-4 text-neon-green shrink-0 mt-0.5" />
                                <div>
                                    <span className="font-mono text-[10px] uppercase text-muted-foreground mb-1 block">Fact</span>
                                    <p className="text-xs text-foreground/90">{experiment.finding}</p>
                                </div>
                            </div>
                        )}
                        {/* 3. Future Work */}
                        {isPlanned && experiment.futureWork && (
                            <div className="flex gap-2 border-t border-border/20 pt-2 mt-1">
                                <Construction className="w-4 h-4 text-neon-red shrink-0 mt-0.5" />
                                <div>
                                    <span className="font-mono text-[10px] uppercase text-muted-foreground mb-1 block">Required Implementation</span>
                                    <p className="text-xs text-foreground/90 font-mono bg-black/30 p-1 rounded">{experiment.futureWork}</p>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {/* Logs */}
                {!isPlanned && (
                    <div className="bg-black/40 rounded-lg p-2 font-mono text-[10px] min-h-[80px] mb-4 overflow-hidden relative">
                        <div className="space-y-1">
                            {logs.length === 0 ? <p className="text-muted-foreground/40 text-center pt-6">Ready to run</p> : logs.map((log, i) => (
                                <p key={i} className="text-neon-cyan/80 truncate">{"> " + log}</p>
                            ))}
                        </div>
                        {isRunning && <div className="absolute bottom-0 left-0 h-0.5 bg-neon-cyan transition-all duration-300" style={{ width: `${progress}%` }} />}
                    </div>
                )}

                {/* Actions */}
                <button
                    onClick={handleRun}
                    disabled={isRunning || isPlanned}
                    className={`
            w-full py-2 rounded-lg font-mono text-xs font-bold flex items-center justify-center gap-2 transition-all mt-auto
            ${isPlanned
                            ? "bg-border/20 text-muted-foreground cursor-not-allowed border border-border/10"
                            : isRunning
                                ? "bg-background/20 text-muted-foreground cursor-not-allowed"
                                : "bg-neon-cyan/10 text-neon-cyan hover:bg-neon-cyan/20 border border-neon-cyan/30"}
          `}
                >
                    {isPlanned ? (
                        <>
                            <Microscope className="w-3 h-3" /> PENDING INTERFACE
                        </>
                    ) : isRunning ? "EXECUTING PROTOCOL..." : (
                        <>
                            <Play className="w-3 h-3" /> RUN EXPERIMENT
                        </>
                    )}
                </button>
            </div>
        </motion.div>
    );
}
