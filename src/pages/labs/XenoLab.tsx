import { motion } from "framer-motion";
import { Eye } from "lucide-react";
import { experiments } from "@/data/experiments";
import { ExperimentCard } from "@/components/labs/ExperimentCard";
import { ClearanceGate } from "@/components/labs/ClearanceGate";

export default function XenoLab() {
    const xenoExperiments = experiments.filter(e => e.category === "Xeno-Cognition");

    return (
        <ClearanceGate requiredLevel={5} title="XENO-COGNITION LABORATORY">
            <div className="p-8">
                {/* Lab Header */}
                <motion.header
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-8"
                >
                    <div className="flex items-center gap-3 mb-2">
                        <div className="p-2 rounded-lg bg-neon-yellow/10 border border-neon-yellow/20">
                            <Eye className="w-8 h-8 text-neon-yellow" />
                        </div>
                        <div>
                            <h1 className="text-3xl font-bold tracking-tight text-foreground">
                                XENO-COGNITION <span className="neon-yellow">LAB</span>
                            </h1>
                            <p className="font-mono text-sm text-muted-foreground mt-1">
                                UNIVERSALITY & ALIEN LOGIC • {xenoExperiments.length} PROTOCOLS
                            </p>
                        </div>
                    </div>
                </motion.header>

                {/* Experiments Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {xenoExperiments.map((experiment) => (
                        <ExperimentCard
                            key={experiment.id}
                            experiment={experiment}
                            onRun={(id) => console.log("Run experiment:", id)}
                        />
                    ))}
                </div>
            </div>
        </ClearanceGate>
    );
}
