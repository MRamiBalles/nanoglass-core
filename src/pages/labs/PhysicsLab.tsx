import { motion } from "framer-motion";
import { Zap } from "lucide-react";
import { experiments } from "@/data/experiments";
import { ExperimentCard } from "@/components/labs/ExperimentCard";

export default function PhysicsLab() {
    const physExperiments = experiments.filter(e => e.category === "Physics");

    return (
        <div className="p-8">
            {/* Lab Header */}
            <motion.header
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-8"
            >
                <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg bg-neon-purple/10 border border-neon-purple/20">
                        <Zap className="w-8 h-8 text-neon-purple" />
                    </div>
                    <div>
                        <h1 className="text-3xl font-bold tracking-tight text-foreground">
                            PHYSICS <span className="neon-purple">LAB</span>
                        </h1>
                        <p className="font-mono text-sm text-muted-foreground mt-1">
                            THERMODYNAMICS OF MEANING & ENTROPY • {physExperiments.length} PROTOCOLS
                        </p>
                    </div>
                </div>
            </motion.header>

            {/* Experiments Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {physExperiments.map((experiment) => (
                    <ExperimentCard
                        key={experiment.id}
                        experiment={experiment}
                        onRun={(id) => console.log("Run experiment:", id)}
                    />
                ))}
            </div>
        </div>
    );
}
