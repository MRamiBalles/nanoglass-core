import { motion } from "framer-motion";
import { Brain } from "lucide-react";
import { experiments } from "@/data/experiments";
import { ExperimentCard } from "@/components/labs/ExperimentCard";

export default function BioLab() {
    const bioExperiments = experiments.filter(e => e.category === "Bio-Mimetic");

    return (
        <div className="p-8">
            {/* Lab Header */}
            <motion.header
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-8"
            >
                <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg bg-neon-green/10 border border-neon-green/20">
                        <Brain className="w-8 h-8 text-neon-green" />
                    </div>
                    <div>
                        <h1 className="text-3xl font-bold tracking-tight text-foreground">
                            BIO-MIMETIC <span className="neon-green">LAB</span>
                        </h1>
                        <p className="font-mono text-sm text-muted-foreground mt-1">
                            NEUROPLASTICITY & BIOLOGICAL CONSTRAINTS • {bioExperiments.length} PROTOCOLS
                        </p>
                    </div>
                </div>
            </motion.header>

            {/* Experiments Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {bioExperiments.map((experiment) => (
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
