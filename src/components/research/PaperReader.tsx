import { motion } from "framer-motion";
import { X, Calendar, User, Tag, Download } from "lucide-react";
import { ResearchPaper } from "@/components/research/ResearchCard";

interface PaperReaderProps {
    paper: ResearchPaper;
    onClose: () => void;
}

export function PaperReader({ paper, onClose }: PaperReaderProps) {
    // Mock content generation based on paper ID
    const getMockContent = (id: string) => {
        if (id.startsWith("eg-")) {
            return (
                <div className="space-y-6">
                    <div className="p-4 rounded-lg bg-neon-purple/10 border border-neon-purple/20">
                        <h3 className="text-lg font-bold text-neon-purple mb-2">Electric Gravity Labs Abstract</h3>
                        <p className="text-muted-foreground">{paper.summary}</p>
                    </div>

                    <div className="prose prose-invert max-w-none">
                        <h4>1. Introduction</h4>
                        <p>
                            The gap between connectionist efficiency and symbolic interpretability ("The Glass Box") remains the central challenge of AGI.
                            In this paper, we demonstrate that <strong>{paper.focus}</strong> is not just a feature but a fundamental necessity for stable intelligence.
                        </p>

                        <h4>2. Methodology: Cortex-13</h4>
                        <p>
              We utilized a Hybrid Mamba-Transformer architecture trained on a curriculum of "Innate" (Pre-training) -> "Social" (Instruction Tuning) -> "Specialized" (MoE Routing).
                            Our results show a statistically significant phase transition (p &lt; 0.05) in the L1 Sparsity of the activation space.
                        </p>

                        <h4>3. Key Findings</h4>
                        <ul className="list-disc pl-5 space-y-2">
                            <li><strong>Sparsity Transition:</strong> The model moves from dense, entropic representations to sparse, boolean-like logic gates.</li>
                            <li><strong>Truth as Attractor:</strong> RLHF acts as a thermodynamic cooling process, settling the system into low-energy states.</li>
                            <li><strong>Convergent Evolution:</strong> Different architectures (Mamba vs Transformer) discover the same "Concept Basins".</li>
                        </ul>

                        <h4>4. Conclusion</h4>
                        <p>
                            "Electric Gravity" is the force that pulls high-dimensional noise into low-dimensional meaning.
                        </p>
                    </div>
                </div>
            );
        }
        return (
            <div className="space-y-4">
                <div className="p-4 rounded-lg bg-muted border border-border">
                    <p className="text-muted-foreground italic">External Reference. Full text available via DOI lookup.</p>
                </div>
                <p className="text-foreground">
                    This paper ({paper.title}) is a foundational reference for the Electric Gravity project.
                    Please refer to the official publication for the full text.
                </p>
            </div>
        );
    };

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm"
            onClick={onClose}
        >
            <div
                className="w-full max-w-3xl max-h-[85vh] overflow-hidden bg-background border border-neon-cyan/30 rounded-xl shadow-2xl flex flex-col"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header */}
                <div className="p-6 border-b border-border bg-muted/20 flex items-start justify-between">
                    <div>
                        <div className="flex items-center gap-2 mb-2">
                            <span className={`px-2 py-0.5 rounded text-[10px] font-mono border ${paper.id.startsWith("eg")
                                    ? "bg-neon-purple/20 text-neon-purple border-neon-purple/30"
                                    : "bg-muted text-muted-foreground border-border"
                                }`}>
                                {paper.category.toUpperCase()}
                            </span>
                            <span className="text-xs font-mono text-muted-foreground">{paper.year}</span>
                        </div>
                        <h2 className="text-2xl font-bold text-foreground leading-tight">{paper.title}</h2>
                        <div className="flex items-center gap-4 mt-3 text-sm text-muted-foreground font-mono">
                            <span className="flex items-center gap-1"><User className="w-3 h-3" /> {paper.authors}</span>
                            <span className="flex items-center gap-1"><Tag className="w-3 h-3" /> {paper.focus}</span>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 rounded-full hover:bg-muted transition-colors"
                    >
                        <X className="w-5 h-5 text-muted-foreground" />
                    </button>
                </div>

                {/* Content - Scrollable */}
                <div className="flex-1 overflow-y-auto p-8 font-serif leading-relaxed text-lg text-foreground/90 bg-background">
                    {getMockContent(paper.id)}
                </div>

                {/* Footer */}
                <div className="p-4 border-t border-border bg-muted/20 flex justify-end gap-3">
                    <button className="flex items-center gap-2 px-4 py-2 rounded-lg bg-background border border-border hover:border-neon-cyan/50 transition-colors text-sm font-mono">
                        <Download className="w-4 h-4" /> Export PDF
                    </button>
                </div>
            </div>
        </motion.div>
    );
}
