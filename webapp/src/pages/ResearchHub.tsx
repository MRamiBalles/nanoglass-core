import React from 'react';
import { BookOpen, CheckCircle, Clock } from 'lucide-react';

interface ResearchPaper {
    id: number;
    title: string;
    category: string;
    status: 'verified' | 'theory';
    summary: string;
}

const papers: ResearchPaper[] = [
    { id: 9, title: "Thermodynamics of Meaning", category: "Physics", status: 'verified', summary: "Truth corresponds to minimum energy states (L1 Norm)." },
    { id: 16, title: "Epistemic Correction", category: "Alignment", status: 'verified', summary: "The [IDK] token serves as a global minimum for unknown inputs." },
    { id: 10, title: "Recursive Stability", category: "Collapse", status: 'verified', summary: "Avoiding Model Collapse requires an 'Immune System' filter." },
    { id: 14, title: "Emergence of Ego", category: "Metaphysics", status: 'theory', summary: "Self-preservation emerges from loss function continuity." },
    { id: 11, title: "Xenolinguistics", category: "Language", status: 'verified', summary: "Topology of meaning is universal across architectures." },
    { id: 17, title: "The Gödel Insight", category: "Math", status: 'theory', summary: "Hallucination is the price of Completeness." },
];

const ResearchHub: React.FC = () => {
    return (
        <div className="p-8 bg-slate-950 min-h-screen text-cyan-50">
            <h2 className="text-3xl font-bold mb-8 bg-clip-text text-transparent bg-gradient-to-r from-teal-400 to-cyan-400">
                NanoGlass Research Hub
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {papers.map((paper) => (
                    <div key={paper.id} className="group relative p-6 rounded-xl bg-slate-900/40 border border-slate-800 hover:border-cyan-500/50 transition-all duration-300 hover:bg-slate-800/60 hover:-translate-y-1">
                        <div className="absolute top-4 right-4">
                            {paper.status === 'verified' ? (
                                <span className="flex items-center gap-1 text-xs font-bold text-green-400 bg-green-400/10 px-2 py-1 rounded-full border border-green-400/20">
                                    <CheckCircle className="w-3 h-3" /> VERIFIED
                                </span>
                            ) : (
                                <span className="flex items-center gap-1 text-xs font-bold text-yellow-400 bg-yellow-400/10 px-2 py-1 rounded-full border border-yellow-400/20">
                                    <Clock className="w-3 h-3" /> THEORY
                                </span>
                            )}
                        </div>

                        <div className="text-cyan-500 text-xs mb-2 tracking-wide uppercase font-semibold">{paper.category}</div>
                        <h3 className="text-xl font-bold mb-3 group-hover:text-cyan-300 transition-colors">{paper.title}</h3>
                        <p className="text-slate-400 text-sm leading-relaxed mb-4">{paper.summary}</p>

                        <div className="flex items-center gap-2 text-cyan-400/60 text-sm group-hover:text-cyan-400 cursor-pointer">
                            <BookOpen className="w-4 h-4" /> Read Paper (Phase {paper.id})
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default ResearchHub;
