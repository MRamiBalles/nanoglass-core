import React from 'react';
import { BookOpen, CheckCircle, Clock } from 'lucide-react';

interface ResearchCardProps {
    id: number;
    title: string;
    category: string;
    status: 'verified' | 'theory';
    summary: string;
}

const ResearchCard: React.FC<ResearchCardProps> = ({ id, title, category, status, summary }) => {
    return (
        <div className="group relative p-6 rounded-xl bg-slate-900/40 border border-slate-800 hover:border-cyan-500/50 transition-all duration-300 hover:bg-slate-800/60 hover:-translate-y-1">
            <div className="absolute top-4 right-4">
                {status === 'verified' ? (
                    <span className="flex items-center gap-1 text-xs font-bold text-green-400 bg-green-400/10 px-2 py-1 rounded-full border border-green-400/20">
                        <CheckCircle className="w-3 h-3" /> VERIFIED
                    </span>
                ) : (
                    <span className="flex items-center gap-1 text-xs font-bold text-yellow-400 bg-yellow-400/10 px-2 py-1 rounded-full border border-yellow-400/20">
                        <Clock className="w-3 h-3" /> THEORY
                    </span>
                )}
            </div>

            <div className="text-cyan-500 text-xs mb-2 tracking-wide uppercase font-semibold">{category}</div>
            <h3 className="text-xl font-bold mb-3 group-hover:text-cyan-300 transition-colors">{title}</h3>
            <p className="text-slate-400 text-sm leading-relaxed mb-4">{summary}</p>

            <button className="flex items-center gap-2 text-cyan-400/60 text-sm group-hover:text-cyan-400 cursor-pointer transition-colors">
                <BookOpen className="w-4 h-4" /> Read Paper (Phase {id})
            </button>
        </div>
    );
};

export default ResearchCard;
