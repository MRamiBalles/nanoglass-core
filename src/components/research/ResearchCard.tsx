import { motion } from "framer-motion";
import { Check, Clock, ExternalLink } from "lucide-react";

export interface ResearchPaper {
  id: string;
  title: string;
  authors: string;
  category: string;
  year: number;
  status: "verified" | "theory";
  summary: string;
  focus?: string;
}

interface ResearchCardProps {
  paper: ResearchPaper;
  index: number;
}

export function ResearchCard({ paper, index }: ResearchCardProps) {
  return (
    <motion.article
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.4 }}
      whileHover={{ y: -4 }}
      className="glass-panel p-6 hover-glow group cursor-pointer relative overflow-hidden"
    >
      {/* Hover glow effect */}
      <div className="absolute inset-0 bg-gradient-to-br from-neon-cyan/0 to-neon-purple/0 group-hover:from-neon-cyan/5 group-hover:to-neon-purple/5 transition-all duration-500" />

      {/* Category & Status */}
      <div className="flex items-center justify-between mb-4 relative z-10">
        <span className="font-mono text-xs text-neon-purple/80 uppercase tracking-wider bg-neon-purple/10 px-2 py-1 rounded">
          {paper.category}
        </span>
        {paper.status === "verified" ? (
          <span className="flex items-center gap-1 font-mono text-xs text-neon-green bg-neon-green/10 px-2 py-1 rounded border border-neon-green/20">
            <Check className="w-3 h-3" />
            VERIFIED
          </span>
        ) : (
          <span className="flex items-center gap-1 font-mono text-xs text-neon-yellow bg-neon-yellow/10 px-2 py-1 rounded border border-neon-yellow/20">
            <Clock className="w-3 h-3" />
            THEORY
          </span>
        )}
      </div>

      {/* Title */}
      <h3 className="text-lg font-semibold text-foreground mb-2 group-hover:text-neon-cyan transition-colors relative z-10 line-clamp-2">
        {paper.title}
      </h3>

      {/* Authors & Year */}
      <p className="font-mono text-xs text-muted-foreground mb-3 relative z-10">
        {paper.authors} • {paper.year}
      </p>

      {/* Summary */}
      <p className="text-sm text-muted-foreground/80 mb-4 relative z-10 line-clamp-3">
        {paper.summary}
      </p>

      {/* Focus Tag */}
      {paper.focus && (
        <div className="flex items-center gap-2 relative z-10">
          <span className="font-mono text-[10px] text-muted-foreground/60 uppercase">Focus:</span>
          <span className="font-mono text-xs text-neon-cyan/70">{paper.focus}</span>
        </div>
      )}

      {/* Hover indicator */}
      <div className="absolute bottom-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
        <ExternalLink className="w-4 h-4 text-neon-cyan" />
      </div>
    </motion.article>
  );
}
