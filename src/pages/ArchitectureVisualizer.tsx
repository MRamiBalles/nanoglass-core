import { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { GlitchText } from "@/components/effects/GlitchText";
import { Cpu, ArrowRight, Layers, GitBranch, Zap } from "lucide-react";

// ── Architecture Definitions ──────────────────────────
interface ArchBlock {
  id: string;
  label: string;
  type: "embed" | "attention" | "ssm" | "mlp" | "moe" | "norm" | "output" | "router";
  color: string; // tailwind token
  description: string;
}

interface Architecture {
  name: string;
  ref: string;
  blocks: ArchBlock[];
  params: string;
  complexity: string;
  strengths: string[];
  weaknesses: string[];
}

const ARCHITECTURES: Record<string, Architecture> = {
  transformer: {
    name: "Transformer (GPT-2)",
    ref: "karpathy/nanoGPT",
    blocks: [
      { id: "emb", label: "Byte Embed", type: "embed", color: "primary", description: "Maps 257 byte tokens to d_model vectors. No tokenizer — raw bytes." },
      { id: "ln1", label: "LayerNorm", type: "norm", color: "muted-foreground", description: "Pre-norm stabilization. RMSNorm variant used in modern models." },
      { id: "attn", label: "Causal Self-Attention", type: "attention", color: "primary", description: "O(n²) full attention with causal mask. Each position attends to all previous positions." },
      { id: "ln2", label: "LayerNorm", type: "norm", color: "muted-foreground", description: "Second pre-norm before feedforward." },
      { id: "mlp", label: "MLP (SwiGLU)", type: "mlp", color: "neon-green", description: "4x expansion with gated activation. Where knowledge is stored." },
      { id: "out", label: "LM Head → 257", type: "output", color: "neon-yellow", description: "Projects to 257 logits. Includes [IDK] token for abstention." },
    ],
    params: "~10M",
    complexity: "O(n²d)",
    strengths: ["Full context attention", "Well-understood circuits", "Strong in-context learning"],
    weaknesses: ["Quadratic in sequence length", "KV cache memory", "No inherent recurrence"],
  },
  mamba: {
    name: "Mamba-2 (SSM)",
    ref: "state-spaces/mamba",
    blocks: [
      { id: "emb", label: "Byte Embed", type: "embed", color: "primary", description: "Same 257-token byte-level embedding." },
      { id: "ln1", label: "RMSNorm", type: "norm", color: "muted-foreground", description: "Efficient pre-norm variant." },
      { id: "ssm", label: "Selective SSM", type: "ssm", color: "secondary", description: "O(n) selective state space. Input-dependent gating compresses context into fixed-size state." },
      { id: "ln2", label: "RMSNorm", type: "norm", color: "muted-foreground", description: "Pre-norm before feedforward." },
      { id: "mlp", label: "MLP (SwiGLU)", type: "mlp", color: "neon-green", description: "Same feedforward, stores factual knowledge." },
      { id: "out", label: "LM Head → 257", type: "output", color: "neon-yellow", description: "Byte-level output with [IDK]." },
    ],
    params: "~8M",
    complexity: "O(nd)",
    strengths: ["Linear in sequence length", "Efficient inference (no KV cache)", "Strong on long sequences"],
    weaknesses: ["Weaker in-context retrieval", "Less interpretable than attention", "Newer, less studied"],
  },
  hybrid: {
    name: "Hybrid (GQA + Mamba)",
    ref: "NanoGlass Cortex-13",
    blocks: [
      { id: "emb", label: "Byte Embed", type: "embed", color: "primary", description: "Shared embedding for both pathways." },
      { id: "ln1", label: "RMSNorm", type: "norm", color: "muted-foreground", description: "Pre-norm." },
      { id: "gqa", label: "Grouped Query Attention", type: "attention", color: "primary", description: "GQA: 6 query heads share 2 KV heads. 3x memory reduction vs MHA." },
      { id: "ln2", label: "RMSNorm", type: "norm", color: "muted-foreground", description: "Intermediate norm." },
      { id: "ssm", label: "Selective SSM", type: "ssm", color: "secondary", description: "Mamba layer for long-range dependencies. Complements GQA's local precision." },
      { id: "ln3", label: "RMSNorm", type: "norm", color: "muted-foreground", description: "Pre-feedforward norm." },
      { id: "mlp", label: "MLP (SwiGLU)", type: "mlp", color: "neon-green", description: "Knowledge storage layer." },
      { id: "out", label: "LM Head → 257", type: "output", color: "neon-yellow", description: "Output with [IDK] abstention." },
    ],
    params: "~12M",
    complexity: "O(n·d + n²·d/g)",
    strengths: ["Best of both worlds", "GQA handles retrieval, SSM handles compression", "Our architecture"],
    weaknesses: ["More complex to train", "Harder to interpret", "Higher parameter count"],
  },
  moe: {
    name: "Mixture of Experts (MoE)",
    ref: "NanoGlass + GranularMoE",
    blocks: [
      { id: "emb", label: "Byte Embed", type: "embed", color: "primary", description: "Standard byte embedding." },
      { id: "ln1", label: "RMSNorm", type: "norm", color: "muted-foreground", description: "Pre-norm." },
      { id: "attn", label: "Causal Attention", type: "attention", color: "primary", description: "Standard causal attention for context mixing." },
      { id: "ln2", label: "RMSNorm", type: "norm", color: "muted-foreground", description: "Pre-MoE norm." },
      { id: "router", label: "Top-K Router", type: "router", color: "neon-red", description: "Learned gating network. Routes each token to K=2 of 8 experts. Load balancing via aux loss." },
      { id: "moe", label: "8× Expert MLPs", type: "moe", color: "neon-purple", description: "8 independent SwiGLU MLPs. Only 2 active per token → 4x effective compute reduction." },
      { id: "out", label: "LM Head → 257", type: "output", color: "neon-yellow", description: "Output projection." },
    ],
    params: "~40M (5M active)",
    complexity: "O(n²d + K·d_ff)",
    strengths: ["Massive capacity, low compute", "Experts specialize naturally", "Scales to huge models"],
    weaknesses: ["Load balancing is hard", "Expert collapse risk", "Communication overhead in distributed"],
  },
};

// ── Block Visual Component ────────────────────────────
function BlockVisual({ block, index, total, isActive }: { block: ArchBlock; index: number; total: number; isActive: boolean }) {
  const typeIcons: Record<string, string> = {
    embed: "▣", attention: "◈", ssm: "≋", mlp: "◆", moe: "❖", norm: "─", output: "▶", router: "⑂",
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.08 }}
      className="relative group"
    >
      <div className={`
        glass-panel p-3 rounded-lg border transition-all duration-300 cursor-pointer
        ${isActive ? `border-${block.color}/50 shadow-[0_0_15px_hsl(var(--${block.color})/0.2)]` : "border-border/30"}
        hover:border-${block.color}/40
      `}>
        <div className="flex items-center gap-2">
          <span className="text-lg">{typeIcons[block.type]}</span>
          <div className="flex-1 min-w-0">
            <p className={`font-mono text-xs font-bold text-${block.color}`}>{block.label}</p>
            <p className="text-[9px] font-mono text-muted-foreground truncate">{block.description.slice(0, 60)}…</p>
          </div>
        </div>
      </div>
      {index < total - 1 && (
        <div className="flex justify-center py-1">
          <ArrowRight className="w-3 h-3 text-muted-foreground/30 rotate-90" />
        </div>
      )}
      {/* Tooltip on hover */}
      <div className="absolute left-full ml-3 top-0 z-50 hidden group-hover:block w-64 glass-panel p-3 rounded-lg border border-border/50 shadow-xl">
        <p className="font-mono text-xs text-foreground font-bold mb-1">{block.label}</p>
        <p className="text-[10px] text-muted-foreground leading-relaxed">{block.description}</p>
        <p className="text-[9px] text-muted-foreground/60 mt-1 uppercase">Type: {block.type}</p>
      </div>
    </motion.div>
  );
}

// ── Data Flow Canvas ──────────────────────────────────
function DataFlowCanvas({ architecture, step }: { architecture: string; step: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animFrame: number;
    let t = step;

    const draw = () => {
      const w = canvas.width;
      const h = canvas.height;
      ctx.fillStyle = "rgba(0,0,0,0.15)";
      ctx.fillRect(0, 0, w, h);

      const numParticles = architecture === "moe" ? 40 : 20;

      for (let i = 0; i < numParticles; i++) {
        const phase = (t * 0.02 + i * 0.3) % 1;
        const x = w * 0.1 + (w * 0.8) * (0.5 + 0.4 * Math.sin(t * 0.01 + i));
        const y = phase * h;

        // Color by architecture
        const colors: Record<string, [number, number, number]> = {
          transformer: [187, 92, 53],
          mamba: [270, 91, 75],
          hybrid: [187, 92, 53],
          moe: [45, 93, 47],
        };
        const [hue, sat, light] = colors[architecture] || [187, 92, 53];

        const r = 2 + Math.sin(t * 0.03 + i) * 1.5;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${hue}, ${sat}%, ${light}%, ${0.6 + 0.3 * Math.sin(phase * Math.PI)})`;
        ctx.fill();

        // MoE: show routing splits
        if (architecture === "moe" && phase > 0.5 && phase < 0.7) {
          const routeX = x + (i % 2 === 0 ? 20 : -20);
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(routeX, y + 10);
          ctx.strokeStyle = `hsla(0, 84%, 60%, 0.3)`;
          ctx.stroke();
        }
      }

      t++;
      animFrame = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(animFrame);
  }, [architecture, step]);

  return <canvas ref={canvasRef} width={200} height={400} className="w-full h-full rounded-lg" />;
}

// ── Comparison Table ──────────────────────────────────
function ComparisonTable({ selected }: { selected: string[] }) {
  const archs = selected.map(k => ARCHITECTURES[k]).filter(Boolean);
  if (archs.length < 2) return null;

  const metrics = [
    { label: "Parameters", key: "params" },
    { label: "Complexity", key: "complexity" },
  ] as const;

  return (
    <div className="glass-panel p-4 rounded-lg overflow-x-auto">
      <h3 className="font-mono text-sm text-foreground font-bold mb-3 flex items-center gap-2">
        <GitBranch className="w-4 h-4 text-neon-purple" /> COMPARISON MATRIX
      </h3>
      <table className="w-full text-xs font-mono">
        <thead>
          <tr className="border-b border-border/30">
            <th className="text-left p-2 text-muted-foreground">Metric</th>
            {archs.map(a => (
              <th key={a.name} className="text-left p-2 text-primary">{a.name.split(" ")[0]}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {metrics.map(m => (
            <tr key={m.label} className="border-b border-border/10">
              <td className="p-2 text-muted-foreground">{m.label}</td>
              {archs.map(a => (
                <td key={a.name} className="p-2 text-foreground">{a[m.key]}</td>
              ))}
            </tr>
          ))}
          <tr className="border-b border-border/10">
            <td className="p-2 text-muted-foreground">Strengths</td>
            {archs.map(a => (
              <td key={a.name} className="p-2 text-neon-green text-[10px]">{a.strengths.join(", ")}</td>
            ))}
          </tr>
          <tr>
            <td className="p-2 text-muted-foreground">Weaknesses</td>
            {archs.map(a => (
              <td key={a.name} className="p-2 text-neon-red text-[10px]">{a.weaknesses.join(", ")}</td>
            ))}
          </tr>
        </tbody>
      </table>
    </div>
  );
}

// ── Main ──────────────────────────────────────────────
export default function ArchitectureVisualizer() {
  const [selected, setSelected] = useState<string[]>(["transformer", "hybrid"]);
  const [activeBlock, setActiveBlock] = useState<string | null>(null);

  const toggleArch = (key: string) => {
    setSelected(prev =>
      prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key].slice(-3)
    );
  };

  return (
    <div className="p-8">
      {/* Header */}
      <motion.header initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight text-foreground">
          <GlitchText text="ARCHITECTURE VISUALIZER" className="text-secondary" />
        </h1>
        <p className="font-mono text-sm text-muted-foreground mt-1">
          Compare Transformer, Mamba, Hybrid & MoE block-by-block
        </p>
      </motion.header>

      {/* Selector */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="glass-panel p-4 rounded-lg mb-6 flex flex-wrap gap-3"
      >
        {Object.entries(ARCHITECTURES).map(([key, arch]) => (
          <button
            key={key}
            onClick={() => toggleArch(key)}
            className={`px-4 py-2 rounded-lg font-mono text-xs border transition-all ${
              selected.includes(key)
                ? "bg-primary/10 border-primary/50 text-primary"
                : "border-border/30 text-muted-foreground hover:border-border/60"
            }`}
          >
            <Cpu className="w-3 h-3 inline mr-2" />
            {arch.name}
          </button>
        ))}
        <span className="ml-auto text-[9px] font-mono text-muted-foreground self-center">
          Select 2-3 to compare
        </span>
      </motion.div>

      {/* Architecture Columns */}
      <div className={`grid gap-4 mb-6 ${
        selected.length === 1 ? "grid-cols-1 max-w-lg" :
        selected.length === 2 ? "grid-cols-1 md:grid-cols-2" :
        "grid-cols-1 md:grid-cols-3"
      }`}>
        {selected.map(key => {
          const arch = ARCHITECTURES[key];
          if (!arch) return null;
          return (
            <motion.div
              key={key}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass-panel p-4 rounded-lg"
            >
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="font-mono text-sm text-foreground font-bold">{arch.name}</h3>
                  <p className="text-[9px] font-mono text-muted-foreground">{arch.ref} • {arch.params} • {arch.complexity}</p>
                </div>
                <Layers className="w-4 h-4 text-primary/50" />
              </div>

              <div className="grid grid-cols-[1fr_60px] gap-2">
                <div className="space-y-0">
                  {arch.blocks.map((block, i) => (
                    <BlockVisual
                      key={block.id}
                      block={block}
                      index={i}
                      total={arch.blocks.length}
                      isActive={activeBlock === block.id}
                    />
                  ))}
                </div>
                <DataFlowCanvas architecture={key} step={0} />
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Comparison Table */}
      <ComparisonTable selected={selected} />

      {/* Footer */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="mt-4 glass-panel p-4 rounded-lg font-mono text-xs text-muted-foreground flex items-center justify-between"
      >
        <div className="flex gap-6">
          <span>◈ Attention O(n²) <span className="text-primary">quadratic</span></span>
          <span>≋ SSM O(n) <span className="text-secondary">linear</span></span>
          <span>❖ MoE K/N <span className="text-neon-purple">sparse</span></span>
        </div>
        <span className="text-muted-foreground/60">
          Refs: <a href="https://github.com/karpathy/nanoGPT" target="_blank" rel="noopener" className="text-primary/60 hover:text-primary mr-2">nanoGPT</a>
          <a href="https://github.com/karpathy/llm.c" target="_blank" rel="noopener" className="text-primary/60 hover:text-primary">llm.c</a>
        </span>
      </motion.div>
    </div>
  );
}
