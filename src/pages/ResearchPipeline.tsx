import { useState, useRef, useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { GlitchText } from "@/components/effects/GlitchText";
import { BookOpen, GitBranch, Search, ExternalLink, ChevronRight, Lightbulb } from "lucide-react";
import { researchPapers } from "@/data/researchPapers";

// ── Types ─────────────────────────────────────────────
interface PaperNode {
  id: string;
  title: string;
  authors: string;
  category: string;
  year: number;
  status: "verified" | "theory";
  summary: string;
  focus: string;
  x: number;
  y: number;
  connections: string[];
}

interface Hypothesis {
  id: string;
  title: string;
  from: string[];
  confidence: number;
  description: string;
}

// ── Build Knowledge Graph ─────────────────────────────
function buildGraph(papers: typeof researchPapers): { nodes: PaperNode[]; hypotheses: Hypothesis[] } {
  // Create nodes with positions
  const nodes: PaperNode[] = papers.map((p, i) => {
    const angle = (i / papers.length) * Math.PI * 2;
    const radius = 250 + (i % 3) * 60;
    return {
      ...p,
      status: p.status as "verified" | "theory",
      x: 400 + Math.cos(angle) * radius,
      y: 350 + Math.sin(angle) * radius,
      connections: [],
    };
  });

  // Auto-detect connections by shared keywords in focus/category
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i];
      const b = nodes[j];
      const aWords = new Set([...a.focus.toLowerCase().split(/[\s,]+/), ...a.category.toLowerCase().split(/[\s,]+/)]);
      const bWords = new Set([...b.focus.toLowerCase().split(/[\s,]+/), ...b.category.toLowerCase().split(/[\s,]+/)]);
      const overlap = [...aWords].filter(w => bWords.has(w) && w.length > 3).length;
      if (overlap >= 1 || a.category === b.category) {
        a.connections.push(b.id);
        b.connections.push(a.id);
      }
    }
  }

  // Auto-generate hypotheses from connected clusters
  const hypotheses: Hypothesis[] = [
    {
      id: "h1",
      title: "Truth is Thermodynamic Equilibrium",
      from: ["eg-02", "6", "9"],
      confidence: 0.82,
      description: "If Free Energy Principle (Friston) + Information Bottleneck (Tishby) + our L1 Sparsity results converge, then Truth = minimum free energy state in weight space.",
    },
    {
      id: "h2",
      title: "Symbolic Reasoning is a Phase Transition",
      from: ["eg-00", "eg-01", "4"],
      confidence: 0.74,
      description: "Cortex-13 + Universality check + Emergent Architectures suggest symbolic logic crystallizes at a specific sparsity threshold, analogous to a physical phase transition.",
    },
    {
      id: "h3",
      title: "[IDK] Token as Metacognitive Probe",
      from: ["8", "eg-00", "5"],
      confidence: 0.91,
      description: "Language Models Know What They Know (Kadavath) + Glass Box abstention + Global Workspace Theory → [IDK] token is a measurable metacognitive signal.",
    },
    {
      id: "h4",
      title: "Neurogenesis Reverses Model Collapse",
      from: ["eg-03", "eg-02"],
      confidence: 0.67,
      description: "If Loss of Plasticity is analogous to biological senescence, then targeted weight re-initialization (neurogenesis) should reopen the critical learning window.",
    },
    {
      id: "h5",
      title: "Hybrid Architecture Converges to Universal Representation",
      from: ["eg-01", "7", "eg-00"],
      confidence: 0.58,
      description: "Platonic Representation Hypothesis + Convergent Evolution in Mambas + Cortex-13 results suggest all sufficiently trained architectures converge to the same geometric structure.",
    },
  ];

  return { nodes, hypotheses };
}

// ── Knowledge Graph Canvas ────────────────────────────
function KnowledgeGraph({
  nodes,
  selectedId,
  onSelect,
  highlightIds,
}: {
  nodes: PaperNode[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  highlightIds: Set<string>;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Draw connections
    for (const node of nodes) {
      for (const connId of node.connections) {
        const target = nodes.find(n => n.id === connId);
        if (!target) continue;
        const isHighlight = highlightIds.has(node.id) && highlightIds.has(target.id);
        ctx.beginPath();
        ctx.moveTo(node.x, node.y);
        ctx.lineTo(target.x, target.y);
        ctx.strokeStyle = isHighlight
          ? "hsla(187, 92%, 53%, 0.5)"
          : "hsla(217, 33%, 30%, 0.2)";
        ctx.lineWidth = isHighlight ? 2 : 0.5;
        ctx.stroke();
      }
    }

    // Draw nodes
    for (const node of nodes) {
      const isSelected = node.id === selectedId;
      const isHighlighted = highlightIds.has(node.id);
      const r = isSelected ? 12 : isHighlighted ? 9 : 6;

      // Glow
      if (isSelected || isHighlighted) {
        const grad = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, r * 3);
        const glowColor = node.status === "verified" ? "187, 92%, 53%" : "45, 93%, 47%";
        grad.addColorStop(0, `hsla(${glowColor}, 0.3)`);
        grad.addColorStop(1, "transparent");
        ctx.fillStyle = grad;
        ctx.fillRect(node.x - r * 3, node.y - r * 3, r * 6, r * 6);
      }

      ctx.beginPath();
      ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
      ctx.fillStyle = node.status === "verified"
        ? `hsla(187, 92%, 53%, ${isSelected ? 1 : isHighlighted ? 0.8 : 0.5})`
        : `hsla(45, 93%, 47%, ${isSelected ? 1 : isHighlighted ? 0.8 : 0.5})`;
      ctx.fill();

      // Label
      if (isSelected || isHighlighted) {
        ctx.font = "9px monospace";
        ctx.fillStyle = "hsla(210, 40%, 90%, 0.9)";
        ctx.fillText(node.title.slice(0, 30), node.x + r + 4, node.y + 3);
      }
    }
  }, [nodes, selectedId, highlightIds]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);

    for (const node of nodes) {
      const dx = node.x - x;
      const dy = node.y - y;
      if (dx * dx + dy * dy < 200) {
        onSelect(node.id);
        return;
      }
    }
  };

  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={700}
      className="w-full h-full cursor-pointer"
      onClick={handleClick}
    />
  );
}

// ── Main Component ────────────────────────────────────
export default function ResearchPipeline() {
  const [selectedPaper, setSelectedPaper] = useState<string | null>(null);
  const [selectedHypothesis, setSelectedHypothesis] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  const { nodes, hypotheses } = useMemo(() => buildGraph(researchPapers), []);

  const selectedNode = nodes.find(n => n.id === selectedPaper);
  const activeHyp = hypotheses.find(h => h.id === selectedHypothesis);

  // Highlight nodes related to selected hypothesis
  const highlightIds = useMemo(() => {
    if (activeHyp) return new Set(activeHyp.from);
    if (selectedPaper) {
      const node = nodes.find(n => n.id === selectedPaper);
      return new Set([selectedPaper, ...(node?.connections || [])]);
    }
    return new Set<string>();
  }, [selectedPaper, activeHyp, nodes]);

  const filteredPapers = searchQuery
    ? nodes.filter(n =>
        n.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        n.category.toLowerCase().includes(searchQuery.toLowerCase()) ||
        n.focus.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : nodes;

  return (
    <div className="p-8">
      {/* Header */}
      <motion.header initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight text-foreground">
          <GlitchText text="RESEARCH PIPELINE" className="text-neon-green" />
        </h1>
        <p className="font-mono text-sm text-muted-foreground mt-1">
          autoresearch-style hypothesis generation • Knowledge graph from {nodes.length} papers
        </p>
      </motion.header>

      {/* Search */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="glass-panel p-3 rounded-lg mb-6 flex items-center gap-3"
      >
        <Search className="w-4 h-4 text-muted-foreground" />
        <input
          type="text"
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          placeholder="Search papers, categories, keywords..."
          className="flex-1 bg-transparent text-sm font-mono text-foreground placeholder:text-muted-foreground/40 outline-none"
        />
        <span className="text-[9px] font-mono text-muted-foreground">{filteredPapers.length} papers</span>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Knowledge Graph — 2/3 */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="lg:col-span-2 glass-panel p-4 rounded-lg"
        >
          <div className="flex items-center gap-2 mb-3">
            <GitBranch className="w-4 h-4 text-primary" />
            <h3 className="font-mono text-sm text-foreground font-bold">KNOWLEDGE GRAPH</h3>
            <div className="ml-auto flex items-center gap-3 text-[9px] font-mono">
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-primary" /> Verified
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-neon-yellow" /> Theory
              </span>
            </div>
          </div>
          <div className="h-[500px] bg-background/50 rounded-lg overflow-hidden">
            <KnowledgeGraph
              nodes={nodes}
              selectedId={selectedPaper}
              onSelect={setSelectedPaper}
              highlightIds={highlightIds}
            />
          </div>
        </motion.div>

        {/* Right Panel */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="space-y-4"
        >
          {/* Selected Paper Detail */}
          <div className="glass-panel p-4 rounded-lg">
            <h3 className="font-mono text-sm text-foreground font-bold mb-3 flex items-center gap-2">
              <BookOpen className="w-4 h-4 text-primary" />
              PAPER DETAIL
            </h3>
            {selectedNode ? (
              <div className="space-y-2">
                <p className="font-mono text-xs text-foreground font-bold">{selectedNode.title}</p>
                <p className="text-[10px] text-muted-foreground">{selectedNode.authors} • {selectedNode.year}</p>
                <div className="flex gap-2">
                  <span className={`text-[9px] font-mono px-2 py-0.5 rounded ${
                    selectedNode.status === "verified" ? "bg-neon-green/10 text-neon-green" : "bg-neon-yellow/10 text-neon-yellow"
                  }`}>
                    {selectedNode.status.toUpperCase()}
                  </span>
                  <span className="text-[9px] font-mono px-2 py-0.5 rounded bg-primary/10 text-primary">
                    {selectedNode.category}
                  </span>
                </div>
                <p className="text-[10px] text-muted-foreground leading-relaxed">{selectedNode.summary}</p>
                <p className="text-[9px] font-mono text-primary/70">Focus: {selectedNode.focus}</p>
                <p className="text-[9px] font-mono text-muted-foreground">
                  Connections: {selectedNode.connections.length} papers
                </p>
              </div>
            ) : (
              <p className="text-xs font-mono text-muted-foreground text-center py-6">
                Click a node in the graph
              </p>
            )}
          </div>

          {/* Hypotheses */}
          <div className="glass-panel p-4 rounded-lg">
            <h3 className="font-mono text-sm text-foreground font-bold mb-3 flex items-center gap-2">
              <Lightbulb className="w-4 h-4 text-neon-yellow" />
              AUTO-HYPOTHESES
            </h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {hypotheses.map(h => (
                <motion.button
                  key={h.id}
                  onClick={() => {
                    setSelectedHypothesis(selectedHypothesis === h.id ? null : h.id);
                    setSelectedPaper(null);
                  }}
                  className={`w-full text-left p-2 rounded-lg border transition-all ${
                    selectedHypothesis === h.id
                      ? "border-neon-yellow/50 bg-neon-yellow/5"
                      : "border-border/20 hover:border-border/40"
                  }`}
                  whileHover={{ scale: 1.01 }}
                >
                  <div className="flex items-start gap-2">
                    <ChevronRight className={`w-3 h-3 mt-0.5 text-neon-yellow transition-transform ${
                      selectedHypothesis === h.id ? "rotate-90" : ""
                    }`} />
                    <div className="flex-1 min-w-0">
                      <p className="font-mono text-[10px] text-foreground font-bold truncate">{h.title}</p>
                      <div className="flex items-center gap-2 mt-1">
                        <div className="flex-1 h-1 bg-background/50 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-neon-yellow/60 rounded-full"
                            style={{ width: `${h.confidence * 100}%` }}
                          />
                        </div>
                        <span className="text-[8px] font-mono text-neon-yellow">{(h.confidence * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  </div>
                </motion.button>
              ))}
            </div>

            {/* Expanded hypothesis */}
            <AnimatePresence>
              {activeHyp && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-3 p-3 bg-neon-yellow/5 rounded-lg border border-neon-yellow/20"
                >
                  <p className="text-[10px] text-muted-foreground leading-relaxed">{activeHyp.description}</p>
                  <p className="text-[9px] font-mono text-neon-yellow/70 mt-2">
                    Sources: {activeHyp.from.map(id => {
                      const p = nodes.find(n => n.id === id);
                      return p?.title.split(":")[0] || id;
                    }).join(" + ")}
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      </div>

      {/* Footer */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="mt-4 glass-panel p-4 rounded-lg font-mono text-xs text-muted-foreground flex items-center justify-between"
      >
        <div className="flex gap-6">
          <span>PAPERS: <span className="text-primary">{nodes.length}</span></span>
          <span>CONNECTIONS: <span className="text-secondary">{nodes.reduce((s, n) => s + n.connections.length, 0) / 2}</span></span>
          <span>HYPOTHESES: <span className="text-neon-yellow">{hypotheses.length}</span></span>
        </div>
        <span className="text-muted-foreground/60">
          Ref: <a href="https://github.com/karpathy/autoresearch" target="_blank" rel="noopener" className="text-primary/60 hover:text-primary">karpathy/autoresearch</a>
        </span>
      </motion.div>
    </div>
  );
}
