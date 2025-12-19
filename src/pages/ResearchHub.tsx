import { motion } from "framer-motion";
import { BookOpen, Filter, Search } from "lucide-react";
import { useState } from "react";
import { ResearchCard } from "@/components/research/ResearchCard";
import { researchPapers } from "@/data/researchPapers";

const categories = ["All", "Interpretability", "Mechanistic", "Consciousness", "Thermodynamics", "Philosophy of Mind", "Information Theory"];

export default function ResearchHub() {
  const [selectedCategory, setSelectedCategory] = useState("All");
  const [searchQuery, setSearchQuery] = useState("");

  const filteredPapers = researchPapers.filter((paper) => {
    const matchesCategory = selectedCategory === "All" || paper.category === selectedCategory;
    const matchesSearch = 
      paper.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      paper.authors.toLowerCase().includes(searchQuery.toLowerCase()) ||
      paper.summary.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  return (
    <div className="p-8">
      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 rounded-lg bg-neon-purple/10 border border-neon-purple/20">
            <BookOpen className="w-6 h-6 text-neon-purple" />
          </div>
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-foreground">
              RESEARCH <span className="neon-purple">HUB</span>
            </h1>
            <p className="font-mono text-sm text-muted-foreground mt-1">
              THE GIANTS ON WHOSE SHOULDERS WE STAND • {researchPapers.length} PAPERS INDEXED
            </p>
          </div>
        </div>
      </motion.header>

      {/* Filters */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="glass-panel p-4 mb-6"
      >
        <div className="flex flex-col lg:flex-row gap-4">
          {/* Search */}
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search papers by title, author, or keywords..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-background/50 border border-border rounded-lg pl-10 pr-4 py-2.5 font-mono text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-neon-cyan/50 focus:ring-1 focus:ring-neon-cyan/30 transition-all"
            />
          </div>

          {/* Category Filter */}
          <div className="flex items-center gap-2 flex-wrap">
            <Filter className="w-4 h-4 text-muted-foreground" />
            {categories.map((category) => (
              <button
                key={category}
                onClick={() => setSelectedCategory(category)}
                className={`
                  px-3 py-1.5 rounded-lg font-mono text-xs transition-all
                  ${selectedCategory === category
                    ? "bg-neon-cyan/20 text-neon-cyan border border-neon-cyan/30"
                    : "bg-background/50 text-muted-foreground border border-border hover:border-neon-cyan/20 hover:text-foreground"
                  }
                `}
              >
                {category}
              </button>
            ))}
          </div>
        </div>
      </motion.div>

      {/* Papers Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {filteredPapers.map((paper, index) => (
          <ResearchCard key={paper.id} paper={paper} index={index} />
        ))}
      </div>

      {/* Empty State */}
      {filteredPapers.length === 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass-panel p-12 text-center"
        >
          <BookOpen className="w-12 h-12 text-muted-foreground/30 mx-auto mb-4" />
          <p className="font-mono text-muted-foreground">No papers found matching your criteria.</p>
          <button
            onClick={() => { setSelectedCategory("All"); setSearchQuery(""); }}
            className="mt-4 font-mono text-sm text-neon-cyan hover:underline"
          >
            Clear filters
          </button>
        </motion.div>
      )}

      {/* Stats Footer */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="mt-8 glass-panel p-4"
      >
        <div className="flex items-center justify-between font-mono text-xs text-muted-foreground">
          <div className="flex items-center gap-6">
            <span>
              VERIFIED: <span className="text-neon-green">{researchPapers.filter(p => p.status === "verified").length}</span>
            </span>
            <span>
              THEORY: <span className="text-neon-yellow">{researchPapers.filter(p => p.status === "theory").length}</span>
            </span>
            <span>
              SHOWING: <span className="text-foreground">{filteredPapers.length}/{researchPapers.length}</span>
            </span>
          </div>
          <span className="text-muted-foreground/60">
            Bibliography sourced from Project NanoGlass archives
          </span>
        </div>
      </motion.div>
    </div>
  );
}
