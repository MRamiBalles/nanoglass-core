#!/usr/bin/env python3
"""
==============================================================================
🕸️ KNOWLEDGE GRAPH GENERATOR - Idea2Story Philosophy Integration
==============================================================================
Generates a lightweight Knowledge Graph (JSON) from project structure.

Node Types (following Idea2Story):
    - Paper: Research papers (from researchPapers.ts + research_*.tex)
    - Pattern: Experimental patterns (from experiments/*.py)
    - Domain: Knowledge domains (Thermodynamics, Interpretability, etc.)

Edge Types:
    - belongs_to: Paper/Pattern -> Domain
    - cites: Paper -> Paper
    - validates: Pattern -> Paper

Usage:
    python scripts/generate_kg.py
    
Output:
    data/knowledge_graph.json
==============================================================================
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RESEARCH_PAPERS_TS = PROJECT_ROOT / "src" / "data" / "researchPapers.ts"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESEARCH_TEX_DIR = PROJECT_ROOT
OUTPUT_FILE = PROJECT_ROOT / "data" / "knowledge_graph.json"


@dataclass
class Node:
    """A node in the Knowledge Graph."""
    id: str
    type: str  # "paper", "pattern", "domain"
    name: str
    metadata: Dict[str, Any]


@dataclass
class Edge:
    """An edge in the Knowledge Graph."""
    source: str
    target: str
    type: str  # "belongs_to", "cites", "validates"


class KnowledgeGraphGenerator:
    def __init__(self):
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.domains: set = set()
    
    def parse_research_papers_ts(self) -> List[Node]:
        """Parse researchPapers.ts to extract paper nodes."""
        papers = []
        
        if not RESEARCH_PAPERS_TS.exists():
            print(f"   ⚠️  {RESEARCH_PAPERS_TS} not found")
            return papers
        
        content = RESEARCH_PAPERS_TS.read_text(encoding="utf-8")
        
        # Regex to extract paper objects
        paper_pattern = r'\{\s*id:\s*"([^"]+)".*?title:\s*"([^"]+)".*?category:\s*"([^"]+)".*?status:\s*"([^"]+)"'
        matches = re.findall(paper_pattern, content, re.DOTALL)
        
        for match in matches:
            paper_id, title, category, status = match
            
            # Track domain
            self.domains.add(category)
            
            papers.append(Node(
                id=f"paper:{paper_id}",
                type="paper",
                name=title[:80],  # Truncate long titles
                metadata={
                    "category": category,
                    "status": status,
                    "source": "researchPapers.ts"
                }
            ))
            
            # Add edge to domain
            self.edges.append(Edge(
                source=f"paper:{paper_id}",
                target=f"domain:{category.lower().replace(' ', '_')}",
                type="belongs_to"
            ))
        
        print(f"   📄 Parsed {len(papers)} papers from researchPapers.ts")
        return papers
    
    def parse_experiments(self) -> List[Node]:
        """Parse experiments/*.py to extract pattern nodes."""
        patterns = []
        
        if not EXPERIMENTS_DIR.exists():
            print(f"   ⚠️  {EXPERIMENTS_DIR} not found")
            return patterns
        
        # Pattern categories based on filename
        category_map = {
            "thermo": "Thermodynamics",
            "causal": "Causality",
            "seal": "Epistemic",
            "truthful": "Truthfulness",
            "rlvr": "Reinforcement",
            "qualia": "Consciousness",
            "xeno": "Xenolinguistics",
            "reliability": "Reliability",
            "adversarial": "Adversarial",
            "plasticity": "Neurogenesis",
        }
        
        for py_file in EXPERIMENTS_DIR.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            
            # Determine category from filename
            category = "General"
            for key, cat in category_map.items():
                if key in py_file.name.lower():
                    category = cat
                    break
            
            self.domains.add(category)
            
            # Extract docstring if present
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
            docstring = docstring_match.group(1).strip()[:200] if docstring_match else ""
            
            patterns.append(Node(
                id=f"pattern:{py_file.stem}",
                type="pattern",
                name=py_file.stem.replace("_", " ").title(),
                metadata={
                    "file": py_file.name,
                    "category": category,
                    "docstring": docstring[:100] + "..." if len(docstring) > 100 else docstring
                }
            ))
            
            # Add edge to domain
            self.edges.append(Edge(
                source=f"pattern:{py_file.stem}",
                target=f"domain:{category.lower().replace(' ', '_')}",
                type="belongs_to"
            ))
        
        print(f"   🧪 Parsed {len(patterns)} patterns from experiments/")
        return patterns
    
    def parse_research_tex(self) -> List[Node]:
        """Parse research_*.tex files for additional papers."""
        papers = []
        
        for tex_file in RESEARCH_TEX_DIR.glob("research_*.tex"):
            content = tex_file.read_text(encoding="utf-8", errors="ignore")
            
            # Extract title
            title_match = re.search(r'\\title\{([^}]+)\}', content)
            title = title_match.group(1) if title_match else tex_file.stem
            
            # Clean up LaTeX
            title = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', title)
            title = title.replace('\\\\', ' ').strip()[:80]
            
            papers.append(Node(
                id=f"paper:tex_{tex_file.stem}",
                type="paper",
                name=title,
                metadata={
                    "file": tex_file.name,
                    "source": "LaTeX",
                    "status": "internal"
                }
            ))
        
        print(f"   📝 Parsed {len(papers)} papers from *.tex files")
        return papers
    
    def create_domain_nodes(self) -> List[Node]:
        """Create domain nodes from collected domains."""
        domains = []
        for domain in self.domains:
            domains.append(Node(
                id=f"domain:{domain.lower().replace(' ', '_')}",
                type="domain",
                name=domain,
                metadata={}
            ))
        print(f"   🏷️  Created {len(domains)} domain nodes")
        return domains
    
    def generate(self) -> Dict[str, Any]:
        """Generate the complete Knowledge Graph."""
        print("\n" + "=" * 60)
        print("  🕸️  Knowledge Graph Generator")
        print("=" * 60)
        
        # Parse sources
        self.nodes.extend(self.parse_research_papers_ts())
        self.nodes.extend(self.parse_experiments())
        self.nodes.extend(self.parse_research_tex())
        self.nodes.extend(self.create_domain_nodes())
        
        # Build graph structure
        graph = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator": "NanoGlass KG Generator v1.0",
                "philosophy": "Idea2Story (pre-computation Knowledge Graph)",
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges)
            },
            "nodes": {
                "papers": [asdict(n) for n in self.nodes if n.type == "paper"],
                "patterns": [asdict(n) for n in self.nodes if n.type == "pattern"],
                "domains": [asdict(n) for n in self.nodes if n.type == "domain"]
            },
            "edges": [asdict(e) for e in self.edges]
        }
        
        print("-" * 60)
        print(f"   Total nodes: {len(self.nodes)}")
        print(f"   Total edges: {len(self.edges)}")
        print(f"   Domains: {len(self.domains)}")
        print("=" * 60)
        
        return graph
    
    def save(self, graph: Dict[str, Any]):
        """Save graph to JSON file."""
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2, ensure_ascii=False)
        
        print(f"\n   💾 Saved to: {OUTPUT_FILE}")


def main():
    generator = KnowledgeGraphGenerator()
    graph = generator.generate()
    generator.save(graph)
    
    # Verify
    print("\n   ✅ Knowledge Graph generated successfully!")
    print(f"   📊 Papers: {len(graph['nodes']['papers'])}")
    print(f"   🧪 Patterns: {len(graph['nodes']['patterns'])}")
    print(f"   🏷️  Domains: {len(graph['nodes']['domains'])}")


if __name__ == "__main__":
    main()
