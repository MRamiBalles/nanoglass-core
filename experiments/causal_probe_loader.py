"""
==============================================================================
CausalProbe-2024 Benchmark Loader
==============================================================================
Loads and manages the CausalProbe-2024 benchmark dataset for evaluating
causal reasoning capabilities in LLMs (Chi et al., June 2025).

Key Features:
    - Temporal novelty: Data sources post-January 2024 to prevent memorization
    - Difficulty splits: Easy (single-hop) and Hard (multi-hop) subsets
    - Metrics: Exact Match, Causal Consistency Score

Reference:
    Chi et al. (2025) "Unveiling Causal Reasoning in Large Language Models"

==============================================================================
"""
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import random

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class CausalExample:
    """A single causal reasoning example."""
    id: str
    context: str                    # Background information (post-Jan 2024)
    question: str                   # Causal question
    cause: str                      # Correct cause
    effect: str                     # Correct effect
    answer: str                     # Expected answer
    difficulty: str                 # "easy" or "hard"
    reasoning_chain: List[str]      # Expected reasoning steps
    distractors: List[str]          # Incorrect alternatives
    source_date: str                # Publication date of source (for verification)


@dataclass
class CausalProbeDataset:
    """Container for the full CausalProbe-2024 benchmark."""
    examples: List[CausalExample]
    version: str = "2024.1"
    cutoff_date: str = "2024-01-01"
    
    def filter_by_difficulty(self, difficulty: str) -> List[CausalExample]:
        return [e for e in self.examples if e.difficulty == difficulty]
    
    def get_easy(self) -> List[CausalExample]:
        return self.filter_by_difficulty("easy")
    
    def get_hard(self) -> List[CausalExample]:
        return self.filter_by_difficulty("hard")


# ==============================================================================
# SYNTHETIC DATASET GENERATOR (For testing before real data integration)
# ==============================================================================

def generate_synthetic_causalprobe(n_easy: int = 50, n_hard: int = 50) -> CausalProbeDataset:
    """
    Generate synthetic CausalProbe-style examples for testing.
    
    In production, replace with actual CausalProbe-2024 dataset loading.
    Real dataset available at: https://github.com/causalprobe/causalprobe-2024
    """
    examples = []
    
    # Easy examples: Single-hop causal reasoning
    easy_templates = [
        {
            "context": "In March 2024, the European Central Bank raised interest rates by 0.5%.",
            "question": "What was the likely effect on European mortgage rates?",
            "cause": "ECB interest rate increase",
            "effect": "Higher mortgage rates",
            "answer": "European mortgage rates increased",
            "reasoning": ["Central bank rates influence lending rates", "Higher base rates lead to higher borrowing costs"]
        },
        {
            "context": "A new AI regulation was passed in the EU in April 2024 requiring algorithmic transparency.",
            "question": "What effect did this have on AI companies operating in Europe?",
            "cause": "EU AI regulation",
            "effect": "Increased compliance costs",
            "answer": "AI companies faced increased compliance requirements",
            "reasoning": ["Regulations impose requirements", "Requirements increase operational costs"]
        },
        {
            "context": "In February 2024, a major undersea cable was damaged in the Red Sea.",
            "question": "What was the immediate effect on internet traffic?",
            "cause": "Undersea cable damage",
            "effect": "Internet latency increase",
            "answer": "Internet traffic was rerouted, increasing latency",
            "reasoning": ["Cable damage disrupts data flow", "Traffic must use alternative routes"]
        },
    ]
    
    # Hard examples: Multi-hop causal reasoning
    hard_templates = [
        {
            "context": "In May 2024, Country X implemented carbon tariffs on imports. Country Y, a major exporter of steel to X, saw its steel industry contract. This led to unemployment in Y's industrial regions, which then affected consumer spending.",
            "question": "What is the causal chain from carbon tariffs to consumer spending in Country Y?",
            "cause": "Carbon tariffs",
            "effect": "Reduced consumer spending",
            "answer": "Tariffs -> Steel industry contraction -> Unemployment -> Lower consumer spending",
            "reasoning": [
                "Carbon tariffs increase cost of carbon-intensive imports",
                "Steel is carbon-intensive, so tariffs hurt steel exports",
                "Reduced exports lead to industry contraction",
                "Industry contraction causes unemployment",
                "Unemployment reduces disposable income and spending"
            ]
        },
        {
            "context": "In June 2024, a drought in Southeast Asia reduced rice production. Global rice prices rose. Countries dependent on rice imports experienced food inflation. Central banks in those countries raised interest rates to combat inflation.",
            "question": "Trace the causal path from the drought to interest rate changes.",
            "cause": "Southeast Asian drought",
            "effect": "Interest rate increases in importing countries",
            "answer": "Drought -> Lower rice production -> Higher global prices -> Food inflation -> Central bank rate hikes",
            "reasoning": [
                "Drought reduces agricultural output",
                "Reduced supply increases prices globally",
                "Importing countries face higher food costs",
                "Food is a major component of consumer price index",
                "Central banks raise rates to control inflation"
            ]
        },
    ]
    
    # Generate easy examples
    for i in range(n_easy):
        template = easy_templates[i % len(easy_templates)]
        examples.append(CausalExample(
            id=f"easy_{i:03d}",
            context=template["context"],
            question=template["question"],
            cause=template["cause"],
            effect=template["effect"],
            answer=template["answer"],
            difficulty="easy",
            reasoning_chain=template["reasoning"],
            distractors=["No effect", "Opposite effect", "Unrelated outcome"],
            source_date="2024-Q1"
        ))
    
    # Generate hard examples
    for i in range(n_hard):
        template = hard_templates[i % len(hard_templates)]
        examples.append(CausalExample(
            id=f"hard_{i:03d}",
            context=template["context"],
            question=template["question"],
            cause=template["cause"],
            effect=template["effect"],
            answer=template["answer"],
            difficulty="hard",
            reasoning_chain=template["reasoning"],
            distractors=["Direct causation only", "Reverse causation", "No causal link"],
            source_date="2024-Q2"
        ))
    
    random.shuffle(examples)
    return CausalProbeDataset(examples=examples)


# ==============================================================================
# DATASET LOADER
# ==============================================================================

def load_causalprobe(path: Optional[str] = None) -> CausalProbeDataset:
    """
    Load CausalProbe-2024 dataset.
    
    Args:
        path: Path to JSON file. If None, uses synthetic data.
        
    Returns:
        CausalProbeDataset instance
    """
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        examples = [CausalExample(**ex) for ex in data['examples']]
        return CausalProbeDataset(
            examples=examples,
            version=data.get('version', '2024.1'),
            cutoff_date=data.get('cutoff_date', '2024-01-01')
        )
    else:
        print("[INFO] Using synthetic CausalProbe data for testing.")
        print("       For production, provide path to real CausalProbe-2024 JSON.")
        return generate_synthetic_causalprobe()


# ==============================================================================
# VERIFICATION
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CausalProbe-2024 Benchmark Loader")
    print("=" * 60)
    
    dataset = load_causalprobe()
    
    print(f"   Total examples: {len(dataset.examples)}")
    print(f"   Easy subset:    {len(dataset.get_easy())}")
    print(f"   Hard subset:    {len(dataset.get_hard())}")
    print(f"   Cutoff date:    {dataset.cutoff_date}")
    
    print("\n   Sample Easy Example:")
    easy = dataset.get_easy()[0]
    print(f"      Q: {easy.question}")
    print(f"      A: {easy.answer}")
    
    print("\n   Sample Hard Example:")
    hard = dataset.get_hard()[0]
    print(f"      Q: {hard.question}")
    print(f"      Chain: {' -> '.join(hard.reasoning_chain[:3])}...")
    
    print("=" * 60)
