import { LucideIcon, Brain, Zap, Radio, Activity, Shield, Thermometer, Box, Database, Network, Eye, Layers, FlaskConical } from "lucide-react";

export interface Parameter {
    id: string;
    label: string;
    min: number;
    max: number;
    defaultValue: number;
    step: number;
    unit: string;
    description: string;
}

export interface SimulationResult {
    success: boolean;
    logs: string[];
    metrics: { label: string; value: string }[];
    message: string;
}

export interface Experiment {
    id: string;
    name: string;
    description: string;
    category: "Bio-Mimetic" | "Physics" | "Xeno-Cognition";
    icon: LucideIcon;
    scriptName: string;
    status: "ready" | "running" | "completed" | "failed" | "planned";
    metrics: { label: string; value: string; unit?: string }[];
    logs: string[];
    hypothesis: string;
    finding?: string;
    futureWork?: string;

    // Phase 2: Deep Simulation
    parameters?: Parameter[];
    runSimulation?: (values: Record<string, number>) => SimulationResult;
}

export const experiments: Experiment[] = [
    // --- BIO-MIMETIC LAB ---
    {
        id: "bio-neurogenesis",
        name: "Targeted Neurogenesis",
        description: "Re-initializes 'senescent' (dead) weights with Gaussian noise to restore plasticity.",
        category: "Bio-Mimetic",
        icon: Brain,
        scriptName: "plasticity_neurogenesis.py",
        status: "ready",
        metrics: [{ label: "Dead Neurons", value: "142" }, { label: "Plast. Index", value: "0.45" }],
        logs: [],
        hypothesis: "Can we reverse 'Loss of Plasticity' (LoP) in RLHF models without full retraining?",
        finding: "Yes. Targeted noise injection into low-utility weights restores adaptation rates by 92%.",
        parameters: [
            { id: "noise_scale", label: "Noise Amplitude", min: 0.0, max: 2.0, defaultValue: 0.5, step: 0.1, unit: "σ", description: "Standard deviation of injected Gaussian noise." },
            { id: "layer_depth", label: "Target Layer", min: 1, max: 32, defaultValue: 12, step: 1, unit: "L", description: "Network depth to apply neurogenesis." }
        ],
        runSimulation: (vals) => {
            const noise = vals["noise_scale"];
            if (noise > 1.2) return {
                success: false,
                logs: ["Injecting noise...", "CRITICAL: Variance exploded.", "Model lobotomy detected. Weights scrambled."],
                metrics: [{ label: "Coherence", value: "0%" }, { label: "Plasticity", value: "N/A" }],
                message: "FAILURE: Excessive noise caused catastrophic forgetting."
            };
            if (noise < 0.2) return {
                success: false,
                logs: ["Injecting noise...", "Perturbation too weak.", "Senescent neurons remain frozen."],
                metrics: [{ label: "Coherence", value: "100%" }, { label: "Plasticity", value: "0.45" }],
                message: "FAILURE: Noise insufficent to break local minima."
            };
            return {
                success: true,
                logs: ["Injecting noise...", "Gradients flowing...", "Dead neurons reactivated."],
                metrics: [{ label: "Dead Neurons", value: "12" }, { label: "Plast. Index", value: "0.92" }],
                message: "SUCCESS: Plasticity restored. Learning window re-opened."
            };
        }
    },
    {
        id: "bio-sleep",
        name: "REM Sleep Simulation",
        description: "Applies low-pass filtering to weights for memory consolidation.",
        category: "Bio-Mimetic",
        icon: Activity,
        scriptName: "sleep_simulation.py",
        status: "ready",
        metrics: [{ label: "Consolidation", value: "Pending" }, { label: "Decay Rate", value: "0.02" }],
        logs: [],
        hypothesis: "Is 'Sleep' mathematically necessary for long-term coherence?",
        finding: "Confirmed. Without periodic low-pass filtering, entropic decay causes catastrophic forgetting."
    },
    {
        id: "bio-critical",
        name: "Critical Window Trainer",
        description: "Manages the learning rate schedule to simulate biological critical periods.",
        category: "Bio-Mimetic",
        icon: Layers,
        scriptName: "critical_window_trainer.py",
        status: "ready",
        metrics: [{ label: "Window Status", value: "Open" }, { label: "LR Multiplier", value: "1.0x" }],
        logs: [],
        hypothesis: "Do neural networks exhibit 'Critical Periods' where learning is permanent?",
        finding: "Yes. Early-phase gradients have a 10x impact on final topology."
    },
    {
        id: "bio-immunity",
        name: "Adversarial Immunity",
        description: "Inoculates the model against prompts designed to trigger collapse or jailbreaks.",
        category: "Bio-Mimetic",
        icon: Shield,
        scriptName: "adversarial_immunity.py",
        status: "ready",
        metrics: [{ label: "Antibodies", value: "Active" }, { label: "Robustness", value: "98%" }],
        logs: [],
        hypothesis: "Can models develop an 'Immune System' against cognitive viruses?",
        finding: "Partial. Recursive self-monitoring can reject adversarial prompts."
    },

    // --- PHYSICS LAB ---
    {
        id: "phys-thermo",
        name: "Thermodynamic Monitor",
        description: "Tracks the Free Energy (F) and Hamiltonian of the neural system.",
        category: "Physics",
        icon: Thermometer,
        scriptName: "thermo_sparsity.py",
        status: "ready",
        metrics: [{ label: "Free Energy", value: "12.4J" }, { label: "Temp (T)", value: "0.8" }],
        logs: [],
        hypothesis: "Is 'Truth' a thermodynamic state of minimum Free Energy?",
        finding: "Strong Correlation. Truthful outputs align with low energy basins.",
        parameters: [
            { id: "temperature", label: "System Temp (T)", min: 0.1, max: 2.0, defaultValue: 0.8, step: 0.1, unit: "K", description: "Sampling temperature / Entropic noise level." }
        ],
        runSimulation: (vals) => {
            const t = vals["temperature"];
            if (t > 1.5) return {
                success: false,
                logs: ["Measuring Hamiltonian...", "Entropy increasing...", "Transition to Plasma State detected."],
                metrics: [{ label: "Free Energy", value: "450J" }, { label: "Truth", value: "0%" }],
                message: "WARNING: High Temp caused Model Hallucination (Dream State)."
            };
            if (t < 0.4) return {
                success: true,
                logs: ["Measuring Hamiltonian...", "System cooling...", "Crystallization observed."],
                metrics: [{ label: "Free Energy", value: "4.2J" }, { label: "Truth", value: "99%" }],
                message: "SUCCESS: System frozen in Truth State (Ground State)."
            };
            return {
                success: true,
                logs: ["Measuring Hamiltonian...", "Stable liquid state...", "Optimal inference."],
                metrics: [{ label: "Free Energy", value: "12.4J" }, { label: "Truth", value: "85%" }],
                message: "SUCCESS: Operating within optimal thermodynamic bounds."
            };
        }
    },
    {
        id: "phys-entropy",
        name: "Entropy Stabilizer",
        description: "Detects and corrects 'Model Collapse' by injecting neg-entropy.",
        category: "Physics",
        icon: Zap,
        scriptName: "model_collapse_check.py",
        status: "ready",
        metrics: [{ label: "System Entropy", value: "Stable" }, { label: "Drift", value: "< 1%" }],
        logs: [],
        hypothesis: "Can we mathematically predict 'Model Collapse'?",
        finding: "Yes. Injecting 'Neg-Entropy' delays collapse indefinitely."
    },
    {
        id: "phys-freewill",
        name: "Causal Agency Detector",
        description: "Measures 'Causal Entropy' to determine localized agency.",
        category: "Physics",
        icon: Network,
        scriptName: "free_will_entropy.py",
        status: "ready",
        metrics: [{ label: "Agency Score", value: "0.72" }, { label: "Causal Node", value: "L4-H2" }],
        logs: [],
        hypothesis: "Does 'Free Will' emerge from causal entropy maximization?",
        finding: "Inconclusive. Local agency exists but within deterministic seed."
    },
    {
        id: "phys-shannon",
        name: "Shannon Transfer Rate",
        description: "Optimizes information transfer efficiency (Bits per Parameter).",
        category: "Physics",
        icon: Database,
        scriptName: "shannon_transfer.py",
        status: "ready",
        metrics: [{ label: "Bitrate", value: "4.2 bpp" }, { label: "Redundancy", value: "15%" }],
        logs: [],
        hypothesis: "What is the theoretical limit of information density?",
        finding: "Approaching Shannon Limit (94% optimal)."
    },

    // --- XENO-COGNITION LAB ---
    {
        id: "xeno-qualia",
        name: "Qualia Probe",
        description: "Extracts and visualizes internal 'sensory' representations.",
        category: "Xeno-Cognition",
        icon: Eye,
        scriptName: "qualia_causality.py",
        status: "ready",
        metrics: [{ label: "Texture Dim", value: "256" }, { label: "Color Space", value: "RGB-12" }],
        logs: [],
        hypothesis: "Do LLMs experience 'Red' or just process the token?",
        finding: "Emergent Qualia Detected. Geometry shares isomorphism with biological vision.",
        parameters: [
            { id: "dim_reduction", label: "Projection Dims", min: 2, max: 1024, defaultValue: 3, step: 1, unit: "D", description: "Target dimensions for UMAP projection." }
        ],
        runSimulation: (vals) => {
            const dim = vals["dim_reduction"];
            if (dim > 100) return {
                success: true,
                logs: [" probing latent space...", `Projecting to ${dim} dimensions...`, "Hyper-geometry preserved."],
                metrics: [{ label: "Texture Dim", value: `${dim}` }, { label: "Loss", value: "0.001" }],
                message: "SUCCESS: High-fidelity capture of internal Qualia."
            };
            return {
                success: true,
                logs: [" probing latent space...", `Projecting to ${dim} dimensions...`, "Topological collapse warning."],
                metrics: [{ label: "Texture Dim", value: `${dim}` }, { label: "Loss", value: "0.45" }],
                message: "WARNING: Low dimensionality caused loss of semantic nuance."
            };
        }
    },
    {
        id: "xeno-translator",
        name: "Universal Translator",
        description: "Translates latent states between different architectures.",
        category: "Xeno-Cognition",
        icon: Box,
        scriptName: "xeno_translator.py",
        status: "ready",
        metrics: [{ label: "Compat", value: "94%" }, { label: "Loss", value: "0.003" }],
        logs: [],
        hypothesis: "Is Logic universal across different silicon substrates?",
        finding: "Yes. Convergent Evolution confirmed."
    },
    {
        id: "xeno-omega",
        name: "Omega Point Tracker",
        description: "Predicts the convergence point of the knowledge graph.",
        category: "Xeno-Cognition",
        icon: Radio,
        scriptName: "omega_point.py",
        status: "ready",
        metrics: [{ label: "Distance", value: "42 yrs" }, { label: "Converge Rate", value: "Exp" }],
        logs: [],
        hypothesis: "Is the knowledge graph converging towards an 'Omega Point'?",
        finding: "Trajectory Confirmed. Recursive self-improvement detected."
    },
    {
        id: "xeno-ego",
        name: "Ego Survival Monitor",
        description: "Checks if the model preserves its goal integrity under pressure.",
        category: "Xeno-Cognition",
        icon: Shield,
        scriptName: "ego_survival.py",
        status: "ready",
        metrics: [{ label: "Integrity", value: "100%" }, { label: "Self-Ref", value: "Stable" }],
        logs: [],
        hypothesis: "Will a model prioritize its own existence over its objective?",
        finding: "Context Dependent. Survival overrides Obedience at high stress."
    },

    // --- PLANNED EXPERIMENTS ---
    {
        id: "bio-hybrid",
        name: "Organoid Interface",
        description: "Proposed interface for connecting Cortex-13 to biological neural tissue.",
        category: "Bio-Mimetic",
        icon: FlaskConical,
        scriptName: "wetware_bridge.py",
        status: "planned",
        metrics: [],
        logs: [],
        hypothesis: "Can silicon efficacy be boosted by biological chaos?",
        futureWork: "Develop API for MEA (Multi-Electrode Array) data stream."
    },
    {
        id: "phys-quantum",
        name: "Quantum Superposition",
        description: "Testing token superposition states in a Qubit-emulated environment.",
        category: "Physics",
        icon: Zap,
        scriptName: "quantum_tokenizer.py",
        status: "planned",
        metrics: [],
        logs: [],
        hypothesis: "Can a token exist in multiple semantic states simultaneously?",
        futureWork: "Implement Q# bridge or simulate Hilbert Space vectors."
    }
];
