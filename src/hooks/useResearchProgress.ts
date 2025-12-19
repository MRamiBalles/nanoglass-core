import { useState, useEffect } from "react";
import { experiments } from "@/data/experiments";

// Key for localStorage
const STORAGE_KEY = "nanoglass_research_progress_v1";

interface ResearchState {
    completedIds: string[];
    level: number;
}

export function useResearchProgress() {
    const [state, setState] = useState<ResearchState>(() => {
        // Initialize from local storage
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
            try {
                return JSON.parse(saved);
            } catch (e) {
                console.error("Failed to parse research progress", e);
            }
        }
        return { completedIds: [], level: 1 };
    });

    // Persist on change
    useEffect(() => {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    }, [state]);

    const completeExperiment = (id: string) => {
        if (state.completedIds.includes(id)) return; // Already done

        const newCompleted = [...state.completedIds, id];

        // Level Up Logic
        // Level 1: Default
        // Level 2: 1 Experiment Done
        // Level 3: 3 Experiments Done
        // Level 4: 5 Experiments Done
        // Level 5: 8 Experiments Done (Unlocks Xeno)
        let newLevel = 1;
        const count = newCompleted.length;

        if (count >= 1) newLevel = 2;
        if (count >= 3) newLevel = 3;
        if (count >= 5) newLevel = 4;
        if (count >= 8) newLevel = 5;

        setState({
            completedIds: newCompleted,
            level: Math.max(state.level, newLevel)
        });
    };

    const resetProgress = () => {
        setState({ completedIds: [], level: 1 });
    };

    // Derived state
    const completedCount = state.completedIds.length;
    const totalBioPhys = experiments.filter(e => e.category !== "Xeno-Cognition").length; // ~8 experiments
    // Progress towards Level 5 (Unlock Xeno)
    const progressToXeno = Math.min((completedCount / 8) * 100, 100);

    return {
        completedIds: state.completedIds,
        level: state.level,
        completeExperiment,
        resetProgress,
        completedCount,
        progressToXeno
    };
}
