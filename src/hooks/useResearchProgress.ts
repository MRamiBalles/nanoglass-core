import { useState, useEffect } from "react";
import { z } from "zod";
import { experiments } from "@/data/experiments";

// Key for localStorage
const STORAGE_KEY = "nanoglass_research_progress_v1";

// Zod schema for runtime validation of localStorage data
const ResearchStateSchema = z.object({
    completedIds: z.array(z.string()),
    level: z.number().int().min(1).max(5)
});

type ResearchState = z.infer<typeof ResearchStateSchema>;

const DEFAULT_STATE: ResearchState = { completedIds: [], level: 1 };

export function useResearchProgress() {
    const [state, setState] = useState<ResearchState>(() => {
        // Initialize from local storage with schema validation
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
            try {
                const parsed = JSON.parse(saved);
                const validated = ResearchStateSchema.safeParse(parsed);
                if (validated.success) {
                    return validated.data;
                }
                // Invalid schema - return default
            } catch {
                // JSON parse failed - return default
            }
        }
        return DEFAULT_STATE;
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
