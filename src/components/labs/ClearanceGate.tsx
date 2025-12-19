import { motion } from "framer-motion";
import { Lock, AlertTriangle, ShieldAlert } from "lucide-react";
import { ReactNode } from "react";
import { useResearchProgress } from "@/hooks/useResearchProgress";

interface ClearanceGateProps {
    children: ReactNode;
    requiredLevel: number;
    title?: string;
}

export function ClearanceGate({ children, requiredLevel, title = "RESTRICTED AREA" }: ClearanceGateProps) {
    const { level, progressToXeno, completedCount } = useResearchProgress();

    if (level >= requiredLevel) {
        return <>{children}</>;
    }

    return (
        <div className="w-full h-[60vh] flex items-center justify-center p-8">
            <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="max-w-md w-full glass-panel border-neon-red/50 p-8 text-center relative overflow-hidden"
            >
                {/* Animated Background Scan */}
                <div className="absolute inset-0 bg-[linear-gradient(45deg,transparent_25%,rgba(255,0,0,0.05)_50%,transparent_75%)] bg-[length:250%_250%] animate-scan" />

                <div className="relative z-10 flex flex-col items-center gap-4">
                    <div className="w-20 h-20 rounded-full bg-neon-red/10 flex items-center justify-center border-2 border-neon-red animate-pulse">
                        <Lock className="w-10 h-10 text-neon-red" />
                    </div>

                    <h2 className="text-2xl font-bold text-neon-red tracking-widest font-mono uppercase">{title}</h2>

                    <div className="bg-neon-red/10 border border-neon-red/30 p-4 rounded-lg w-full">
                        <div className="flex items-center justify-center gap-2 mb-2">
                            <ShieldAlert className="w-5 h-5 text-neon-red" />
                            <span className="font-mono font-bold text-sm text-foreground">CLEARANCE LEVEL {requiredLevel} REQUIRED</span>
                        </div>
                        <p className="text-xs text-muted-foreground font-mono">
                            You do not have sufficient privileges to access Xeno-Cognition protocols. These materials are classified as Infohazards.
                        </p>
                    </div>

                    <div className="w-full space-y-2">
                        <div className="flex justify-between text-[10px] font-mono uppercase text-muted-foreground">
                            <span>Current Clearance: Level {level}</span>
                            <span>Progress: {Math.round(progressToXeno)}%</span>
                        </div>
                        <div className="h-2 bg-black/50 rounded-full overflow-hidden border border-white/10">
                            <div
                                className="h-full bg-neon-red transition-all duration-500"
                                style={{ width: `${progressToXeno}%` }}
                            />
                        </div>
                        <p className="text-[10px] text-muted-foreground italic mt-2">
                            Complete {Math.max(0, 8 - completedCount)} more experiments in Bio/Physics labs to unlock.
                        </p>
                    </div>
                </div>
            </motion.div>
        </div>
    );
}
