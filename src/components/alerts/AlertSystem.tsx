import { useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, CheckCircle, XCircle, Info, X, Bell } from "lucide-react";
import { create } from "zustand";

export type AlertType = "info" | "success" | "warning" | "error";

export interface Alert {
  id: string;
  type: AlertType;
  title: string;
  message: string;
  timestamp: number;
  dismissed: boolean;
}

interface AlertStore {
  alerts: Alert[];
  addAlert: (type: AlertType, title: string, message: string) => void;
  dismissAlert: (id: string) => void;
  clearAll: () => void;
}

export const useAlertStore = create<AlertStore>((set) => ({
  alerts: [],
  addAlert: (type, title, message) => {
    const id = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    set((state) => ({
      alerts: [
        ...state.alerts.slice(-9), // Keep max 10 alerts
        { id, type, title, message, timestamp: Date.now(), dismissed: false },
      ],
    }));
  },
  dismissAlert: (id) => {
    set((state) => ({
      alerts: state.alerts.map((a) => (a.id === id ? { ...a, dismissed: true } : a)),
    }));
    // Remove after animation
    setTimeout(() => {
      set((state) => ({
        alerts: state.alerts.filter((a) => a.id !== id),
      }));
    }, 300);
  },
  clearAll: () => set({ alerts: [] }),
}));

// Hook to trigger alerts from simulation
export function useSimulationAlerts(enabled: boolean) {
  const { addAlert } = useAlertStore();
  const lastAlertRef = useRef<string | null>(null);

  const triggerAlert = useCallback(
    (type: AlertType, title: string, message: string) => {
      if (!enabled) return;
      
      // Prevent duplicate alerts
      const key = `${type}-${title}`;
      if (lastAlertRef.current === key) return;
      lastAlertRef.current = key;
      
      addAlert(type, title, message);
      
      // Reset after 2 seconds to allow same alert again
      setTimeout(() => {
        if (lastAlertRef.current === key) {
          lastAlertRef.current = null;
        }
      }, 2000);
    },
    [enabled, addAlert]
  );

  return { triggerAlert };
}

const alertConfig: Record<AlertType, { icon: typeof Info; bgClass: string; borderClass: string; textClass: string }> = {
  info: {
    icon: Info,
    bgClass: "bg-neon-cyan/10",
    borderClass: "border-neon-cyan/30",
    textClass: "text-neon-cyan",
  },
  success: {
    icon: CheckCircle,
    bgClass: "bg-neon-green/10",
    borderClass: "border-neon-green/30",
    textClass: "text-neon-green",
  },
  warning: {
    icon: AlertTriangle,
    bgClass: "bg-neon-yellow/10",
    borderClass: "border-neon-yellow/30",
    textClass: "text-neon-yellow",
  },
  error: {
    icon: XCircle,
    bgClass: "bg-neon-red/10",
    borderClass: "border-neon-red/30",
    textClass: "text-neon-red",
  },
};

export function AlertSystem() {
  const { alerts, dismissAlert } = useAlertStore();
  const activeAlerts = alerts.filter((a) => !a.dismissed);

  // Auto-dismiss after 5 seconds
  useEffect(() => {
    activeAlerts.forEach((alert) => {
      const age = Date.now() - alert.timestamp;
      if (age < 5000) {
        const timeout = setTimeout(() => dismissAlert(alert.id), 5000 - age);
        return () => clearTimeout(timeout);
      }
    });
  }, [activeAlerts, dismissAlert]);

  return (
    <div className="fixed top-4 right-4 z-[200] flex flex-col gap-2 max-w-sm">
      <AnimatePresence mode="popLayout">
        {activeAlerts.map((alert) => {
          const config = alertConfig[alert.type];
          const Icon = config.icon;

          return (
            <motion.div
              key={alert.id}
              initial={{ opacity: 0, x: 100, scale: 0.9 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: 100, scale: 0.9 }}
              transition={{ type: "spring", damping: 20, stiffness: 300 }}
              className={`glass-panel p-4 rounded-lg border ${config.borderClass} ${config.bgClass} backdrop-blur-lg`}
            >
              <div className="flex items-start gap-3">
                <div className={`mt-0.5 ${config.textClass}`}>
                  <Icon className="w-5 h-5" />
                </div>
                <div className="flex-1 min-w-0">
                  <h4 className={`font-mono text-sm font-bold ${config.textClass}`}>
                    {alert.title}
                  </h4>
                  <p className="font-mono text-xs text-muted-foreground mt-1 leading-relaxed">
                    {alert.message}
                  </p>
                  <p className="font-mono text-[9px] text-muted-foreground/50 mt-2">
                    {new Date(alert.timestamp).toLocaleTimeString()}
                  </p>
                </div>
                <button
                  onClick={() => dismissAlert(alert.id)}
                  className="w-6 h-6 rounded flex items-center justify-center hover:bg-background/50 transition-colors"
                >
                  <X className="w-3 h-3 text-muted-foreground" />
                </button>
              </div>

              {/* Progress bar for auto-dismiss */}
              <motion.div
                initial={{ scaleX: 1 }}
                animate={{ scaleX: 0 }}
                transition={{ duration: 5, ease: "linear" }}
                className={`absolute bottom-0 left-0 right-0 h-0.5 origin-left ${config.textClass.replace("text-", "bg-")}`}
                style={{ opacity: 0.5 }}
              />
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}

// Indicator button to show alert count
export function AlertIndicator() {
  const { alerts, clearAll } = useAlertStore();
  const activeCount = alerts.filter((a) => !a.dismissed).length;

  if (activeCount === 0) return null;

  return (
    <motion.button
      initial={{ scale: 0 }}
      animate={{ scale: 1 }}
      exit={{ scale: 0 }}
      onClick={clearAll}
      className="fixed top-4 right-[340px] z-[200] w-10 h-10 rounded-full glass-panel flex items-center justify-center hover-glow"
      title="Clear all alerts"
    >
      <Bell className="w-4 h-4 text-neon-cyan" />
      <span className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-neon-red text-[10px] font-bold flex items-center justify-center text-background">
        {activeCount}
      </span>
    </motion.button>
  );
}
