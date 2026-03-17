import { NavLink, useLocation } from "react-router-dom";
import { LayoutDashboard, BookOpen, Activity, Brain, Zap, Eye, Lock, ShieldCheck, FlaskConical } from "lucide-react";
import { motion } from "framer-motion";
import { useResearchProgress } from "@/hooks/useResearchProgress";

const navItems = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/research", label: "Research Hub", icon: BookOpen },
  { to: "/training", label: "Training Lab", icon: FlaskConical },
  { type: "divider", label: "EXPERIMENTAL LABS" },
  { to: "/labs/bio", label: "Bio-Mimetic Lab", icon: Brain },
  { to: "/labs/physics", label: "Physics Lab", icon: Zap },
  { to: "/labs/xeno", label: "Xeno Lab", icon: Eye, restricted: true, requiredLevel: 5 },
];

export function Sidebar() {
  const location = useLocation();
  const { level, progressToXeno } = useResearchProgress();

  return (
    <aside className="fixed left-0 top-0 h-screen w-64 glass-panel border-r border-border/50 flex flex-col z-50">
      {/* Logo/Brand */}
      <div className="p-6 border-b border-border/30">
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-3"
        >
          <div className="relative">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-neon-cyan/20 to-neon-purple/20 flex items-center justify-center border border-neon-cyan/30">
              <Activity className="w-5 h-5 text-neon-cyan" />
            </div>
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-neon-green rounded-full heartbeat" />
          </div>
          <div>
            <h1 className="font-bold text-foreground tracking-tight">NANOGLASS</h1>
            <p className="text-xs text-muted-foreground font-mono">NAVIGATOR</p>
          </div>
        </motion.div>

        {/* Level Badge */}
        <div className="mt-4 bg-background/30 p-2 rounded-lg border border-white/10 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ShieldCheck className={`w-4 h-4 ${level >= 5 ? "text-neon-yellow" : "text-muted-foreground"}`} />
            <div>
              <p className="text-[9px] font-mono text-muted-foreground uppercase">Clearance</p>
              <p className="text-xs font-bold font-mono text-foreground">LEVEL {level}</p>
            </div>
          </div>
          {level < 5 && (
            <div className="w-16 h-1 bg-black/50 rounded-full overflow-hidden">
              <div className="h-full bg-neon-cyan" style={{ width: `${progressToXeno}%` }} />
            </div>
          )}
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4">
        <ul className="space-y-2">
          {navItems.map((item, index) => {
            if (item.type === "divider") {
              return (
                <li key={index} className="pt-4 pb-2 px-4 text-[10px] font-mono font-bold text-muted-foreground/60 uppercase tracking-widest">
                  {item.label}
                </li>
              );
            }

            const isActive = location.pathname === item.to;
            const Icon = item.icon;
            // Check restriction
            const isLocked = item.restricted && level < (item.requiredLevel || 1);

            return (
              <motion.li
                key={item.to}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                {isLocked ? (
                  <div className="relative flex items-center gap-3 px-4 py-3 rounded-lg font-mono text-sm text-muted-foreground/40 cursor-not-allowed border border-transparent bg-background/20 opacity-60">
                    {Icon && <Icon className="w-4 h-4" />}
                    <span>{item.label}</span>
                    <Lock className="w-3 h-3 ml-auto text-neon-red/50" />
                  </div>
                ) : (
                  <NavLink
                    to={item.to!}
                    className={`
                    relative flex items-center gap-3 px-4 py-3 rounded-lg font-mono text-sm
                    transition-all duration-300 group
                    ${isActive
                        ? "bg-neon-cyan/10 text-neon-cyan border border-neon-cyan/30"
                        : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
                      }
                  `}
                  >
                    {isActive && (
                      <motion.div
                        layoutId="activeNav"
                        className="absolute inset-0 rounded-lg bg-neon-cyan/5 border border-neon-cyan/20"
                        transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                      />
                    )}
                    {Icon && <Icon className={`w-4 h-4 relative z-10 ${isActive ? "text-neon-cyan" : "group-hover:text-neon-cyan/70"}`} />}
                    <span className="relative z-10">{item.label}</span>
                    {isActive && (
                      <div className="absolute right-3 w-1.5 h-1.5 rounded-full bg-neon-cyan shadow-[0_0_8px_hsl(var(--neon-cyan))]" />
                    )}
                  </NavLink>
                )}
              </motion.li>
            );
          })}
        </ul>
      </nav>

      {/* Status Indicator */}
      <div className="p-4 border-t border-border/30">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="glass-panel p-4 rounded-lg"
        >
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-3 h-3 rounded-full bg-neon-green heartbeat" />
              <div className="absolute inset-0 w-3 h-3 rounded-full bg-neon-green/50 pulse-ring" />
            </div>
            <div>
              <p className="text-xs font-mono neon-green">TruthRL ACTIVE</p>
              <p className="text-[10px] text-muted-foreground font-mono">Monitoring...</p>
            </div>
          </div>
        </motion.div>
      </div>
    </aside>
  );
}
