import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Settings, X, Palette, Zap, Sparkles, Monitor, SlidersHorizontal } from "lucide-react";
import { useSettings, ThemeVariant, AnimationSpeed } from "@/contexts/SettingsContext";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";

const themes: { id: ThemeVariant; name: string; colors: string[] }[] = [
  { id: "cyber", name: "Cyber", colors: ["#22d3ee", "#c084fc"] },
  { id: "matrix", name: "Matrix", colors: ["#22c55e", "#16a34a"] },
  { id: "plasma", name: "Plasma", colors: ["#ec4899", "#eab308"] },
  { id: "void", name: "Void", colors: ["#818cf8", "#6b7280"] },
];

const speeds: { id: AnimationSpeed; name: string; icon: string }[] = [
  { id: "slow", name: "Slow", icon: "🐢" },
  { id: "normal", name: "Normal", icon: "⚡" },
  { id: "fast", name: "Fast", icon: "🚀" },
  { id: "insane", name: "Insane", icon: "💀" },
];

export function SettingsPanel() {
  const [isOpen, setIsOpen] = useState(false);
  const settings = useSettings();

  return (
    <>
      {/* Toggle Button */}
      <motion.button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 z-50 w-12 h-12 rounded-full glass-panel flex items-center justify-center hover-glow group"
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
      >
        <Settings className="w-5 h-5 text-neon-cyan group-hover:animate-spin" style={{ animationDuration: "2s" }} />
      </motion.button>

      {/* Panel */}
      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsOpen(false)}
              className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50"
            />

            {/* Panel */}
            <motion.div
              initial={{ x: "100%", opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: "100%", opacity: 0 }}
              transition={{ type: "spring", damping: 25, stiffness: 200 }}
              className="fixed right-0 top-0 h-full w-96 glass-panel border-l border-border/50 z-50 overflow-y-auto"
            >
              {/* Header */}
              <div className="p-6 border-b border-border/30 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <SlidersHorizontal className="w-5 h-5 text-neon-cyan" />
                  <h2 className="text-lg font-bold text-foreground">Settings</h2>
                </div>
                <button
                  onClick={() => setIsOpen(false)}
                  className="w-8 h-8 rounded-lg bg-background/50 flex items-center justify-center hover:bg-background transition-colors"
                >
                  <X className="w-4 h-4 text-muted-foreground" />
                </button>
              </div>

              <div className="p-6 space-y-8">
                {/* Theme Selection */}
                <section>
                  <div className="flex items-center gap-2 mb-4">
                    <Palette className="w-4 h-4 text-neon-purple" />
                    <h3 className="font-mono text-sm text-foreground">COLOR THEME</h3>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    {themes.map((t) => (
                      <motion.button
                        key={t.id}
                        onClick={() => settings.setTheme(t.id)}
                        className={`p-3 rounded-lg border transition-all ${
                          settings.theme === t.id
                            ? "border-neon-cyan bg-neon-cyan/10"
                            : "border-border/50 bg-background/30 hover:border-border"
                        }`}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <div className="flex items-center gap-2 mb-2">
                          <div
                            className="w-4 h-4 rounded-full"
                            style={{ background: `linear-gradient(135deg, ${t.colors[0]}, ${t.colors[1]})` }}
                          />
                          <span className="font-mono text-xs text-foreground">{t.name}</span>
                        </div>
                        <div className="flex gap-1">
                          {t.colors.map((c, i) => (
                            <div key={i} className="flex-1 h-1 rounded" style={{ background: c }} />
                          ))}
                        </div>
                      </motion.button>
                    ))}
                  </div>
                </section>

                {/* Animation Speed */}
                <section>
                  <div className="flex items-center gap-2 mb-4">
                    <Zap className="w-4 h-4 text-neon-yellow" />
                    <h3 className="font-mono text-sm text-foreground">ANIMATION SPEED</h3>
                  </div>
                  <div className="grid grid-cols-4 gap-2">
                    {speeds.map((s) => (
                      <motion.button
                        key={s.id}
                        onClick={() => settings.setAnimationSpeed(s.id)}
                        className={`p-2 rounded-lg border text-center transition-all ${
                          settings.animationSpeed === s.id
                            ? "border-neon-cyan bg-neon-cyan/10"
                            : "border-border/50 bg-background/30 hover:border-border"
                        }`}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                      >
                        <div className="text-lg mb-1">{s.icon}</div>
                        <span className="font-mono text-[10px] text-muted-foreground">{s.name}</span>
                      </motion.button>
                    ))}
                  </div>
                </section>

                {/* Neon Intensity */}
                <section>
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-neon-cyan" />
                      <h3 className="font-mono text-sm text-foreground">NEON INTENSITY</h3>
                    </div>
                    <span className="font-mono text-xs text-neon-cyan">{settings.neonIntensity}%</span>
                  </div>
                  <Slider
                    value={[settings.neonIntensity]}
                    onValueChange={([v]) => settings.setNeonIntensity(v)}
                    min={0}
                    max={100}
                    step={5}
                    className="w-full"
                  />
                </section>

                {/* Effect Toggles */}
                <section>
                  <div className="flex items-center gap-2 mb-4">
                    <Monitor className="w-4 h-4 text-neon-green" />
                    <h3 className="font-mono text-sm text-foreground">EFFECTS</h3>
                  </div>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-3 rounded-lg bg-background/30 border border-border/30">
                      <div>
                        <p className="font-mono text-sm text-foreground">Particles</p>
                        <p className="font-mono text-[10px] text-muted-foreground">Floating particle network</p>
                      </div>
                      <Switch
                        checked={settings.particlesEnabled}
                        onCheckedChange={settings.setParticlesEnabled}
                      />
                    </div>
                    <div className="flex items-center justify-between p-3 rounded-lg bg-background/30 border border-border/30">
                      <div>
                        <p className="font-mono text-sm text-foreground">Glitch Effects</p>
                        <p className="font-mono text-[10px] text-muted-foreground">Random text corruption</p>
                      </div>
                      <Switch
                        checked={settings.glitchEnabled}
                        onCheckedChange={settings.setGlitchEnabled}
                      />
                    </div>
                    <div className="flex items-center justify-between p-3 rounded-lg bg-background/30 border border-border/30">
                      <div>
                        <p className="font-mono text-sm text-foreground">Scanlines</p>
                        <p className="font-mono text-[10px] text-muted-foreground">CRT monitor effect</p>
                      </div>
                      <Switch
                        checked={settings.scanlineEnabled}
                        onCheckedChange={settings.setScanlineEnabled}
                      />
                    </div>
                  </div>
                </section>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
