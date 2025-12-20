import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";

export type ThemeVariant = "cyber" | "matrix" | "plasma" | "void";
export type AnimationSpeed = "slow" | "normal" | "fast" | "insane";

interface SettingsState {
  theme: ThemeVariant;
  particlesEnabled: boolean;
  glitchEnabled: boolean;
  animationSpeed: AnimationSpeed;
  scanlineEnabled: boolean;
  neonIntensity: number; // 0-100
}

interface SettingsContextType extends SettingsState {
  setTheme: (theme: ThemeVariant) => void;
  setParticlesEnabled: (enabled: boolean) => void;
  setGlitchEnabled: (enabled: boolean) => void;
  setAnimationSpeed: (speed: AnimationSpeed) => void;
  setScanlineEnabled: (enabled: boolean) => void;
  setNeonIntensity: (intensity: number) => void;
  getSpeedMultiplier: () => number;
}

const STORAGE_KEY = "nanoglass_settings_v1";

const defaultSettings: SettingsState = {
  theme: "cyber",
  particlesEnabled: true,
  glitchEnabled: true,
  animationSpeed: "normal",
  scanlineEnabled: true,
  neonIntensity: 70,
};

const SettingsContext = createContext<SettingsContextType | undefined>(undefined);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<SettingsState>(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        return { ...defaultSettings, ...JSON.parse(saved) };
      } catch {
        return defaultSettings;
      }
    }
    return defaultSettings;
  });

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    
    // Apply theme to CSS variables
    const root = document.documentElement;
    const themes: Record<ThemeVariant, { primary: string; secondary: string }> = {
      cyber: { primary: "187 92% 53%", secondary: "270 91% 75%" },
      matrix: { primary: "142 76% 45%", secondary: "142 50% 30%" },
      plasma: { primary: "320 80% 60%", secondary: "45 93% 47%" },
      void: { primary: "240 50% 60%", secondary: "0 0% 40%" },
    };
    
    const theme = themes[settings.theme];
    root.style.setProperty("--neon-cyan", theme.primary);
    root.style.setProperty("--primary", theme.primary);
    root.style.setProperty("--neon-purple", theme.secondary);
    root.style.setProperty("--secondary", theme.secondary);
    
    // Apply neon intensity
    root.style.setProperty("--neon-intensity", `${settings.neonIntensity / 100}`);
  }, [settings]);

  const getSpeedMultiplier = () => {
    const speeds: Record<AnimationSpeed, number> = {
      slow: 2,
      normal: 1,
      fast: 0.5,
      insane: 0.25,
    };
    return speeds[settings.animationSpeed];
  };

  const value: SettingsContextType = {
    ...settings,
    setTheme: (theme) => setSettings((s) => ({ ...s, theme })),
    setParticlesEnabled: (enabled) => setSettings((s) => ({ ...s, particlesEnabled: enabled })),
    setGlitchEnabled: (enabled) => setSettings((s) => ({ ...s, glitchEnabled: enabled })),
    setAnimationSpeed: (speed) => setSettings((s) => ({ ...s, animationSpeed: speed })),
    setScanlineEnabled: (enabled) => setSettings((s) => ({ ...s, scanlineEnabled: enabled })),
    setNeonIntensity: (intensity) => setSettings((s) => ({ ...s, neonIntensity: intensity })),
    getSpeedMultiplier,
  };

  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

export function useSettings() {
  const context = useContext(SettingsContext);
  if (!context) {
    throw new Error("useSettings must be used within a SettingsProvider");
  }
  return context;
}
