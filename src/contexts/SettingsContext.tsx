import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";

export type ThemeVariant = "cyber" | "matrix" | "plasma" | "void";
export type AnimationSpeed = "slow" | "normal" | "fast" | "insane";
export type ColorMode = "dark" | "light";

interface SettingsState {
  theme: ThemeVariant;
  colorMode: ColorMode;
  particlesEnabled: boolean;
  glitchEnabled: boolean;
  animationSpeed: AnimationSpeed;
  scanlineEnabled: boolean;
  neonIntensity: number;
  alertsEnabled: boolean;
}

interface SettingsContextType extends SettingsState {
  setTheme: (theme: ThemeVariant) => void;
  setColorMode: (mode: ColorMode) => void;
  toggleColorMode: () => void;
  setParticlesEnabled: (enabled: boolean) => void;
  setGlitchEnabled: (enabled: boolean) => void;
  setAnimationSpeed: (speed: AnimationSpeed) => void;
  setScanlineEnabled: (enabled: boolean) => void;
  setNeonIntensity: (intensity: number) => void;
  setAlertsEnabled: (enabled: boolean) => void;
  getSpeedMultiplier: () => number;
}

const STORAGE_KEY = "nanoglass_settings_v2";

const defaultSettings: SettingsState = {
  theme: "cyber",
  colorMode: "dark",
  particlesEnabled: true,
  glitchEnabled: true,
  animationSpeed: "normal",
  scanlineEnabled: true,
  neonIntensity: 70,
  alertsEnabled: true,
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
    
    // Apply color mode class to document
    const root = document.documentElement;
    root.classList.remove("dark", "light");
    root.classList.add(settings.colorMode);
    
    // Apply theme colors
    const themes: Record<ThemeVariant, { primary: string; secondary: string }> = {
      cyber: { primary: "187 92% 53%", secondary: "270 91% 75%" },
      matrix: { primary: "142 76% 45%", secondary: "142 50% 30%" },
      plasma: { primary: "320 80% 60%", secondary: "45 93% 47%" },
      void: { primary: "240 50% 60%", secondary: "0 0% 40%" },
    };
    
    // Adjust for light mode
    const themesLight: Record<ThemeVariant, { primary: string; secondary: string }> = {
      cyber: { primary: "187 92% 35%", secondary: "270 70% 50%" },
      matrix: { primary: "142 76% 36%", secondary: "142 50% 25%" },
      plasma: { primary: "320 80% 45%", secondary: "45 93% 40%" },
      void: { primary: "240 50% 45%", secondary: "0 0% 35%" },
    };
    
    const themeColors = settings.colorMode === "dark" ? themes : themesLight;
    const theme = themeColors[settings.theme];
    
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
    setColorMode: (colorMode) => setSettings((s) => ({ ...s, colorMode })),
    toggleColorMode: () => setSettings((s) => ({ ...s, colorMode: s.colorMode === "dark" ? "light" : "dark" })),
    setParticlesEnabled: (enabled) => setSettings((s) => ({ ...s, particlesEnabled: enabled })),
    setGlitchEnabled: (enabled) => setSettings((s) => ({ ...s, glitchEnabled: enabled })),
    setAnimationSpeed: (speed) => setSettings((s) => ({ ...s, animationSpeed: speed })),
    setScanlineEnabled: (enabled) => setSettings((s) => ({ ...s, scanlineEnabled: enabled })),
    setNeonIntensity: (intensity) => setSettings((s) => ({ ...s, neonIntensity: intensity })),
    setAlertsEnabled: (enabled) => setSettings((s) => ({ ...s, alertsEnabled: enabled })),
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
