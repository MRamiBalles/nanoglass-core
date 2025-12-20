import { useSettings } from "@/contexts/SettingsContext";

export function ScanlineOverlay() {
  const { scanlineEnabled } = useSettings();

  if (!scanlineEnabled) return null;

  return (
    <div className="fixed inset-0 pointer-events-none z-[100]">
      {/* Scanlines */}
      <div
        className="absolute inset-0"
        style={{
          background: `repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0, 0, 0, 0.03) 2px,
            rgba(0, 0, 0, 0.03) 4px
          )`,
        }}
      />

      {/* Moving scanline */}
      <div
        className="absolute left-0 right-0 h-[2px]"
        style={{
          background: "linear-gradient(90deg, transparent, hsl(var(--neon-cyan) / 0.1), transparent)",
          animation: "scanline-move 4s linear infinite",
        }}
      />

      {/* Vignette */}
      <div
        className="absolute inset-0"
        style={{
          background: "radial-gradient(ellipse at center, transparent 0%, rgba(0,0,0,0.3) 100%)",
        }}
      />

      <style>{`
        @keyframes scanline-move {
          0% { top: -2px; }
          100% { top: 100%; }
        }
      `}</style>
    </div>
  );
}
