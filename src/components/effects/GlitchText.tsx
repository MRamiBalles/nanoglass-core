import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { useSettings } from "@/contexts/SettingsContext";

interface GlitchTextProps {
  text: string;
  className?: string;
  as?: "h1" | "h2" | "h3" | "span" | "p";
  glitchColor1?: string;
  glitchColor2?: string;
}

export function GlitchText({
  text,
  className = "",
  as: Component = "span",
  glitchColor1 = "hsl(var(--neon-cyan))",
  glitchColor2 = "hsl(var(--neon-purple))",
}: GlitchTextProps) {
  const { glitchEnabled, getSpeedMultiplier } = useSettings();
  const [isGlitching, setIsGlitching] = useState(false);
  const [glitchText, setGlitchText] = useState(text);

  useEffect(() => {
    if (!glitchEnabled) return;

    const speedMultiplier = getSpeedMultiplier();
    const glitchChars = "!@#$%^&*()_+-=[]{}|;':\",./<>?0123456789";

    const triggerGlitch = () => {
      setIsGlitching(true);

      // Corrupt text temporarily
      let iterations = 0;
      const maxIterations = 5;

      const glitchInterval = setInterval(() => {
        setGlitchText(
          text
            .split("")
            .map((char, i) => {
              if (char === " ") return char;
              if (Math.random() < 0.3) {
                return glitchChars[Math.floor(Math.random() * glitchChars.length)];
              }
              return char;
            })
            .join("")
        );

        iterations++;
        if (iterations >= maxIterations) {
          clearInterval(glitchInterval);
          setGlitchText(text);
          setIsGlitching(false);
        }
      }, 50 * speedMultiplier);
    };

    // Random glitch trigger
    const scheduleGlitch = () => {
      const delay = (3000 + Math.random() * 5000) * speedMultiplier;
      return setTimeout(() => {
        triggerGlitch();
        scheduleGlitch();
      }, delay);
    };

    const timeoutId = scheduleGlitch();

    return () => clearTimeout(timeoutId);
  }, [text, glitchEnabled, getSpeedMultiplier]);

  if (!glitchEnabled) {
    return <Component className={className}>{text}</Component>;
  }

  return (
    <Component className={`relative inline-block ${className}`}>
      {/* Main text */}
      <span className="relative z-10">{glitchText}</span>

      {/* Glitch layers */}
      {isGlitching && (
        <>
          <motion.span
            initial={{ x: 0 }}
            animate={{ x: [-2, 2, -1, 1, 0] }}
            transition={{ duration: 0.1, repeat: 3 }}
            className="absolute inset-0 z-0"
            style={{
              color: glitchColor1,
              clipPath: "polygon(0 0, 100% 0, 100% 45%, 0 45%)",
              transform: "translateX(-2px)",
            }}
          >
            {glitchText}
          </motion.span>
          <motion.span
            initial={{ x: 0 }}
            animate={{ x: [2, -2, 1, -1, 0] }}
            transition={{ duration: 0.1, repeat: 3 }}
            className="absolute inset-0 z-0"
            style={{
              color: glitchColor2,
              clipPath: "polygon(0 55%, 100% 55%, 100% 100%, 0 100%)",
              transform: "translateX(2px)",
            }}
          >
            {glitchText}
          </motion.span>
        </>
      )}
    </Component>
  );
}
