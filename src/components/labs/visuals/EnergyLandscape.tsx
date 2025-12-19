import { useRef, useEffect } from "react";

interface EnergyLandscapeProps {
    temperature: number; // e.g., 0.1 - 2.0
    active: boolean;
}

export function EnergyLandscape({ temperature, active }: EnergyLandscapeProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        let time = 0;
        let animationFrameId: number;

        const draw = () => {
            if (!active) return;

            const width = canvas.width;
            const height = canvas.height;
            ctx.clearRect(0, 0, width, height);

            // Pseudo-3D Terrain Logic
            const rows = 20;
            const cols = 20;
            const spacingX = width / cols;
            const spacingY = height / 3 / rows; // flattened perspective

            const centerX = width / 2;
            const centerY = height / 2;

            ctx.strokeStyle = temperature > 1.5 ? "rgba(255, 50, 50, 0.6)" : "rgba(180, 50, 255, 0.6)"; // Red if hallucinating
            ctx.lineWidth = 1;

            for (let y = 0; y < rows; y++) {
                ctx.beginPath();
                for (let x = 0; x <= cols; x++) {
                    // Calculate 3D-ish coordinates
                    // Perspective push
                    const xPos = (x - cols / 2) * spacingX * (1 + y * 0.1) + centerX;
                    const yBase = centerY + (y - rows / 2) * 10;

                    // Height modulation (The "Energy")
                    // High Temp = fast moving, high amplitude waves
                    // Low Temp = slow moving, smooth valleys
                    const freq = 0.5;
                    const amp = temperature * 20;
                    const noise = Math.sin((x * freq) + time * temperature) * Math.cos((y * freq) + time * 0.5) * amp;

                    const yPos = yBase - noise;

                    if (x === 0) ctx.moveTo(xPos, yPos);
                    else ctx.lineTo(xPos, yPos);
                }
                ctx.stroke();
            }

            // Vertical lines for wireframe feel
            // (Simplified to reduce CPU load - horizontal only looks good enough for "Landscape")

            time += 0.05;
            animationFrameId = requestAnimationFrame(draw);
        };

        if (active) {
            draw();
        } else {
            // Draw one static frame
            // ... or simply clear
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = "rgba(255,255,255,0.05)";
            ctx.font = "10px monospace";
            ctx.fillText("SYSTEM OFFLINE", 10, 20);
        }

        return () => cancelAnimationFrame(animationFrameId);
    }, [temperature, active]);

    return (
        <div className="w-full h-64 bg-black/40 rounded-lg overflow-hidden relative border border-white/5">
            <canvas ref={canvasRef} width={400} height={300} className="w-full h-full" />
            <div className={`absolute bottom-2 left-2 text-[10px] font-mono px-2 py-1 rounded ${temperature > 1.5 ? "bg-neon-red/20 text-neon-red" : "bg-neon-purple/20 text-neon-purple"}`}>
                System Temp: {temperature.toFixed(1)} K
            </div>
        </div>
    );
}
