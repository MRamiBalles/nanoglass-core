import { useRef, useEffect } from "react";

interface QualiaFieldProps {
    dimensions: number; // e.g., 2 - 1024
    active: boolean;
}

export function QualiaField({ dimensions, active }: QualiaFieldProps) {
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

            // Clear with trail effect
            ctx.fillStyle = "rgba(0, 0, 0, 0.1)";
            ctx.fillRect(0, 0, width, height);

            // Qualia Orbs
            // Number of orbs related to log(dimensions)
            // Low dim = 1-2 boring blobs. High dim = Many complex interactive blobs.
            const numOrbs = Math.max(2, Math.min(10, Math.log2(dimensions)));

            for (let i = 0; i < numOrbs; i++) {
                const t = time * (0.5 + i * 0.1);

                // Position
                const x = width / 2 + Math.cos(t * 0.7) * (width * 0.3);
                const y = height / 2 + Math.sin(t * 1.1) * (height * 0.3);
                const r = 40 + Math.sin(t) * 20;

                // Color based on dimensions (Higher dim = more spectral range)
                const hue = (time * 20 + i * (360 / numOrbs)) % 360;
                const sat = dimensions < 10 ? "20%" : "80%";

                const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
                grad.addColorStop(0, `hsla(${hue}, ${sat}, 60%, 0.6)`);
                grad.addColorStop(1, "transparent"); // `hsla(${hue}, ${sat}, 20%, 0)`);

                ctx.beginPath();
                ctx.arc(x, y, r, 0, Math.PI * 2);
                ctx.fillStyle = grad;
                // Global composite operation for 'mixing' qualia
                ctx.globalCompositeOperation = "screen";
                ctx.fill();
            }

            // Reset composite for next frame clear
            ctx.globalCompositeOperation = "source-over";

            time += 0.03 + (dimensions > 500 ? 0.05 : 0); // Faster chaos at high dims
            animationFrameId = requestAnimationFrame(draw);
        };

        if (active) {
            draw();
        } else {
            ctx.fillStyle = "#000";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = "#333";
            ctx.font = "10px monospace";
            ctx.fillText("NO SIGNAL", 10, 20);
        }

        return () => cancelAnimationFrame(animationFrameId);
    }, [dimensions, active]);

    return (
        <div className="w-full h-64 bg-black rounded-lg overflow-hidden relative border border-neon-yellow/20">
            <canvas ref={canvasRef} width={400} height={300} className="w-full h-full" />
            <div className="absolute top-2 right-2 text-[10px] font-mono text-neon-yellow/80 bg-black/60 px-2 rounded">
                Dims: {dimensions} | Topo: {dimensions < 10 ? "Collapse" : "Stable"}
            </div>
        </div>
    );
}
