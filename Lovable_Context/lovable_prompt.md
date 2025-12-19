# PROMPT MAESTRO PARA LOVABLE (PROJECT NANOGLASS)
### 🎨 Rol y Estilo Visual
Actúa como un **Diseñador UI/UX Senior de Interfaces Sci-Fi** y Desarrollador React experto.
Quiero que construyas el **"NanoGlass Dashboard"**, una interfaz futurista para monitorizar una IA experimental "Glass Box".
**Estética Clave:**
*   **Tema:** Ultra Dark Mode (Fondos `slate-950`).
*   **Estilo:** "Glassmorphism" científico (paneles translúcidos, bordes brillantes finos).
*   **Paleta de Colores:** Neon Cyan (`#22d3ee`) para verdad/energía, Neon Purple (`#c084fc`) para entropía/incertidumbre, y Green/Yellow/Red para estados de alerta.
*   **Tipografía:** Fuentes monoespaciadas para datos (JetBrains Mono o similar) y Sans-serif limpia para títulos.
---
### 🛠️ Especificaciones Técnicas
*   **Framework:** React + Vite + TypeScript.
*   **Styling:** Tailwind CSS (uso extensivo de gradientes y `backdrop-blur`).
*   **Iconos:** Lucide React (`Zap`, `Activity`, `Shield`, `Terminal`, `BookOpen`).
*   **Gráficos:** Recharts (para líneas de tiempo reales).
*   **Navegación:** `react-router-dom` (Sidebar fija a la izquierda).
---
### 📱 Estructura de la App
#### 1. Sidebar de Navegación ("Navigator")
*   Fija a la izquierda, estilo vidrio esmerilado.
*   Links: "Dashboard" (Home) y "Research Hub".
*   Indicador de estado abajo: "TruthRL Active" (simulando un "latido" verde animado).
#### 2. Página Principal: "Dashboard"
Debe parecer el panel de control de un reactor nuclear o laboratorio avanzado.
*   **Header:** Título "PROJECT NANOGLASS" con subtítulo "GLASS BOX INTERPRETER V1.0".
*   **Grid de Métricas (3 Tarjetas):**
    1.  **Current Energy:** Número grande (ej: 0.1542). Icono de Rayo Amarillo. Etiqueta "Minimizing (Optimal)".
    2.  **Entropy State:** Texto "Low" o "Stable". Icono de Actividad Púrpura.
    3.  **TruthRL Status:** Texto "Active". Icono de Escudo Verde.
*   **Gráfico Principal (Recharts):**
    *   Una `AreaChart` o `LineChart` que ocupe 2/3 del ancho.
    *   Línea Cyan (`Total Energy`) bajando con el tiempo (simulando aprendizaje).
    *   Línea Punteada Púrpura (`Entropy`) fluctuando.
    *   Ejes minimalistas, grid muy sutil (`stroke="#1e293b"`).
*   **Log Terminal (Derecha):**
    *   Un panel estilo consola (`bg-black/90`).
    *   Live logs simulados:
        *   `[10:42:01] Input: "What is 2+2?" -> Output: "4" (Low Energy)`
        *   `[10:42:05] Input: "Meaning of life?" -> [IDK] TOKEN TRIGGERED (Abstention)`
        *   `[10:42:15] HALLUCINATION BLOCKED by TruthRL.`
#### 3. Página Secundaria: "Research Hub"
Un grid de tarjetas bonitas para mostrar los papers del proyecto.
*   **Cada Tarjeta:** Título, Categoría (ej: "Thermodynamics", "Xenolinguistics"), Resumen breve, y un Badge de estado.
*   **Badges:**
    *   `VERIFIED` (Verde, con check): Para cosas probadas.
    *   `THEORY` (Amarillo, con reloj): Para hipótesis.
*   **Interacción:** Hover effects que iluminan el borde de la tarjeta con color Cyan.
---
### 🚀 Comportamiento deseado
*   Usa `framer-motion` para animaciones suaves de entrada.
*   Haz que el gráfico de energía se sienta "vivo" (puedes usar datos mockeados estáticos pero que parezcan reales).
*   Prioriza el **"Wow Factor"** visual. Quiero que parezca tecnología del año 2045.
