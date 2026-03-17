import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { MainLayout } from "@/components/layout/MainLayout";
import { SettingsProvider } from "@/contexts/SettingsContext";
import Dashboard from "./pages/Dashboard";
import ResearchHub from "./pages/ResearchHub";
import TrainingPlayground from "./pages/TrainingPlayground";
import InferenceDemo from "./pages/InferenceDemo";
import ArchitectureVisualizer from "./pages/ArchitectureVisualizer";
import ResearchPipeline from "./pages/ResearchPipeline";
import TruthRLVisualizer from "./pages/TruthRLVisualizer";
import BioLab from "./pages/labs/BioLab";
import PhysicsLab from "./pages/labs/PhysicsLab";
import XenoLab from "./pages/labs/XenoLab";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <SettingsProvider>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route element={<MainLayout />}>
              <Route path="/" element={<Dashboard />} />
              <Route path="/research" element={<ResearchHub />} />
              <Route path="/training" element={<TrainingPlayground />} />
              <Route path="/inference" element={<InferenceDemo />} />
              <Route path="/architectures" element={<ArchitectureVisualizer />} />
              <Route path="/pipeline" element={<ResearchPipeline />} />
              <Route path="/labs/bio" element={<BioLab />} />
              <Route path="/labs/physics" element={<PhysicsLab />} />
              <Route path="/labs/xeno" element={<XenoLab />} />
            </Route>
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </SettingsProvider>
  </QueryClientProvider>
);

export default App;
