import { Outlet } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { ParticleField } from "@/components/effects/ParticleField";
import { ScanlineOverlay } from "@/components/effects/ScanlineOverlay";
import { SettingsPanel } from "@/components/settings/SettingsPanel";
import { AlertSystem } from "@/components/alerts/AlertSystem";

export function MainLayout() {
  return (
    <div className="min-h-screen bg-background grid-bg relative">
      <ParticleField />
      <ScanlineOverlay />
      <AlertSystem />
      <Sidebar />
      <main className="ml-64 min-h-screen relative z-10">
        <Outlet />
      </main>
      <SettingsPanel />
    </div>
  );
}
