import { useState, useEffect } from "react";
import { KPICards } from "@/components/dashboard/KPICards";
import { RiskHeatmap } from "@/components/dashboard/RiskHeatmap";
import { TrendCharts } from "@/components/dashboard/TrendCharts";
import { DiagnosisCard } from "@/components/dashboard/DiagnosisCard";
import { AlertPanel } from "@/components/dashboard/AlertPanel";
import { TicketPanel } from "@/components/dashboard/TicketPanel";
import { getInverters } from "@/services/api";
import { wsService } from "@/services/websocket";
import type { Inverter } from "@/data/mockData";
import { AppLayout } from "@/components/layout/AppLayout";

import { useInverters } from "@/hooks/useInverters";

export default function Dashboard() {
  const { inverters, isLoading, error } = useInverters();
  const [selectedInverter, setSelectedInverter] = useState("INV_001");

  if (isLoading) return <div className="flex h-screen items-center justify-center font-mono text-primary">Loading monitoring data...</div>;
  if (error) return <div className="flex h-screen items-center justify-center text-destructive">{error}</div>;


  return (
    <AppLayout>
      <div className="p-4 lg:p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Plant Overview</h1>
            <p className="text-sm text-muted-foreground font-mono">Solar Farm Alpha · Real-time Monitoring</p>
          </div>
          <select
            value={selectedInverter}
            onChange={(e) => setSelectedInverter(e.target.value)}
            className="rounded-lg border border-border/40 bg-muted/30 px-3 py-2 text-sm text-foreground font-mono focus:outline-none focus:ring-1 focus:ring-primary/50"
          >
            {inverters.map(inv => (
              <option key={inv.id} value={inv.id}>{inv.id} — {inv.risk_level}</option>
            ))}
          </select>
        </div>

        <KPICards inverters={inverters} />

        <div className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <RiskHeatmap inverters={inverters} />
          </div>
          <div className="h-[400px]">
            <AlertPanel />
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <TrendCharts inverterId={selectedInverter} />
            <DiagnosisCard inverterId={selectedInverter} />
          </div>
          <div className="h-[650px]">
            <TicketPanel />
          </div>
        </div>
      </div>
    </AppLayout>
  );
}
