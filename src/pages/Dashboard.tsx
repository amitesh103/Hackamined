import { useState, useEffect } from "react";
import { KPICards } from "@/components/dashboard/KPICards";
import { RiskHeatmap } from "@/components/dashboard/RiskHeatmap";
import { TrendCharts } from "@/components/dashboard/TrendCharts";
import { DiagnosisCard } from "@/components/dashboard/DiagnosisCard";
import { getInverters } from "@/services/api";
import { wsService } from "@/services/websocket";
import type { Inverter } from "@/data/mockData";
import { AppLayout } from "@/components/layout/AppLayout";

export default function Dashboard() {
  const [inverters, setInverters] = useState<Inverter[]>([]);
  const [selectedInverter, setSelectedInverter] = useState("INV-14");

  useEffect(() => {
    getInverters().then(setInverters);
    wsService.connect();
    const unsub = wsService.subscribe(setInverters);
    return () => { unsub(); wsService.disconnect(); };
  }, []);

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

        <div className="grid lg:grid-cols-2 gap-6">
          <RiskHeatmap inverters={inverters} />
          <TrendCharts inverterId={selectedInverter} />
        </div>

        <DiagnosisCard inverterId={selectedInverter} />
      </div>
    </AppLayout>
  );
}
