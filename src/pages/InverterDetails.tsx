import { useParams, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { SHAPChart } from "@/components/inverter/SHAPChart";
import { DeltaSHAPChart } from "@/components/inverter/DeltaSHAPChart";
import { TrendCharts } from "@/components/dashboard/TrendCharts";
import { DiagnosisCard } from "@/components/dashboard/DiagnosisCard";
import { AppLayout } from "@/components/layout/AppLayout";
import { AlertTriangle, Thermometer, Gauge, Zap, ChevronLeft } from "lucide-react";
import { useInverters } from "@/hooks/useInverters";

export default function InverterDetails() {
  const { id = "INV_001" } = useParams();
  const navigate = useNavigate();
  const { inverters, inverterMap, isLoading } = useInverters();
  const inverter = inverterMap[id];

  if (isLoading || !inverter) {
    return (
      <div className="flex h-screen items-center justify-center font-mono text-primary">
        Loading diagnostics...
      </div>
    );
  }

  const riskPercent = (inverter.risk_score || 0) * 100;
  const riskColor = riskPercent > 85 ? "text-destructive" : riskPercent > 60 ? "text-warning" : riskPercent > 30 ? "text-yellow-400" : "text-primary";

  return (
    <AppLayout>
      <div className="p-4 lg:p-6 space-y-6">
        {/* Navigation Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 glass-card p-4">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/')}
              className="p-2 hover:bg-muted/50 rounded-lg transition-colors group"
            >
              <ChevronLeft className="h-5 w-5 text-muted-foreground group-hover:text-primary" />
            </button>
            <div>
              <h1 className="text-xl font-bold tracking-tight">{inverter.id}</h1>
              <p className="text-[10px] text-muted-foreground font-mono uppercase tracking-widest">Asset Diagnostics</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-xs text-muted-foreground font-mono hidden sm:inline">Switch Asset:</span>
            <select
              value={id}
              onChange={(e) => navigate(`/inverter/${e.target.value}`)}
              className="rounded-lg border border-border/40 bg-muted/20 px-3 py-1.5 text-xs text-foreground font-mono focus:outline-none focus:ring-1 focus:ring-primary/50 cursor-pointer"
            >
              {inverters.map(inv => (
                <option key={inv.id} value={inv.id}>
                  {inv.id} — {(inv.risk_score * 100).toFixed(0)}% Risk
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Risk Gauge & Quick Stats */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-card p-5 text-center">
            <Gauge className={`h-6 w-6 mx-auto mb-2 ${riskColor}`} />
            <div className={`text-4xl font-mono font-black ${riskColor}`}>{riskPercent.toFixed(0)}%</div>
            <div className="text-xs text-muted-foreground mt-1">Risk Score</div>
          </motion.div>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="glass-card p-5 text-center">
            <Thermometer className="h-6 w-6 mx-auto mb-2 text-destructive" />
            <div className="text-4xl font-mono font-black text-foreground">{inverter.temperature?.toFixed(1) || "0.0"}°</div>
            <div className="text-xs text-muted-foreground mt-1">Temperature</div>
          </motion.div>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="glass-card p-5 text-center">
            <Zap className="h-6 w-6 mx-auto mb-2 text-primary" />
            <div className="text-4xl font-mono font-black text-foreground">{inverter.efficiency?.toFixed(1) || "0.0"}%</div>
            <div className="text-xs text-muted-foreground mt-1">Efficiency</div>
          </motion.div>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="glass-card p-5 text-center">
            <AlertTriangle className="h-6 w-6 mx-auto mb-2 text-warning" />
            <div className="text-4xl font-mono font-black text-foreground">{inverter.string_mismatch?.toFixed(2) || "0.00"}</div>
            <div className="text-xs text-muted-foreground mt-1">String Mismatch</div>
          </motion.div>
        </div>

        <TrendCharts inverterId={id} />

        <div className="grid lg:grid-cols-2 gap-6">
          <SHAPChart inverterId={id} />
          <DeltaSHAPChart inverterId={id} />
        </div>

        <DiagnosisCard inverterId={id} />
      </div>
    </AppLayout>
  );
}
