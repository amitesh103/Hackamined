import { useParams } from "react-router-dom";
import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { SHAPChart } from "@/components/inverter/SHAPChart";
import { DeltaSHAPChart } from "@/components/inverter/DeltaSHAPChart";
import { TrendCharts } from "@/components/dashboard/TrendCharts";
import { DiagnosisCard } from "@/components/dashboard/DiagnosisCard";
import { getInverters } from "@/services/api";
import type { Inverter } from "@/data/mockData";
import { AppLayout } from "@/components/layout/AppLayout";
import { AlertTriangle, Thermometer, Gauge, Zap } from "lucide-react";

export default function InverterDetails() {
  const { id = "INV-14" } = useParams();
  const [inverter, setInverter] = useState<Inverter | null>(null);

  useEffect(() => {
    getInverters().then(data => {
      setInverter(data.find(i => i.id === id) || data[0]);
    });
  }, [id]);

  if (!inverter) return null;

  const riskPercent = inverter.risk_score * 100;
  const riskColor = riskPercent > 85 ? "text-destructive" : riskPercent > 60 ? "text-warning" : riskPercent > 30 ? "text-yellow-400" : "text-primary";

  return (
    <AppLayout>
      <div className="p-4 lg:p-6 space-y-6">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold tracking-tight">{inverter.id}</h1>
          <span className={`text-xs font-mono font-bold px-2.5 py-1 rounded-full border ${
            inverter.status === "critical" ? "border-destructive/30 bg-destructive/10 text-destructive" :
            inverter.status === "high_risk" ? "border-warning/30 bg-warning/10 text-warning" :
            "border-primary/30 bg-primary/10 text-primary"
          }`}>
            {inverter.risk_level}
          </span>
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
            <div className="text-4xl font-mono font-black text-foreground">{inverter.temperature.toFixed(1)}°</div>
            <div className="text-xs text-muted-foreground mt-1">Temperature</div>
          </motion.div>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="glass-card p-5 text-center">
            <Zap className="h-6 w-6 mx-auto mb-2 text-primary" />
            <div className="text-4xl font-mono font-black text-foreground">{inverter.efficiency.toFixed(1)}%</div>
            <div className="text-xs text-muted-foreground mt-1">Efficiency</div>
          </motion.div>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="glass-card p-5 text-center">
            <AlertTriangle className="h-6 w-6 mx-auto mb-2 text-warning" />
            <div className="text-4xl font-mono font-black text-foreground">{inverter.string_mismatch.toFixed(2)}</div>
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
