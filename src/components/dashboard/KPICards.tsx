import { motion } from "framer-motion";
import { Server, ShieldCheck, AlertTriangle, AlertOctagon } from "lucide-react";
import type { Inverter } from "@/data/mockData";

interface KPICardsProps {
  inverters: Inverter[];
}

export function KPICards({ inverters }: KPICardsProps) {
  const total = inverters.length;
  const healthy = inverters.filter(i => i.status === "healthy").length;
  const highRisk = inverters.filter(i => i.status === "high_risk" || i.status === "warning").length;
  const critical = inverters.filter(i => i.status === "critical").length;

  const cards = [
    { label: "Total Inverters", value: total, icon: Server, color: "text-secondary", bg: "bg-secondary/10", border: "border-secondary/20" },
    { label: "Healthy", value: healthy, icon: ShieldCheck, color: "text-primary", bg: "bg-primary/10", border: "border-primary/20" },
    { label: "High Risk", value: highRisk, icon: AlertTriangle, color: "text-warning", bg: "bg-warning/10", border: "border-warning/20" },
    { label: "Critical", value: critical, icon: AlertOctagon, color: "text-destructive", bg: "bg-destructive/10", border: "border-destructive/20", glow: critical > 0 },
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card, i) => (
        <motion.div
          key={card.label}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.1 }}
          className={`glass-card p-5 border ${card.border} ${card.glow ? "neon-glow-red animate-pulse-glow" : ""}`}
        >
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">{card.label}</span>
            <div className={`p-2 rounded-lg ${card.bg}`}>
              <card.icon className={`h-4 w-4 ${card.color}`} />
            </div>
          </div>
          <div className={`text-3xl font-bold font-mono ${card.color}`}>
            {card.value}
          </div>
        </motion.div>
      ))}
    </div>
  );
}
