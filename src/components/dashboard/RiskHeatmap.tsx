import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import type { Inverter } from "@/data/mockData";

interface RiskHeatmapProps {
  inverters: Inverter[];
}

function getRiskColor(score: number) {
  if (score > 0.85) return { bg: "bg-destructive/20", border: "border-destructive/50", text: "text-destructive" };
  if (score > 0.6) return { bg: "bg-warning/20", border: "border-warning/50", text: "text-warning" };
  if (score > 0.3) return { bg: "bg-yellow-500/20", border: "border-yellow-500/40", text: "text-yellow-400" };
  return { bg: "bg-primary/20", border: "border-primary/40", text: "text-primary" };
}

export function RiskHeatmap({ inverters }: RiskHeatmapProps) {
  const navigate = useNavigate();

  return (
    <div className="glass-card p-5">
      <h3 className="text-sm font-semibold text-foreground mb-4 uppercase tracking-wider">Risk Heatmap</h3>
      <div className="grid grid-cols-4 sm:grid-cols-5 gap-2">
        {inverters.map((inv, i) => {
          const colors = getRiskColor(inv.risk_score);
          return (
            <motion.button
              key={inv.id}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: i * 0.03 }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => navigate(`/inverter/${inv.id}`)}
              className={`relative p-3 rounded-lg border ${colors.bg} ${colors.border} cursor-pointer transition-all hover:shadow-lg group`}
            >
              <div className={`text-xs font-mono font-bold ${colors.text}`}>{inv.id}</div>
              <div className={`text-lg font-mono font-bold mt-1 ${colors.text}`}>
                {(inv.risk_score * 100).toFixed(0)}%
              </div>
              <div className="text-[10px] text-muted-foreground mt-0.5">{inv.risk_level}</div>
              {inv.risk_score > 0.85 && (
                <div className="absolute -top-1 -right-1 h-2.5 w-2.5 rounded-full bg-destructive animate-pulse-glow" />
              )}
            </motion.button>
          );
        })}
      </div>
      <div className="flex items-center gap-4 mt-4 text-[10px] text-muted-foreground">
        <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-primary" /> Low</span>
        <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-yellow-400" /> Medium</span>
        <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-warning" /> High</span>
        <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-destructive" /> Critical</span>
      </div>
    </div>
  );
}
