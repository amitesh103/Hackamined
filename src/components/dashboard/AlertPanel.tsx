import { useState, useEffect } from "react";
import { getAlerts, type Alert } from "@/services/api";
import { AlertCircle, AlertTriangle } from "lucide-react";
import { motion } from "framer-motion";

export function AlertPanel() {
    const [alerts, setAlerts] = useState<Alert[]>([]);

    useEffect(() => {
        // Poll for alerts every 5 seconds
        const fetchAlerts = () => getAlerts().then(setAlerts);
        fetchAlerts();
        const interval = setInterval(fetchAlerts, 5000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="glass-card p-5 h-full flex flex-col">
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-foreground uppercase tracking-wider">Active Alerts</h3>
                <span className="text-xs px-2 py-1 rounded-full bg-muted/50 text-muted-foreground font-mono">
                    {alerts.length} Total
                </span>
            </div>

            <div className="flex-1 overflow-y-auto space-y-3 pr-2 scrollbar-thin">
                {alerts.length === 0 ? (
                    <div className="text-center text-sm text-muted-foreground mt-8">No active alerts. System healthy.</div>
                ) : (
                    alerts.map((alert, i) => (
                        <motion.div
                            key={alert.id}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.05 }}
                            className={`p-3 rounded-lg border ${alert.level === "critical"
                                    ? "bg-destructive/10 border-destructive/30"
                                    : "bg-warning/10 border-warning/30"
                                }`}
                        >
                            <div className="flex items-start gap-3">
                                {alert.level === "critical" ? (
                                    <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                                ) : (
                                    <AlertTriangle className="h-5 w-5 text-warning shrink-0 mt-0.5" />
                                )}
                                <div>
                                    <div className="flex items-center gap-2 mb-1">
                                        <span className="font-mono text-sm font-bold text-foreground">{alert.inverter_id}</span>
                                        <span className={`text-[10px] px-1.5 py-0.5 rounded uppercase tracking-wider ${alert.level === "critical" ? "bg-destructive/20 text-destructive" : "bg-warning/20 text-warning"
                                            }`}>
                                            {alert.level}
                                        </span>
                                    </div>
                                    <p className="text-xs text-muted-foreground leading-relaxed">{alert.message}</p>
                                    <div className="text-[10px] text-muted-foreground/60 mt-2 font-mono">
                                        Score: {(alert.risk_score * 100).toFixed(1)}% • {new Date(alert.timestamp).toLocaleTimeString()}
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    ))
                )}
            </div>
        </div>
    );
}
