import { useState, useEffect } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { getTelemetry } from "@/services/api";
import type { TelemetryPoint } from "@/data/mockData";

interface TrendChartsProps {
  inverterId?: string;
}

export function TrendCharts({ inverterId = "INV-14" }: TrendChartsProps) {
  const [data, setData] = useState<TelemetryPoint[]>([]);
  const [activeChart, setActiveChart] = useState<"temperature" | "efficiency" | "string_mismatch">("temperature");

  useEffect(() => {
    getTelemetry(inverterId).then(setData);
  }, [inverterId]);

  const chartConfig = {
    temperature: { label: "Temperature (°C)", color: "hsl(0, 84%, 60%)", key: "temperature" as const },
    efficiency: { label: "Efficiency (%)", color: "hsl(142, 72%, 50%)", key: "efficiency" as const },
    string_mismatch: { label: "String Mismatch", color: "hsl(210, 100%, 56%)", key: "string_mismatch" as const },
  };

  const config = chartConfig[activeChart];
  const formatted = data.map(d => ({
    ...d,
    time: new Date(d.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
  }));

  return (
    <div className="glass-card p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-foreground uppercase tracking-wider">Telemetry — {inverterId}</h3>
        <div className="flex gap-1">
          {Object.entries(chartConfig).map(([key, cfg]) => (
            <button
              key={key}
              onClick={() => setActiveChart(key as typeof activeChart)}
              className={`px-3 py-1 text-xs rounded-md font-mono transition-colors ${
                activeChart === key
                  ? "bg-primary/20 text-primary border border-primary/30"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
              }`}
            >
              {cfg.label.split(" (")[0]}
            </button>
          ))}
        </div>
      </div>
      <div className="h-[250px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={formatted}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 14%, 18%)" />
            <XAxis
              dataKey="time"
              stroke="hsl(215, 20%, 55%)"
              fontSize={10}
              tickLine={false}
              interval={Math.floor(formatted.length / 8)}
            />
            <YAxis stroke="hsl(215, 20%, 55%)" fontSize={10} tickLine={false} />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(220, 18%, 10%)",
                border: "1px solid hsl(220, 14%, 18%)",
                borderRadius: "8px",
                fontSize: "12px",
              }}
            />
            <Line
              type="monotone"
              dataKey={config.key}
              stroke={config.color}
              strokeWidth={2}
              dot={false}
              animationDuration={1000}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
