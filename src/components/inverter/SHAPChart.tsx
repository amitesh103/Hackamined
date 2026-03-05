import { useState, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from "recharts";
import { getSHAPValues } from "@/services/api";
import type { SHAPValue } from "@/data/mockData";

interface SHAPChartProps {
  inverterId: string;
}

export function SHAPChart({ inverterId }: SHAPChartProps) {
  const [data, setData] = useState<SHAPValue[]>([]);

  useEffect(() => {
    getSHAPValues(inverterId).then(setData);
  }, [inverterId]);

  const sorted = [...data].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  return (
    <div className="glass-card p-5">
      <h3 className="text-sm font-semibold text-foreground uppercase tracking-wider mb-4">
        SHAP Feature Contributions
      </h3>
      <div className="h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={sorted} layout="vertical">
            <XAxis type="number" stroke="hsl(215, 20%, 55%)" fontSize={10} />
            <YAxis dataKey="feature" type="category" stroke="hsl(215, 20%, 55%)" fontSize={9} width={150} />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(220, 18%, 10%)",
                border: "1px solid hsl(220, 14%, 18%)",
                borderRadius: "8px",
                fontSize: "12px",
              }}
              formatter={(value: number) => [value.toFixed(4), "SHAP Value"]}
            />
            <ReferenceLine x={0} stroke="hsl(220, 14%, 25%)" />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {sorted.map((entry, i) => (
                <Cell key={i} fill={entry.value >= 0 ? "hsl(0, 84%, 60%)" : "hsl(210, 100%, 56%)"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-[10px] text-muted-foreground mt-2">
        Red = increases risk prediction · Blue = decreases risk prediction
      </p>
    </div>
  );
}
