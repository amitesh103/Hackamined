import { useState, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from "recharts";
import { getDeltaSHAP } from "@/services/api";
import type { DeltaSHAPValue } from "@/data/mockData";

interface DeltaSHAPChartProps {
  inverterId: string;
}

export function DeltaSHAPChart({ inverterId }: DeltaSHAPChartProps) {
  const [data, setData] = useState<DeltaSHAPValue[]>([]);

  useEffect(() => {
    getDeltaSHAP(inverterId).then(setData);
  }, [inverterId]);

  const sorted = [...data].sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));

  return (
    <div className="glass-card p-5">
      <h3 className="text-sm font-semibold text-foreground uppercase tracking-wider mb-4">
        Δ-SHAP Change Analysis
      </h3>
      <div className="h-[250px]">
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
              formatter={(value: number) => [value.toFixed(4), "Delta"]}
            />
            <ReferenceLine x={0} stroke="hsl(220, 14%, 25%)" />
            <Bar dataKey="delta" radius={[0, 4, 4, 0]}>
              {sorted.map((entry, i) => (
                <Cell key={i} fill={entry.delta >= 0 ? "hsl(30, 95%, 55%)" : "hsl(142, 72%, 50%)"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-[10px] text-muted-foreground mt-2">
        Orange = worsening trend · Green = improving trend
      </p>
    </div>
  );
}
