export interface Inverter {
  id: string;
  name: string;
  risk_score: number;
  risk_level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  status: "healthy" | "warning" | "high_risk" | "critical";
  temperature: number;
  efficiency: number;
  power_output: number;
  string_mismatch: number;
  location: string;
  last_updated: string;
}

export interface TelemetryPoint {
  timestamp: string;
  temperature: number;
  efficiency: number;
  string_mismatch: number;
  power_output: number;
}

export interface DiagnosticReport {
  inverter_id: string;
  risk_score: number;
  risk_level: string;
  summary: string;
  root_cause: string;
  action: string;
  confidence: number;
  causal_drivers: { feature: string; contribution: number }[];
}

export interface SHAPValue {
  feature: string;
  value: number;
  base_value: number;
}

export interface DeltaSHAPValue {
  feature: string;
  current: number;
  previous: number;
  delta: number;
}

function getRiskLevel(score: number): Inverter["risk_level"] {
  if (score > 0.85) return "CRITICAL";
  if (score > 0.6) return "HIGH";
  if (score > 0.3) return "MEDIUM";
  return "LOW";
}

function getStatus(score: number): Inverter["status"] {
  if (score > 0.85) return "critical";
  if (score > 0.6) return "high_risk";
  if (score > 0.3) return "warning";
  return "healthy";
}

const riskScores = [
  0.12, 0.08, 0.45, 0.22, 0.67, 0.15, 0.91, 0.33, 0.05, 0.78,
  0.19, 0.55, 0.27, 0.82, 0.11, 0.43, 0.88, 0.14, 0.36, 0.71
];

export const mockInverters: Inverter[] = Array.from({ length: 20 }, (_, i) => {
  const risk = riskScores[i];
  return {
    id: `INV-${String(i + 1).padStart(2, "0")}`,
    name: `Inverter ${i + 1}`,
    risk_score: risk,
    risk_level: getRiskLevel(risk),
    status: getStatus(risk),
    temperature: 35 + risk * 40 + Math.random() * 5,
    efficiency: Math.max(60, 98 - risk * 30 - Math.random() * 5),
    power_output: Math.max(5, 25 - risk * 15 + Math.random() * 3),
    string_mismatch: risk * 0.5 + Math.random() * 0.1,
    location: `Zone ${String.fromCharCode(65 + (i % 4))}-${Math.floor(i / 4) + 1}`,
    last_updated: new Date().toISOString(),
  };
});

export function generateTelemetry(inverterId: string, hours = 48): TelemetryPoint[] {
  const inv = mockInverters.find(i => i.id === inverterId) || mockInverters[0];
  const baseTemp = 35 + inv.risk_score * 20;
  const baseEff = 95 - inv.risk_score * 25;
  const now = Date.now();

  return Array.from({ length: hours * 2 }, (_, i) => {
    const t = new Date(now - (hours * 2 - i) * 30 * 60 * 1000);
    const hourOfDay = t.getHours();
    const dayCycle = Math.sin((hourOfDay - 6) * Math.PI / 12);
    
    return {
      timestamp: t.toISOString(),
      temperature: baseTemp + dayCycle * 8 + Math.random() * 3 + (inv.risk_score > 0.6 ? i * 0.05 : 0),
      efficiency: Math.max(50, baseEff - dayCycle * 3 + Math.random() * 2 - (inv.risk_score > 0.6 ? i * 0.03 : 0)),
      string_mismatch: inv.string_mismatch + Math.random() * 0.05 + (inv.risk_score > 0.6 ? i * 0.002 : 0),
      power_output: Math.max(0, (inv.power_output + dayCycle * 8) * (dayCycle > 0 ? 1 : 0.1) + Math.random() * 2),
    };
  });
}

export function generateReport(inverterId: string): DiagnosticReport {
  const inv = mockInverters.find(i => i.id === inverterId) || mockInverters[0];
  
  const reports: Record<string, Partial<DiagnosticReport>> = {
    critical: {
      summary: `${inv.id} shows critical degradation patterns. Immediate intervention required to prevent complete failure within 48 hours.`,
      root_cause: "Persistent thermal runaway detected in IGBT module. Fan subsystem degradation accelerating junction temperatures beyond safe operating limits.",
      action: "IMMEDIATE: Reduce load to 50%. Schedule emergency IGBT replacement and cooling system inspection within 24 hours.",
    },
    high_risk: {
      summary: `${inv.id} exhibiting progressive efficiency loss with thermal anomalies indicating accelerated component wear.`,
      root_cause: "String current mismatch combined with rising DC-side temperatures suggests partial shading or connector degradation on strings 3-4.",
      action: "Schedule maintenance within 72 hours. Inspect string connectors and clean affected panels. Monitor thermal trends.",
    },
    warning: {
      summary: `${inv.id} shows minor deviations from baseline performance. Monitoring recommended.`,
      root_cause: "Slight efficiency degradation consistent with seasonal soiling patterns. No critical component issues detected.",
      action: "Include in next scheduled maintenance cycle. Consider panel cleaning if efficiency drops below 85%.",
    },
    healthy: {
      summary: `${inv.id} operating within normal parameters. All subsystems nominal.`,
      root_cause: "No anomalies detected. Performance consistent with expected baseline.",
      action: "Continue standard monitoring. Next scheduled maintenance in 30 days.",
    },
  };

  const report = reports[inv.status] || reports.healthy;

  return {
    inverter_id: inv.id,
    risk_score: inv.risk_score,
    risk_level: inv.risk_level,
    summary: report.summary!,
    root_cause: report.root_cause!,
    action: report.action!,
    confidence: 0.85 + Math.random() * 0.12,
    causal_drivers: [
      { feature: "Temperature Rise Rate", contribution: 0.15 + inv.risk_score * 0.3 },
      { feature: "Efficiency Delta", contribution: 0.1 + inv.risk_score * 0.25 },
      { feature: "String Mismatch Coeff", contribution: 0.08 + inv.risk_score * 0.2 },
      { feature: "Power Variance", contribution: 0.05 + inv.risk_score * 0.15 },
      { feature: "Operating Hours", contribution: 0.03 + Math.random() * 0.1 },
      { feature: "Ambient Temp Delta", contribution: 0.02 + Math.random() * 0.08 },
    ].sort((a, b) => b.contribution - a.contribution),
  };
}

export function generateSHAPValues(inverterId: string): SHAPValue[] {
  const inv = mockInverters.find(i => i.id === inverterId) || mockInverters[0];
  return [
    { feature: "dc_voltage_std", value: 0.12 + inv.risk_score * 0.2, base_value: 0.5 },
    { feature: "temperature_rise_rate", value: 0.1 + inv.risk_score * 0.25, base_value: 0.5 },
    { feature: "efficiency_rolling_mean", value: -(0.08 + inv.risk_score * 0.15), base_value: 0.5 },
    { feature: "string_mismatch_coeff", value: 0.06 + inv.risk_score * 0.18, base_value: 0.5 },
    { feature: "power_output_variance", value: 0.04 + inv.risk_score * 0.1, base_value: 0.5 },
    { feature: "operating_hours", value: 0.02 + Math.random() * 0.05, base_value: 0.5 },
    { feature: "ambient_temp_delta", value: -(0.01 + Math.random() * 0.03), base_value: 0.5 },
    { feature: "ac_frequency_std", value: 0.01 + Math.random() * 0.02, base_value: 0.5 },
  ];
}

export function generateDeltaSHAP(inverterId: string): DeltaSHAPValue[] {
  const inv = mockInverters.find(i => i.id === inverterId) || mockInverters[0];
  const features = [
    "temperature_rise_rate", "efficiency_rolling_mean", "string_mismatch_coeff",
    "dc_voltage_std", "power_output_variance", "operating_hours"
  ];
  return features.map(f => {
    const prev = Math.random() * 0.2;
    const curr = prev + (inv.risk_score > 0.5 ? Math.random() * 0.15 : -Math.random() * 0.05);
    return { feature: f, current: curr, previous: prev, delta: curr - prev };
  });
}

export const mockChatResponses: Record<string, string> = {
  "default": "I've analyzed the plant data. Based on current telemetry, **3 inverters** require attention. INV-07 and INV-17 are in critical status, while INV-14 is at high risk with a score of 0.82.\n\nWould you like me to provide detailed diagnostics for any specific inverter?",
  "thermal": "## Thermal Anomaly Analysis\n\nThe inverter with the highest thermal anomaly is **INV-07** with a temperature rise rate of `2.3°C/hr`, significantly above the baseline of `0.5°C/hr`.\n\n### Top 3 Thermal Anomalies:\n1. **INV-07** — 72.4°C (Critical)\n2. **INV-17** — 68.1°C (Critical)\n3. **INV-14** — 61.3°C (High Risk)\n\nRecommendation: Immediate cooling system inspection for INV-07.",
  "risk": "## Weekly Risk Assessment\n\n| Inverter | Risk Score | Trend |\n|----------|-----------|-------|\n| INV-07 | 0.91 | ↑ Rising |\n| INV-17 | 0.88 | ↑ Rising |\n| INV-14 | 0.82 | → Stable |\n| INV-10 | 0.78 | ↑ Rising |\n| INV-05 | 0.67 | → Stable |\n\n**4 inverters** show rising risk trends. Proactive maintenance recommended for INV-07 and INV-17 within 48 hours.",
  "inv-14": "## INV-14 Diagnostic Report\n\n**Risk Score:** 0.82 (HIGH)\n\nINV-14 is experiencing progressive efficiency loss driven by:\n\n1. **String current mismatch** — Strings 3-4 show 15% lower current\n2. **Rising DC-side temperatures** — +1.2°C/hr above baseline\n3. **Connector degradation** — Impedance measurements suggest oxidation\n\n### Recommended Actions:\n- Schedule connector inspection within 72 hours\n- Clean panels on strings 3-4\n- Monitor thermal trend for further acceleration",
};
