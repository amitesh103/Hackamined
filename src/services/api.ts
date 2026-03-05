import axios from "axios";
import {
  mockInverters,
  generateTelemetry,
  generateReport,
  generateSHAPValues,
  generateDeltaSHAP,
  mockChatResponses,
  type Inverter,
  type TelemetryPoint,
  type DiagnosticReport,
  type SHAPValue,
  type DeltaSHAPValue,
} from "@/data/mockData";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 10000,
  headers: { "Content-Type": "application/json" },
});

let useMock = true;

export async function getInverters(): Promise<Inverter[]> {
  if (useMock) return mockInverters;
  try {
    const { data } = await api.get("/inverters");
    return data;
  } catch {
    useMock = true;
    return mockInverters;
  }
}

export async function getInverterReport(id: string): Promise<DiagnosticReport> {
  if (useMock) return generateReport(id);
  try {
    const { data } = await api.get(`/inverters/${id}/report`);
    return data;
  } catch {
    return generateReport(id);
  }
}

export async function getTelemetry(id: string): Promise<TelemetryPoint[]> {
  if (useMock) return generateTelemetry(id);
  try {
    const { data } = await api.get(`/inverters/${id}/telemetry`);
    return data;
  } catch {
    return generateTelemetry(id);
  }
}

export async function getSHAPValues(id: string): Promise<SHAPValue[]> {
  if (useMock) return generateSHAPValues(id);
  try {
    const { data } = await api.get(`/inverters/${id}/shap`);
    return data;
  } catch {
    return generateSHAPValues(id);
  }
}

export async function getDeltaSHAP(id: string): Promise<DeltaSHAPValue[]> {
  if (useMock) return generateDeltaSHAP(id);
  try {
    const { data } = await api.get(`/inverters/${id}/delta-shap`);
    return data;
  } catch {
    return generateDeltaSHAP(id);
  }
}

export async function queryAI(question: string): Promise<string> {
  if (useMock) {
    await new Promise(r => setTimeout(r, 800 + Math.random() * 1200));
    const q = question.toLowerCase();
    if (q.includes("thermal") || q.includes("temperature") || q.includes("heat")) return mockChatResponses.thermal;
    if (q.includes("risk") || q.includes("week") || q.includes("at risk")) return mockChatResponses.risk;
    if (q.includes("inv-14") || q.includes("inverter 14")) return mockChatResponses["inv-14"];
    return mockChatResponses.default;
  }
  try {
    const { data } = await api.post("/query", { question });
    return data.response || data.answer || JSON.stringify(data);
  } catch {
    return mockChatResponses.default;
  }
}

export async function predictInverter(id: string): Promise<{ risk_score: number; risk_level: string }> {
  if (useMock) {
    const inv = mockInverters.find(i => i.id === id);
    return { risk_score: inv?.risk_score || 0.5, risk_level: inv?.risk_level || "MEDIUM" };
  }
  const { data } = await api.post("/predict", { inverter_id: id });
  return data;
}
