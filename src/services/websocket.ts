import { mockInverters, type Inverter } from "@/data/mockData";

type WSCallback = (data: Inverter[]) => void;

const WS_BASE = import.meta.env.VITE_WS_BASE_URL || "ws://localhost:8000";

class WebSocketService {
  private ws: WebSocket | null = null;
  private callbacks: Set<WSCallback> = new Set();
  private mockInterval: ReturnType<typeof setInterval> | null = null;
  private useMock = true;

  connect(plantId = "plant-1") {
    if (!this.useMock) {
      try {
        this.ws = new WebSocket(`${WS_BASE}/ws/stream/${plantId}`);
        this.ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          this.callbacks.forEach(cb => cb(data));
        };
        this.ws.onerror = () => {
          this.useMock = true;
          this.startMockStream();
        };
        return;
      } catch {
        this.useMock = true;
      }
    }
    this.startMockStream();
  }

  private startMockStream() {
    this.mockInterval = setInterval(() => {
      const updated = mockInverters.map(inv => ({
        ...inv,
        temperature: inv.temperature + (Math.random() - 0.48) * 0.5,
        efficiency: Math.max(50, Math.min(100, inv.efficiency + (Math.random() - 0.52) * 0.3)),
        risk_score: Math.max(0, Math.min(1, inv.risk_score + (Math.random() - 0.48) * 0.01)),
        last_updated: new Date().toISOString(),
      }));
      this.callbacks.forEach(cb => cb(updated));
    }, 5000);
  }

  subscribe(cb: WSCallback) {
    this.callbacks.add(cb);
    return () => this.callbacks.delete(cb);
  }

  disconnect() {
    this.ws?.close();
    if (this.mockInterval) clearInterval(this.mockInterval);
    this.callbacks.clear();
  }
}

export const wsService = new WebSocketService();
