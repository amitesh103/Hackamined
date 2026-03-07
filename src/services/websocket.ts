import { mockInverters, type Inverter } from "@/data/mockData";

type WSCallback = (data: Inverter[]) => void;

const WS_BASE = import.meta.env.VITE_WS_BASE_URL || "ws://localhost:8000";

class WebSocketService {
  private ws: WebSocket | null = null;
  private callbacks: Set<WSCallback> = new Set();
  private reconnectAttempts = 0;
  private maxReconnectDelay = 30000;

  private getRiskLevel(score: number): Inverter["risk_level"] {
    if (score > 0.85) return "CRITICAL";
    if (score > 0.6) return "HIGH";
    if (score > 0.3) return "MEDIUM";
    return "LOW";
  }

  private getStatus(score: number): Inverter["status"] {
    if (score > 0.85) return "critical";
    if (score > 0.6) return "high_risk";
    if (score > 0.3) return "warning";
    return "healthy";
  }

  connect(plantId = "PLANT_1") {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    const url = `${WS_BASE}/ws/stream/${plantId}`;
    console.log(`[WS] Connecting to ${url}...`);

    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        console.log("[WS] Connected");
        this.reconnectAttempts = 0;
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'update' && data.inverters) {
            const mapped: Inverter[] = data.inverters.map((inv: any) => ({
              id: inv.inverter_id,
              name: `Inverter ${inv.inverter_id.split('_').pop()}`,
              risk_score: inv.risk_score,
              risk_level: this.getRiskLevel(inv.risk_score),
              status: this.getStatus(inv.risk_score),
              temperature: inv.temperature,
              efficiency: inv.efficiency,
              power_output: inv.power,
              string_mismatch: 0,
              location: "Main Array",
              last_updated: data.timestamp
            }));
            this.callbacks.forEach(cb => cb(mapped));
          }
        } catch (e) {
          console.error("[WS] Parse error", e);
        }
      };

      this.ws.onclose = (e) => {
        this.ws = null;
        if (e.code !== 1000) {
          const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), this.maxReconnectDelay);
          console.warn(`[WS] Disconnected. Reconnecting in ${delay}ms...`);
          this.reconnectAttempts++;
          setTimeout(() => this.connect(plantId), delay);
        }
      };

      this.ws.onerror = (err) => {
        console.error("[WS] Error", err);
      };
    } catch (err) {
      console.error("[WS] Sync error", err);
    }
  }

  subscribe(cb: WSCallback) {
    this.callbacks.add(cb);
    return () => this.callbacks.delete(cb);
  }

  disconnect() {
    if (this.ws) {
      this.ws.close(1000); // Normal closure
      this.ws = null;
    }
    this.callbacks.clear();
  }
}

export const wsService = new WebSocketService();
