import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { Zap, Brain, Shield, Workflow, BarChart3, Bot, ArrowRight, Activity } from "lucide-react";

const features = [
  { icon: Brain, title: "ML Failure Prediction", desc: "LSTM + XGBoost models predict inverter failures 48 hours in advance with 94% accuracy." },
  { icon: Shield, title: "SHAP Explainability", desc: "Understand exactly why an inverter is at risk with transparent feature-level explanations." },
  { icon: Workflow, title: "Autonomous Maintenance", desc: "AI agents automatically generate work orders and prioritize maintenance schedules." },
  { icon: BarChart3, title: "Real-Time Monitoring", desc: "WebSocket-powered live telemetry streaming with sub-second dashboard updates." },
  { icon: Bot, title: "RAG AI Assistant", desc: "Ask questions in natural language about plant performance and get instant diagnostic answers." },
  { icon: Activity, title: "Delta-SHAP Analysis", desc: "Track how risk drivers change over time to catch emerging failure patterns early." },
];

const archLayers = [
  { label: "Data Pipeline", desc: "SCADA · IoT Sensors · Weather API", color: "border-secondary/40 bg-secondary/5" },
  { label: "ML Engine", desc: "LSTM · XGBoost · Anomaly Detection", color: "border-primary/40 bg-primary/5" },
  { label: "GenAI Layer", desc: "LLM Diagnostics · RAG · SHAP", color: "border-neon-purple/40 bg-neon-purple/5" },
  { label: "FastAPI Backend", desc: "REST · WebSocket · Agent Orchestration", color: "border-warning/40 bg-warning/5" },
  { label: "React Dashboard", desc: "Monitoring · Analytics · Assistant", color: "border-secondary/40 bg-secondary/5" },
];

export default function Landing() {
  return (
    <div className="min-h-screen bg-background grid-pattern">
      {/* Hero */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-primary/5 via-transparent to-transparent" />
        <div className="relative z-10 text-center px-6 max-w-4xl mx-auto">
          <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}>
            <div className="flex items-center justify-center gap-3 mb-6">
              <Zap className="h-12 w-12 text-primary" />
              <h1 className="text-6xl md:text-8xl font-black tracking-tighter">
                Solar<span className="text-primary neon-text">Mind</span>
              </h1>
            </div>
            <div className="flex items-center justify-center gap-2 mb-4">
              <span className="text-secondary font-mono text-sm tracking-widest uppercase">AI-Powered Platform</span>
            </div>
            <p className="text-2xl md:text-3xl font-light text-muted-foreground mb-2">
              Predict. Explain. Act. <span className="text-primary font-semibold">Automatically.</span>
            </p>
            <p className="text-muted-foreground max-w-2xl mx-auto mb-10">
              AI-driven solar inverter monitoring and predictive maintenance platform. Machine learning predictions, SHAP explainability, and autonomous maintenance — all in real time.
            </p>
            <div className="flex items-center justify-center gap-4">
              <Link
                to="/dashboard"
                className="inline-flex items-center gap-2 rounded-xl bg-primary px-8 py-4 text-primary-foreground font-semibold hover:bg-primary/90 transition-colors neon-glow"
              >
                Enter Dashboard <ArrowRight className="h-4 w-4" />
              </Link>
              <a
                href="#architecture"
                className="inline-flex items-center gap-2 rounded-xl border border-border/50 px-8 py-4 text-foreground hover:bg-muted/30 transition-colors"
              >
                View Architecture
              </a>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features */}
      <section className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <motion.h2
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-3xl font-bold text-center mb-4"
          >
            AI-Powered <span className="text-primary">Capabilities</span>
          </motion.h2>
          <p className="text-center text-muted-foreground mb-12 max-w-2xl mx-auto">
            End-to-end intelligent monitoring from data ingestion to autonomous action.
          </p>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((f, i) => (
              <motion.div
                key={f.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="glass-card-hover p-6 group"
              >
                <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center mb-4 group-hover:neon-glow transition-all">
                  <f.icon className="h-5 w-5 text-primary" />
                </div>
                <h3 className="font-semibold text-foreground mb-2">{f.title}</h3>
                <p className="text-sm text-muted-foreground">{f.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Architecture */}
      <section id="architecture" className="py-24 px-6 bg-muted/10">
        <div className="max-w-3xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">
            System <span className="text-secondary neon-text-blue">Architecture</span>
          </h2>
          <div className="space-y-3">
            {archLayers.map((layer, i) => (
              <motion.div
                key={layer.label}
                initial={{ opacity: 0, x: -30 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className={`flex items-center gap-4 rounded-xl border p-5 ${layer.color}`}
              >
                <div className="text-2xl font-mono font-bold text-muted-foreground w-8">{i + 1}</div>
                <div>
                  <div className="font-semibold text-foreground">{layer.label}</div>
                  <div className="text-xs text-muted-foreground font-mono">{layer.desc}</div>
                </div>
                {i < archLayers.length - 1 && (
                  <ArrowRight className="ml-auto h-4 w-4 text-muted-foreground" />
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-24 px-6 text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          className="max-w-2xl mx-auto glass-card p-12 neon-glow"
        >
          <Zap className="h-10 w-10 text-primary mx-auto mb-4" />
          <h2 className="text-3xl font-bold mb-4">Ready to Monitor Smarter?</h2>
          <p className="text-muted-foreground mb-8">
            Access real-time AI diagnostics, predictive analytics, and autonomous maintenance workflows.
          </p>
          <Link
            to="/dashboard"
            className="inline-flex items-center gap-2 rounded-xl bg-primary px-8 py-4 text-primary-foreground font-semibold hover:bg-primary/90 transition-colors"
          >
            Launch Dashboard <ArrowRight className="h-4 w-4" />
          </Link>
        </motion.div>
      </section>

      <footer className="border-t border-border/30 py-8 text-center text-xs text-muted-foreground">
        © 2026 SolarMind AI · Predictive Solar Intelligence Platform
      </footer>
    </div>
  );
}
