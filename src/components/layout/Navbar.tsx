import { Link, useLocation } from "react-router-dom";
import { Activity, Zap } from "lucide-react";
import { motion } from "framer-motion";
import { SidebarTrigger } from "@/components/ui/sidebar";

export function Navbar() {
  const location = useLocation();
  const isLanding = location.pathname === "/";

  return (
    <header className="sticky top-0 z-50 h-14 border-b border-border/40 bg-background/80 backdrop-blur-xl">
      <div className="flex h-full items-center gap-3 px-4">
        {!isLanding && <SidebarTrigger className="text-muted-foreground hover:text-foreground" />}
        <Link to="/" className="flex items-center gap-2">
          <div className="relative">
            <Zap className="h-6 w-6 text-primary" />
            <div className="absolute inset-0 animate-pulse-glow">
              <Zap className="h-6 w-6 text-primary opacity-50" />
            </div>
          </div>
          <span className="text-lg font-bold tracking-tight">
            Solar<span className="text-primary neon-text">Mind</span> AI
          </span>
        </Link>

        <div className="ml-auto flex items-center gap-3">
          <motion.div
            className="flex items-center gap-1.5 rounded-full border border-primary/30 bg-primary/10 px-3 py-1 text-xs font-mono"
            animate={{ opacity: [0.7, 1, 0.7] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <Activity className="h-3 w-3 text-primary" />
            <span className="text-primary">LIVE</span>
          </motion.div>
          <div className="text-xs text-muted-foreground font-mono">
            {new Date().toLocaleTimeString()}
          </div>
        </div>
      </div>
    </header>
  );
}
