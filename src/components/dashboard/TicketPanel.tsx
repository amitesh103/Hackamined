import { useState, useEffect } from "react";
import { getTickets, type Ticket } from "@/services/api";
import { ClipboardList, CheckCircle2, Clock } from "lucide-react";
import { motion } from "framer-motion";

export function TicketPanel() {
    const [tickets, setTickets] = useState<Ticket[]>([]);

    useEffect(() => {
        // Poll for tickets every 5 seconds
        const fetchTickets = () => getTickets().then(setTickets);
        fetchTickets();
        const interval = setInterval(fetchTickets, 5000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="glass-card p-5 h-full flex flex-col">
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-foreground uppercase tracking-wider">Draft Tickets</h3>
                <ClipboardList className="h-4 w-4 text-muted-foreground" />
            </div>

            <div className="flex-1 overflow-y-auto space-y-3 pr-2 scrollbar-thin">
                {tickets.length === 0 ? (
                    <div className="text-center text-sm text-muted-foreground mt-8">No maintenance tickets required.</div>
                ) : (
                    tickets.map((ticket, i) => (
                        <motion.div
                            key={ticket.id}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: i * 0.05 }}
                            className="p-3 bg-muted/20 border border-border/50 rounded-lg hover:border-primary/30 transition-colors cursor-pointer group"
                        >
                            <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center gap-2">
                                    <span className="font-mono text-xs font-bold text-primary">{ticket.id}</span>
                                    <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-secondary/20 text-secondary uppercase tracking-wider">
                                        {ticket.status}
                                    </span>
                                </div>
                                <span className="text-[10px] text-muted-foreground font-mono">
                                    {new Date(ticket.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                </span>
                            </div>

                            <div className="mb-2">
                                <div className="font-medium text-sm text-foreground mb-0.5">{ticket.suspected_issue}</div>
                                <div className="text-xs text-muted-foreground flex items-start gap-1.5">
                                    <span className="shrink-0 mt-0.5 text-primary">↳</span>
                                    {ticket.recommended_action}
                                </div>
                            </div>

                            <div className="flex items-center justify-between mt-3 pt-3 border-t border-border/30">
                                <span className="font-mono text-[10px] text-muted-foreground bg-background/50 px-2 py-1 rounded">
                                    {ticket.inverter_id}
                                </span>
                                <span className="text-[10px] text-destructive/80 font-mono">
                                    Risk: {(ticket.risk_score * 100).toFixed(0)}%
                                </span>
                            </div>
                        </motion.div>
                    ))
                )}
            </div>
        </div>
    );
}
