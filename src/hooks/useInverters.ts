import { useState, useEffect, useMemo } from "react";
import { getInverters } from "@/services/api";
import { wsService } from "@/services/websocket";
import type { Inverter } from "@/data/mockData";

export function useInverters() {
    const [inverterMap, setInverterMap] = useState<Record<string, Inverter>>({});
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        let isMounted = true;

        async function init() {
            try {
                const data = await getInverters();
                if (!isMounted) return;

                const initialMap: Record<string, Inverter> = {};
                data.forEach((inv) => (initialMap[inv.id] = inv));
                setInverterMap(initialMap);
                console.log("[INVERTER-DEBUG] Initial inverter count:", Object.keys(initialMap).length);
                setIsLoading(false);
            } catch (err) {
                if (isMounted) {
                    // FIX: Start with empty map, WebSocket will populate real data
                    console.warn("[INVERTER-DEBUG] Init failed, starting with empty map");
                    setInverterMap({});
                    setIsLoading(false);
                }
            }
        }

        init();
        wsService.connect();

        const unsub = wsService.subscribe((incoming) => {
            setInverterMap((prev) => {
                const next = { ...prev };
                incoming.forEach((inv) => {
                    next[inv.id] = { ...next[inv.id], ...inv };
                });
                console.log("[INVERTER-DEBUG] Dashboard inverter count:", Object.keys(next).length);
                return next;
            });
        });

        return () => {
            isMounted = false;
            unsub();
            wsService.disconnect();
        };
    }, []);

    const inverters = useMemo(() => {
        return Object.values(inverterMap).sort((a, b) => a.id.localeCompare(b.id));
    }, [inverterMap]);

    return { inverters, inverterMap, isLoading, error };
}
