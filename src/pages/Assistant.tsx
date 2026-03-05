import { ChatWindow } from "@/components/chat/ChatWindow";
import { AppLayout } from "@/components/layout/AppLayout";

export default function Assistant() {
  return (
    <AppLayout>
      <div className="flex flex-col h-[calc(100vh-3.5rem)]">
        <div className="px-4 lg:px-6 pt-4 pb-2">
          <h1 className="text-2xl font-bold tracking-tight">AI Assistant</h1>
          <p className="text-sm text-muted-foreground font-mono">RAG-powered diagnostics · Natural language queries</p>
        </div>
        <div className="flex-1 min-h-0">
          <ChatWindow />
        </div>
      </div>
    </AppLayout>
  );
}
