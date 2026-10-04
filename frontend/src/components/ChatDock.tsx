"use client";

import { useEffect, useRef, useState } from "react";
import { usePathname } from "next/navigation";
import clsx from "clsx";
import { Bot, Loader2, MessageSquare, Send, X } from "lucide-react";
import { api } from "@/lib/api";

interface Msg {
  role: "user" | "assistant";
  content: string;
  tools?: string[];
}

const SUGGESTIONS = [
  "When should Russell pit tonight?",
  "Compare one-stop and two-stop for Verstappen",
  "How did Baku's strategies compare with the model?",
  "What happens to Leclerc if there's a Safety Car on lap 20?",
];

/** "Ask the pit wall": a chat grounded in the strategy engine via tool calls. */
export function ChatDock() {
  const [open, setOpen] = useState(false);
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState<string | null>(null);
  const path = usePathname();
  const end = useRef<HTMLDivElement>(null);
  const roundFromPath = path.startsWith("/race/") ? Number(path.split("/")[2]) : null;

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs, busy]);

  const send = async (q: string) => {
    if (!q.trim() || busy) return;
    const next = [...msgs, { role: "user" as const, content: q.trim() }];
    setMsgs(next);
    setText("");
    setBusy(true);
    try {
      const res = await api<{ reply: string; tools_used: string[]; mode: string }>("/api/chat", {
        method: "POST",
        body: JSON.stringify({ messages: next.map(({ role, content }) => ({ role, content })), context: { round: roundFromPath, page: path } }),
      });
      setMode(res.mode);
      setMsgs([...next, { role: "assistant", content: res.reply, tools: res.tools_used }]);
    } catch (e) {
      setMsgs([...next, { role: "assistant", content: `Sorry — ${(e as Error).message}` }]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <button
        onClick={() => setOpen((o) => !o)}
        className="fixed bottom-5 right-5 z-50 flex items-center gap-2 rounded-full bg-accent px-4 py-3 font-semibold shadow-2xl shadow-black/50 hover:brightness-110"
        aria-label="Ask the pit wall"
      >
        {open ? <X size={18} /> : <MessageSquare size={18} />}
        <span className="hidden sm:inline">{open ? "Close" : "Ask the pit wall"}</span>
      </button>
      {open && (
        <div className="fixed bottom-20 right-4 left-4 sm:left-auto z-50 sm:w-[420px] h-[min(620px,75vh)] card flex flex-col shadow-2xl shadow-black/60">
          <div className="flex items-center gap-2 border-b border-line px-4 py-3">
            <Bot size={18} className="text-accent" />
            <div>
              <div className="font-semibold leading-tight">Race engineer</div>
              <div className="text-xs text-text-3">Answers come from the strategy engine{mode ? ` · ${mode}` : ""}</div>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3 text-sm">
            {msgs.length === 0 && (
              <div className="space-y-2">
                <p className="text-text-2">Ask about any 2026 race, a driver&apos;s strategy, or the live race.</p>
                {SUGGESTIONS.map((s) => (
                  <button key={s} onClick={() => send(s)} className="block w-full text-left rounded-lg border border-line px-3 py-2 text-text-2 hover:bg-surface-2">
                    {s}
                  </button>
                ))}
              </div>
            )}
            {msgs.map((m, i) => (
              <div key={i} className={clsx("flex", m.role === "user" ? "justify-end" : "justify-start")}>
                <div className={clsx("max-w-[88%] rounded-2xl px-3 py-2 whitespace-pre-wrap", m.role === "user" ? "bg-accent/90 text-white" : "bg-surface-2 text-text")}>
                  {m.content}
                  {m.tools && m.tools.length > 0 && <div className="mt-1.5 text-[10px] text-text-3">engine calls: {m.tools.join(", ")}</div>}
                </div>
              </div>
            ))}
            {busy && (
              <div className="flex items-center gap-2 text-text-3">
                <Loader2 size={14} className="animate-spin" /> running the numbers…
              </div>
            )}
            <div ref={end} />
          </div>
          <form
            className="border-t border-line p-3 flex gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              send(text);
            }}
          >
            <input
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="e.g. Best strategy for Norris at Monza?"
              className="flex-1 rounded-lg border border-line bg-surface-2 px-3 py-2 text-sm outline-none focus:border-line-strong"
            />
            <button type="submit" disabled={busy} className="rounded-lg bg-accent px-3 disabled:opacity-50" aria-label="Send">
              <Send size={16} />
            </button>
          </form>
        </div>
      )}
    </>
  );
}
