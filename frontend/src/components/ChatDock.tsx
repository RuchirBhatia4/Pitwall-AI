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
  "Who's leading right now, and on what tyres?",
  "What happens to Leclerc if there's a Safety Car on lap 20?",
];

/** Bold and `code` inside one line of a reply. */
function inline(text: string) {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, i) =>
    part.length > 4 && part.startsWith("**") && part.endsWith("**") ? (
      <strong key={i}>{part.slice(2, -2)}</strong>
    ) : part.length > 2 && part.startsWith("`") && part.endsWith("`") ? (
      <code key={i} className="font-mono text-[0.92em]">
        {part.slice(1, -1)}
      </code>
    ) : (
      part
    ),
  );
}

/** The light Markdown the AI mode writes (bold, `code`, bullets, headings), rendered as elements rather than raw asterisks. */
function Reply({ text }: { text: string }) {
  return (
    <>
      {text.split("\n").map((line, i) => {
        if (!line.trim()) return <div key={i} className="h-2" />;
        const bullet = /^\s*(?:[-*•]|\d+[.)])\s+/.exec(line);
        const heading = /^#{1,6}\s+/.exec(line);
        const body = inline(line.slice((bullet ?? heading)?.[0].length ?? 0));
        if (bullet) {
          const mark = /\d/.test(bullet[0]) ? bullet[0].trim() : "•";
          return (
            <div key={i} className="flex gap-1.5 pl-1">
              <span className="shrink-0 text-text-3">{mark}</span>
              <span>{body}</span>
            </div>
          );
        }
        return <div key={i}>{heading ? <strong>{body}</strong> : body}</div>;
      })}
    </>
  );
}

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
        className="fixed bottom-5 right-5 z-50 flex items-center gap-2 rounded-full bg-accent px-4 py-3 font-semibold text-on-accent shadow-2xl shadow-black/50 hover:brightness-110"
        aria-label="Ask the pit wall"
      >
        {open ? <X size={18} /> : <MessageSquare size={18} />}
        <span className="hidden sm:inline">{open ? "Close" : "Ask the pit wall"}</span>
      </button>
      {open && (
        <div className="fixed bottom-20 right-4 left-4 sm:left-auto z-50 sm:w-[420px] h-[min(620px,75vh)] card flex flex-col shadow-2xl shadow-black/60">
          <div className="flex items-center gap-2 border-b border-line px-4 py-3">
            <Bot size={18} className="text-accent-ink" />
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
                <div className={clsx("max-w-[88%] rounded-2xl px-3 py-2 whitespace-pre-wrap", m.role === "user" ? "bg-accent/90 text-on-accent" : "bg-surface-2 text-text")}>
                  {m.role === "assistant" ? <Reply text={m.content} /> : m.content}
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
            <button type="submit" disabled={busy} className="rounded-lg bg-accent px-3 text-on-accent disabled:opacity-50" aria-label="Send">
              <Send size={16} />
            </button>
          </form>
        </div>
      )}
    </>
  );
}
