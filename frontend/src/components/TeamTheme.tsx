"use client";

import { createContext, useCallback, useContext, useEffect, useRef, useState, useSyncExternalStore } from "react";
import clsx from "clsx";
import { Check, ChevronDown, X } from "lucide-react";
import { NO_TEAM, TEAM_ASKED_KEY, TEAM_KEY, TEAMS, teamById, teamIdOf, type TeamTheme } from "@/lib/teams";

interface TeamState {
  /** Theme id, or NO_TEAM. */
  team: string;
  theme: TeamTheme | null;
  setTeam: (id: string) => void;
  /** Is this timing-data team name the visitor's team? */
  isMine: (teamName: string | null | undefined) => boolean;
  /** Has the visitor answered (or dismissed) the "who do you support" prompt? */
  asked: boolean;
  dismiss: () => void;
  ready: boolean;
}

const TeamContext = createContext<TeamState>({
  team: NO_TEAM,
  theme: null,
  setTeam: () => undefined,
  isMine: () => false,
  asked: true,
  dismiss: () => undefined,
  ready: false,
});

export const useTeam = () => useContext(TeamContext);

// The choice lives in the visitor's browser. Storage can be missing or blocked
// (private windows): the choice then lasts for the visit only.
function load(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}
function save(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {}
}

// A tiny external store ("team|asked") read with useSyncExternalStore, so the
// server render and hydration agree and other tabs stay in step.
const SERVER = "server";
const listeners = new Set<() => void>();
let current: string | null = null;
const read = () => `${load(TEAM_KEY) ?? NO_TEAM}|${load(TEAM_ASKED_KEY) ?? ""}`;
const snapshot = () => (current ??= read());
function subscribe(cb: () => void) {
  const onStorage = () => {
    current = read();
    cb();
  };
  listeners.add(cb);
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(cb);
    window.removeEventListener("storage", onStorage);
  };
}
function write(team: string) {
  save(TEAM_KEY, team);
  save(TEAM_ASKED_KEY, "1");
  current = `${team}|1`;
  listeners.forEach((l) => l());
}

export function TeamProvider({ children }: { children: React.ReactNode }) {
  const snap = useSyncExternalStore(subscribe, snapshot, () => SERVER);
  const ready = snap !== SERVER;
  const [saved, savedAsked] = ready ? snap.split("|") : [NO_TEAM, "1"];
  const team = teamById(saved) ? saved : NO_TEAM;
  const asked = savedAsked === "1" || team !== NO_TEAM;

  const setTeam = useCallback((id: string) => {
    const valid = teamById(id) ? id : NO_TEAM;
    if (valid === NO_TEAM) delete document.documentElement.dataset.team;
    else document.documentElement.dataset.team = valid;
    write(valid);
  }, []);

  const dismiss = useCallback(() => write(team), [team]);

  const isMine = useCallback((name: string | null | undefined) => team !== NO_TEAM && teamIdOf(name) === team, [team]);

  return (
    <TeamContext.Provider value={{ team, theme: teamById(team), setTeam, isMine, asked, dismiss, ready }}>{children}</TeamContext.Provider>
  );
}

function Dot({ color, size = 10 }: { color: string; size?: number }) {
  return <span className="shrink-0 rounded-full ring-1 ring-white/15" style={{ background: color, width: size, height: size }} />;
}

/** Header control: the visitor's team. */
export function TeamPicker() {
  const { theme, team, setTeam, ready } = useTeam();
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (e instanceof KeyboardEvent ? e.key === "Escape" : !box.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", close);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", close);
    };
  }, [open]);

  const choose = (id: string) => {
    setTeam(id);
    setOpen(false);
  };

  return (
    <div ref={box} className="relative ml-auto shrink-0">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={theme ? `Your team: ${theme.name}. Change team` : "Pick your team"}
        className={clsx(
          "flex items-center gap-2 rounded-md border border-line px-2.5 py-1.5 text-sm text-text-2 hover:border-line-strong hover:text-text",
          !ready && "invisible",
        )}
      >
        <Dot color={theme ? (theme.mark ?? theme.accent) : "var(--text-3)"} />
        <span className="hidden sm:inline whitespace-nowrap">{theme ? theme.name : "Your team"}</span>
        <ChevronDown size={14} />
      </button>
      {open && (
        <div role="listbox" aria-label="Your team" className="absolute right-0 top-full mt-2 w-56 rounded-xl border border-line bg-surface p-1.5 shadow-2xl shadow-black/50">
          {[...TEAMS.map((t) => ({ id: t.id, name: t.name, color: t.mark ?? t.accent })), { id: NO_TEAM, name: "No team", color: "var(--text-3)" }].map((t) => (
            <button
              key={t.id}
              role="option"
              aria-selected={team === t.id}
              onClick={() => choose(t.id)}
              className={clsx("flex w-full items-center gap-2.5 rounded-md px-2.5 py-1.5 text-left text-sm", team === t.id ? "bg-surface-3 text-text" : "text-text-2 hover:bg-surface-2 hover:text-text")}
            >
              <Dot color={t.color} />
              <span className="flex-1">{t.name}</span>
              {team === t.id && <Check size={14} />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/** One-time question for new visitors, under the header. */
export function TeamPrompt() {
  const { ready, asked, setTeam, dismiss } = useTeam();
  if (!ready || asked) return null;
  return (
    <div className="border-b border-line bg-surface">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3 sm:px-6">
        <div className="text-sm">
          <span className="font-semibold">Who do you support?</span>{" "}
          <span className="text-text-2">Your team colours the site and its drivers are highlighted.</span>
        </div>
        <div className="flex flex-1 flex-wrap gap-1.5">
          {TEAMS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTeam(t.id)}
              className="flex items-center gap-1.5 rounded-md border border-line bg-surface-2 px-2 py-1 text-xs text-text-2 hover:border-line-strong hover:text-text"
            >
              <Dot color={t.mark ?? t.accent} size={8} />
              {t.name}
            </button>
          ))}
        </div>
        <button onClick={dismiss} className="flex items-center gap-1 text-xs text-text-3 hover:text-text" aria-label="Not now">
          Not now <X size={14} />
        </button>
      </div>
    </div>
  );
}

/** Small "your team" mark next to a driver. */
export function YourTeamTag() {
  return <span className="rounded border border-[var(--accent-2)] px-1 text-[10px] font-semibold uppercase tracking-wide text-[var(--accent-2)]">Your team</span>;
}
