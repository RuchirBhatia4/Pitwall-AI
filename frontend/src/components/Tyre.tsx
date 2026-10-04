import clsx from "clsx";

export const COMPOUND_COLOR: Record<string, string> = {
  SOFT: "var(--soft)",
  MEDIUM: "var(--medium)",
  HARD: "var(--hard)",
  INTERMEDIATE: "var(--inter)",
  WET: "var(--wet)",
};
export const COMPOUND_LETTER: Record<string, string> = { SOFT: "S", MEDIUM: "M", HARD: "H", INTERMEDIATE: "I", WET: "W" };

export const compoundName = (c?: string | null) => (c ? c.charAt(0) + c.slice(1).toLowerCase() : "—");

/** Tyre icon: coloured ring + letter, so identity never relies on colour alone. */
export function TyreBadge({ compound, size = 22, age, className }: { compound: string; size?: number; age?: number | null; className?: string }) {
  const color = COMPOUND_COLOR[compound] ?? "var(--text-3)";
  return (
    <span className={clsx("inline-flex items-center gap-1.5", className)} title={`${compoundName(compound)}${age != null ? `, ${age} laps old` : ""}`}>
      <span
        className="grid place-items-center rounded-full font-bold bg-black/60"
        style={{ width: size, height: size, border: `${Math.max(2, size / 7)}px solid ${color}`, fontSize: size * 0.45, color: "var(--text)" }}
      >
        {COMPOUND_LETTER[compound] ?? "?"}
      </span>
      {age != null && <span className="tabular text-xs text-text-2">{age}</span>}
    </span>
  );
}

export function SequencePills({ sequence }: { sequence: string[] }) {
  return (
    <span className="inline-flex items-center gap-1">
      {sequence.map((c, i) => (
        <span key={i} className="inline-flex items-center gap-1">
          {i > 0 && <span className="text-text-3 text-xs">›</span>}
          <TyreBadge compound={c} size={18} />
        </span>
      ))}
    </span>
  );
}
