import { pct } from "@/lib/api";

const SHADES = ["var(--text-3)", "var(--info)", "#8b5cf6", "var(--warn)"];

/** Probability that the fastest plan has k (more) stops, from the Monte Carlo. */
export function StopOdds({ probs, label = "stops" }: { probs: Record<string, number>; label?: string }) {
  const entries = Object.entries(probs)
    .map(([k, v]) => [Number(k), v] as const)
    .filter(([, v]) => v > 0)
    .sort((a, b) => a[0] - b[0]);
  if (!entries.length) return null;
  return (
    <div>
      <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-surface-3" role="img" aria-label="Stop-count odds">
        {entries.map(([k, v]) => (
          <div key={k} style={{ width: `${v * 100}%`, background: SHADES[k] ?? "var(--text-2)" }} className="h-full border-r-2 border-surface last:border-r-0" title={`${k} ${label}: ${pct(v)}`} />
        ))}
      </div>
      <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1 text-xs text-text-2 tabular">
        {entries.map(([k, v]) => (
          <span key={k} className="inline-flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: SHADES[k] ?? "var(--text-2)" }} />
            {k}-{label.replace(/s$/, "")} {pct(v)}
          </span>
        ))}
      </div>
    </div>
  );
}
