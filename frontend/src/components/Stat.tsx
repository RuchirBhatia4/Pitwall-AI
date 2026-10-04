import clsx from "clsx";
import type { ReactNode } from "react";

export function Stat({ label, value, sub, tone, className }: { label: string; value: ReactNode; sub?: ReactNode; tone?: "good" | "warn" | "bad"; className?: string }) {
  return (
    <div className={clsx("card px-4 py-3", className)}>
      <div className="label">{label}</div>
      <div className={clsx("mt-1 text-2xl font-semibold tabular", tone === "good" && "text-good", tone === "warn" && "text-warn", tone === "bad" && "text-bad")}>
        {value}
      </div>
      {sub && <div className="mt-0.5 text-xs text-text-3">{sub}</div>}
    </div>
  );
}

export function Section({ title, subtitle, right, children, className }: { title: string; subtitle?: ReactNode; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={clsx("card p-4 sm:p-5", className)}>
      <div className="flex flex-wrap items-start justify-between gap-3 mb-4">
        <div>
          <h2 className="font-semibold tracking-tight">{title}</h2>
          {subtitle && <p className="text-sm text-text-3 mt-0.5">{subtitle}</p>}
        </div>
        {right}
      </div>
      {children}
    </section>
  );
}
