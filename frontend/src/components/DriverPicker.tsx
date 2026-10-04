"use client";

import clsx from "clsx";

export interface PickerDriver {
  driver: string;
  team: string;
  color: string;
  sub?: string | number | null;
}

export function DriverPicker({ drivers, value, onChange }: { drivers: PickerDriver[]; value: string | null; onChange: (d: string) => void }) {
  return (
    <div className="flex flex-wrap gap-1.5" role="listbox" aria-label="Choose driver">
      {drivers.map((d) => (
        <button
          key={d.driver}
          role="option"
          aria-selected={value === d.driver}
          onClick={() => onChange(d.driver)}
          className={clsx(
            "group flex items-center gap-2 rounded-md border px-2.5 py-1.5 text-sm transition-colors",
            value === d.driver ? "border-text-2 bg-surface-3" : "border-line bg-surface hover:border-line-strong",
          )}
          title={d.team}
        >
          <span className="h-3.5 w-1 rounded-full" style={{ background: d.color }} />
          <span className="font-semibold tracking-wide">{d.driver}</span>
          {d.sub != null && <span className="tabular text-xs text-text-3">{d.sub}</span>}
        </button>
      ))}
    </div>
  );
}
