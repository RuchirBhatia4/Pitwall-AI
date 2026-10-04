"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";
import { Activity, Flag, FlaskConical, Radio } from "lucide-react";

const links = [
  { href: "/", label: "Season", icon: Flag },
  { href: "/live", label: "Live pit wall", icon: Radio },
  { href: "/methodology", label: "Method & validation", icon: FlaskConical },
];

export function Nav() {
  const path = usePathname();
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-bg/85 backdrop-blur">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 h-14 flex items-center gap-6">
        <Link href="/" className="flex shrink-0 items-center gap-2 whitespace-nowrap font-semibold tracking-tight">
          <span className="grid place-items-center h-7 w-7 rounded-md bg-accent">
            <Activity size={16} strokeWidth={2.5} />
          </span>
          <span>
            PitWall <span className="text-accent">AI</span>
          </span>
        </Link>
        <nav className="flex items-center gap-1 overflow-x-auto">
          {links.map(({ href, label, icon: Icon }) => {
            const active = href === "/" ? path === "/" || path.startsWith("/race") : path.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={clsx(
                  "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm whitespace-nowrap transition-colors",
                  active ? "bg-surface-3 text-text" : "text-text-2 hover:text-text hover:bg-surface-2",
                )}
              >
                <Icon size={15} />
                {label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
