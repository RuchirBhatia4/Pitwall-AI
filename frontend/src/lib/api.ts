"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not json */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

/** Fetch once (and re-fetch when `path` changes). */
export function useApi<T>(path: string | null) {
  const [res, setRes] = useState<{ path: string | null; data: T | null; error: string | null }>({ path: null, data: null, error: null });
  useEffect(() => {
    if (!path) return;
    let alive = true;
    api<T>(path)
      .then((d) => alive && setRes({ path, data: d, error: null }))
      .catch((e: Error) => alive && setRes({ path, data: null, error: e.message }));
    return () => {
      alive = false;
    };
  }, [path]);
  const current = res.path === path;
  return { data: current ? res.data : null, error: current ? res.error : null, loading: !!path && !current };
}

/** Poll an endpoint every `ms` while `enabled`. Keeps the last good payload on errors. */
export function usePoll<T>(path: string | null, ms: number, enabled = true) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<number | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const tick = useCallback(async () => {
    if (!path) return;
    try {
      const d = await api<T>(path);
      setData(d);
      setError(null);
      setUpdatedAt(Date.now());
    } catch (e) {
      setError((e as Error).message);
    }
  }, [path]);
  useEffect(() => {
    if (!enabled || !path) return;
    let stopped = false;
    const loop = async () => {
      await tick();
      if (!stopped) timer.current = setTimeout(loop, ms);
    };
    loop();
    return () => {
      stopped = true;
      if (timer.current) clearTimeout(timer.current);
    };
  }, [tick, ms, enabled, path]);
  return { data, error, updatedAt, refresh: tick, setData };
}

export const pct = (v: number | null | undefined, digits = 0) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

export const secs = (v: number | null | undefined, digits = 1, sign = false) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${sign && v > 0 ? "+" : ""}${v.toFixed(digits)}s`;

export function countdown(seconds: number): string {
  if (seconds <= 0) return "Lights out";
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (d > 0) return `${d}d ${h}h ${m}m`;
  return `${h}h ${String(m).padStart(2, "0")}m ${String(s).padStart(2, "0")}s`;
}
