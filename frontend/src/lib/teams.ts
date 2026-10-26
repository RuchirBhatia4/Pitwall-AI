// Favourite-team themes: a visitor's team recolours the site (accent, a second
// livery colour, a background tinted to match) and its drivers are highlighted.
// Team names and colours only, no logos. Tyre colours (Pirelli convention) and
// status colours (good / warn / bad / alert) never change with the team.
// Every accent reads at >= 4.5:1 as text on its background and under its
// button text; the reds, mid blue and green therefore take near-black button
// text (no red is bright enough for one and dark enough for white text).
// A livery with several colours can split the roles: `accent` fills buttons,
// `ink` colours accent text and labels, `mark` highlights the team's drivers
// (Red Bull: red buttons, yellow writing, blue drivers on a navy background).

export interface TeamTheme {
  id: string;
  name: string;
  /** Lower-case fragments that identify the team in timing-data team names. */
  match: string[];
  bg: string;
  surface: string;
  surface2: string;
  surface3: string;
  line: string;
  lineStrong: string;
  text: string;
  text2: string;
  text3: string;
  /** Fills: buttons, the logo, chat bubbles. */
  accent: string;
  /** Second livery colour, for small marks (your-team tags). */
  accent2: string;
  /** Text on an accent fill. */
  onAccent: string;
  /** Accent-coloured text (default: accent). */
  ink?: string;
  /** Small uppercase labels (default: text3). */
  label?: string;
  /** Highlight of the team's drivers (default: accent). */
  mark?: string;
}

export const NO_TEAM = "none";
export const TEAM_KEY = "pitwall-team";
export const TEAM_ASKED_KEY = "pitwall-team-asked";

export const TEAMS: TeamTheme[] = [
  { id: "mercedes", name: "Mercedes", match: ["mercedes"], bg: "#060b0b", surface: "#0d1514", surface2: "#13201e", surface3: "#1a2b28", line: "#1e302d", lineStrong: "#2a4440", text: "#eef6f5", text2: "#a4b8b5", text3: "#677a77", accent: "#00d7b6", accent2: "#c3cbd1", onAccent: "#00241e" },
  { id: "ferrari", name: "Ferrari", match: ["ferrari"], bg: "#0d0606", surface: "#160c0c", surface2: "#211111", surface3: "#2c1718", line: "#33191a", lineStrong: "#47242a", text: "#fbf2f2", text2: "#c4a9aa", text3: "#85696a", accent: "#ed1131", accent2: "#ffe100", onAccent: "#1a0003" },
  { id: "mclaren", name: "McLaren", match: ["mclaren"], bg: "#0b0b0c", surface: "#141416", surface2: "#1c1c20", surface3: "#25252b", line: "#2b2b31", lineStrong: "#3a3a42", text: "#f5f4f2", text2: "#b6b2ab", text3: "#7a766f", accent: "#f47600", accent2: "#b8bcc2", onAccent: "#251100" },
  { id: "red-bull", name: "Red Bull", match: ["red bull"], bg: "#070a17", surface: "#0d1224", surface2: "#131a33", surface3: "#1a2342", line: "#1f2848", lineStrong: "#2c3862", text: "#eef1fb", text2: "#a7b1cf", text3: "#687293", accent: "#cc1e4a", accent2: "#ffc906", onAccent: "#ffffff", ink: "#ffc906", label: "#ffc906", mark: "#4781d7" },
  { id: "aston-martin", name: "Aston Martin", match: ["aston martin"], bg: "#050d0a", surface: "#0b1712", surface2: "#11211a", surface3: "#172c23", line: "#1b3328", lineStrong: "#274838", text: "#eef6f2", text2: "#a3baae", text3: "#65796e", accent: "#229971", accent2: "#cedc00", onAccent: "#03140d" },
  { id: "alpine", name: "Alpine", match: ["alpine"], bg: "#050a13", surface: "#0b1220", surface2: "#111a2d", surface3: "#17233b", line: "#1c2944", lineStrong: "#29395d", text: "#eef3fb", text2: "#a6b2c9", text3: "#67728b", accent: "#00a1e8", accent2: "#ff87bc", onAccent: "#00202e" },
  { id: "williams", name: "Williams", match: ["williams"], bg: "#050918", surface: "#0a1126", surface2: "#101a36", surface3: "#162245", line: "#1b2850", lineStrong: "#28386b", text: "#eef2fc", text2: "#a6b1d1", text3: "#677196", accent: "#2f7ae6", accent2: "#00a3e0", onAccent: "#020617" },
  { id: "racing-bulls", name: "Racing Bulls", match: ["racing bulls", "visa cash app", "rb f1"], bg: "#08091a", surface: "#0e1127", surface2: "#151a38", surface3: "#1c2348", line: "#222a52", lineStrong: "#313c6f", text: "#f0f2fc", text2: "#aab1d4", text3: "#6a7199", accent: "#6c98ff", accent2: "#f0f2fc", onAccent: "#0b1433" },
  { id: "haas", name: "Haas", match: ["haas"], bg: "#0a0a0a", surface: "#131313", surface2: "#1b1b1c", surface3: "#242426", line: "#2a2a2c", lineStrong: "#3a3a3d", text: "#f4f4f5", text2: "#b0b0b4", text3: "#737378", accent: "#dadde0", accent2: "#ff4058", onAccent: "#111111" },
  { id: "audi", name: "Audi", match: ["audi", "sauber"], bg: "#0b0b0c", surface: "#141415", surface2: "#1c1c1e", surface3: "#252528", line: "#2b2b2e", lineStrong: "#3b3b3f", text: "#f4f4f5", text2: "#b3b3b8", text3: "#75757b", accent: "#f50537", accent2: "#c5c9cc", onAccent: "#1a0006" },
  { id: "cadillac", name: "Cadillac", match: ["cadillac"], bg: "#09090a", surface: "#121213", surface2: "#1a1a1c", surface3: "#232326", line: "#29292c", lineStrong: "#39393d", text: "#f4f4f4", text2: "#b2b2b2", text3: "#747474", accent: "#e8e8e8", accent2: "#909090", onAccent: "#111111" },
];

export const teamById = (id: string | null | undefined) => TEAMS.find((t) => t.id === id) ?? null;

/** Theme id of a team name from the timing data ("Haas F1 Team" -> "haas"). */
export function teamIdOf(name: string | null | undefined): string | null {
  const n = (name ?? "").toLowerCase();
  return TEAMS.find((t) => t.match.some((m) => n.includes(m)))?.id ?? null;
}

/** CSS overriding the :root tokens for each team (html[data-team=...]). */
export function teamThemeCss(): string {
  return TEAMS.map(
    (t) =>
      `html[data-team="${t.id}"]{--bg:${t.bg};--surface:${t.surface};--surface-2:${t.surface2};--surface-3:${t.surface3};` +
      `--line:${t.line};--line-strong:${t.lineStrong};--text:${t.text};--text-2:${t.text2};--text-3:${t.text3};` +
      `--accent:${t.accent};--accent-2:${t.accent2};--on-accent:${t.onAccent}` +
      (t.ink ? `;--accent-ink:${t.ink}` : "") +
      (t.label ? `;--label:${t.label}` : "") +
      (t.mark ? `;--mark:${t.mark}` : "") +
      "}",
  ).join("\n");
}

/** Runs before first paint: apply the saved team so the page never flashes the default theme. */
export const teamBootScript = `try{var t=localStorage.getItem("${TEAM_KEY}");if(t&&t!=="${NO_TEAM}")document.documentElement.dataset.team=t}catch(e){}`;
