import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Nav } from "@/components/Nav";
import { ChatDock } from "@/components/ChatDock";
import { FooterLogo } from "@/components/Brand";
import { TeamPrompt, TeamProvider } from "@/components/TeamTheme";
import { teamBootScript, teamThemeCss } from "@/lib/teams";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

// Canonical address; the old pitwall-ai-f1.vercel.app redirects here (next.config.ts).
const SITE_URL = "https://pitwallai.ruchirbhatia.com";
const TITLE = "PitWall AI — F1 race strategy engine";
const DESCRIPTION = "Tyre-degradation modelling, exact strategy optimisation and a live pit-wall tracker for the 2026 Formula 1 season.";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: TITLE,
  description: DESCRIPTION,
  // app/favicon.ico is the fallback; the simplified chequered mark is the small icon.
  icons: { icon: [{ url: "/branding/icons/favicon-vector-simplified.svg", type: "image/svg+xml" }], apple: "/branding/icons/apple-touch-icon-180x180.png" },
  manifest: "/site.webmanifest",
  openGraph: {
    type: "website",
    url: "/",
    siteName: "PitWall AI",
    title: TITLE,
    description: DESCRIPTION,
    images: [{ url: "/branding/social/open-graph-1200x630.png", width: 1200, height: 630, alt: "PitWall AI — Predict. Pit. Win." }],
  },
  twitter: { card: "summary_large_image", title: TITLE, description: DESCRIPTION, images: ["/branding/social/twitter-card-1200x600.png"] },
};

export const viewport: Viewport = { themeColor: "#080b0d" };

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    // suppressHydrationWarning: the boot script sets data-team on <html> before React hydrates.
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`} suppressHydrationWarning>
      <head>
        <style dangerouslySetInnerHTML={{ __html: teamThemeCss() }} />
        <script dangerouslySetInnerHTML={{ __html: teamBootScript }} />
      </head>
      <body className="min-h-full flex flex-col">
        <TeamProvider>
          <Nav />
          <TeamPrompt />
          <main className="flex-1">{children}</main>
          <footer className="border-t border-line mt-16">
            <div className="mx-auto max-w-7xl px-4 sm:px-6 py-6 text-xs text-text-3 flex flex-wrap gap-x-6 gap-y-2 justify-between">
              <span className="flex items-center gap-3">
                <FooterLogo />
                Independent project, not affiliated with Formula 1, FIA or any team.
              </span>
              <span>Data: FastF1 / F1 live timing, OpenF1. Model outputs are estimates with stated uncertainty.</span>
            </div>
          </footer>
          <ChatDock />
        </TeamProvider>
      </body>
    </html>
  );
}
