import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Nav } from "@/components/Nav";
import { ChatDock } from "@/components/ChatDock";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "PitWall AI — F1 race strategy engine",
  description:
    "Tyre-degradation modelling, exact strategy optimisation and a live pit-wall tracker for the 2026 Formula 1 season.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col">
        <Nav />
        <main className="flex-1">{children}</main>
        <footer className="border-t border-line mt-16">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 py-6 text-xs text-text-3 flex flex-wrap gap-x-6 gap-y-2 justify-between">
            <span>PitWall AI · independent project, not affiliated with Formula 1, FIA or any team.</span>
            <span>Data: FastF1 / F1 live timing, OpenF1. Model outputs are estimates with stated uncertainty.</span>
          </div>
        </footer>
        <ChatDock />
      </body>
    </html>
  );
}
