import type { NextConfig } from "next";

// All /api/* calls are proxied to the FastAPI strategy engine, so the browser
// never needs CORS and the backend URL is configured in one place.
const API_URL = process.env.PITWALL_API_URL ?? "http://127.0.0.1:8010";

// The first public address, printed on CVs: every page of it now sends visitors
// to the same page on the canonical domain (308, permanent).
const OLD_HOST = "pitwall-ai-f1.vercel.app";
const SITE_URL = "https://pitwallai.ruchirbhatia.com";

const nextConfig: NextConfig = {
  agentRules: false,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/api/:path*` }];
  },
  async redirects() {
    return [{ source: "/:path*", has: [{ type: "host", value: OLD_HOST }], destination: `${SITE_URL}/:path*`, permanent: true }];
  },
};

export default nextConfig;
