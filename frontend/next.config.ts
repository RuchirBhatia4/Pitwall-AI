import type { NextConfig } from "next";

// All /api/* calls are proxied to the FastAPI strategy engine, so the browser
// never needs CORS and the backend URL is configured in one place.
const API_URL = process.env.PITWALL_API_URL ?? "http://127.0.0.1:8010";

const nextConfig: NextConfig = {
  agentRules: false,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/api/:path*` }];
  },
};

export default nextConfig;
