# PitWall AI — frontend

Next.js 16 (App Router, TypeScript, Tailwind CSS 4, Recharts).

```bash
npm install
npm run dev      # http://localhost:3000
```

All `/api/*` requests are proxied to the FastAPI strategy engine at `PITWALL_API_URL`
(default `http://127.0.0.1:8010`) via `next.config.ts` rewrites.

Routes: `/` season, `/race/[round]` strategy predictor, `/live` live pit wall, `/methodology`.
See the repository README for the model and deployment.
