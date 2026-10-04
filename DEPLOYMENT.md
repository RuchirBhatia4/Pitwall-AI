# Deploying PitWall AI

Live: frontend **https://pitwall-ai-f1.vercel.app** (Vercel, production branch `main`), API **https://pitwall-ai-2i89.onrender.com** (Render, `0.5c-512mb`, branch `main`). Both redeploy automatically on every push to `main`.

Two services:

- **API** – FastAPI (`src/api/main.py`) on Render (or any Python host).
- **Frontend** – Next.js (`frontend/`) on Vercel.

The API serves the pre-built JSON in `data/season/2026/` (committed), so it does not need the multi-GB FastF1 cache. Replays and the live tracker fetch what they need on demand.

## 1. API on Render

`render.yaml` is a Render Blueprint for the API service:

- Plan `0.5c-512mb` (0.5 CPU / 512 MB; the API peaks at ~0.3 GB). The free plan's 0.1 CPU is too slow for live per-lap calls and it sleeps when idle.
- Region `virginia`, next to Vercel's default region, which proxies `/api` to it.
- Python 3.11 via `.python-version` (Render's default is newer than the pinned scientific stack supports).
- Build `pip install -r requirements.txt` (versions pinned), start `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT --workers 1`. Keep **one worker**: the live tracker holds its connections in-process. Each browser sends an `X-Pitwall-Session` id; sessions choose what to watch, while F1 live timing / OpenF1 feeds are shared per race (max 3, stopped after 20 idle minutes).
- Health check `/health`; redeploys on every push to the selected branch.

Steps (Render dashboard):

1. **New → Blueprint**, connect the GitHub repo, pick the branch (`pitwall-strategy-engine` until the PR is merged, then `main`).
2. Fill the prompted optional secrets or leave them empty: `ANTHROPIC_API_KEY` (LLM chat), `OPENF1_USERNAME` / `OPENF1_PASSWORD` (OpenF1 live source).
3. Apply. When the deploy is live, check `https://<service>.onrender.com/health` and `/api/season`.

For an F1TV-authenticated live feed, run `python -m src.pitwall.f1tv_login` in the service shell (the token is stored by FastF1 in the user data directory). Without it the public feed is used.

## 2. Frontend on Vercel

- Root directory: `frontend`
- Framework preset: Next.js (build `next build`)
- Env: `PITWALL_API_URL=https://<service>.onrender.com`

The frontend proxies `/api/*` to `PITWALL_API_URL` through Next.js rewrites, so the browser never calls the API cross-origin.

## 3. Updating after each race

```bash
python -m src.pitwall.prefetch       # new sessions into the FastF1 cache
python -m src.pitwall.build_season   # rebuild rounds, backtest, hybrid model
git add data/season/2026 && git commit -m "Add round N" && git push
```

## Local development

```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8010
cd frontend && npm run dev
```
