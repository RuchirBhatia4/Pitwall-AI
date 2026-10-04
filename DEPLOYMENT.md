# Deploying PitWall AI

Two services:

- **API** – FastAPI (`src/api/main.py`) on Render (or any Python host).
- **Frontend** – Next.js (`frontend/`) on Vercel.

The API serves the pre-built JSON in `data/season/2026/` (committed), so it does not need the multi-GB FastF1 cache. Replays and the live tracker fetch what they need on demand.

## 1. API on Render

`render.yaml` defines the service. Settings:

- Build: `pip install -r requirements.txt`
- Start: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
- Env:
  - `CORS_ALLOW_ORIGIN_REGEX=https://.*\.vercel\.app` (and/or `CORS_ALLOW_ORIGINS` with your domain)
  - `ANTHROPIC_API_KEY` – optional, enables LLM answers in the chat (`PITWALL_CHAT_MODEL` to override the model)
  - `OPENF1_USERNAME` / `OPENF1_PASSWORD` – optional, OpenF1 live source
  - `PITWALL_STORAGE_BACKEND`, `DATABASE_URL`, `SUPABASE_*` – only for the legacy CSV/Supabase endpoints

Check `https://<service>.onrender.com/health` and `/api/season`.

The F1 live-timing source keeps a websocket open from the API process; use an instance type that is not put to sleep during a race. For an F1TV-authenticated feed, run `python -m src.pitwall.f1tv_login` where the API runs (the token is stored by FastF1 in the user data directory).

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
