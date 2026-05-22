# PitWall AI Live Deployment

This setup uses:
- Supabase for hosted Postgres storage.
- Render for the FastAPI backend.
- Vercel for the React/Vite frontend.

## 1. Create Supabase Project

1. Create a Supabase project.
2. Open the SQL editor.
3. Run `migrations/001_live_storage.sql`.
4. Copy the project database connection string.

Use the connection string in pooler/direct Postgres form as `DATABASE_URL`.

## 2. Configure Backend On Render

Create a Render Web Service from this repository.

Settings:
- Runtime: Python
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`

Environment variables:

```bash
PITWALL_STORAGE_BACKEND=database
DATABASE_URL=your_supabase_postgres_connection_string
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
```

After deploy, verify:

```text
https://your-render-service.onrender.com/health
https://your-render-service.onrender.com/api/storage/status
```

## 3. Seed Current Outputs Into Supabase

After the backend is deployed and `PITWALL_STORAGE_BACKEND=database`, call:

```bash
curl -X POST https://your-render-service.onrender.com/api/storage/sync-current-outputs
```

This copies the current generated CSV outputs into Supabase-backed app storage.

## 4. Configure Frontend On Vercel

Create a Vercel project from the `frontend/` directory.

Settings:
- Framework: Vite
- Build command: `npm run build`
- Output directory: `dist`

Environment variable:

```bash
VITE_API_BASE_URL=https://your-render-service.onrender.com
```

## 5. Custom Domain

Attach your domain to the Vercel frontend project.

Recommended shape:

```text
https://pitwall-ai.com        -> Vercel frontend
https://api.pitwall-ai.com    -> Render backend, optional
```

If you use an API subdomain, update:

```bash
VITE_API_BASE_URL=https://api.pitwall-ai.com
```

## 6. Local Development Modes

CSV mode:

```bash
PITWALL_STORAGE_BACKEND=csv
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Database mode:

```bash
PITWALL_STORAGE_BACKEND=database
DATABASE_URL=your_supabase_postgres_connection_string
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```
