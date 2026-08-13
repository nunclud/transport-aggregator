# NaijaFare Multi-Modal Ticketing & Dynamic Pricing (Project 03)

A **runnable, end-to-end reference implementation** of the "Skyscanner for Nigeria"
aggregator: it ingests fares from multiple road and air carriers, unifies them into
a single `offers` feed, predicts whether each fare will **rise in the next 24 hours**
with a LightGBM model, and serves cheapest/fastest search + natural-language search +
booking through a FastAPI web app.

Because live GIGM / Air Peace / Wakanow feeds aren't publicly available (and scraping
real carrier sites is fragile and often blocked), the ingestion stage produces
**realistic synthetic fare trajectories** — calibrated against a real public
flight-price dataset rather than hand-picked (see `src/external_data.py`). Every
downstream stage: normalization, modeling, serving, UI - is real and production-shaped,
and several pieces of the production architecture (Postgres, Redis, Ray, MLflow,
Claude) are already wired in behind environment variables, not just documented as a
future swap — see [Optional production backends](#optional-production-backends).

## Quick start

## Application Test page
## frontend for testing deployed on vercel https://transport-aggregator-gp3.vercel.app/


```bash
pip install -r requirements.txt
python run_all.py            # generate -> normalize -> train -> serve on :8000
# then open http://127.0.0.1:8000 (for running on local machine)
```

Build the data + model without starting the server:

```bash
python run_all.py --no-serve
python src/api.py            # start the API separately
```

## Pipeline stages

| Stage | File | Production equivalent | Status here |
|---|---|---|---|
| 1. Ingest fares | `src/generate_fares.py` | Per-carrier **Kafka source connectors** (REST poll + Scrapy/Playwright scrapers), one topic per carrier | Synthetic, but price-rise dynamics for air are calibrated from a real public fare dataset (`external_data.py`) |
| 2. Normalize | `src/normalize.py` | **ksqlDB** stream queries unifying carrier topics into one `offers` topic, sunk to **PostgreSQL** | Runs on SQLite by default, on **real PostgreSQL** when `DATABASE_URL` is set — same code path (`db.py`) |
| 3. Model | `src/train_model.py` + `src/features.py` | **LightGBM** "price will rise in 24h" classifier, distributed with **Ray**, tracked in MLflow | Distributes time-fold cross-validation across **Ray** workers and logs every run to **MLflow** when both are installed; falls back to a plain single-split run otherwise |
| 4. Serve | `src/api.py` + `src/ui/` | **FastAPI** search + booking-orchestrator, **Redis**-cached reads, **React/Streamlit + R Shiny** UI, **Anthropic Claude** for NL search | FastAPI reads/writes cached through **Redis** when `REDIS_URL` is set (`cache.py`); NL search auto-routes through **Claude** when `ANTHROPIC_API_KEY` is set; UI is still a static page, not React/Shiny |

## Optional production backends

Nothing below is required — every one of these degrades gracefully to the zero-setup
prototype behavior if its env var is unset or unreachable.

| Backend | Env var | What it changes |
|---|---|---|
| PostgreSQL | `DATABASE_URL` | `normalize.py`, `train_model.py`, `api.py` read/write Postgres instead of SQLite (`data/aggregator.db`). A free instance (Neon, Supabase) works fine. |
| Redis | `REDIS_URL` | `/search`, `/search/nl`, `/routes`, `/metrics` are cached (60–300s TTL) instead of hitting the DB every request. A free instance (Upstash) works fine. |
| Anthropic Claude | `ANTHROPIC_API_KEY` | `/search/nl` parses free-text queries through Claude instead of the rule-based parser. Set `USE_CLAUDE=0` to force the rule-based parser even with a key present. |
| Ray + MLflow | *(installed, not env-gated)* | `train_model.py` runs a 4-fold time-series cross-validation across Ray workers and logs params/metrics/artifacts to a local MLflow run (`src/mlruns/`). Install via `pip install -r requirements-train.txt` — kept **out of `requirements.txt`** deliberately, since that's what Vercel's `@vercel/python` builder installs for `api/index.py`, and the API never imports either package; pulling Ray+MLflow's dependency trees into the deploy bundle would bloat/risk the serverless build for a training-only feature. **Ray also has no Windows wheel for Python 3.13** — use a Python 3.10–3.12 environment for training if you're on Windows (e.g. `py -3.12 -m venv .venv312`). |

Verified locally: `normalize.py`/`train_model.py`/`api.py` against a real Postgres
container, and the Redis cache-aside logic against a fake Redis client — both took the
DATABASE_URL/REDIS_URL path correctly and fell back cleanly when unset.

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Search web UI |
| GET | `/health` | Readiness check |
| GET | `/routes` | Available routes |
| GET | `/metrics` | Model AUC / accuracy / feature importance |
| GET | `/search?origin=LOS&destination=ABV&sort=cheapest&mode=air` | Structured search |
| POST | `/search/nl` `{"q":"cheapest Lagos to Abuja tomorrow morning"}` | Natural-language search |
| POST | `/book` `{"trip_id":"..."}` | Simulated carrier booking |

## The model

Target: for each fare snapshot, **will the price be higher at the next daily snapshot (~24h)?**
Features: current price, days-to-departure, duration, departure hour, day-of-week,
mode (air/road), price relative to the route median, recent 3-snapshot price trend,
and carrier cheapness rank. A time-aware 80/20 split trains LightGBM; the latest
snapshot per trip is scored and written to a `predictions` table the API serves.

Typical run: **AUC ≈ 0.63, accuracy ≈ 70%** on held-out data, with `days_to_departure`
and `price_trend_3d` the dominant features — i.e. fares are most predictable close to
departure and when they already have upward momentum. (These are synthetic-data numbers;
real fare feeds would re-fit the same pipeline.)

## Plugging in real data

1. Replace `generate_fares.py` with Kafka source connectors / scrapers that write the
   same raw columns per carrier.
2. Set `DATABASE_URL` to point `normalize.py`/`train_model.py`/`api.py` at PostgreSQL —
   no code changes needed, see [Optional production backends](#optional-production-backends).
3. Add a scheduler (cron/Airflow) to re-run stages 1–3 on an interval; keep stage 4 running.
4. Set `ANTHROPIC_API_KEY` to route NL search through Claude automatically
   (see `src/nl_search.py:parse_with_claude`).
5. Set `REDIS_URL` to cache serving reads instead of hitting Postgres every request.

## Config

| Env var | Default | Purpose |
|---|---|---|
| `AGG_DB` | `data/aggregator.db` | SQLite path, used when `DATABASE_URL` isn't set. Change this if your filesystem doesn't support SQLite locking (e.g. some network mounts). |
| `DATABASE_URL` | *(unset → SQLite)* | PostgreSQL DSN. Switches `normalize.py`/`train_model.py`/`api.py` to Postgres. |
| `REDIS_URL` | *(unset → no cache)* | Redis DSN. Enables caching for `/search`, `/search/nl`, `/routes`, `/metrics`. |
| `ANTHROPIC_API_KEY` | *(unset → rule-based parser)* | Enables Claude-powered NL search automatically. |
| `USE_CLAUDE` | *(auto)* | Set to `0` to force the rule-based NL parser even when a Claude key is present. |

## Layout

```
transport_aggregator/
├── run_all.py            # one-command end-to-end runner
├── requirements.txt
├── README.md
└── src/
    ├── generate_fares.py # stage 1 — synthetic per-carrier feeds
    ├── external_data.py  # calibrates fare urgency from a real public flight-price dataset
    ├── normalize.py      # stage 2 — unify into canonical offers (SQLite or Postgres)
    ├── db.py             # shared SQLAlchemy engine (SQLite default, Postgres via DATABASE_URL)
    ├── features.py       # shared feature engineering + label
    ├── train_model.py    # stage 3 — LightGBM training + scoring (Ray CV + MLflow tracking)
    ├── nl_search.py      # NL query parser (rule-based, or Claude when ANTHROPIC_API_KEY is set)
    ├── api.py            # stage 4 — FastAPI service
    ├── cache.py           # Redis cache-aside helper (REDIS_URL), no-op fallback otherwise
    └── ui/index.html      # single-page search UI
```