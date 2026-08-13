"""
Stage 2 — Normalisation layer (stands in for ksqlDB unifying carrier topics).

Reads every per-carrier raw feed from data/raw/*.csv, cleans/deduplicates,
and unifies them into a single canonical `offers` table plus a `routes`
reference table. In production this is a set of ksqlDB stream queries writing
to PostgreSQL; here we default to SQLite so it runs with zero setup, but the
same code writes to PostgreSQL instead when DATABASE_URL is set (db.py).

Output: aggregator DB (tables: offers, routes)
"""
from __future__ import annotations
import os
import csv
import glob
from sqlalchemy import text

import db

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW_DIR = os.path.join(ROOT, "data", "raw")

CITY = {
    "LOS": "Lagos", "ABV": "Abuja", "ONI": "Onitsha",
    "PHC": "Port Harcourt", "LON": "London",
}

_OFFERS_PK = ("offer_id INTEGER PRIMARY KEY AUTOINCREMENT" if not db.IS_POSTGRES
              else "offer_id SERIAL PRIMARY KEY")

SCHEMA = f"""
DROP TABLE IF EXISTS offers;
DROP TABLE IF EXISTS routes;

CREATE TABLE routes (
    route_id     TEXT PRIMARY KEY,
    origin       TEXT NOT NULL,
    destination  TEXT NOT NULL,
    origin_city  TEXT NOT NULL,
    dest_city    TEXT NOT NULL
);

CREATE TABLE offers (
    {_OFFERS_PK},
    trip_id           TEXT NOT NULL,
    route_id          TEXT NOT NULL,
    carrier           TEXT NOT NULL,
    mode              TEXT NOT NULL,           -- 'road' | 'air'
    departure_date    TEXT NOT NULL,
    departure_hour    INTEGER NOT NULL,
    duration_min      INTEGER NOT NULL,
    captured_at       TEXT NOT NULL,
    days_to_departure INTEGER NOT NULL,
    price_ngn         REAL NOT NULL
);
CREATE INDEX idx_offers_route_date ON offers(route_id, departure_date);
CREATE INDEX idx_offers_trip ON offers(trip_id, captured_at);
"""


def normalize():
    if not db.IS_POSTGRES and os.path.exists(db.AGG_DB):
        os.remove(db.AGG_DB)

    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not files:
        raise SystemExit("No raw feeds found — run generate_fares.py first.")

    seen = set()          # (trip_id, captured_at) dedupe key
    routes = {}
    offers = []
    for path in files:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                key = (r["trip_id"], r["captured_at"])
                if key in seen:
                    continue
                seen.add(key)
                o, d = r["origin"], r["destination"]
                route_id = f"{o}-{d}"
                routes.setdefault(route_id, (o, d, CITY.get(o, o), CITY.get(d, d)))
                offers.append({
                    "trip_id": r["trip_id"], "route_id": route_id,
                    "carrier": r["carrier"], "mode": r["mode"],
                    "departure_date": r["departure_date"],
                    "departure_hour": int(r["departure_hour"]),
                    "duration_min": int(r["duration_min"]),
                    "captured_at": r["captured_at"],
                    "days_to_departure": int(r["days_to_departure"]),
                    "price_ngn": float(r["price_ngn"]),
                })

    engine = db.get_engine()
    with engine.begin() as con:
        for stmt in SCHEMA.strip().split(";"):
            if stmt.strip():
                con.execute(text(stmt))

        con.execute(
            text("INSERT INTO routes VALUES (:route_id, :origin, :destination, :origin_city, :dest_city)"),
            [{"route_id": rid, "origin": o, "destination": d, "origin_city": oc, "dest_city": dc}
             for rid, (o, d, oc, dc) in routes.items()],
        )
        con.execute(
            text("""INSERT INTO offers
                    (trip_id, route_id, carrier, mode, departure_date, departure_hour,
                     duration_min, captured_at, days_to_departure, price_ngn)
                    VALUES (:trip_id, :route_id, :carrier, :mode, :departure_date, :departure_hour,
                            :duration_min, :captured_at, :days_to_departure, :price_ngn)"""),
            offers,
        )
        n_offers = con.execute(text("SELECT COUNT(*) FROM offers")).scalar()
        n_routes = con.execute(text("SELECT COUNT(*) FROM routes")).scalar()
        n_carriers = con.execute(text("SELECT COUNT(DISTINCT carrier) FROM offers")).scalar()

    print(f"  unified {n_offers} offers | {n_carriers} carriers | {n_routes} routes")
    print(f"  wrote {'PostgreSQL (' + db.DATABASE_URL.split('@')[-1] + ')' if db.IS_POSTGRES else db.AGG_DB}")


if __name__ == "__main__":
    print("Stage 2: normalising carrier feeds into canonical `offers` table...")
    normalize()