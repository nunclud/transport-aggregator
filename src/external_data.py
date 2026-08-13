"""
Real external flight-price data used to ground the synthetic generator.

Live GIGM / Air Peace / Wakanow feeds aren't publicly scrapable (see README),
so generate_fares.py's price trajectories are synthetic. But *how much fares
climb as departure approaches* doesn't have to be a hand-picked guess: this
module downloads a public flight-price dataset (Kaggle's "Flight Price
Prediction" set, mirrored on GitHub — Indian domestic routes, not Nigerian,
but the booking-window price elasticity it captures is a real, general air
travel pattern) and uses it to calibrate generate_fares.py's URGENCY["air"]
constant instead of that number being invented.
"""
from __future__ import annotations
import os
import json
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXTERNAL_DIR = os.path.join(ROOT, "data", "external")
RAW_CSV = os.path.join(EXTERNAL_DIR, "flight_price_reference.csv")
CALIBRATION_JSON = os.path.join(EXTERNAL_DIR, "calibration.json")

SOURCE_URL = ("https://raw.githubusercontent.com/BelideSaiTeja/"
              "Flight-Price-Prediction/main/Flight%20Price%20Prediction%20dataset.csv")

FALLBACK_URGENCY = 0.55  # generate_fares.py's original hand-picked constant
# Map the reference dataset's near/far price ratio onto the operating range
# price_trajectory()'s p_up formula stays well-behaved in (see generate_fares.py).
URGENCY_RANGE = (0.30, 0.75)


def _download():
    os.makedirs(EXTERNAL_DIR, exist_ok=True)
    urllib.request.urlretrieve(SOURCE_URL, RAW_CSV)


def calibrate_air_urgency() -> float:
    """
    Fit the real relationship between price and days-left-to-book, and turn
    it into an urgency coefficient in the same range generate_fares.py's
    URGENCY dict uses (higher = fares climb harder as departure approaches).
    Falls back to the prototype's original constant if the reference dataset
    can't be fetched (e.g. offline).
    """
    try:
        import pandas as pd
        import numpy as np
        if not os.path.exists(RAW_CSV):
            _download()
        df = pd.read_csv(RAW_CSV, usecols=["days_left", "price"])
        by_days = df.groupby("days_left")["price"].mean()
        days = np.array(sorted(by_days.index))
        far_cut, near_cut = np.quantile(days, 0.75), np.quantile(days, 0.10)
        far_avg = float(by_days.loc[by_days.index >= far_cut].mean())
        near_avg = float(by_days.loc[by_days.index <= near_cut].mean())
        gap = (near_avg - far_avg) / far_avg  # relative price rise close to departure
    except Exception as e:
        print(f"  external_data: reference dataset unavailable ({e}); "
              f"using fallback urgency={FALLBACK_URGENCY}")
        return FALLBACK_URGENCY

    lo, hi = URGENCY_RANGE
    urgency = lo + (hi - lo) * min(1.0, gap / 2.0)

    os.makedirs(EXTERNAL_DIR, exist_ok=True)
    with open(CALIBRATION_JSON, "w") as f:
        json.dump({
            "source": SOURCE_URL,
            "far_price_avg": round(far_avg, 1),
            "near_price_avg": round(near_avg, 1),
            "near_vs_far_price_gap": round(gap, 3),
            "calibrated_air_urgency": round(urgency, 3),
        }, f, indent=2)
    print(f"  external_data: real fares are {gap:.0%} pricier close to departure "
          f"(avg {near_avg:.0f} vs {far_avg:.0f}) -> air urgency = {urgency:.3f}")
    return urgency
