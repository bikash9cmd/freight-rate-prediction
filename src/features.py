"""Data cleaning and feature engineering for the freight rate model."""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_MI = 3958.8
# Road distance is normally 1.05x-1.6x straight-line distance on these lanes.
ROAD_FACTOR_MIN, ROAD_FACTOR_MAX = 1.05, 1.60
CAT_COLS = ["equipment"]
FEATURES = [
    "log_distance",
    "straight_line_mi",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "lat_change",
    "lon_change",
    "weight",
    "market_index",
    "daily_market_index",
    "quote_signal",
    "equipment",
]


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(a))


def clean(df: pd.DataFrame, road_factor: float, daily_index: pd.Series) -> pd.DataFrame:
    """Fix the data-quality issues found during exploration.

    road_factor and daily_index are learned on training data only and passed in,
    so validation rows never leak into the fitted cleaning rules.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    # 1. Weight: a few values are negative (sign flipped) -> use absolute value.
    #    Missing weights are left as NaN; the gradient boosted model handles them natively.
    out["weight"] = out["weight"].abs()

    # 2. Distance: some rows have impossible distances (e.g. clipped at 70 mi for a
    #    ~600 mi lane). If road/straight-line ratio is implausible, rebuild it from
    #    coordinates using the typical road factor.
    out["straight_line_mi"] = haversine_miles(
        out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"]
    )
    ratio = out["distance"] / out["straight_line_mi"]
    bad = (ratio < ROAD_FACTOR_MIN) | (ratio > ROAD_FACTOR_MAX)
    out["distance_was_fixed"] = bad
    out.loc[bad, "distance"] = out.loc[bad, "straight_line_mi"] * road_factor

    # 3. market_index: a daily market level plus load-level noise. Missing values
    #    are filled with that day's average index.
    out["daily_market_index"] = out["date"].map(daily_index)
    out["market_index"] = out["market_index"].fillna(out["daily_market_index"])
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["log_distance"] = np.log(df["distance"])
    x["straight_line_mi"] = df["straight_line_mi"]
    for c in [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "weight",
        "market_index",
        "daily_market_index",
        "quote_signal",
    ]:
        x[c] = df[c]
    x["lat_change"] = df["delivery_lat"] - df["pickup_lat"]
    x["lon_change"] = df["delivery_lon"] - df["pickup_lon"]
    x["equipment"] = df["equipment"].astype(
        pd.CategoricalDtype(["Dry Van", "Reefer", "Flatbed"])
    )
    return x[FEATURES]


def fit_cleaning_stats(train: pd.DataFrame) -> tuple[float, pd.Series]:
    straight = haversine_miles(
        train["pickup_lat"],
        train["pickup_lon"],
        train["delivery_lat"],
        train["delivery_lon"],
    )
    ratio = train["distance"] / straight
    good = ratio.between(ROAD_FACTOR_MIN, ROAD_FACTOR_MAX)
    road_factor = float(ratio[good].median())
    return road_factor, daily_index_from(train)


def daily_index_from(*frames: pd.DataFrame) -> pd.Series:
    """Average market_index per calendar day (feature-only, no labels used)."""
    all_rows = pd.concat([f[["date", "market_index"]] for f in frames])
    all_rows["date"] = pd.to_datetime(all_rows["date"])
    return all_rows.groupby("date")["market_index"].mean()


def flag_label_outliers(df: pd.DataFrame, factor: float = 1.6) -> pd.Series:
    """Flag corrupted labels: rate per mile far from the same lane+equipment median.

    About 1.4% of rows sit at roughly 0.2x-0.5x or 2x-5x their lane's normal rate,
    which looks like unit/entry errors rather than real market moves.
    """
    rpm = df["posted_rate"] / df["distance"]
    lane_median = rpm.groupby(
        [df["pickup"], df["delivery"], df["equipment"]]
    ).transform("median")
    # Lanes with few loads: fall back to equipment + distance band median.
    band = pd.cut(df["distance"], [0, 250, 500, 1000, 1500, 2000, 5000])
    fallback = rpm.groupby([df["equipment"], band], observed=True).transform("median")
    lane_count = rpm.groupby([df["pickup"], df["delivery"], df["equipment"]]).transform(
        "size"
    )
    reference = lane_median.where(lane_count >= 5, fallback)
    ratio = rpm / reference
    return (ratio > factor) | (ratio < 1 / factor)
