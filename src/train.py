"""Train the freight rate model, validate it over time, and write both prediction files.

Usage:  python src/train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import (  # noqa: E402
    build_features,
    clean,
    daily_index_from,
    fit_cleaning_stats,
    flag_label_outliers,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
SEED = 42

# Target is log(rate per mile): rate scales roughly with distance, so this makes the
# target near-stationary across short and long lanes. L1 loss limits the pull of any
# corrupted labels that survive cleaning.
PARAMS = dict(
    objective="l1",
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
)


def fit(train: pd.DataFrame) -> tuple[lgb.LGBMRegressor, float]:
    road_factor, daily = fit_cleaning_stats(train)
    train = clean(train, road_factor, daily)
    keep = ~flag_label_outliers(train)
    X = build_features(train[keep])
    y = np.log(train.loc[keep, "posted_rate"] / train.loc[keep, "distance"])
    model = lgb.LGBMRegressor(**PARAMS).fit(X, y)
    return model, road_factor


def predict(
    model, road_factor: float, frame: pd.DataFrame, daily: pd.Series
) -> np.ndarray:
    cleaned = clean(frame, road_factor, daily)
    return (
        np.exp(model.predict(build_features(cleaned))) * cleaned["distance"].to_numpy()
    )


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    e = np.abs(p - y)
    return {
        "MAE": round(float(e.mean()), 2),
        "MAPE_%": round(float((e / y).mean() * 100), 2),
        "median_APE_%": round(float(np.median(e / y) * 100), 2),
    }


def time_based_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window backtest: always train on the past, test on the next 2 months.

    validation.csv is Nov-Dec 2025 (after all labelled data), so a random split
    would be optimistic. Each fold mimics the real task: predict future months.
    """
    rows = []
    folds = [
        ("2025-05-01", "2025-07-01"),
        ("2025-07-01", "2025-09-01"),
        ("2025-09-01", "2025-11-01"),
    ]
    for start, end in folds:
        train = df[df["date"] < start]
        test = df[(df["date"] >= start) & (df["date"] < end)]
        model, road_factor = fit(train)
        pred = predict(model, road_factor, test, daily_index_from(test))
        test_clean = clean(test, road_factor, daily_index_from(test))
        good = ~flag_label_outliers(test_clean).to_numpy()
        y = test["posted_rate"].to_numpy()
        # Naive baseline: equipment median rate-per-mile x distance (from training months).
        rpm = (
            (train["posted_rate"] / train["distance"])
            .groupby(train["equipment"])
            .median()
        )
        base = test["equipment"].map(rpm).to_numpy() * test_clean["distance"].to_numpy()
        rows.append(
            {
                "train_until": start,
                "test_months": f"{pd.Timestamp(start):%b}-{(pd.Timestamp(end) - pd.Timedelta(days=1)):%b %Y}",
                "n_train": len(train),
                "n_test": len(test),
                **{f"model_{k}": v for k, v in metrics(y[good], pred[good]).items()},
                "model_MAE_all_rows": metrics(y, pred)["MAE"],
                "baseline_MAPE_%": metrics(y[good], base[good])["MAPE_%"],
            }
        )
    return pd.DataFrame(rows)


def december_inputs(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Add the model inputs the December chart file does not include.

    Coordinates come from the city lookup in the training data. market_index and
    quote_signal are not given per day, so we use the December loads in
    validation.csv: that day's average market_index and the month's median quote_signal.
    """
    dec = pd.read_csv(DATA / "december_chart_inputs.csv")
    cities = (
        pd.concat(
            [
                train[["pickup", "pickup_lat", "pickup_lon"]].set_axis(
                    ["city", "lat", "lon"], axis=1
                ),
                train[["delivery", "delivery_lat", "delivery_lon"]].set_axis(
                    ["city", "lat", "lon"], axis=1
                ),
            ]
        )
        .drop_duplicates("city")
        .set_index("city")
    )
    v = validation.copy()
    v["date"] = pd.to_datetime(v["date"])
    v_dec = v[v["date"].dt.month == 12]
    frame = dec.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["pickup_lat"] = frame["pickup"].map(cities["lat"])
    frame["pickup_lon"] = frame["pickup"].map(cities["lon"])
    frame["delivery_lat"] = frame["delivery"].map(cities["lat"])
    frame["delivery_lon"] = frame["delivery"].map(cities["lon"])
    frame["market_index"] = frame["date"].map(
        v_dec.groupby("date")["market_index"].mean()
    )
    frame["quote_signal"] = v_dec["quote_signal"].median()
    return dec, frame


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    df = pd.read_csv(DATA / "train_test.csv")
    df["date"] = pd.to_datetime(df["date"])
    validation = pd.read_csv(DATA / "validation.csv")

    cv = time_based_validation(df)
    cv.to_csv(REPORTS / "backtest_results.csv", index=False)
    print(cv.to_string(index=False))

    # Final model: all labelled data (Jan-Oct 2025).
    model, road_factor = fit(df)
    daily_val = daily_index_from(validation)

    preds = predict(model, road_factor, validation, daily_val)
    out = pd.read_csv(DATA / "validation_predictions_template.csv")[["load_id"]]
    out = out.merge(
        pd.DataFrame(
            {"load_id": validation["load_id"], "predicted_rate": preds.round(2)}
        ),
        on="load_id",
        how="left",
        validate="one_to_one",
    )
    assert out["predicted_rate"].notna().all() and len(out) == 12_000
    out.to_csv(ROOT / "validation_predictions.csv", index=False)

    dec, dec_frame = december_inputs(df, validation)
    dec["predicted_rate"] = predict(model, road_factor, dec_frame, daily_val).round(2)
    dec.to_csv(DATA / "december_chart_inputs.csv", index=False)

    imp = pd.Series(
        model.booster_.feature_importance("gain"), model.booster_.feature_name()
    )
    (imp / imp.sum()).sort_values(ascending=False).round(4).to_csv(
        REPORTS / "feature_importance.csv", header=["gain_share"]
    )
    road_factor_info = {"road_factor": round(road_factor, 4)}
    (REPORTS / "cleaning_stats.json").write_text(json.dumps(road_factor_info, indent=2))
    print(
        f"Wrote validation_predictions.csv ({len(out):,} rows) and December predictions."
    )
    print(dec[["date", "predicted_rate"]].describe())


if __name__ == "__main__":
    main()
