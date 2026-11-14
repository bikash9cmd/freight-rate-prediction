# Freight Rate Prediction

Predicts the posted rate for a truckload from lane, equipment, weight, date and market signals.

## Run

Requires Python 3.10-3.13 (pandas 2.x does not run on Python 3.14+).

```bash
python3.13 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python src/train.py          # backtest, train final model, write predictions
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Outputs:

- `validation_predictions.csv`: `load_id,predicted_rate` for all 12,000 validation loads
- `data/december_chart_inputs.csv`: filled `predicted_rate` for the fixed December lane
- `scorer_results/candidate_december.png`: chart from the provided scorer
- `reports/backtest_results.csv`, `reports/feature_importance.csv`

## Approach in short

1. **Cleaning** (`src/features.py`)
   - Negative weights (292 rows) are sign errors, so the absolute value is used. Missing weights (300) stay missing and LightGBM handles them.
   - Distances that are impossible given the city coordinates (95 rows, e.g. 70 mi on a 600 mi lane) are rebuilt from straight-line distance x the typical road factor (about 1.19).
   - Missing `market_index` (374) is filled with that day's average index.
   - About 1.4% of labels are 2-5x above or below their lane's normal rate per mile. These look like entry errors, so they are excluded from training.
2. **Features**: log distance, straight-line distance, coordinates, equipment, weight, load and daily market index, quote signal. City names are not used because 12% of validation loads involve cities never seen in training.
3. **Model**: LightGBM, L1 loss, target = log(rate per mile). Rate is predicted as exp(model) x distance.
4. **Validation**: the validation set is Nov-Dec 2025, after all labelled data, so the split is by time, never random. An expanding-window backtest trains on earlier months and tests on the next two months (3 folds).

| Test months | Train rows | MAPE (clean labels) | Baseline MAPE |
|---|---|---|---|
| May-Jun | 19,110 | 1.9% | 9.0% |
| Jul-Aug | 28,806 | 4.2% | 8.0% |
| Sep-Oct | 38,477 | 3.0% | 8.1% |

Baseline = equipment median rate per mile x distance.
