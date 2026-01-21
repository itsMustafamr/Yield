import argparse
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm

# ---- KumoRFM import ----
import kumoai.experimental.rfm as rfm

os.environ["KUMO_API_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1OWVjMjRhZTVhMmZkMjI5OTcwZTEzYTU4MjY0ZmI0MSIsImp0aSI6ImVkM2M5ZDM3LThmYjctNDQ3Yi1hY2QxLWYzOGRjMTVhMzM3OSIsImlhdCI6MTc1NDAwMTAwMiwiZXhwIjoxNzU5MTg1MDAyfQ.HKsPrNXa18I4VtO0ZzNkvyoVLHxIilDOj8UnFfdys9k"
rfm.init()

# -----------------------
# Feature definitions (same as your original script)
temporal_feats = ['MaxTemp', 'MinTemp', 'AvgTemp', 'AvgHumidity', 'Precipitation', 'Radiation']
static_feats   = ['Latitude', 'Longitude', 'Row.Spacing']
plant_feats    = ['Lodging', 'PlantHeight', 'SeedSize', 'Protein', 'Oil']
cluster_feats  = [f'Cluster_{i}' for i in range(40)]
target_col = 'Yield'


def parse_args():
    parser = argparse.ArgumentParser(description="KumoRFM on yield timeseries")
    parser.add_argument("--api-key", required=True, help="Kumo API key")
    parser.add_argument("--data-dir", required=True, help="Directory containing train.csv,val.csv,test.csv")
    parser.add_argument("--sample-test", type=int, default=None, help="If set, subsample this many test series for speed")
    parser.add_argument("--skip-visualize", action="store_true", help="Skip graph.visualize()")
    parser.add_argument("--agg-target", choices=['mean', 'last'], default='mean', help="Aggregation strategy for target")
    return parser.parse_args()


def load_and_impute(train, val, test):
    # Compute TRAIN-only statistics
    mg_mode    = train['MG'].mode()[0]
    lon_med    = train['Longitude'].median()
    mean_cols  = ['Lodging', 'PlantHeight', 'SeedSize', 'Protein', 'Oil']
    mean_vals  = train[mean_cols].mean()

    for df in (train, val, test):
        df['MG']        = df['MG'].fillna(mg_mode)
        df['Longitude'] = df['Longitude'].fillna(lon_med)
        for c in mean_cols:
            df[c]      = df[c].fillna(mean_vals[c])

    return train, val, test


def aggregate_sequences(df, target=target_col, agg_target='mean'):
    """
    Aggregates each TimeSeriesLabel into one row. Keeps TimeSeriesLabel as column.
    """
    agg = {}
    # temporal: mean & std
    for f in temporal_feats:
        agg[f'{f}_mean'] = (f, 'mean')
        agg[f'{f}_std']  = (f, 'std')
    # static geography: first
    for f in static_feats:
        agg[f] = (f, 'first')
    # plant: MG by mode, others by first
    agg['MG'] = ('MG', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
    for f in plant_feats:
        agg[f] = (f, 'first')
    # clusters: mean & std
    for f in cluster_feats:
        agg[f'{f}_mean'] = (f, 'mean')
        agg[f'{f}_std']  = (f, 'std')
    # target
    if agg_target == 'mean':
        agg[target] = (target, 'mean')
    else:
        agg[target] = (target, lambda x: x.iloc[-1])

    grouped = df.groupby('TimeSeriesLabel').agg(**agg).reset_index()
    grouped = grouped.rename(columns={'TimeSeriesLabel': 'series_id'})
    return grouped


def safe_extract_prediction(result):
    # KumoRFM .predict output format can vary; try common keys.
    if isinstance(result, dict):
        for key in ['prediction', 'pred', 'value', 'y_pred']:
            if key in result:
                return result[key]
        # sometimes nested
        if 'predictions' in result and isinstance(result['predictions'], list):
            return result['predictions'][0]
        # fallback: try any numeric-like leaf
        for v in result.values():
            if isinstance(v, (int, float, np.floating)):
                return v
    # if bare scalar
    if isinstance(result, (int, float, np.floating)):
        return result
    raise ValueError(f"Can't extract prediction from result: {result}")


def predict_batch(model, df, max_tries=3, backoff_base=1.0):
    y_pred = []
    for sid in tqdm(df["series_id"].tolist(), desc="RFM predict"):
        query = f"PREDICT yield_data.{target_col} FOR yield_data.series_id={sid}"
        attempt = 0
        while True:
            try:
                res = model.predict(query)
                pred = safe_extract_prediction(res)
                y_pred.append(pred)
                break
            except Exception as e:
                attempt += 1
                if attempt >= max_tries:
                    raise RuntimeError(f"Failed to get prediction for {sid} after {max_tries} attempts: {e}")
                sleep = backoff_base * (2 ** (attempt - 1))
                time.sleep(sleep)
    return np.array(y_pred, dtype=float)


def main():
    args = parse_args()
    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1OWVjMjRhZTVhMmZkMjI5OTcwZTEzYTU4MjY0ZmI0MSIsImp0aSI6ImVkM2M5ZDM3LThmYjctNDQ3Yi1hY2QxLWYzOGRjMTVhMzM3OSIsImlhdCI6MTc1NDAwMTAwMiwiZXhwIjoxNzU5MTg1MDAyfQ.HKsPrNXa18I4VtO0ZzNkvyoVLHxIilDOj8UnFfdys9k"
    os.environ["KUMO_API_KEY"] = api_key
    rfm.init()

    # Load raw CSVs
    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    val   = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    test  = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

    # Clean/impute
    train, val, test = load_and_impute(train, val, test)

    # Aggregate sequences
    train_agg = aggregate_sequences(train, agg_target=args.agg_target)
    val_agg   = aggregate_sequences(val,   agg_target=args.agg_target)
    test_agg  = aggregate_sequences(test,  agg_target=args.agg_target)

    # Optionally subsample test for speed
    if args.sample_test is not None:
        test_agg = test_agg.sample(min(args.sample_test, len(test_agg)), random_state=42).reset_index(drop=True)

    # Build full context table (train + val)
    full_train = pd.concat([train_agg, val_agg], axis=0).reset_index(drop=True)

    # Sanity: ensure Yield exists
    assert target_col in full_train.columns, f"{target_col} missing in aggregated data"

    graph = rfm.LocalGraph.from_data({
    "yield_data": full_train,
    "yield_data": test_agg,
    "yield_data": val_agg,
    })

    
    graph.validate()
    if not args.skip_visualize:
        try:
            graph.visualize()
        except Exception as e:
            print(f"[warning] graph.visualize() failed (maybe graphviz missing): {e}")

    model = rfm.KumoRFM(graph)

    # Extract ground truth for test
    y_true = test_agg[target_col].to_numpy()

    # Predict
    y_pred = predict_batch(model, test_agg)

    # Metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("\n===== Evaluation on test set =====")
    print(f"Examples: {len(y_true)}")
    print(f"R²   : {r2:.4f}")
    print(f"RMSE : {rmse:.4f}")

    # Save predictions
    out_df = test_agg[["series_id", target_col]].copy()
    out_df["rfm_pred"] = y_pred
    out_csv = "rfm_test_predictions.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"Saved test predictions to {out_csv}")


if __name__ == "__main__":
    main()
