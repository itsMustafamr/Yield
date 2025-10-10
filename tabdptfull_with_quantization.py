# %%
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import r2_score, mean_squared_error
from tabdpt import TabDPTRegressor

# %%# 1. Load your datasets
train = pd.read_csv('train.csv')
val   = pd.read_csv('val.csv')
test  = pd.read_csv('test.csv')

# %%
# 2. Compute fill‐values on TRAIN only
mg_mode      = train['MG'].mode()[0]
lon_median   = train['Longitude'].median()
mean_cols    = ['Lodging','PlantHeight','SeedSize','Protein','Oil']
mean_values  = train[mean_cols].mean()

# 3. Fill missing in all splits
for df in (train, val, test):
    df['MG']        = df['MG'].fillna(mg_mode)
    df['Longitude'] = df['Longitude'].fillna(lon_median)
    for col in mean_cols:
        df[col]     = df[col].fillna(mean_values[col])

# %%
# 4. Aggregation helper
temporal_feats = ['MaxTemp','MinTemp','AvgTemp','AvgHumidity','Precipitation','Radiation']
static_feats   = ['Latitude','Longitude','Row.Spacing']
plant_feats    = ['Lodging','PlantHeight','SeedSize','Protein','Oil']
cluster_feats  = [f'Cluster_{i}' for i in range(40)]

def aggregate_sequences(df, target='Yield', agg_target='mean'):
    agg_dict = {}
    # temporal: mean & std
    for f in temporal_feats:
        agg_dict[f'{f}_mean'] = (f, 'mean')
        agg_dict[f'{f}_std']  = (f, 'std')
    # static: first
    for f in static_feats:
        agg_dict[f] = (f, 'first')
    # MG: mode; plant_feats: first
    agg_dict['MG'] = ('MG', lambda x: x.mode().iloc[0])
    for f in plant_feats:
        agg_dict[f] = (f, 'first')
    # clusters: mean & std
    for f in cluster_feats:
        agg_dict[f'{f}_mean'] = (f, 'mean')
        agg_dict[f'{f}_std']  = (f, 'std')
    # target
    if agg_target=='mean':
        agg_dict[target] = (target, 'mean')
    elif agg_target=='final':
        agg_dict[target] = (target, lambda x: x.iloc[-1])
    else:
        raise ValueError("agg_target must be 'mean' or 'final'")
    return df.groupby('TimeSeriesLabel').agg(**agg_dict).reset_index(drop=True)

# %%
# 5. Aggregate
train_agg = aggregate_sequences(train, agg_target='mean')
val_agg   = aggregate_sequences(val,   agg_target='mean')
test_agg  = aggregate_sequences(test,  agg_target='mean')

# %%
# 6. Split into X (DataFrame) and y (Series → to be converted)
X_train_df = train_agg.drop('Yield', axis=1)
y_train_sr = train_agg['Yield']
X_val_df   = val_agg.drop('Yield', axis=1)
y_val_sr   = val_agg['Yield']
X_test_df  = test_agg.drop('Yield', axis=1)
y_test_sr  = test_agg['Yield']

# %%
# 7. Define which columns get which binner
uniform_cols = (
    [f"{f}_mean" for f in temporal_feats if f not in ('Precipitation','Radiation')]
  + static_feats
  + plant_feats
)
quantile_cols = (
    [f"{f}_mean" for f in ('Precipitation','Radiation')]
  + [f"{f}_std"  for f in temporal_feats]
  + [f"{c}_mean" for c in cluster_feats]
  + [f"{c}_std"  for c in cluster_feats]
)

ct = ColumnTransformer([
    ("uniform",  KBinsDiscretizer(n_bins=256, encode='ordinal', strategy='uniform'), uniform_cols),
    ("quantile", KBinsDiscretizer(n_bins=5,   encode='ordinal', strategy='quantile'), quantile_cols),
], remainder="passthrough", sparse_threshold=0)

# %%
# 8. Fit/transform your features
X_train = ct.fit_transform(X_train_df)
X_val   = ct.transform(X_val_df)
X_test  = ct.transform(X_test_df)

# 9. **Convert y to NumPy arrays** so TabDPT is happy
y_train = y_train_sr.to_numpy()
y_val   = y_val_sr.to_numpy()
y_test  = y_test_sr.to_numpy()

# %%
# 10. Train & Evaluate
model = TabDPTRegressor()
model.fit(X_train, y_train)                      # no more Series here

for name, X, y in [('Val', X_val, y_val), ('Test', X_test, y_test)]:
    preds = model.predict(
        X,
        n_ensembles=8,
        context_size=len(X_train),
    )
    rmse = mean_squared_error(y, preds, squared=False)
    print(f"{name} R²: {r2_score(y, preds):.3f}, RMSE: {rmse:.3f}")
