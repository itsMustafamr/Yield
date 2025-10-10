# %%
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error




# %%

# 1. Load
train = pd.read_csv('train.csv')
val   = pd.read_csv('val.csv')
test  = pd.read_csv('test.csv')

# %%
# 1. Compute fill‐values on TRAIN only
mg_mode      = train['MG'].mode()[0]           # mode for categorical MG
lon_median   = train['Longitude'].median()     # median for longitude
mean_cols    = ['Lodging','PlantHeight','SeedSize','Protein','Oil']
mean_values  = train[mean_cols].mean()         # means for plant characteristics

# 2. Fill missing in train/val/test
for df in (train, val, test):
    df['MG']        = df['MG'].fillna(mg_mode)
    df['Longitude'] = df['Longitude'].fillna(lon_median)
    for col in mean_cols:
        df[col]     = df[col].fillna(mean_values[col])

# 3. Define your features

# %%
# make sure these are defined at top‐level:
temporal_feats = ['MaxTemp','MinTemp','AvgTemp','AvgHumidity','Precipitation','Radiation']
static_feats   = ['Latitude','Longitude','Row.Spacing']
plant_feats    = ['Lodging','PlantHeight','SeedSize','Protein','Oil']
cluster_feats  = [f'Cluster_{i}' for i in range(40)]

def aggregate_sequences(df, target='Yield', agg_target='mean'):
    agg_dict = {}

    # 1. temporal: mean & std
    for feat in temporal_feats:
        agg_dict[f'{feat}_mean'] = (feat, 'mean')
        agg_dict[f'{feat}_std']  = (feat, 'std')

    # 2. static geography: take first (constant per sequence)
    for feat in static_feats:
        agg_dict[feat] = (feat, 'first')

    # 3. plant features:
    #    - MG (categorical) → mode  
    agg_dict['MG'] = ('MG', lambda x: x.mode().iloc[0])
    #    - Lodging, PlantHeight, SeedSize, Protein, Oil → first
    for feat in plant_feats:
        agg_dict[feat] = (feat, 'first')

    # 4. cluster indicators: proportion of time in each cluster + variability
    for feat in cluster_feats:
        agg_dict[f'{feat}_mean'] = (feat, 'mean')
        agg_dict[f'{feat}_std']  = (feat, 'std')

    # 5. target: mean or final
    if agg_target == 'mean':
        agg_dict[target] = (target, 'mean')
    elif agg_target == 'final':
        agg_dict[target] = (target, lambda x: x.iloc[-1])
    else:
        raise ValueError("agg_target must be 'mean' or 'final'")

    # apply the aggregation
    grouped = df.groupby('TimeSeriesLabel').agg(**agg_dict)
    return grouped.reset_index(drop=True)


# %%
train_agg = aggregate_sequences(train, agg_target='mean')
val_agg   = aggregate_sequences(val,   agg_target='mean')
test_agg  = aggregate_sequences(test,  agg_target='mean')

# %%
# Prepare feature / target matrices as before, but convert to numpy:
X_train = train_agg.drop('Yield', axis=1).to_numpy()
y_train = train_agg['Yield'].to_numpy()
X_val   = val_agg.drop('Yield',   axis=1).to_numpy()
y_val   = val_agg['Yield'].to_numpy()
X_test  = test_agg.drop('Yield',  axis=1).to_numpy()
y_test  = test_agg['Yield'].to_numpy()



# %%
import numpy as np
from tabdpt import TabDPTRegressor
from sklearn.metrics import r2_score, root_mean_squared_error


# Instantiate and train on all ~87k rows:
model = TabDPTRegressor()
model.fit(X_train, y_train)


for name, X_np, y_np in [
        ('Val',  X_val,  y_val),
        ('Test', X_test, y_test)
    ]:
    preds = model.predict(
        X_np,
        n_ensembles=8,                    # trade speed vs. stability
        context_size=len(X_train),        # ≥ number of rows in X_train
    )
    print(f"{name} R²:  {r2_score(y_np, preds):.3f},",
          f"RMSE: {root_mean_squared_error(y_np, preds):.3f}")



# %%



