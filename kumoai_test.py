# %%
import kumoai.experimental.rfm as rfm, os

os.environ["KUMO_API_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1OWVjMjRhZTVhMmZkMjI5OTcwZTEzYTU4MjY0ZmI0MSIsImp0aSI6ImVkM2M5ZDM3LThmYjctNDQ3Yi1hY2QxLWYzOGRjMTVhMzM3OSIsImlhdCI6MTc1NDAwMTAwMiwiZXhwIjoxNzU5MTg1MDAyfQ.HKsPrNXa18I4VtO0ZzNkvyoVLHxIilDOj8UnFfdys9k"

rfm.init()

# %%
import pandas as pd

dataset_url = "s3://kumo-sdk-public/rfm-datasets/online-shopping"

users_df = pd.read_parquet(f"{dataset_url}/users.parquet")
items_df = pd.read_parquet(f"{dataset_url}/items.parquet")
orders_df = pd.read_parquet(f"{dataset_url}/orders.parquet")

# %%
graph = rfm.LocalGraph.from_data({
    "users": users_df,
    "items": items_df,
    "orders": orders_df,
})

# Inspect the graph - requires graphviz to be installed
graph.visualize()

# %%
model = rfm.KumoRFM(graph)

# Forecast 30-day product demand
query = "PREDICT SUM(orders.price, 0, 30, days) FOR items.item_id=1"

prediction_result = model.predict(query)
print(prediction_result)


