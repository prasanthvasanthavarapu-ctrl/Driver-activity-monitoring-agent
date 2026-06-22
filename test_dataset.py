import pandas as pd

df = pd.read_csv("mental_health_monitoring_dataset.csv")

print(df.head())
print(df.columns.tolist())
print(df.shape)