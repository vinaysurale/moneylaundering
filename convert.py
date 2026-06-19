import pandas as pd
df = pd.read_parquet('data/elliptic_features_optimized.parquet')
df.to_csv('data/elliptic_features_optimized.csv', header=False, index=False)
