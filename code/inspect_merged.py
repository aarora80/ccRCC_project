import pandas as pd
import numpy as np

df = pd.read_parquet("data/ccRCC_data.parquet")

print("\nDataset shape:", df.shape)

print("\nModality availability:")
print("WSI nulls:", df["wsi_data"].isna().sum())
print("MRI nulls:", df["mri_data"].isna().sum())
print("CT  nulls:", df["ct_data"].isna().sum())

print("\nValue counts:")
print(df[["mri_data", "ct_data"]].isna().value_counts())

print("\nExample MRI shapes (first 5 non-null):")
example_mri = df.dropna(subset=["mri_data"]).head()
for i, row in example_mri.iterrows():
    print(row["case_id"], "MRI shape:", np.array(row["mri_data"]).shape)

print("\nExample CT shapes (first 5 non-null):")
example_ct = df.dropna(subset=["ct_data"]).head()
for i, row in example_ct.iterrows():
    print(row["case_id"], "CT shape:", np.array(row["ct_data"]).shape)
