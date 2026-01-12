import pandas as pd
import numpy as np
import os
from glob import glob

# ============================
# PATHS
# ============================
BASE = "multi-modal-ist.github.io/datasets/ccRCC/Code"
WSI_PATH = f"{BASE}/Features/WSI Features/"
MRI_PATH = f"{BASE}/Features/MRI Features/"
CT_PATH  = f"{BASE}/Features/CT Features/"

# ============================
# LOAD CLINICAL
# ============================
clinical = pd.read_csv(f"{BASE}/clinical+genomic_split.csv")

print("Loaded clinical:", clinical.shape)

# ============================
# LOAD CORRECT MAPPING FILES
# ============================
wsi_files = pd.read_csv(f"{BASE}/WSI_patientfiles.csv")
mri_files = pd.read_csv(f"{BASE}/patients_with_labels_MR_final.csv")[["case_id"]].drop_duplicates()
ct_files  = pd.read_csv(f"{BASE}/patients_with_labels_CT_final.csv")[["case_id"]].drop_duplicates()

print("WSI rows:", len(wsi_files))
print("MRI unique cases:", len(mri_files))
print("CT unique cases:", len(ct_files))

# ============================
# LOAD WSI FEATURES
# ============================
def load_wsi(fname):
    path = os.path.join(WSI_PATH, fname)
    arr = np.load(path)["arr_0"]
    return arr.squeeze()

wsi_files["wsi_data"] = wsi_files["chosen_exam"].apply(load_wsi)
wsi_files = wsi_files.drop(columns=["chosen_exam"])

print("WSI loaded.")

# ============================
# MRI LOADING FUNCTION
# ============================
def load_mri_for_case(case_id):
    pattern = os.path.join(MRI_PATH, f"{case_id}-*.npz")
    files = sorted(glob(pattern))
    if len(files) == 0:
        return None

    feats = []
    for f in files:
        arr = np.load(f)["arr_0"].squeeze()
        feats.append(arr)

    return np.mean(feats, axis=0)  # average fusion across slices

mri_files["mri_data"] = mri_files["case_id"].apply(load_mri_for_case)
print("MRI loaded.")

# ============================
# CT LOADING FUNCTION
# ============================
def load_ct_for_case(case_id):
    pattern = os.path.join(CT_PATH, f"{case_id}-*.npz")
    files = sorted(glob(pattern))
    if len(files) == 0:
        return None

    feats = []
    for f in files:
        arr = np.load(f)["arr_0"].squeeze()
        feats.append(arr)

    return np.mean(feats, axis=0)

ct_files["ct_data"] = ct_files["case_id"].apply(load_ct_for_case)
print("CT loaded.")

# ============================
# MERGE EVERYTHING
# ============================
all_data = clinical.merge(wsi_files, on="case_id", how="left")
all_data = all_data.merge(mri_files, on="case_id", how="left")
all_data = all_data.merge(ct_files, on="case_id", how="left")

print("\nFinal shape:", all_data.shape)
print(all_data.head())

# ============================
# SAVE TO PARQUET
# ============================
os.makedirs("data", exist_ok=True)
all_data.to_parquet("data/ccRCC_data.parquet", index=False)
print("Saved to data/ccRCC_data.parquet")
