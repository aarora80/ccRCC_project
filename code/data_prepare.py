import pandas as pd
import torch
import numpy as np

DATA_PATH = "data/ccRCC_data.parquet"
SAVE_PATH = "data/processed_ccRCC.pt"

df = pd.read_parquet(DATA_PATH)
print("Loaded merged dataset:", df.shape)

# -------------------------------------------------
# 1. Extract labels SEPARATELY (VERY IMPORTANT)
# -------------------------------------------------
labels = df["vital_status_12"].astype("int64").values

# -------------------------------------------------
# 2. Define clinical feature columns (exclude label + modalities)
# -------------------------------------------------
clinical_cols = [
    c for c in df.columns
    if c not in ["case_id", "Split", "wsi_data", "mri_data", "ct_data", "vital_status_12"]
]

print("Clinical feature count:", len(clinical_cols))

clinical_np = df[clinical_cols].fillna(0).astype("float32").values

# -------------------------------------------------
# 3. Extract modalities
# -------------------------------------------------
def modality_to_array(x):
    if isinstance(x, np.ndarray):
        return x.astype("float32")
    return None

wsi = [modality_to_array(x) for x in df["wsi_data"]]
mri = [modality_to_array(x) for x in df["mri_data"]]
ct  = [modality_to_array(x) for x in df["ct_data"]]

WSI_DIM = len(wsi[0])
MRI_DIM = 512
CT_DIM  = 512

# -------------------------------------------------
# 4. Fill missing modalities + create masks
# -------------------------------------------------
def fill_and_mask(mod_list, dim):
    filled = []
    masks = []
    for x in mod_list:
        if x is None:
            filled.append(np.zeros(dim, dtype="float32"))
            masks.append(0.0)
        else:
            filled.append(x)
            masks.append(1.0)
    return np.stack(filled), np.array(masks).reshape(-1,1)

wsi_np, wsi_mask = fill_and_mask(wsi, WSI_DIM)
mri_np, mri_mask = fill_and_mask(mri, MRI_DIM)
ct_np,  ct_mask  = fill_and_mask(ct,  CT_DIM)

# -------------------------------------------------
# 5. Train/test split
# -------------------------------------------------
train_idx = df["Split"] == "train"
test_idx  = df["Split"] == "test"

data = {
    "train": {
        "clinical": torch.tensor(clinical_np[train_idx], dtype=torch.float32),
        "wsi":      torch.tensor(wsi_np[train_idx], dtype=torch.float32),
        "mri":      torch.tensor(mri_np[train_idx], dtype=torch.float32),
        "ct":       torch.tensor(ct_np[train_idx], dtype=torch.float32),

        "mask_wsi": torch.tensor(wsi_mask[train_idx], dtype=torch.float32),
        "mask_mri": torch.tensor(mri_mask[train_idx], dtype=torch.float32),
        "mask_ct":  torch.tensor(ct_mask[train_idx], dtype=torch.float32),

        "labels":   torch.tensor(labels[train_idx], dtype=torch.long)
    },
    "test": {
        "clinical": torch.tensor(clinical_np[test_idx], dtype=torch.float32),
        "wsi":      torch.tensor(wsi_np[test_idx], dtype=torch.float32),
        "mri":      torch.tensor(mri_np[test_idx], dtype=torch.float32),
        "ct":       torch.tensor(ct_np[test_idx], dtype=torch.float32),

        "mask_wsi": torch.tensor(wsi_mask[test_idx], dtype=torch.float32),
        "mask_mri": torch.tensor(mri_mask[test_idx], dtype=torch.float32),
        "mask_ct":  torch.tensor(ct_mask[test_idx], dtype=torch.float32),

        "labels":   torch.tensor(labels[test_idx], dtype=torch.long)
    }
}

torch.save(data, SAVE_PATH)
print("Saved:", SAVE_PATH)
print("Train size:", data["train"]["clinical"].shape[0])
print("Test size:", data["test"]["clinical"].shape[0])
