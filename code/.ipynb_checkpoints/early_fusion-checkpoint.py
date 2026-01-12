import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

DATA_PATH = "data/processed_ccRCC.pt"
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 40

# -------------------------
# 1. Load processed data
# -------------------------
data = torch.load(DATA_PATH)

train = data["train"]
test  = data["test"]

# Clinical features (label already removed in data_prepare)
clinical_train = train["clinical"]
clinical_test  = test["clinical"]

# Modalities
wsi_train = train["wsi"]
mri_train = train["mri"]
ct_train  = train["ct"]

wsi_test = test["wsi"]
mri_test = test["mri"]
ct_test  = test["ct"]

# Masks
mask_wsi_train = train["mask_wsi"]
mask_mri_train = train["mask_mri"]
mask_ct_train  = train["mask_ct"]

mask_wsi_test = test["mask_wsi"]
mask_mri_test = test["mask_mri"]
mask_ct_test  = test["mask_ct"]

# Labels (correct)
y_train = train["labels"].long().view(-1)
y_test  = test["labels"].long().view(-1)

print("Label distribution TRAIN:", torch.bincount(y_train))
print("Label distribution TEST :", torch.bincount(y_test))

# -------------------------
# 2. Build Early Fusion Input
# -------------------------

def fuse_inputs(clinical, wsi, mri, ct, mask_wsi, mask_mri, mask_ct):
    """
    Concatenate:
    [clinical | wsi | mri | ct | mask_wsi | mask_mri | mask_ct]
    """
    return torch.cat(
        [clinical, wsi, mri, ct, mask_wsi, mask_mri, mask_ct],
        dim=1
    )

X_train = fuse_inputs(
    clinical_train, wsi_train, mri_train, ct_train,
    mask_wsi_train, mask_mri_train, mask_ct_train
)

X_test = fuse_inputs(
    clinical_test, wsi_test, mri_test, ct_test,
    mask_wsi_test, mask_mri_test, mask_ct_test
)

print("Early Fusion input dimension:", X_train.shape[1])

# -------------------------
# 3. Dataloaders
# -------------------------
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -------------------------
# 4. Define MLP model
# -------------------------

class EarlyFusionMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)  # 2 classes
        )

    def forward(self, x):
        return self.net(x)


model = EarlyFusionMLP(X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------
# 5. Training Loop
# -------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Evaluate
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.3f} | Test Acc: {acc:.4f}")

# -------------------------
# 6. Save Model
# -------------------------
torch.save(model.state_dict(), "models/early_fusion.pth")
print("Saved early fusion model → models/early_fusion.pth")
