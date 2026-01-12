import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, confusion_matrix

DATA_PATH = "data/processed_ccRCC.pt"
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 40

# ---------------------------------------
# 1. Load processed dataset
# ---------------------------------------
data = torch.load(DATA_PATH)
train = data["train"]
test  = data["test"]

# ---------------------------------------
# 2. Extract labels (last clinical column)
# ---------------------------------------
clinical_train = train["clinical"]
clinical_test  = test["clinical"]

y_train = clinical_train[:, -1]
y_test  = clinical_test[:, -1]

# Replace -1 with 0 for alive
y_train = torch.where(y_train == -1, torch.tensor(0.0), y_train)
y_test  = torch.where(y_test == -1, torch.tensor(0.0), y_test)

y_train = y_train.long().view(-1)
y_test  = y_test.long().view(-1)

print("Label distribution TRAIN:", torch.bincount(y_train))
print("Label distribution TEST :", torch.bincount(y_test))

# Remove labels from clinical feature vectors
clinical_train = clinical_train[:, :-1]
clinical_test  = clinical_test[:, :-1]

# ---------------------------------------
# 3. Modality tensors
# ---------------------------------------
wsi_train = train["wsi"]
mri_train = train["mri"]
ct_train  = train["ct"]

wsi_test = test["wsi"]
mri_test = test["mri"]
ct_test  = test["ct"]

mask_wsi_train = train["mask_wsi"]
mask_mri_train = train["mask_mri"]
mask_ct_train  = train["mask_ct"]

mask_wsi_test = test["mask_wsi"]
mask_mri_test = test["mask_mri"]
mask_ct_test  = test["mask_ct"]

# ---------------------------------------
# 4. Define small classifier for each modality
# ---------------------------------------
class SmallMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        return self.net(x)

clinical_head = SmallMLP(clinical_train.shape[1])
wsi_head      = SmallMLP(wsi_train.shape[1])
mri_head      = SmallMLP(mri_train.shape[1])
ct_head       = SmallMLP(ct_train.shape[1])

# ---------------------------------------
# 5. Learnable fusion weights
# ---------------------------------------
fusion_weights = nn.Parameter(torch.ones(4))  # [clinical, wsi, mri, ct]
softmax = nn.Softmax(dim=0)

params = (
    list(clinical_head.parameters()) +
    list(wsi_head.parameters()) +
    list(mri_head.parameters()) +
    list(ct_head.parameters()) +
    [fusion_weights]
)

# Weighted CE for class imbalance
alive = torch.sum(y_train == 0)
dead  = torch.sum(y_train == 1)
class_weights = torch.tensor([1.0, alive.float()/dead.float()])
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(params, lr=LR)

# ---------------------------------------
# 6. Dataloaders
# ---------------------------------------
train_loader = DataLoader(
    TensorDataset(
        clinical_train, wsi_train, mri_train, ct_train,
        mask_wsi_train, mask_mri_train, mask_ct_train,
        y_train
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(
        clinical_test, wsi_test, mri_test, ct_test,
        mask_wsi_test, mask_mri_test, mask_ct_test,
        y_test
    ),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ---------------------------------------
# 7. Fusion function
# ---------------------------------------
def fuse_logits(clin_logits, wsi_logits, mri_logits, ct_logits,
                mask_wsi, mask_mri, mask_ct):

    present = torch.cat([
        torch.ones_like(mask_wsi),  # clinical always present
        mask_wsi,
        mask_mri,
        mask_ct
    ], dim=1)  # [B, 4]

    logits_list = torch.stack([
        clin_logits,
        wsi_logits,
        mri_logits,
        ct_logits
    ], dim=1)  # [B, 4, 2]

    w = softmax(fusion_weights)
    w = w.unsqueeze(0).unsqueeze(2)
    w = w * present.unsqueeze(2)
    w = w / (w.sum(dim=1, keepdim=True) + 1e-8)

    return (logits_list * w).sum(dim=1)

# ---------------------------------------
# 8. Training loop with AUROC
# ---------------------------------------
for epoch in range(EPOCHS):
    clinical_head.train()
    wsi_head.train()
    mri_head.train()
    ct_head.train()

    total_loss = 0

    for clin, wsi, mri, ct, mask_wsi, mask_mri, mask_ct, labels in train_loader:
        optimizer.zero_grad()

        clin_log = clinical_head(clin)
        wsi_log  = wsi_head(wsi)
        mri_log  = mri_head(mri)
        ct_log   = ct_head(ct)

        fused = fuse_logits(clin_log, wsi_log, mri_log, ct_log,
                            mask_wsi, mask_mri, mask_ct)

        loss = criterion(fused, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Evaluation
    all_probs = []
    all_labels = []

    clinical_head.eval()
    wsi_head.eval()
    mri_head.eval()
    ct_head.eval()

    with torch.no_grad():
        for clin, wsi, mri, ct, mask_wsi, mask_mri, mask_ct, labels in test_loader:
            logits = fuse_logits(
                clinical_head(clin),
                wsi_head(wsi),
                mri_head(mri),
                ct_head(ct),
                mask_wsi, mask_mri, mask_ct
            )
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.3f} | AUROC: {auc:.4f}")
    print("Fusion weights:", softmax(fusion_weights).detach().cpu().numpy())

# ---------------------------------------
# 9. Confusion matrix on test set
# ---------------------------------------
preds = (all_probs > 0.5).astype(int)
cm = confusion_matrix(all_labels, preds)
print("Confusion Matrix:\n", cm)

# ---------------------------------------
# 10. Save model
# ---------------------------------------
torch.save({
    "clinical_head": clinical_head.state_dict(),
    "wsi_head": wsi_head.state_dict(),
    "mri_head": mri_head.state_dict(),
    "ct_head": ct_head.state_dict(),
    "fusion_weights": fusion_weights.detach(),
}, "models/late_fusion.pth")

print("Saved → models/late_fusion.pth")
