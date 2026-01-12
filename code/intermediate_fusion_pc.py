import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, confusion_matrix

from cirkit.pipeline import compile
from cirkit.templates.region_graph import RandomBinaryTree
from cirkit.templates.utils import Parameterization, parameterization_to_factory
from cirkit.symbolic.layers import GaussianLayer
from cirkit.symbolic.parameters import mixing_weight_factory

import numpy as np

DATA_PATH = "data/processed_ccRCC.pt"
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 30

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# Load Dataset
# ==========================================================
data = torch.load(DATA_PATH)

train = data["train"]
test = data["test"]

clinical_train = train["clinical"]
clinical_test = test["clinical"]

wsi_train = train["wsi"]
wsi_test = test["wsi"]

mri_train = train["mri"]
mri_test = test["mri"]

ct_train = train["ct"]
ct_test = test["ct"]

y_train = train["labels"].long().view(-1)
y_test = test["labels"].long().view(-1)

print("Label distribution TRAIN:", torch.bincount(y_train))
print("Label distribution TEST :", torch.bincount(y_test))

# ==========================================================
# Define Encoders (Joint Training)
# ==========================================================
class ClinicalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(17, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )

    def forward(self, x):
        return self.net(x)

class WSIEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 16)
        )

    def forward(self, x):
        return self.net(x)

class MRIEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 16)
        )

    def forward(self, x):
        return self.net(x)

class CTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 16)
        )

    def forward(self, x):
        return self.net(x)

# Instantiate encoders
clinical_encoder = ClinicalEncoder().to(device)
wsi_encoder = WSIEncoder().to(device)
mri_encoder = MRIEncoder().to(device)
ct_encoder = CTEncoder().to(device)

# ==========================================================
# Fuse embedding + LayerNorm
# ==========================================================
layernorm = nn.LayerNorm(56).to(device)

def fuse_embeddings(c, w, m, ct):
    fused = torch.cat([c, w, m, ct], dim=1)
    return layernorm(fused)

# ==========================================================
# Build Probabilistic Circuit with 56 Gaussian variables
# ==========================================================
NUM_VARS = 56
NUM_INPUT_UNITS = 64
NUM_SUM_UNITS = 64

rg = RandomBinaryTree(NUM_VARS, depth=None, num_repetitions=1)

# Gaussian input factory
def input_factory(scope, num_units):
    return GaussianLayer(scope=scope, num_output_units=num_units)

sum_param = Parameterization(activation="softmax", initialization="normal")
sum_weight_factory = parameterization_to_factory(sum_param)
nary_sum_weight_factory = lambda *args, **kwargs: mixing_weight_factory(
    *args, param_factory=sum_weight_factory, **kwargs
)

symbolic_pc = rg.build_circuit(
    input_factory=input_factory,
    sum_weight_factory=sum_weight_factory,
    nary_sum_weight_factory=nary_sum_weight_factory,
    num_input_units=NUM_INPUT_UNITS,
    num_sum_units=NUM_SUM_UNITS,
    sum_product="cp"
)

pc = compile(symbolic_pc).to(device)

# PC outputs log-likelihood per class
#cls_head = nn.Linear(pc.output_dim, 2).to(device)
cls_head = nn.Linear(1, 2).to(device)


# ==========================================================
# Dataloaders
# ==========================================================
train_loader = DataLoader(
    TensorDataset(clinical_train, wsi_train, mri_train, ct_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(clinical_test, wsi_test, mri_test, ct_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================================================
# Training Setup
# ==========================================================
params = list(pc.parameters()) + \
         list(clinical_encoder.parameters()) + \
         list(wsi_encoder.parameters()) + \
         list(mri_encoder.parameters()) + \
         list(ct_encoder.parameters()) + \
         list(layernorm.parameters()) + \
         list(cls_head.parameters())

optimizer = optim.Adam(params, lr=LR)
criterion = nn.CrossEntropyLoss()

# ==========================================================
# Training Loop
# ==========================================================
for epoch in range(EPOCHS):
    pc.train()
    total_loss = 0

    for i, (c, w, m, ct, y) in enumerate(train_loader):
        c, w, m, ct, y = c.to(device), w.to(device), m.to(device), ct.to(device), y.to(device)

        optimizer.zero_grad()

        zc = clinical_encoder(c)
        zw = wsi_encoder(w)
        zm = mri_encoder(m)
        zct = ct_encoder(ct)

        fused = fuse_embeddings(zc, zw, zm, zct)

        ll = pc(fused)             # (batch, dim)
        ll = ll.view(ll.size(0), -1) 
        # DEBUG: print PC output shape once
        if epoch == 0 and i == 0:
            print("PC output shape:", ll.shape)
        logits = cls_head(ll)      # map to 2-class logits

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Evaluation
    pc.eval()
    preds_list = []
    labels_list = []

    with torch.no_grad():
    
        for c, w, m, ct, y in test_loader:
            c = c.to(device)
            w = w.to(device)
            m = m.to(device)
            ct = ct.to(device)
            y = y.to(device)
    
            zc = clinical_encoder(c)
            zw = wsi_encoder(w)
            zm = mri_encoder(m)
            zct = ct_encoder(ct)
    
            fused = fuse_embeddings(zc, zw, zm, zct)
    
            ll = pc(fused)
            ll = ll.view(ll.size(0), -1)      # flatten (batch,1,1) → (batch,1)
    
            logits = cls_head(ll)             # (batch,2)
            preds = logits.argmax(dim=1)      # (batch,)
    
            preds_list.append(preds.cpu())
            labels_list.append(y.cpu())

    preds = torch.cat(preds_list, dim=0)       # (N,)
    labels = torch.cat(labels_list, dim=0)     # (N,)

    acc = (preds == labels).float().mean().item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Test Acc: {acc:.4f}")

# ==========================================================
# Save model components
# ==========================================================
torch.save(
    {
        "pc": pc.state_dict(),
        "clinical_encoder": clinical_encoder.state_dict(),
        "wsi_encoder": wsi_encoder.state_dict(),
        "mri_encoder": mri_encoder.state_dict(),
        "ct_encoder": ct_encoder.state_dict(),
        "layernorm": layernorm.state_dict(),
        "cls_head": cls_head.state_dict(),
    },
    "models/intermediate_fusion_pc.pth"
)

print("Saved → models/intermediate_fusion_pc.pth")
