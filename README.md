- This vector is passed directly into a neural network (MLP).
- The network outputs a prediction: **Alive vs Dead**.

### What happens when you run it
- Trains a neural network end-to-end
- Prints **test accuracy per epoch**
- Saves the model to `models/early_fusion.pth`

### Why performance looks high
- The dataset is **highly imbalanced** (many more patients in one class).
- Accuracy can look very high even if the model mostly predicts the majority class.
- Early fusion has access to *all raw features*, so it can easily learn shortcuts.

---

## 2. Intermediate Fusion with Probabilistic Circuits

**File:** `code/intermediate_fusion_pc.py`

### What it does
1. Each modality is first compressed using a small neural network:
 - Clinical → 8 dims
 - WSI → 16 dims
 - MRI → 16 dims
 - CT → 16 dims
2. These embeddings are concatenated into a **56-dimensional fused representation**.
3. A **Probabilistic Circuit (PC)** models the joint distribution of these features.
4. The PC outputs a **log-likelihood score**.
5. A small classifier uses this score to predict survival.

### What happens when you run it
- Trains encoders + probabilistic circuit jointly
- Prints **test accuracy per epoch**
- Saves all components to `models/intermediate_fusion_pc.pth`

### Why performance plateaus
- The PC compresses all information into **one scalar** (log-likelihood).
- This limits how much class-specific detail reaches the classifier.
- With class imbalance, the model can converge to majority-class predictions.

---

## 3. Late Fusion (Weighted Voting)

**File:** `code/late_fusion.py`

### What it does
- Trains **separate classifiers** for:
- Clinical
- WSI
- MRI
- CT
- Each classifier produces its own prediction.
- A set of **learnable fusion weights** decides how much to trust each modality.
- Missing modalities are automatically ignored using masks.

### What happens when you run it
- Prints:
- Training loss
- **AUROC** (better than accuracy for imbalanced data)
- Learned fusion weights per epoch
- Outputs a **confusion matrix**
- Saves the model to `models/late_fusion.pth`

### Why AUROC is used
- Accuracy can be misleading with imbalanced datasets.
- AUROC measures how well the model ranks high-risk vs low-risk patients.

---

## Key Takeaways

- **Early fusion** is simple and powerful, but prone to shortcut learning.
- **Intermediate fusion** enforces structured reasoning but may over-compress information.
- **Late fusion** is interpretable and robust to missing data, but depends on label quality.

This repository is designed to **compare behaviors**, not just maximize metrics.

---

## Notes

- Large data files and trained models are excluded from GitHub.
- Code is structured for clarity and reproducibility.
- Intended for research and thesis work.

---

## Author
Arnav Arora

