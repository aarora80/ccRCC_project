# Multimodal Fusion for ccRCC Survival Prediction

This repository explores **how different multimodal fusion strategies combine heterogeneous medical data** to predict a binary clinical outcome for patients with clear cell Renal Cell Carcinoma (ccRCC).

Rather than optimizing for a single metric, the goal is to **understand how fusion design choices affect model behavior, robustness, and interpretability**

We compare three standard fusion paradigms used in multimodal machine learning:

1. **Early Fusion** – combine all features at once  
2. **Intermediate Fusion** – learn modality representations, then combine  
3. **Late Fusion** – combine predictions instead of features  

---

## Dataset Overview

Each patient has the following modalities:

- Clinical features  
- Whole Slide Image (WSI) embeddings  
- MRI embeddings  
- CT embeddings  
- Modality presence masks  
- Binary survival label 

**Important:**  
The dataset is **highly class-imbalanced**, which strongly affects how metrics such as accuracy should be interpreted.

---

## 1. Early Fusion (Single Network on All Features)

**File:** `code/early_fusion.py`

### Method

All available modalities are **concatenated into one large feature vector**, including:

- Clinical features  
- WSI embeddings  
- MRI embeddings  
- CT embeddings  
- Missing-modality masks  

This vector is passed directly into a **multi-layer perceptron (MLP)** that outputs a binary prediction:

> **Alive vs Not**

No modality structure is preserved after concatenation.

---

### What Happens When You Run It

- Trains a single neural network end-to-end  
- Prints **test accuracy per epoch**  
- Saves the trained model to:  


---

### Why Accuracy Appears Very High

Early fusion often reports **very high accuracy (~95%)**, but this must be interpreted carefully:

- The dataset is **strongly imbalanced**
- A model can achieve high accuracy by mostly predicting the majority class
- Early fusion exposes *all raw features at once*, allowing the model to:
- Learn shortcuts  
- Exploit spurious correlations  
- Overfit dataset-specific patterns  

**Takeaway:**  
Early fusion is powerful but **least constrained**, making it prone to misleadingly high performance.

---

## 2. Intermediate Fusion with Probabilistic Circuits

**File:** `code/intermediate_fusion_pc.py`

### Method

Each modality is first **compressed into a latent embedding** using a small neural network:

- Clinical → 8 dimensions  
- WSI → 16 dimensions  
- MRI → 16 dimensions  
- CT → 16 dimensions  

These embeddings are concatenated into a **56-dimensional fused representation**.

This representation is passed into a **Probabilistic Circuit (PC)**, which models the **joint distribution** of the fused features rather than directly predicting a label.

The PC outputs a **log-likelihood score**, which is then mapped to a binary prediction using a small classifier head.

---

### What Happens When You Run It

- Jointly trains:
- Modality-specific encoders  
- The probabilistic circuit  
- The classifier head  
- Prints **test accuracy per epoch**  
- Saves all components to:  


---

### Why Performance Plateaus

This behavior is expected and informative:

- The probabilistic circuit compresses all information into **a single scalar**
- This limits how much class-specific information reaches the classifier
- Under class imbalance, the model may converge to majority-class predictions

**Takeaway:**  
Intermediate fusion enforces **structured, probabilistic reasoning**, trading raw predictive power for interpretability and theoretical guarantees.

---

## 3. Late Fusion (Weighted Prediction Aggregation)

**File:** `code/late_fusion.py`

### Method

Each modality is modeled **independently** using its own classifier:

- Clinical classifier  
- WSI classifier  
- MRI classifier  
- CT classifier  

Each classifier produces logits.  
A set of **learnable fusion weights** determines how much each modality contributes to the final prediction.

Missing modalities are handled automatically using **presence masks**.

---

### What Happens When You Run It

- Trains four independent classifiers  
- Learns modality importance weights  
- Prints per epoch:
- Training loss  
- **AUROC** (preferred for imbalanced data)  
- Learned fusion weights  
- Outputs a **confusion matrix**  
- Saves the model to:  


---

### Why AUROC Is Used

- Accuracy is misleading for imbalanced datasets  
- **AUROC** measures how well the model:
- Ranks high-risk vs low-risk patients  
- Separates classes independent of a threshold  

**Takeaway:**  
Late fusion is **interpretable, robust, and clinically meaningful**, especially when modalities are inconsistently available.

---

## Fusion Strategy Comparison

| Fusion Type | Strengths | Limitations |
|------------|----------|-------------|
| Early Fusion | High raw accuracy, simple | Overfitting, shortcut learning |
| Intermediate Fusion | Probabilistic, structured, explainable | Information compression |
| Late Fusion | Robust, interpretable, handles missing data | Depends on modality quality |

---

## Project Philosophy

This repository is designed to **compare fusion behaviors**, not to maximize leaderboard metrics.

Performance should be interpreted **in context**, with attention to:

- Class imbalance  
- Model inductive bias  
- Clinical interpretability  

---

## Notes

- Large datasets and trained models are excluded from GitHub  
- Code is structured for clarity and reproducibility  

---

## Author

**Arnav Arora**

