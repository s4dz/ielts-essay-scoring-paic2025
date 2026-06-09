# Automated Essay Scoring (AES) System for IELTS Writing
### 🏆 1st Place Solution – PTNK AI Challenge (PAIC 2025)

An advanced, end-to-end Machine Learning pipeline designed to objectively score IELTS writing tasks (Task 1 and Task 2) on the official 1.0 - 9.0 band scale. 

This system utilizes a hybrid architecture: leveraging a fine-tuned Deep Learning Transformer (`DeBERTa-v3-large`) for semantic representation, paired with a robust ensemble of Gradient Boosted Decision Trees (XGBoost, LightGBM, CatBoost) trained on over 35+ engineered linguistic features.

## 📊 Performance Summary
* **Primary Metric:** Mean Absolute Error (MAE)
* **Local 5-Fold CV MAE:** `0.4150`
* **Official IELTS Half-Band Mapping:** Implemented via custom clipping and rounding logic.

---

## 🧠 System Architecture & Methodology

The solution is engineered as a 5-stage sequential pipeline:

### Stage 1: Data Preparation & Linguistic Feature Engineering
Extracts **35+ distinct linguistic, structural, and stylistic attributes** from raw text:
* **Readability Statistics:** Flesch Reading Ease, Gunning Fog, and SMOG indices; sentence/word length profiles.
* **Vocabulary Richness:** Type-Token Ratio (TTR) and long-word (>6 chars) density.
* **Error Heuristics (Grammar Proxies):** Detection of mechanics errors (irregular spacing, punctuation anomalies, uncapitalized sentence starters).
* **Prompt Relevance:** Semantic overlap calculation between the essay and the writing prompt after stopword filtration.

### Stage 2: Deep Learning NLP Core (DeBERTa)
* **Base Model:** `microsoft/deberta-v3-large` fine-tuned on the raw essay text.
* **Head:** Custom **Attention Pooling layer** over the final hidden states to compress sequence length dynamically, feeding into a linear regression layer.
* **Ordinal Regression:** Modeled as a 16-logit classification problem with a sigmoid activation function. Continuous scores are aggregated via:
  $$Score = 1.0 + \left(\sum P(\text{class}) \times 0.5\right)$$
* **Validation:** 5-Fold Stratified K-Fold setup. Out-Of-Fold (OOF) predictions are saved as a meta-feature (`deberta_pred`).

### Stage 3: Advanced Tabular Feature Generation
To model non-linear interactions explicitly, **11 "Golden Features"** (including `deberta_pred` and key readability metrics) are transformed via:
* **Squared Terms:** $X^2$
* **Pairwise Interactions:** $A \times B$
* **Pairwise Ratios:** $A / (B + 1e-6)$ to discover composite metrics (e.g., error-to-length ratios). All infinite/NaN values are strictly sanitized to zero.

### Stage 4: Meta-Feature Generation via Bagging
* To prevent overfitting and capture discrete score class tendencies, a **LightGBM Multi-class Classifier** is introduced.
* Modeled across 3 distinct random seeds over 5 folds (Bagging).
* The resulting probability distributions are transformed back into a continuous expected score using a dot product, outputting the `lgb_cls_pred` feature.

### Stage 5: Final Ensemble & Weight Optimization
The tabular matrix (original features + engineered interactions + NLP OOF predictions + Bagging meta-features) is passed to three optimized regressors: **XGBoost, LightGBM, and CatBoost**.
* **Ensemble Blending:** Instead of uniform averaging, **SciPy's `minimize` API (SLSQP algorithm)** optimizes the blending weights to target minimum absolute error directly.
* **Post-Processing:** Predictions are clipped to $[1.0, 9.0]$, multiplied by 2, rounded to the nearest integer, and divided by 2 to align with standard IELTS half-band conventions.

---

## 📂 Repository Layout
* `src/features.py`: Linguistic feature extraction and mathematical interaction generators.
* `src/models_nlp.py`: PyTorch implementation of DeBERTa-v3 with Attention Pooling and ordinal regression loss.
* `src/models_tabular.py`: GBDT implementations (XGBoost, LightGBM, CatBoost) and bagging wrappers.
* `src/utils.py`: Post-processing, formatting scales, and SciPy optimization routines.
* `main.py`: The wrapper script to run training or inference locally.

## 🛠️ Installation & Usage

1. **Clone the repository:**
```bash
   git clone [https://github.com/s4dz/ielts-essay-scoring-paic2025.git](https://github.com/s4dz/ielts-essay-scoring-paic2025.git)
   cd ielts-essay-scoring-paic2025
```

2. **Install dependencies:**
```bash
   pip install -r requirements.txt
```

3. **Run Inference on Sample Data:**
```bash
   python main.py --mode predict --input data/sample_essays.csv
```

Developed by Khoa Phan as part of the PAIC 2025 Competition.
