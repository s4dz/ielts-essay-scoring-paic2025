# Automated Essay Scoring (AES) System for IELTS Writing
### 🏆 1st Place Solution (Team Mango) – PTNK AI Challenge (PAIC 2026)

An end-to-end machine-learning pipeline that scores IELTS Writing Task 1 and Task 2 essays on the official band scale, up to 9.0, in half-band steps.

The system is a hybrid. A fine-tuned `DeBERTa-v3-large` model reads the prompt and the essay and predicts a band. That prediction, together with 22 hand-engineered linguistic features and their interactions, goes into an ensemble of gradient-boosted trees (XGBoost, LightGBM, CatBoost).

## 📊 Performance
* **Metric:** Mean Absolute Error (MAE)
* **Local 5-fold CV MAE:** `0.4150`. This is the out-of-fold MAE of the weighted ensemble, measured before half-band rounding.

---

## 🧠 Pipeline

The competition pipeline lives in `notebooks/PAIC2025.ipynb`, the exact notebook used during PAIC 2026. `src/` and `main.py` were split out of it after the competition. The notebook's four code cells run in order on Kaggle with a GPU. All stages share the same 5-fold `StratifiedKFold` split (seed 42, stratified by band), so every out-of-fold prediction lines up across stages.

### Stage 1: DeBERTa ordinal model (cell 1)
* **Input text:** `PROMPT: … [SEP] ESSAY: …`. For Task 1 the chart's image description is inserted as `IMAGE: …` between the two. Texts are truncated to 640 tokens.
* **Architecture:** `microsoft/deberta-v3-large`, followed by attention pooling over the last hidden states, followed by a linear head with **16 outputs**.
* **Ordinal decoding:** the scale from 1.0 to 9.0 in half-band steps has 17 bands, separated by 16 thresholds. Output *k* (k = 0, …, 15) is read through a sigmoid as the probability that the essay scores above 1.0 + 0.5k. The predicted band is

  $$\hat{y} = 1.0 + 0.5 \sum_{k=0}^{15} \sigma(z_k)$$

  If every probability is near 0, the prediction is 1.0. If every probability is near 1, it is 1.0 + 0.5 × 16 = **9.0**. Each output is a separate binary decision, which is why the model uses a sigmoid on each output rather than a softmax across them.
* **Out-of-fold predictions:** each of the five fold models predicts the essays it did not see in training. These predictions become the `deberta_pred` feature for the training set. Test predictions are the average of the five fold models.

### Stage 2: Linguistic features (cell 1, `src/features.py`)
There are 22 features in total:
* **Length and structure:** characters, words, sentences, paragraphs.
* **Punctuation:** counts of question marks, exclamation marks, semicolons and quotes.
* **Readability:** Flesch Reading Ease, Gunning Fog, SMOG, average sentence length, number of complex (3+ syllable) words.
* **Vocabulary:** unique words, type-token ratio, share of words longer than 6 characters.
* **Error proxies:** double spaces, a space before a period or comma, sentences that start with a lowercase letter, immediately repeated words.
* **Prompt relevance:** number and share of prompt words (after removing stopwords) that appear in the essay.

### Stage 3: Interaction features (cell 2, `src/features.py`)
Eleven key features are expanded: `deberta_pred`, word / sentence / paragraph counts, average sentence length, unique words, complex words, TTR, Flesch, Gunning Fog and SMOG.
* Squared terms: 11 features.
* Pairwise products, A × B: 55 features.
* Pairwise ratios in both directions, A / (B + 1e-6) and B / (A + 1e-6): 110 features.

This adds 176 features. Any infinite or missing value becomes 0.

### Stage 4: Bagged classifier meta-feature (cell 3)
* A LightGBM **multiclass** classifier treats each band as its own class.
* Each fold is trained with three seeds (42, 2024, 777), and their probabilities are averaged.
* The expected band, Σ P(band) × band, becomes the `lgb_cls_pred` feature.

### Stage 5: Final ensemble (cell 4)
* XGBoost, LightGBM and CatBoost regressors are each trained with an MAE objective, learning rate 0.01 and early stopping after 100 rounds.
* **Blending weights:** found with SciPy `minimize` (SLSQP) to minimize out-of-fold MAE. The weights are between 0 and 1 and sum to 1.
* **Post-processing:** predictions are clipped to [0.5, 9.0] and rounded to the nearest half band: `round(x × 2) / 2`.

---

## 📂 Repository layout
| Path | Contents |
| :-- | :-- |
| `notebooks/PAIC2025.ipynb` | The full competition pipeline: four cells, run on Kaggle |
| `src/features.py` | Stage 2 and Stage 3 feature code, moved out of the notebook |
| `src/models_nlp.py`, `src/models_tabular.py`, `src/utils.py` | **Empty for now.** The DeBERTa model, the bagging step and the weight optimization still need to be moved here from the notebook |
| `main.py` | Single-script version of the pipeline. It imports from the three empty files above, so it **does not run yet** |
| `data/sample_essays.csv` | A sample showing the input format |
| `requirements.txt` | Python dependencies |

## 🛠️ Reproducing the results
1. Open `notebooks/PAIC2025.ipynb` on Kaggle with a GPU.
2. Attach two inputs:
   * the competition data (`public_train.csv`, `private_train.csv`, `private_test.csv`) under `/kaggle/input/ieltsdata/`;
   * the fine-tuned DeBERTa weights for the five folds (`model_fold_{0..4}.pth`) under `/kaggle/input/large-sigma-private/`.
3. Run the four cells in order. The last cell writes `submission_enhanced.csv`.

To run locally:
```bash
git clone https://github.com/s4dz/ielts-essay-scoring-paic2025.git
cd ielts-essay-scoring-paic2025
pip install -r requirements.txt
```
Then change the `/kaggle/...` paths at the top of each cell to your local paths.

## ⚠️ Limitations
* This repository does not include the training run that produced the final DeBERTa weights, the weights themselves or the competition data. Only inference with existing weights and the tabular stages can be reproduced from it. An earlier training notebook is included; see below.
* The MAE above is measured before the final rounding to half bands.

## 🧪 Development notebook

`notebooks/deberta_training_dev.ipynb` is Khoa's DeBERTa training notebook from before the final run (Kaggle notebook `IELTSv2`, version 9, run completed 12 January 2026). It trains `deberta-v3-large` with attention pooling and the same 16-threshold ordinal target, plus:
* an auxiliary regression head, trained jointly (BCE on the 16 thresholds + 0.5 × MSE on the band);
* layer-wise learning-rate decay (0.9 per layer, 5× learning rate on the heads);
* adversarial weight perturbation (AWP) from the third epoch;
* a best-checkpoint per fold, selected on validation MAE.

On 5-fold CV over the public training set, the best validation MAE per fold was 0.51–0.53 (before rounding). The `Final CV MAE: 1.4949` printed at the end is a bug, explained at the top of the notebook.

This notebook did **not** produce the weights used in `PAIC2025.ipynb`. Those weights load into a model with a single 16-output head (`fc`), whereas this notebook has two heads (`fc_ordinal`, `fc_reg`). It also trains on `public_train.csv` only.

---

Part of Team Mango's winning entry at PAIC 2026 (January 2026). Repository maintained by Khoa Phan.
