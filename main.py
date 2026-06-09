import os
import gc
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Import custom modules
from src.features import extract_base_features, add_advanced_features
from src.models_nlp import IELTS_Model, IELTSOrdinalDataset, SmartCollator
from src.models_tabular import generate_lgb_bagging_feature
from src.utils import optimize_weights, post_process_predictions

class Config:
    TRAIN_FILE_PUB = "/kaggle/input/ieltsdata/public_train.csv"
    TRAIN_FILE_PRIV = "/kaggle/input/ieltsdata/private_train.csv"
    TEST_FILE = "/kaggle/input/ieltsdata/private_test.csv"
    MODEL_NAME = "microsoft/deberta-v3-large" 
    MAX_LEN = 640
    FOLDS = 5
    BATCH_SIZE = 4
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_nlp_fn(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="Inferencing"):
            input_ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            logits = model(input_ids, mask)
            probs = torch.sigmoid(logits)
            pred_scores = 1.0 + (probs.sum(dim=1) * 0.5)
            preds.extend(pred_scores.cpu().numpy())
    return np.array(preds)

def main():
    print(f"🔹 Running Pipeline on Device: {Config.DEVICE}")
    
    # 1. LOAD AND MERGE DATA
    print("📂 Loading & Merging Datasets...")
    df_pub = pd.read_csv(Config.TRAIN_FILE_PUB)
    df_priv = pd.read_csv(Config.TRAIN_FILE_PRIV)
    df_test = pd.read_csv(Config.TEST_FILE)
    
    for df in [df_pub, df_priv, df_test]:
        df.columns = df.columns.str.strip()
        df.rename(columns={'prompt': 'Prompt', 'essay': 'Essay', 'Overall Score': 'Overall', 'image description': 'Image Description'}, inplace=True)
    
    df_train = pd.concat([df_pub, df_priv], axis=0).reset_index(drop=True)
    df_train.dropna(subset=['Prompt', 'Essay', 'Overall'], inplace=True)
    
    # 2. STAGE 2: GENERATE OOF NLP PREDICTIONS 
    df_train['deberta_pred'] = 0.0
    test_accum = []
    
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    collator = SmartCollator(tokenizer)
    skf = StratifiedKFold(n_splits=Config.FOLDS, shuffle=True, random_state=Config.SEED)
    y_stratify = df_train['Overall'].astype(str)
    
    print("\n🔮 Generating NLP Predictions via DeBERTa...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, y_stratify)):
        model_path = f"/kaggle/input/large-sigma-private/model_fold_{fold}.pth"
        if not os.path.exists(model_path):
            alt_path = f"/kaggle/input/best-model/pytorch/default/2/model_fold_{fold}.pth"
            if os.path.exists(alt_path): model_path = alt_path
            else: print(f"❌ Weight file for fold {fold} missing! Skipping."); continue
            
        model = IELTS_Model(Config.MODEL_NAME)
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        model.to(Config.DEVICE)
        if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
        
        # Validation OOF
        df_val = df_train.iloc[val_idx]
        val_loader = DataLoader(IELTSOrdinalDataset(df_val, tokenizer, Config.MAX_LEN), batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collator)
        df_train.loc[df_train.index[val_idx], 'deberta_pred'] = predict_nlp_fn(model, val_loader, Config.DEVICE)
        
        # Test Inference
        test_loader = DataLoader(IELTSOrdinalDataset(df_test, tokenizer, Config.MAX_LEN), batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collator)
        test_accum.append(predict_nlp_fn(model, test_loader, Config.DEVICE))
        
        del model; torch.cuda.empty_cache(); gc.collect()
        
    df_test['deberta_pred'] = np.mean(test_accum, axis=0) if test_accum else 0.0

    # 3. STAGE 1 & 3: LINGUISTIC + TABULAR INTERACTION ENGINE
    df_train = extract_base_features(df_train)
    df_test = extract_base_features(df_test)
    df_train, df_test = add_advanced_features(df_train, df_test)

    # 4. STAGE 4: BAGGING META-FEATURE GENERATION
    ignore = ['ID', 'Prompt', 'Image Description', 'Essay', 'Overall', 'TR', 'CC', 'LR', 'GRA', 'kfold', 'id']
    base_features = [c for c in df_train.columns if c not in ignore]
    df_train, df_test = generate_lgb_bagging_feature(df_train, df_test, base_features, folds=Config.FOLDS, kfold_seed=Config.SEED)

    # Update features to include the new bagged meta-feature
    final_features = [c for c in df_train.columns if c not in ignore]

    # 5. STAGE 5: FINAL ENSEMBLE TRAINING (XGB, LGB, CAT)
    print("\n⚡ Training Final GBDT Ensemble Models...")
    y = df_train['Overall'].values
    X = df_train[final_features]
    X_test = df_test[final_features]

    oof_preds = np.zeros((len(X), 3)) 
    test_preds = np.zeros((len(X_test), 3))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_stratify)):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        # XGBoost
        m1 = xgb.XGBRegressor(objective='reg:absoluteerror', tree_method='hist', device='cuda' if torch.cuda.is_available() else 'cpu', n_estimators=2000, learning_rate=0.01, max_depth=6, early_stopping_rounds=100)
        m1.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
        oof_preds[val_idx, 0] = m1.predict(X_val)
        test_preds[:, 0] += m1.predict(X_test) / Config.FOLDS

        # LightGBM
        m2 = lgb.LGBMRegressor(objective='mae', device='gpu' if torch.cuda.is_available() else 'cpu', n_estimators=2000, learning_rate=0.01, num_leaves=31, verbosity=-1)
        m2.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_preds[val_idx, 1] = m2.predict(X_val)
        test_preds[:, 1] += m2.predict(X_test) / Config.FOLDS

        # CatBoost
        m3 = CatBoostRegressor(loss_function='MAE', task_type='GPU' if torch.cuda.is_available() else 'CPU', iterations=2000, learning_rate=0.01, depth=6, verbose=0, early_stopping_rounds=100)
        m3.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        oof_preds[val_idx, 2] = m3.predict(X_val)
        test_preds[:, 2] += m3.predict(X_test) / Config.FOLDS

    # 6. OPTIMIZATION AND POST-PROCESSING
    best_weights = optimize_weights(oof_preds, y)
    final_test_preds = np.average(test_preds, axis=1, weights=best_weights)
    final_test_preds = post_process_predictions(final_test_preds)

    # Generate Final Submission File
    id_col = 'id' if 'id' in df_test.columns else ('ID' if 'ID' in df_test.columns else df_test.index)
    sub = pd.DataFrame({'id': df_test[id_col], 'score': final_test_preds})
    sub.to_csv("submission_enhanced.csv", index=False)
    print(f"\n🏆 SUCCESS: 'submission_enhanced.csv' generated with mean target score: {sub['score'].mean():.4f}")

if __name__ == "__main__":
    main()