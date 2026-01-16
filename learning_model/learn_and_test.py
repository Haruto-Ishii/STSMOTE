import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from strong_model.tabm_learning import main as tabm_main
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def log_all_metrics(model_name: str, y_true, y_pred, y_score, n_classes, minority_class_int):
    
    y_true_flat = np.array(y_true).flatten()
    y_pred_flat = np.array(y_pred).flatten()

    is_min_true = (y_true_flat == minority_class_int)
    is_min_pred = (y_pred_flat == minority_class_int)

    cnt_min_as_maj = np.sum(is_min_true & (~is_min_pred))

    cnt_maj_as_min = np.sum((~is_min_true) & is_min_pred)

    tp_min = np.sum(is_min_true & is_min_pred)
    total_min_true = np.sum(is_min_true)
    total_min_pred = np.sum(is_min_pred)

    recall_min = tp_min / (total_min_true + 1e-9)
    precision_min = tp_min / (total_min_pred + 1e-9)
    f1_minority = 2 * (precision_min * recall_min) / (precision_min + recall_min + 1e-9)

    print(f"[{model_name}] F1(Min): {f1_minority:.4f}, FN: {cnt_min_as_maj}, FP: {cnt_maj_as_min}")
    
    return {
        "f1_minority": f1_minority,
        "min_as_maj": cnt_min_as_maj,
        "maj_as_min": cnt_maj_as_min
    }

def run_training_evaluation(X_train, y_train, X_test, y_test, num_numerical_features, minority_class_int, random_state):
    
    numerical_idx = list(range(num_numerical_features))
    categorical_idx = list(range(num_numerical_features, X_train.shape[1]))
    n_classes = len(np.unique(y_train))
    
    all_model_results = {}
    
    df_train = pd.DataFrame(X_train)
    df_test = pd.DataFrame(X_test)
    
    for col in numerical_idx:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(0).astype(float)
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce').fillna(0).astype(float)
        
    for col in categorical_idx:
        df_train[col] = df_train[col].astype('str').astype('category')
        df_test[col] = df_test[col].astype('str').astype('category')
        
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
        
    print("\n--- 1. CatBoost ---")
    model_cat = CatBoostClassifier(
        task_type="GPU",
        devices='0',
        logging_level='Silent',
        allow_writing_files=False,
        random_state=random_state
    )
    
    model_cat.fit(df_train, y_train, cat_features=categorical_idx)
    
    y_pred_cat = model_cat.predict(df_test).flatten()
    y_score_cat = model_cat.predict_proba(df_test)
    
    res_cat = log_all_metrics("CatBoost", y_test, y_pred_cat, y_score_cat, n_classes, minority_class_int)
    all_model_results["CatBoost"] = res_cat
    
    print("\n--- 2. XGBoost ---")
    model_xgb = XGBClassifier(
        device='cuda',
        eval_metric='mlogloss',
        objective='multi:softmax',
        enable_categorical=True,
        num_classes=n_classes,
        random_state=random_state
    )
    
    model_xgb.fit(df_train, y_train)
    
    y_pred_xgb = model_xgb.predict(df_test)
    y_score_xgb = model_xgb.predict_proba(df_test)
    
    all_model_results["XGBoost"] = log_all_metrics(
        "XGBoost", y_test, y_pred_xgb, y_score_xgb, n_classes, minority_class_int
    )
    
    print("\n--- 3. LightGBM ---")
    model_lgbm = LGBMClassifier(
        device='cpu',
        verbose=-1,
        random_state=random_state
    )
    
    model_lgbm.fit(df_train, y_train, categorical_feature=categorical_idx)
    
    y_pred_lgbm = model_lgbm.predict(df_test)
    y_score_lgbm = model_lgbm.predict_proba(df_test)
    
    all_model_results["LightGBM"] = log_all_metrics(
        "LightGBM", y_test, y_pred_lgbm, y_score_lgbm, n_classes, minority_class_int
    )
    
    print("\n--- 4. TabM ---")
    
    y_pred_tabm, y_score_tabm = tabm_main(X_train, y_train, X_test, y_test, num_numerical_features, random_state)
    res_tabm = log_all_metrics("TabM", y_test, y_pred_tabm, y_score_tabm, n_classes, minority_class_int)
    all_model_results["TabM"] = res_tabm

    return all_model_results