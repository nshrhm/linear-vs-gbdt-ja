#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
機械学習モデルの比較実験 - 実データでの解釈性比較
UCIドイツ信用データセットを使用した分類タスクでの解釈性評価
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import shap
import os
import warnings
import optuna
from scipy import stats

# 特定の警告を抑制
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', message='LightGBM binary classifier with TreeExplainer')
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore', category=FutureWarning, module='optuna')

# データ取得と前処理
from sklearn.datasets import fetch_openml # ucimlrepoの代わりにfetch_openmlを使用
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# モデル
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier # Random Forestを追加
import xgboost as xgb
import lightgbm as lgb
from lightgbm.callback import early_stopping

# 評価指標
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss, classification_report # Brierスコアを追加

# plotting.pyからグラフ作成関数をインポート
from plotting import create_and_save_plot, create_extrapolation_plot, create_shap_summary_plot

def main():
    print("=== 機械学習モデル比較実験 (実データでの解釈性比較) ===")
    print("30回実行の平均値と標準偏差による統計的に信頼性の高い評価を実施します。")
    
    n_experiments = 30
    all_results = []
    
    print("\n1. データの読み込みと前処理...")
    try:
        # OpenMLからGerman Creditデータセットを取得
        german_credit = fetch_openml(data_id=31, as_frame=True, return_X_y=False) # nameの代わりにdata_idを使用
        X = german_credit.data
        y = german_credit.target.map({'good': 0, 'bad': 1}) # 'good'を0, 'bad'を1にマッピング
        print(f"データセット取得完了: {X.shape}")
    except Exception as e:
        print(f"OpenMLデータセットの取得に失敗しました: {e}。代替として人工データを生成します。")
        from sklearn.datasets import make_classification
        X_array, y_array = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, n_classes=2, random_state=42)
        feature_names = [f'num_feature_{i}' for i in range(10)] + [f'cat_feature_{i}' for i in range(10)]
        X = pd.DataFrame(X_array, columns=feature_names)
        for i in range(10, 20):
            X.iloc[:, i] = pd.cut(X.iloc[:, i], bins=3, labels=['low', 'medium', 'high'])
        y = pd.Series(y_array)
        print(f"人工データ生成完了: {X.shape}")
    
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=np.number).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Optunaによるハイパーパラメータチューニング
    tuned_params_xgb = {}
    tuned_params_lgbm = {}

    print("\n=== GBDTモデルのハイパーパラメータチューニング (Optuna) ===")
    # チューニング用のデータ生成 (最初の実験のシードを使用)
    current_seed_tune = 42
    X_train_tune, X_test_tune, y_train_tune, y_test_tune = train_test_split(X, y, test_size=0.2, random_state=current_seed_tune, stratify=y)
    
    preprocessor_fitted_tune = preprocessor.fit(X_train_tune)
    X_train_transformed_tune = preprocessor_fitted_tune.transform(X_train_tune)
    X_test_transformed_tune = preprocessor_fitted_tune.transform(X_test_tune)

    X_train_new_tune, X_val_tune, y_train_new_tune, y_val_tune = train_test_split(
        X_train_transformed_tune, y_train_tune, test_size=0.25, random_state=current_seed_tune, stratify=y_train_tune
    )

    def objective_xgb(trial):
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'seed': current_seed_tune,
            'n_jobs': -1,
        }
        model = xgb.XGBClassifier(**param, callbacks=[xgb.callback.EarlyStopping(rounds=10)])
        model.fit(X_train_new_tune, y_train_new_tune,
                  eval_set=[(X_val_tune, y_val_tune)],
                  verbose=False)
        y_pred_proba = model.predict_proba(X_test_transformed_tune)[:, 1]
        roc_auc = roc_auc_score(y_test_tune, y_pred_proba)
        return 1 - roc_auc # Minimize 1 - ROC AUC

    def objective_lgbm(trial):
        param = {
            'objective': 'binary',
            'metric': 'auc',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'random_state': current_seed_tune,
            'n_jobs': -1,
            'verbose': -1,
        }
        model = lgb.LGBMClassifier(**param)
        model.fit(X_train_new_tune, y_train_new_tune,
                  eval_set=[(X_val_tune, y_val_tune)],
                  callbacks=[early_stopping(stopping_rounds=10, verbose=False)])
        y_pred_proba = model.predict_proba(X_test_transformed_tune)[:, 1]
        roc_auc = roc_auc_score(y_test_tune, y_pred_proba)
        return 1 - roc_auc # Minimize 1 - ROC AUC

    # XGBoostチューニング
    print("XGBoostのハイパーパラメータチューニング中...")
    study_xgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=current_seed_tune))
    study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=True)
    tuned_params_xgb = study_xgb.best_params
    print(f"XGBoostの最適ハイパーパラメータ: {tuned_params_xgb}")

    # LightGBMチューニング
    print("LightGBMのハイパーパラメータチューニング中...")
    study_lgbm = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=current_seed_tune))
    study_lgbm.optimize(objective_lgbm, n_trials=50, show_progress_bar=True)
    tuned_params_lgbm = study_lgbm.best_params
    print(f"LightGBMの最適ハイパーパラメータ: {tuned_params_lgbm}")
    
    for experiment_id in range(n_experiments):
        print(f"\n{'='*20} 実験 {experiment_id + 1}/{n_experiments} {'='*20}")
        current_seed = 42 + experiment_id
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=current_seed, stratify=y)
        
        # ADDED: GBDTモデルの早期停止のために訓練データをさらに分割
        X_train_new, X_val, y_train_new, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=current_seed, stratify=y_train
        )
        
        print(f"データ分割完了。訓練: {X_train_new.shape}, 検証: {X_val.shape}, テスト: {X_test.shape}")
        
        print("\n2. モデルの定義...")
        models = {
            "Logistic Regression (L2)": LogisticRegression(penalty='l2', solver='liblinear', random_state=current_seed),
            "Logistic Regression (L1)": LogisticRegression(penalty='l1', solver='liblinear', random_state=current_seed),
            "Random Forest": RandomForestClassifier(random_state=current_seed, n_jobs=-1), # Random Forestを追加
            "XGBoost": xgb.XGBClassifier(random_state=current_seed, callbacks=[xgb.callback.EarlyStopping(rounds=10)], **tuned_params_xgb), # チューニング済みパラメータを適用
            "LightGBM": lgb.LGBMClassifier(random_state=current_seed, verbosity=-1, **tuned_params_lgbm) # チューニング済みパラメータを適用
        }
        
        pipelines = {name: Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)]) for name, model in models.items()}
        
        print("\n3. 実験の実行と評価...")
        experiment_results = []
        for name, pipeline in pipelines.items():
            print(f"--- {name} の評価を開始 ---")
            
            start_train_time = time.time()
            if name in ["XGBoost", "LightGBM"]:
                preprocessor_for_val = Pipeline(steps=[('preprocessor', preprocessor)])
                preprocessor_for_val.fit(X_train_new)
                X_val_transformed = preprocessor_for_val.transform(X_val)
                eval_set = [(X_val_transformed, y_val)] 
                
                if name == "XGBoost":
                    pipeline.fit(X_train_new, y_train_new, classifier__eval_set=eval_set, classifier__verbose=False)
                elif name == "LightGBM":
                    callbacks = [lgb.early_stopping(10, verbose=False)]
                    pipeline.fit(X_train_new, y_train_new, classifier__eval_set=eval_set, classifier__callbacks=callbacks)
            else:
                pipeline.fit(X_train, y_train)
            train_time = time.time() - start_train_time
            
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            
            experiment_results.append({
                "Model": name, 
                "Train Time (s)": train_time, 
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred), 
                "ROC AUC": roc_auc_score(y_test, y_prob),
                "Brier Score": brier_score_loss(y_test, y_prob) # Brierスコアを追加
            })
            print(f"--- {name} の評価が完了 ---")
        all_results.extend(experiment_results)

    print(f"\n4. 統計的評価（{n_experiments}回実験の平均値と標準偏差計算）...")
    all_results_df = pd.DataFrame(all_results)
    summary_results = []
    for model_name in all_results_df['Model'].unique():
        model_data = all_results_df[all_results_df['Model'] == model_name]
        
        mean_result = {
            "Model": model_name,
            "Train Time (s) Mean": model_data["Train Time (s)"].mean(),
            "Train Time (s) Std": model_data["Train Time (s)"].std(),
            "Accuracy Mean": model_data["Accuracy"].mean(),
            "Accuracy Std": model_data["Accuracy"].std(),
            "F1 Score Mean": model_data["F1 Score"].mean(),
            "F1 Score Std": model_data["F1 Score"].std(),
            "ROC AUC Mean": model_data["ROC AUC"].mean(),
            "ROC AUC Std": model_data["ROC AUC"].std(),
            "Brier Score Mean": model_data["Brier Score"].mean(),
            "Brier Score Std": model_data["Brier Score"].std(),
        }
        summary_results.append(mean_result)
        
        print(f"{model_name}: ROC AUC平均={mean_result['ROC AUC Mean']:.4f}±{mean_result['ROC AUC Std']:.4f}, 学習時間平均={mean_result['Train Time (s) Mean']:.4f}s")

    # ===================================================================
    # 5. 統計検定 (Wilcoxon符号順位検定)
    # ===================================================================
    print("\n5. 統計検定 (Wilcoxon符号順位検定)...")
    # 例: Logistic Regression (L2) vs. Tuned LightGBM のROC AUCを比較
    roc_auc_logreg_l2 = all_results_df[all_results_df['Model'] == 'Logistic Regression (L2)']['ROC AUC']
    roc_auc_lgbm = all_results_df[all_results_df['Model'] == 'LightGBM']['ROC AUC']
    
    if len(roc_auc_logreg_l2) == len(roc_auc_lgbm) and len(roc_auc_logreg_l2) > 1:
        stat, p_value = stats.wilcoxon(roc_auc_logreg_l2, roc_auc_lgbm, alternative='greater') # LogReg > LightGBM を仮説
        print(f"Logistic Regression (L2) vs. LightGBM (ROC AUC): Wilcoxon p-value = {p_value:.4f}")
        if p_value < 0.05:
            print("  -> Logistic Regression (L2)のROC AUCはLightGBMより統計的に有意に高いです (p < 0.05)。")
        else:
            print("  -> Logistic Regression (L2)のROC AUCはLightGBMより統計的に有意に高いとは言えません (p >= 0.05)。")
    else:
        print("Wilcoxon検定に必要なデータ数が不足しているか、データフレームの構造が異なります。")

    print("\n6. 結果の保存と可視化...")
    results_df = pd.DataFrame(summary_results).sort_values(by="ROC AUC Mean", ascending=False).reset_index(drop=True)
    results_df.to_csv("ex5_interpretability_summary_results.csv", index=False, encoding='utf-8')
    print("要約結果をex5_interpretability_summary_results.csvに保存しました。")
    all_results_df.to_csv("ex5_interpretability_all_results.csv", index=False, encoding='utf-8')
    print("全実験結果をex5_interpretability_all_results.csvに保存しました。")

    create_and_save_plot(all_results_df, language='en', experiment_name='ex5_performance')
    create_and_save_plot(all_results_df, language='ja', experiment_name='ex5_performance')

    print("\n7. 解釈性の比較...")
    # --- 7.1. 線形モデルの解釈 ---
    print("\n7.1. 線形モデル（ロジスティック回帰）の解釈...")
    # 最初の実験のデータでパイプラインを再フィットして係数を取得
    current_seed_for_coef = 42
    X_train_coef, _, y_train_coef, _ = train_test_split(X, y, test_size=0.2, random_state=current_seed_for_coef, stratify=y)
    
    log_reg_pipeline_coef = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(penalty='l2', solver='liblinear', random_state=current_seed_for_coef))])
    log_reg_pipeline_coef.fit(X_train_coef, y_train_coef)

    log_reg_model_coef = log_reg_pipeline_coef.named_steps['classifier']
    preprocessor_fitted_coef = log_reg_pipeline_coef.named_steps['preprocessor']
    
    try:
        ohe_feature_names = preprocessor_fitted_coef.named_transformers_['cat'].get_feature_names_out(categorical_features)
        feature_names_final = np.concatenate([numerical_features, ohe_feature_names])
    except Exception:
        feature_names_final = list(numerical_features)
        num_ohe_features = log_reg_model_coef.coef_.shape[1] - len(numerical_features)
        feature_names_final.extend([f'cat_feature_{i}' for i in range(num_ohe_features)])

    coef_df = pd.DataFrame({'Feature': feature_names_final, 'Coefficient': log_reg_model_coef.coef_[0]}).sort_values('Coefficient', ascending=False)
    coef_df.to_csv("ex5_logistic_coefficients.csv", index=False, encoding='utf-8')
    print("ロジスティック回帰係数をex5_logistic_coefficients.csvに保存しました。")
    create_and_save_plot(coef_df, language='en', experiment_name='ex5_coefficients')
    create_and_save_plot(coef_df, language='ja', experiment_name='ex5_coefficients')

    # --- 7.2. GDBTモデルの解釈 (LightGBM + SHAP) ---
    print("\n7.2. GDBTモデル（LightGBM + SHAP）の解釈...")
    # 最初の実験のデータでパイプラインを再フィットしてSHAP値を計算
    lgbm_pipeline_shap = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', lgb.LGBMClassifier(random_state=current_seed_for_coef, verbosity=-1, **tuned_params_lgbm))])
    
    X_train_shap, X_test_shap, y_train_shap, y_test_shap = train_test_split(X, y, test_size=0.2, random_state=current_seed_for_coef, stratify=y)
    X_train_new_shap, X_val_shap, y_train_new_shap, y_val_shap = train_test_split(
        X_train_shap, y_train_shap, test_size=0.25, random_state=current_seed_for_coef, stratify=y_train_shap
    )

    preprocessor_for_val_shap = Pipeline(steps=[('preprocessor', preprocessor)])
    preprocessor_for_val_shap.fit(X_train_new_shap)
    X_val_transformed_shap = preprocessor_for_val_shap.transform(X_val_shap)
    eval_set_shap = [(X_val_transformed_shap, y_val_shap)] 

    lgbm_pipeline_shap.fit(X_train_new_shap, y_train_new_shap, classifier__eval_set=eval_set_shap, classifier__callbacks=[early_stopping(10, verbose=False)])

    lgbm_model_shap = lgbm_pipeline_shap.named_steps['classifier']
    preprocessor_fitted_lgbm_shap = lgbm_pipeline_shap.named_steps['preprocessor']
    X_test_transformed_shap = preprocessor_fitted_lgbm_shap.transform(X_test_shap)
    X_test_transformed_df_shap = pd.DataFrame(X_test_transformed_shap, columns=feature_names_final)

    print("SHAP値の計算中...")
    try:
        explainer = shap.TreeExplainer(lgbm_model_shap)
        sample_size = min(100, len(X_test_transformed_df_shap))
        X_sample = X_test_transformed_df_shap.sample(n=sample_size, random_state=42)
        shap_values = explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        print("SHAP値の計算完了。")
        create_shap_summary_plot(shap_values, X_sample, language='en', experiment_name='ex5_shap')
        create_shap_summary_plot(shap_values, X_sample, language='ja', experiment_name='ex5_shap')
        
    except Exception as e:
        print(f"SHAP計算でエラーが発生しました: {e}")

    # 8. 考察の保存機能は削除
    print("\n=== 実験完了 ===")

if __name__ == "__main__":
    main()
