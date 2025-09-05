#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
機械学習モデルの比較実験 - 小規模データでの性能比較
サンプル数が少ない状況での過学習への耐性を評価
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import os
import warnings
import optuna
from scipy import stats

# 特定の警告を抑制
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', message='LightGBM binary classifier with TreeExplainer')
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore', category=FutureWarning, module='optuna')

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor # Random Forestを追加

import xgboost as xgb
import lightgbm as lgb
from lightgbm.callback import early_stopping

# plotting.pyからグラフ作成関数をインポート
from plotting import create_and_save_plot, create_extrapolation_plot, create_shap_summary_plot

def main():
    print("=== 機械学習モデル比較実験 (小規模データ) ===")
    print("30回実行の平均値と標準偏差による統計的に信頼性の高い評価を実施します。")
    
    # 実験設定
    n_experiments = 30
    all_results = []

    # 正則化パラメータalphaの探索範囲を定義
    alphas = np.logspace(-4, 4, 100)

    # Optunaによるハイパーパラメータチューニング
    tuned_params_xgb = {}
    tuned_params_lgbm = {}

    print("\n=== GBDTモデルのハイパーパラメータチューニング (Optuna) ===")
    # チューニング用のデータ生成 (最初の実験のシードを使用)
    current_seed_tune = 42
    X_tune, y_tune = make_regression(
        n_samples=100,
        n_features=200,
        n_informative=10,
        noise=30.0,
        random_state=current_seed_tune
    )
    X_train_tune, X_test_tune, y_train_tune, y_test_tune = train_test_split(X_tune, y_tune, test_size=0.2, random_state=current_seed_tune)
    feature_names_tune = [f'feature_{i}' for i in range(X_tune.shape[1])]
    X_train_df_tune = pd.DataFrame(X_train_tune, columns=feature_names_tune)
    X_test_df_tune = pd.DataFrame(X_test_tune, columns=feature_names_tune)
    X_train_new_tune, X_val_tune, y_train_new_tune, y_val_tune = train_test_split(
        X_train_df_tune, y_train_tune, test_size=0.25, random_state=current_seed_tune # 80*0.25=20 for val
    )

    def objective_xgb(trial):
        param = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
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
        model = xgb.XGBRegressor(**param, callbacks=[xgb.callback.EarlyStopping(rounds=10)])
        model.fit(X_train_new_tune, y_train_new_tune,
                  eval_set=[(X_val_tune, y_val_tune)],
                  verbose=False)
        y_pred = model.predict(X_test_tune)
        rmse = np.sqrt(mean_squared_error(y_test_tune, y_pred))
        return rmse

    def objective_lgbm(trial):
        param = {
            'objective': 'regression',
            'metric': 'rmse',
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
        model = lgb.LGBMRegressor(**param)
        model.fit(X_train_new_tune, y_train_new_tune,
                  eval_set=[(X_val_tune, y_val_tune)],
                  callbacks=[early_stopping(stopping_rounds=10, verbose=False)])
        y_pred = model.predict(X_test_tune)
        rmse = np.sqrt(mean_squared_error(y_test_tune, y_pred))
        return rmse

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
    
    # ===================================================================
    # 複数回実験の実行
    # ===================================================================
    for experiment_id in range(n_experiments):
        print(f"\n{'='*20} 実験 {experiment_id + 1}/{n_experiments} {'='*20}")
        
        # ===================================================================
        # 1. 人工データの生成 (Small-Sample Data)
        # ===================================================================
        print(f"\n1. 小規模データの生成中... (実験 {experiment_id + 1})")
        print("サンプル数が少なく、特徴量が多いデータを生成し、過学習しやすい状況を作成")
        
        current_seed = 42 + experiment_id
        
        X, y = make_regression(
            n_samples=100,
            n_features=200,
            n_informative=10,
            noise=30.0,
            random_state=current_seed
        )
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=current_seed)
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # ADDED: GBDTモデルの早期停止のために訓練データをさらに分割
        X_train_new, X_val, y_train_new, y_val = train_test_split(
            X_train_df, y_train, test_size=0.25, random_state=current_seed # 80*0.25=20 for val
        )
        
        print(f"データ生成完了。")
        print(f"全データ: {X.shape}, 訓練データ: {X_train.shape}, テストデータ: {X_test.shape}")
        print(f"GBDT訓練データ: {X_train_new.shape}, GBDT検証データ: {X_val.shape}")
        
        # ===================================================================
        # 2. モデルの定義
        # ===================================================================
        print("\n2. モデルの定義...")
        
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": RidgeCV(alphas=alphas, cv=5),
            "Lasso": LassoCV(alphas=alphas, cv=5, random_state=current_seed, max_iter=10000),
            "Random Forest": RandomForestRegressor(random_state=current_seed, n_jobs=-1), # Random Forestを追加
            "XGBoost": xgb.XGBRegressor(random_state=current_seed, callbacks=[xgb.callback.EarlyStopping(rounds=10)], **tuned_params_xgb), # チューニング済みパラメータを適用
            "LightGBM": lgb.LGBMRegressor(random_state=current_seed, verbosity=-1, **tuned_params_lgbm) # チューニング済みパラメータを適用
        }
        
        # ===================================================================
        # 3. 実験の実行と評価
        # ===================================================================
        print("\n3. 実験の実行と評価...")
        
        experiment_results = []
        
        for name, model in models.items():
            print(f"--- {name} の評価を開始 ---")
            
            # --- 学習と予測 ---
            if name in ["Linear Regression", "Ridge", "Lasso", "Random Forest"]:
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
            else: # GBDT Models
                if name == "XGBoost":
                    model.fit(X_train_new, y_train_new,
                              eval_set=[(X_val, y_val)],
                              verbose=False)
                elif name == "LightGBM":
                    model.fit(X_train_new, y_train_new,
                              eval_set=[(X_val, y_val)],
                              callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)])
                
                y_pred_train = model.predict(X_train_new)
                y_pred_test = model.predict(X_test_df)
            
            # --- 訓練データとテストデータの両方で評価 ---
            actual_y_train = y_train_new if name in ["XGBoost", "LightGBM"] else y_train
            rmse_train = np.sqrt(mean_squared_error(actual_y_train, y_pred_train))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            r2_train = r2_score(actual_y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            
            overfitting_factor = rmse_test / rmse_train if rmse_train > 0 else float('inf')
            
            # 結果をリストに追加
            experiment_results.append({
                "Experiment": experiment_id + 1,
                "Model": name,
                "RMSE (Train)": rmse_train,
                "RMSE (Test)": rmse_test,
                "R2 (Train)": r2_train,
                "R2 (Test)": r2_test,
                "Overfitting": overfitting_factor
            })
            print(f"--- {name} の評価が完了 ---")
        
        all_results.extend(experiment_results)
    
    # ===================================================================
    # 4. 統計的評価（30回実験の平均値と標準偏差計算）
    # ===================================================================
    print(f"\n4. 統計的評価（{n_experiments}回実験の平均値と標準偏差計算）...")
    
    all_results_df = pd.DataFrame(all_results)
    
    summary_results = []
    model_names = all_results_df['Model'].unique()
    
    for model_name in model_names:
        model_data = all_results_df[all_results_df['Model'] == model_name]
        mean_result = {
            "Model": model_name,
            "RMSE (Train) Mean": model_data["RMSE (Train)"].mean(),
            "RMSE (Train) Std": model_data["RMSE (Train)"].std(),
            "RMSE (Test) Mean": model_data["RMSE (Test)"].mean(),
            "RMSE (Test) Std": model_data["RMSE (Test)"].std(),
            "R2 (Train) Mean": model_data["R2 (Train)"].mean(),
            "R2 (Train) Std": model_data["R2 (Train)"].std(),
            "R2 (Test) Mean": model_data["R2 (Test)"].mean(),
            "R2 (Test) Std": model_data["R2 (Test)"].std(),
            "Overfitting Mean": model_data["Overfitting"].mean(),
            "Overfitting Std": model_data["Overfitting"].std(),
        }
        summary_results.append(mean_result)
        
        print(f"{model_name}: テストRMSE平均={mean_result['RMSE (Test) Mean']:.4f}±{mean_result['RMSE (Test) Std']:.4f}, 過学習係数平均={mean_result['Overfitting Mean']:.2f}±{mean_result['Overfitting Std']:.2f}")

    # ===================================================================
    # 5. 統計検定 (Wilcoxon符号順位検定)
    # ===================================================================
    print("\n5. 統計検定 (Wilcoxon符号順位検定)...")
    # 例: Lasso vs. Tuned LightGBM の過学習係数を比較 (Lassoの方が低いことを仮説)
    of_lasso = all_results_df[all_results_df['Model'] == 'Lasso']['Overfitting']
    of_lgbm = all_results_df[all_results_df['Model'] == 'LightGBM']['Overfitting']
    
    if len(of_lasso) == len(of_lgbm) and len(of_lasso) > 1:
        stat, p_value = stats.wilcoxon(of_lasso, of_lgbm, alternative='less') # Lasso < LightGBM を仮説
        print(f"Lasso vs. LightGBM (Overfitting Factor): Wilcoxon p-value = {p_value:.4f}")
        if p_value < 0.05:
            print("  -> Lassoの過学習係数はLightGBMより統計的に有意に低いです (p < 0.05)。")
        else:
            print("  -> Lassoの過学習係数はLightGBMより統計的に有意に低いとは言えません (p >= 0.05)。")
    else:
        print("Wilcoxon検定に必要なデータ数が不足しているか、データフレームの構造が異なります。")

    # ===================================================================
    # 6. 結果の保存と可視化
    # ===================================================================
    print("\n6. 結果の保存と可視化...")
    
    results_df = pd.DataFrame(summary_results).sort_values(by="RMSE (Test) Mean").reset_index(drop=True)
    
    results_df.to_csv("ex4_small_sample_summary_results.csv", index=False, encoding='utf-8')
    print("要約結果をex4_small_sample_summary_results.csvに保存しました。")
    
    all_results_df.to_csv("ex4_small_sample_all_results.csv", index=False, encoding='utf-8')
    print("全実験結果をex4_small_sample_all_results.csvに保存しました。")
    
    print("\n--- 要約実験結果 (小規模データ) ---")
    print(results_df.to_string(index=False))
    
    create_and_save_plot(all_results_df, language='en', experiment_name='ex4_train_test')
    create_and_save_plot(all_results_df, language='ja', experiment_name='ex4_train_test')
    
    create_and_save_plot(all_results_df, language='en', experiment_name='ex4_overfitting')
    create_and_save_plot(all_results_df, language='ja', experiment_name='ex4_overfitting')
    
    # 7. 考察の保存機能は削除
    print("\n=== 実験完了 ===")

if __name__ == "__main__":
    main()
