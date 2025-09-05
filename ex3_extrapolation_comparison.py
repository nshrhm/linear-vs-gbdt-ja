#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
機械学習モデルの比較実験 - 外挿（Extrapolation）タスクでの性能比較
線形モデルとGDBTモデルの外挿性能の根本的な違いを評価
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib # 追加
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

# scikit-learnからモデルとメトリクスをインポート
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # Random Forestを追加

# GDBTモデルをインポート
import xgboost as xgb
import lightgbm as lgb
from lightgbm.callback import early_stopping

# plotting.pyからグラフ作成関数をインポート
from plotting import create_and_save_plot, create_extrapolation_plot, create_shap_summary_plot

def main():
    print("=== 機械学習モデル比較実験 (外挿タスク) ===")
    print("30回実行の平均値と標準偏差による統計的に信頼性の高い評価を実施します。")
    
    # 実験設定
    n_experiments = 30
    all_results = []
    all_predictions = {} # 最初の実験の予測結果のみ可視化用に保存

    # 正則化パラメータalphaの探索範囲を定義
    alphas = np.logspace(-4, 4, 100)  # 10^-4 から 10^4 までを100分割

    # Optunaによるハイパーパラメータチューニング
    tuned_params_xgb = {}
    tuned_params_lgbm = {}

    print("\n=== GBDTモデルのハイパーパラメータチューニング (Optuna) ===")
    # チューニング用のデータ生成 (最初の実験のシードを使用)
    current_seed_tune = 42
    np.random.seed(current_seed_tune)
    X_tune = np.linspace(-2, 2, 500).reshape(-1, 1)
    y_tune = 2 * X_tune.flatten() + 30 + np.random.normal(0, 10, 500)
    train_mask_tune = np.abs(X_tune.flatten()) <= 1
    X_train_tune, y_train_tune = X_tune[train_mask_tune], y_tune[train_mask_tune]
    feature_names_tune = ['feature_0']
    X_train_df_tune = pd.DataFrame(X_train_tune, columns=feature_names_tune)
    X_train_new_tune, X_val_tune, y_train_new_tune, y_val_tune = train_test_split(
        X_train_df_tune, y_train_tune, test_size=0.2, random_state=current_seed_tune
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
        y_pred = model.predict(X_train_tune) # 訓練データでのRMSEを最小化
        rmse = np.sqrt(mean_squared_error(y_train_tune, y_pred))
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
        y_pred = model.predict(X_train_tune) # 訓練データでのRMSEを最小化
        rmse = np.sqrt(mean_squared_error(y_train_tune, y_pred))
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
        # 1. 人工データの生成 (Extrapolation-Task Data)
        # ===================================================================
        print(f"\n1. 外挿タスク用データの生成中... (実験 {experiment_id + 1})")
        print("単純な線形データを生成し、訓練範囲外での予測性能を評価")
        
        # 実験ごとに異なるランダムシードを使用
        current_seed = 42 + experiment_id
        np.random.seed(current_seed)
        
        # 単一特徴量の線形データを作成
        X = np.linspace(-2, 2, 500).reshape(-1, 1)
        y = 2 * X.flatten() + 30 + np.random.normal(0, 10, 500)
        
        # データを訓練用（内挿範囲）とテスト用（外挿範囲）に分割
        train_mask = np.abs(X.flatten()) <= 1
        test_mask = np.abs(X.flatten()) > 1
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # LightGBMの警告を回避するため、DataFrameに変換
        feature_names = ['feature_0']
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        X_df = pd.DataFrame(X, columns=feature_names)

        # ADDED: GBDTモデルの早期停止のために訓練データをさらに分割
        X_train_new, X_val, y_train_new, y_val = train_test_split(
            X_train_df, y_train, test_size=0.2, random_state=current_seed
        )
        
        print(f"データ生成完了。")
        print(f"訓練データ範囲: X ∈ [-1, 1], テストデータ範囲: X ∈ [-2, -1) U (1, 2]")
        print(f"訓練データ: {X_train.shape}, テストデータ: {X_test.shape}")
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
        predictions = {}  # 可視化用の予測結果を保存
        
        for name, model in models.items():
            print(f"--- {name} の評価を開始 ---")
            
            # --- 学習時間の計測 ---
            start_train_time = time.time()
            if name in ["Linear Regression", "Ridge", "Lasso", "Random Forest"]:
                # 線形モデルとRandom Forestはnumpy配列を使用
                model.fit(X_train, y_train)
            elif name == "XGBoost":
                model.fit(X_train_new, y_train_new,
                          eval_set=[(X_val, y_val)],
                          verbose=False)
            elif name == "LightGBM":
                model.fit(X_train_new, y_train_new,
                          eval_set=[(X_val, y_val)],
                          callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)])
            train_time = time.time() - start_train_time
            
            # --- 推論時間の計測 ---
            start_pred_time = time.time()
            if name in ["Linear Regression", "Ridge", "Lasso", "Random Forest"]:
                y_pred = model.predict(X_test)
                predictions[name] = model.predict(X)  # 全範囲での予測
            else:
                y_pred = model.predict(X_test_df)
                predictions[name] = model.predict(X_df)  # 全範囲での予測
            pred_time = time.time() - start_pred_time
            
            # --- 予測精度の評価 ---
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)  # GDBTでは大きな負の値になることが予想される
            
            experiment_results.append({
                "Experiment": experiment_id + 1,
                "Model": name,
                "Train Time (s)": train_time,
                "Inference Time (s)": pred_time,
                "RMSE": rmse,
                "MAE": mae,
                "R2 Score": r2,
            })
            print(f"--- {name} の評価が完了 ---")
        
        # 実験結果を全体の結果に追加
        all_results.extend(experiment_results)
        if experiment_id == 0:  # 最初の実験の予測結果のみ可視化用に保存
            all_predictions = predictions
    
    # ===================================================================
    # 4. 統計的評価（30回実験の平均値と標準偏差計算）
    # ===================================================================
    print(f"\n4. 統計的評価（{n_experiments}回実験の平均値と標準偏差計算）...")
    
    # 全実験結果をDataFrameに変換
    all_results_df = pd.DataFrame(all_results)
    
    # モデルごとに平均値と標準偏差を計算
    summary_results = []
    model_names = all_results_df['Model'].unique()
    
    for model_name in model_names:
        model_data = all_results_df[all_results_df['Model'] == model_name]
        mean_result = {
            "Model": model_name,
            "Train Time (s) Mean": model_data["Train Time (s)"].mean(),
            "Train Time (s) Std": model_data["Train Time (s)"].std(),
            "Inference Time (s) Mean": model_data["Inference Time (s)"].mean(),
            "Inference Time (s) Std": model_data["Inference Time (s)"].std(),
            "RMSE Mean": model_data["RMSE"].mean(),
            "RMSE Std": model_data["RMSE"].std(),
            "MAE Mean": model_data["MAE"].mean(),
            "MAE Std": model_data["MAE"].std(),
            "R2 Score Mean": model_data["R2 Score"].mean(),
            "R2 Score Std": model_data["R2 Score"].std(),
        }
        summary_results.append(mean_result)
        
        print(f"{model_name}: RMSE平均={mean_result['RMSE Mean']:.4f}±{mean_result['RMSE Std']:.4f}, R2平均={mean_result['R2 Score Mean']:.4f}±{mean_result['R2 Score Std']:.4f}")

    # ===================================================================
    # 5. 統計検定 (Wilcoxon符号順位検定)
    # ===================================================================
    print("\n5. 統計検定 (Wilcoxon符号順位検定)...")
    # 例: Lasso vs. Tuned LightGBM のR2スコアを比較
    r2_lasso = all_results_df[all_results_df['Model'] == 'Lasso']['R2 Score']
    r2_lgbm = all_results_df[all_results_df['Model'] == 'LightGBM']['R2 Score']
    
    if len(r2_lasso) == len(r2_lgbm) and len(r2_lasso) > 1:
        stat, p_value = stats.wilcoxon(r2_lasso, r2_lgbm, alternative='greater') # Lasso > LightGBM を仮説
        print(f"Lasso vs. LightGBM (R2 Score): Wilcoxon p-value = {p_value:.4f}")
        if p_value < 0.05:
            print("  -> LassoのR2スコアはLightGBMより統計的に有意に高いです (p < 0.05)。")
        else:
            print("  -> LassoのR2スコアはLightGBMより統計的に有意に高いとは言えません (p >= 0.05)。")
    else:
        print("Wilcoxon検定に必要なデータ数が不足しているか、データフレームの構造が異なります。")

    # ===================================================================
    # 6. 結果の保存と可視化
    # ===================================================================
    print("\n6. 結果の保存と可視化...")
    
    # 平均値と標準偏差の結果をDataFrameに変換
    summary_results_df = pd.DataFrame(summary_results).sort_values(by="RMSE Mean").reset_index(drop=True)
    
    # CSVファイルとして保存（要約結果）
    summary_results_df.to_csv("ex3_extrapolation_summary_results.csv", index=False, encoding='utf-8')
    print("要約結果をex3_extrapolation_summary_results.csvに保存しました。")
    
    # 全実験結果も保存
    all_results_df.to_csv("ex3_extrapolation_all_results.csv", index=False, encoding='utf-8')
    print("全実験結果をex3_extrapolation_all_results.csvに保存しました。")
    
    # 結果の表示
    print("\n--- 要約実験結果 (外挿性能) ---")
    print(summary_results_df.to_string(index=False))
    
    # --- 結果の可視化 ---
    # 外挿プロットは最初の実験の予測結果を使用
    create_extrapolation_plot(X, y, all_predictions, X_train, y_train, X_test, y_test, language='en')
    create_extrapolation_plot(X, y, all_predictions, X_train, y_train, X_test, y_test, language='ja')
    
    # 性能比較の可視化 (箱ひげ図)
    create_and_save_plot(all_results_df, language='en', experiment_name='ex3_metrics')
    create_and_save_plot(all_results_df, language='ja', experiment_name='ex3_metrics')
    
    # 7. 考察の保存機能は削除
    print("\n=== 実験完了 ===")

if __name__ == "__main__":
    main()
