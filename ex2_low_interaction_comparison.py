#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
機械学習モデルの比較実験 - 特徴量間の交互作用が少ないデータでの性能比較
Friedman1データセットを使用した低交互作用データでの実験
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

# scikit-learnからモデルとデータ生成ツールをインポート
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor # Random Forestを追加

# GDBTモデルをインポート
import xgboost as xgb
import lightgbm as lgb
from lightgbm.callback import early_stopping

# plotting.pyからグラフ作成関数をインポート
from plotting import create_and_save_plot, create_extrapolation_plot, create_shap_summary_plot

def main():
    print("=== 機械学習モデル比較実験 (特徴量間の交互作用が少ないデータ) ===")
    print("30回実行の平均値と標準偏差による統計的に信頼性の高い評価を実施します。")
    
    # 実験回数
    n_experiments = 30
    
    # 結果を格納するリスト
    all_results = []

    # 正則化パラメータalphaの探索範囲を定義
    alphas = np.logspace(-4, 4, 100)  # 10^-4 から 10^4 までを100分割

    # Optunaによるハイパーパラメータチューニング
    tuned_params_xgb = {}
    tuned_params_lgbm = {}

    print("\n=== GBDTモデルのハイパーパラメータチューニング (Optuna) ===")
    # チューニング用のデータ生成 (最初の実験のシードを使用)
    X_tune, y_tune = make_friedman1(
        n_samples=10000,
        n_features=10,
        noise=5.0,
        random_state=42
    )
    X_train_tune, X_test_tune, y_train_tune, y_test_tune = train_test_split(
        X_tune, y_tune, test_size=0.2, random_state=42
    )
    feature_names_tune = [f'feature_{i}' for i in range(X_tune.shape[1])]
    X_train_df_tune = pd.DataFrame(X_train_tune, columns=feature_names_tune)
    X_test_df_tune = pd.DataFrame(X_test_tune, columns=feature_names_tune)
    X_train_new_tune, X_val_tune, y_train_new_tune, y_val_tune = train_test_split(
        X_train_df_tune, y_train_tune, test_size=0.2, random_state=42
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
            'seed': 42,
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
            'random_state': 42,
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
    study_xgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=True)
    tuned_params_xgb = study_xgb.best_params
    print(f"XGBoostの最適ハイパーパラメータ: {tuned_params_xgb}")

    # LightGBMチューニング
    print("LightGBMのハイパーパラメータチューニング中...")
    study_lgbm = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study_lgbm.optimize(objective_lgbm, n_trials=50, show_progress_bar=True)
    tuned_params_lgbm = study_lgbm.best_params
    print(f"LightGBMの最適ハイパーパラメータ: {tuned_params_lgbm}")
    
    # ===================================================================
    # 複数回実験のループ
    # ===================================================================
    for experiment_id in range(n_experiments):
        print(f"\n{'='*20} 実験 {experiment_id + 1}/{n_experiments} {'='*20}")
        
        # ===================================================================
        # 1. 人工データの生成 (Low-Interaction Data)
        # ===================================================================
        print(f"\n1. 人工データの生成中... (実験 {experiment_id + 1})")
        print("make_friedman1データセットを使用:")
        print("y = 10*sin(pi*X1*X2) + 20*(X3-0.5)^2 + 10*X4 + 5*X5 + noise")
        print("交互作用項(X1*X2)が1つのみで、他は線形項や単純な非線形項")
        
        X, y = make_friedman1(
            n_samples=10000,
            n_features=10,  # friedman1は最初の5つの特徴量のみ使用
            noise=5.0,
            random_state=42 + experiment_id  # 各実験で異なるシード
        )
        
        # データを訓練用とテスト用に分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42 + experiment_id
        )
        
        # LightGBMの警告を回避するため、DataFrameに変換
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # ADDED: GBDTモデルの早期停止のために訓練データをさらに分割
        X_train_new, X_val, y_train_new, y_val = train_test_split(
            X_train_df, y_train, test_size=0.2, random_state=42 + experiment_id
        )
        
        print(f"データ生成完了。")
        print(f"訓練データ: {X_train_df.shape}, テストデータ: {X_test_df.shape}")
        print(f"GBDT訓練データ: {X_train_new.shape}, GBDT検証データ: {X_val.shape}")

        # ===================================================================
        # 2. モデルの定義
        # ===================================================================
        print("\n2. モデルの定義...")
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": RidgeCV(alphas=alphas, cv=5),
            "Lasso": LassoCV(alphas=alphas, cv=5, random_state=42 + experiment_id, max_iter=10000),
            "Random Forest": RandomForestRegressor(random_state=42 + experiment_id, n_jobs=-1), # Random Forestを追加
            "XGBoost": xgb.XGBRegressor(random_state=42 + experiment_id, callbacks=[xgb.callback.EarlyStopping(rounds=10)], **tuned_params_xgb), # チューニング済みパラメータを適用
            "LightGBM": lgb.LGBMRegressor(random_state=42 + experiment_id, verbosity=-1, **tuned_params_lgbm) # チューニング済みパラメータを適用
        }
        
        # ===================================================================
        # 3. 実験の実行と評価
        # ===================================================================
        print("\n3. 実験の実行と評価...")
        experiment_results = []
        
        # 各モデルでループ処理
        for name, model in models.items():
            print(f"--- {name} の評価を開始 (実験 {experiment_id + 1}) ---")
            
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
            end_train_time = time.time()
            train_time = end_train_time - start_train_time
            
            # --- 推論時間の計測 ---
            start_pred_time = time.time()
            if name in ["Linear Regression", "Ridge", "Lasso", "Random Forest"]:
                y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test_df)
            end_pred_time = time.time()
            pred_time = end_pred_time - start_pred_time
            
            # --- 予測精度の評価 ---
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # --- 解釈性コストの計測 (SHAP値の計算時間) ---
            start_shap_time = time.time()
            if hasattr(model, 'coef_'):
                explainer = shap.LinearExplainer(model, X_train)
                num_shap_samples = min(1000, X_test.shape[0])
                shap_sample_indices = np.random.choice(X_test.shape[0], num_shap_samples, replace=False)
                X_test_subset = X_test[shap_sample_indices]
                shap_values = explainer.shap_values(X_test_subset)
            elif name == "Random Forest":
                explainer = shap.TreeExplainer(model, X_train)
                num_shap_samples = min(1000, X_test.shape[0])
                shap_sample_indices = np.random.choice(X_test.shape[0], num_shap_samples, replace=False)
                X_test_subset = X_test[shap_sample_indices]
                shap_values = explainer.shap_values(X_test_subset)
            else:
                explainer = shap.TreeExplainer(model, X_train_new)
                num_shap_samples = min(1000, X_test_df.shape[0])
                shap_sample_indices = np.random.choice(X_test_df.shape[0], num_shap_samples, replace=False)
                X_test_subset = X_test_df.iloc[shap_sample_indices]
                shap_values = explainer.shap_values(X_test_subset)
            end_shap_time = time.time()
            shap_time = end_shap_time - start_shap_time
            
            # 結果をリストに追加
            experiment_results.append({
                "Experiment": experiment_id + 1,
                "Model": name,
                "Train Time (s)": train_time,
                "Inference Time (s)": pred_time,
                "RMSE": rmse,
                "MAE": mae,
                "R2 Score": r2,
                "SHAP Time (s)": shap_time,
            })
            print(f"--- {name} の評価が完了 (実験 {experiment_id + 1}) ---")
        
        # 各実験の結果を全体リストに追加
        all_results.extend(experiment_results)
    
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
            "SHAP Time (s) Mean": model_data["SHAP Time (s)"].mean(),
            "SHAP Time (s) Std": model_data["SHAP Time (s)"].std(),
        }
        summary_results.append(mean_result)
        
        print(f"{model_name}: RMSE平均={mean_result['RMSE Mean']:.4f}±{mean_result['RMSE Std']:.4f}, R2平均={mean_result['R2 Score Mean']:.4f}±{mean_result['R2 Score Std']:.4f}")

    # ===================================================================
    # 5. 統計検定 (Wilcoxon符号順位検定)
    # ===================================================================
    print("\n5. 統計検定 (Wilcoxon符号順位検定)...")
    # 例: LightGBM vs. Random Forest のR2スコアを比較
    r2_lgbm = all_results_df[all_results_df['Model'] == 'LightGBM']['R2 Score']
    r2_rf = all_results_df[all_results_df['Model'] == 'Random Forest']['R2 Score']
    
    if len(r2_lgbm) == len(r2_rf) and len(r2_lgbm) > 1:
        stat, p_value = stats.wilcoxon(r2_lgbm, r2_rf, alternative='greater') # LightGBM > Random Forest を仮説
        print(f"LightGBM vs. Random Forest (R2 Score): Wilcoxon p-value = {p_value:.4f}")
        if p_value < 0.05:
            print("  -> LightGBMのR2スコアはRandom Forestより統計的に有意に高いです (p < 0.05)。")
        else:
            print("  -> LightGBMのR2スコアはRandom Forestより統計的に有意に高いとは言えません (p >= 0.05)。")
    else:
        print("Wilcoxon検定に必要なデータ数が不足しているか、データフレームの構造が異なります。")

    # ===================================================================
    # 6. 結果の保存と可視化
    # ===================================================================
    print("\n6. 結果の保存と可視化...")
    
    # 平均値と標準偏差の結果をDataFrameに変換
    summary_results_df = pd.DataFrame(summary_results).sort_values(by="RMSE Mean").reset_index(drop=True)
    
    # CSVファイルとして保存（要約結果）
    summary_results_df.to_csv("ex2_low_interaction_summary_results.csv", index=False, encoding='utf-8')
    print("要約結果をex2_low_interaction_summary_results.csvに保存しました。")
    
    # 全実験結果も保存
    all_results_df.to_csv("ex2_low_interaction_all_results.csv", index=False, encoding='utf-8')
    print("全実験結果をex2_low_interaction_all_results.csvに保存しました。")
    
    # 結果の表示
    print("\n--- 要約実験結果 ---")
    print(summary_results_df.to_string(index=False))
    
    # --- 結果の可視化 (箱ひげ図) ---
    # 英語バージョン
    create_and_save_plot(all_results_df, language='en', experiment_name='ex2')
    
    # 日本語バージョン
    create_and_save_plot(all_results_df, language='ja', experiment_name='ex2')
    
    # plt.show() # GUIが必要なため、自動実行ではコメントアウト
    
    # 7. 考察の保存機能は削除
    print("\n=== 実験完了 ===")

if __name__ == "__main__":
    main()
