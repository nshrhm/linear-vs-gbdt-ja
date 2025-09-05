# linear-vs-gbdt-ja

![Python 3.10–3.12](https://img.shields.io/badge/Python-3.10%E2%80%933.12-blue)
[![License](https://img.shields.io/github/license/nshrhm/linear-vs-gbdt-ja)](LICENSE)
![Last Commit](https://img.shields.io/github/last-commit/nshrhm/linear-vs-gbdt-ja)
![Issues](https://img.shields.io/github/issues/nshrhm/linear-vs-gbdt-ja)
![Stars](https://img.shields.io/github/stars/nshrhm/linear-vs-gbdt-ja?style=social)

[![CI](https://github.com/nshrhm/linear-vs-gbdt-ja/actions/workflows/ci.yml/badge.svg)](https://github.com/nshrhm/linear-vs-gbdt-ja/actions/workflows/ci.yml)
![Tag](https://img.shields.io/github/v/tag/nshrhm/linear-vs-gbdt-ja?sort=semver)
![Downloads](https://img.shields.io/github/downloads/nshrhm/linear-vs-gbdt-ja/total)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

依存関係: 
[![numpy](https://img.shields.io/badge/numpy-1.26.4-013243)](https://pypi.org/project/numpy/)
[![pandas](https://img.shields.io/badge/pandas-2.2.2-150458)](https://pypi.org/project/pandas/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.4.2-f89939)](https://pypi.org/project/scikit-learn/)
[![scipy](https://img.shields.io/badge/scipy-1.11.4-8CAAE6)](https://pypi.org/project/scipy/)
[![matplotlib](https://img.shields.io/badge/matplotlib-3.8.4-11557c)](https://pypi.org/project/matplotlib/)
[![seaborn](https://img.shields.io/badge/seaborn-0.13.2-4c72b0)](https://pypi.org/project/seaborn/)
[![optuna](https://img.shields.io/badge/optuna-3.6.1-3E79B5)](https://pypi.org/project/optuna/)
[![shap](https://img.shields.io/badge/SHAP-0.42.1-ff7f0e)](https://pypi.org/project/shap/)
[![xgboost](https://img.shields.io/badge/xgboost-3.0.0-EB5E28)](https://pypi.org/project/xgboost/)
[![lightgbm](https://img.shields.io/badge/lightgbm-4.3.0-017E4A)](https://pypi.org/project/lightgbm/)

本リポジトリは、論文「Linear Models vs. Gradient Boosting: A Systematic Comparison of Efficiency, Extrapolation, and Interpretability」（JMLR 投稿予定）の実験コード・生成物（図表・CSV）を、日本語利用者向けに公開するものです。コード内コメント・標準出力・図表タイトルなどは日本語対応です。英語版リポジトリは別途公開予定です。

## 概要
- 対象モデル: 線形モデル（Linear/Ridge/Lasso/Logistic）と GBDT（XGBoost/LightGBM）。
- 評価軸: 予測性能、計算コスト（学習/推論）、解釈性コスト（SHAP 計算時間）、外挿性能、過学習耐性。
- 再現性: 30 回反復＋Optuna によるハイパーパラメータチューニング（各 50 試行）で統計的に安定な比較を実施。ランダムシードは各実験で固定（例: 42 + 実験ID）。

## 実験一覧（スクリプト）
- ex1: 線形性支配データでの性能比較（回帰）
  - `ex1_model_comparison_linearity.py`
  - 合成回帰データ、RMSE/MAE/R2 に加え計算・解釈性コストを評価
- ex2: 交互作用が少ないデータ（Friedman1）での性能比較（回帰）
  - `ex2_low_interaction_comparison.py`
- ex3: 外挿タスク（訓練範囲外での予測挙動）
  - `ex3_extrapolation_comparison.py`
  - 1 次元の線形データに対し、内挿/外挿での挙動を可視化
- ex4: 小規模データにおける過学習耐性（回帰）
  - `ex4_small_sample_comparison.py`
  - 訓練/テストの RMSE・R2 と過学習係数を比較
- ex5: 実データ（UCI German Credit）での分類性能と解釈性
  - `ex5_interpretability_comparison.py`
  - 前処理（StandardScaler/OneHotEncoder）＋パイプライン、ROC AUC/Accuracy/F1/Brier を評価、係数可視化と LightGBM+SHAP による解釈

補助スクリプト
- 一括実行: `run_batch.sh`（5 実験を順番に実行し `execution_log.txt` に記録）
- 図の再生成: `test_plots.py`（既存 CSV から図を再出力）

## 生成物（主なファイル）
スクリプト実行により以下が生成されます（英語版/日本語版の両方を保存）。
- 図（PNG/PDF）: `ex1_model_comparison_plot[ _ja].{png,pdf}`、`ex2_low_interaction_plot[ _ja].{png,pdf}`、`ex3_extrapolation_comparison[ _ja].{png,pdf}`、`ex3_performance_metrics[ _ja].{png,pdf}`、`ex4_train_test_comparison[ _ja].{png,pdf}`、`ex4_overfitting_metrics[ _ja].{png,pdf}`、`ex5_classification_performance[ _ja].{png,pdf}`、`ex5_logistic_coefficients[ _ja].{png,pdf}`、`ex5_shap_summary[ _ja].{png,pdf}`
- 結果 CSV: `ex*_*_all_results.csv`、`ex*_*_summary_results.csv`、`ex5_logistic_coefficients.csv`

本リポジトリには代表的な出力例（PNG/PDF/CSV）も同梱しています。

## 必要環境
- Python: 3.10 以上推奨（3.11/3.12 でも可）
- CPU で実行可能（GPU 不要）
- 必要パッケージ
  - numpy, pandas, scikit-learn, matplotlib, seaborn, japanize-matplotlib
  - scipy, optuna, shap
  - xgboost, lightgbm（XGBoost 3.x での動作を想定）

インストール例（仮想環境の利用を推奨）
```
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install numpy pandas scikit-learn matplotlib seaborn japanize-matplotlib \
            scipy optuna shap xgboost lightgbm
```

備考
- `lightgbm` は公式ホイールで多くの環境に対応しています。ビルドに失敗する場合は公式ドキュメントをご参照ください。
- ネットワークが使えない環境でも実行できますが、ex5 は OpenML からのデータ取得を試みます。失敗時は自動的に人工データへフォールバックします。

## 実行方法
- 単体実行（例: ex1）
```
python3 ex1_model_comparison_linearity.py
```
- まとめて実行
```
bash run_batch.sh
```
- 図の再生成（既存 CSV から）
```
python3 test_plots.py
```

出力
- 実行ディレクトリ直下に PNG/PDF/CSV を自動保存します。
- 実行ログは `execution_log.txt` に追記されます。

## 実行時間の目安とクイック実行
本設定（各 30 回＋Optuna 50 試行）は比較的重く、マシン環境によっては時間を要します。動作確認やクイック再現の際は以下を小さくしてください。
- 反復回数（`n_experiments`）を 30 → 5 程度
  - `ex1_model_comparison_linearity.py:47`
  - `ex2_low_interaction_comparison.py:46`
  - `ex3_extrapolation_comparison.py:45`
  - `ex4_small_sample_comparison.py:43`
  - `ex5_interpretability_comparison.py:50`
- Optuna の試行回数（`n_trials`）を 50 → 5–10 程度
  - XGBoost: `ex*_*.py` 内の `study_xgb.optimize(..., n_trials=50, ...)`
  - LightGBM: `ex*_*.py` 内の `study_lgbm.optimize(..., n_trials=50, ...)`
  - 例: `ex1_model_comparison_linearity.py:120` と `:126`

## データ取り扱い
- ex1/2/3/4: すべて合成データ（外部配布データ不要）
- ex5: OpenML の German Credit（data_id=31）を取得します。取得不可の場合は自動で人工データに切替えます。

## 再現性メモ
- 乱数シードは明示的に固定（例: 42 + 実験ID、Optuna は `TPESampler(seed=42)`）。
- ライブラリのメジャーバージョン差分で統計値がわずかに変動する場合があります。論文掲載値と大きく乖離する場合はご連絡ください。

## 論文・引用
- 論文題目: Linear Models vs. Gradient Boosting: A Systematic Comparison of Efficiency, Extrapolation, and Interpretability
- 著者: Naruki Shirahama
- 投稿先: Journal of Machine Learning Research（投稿予定）
- 引用例（暫定）
  - Shirahama, N. (2025). Linear Models vs. Gradient Boosting: A Systematic Comparison of Efficiency, Extrapolation, and Interpretability. Manuscript submitted for publication. Code: https://github.com/nshrhm/linear-vs-gbdt-ja

## ライセンス
- 本リポジトリのコンテンツは CC BY 4.0（Creative Commons Attribution 4.0 International）で提供します。
  - あなたは共有・改変・商用利用が可能です。出典表示（著者名・作品名・ライセンス名・変更の有無）を行ってください。
  - ライセンス全文: https://creativecommons.org/licenses/by/4.0/
  - ライセンスファイル: `LICENSE`

## 連絡先
- Email: nshirahama@ieee.org
- 所属: Shimonoseki City University, Faculty of Data Science

## よくある質問（FAQ）
- Q. GPU は必要ですか？
  - A. 不要です。CPU で十分に再現可能です。
- Q. 図だけ再生成したいです。
  - A. 既存の `ex*_all_results.csv` があれば `python3 test_plots.py` で図のみ再出力できます。
- Q. 実行が長いです。
  - A. 上記「クイック実行」を参考に `n_experiments` と `n_trials` を小さくしてください。

---
本リポジトリは日本語利用者向けの補助公開です。英語版（英語 UI/コメント）の公開準備が整い次第、README にリンクを追記します。
