import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
import os
import shap

TEXT_MAPS = {
    'en': {
        'title_map': {
            'R2 Score': 'Prediction Accuracy (R2 Score)',
            'Train Time (s)': 'Computation Cost (Training Time)',
            'SHAP Time (s)': 'Interpretability Cost (SHAP Calculation Time)',
            'RMSE': 'RMSE (Lower is Better)',
            'MAE': 'MAE (Lower is Better)',
            'Accuracy': 'Accuracy (Higher is Better)',
            'F1 Score': 'F1 Score (Higher is Better)',
            'ROC AUC': 'ROC AUC (Higher is Better)',
            'Brier Score': 'Brier Score (Lower is Better)',
            'Overfitting': 'Overfitting Factor (Lower is Better)',
            'RMSE (Train)': 'RMSE (Train)',
            'RMSE (Test)': 'RMSE (Test)',
            'R2 (Test)': 'R2 Score (Test)',
            'Model': 'Model',
            'Coefficient': 'Coefficient',
            'Feature': 'Feature'
        },
        'filename_base_map': {
            'ex1': 'ex1_model_comparison_plot',
            'ex2': 'ex2_low_interaction_plot',
            'ex3_extrapolation': 'ex3_extrapolation_performance',
            'ex3_metrics': 'ex3_performance_metrics',
            'ex4_train_test': 'ex4_train_test_comparison',
            'ex4_overfitting': 'ex4_overfitting_metrics',
            'ex5_performance': 'ex5_classification_performance',
            'ex5_coefficients': 'ex5_logistic_coefficients',
            'ex5_shap': 'ex5_shap_summary'
        },
        'main_title_map': {
            'ex1': 'Model Performance Comparison (Linearity-Dominant Data)',
            'ex2': 'Model Performance Comparison (Low-Interaction Data)',
            'ex3_extrapolation': 'Model Behavior Comparison on Extrapolation Task',
            'ex3_metrics': 'Extrapolation Performance Comparison',
            'ex4_train_test': 'Comparison of Training and Test Errors (Small Data)',
            'ex4_overfitting': 'Overfitting Metrics Comparison (Small Data)',
            'ex5_performance': 'Classification Performance Comparison',
            'ex5_coefficients': 'Logistic Regression Coefficients (Top/Bottom 10)',
            'ex5_shap': 'SHAP Summary Plot for LightGBM'
        }
    },
    'ja': {
        'title_map': {
            'R2 Score': '予測精度 (R2スコア)',
            'Train Time (s)': '計算コスト (学習時間)',
            'SHAP Time (s)': '解釈性コスト (SHAP計算時間)',
            'RMSE': 'RMSE (低いほど良い)',
            'MAE': 'MAE (低いほど良い)',
            'Accuracy': '正解率（高いほど良い）',
            'F1 Score': 'F1スコア（高いほど良い）',
            'ROC AUC': 'ROC AUC（高いほど良い）',
            'Brier Score': 'Brierスコア（低いほど良い）',
            'Overfitting': '過学習係数（低いほど良い）',
            'RMSE (Train)': 'RMSE (訓練)',
            'RMSE (Test)': 'RMSE (テスト)',
            'R2 (Test)': 'R2スコア (テスト)',
            'Model': 'モデル',
            'Coefficient': '係数',
            'Feature': '特徴量'
        },
        'filename_base_map': {
            'ex1': 'ex1_model_comparison_plot_ja',
            'ex2': 'ex2_low_interaction_plot_ja',
            'ex3_extrapolation': 'ex3_extrapolation_performance_ja',
            'ex3_metrics': 'ex3_performance_metrics_ja',
            'ex4_train_test': 'ex4_train_test_comparison_ja',
            'ex4_overfitting': 'ex4_overfitting_metrics_ja',
            'ex5_performance': 'ex5_classification_performance_ja',
            'ex5_coefficients': 'ex5_logistic_coefficients_ja',
            'ex5_shap': 'ex5_shap_summary_ja'
        },
        'main_title_map': {
            'ex1': 'モデル性能比較（線形性支配的なデータ）',
            'ex2': 'モデル性能比較（特徴量間の交互作用が少ないデータ）',
            'ex3_extrapolation': '外挿タスクでのモデル挙動比較',
            'ex3_metrics': '外挿性能比較',
            'ex4_train_test': '訓練誤差とテスト誤差の比較（小規模データ）',
            'ex4_overfitting': '過学習指標の比較（小規模データ）',
            'ex5_performance': '分類性能の比較',
            'ex5_coefficients': 'ロジスティック回帰の係数（上位/下位10）',
            'ex5_shap': 'LightGBMのSHAPサマリープロット'
        }
    }
}

def create_and_save_plot(results_df, language='en', experiment_name='ex1'):
    """
    言語設定に基づいてモデル性能比較のグラフを作成し、保存する。
    棒グラフから箱ひげ図に変更し、平均値と標準偏差を表示。
    
    Args:
        results_df (pd.DataFrame): モデルの性能評価結果を含むDataFrame。
                                   'Model', 'Metric', 'Value' の形式を想定。
        language (str): 'en' (英語) または 'ja' (日本語) を指定。
        experiment_name (str): 実験名 (例: 'ex1', 'ex2')。ファイル名に使用。
    """
    title_map = TEXT_MAPS[language]['title_map']
    filename_base_map = TEXT_MAPS[language]['filename_base_map']
    main_title_map = TEXT_MAPS[language]['main_title_map']

    plt.rcParams['font.size'] = 12

    # ヘルパー関数を定義
    def _plot_ex1_ex2(results_df, title_map, filename_base_map, main_title_map, experiment_name):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(main_title_map[experiment_name], fontsize=16)
        
        metrics = ['R2 Score', 'Train Time (s)', 'SHAP Time (s)']
        for i, metric in enumerate(metrics):
            sns.violinplot(x=metric, y='Model', data=results_df, ax=axes[i], palette='viridis', hue='Model', legend=False, inner=None, cut=0)  # cut=0
            sns.stripplot(x=metric, y='Model', hue='Model', data=results_df, ax=axes[i], palette='dark:black', size=3, jitter=True, alpha=0.5, legend=False)
            axes[i].set_title(title_map[metric])
            if metric == 'Train Time (s)':
                axes[i].set_xscale('log')
                axes[i].set_xlim(left=1e-3) # 下限を1e-3に設定
            elif metric == 'SHAP Time (s)':
                axes[i].set_xscale('log')
                axes[i].set_xlim(left=1e-4) # 下限を1e-4に設定
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filename_base = filename_base_map[experiment_name]
        plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{filename_base}.pdf", bbox_inches='tight')
        print(f"グラフを{filename_base}.png/.pdfに保存しました。")
        plt.close(fig)

    def _plot_ex3_metrics(results_df, title_map, filename_base_map, main_title_map, experiment_name):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(main_title_map[experiment_name], fontsize=16)
        
        metrics = ['RMSE', 'MAE', 'R2 Score']
        palettes = ['Reds', 'Oranges', 'Blues']
        for i, metric in enumerate(metrics):
            sns.violinplot(x=metric, y='Model', data=results_df, ax=axes[i], palette=palettes[i], hue='Model', legend=False, inner=None)
            sns.stripplot(x=metric, y='Model', hue='Model', data=results_df, ax=axes[i], palette='dark:black', size=3, jitter=True, alpha=0.5, legend=False)
            axes[i].set_title(title_map[metric])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filename_base = filename_base_map[experiment_name]
        plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{filename_base}.pdf", bbox_inches='tight')
        print(f"性能比較グラフを{filename_base}.png/.pdfに保存しました。")
        plt.close(fig)

    def _plot_ex4_train_test(results_df, title_map, filename_base_map, main_title_map, experiment_name):
        fig, ax = plt.subplots(figsize=(12, 6))

        # データを整形: 'RMSE (Train)'と'RMSE (Test)'を一つの列にまとめる
        df_melted = results_df.melt(id_vars=['Model'], value_vars=['RMSE (Train)', 'RMSE (Test)'],
                                    var_name='Type', value_name='RMSE')
        # 'Type'列の値をより分かりやすいラベルに変更
        df_melted['Type'] = df_melted['Type'].map({'RMSE (Train)': title_map['RMSE (Train)'], 'RMSE (Test)': title_map['RMSE (Test)']})

        # モデルの順序を固定
        model_order = results_df['Model'].unique()

        # 左右分割バイオリンプロット
        # パレットを言語に応じて動的に生成
        current_palette = {
            title_map['RMSE (Train)']: 'skyblue',
            title_map['RMSE (Test)']: 'orange'
        }
        sns.violinplot(x='Model', y='RMSE', hue='Type', data=df_melted, ax=ax,
                       palette=current_palette,
                       split=True, inner=None, order=model_order, cut=0) # cut=0
        
        # ストリッププロットを重ねる
        sns.stripplot(x='Model', y='RMSE', hue='Type', data=df_melted, ax=ax,
                      palette='dark:black', size=3, jitter=True, alpha=0.5, dodge=True, order=model_order, legend=False)

        ax.set_xlabel(title_map['Model'])
        ax.set_ylabel('RMSE')
        ax.set_title(main_title_map[experiment_name])
        
        # x軸の目盛りとラベルを明示的に設定
        ax.set_xticks(np.arange(len(model_order)))
        ax.set_xticklabels(model_order, rotation=45, ha='right')
        
        ax.legend(title='データセット') # 凡例のタイトルを調整
        ax.grid(True, alpha=0.3)

        # Y軸を対数スケールに設定
        ax.set_yscale('log')
        # Y軸の表示範囲を調整 (最小値を1e-14に設定)
        ax.set_ylim(1e-14, None) # 上限は自動調整に任せる

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filename_base = filename_base_map[experiment_name]
        plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{filename_base}.pdf", bbox_inches='tight')
        print(f"訓練・テスト誤差比較グラフを{filename_base}.png/.pdfに保存しました。")
        plt.close(fig)

    def _plot_ex4_overfitting(results_df, title_map, filename_base_map, main_title_map, experiment_name):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.violinplot(x='Overfitting', y='Model', data=results_df, ax=axes[0], palette='Reds', hue='Model', legend=False, inner=None)
        sns.stripplot(x='Overfitting', y='Model', hue='Model', data=results_df, ax=axes[0], palette='dark:black', size=3, jitter=True, alpha=0.5, legend=False)
        axes[0].set_title(title_map['Overfitting'])
        axes[0].set_xlabel(title_map['Overfitting'])
        
        sns.violinplot(x='R2 (Test)', y='Model', data=results_df, ax=axes[1], palette='Blues', hue='Model', legend=False, inner=None)
        sns.stripplot(x='R2 (Test)', y='Model', hue='Model', data=results_df, ax=axes[1], palette='dark:black', size=3, jitter=True, alpha=0.5, legend=False)
        axes[1].set_title(title_map['R2 (Test)'])
        axes[1].set_xlabel(title_map['R2 (Test)'])
        
        plt.tight_layout()
        filename_base = filename_base_map[experiment_name]
        plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{filename_base}.pdf", bbox_inches='tight')
        print(f"過学習指標グラフを{filename_base}.png/.pdfに保存しました。")
        plt.close(fig)

    def _plot_ex5_performance(results_df, title_map, filename_base_map, main_title_map, experiment_name):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(main_title_map[experiment_name], fontsize=16)
        
        metrics = ['Accuracy', 'F1 Score', 'ROC AUC']
        palettes = ['Blues', 'Greens', 'Purples']
        for i, metric in enumerate(metrics):
            sns.violinplot(x=metric, y='Model', data=results_df, ax=axes[i], palette=palettes[i], hue='Model', legend=False, inner=None)
            sns.stripplot(x=metric, y='Model', hue='Model', data=results_df, ax=axes[i], palette='dark:black', size=3, jitter=True, alpha=0.5, legend=False)
            axes[i].set_title(title_map[metric])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filename_base = filename_base_map[experiment_name]
        plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{filename_base}.pdf", bbox_inches='tight')
        print(f"性能比較グラフを{filename_base}.png/.pdfに保存しました。")
        plt.close(fig)

    def _plot_ex5_coefficients(results_df, title_map, filename_base_map, main_title_map, experiment_name):
        fig = plt.figure(figsize=(12, 10))
        top_bottom_features = pd.concat([results_df.head(10), results_df.tail(10)])
        sns.barplot(x='Coefficient', y='Feature', data=top_bottom_features, hue='Feature', palette='coolwarm', legend=False)
        plt.title(main_title_map[experiment_name], fontsize=16)
        plt.xlabel(title_map['Coefficient'], fontsize=12)
        plt.ylabel(title_map['Feature'], fontsize=12)
        plt.tight_layout()
        filename_base = filename_base_map[experiment_name]
        plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{filename_base}.pdf", bbox_inches='tight')
        print(f"ロジスティック回帰係数グラフを{filename_base}.png/.pdfに保存しました。")
        plt.close(fig)

    # experiment_name に基づいて適切なヘルパー関数を呼び出す
    if experiment_name in ['ex1', 'ex2']:
        _plot_ex1_ex2(results_df, title_map, filename_base_map, main_title_map, experiment_name)
    elif experiment_name == 'ex3_extrapolation':
        # この関数は生データと予測値を受け取るため、results_dfとは異なる形式
        # main関数内で直接プロットを生成し、この関数は呼び出さない
        pass
    elif experiment_name == 'ex3_metrics':
        _plot_ex3_metrics(results_df, title_map, filename_base_map, main_title_map, experiment_name)
    elif experiment_name == 'ex4_train_test':
        _plot_ex4_train_test(results_df, title_map, filename_base_map, main_title_map, experiment_name)
    elif experiment_name == 'ex4_overfitting':
        _plot_ex4_overfitting(results_df, title_map, filename_base_map, main_title_map, experiment_name)
    elif experiment_name == 'ex5_performance':
        _plot_ex5_performance(results_df, title_map, filename_base_map, main_title_map, experiment_name)
    elif experiment_name == 'ex5_coefficients':
        _plot_ex5_coefficients(results_df, title_map, filename_base_map, main_title_map, experiment_name)
    elif experiment_name == 'ex5_shap':
        # この関数はshap_valuesとX_sampleを受け取るため、results_dfとは異なる形式
        # main関数内で直接プロットを生成し、この関数は呼び出さない
        pass

def create_extrapolation_plot(X, y, all_predictions, X_train, y_train, X_test, y_test, language='en'):
    """
    外挿タスクのモデル挙動比較グラフを作成し、保存する。
    """
    if language == 'ja':
        title = '外挿タスクでのモデル挙動比較'
        train_label = '訓練データ'
        extrap_label = '外挿データ'
        lin_reg_pred_label = '線形回帰予測'
        lgbm_pred_label = 'LightGBM予測'
        max_train_val_label = '訓練データの最大値'
        filename_base = 'ex3_extrapolation_comparison_ja'
    else:
        title = 'Model Behavior Comparison on Extrapolation Task'
        train_label = 'Training Data'
        extrap_label = 'Extrapolation Data'
        lin_reg_pred_label = 'Linear Regression Prediction'
        lgbm_pred_label = 'LightGBM Prediction'
        max_train_val_label = 'Max value in Training Data'
        filename_base = 'ex3_extrapolation_comparison'

    plt.figure(figsize=(12, 8))
    
    plt.scatter(X_train, y_train, label=train_label, color='blue', alpha=0.6, s=50)
    plt.scatter(X_test, y_test, label=extrap_label, color='green', alpha=0.6, s=50)
    
    plt.plot(X, all_predictions['Linear Regression'], color='red', linestyle='-', linewidth=2, 
             label=lin_reg_pred_label)
    plt.plot(X, all_predictions['LightGBM'], color='purple', linestyle='--', linewidth=2, 
             label=lgbm_pred_label)
    
    max_train_y = y_train.max()
    plt.axhline(y=max_train_y, color='orange', linestyle=':', linewidth=2, 
                label=f'{max_train_val_label} ({max_train_y:.2f})')
    plt.axvline(x=-1, color='gray', linestyle=':', linewidth=1.5)
    plt.axvline(x=1, color='gray', linestyle=':', linewidth=1.5)
    
    plt.title(title, fontsize=18)
    plt.xlabel('Feature (X)', fontsize=14)
    plt.ylabel('Target (y)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename_base}.pdf", bbox_inches='tight')
    print(f"グラフを{filename_base}.png/.pdfに保存しました。")
    plt.close()

def create_shap_summary_plot(shap_values, X_sample, language='en', experiment_name='ex5'):
    """
    SHAPサマリープロットを作成し、保存する。
    """
    if language == 'ja':
        title = 'LightGBMのSHAPサマリープロット'
        filename_base = 'ex5_shap_summary_ja'
    else:
        title = 'SHAP Summary Plot for LightGBM'
        filename_base = 'ex5_shap_summary'

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="dot", show=False)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename_base}.pdf", bbox_inches='tight')
    print(f"SHAPサマリープロットを{filename_base}.png/.pdfに保存しました。")
    plt.close()
