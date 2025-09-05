import pandas as pd
import os
from plotting import create_and_save_plot

def main():
    print("=== プロットテストスクリプトを開始します ===")

    # ex1のプロットをテスト
    print("\n--- ex1_performance プロットを生成中 ---")
    ex1_results_path = "ex1_model_performance_all_results.csv"
    if os.path.exists(ex1_results_path):
        ex1_df = pd.read_csv(ex1_results_path)
        create_and_save_plot(ex1_df, language='en', experiment_name='ex1')
        create_and_save_plot(ex1_df, language='ja', experiment_name='ex1')
        print(f"ex1_performance プロットの生成が完了しました。ファイル: ex1_model_comparison_plot.png/.pdf, ex1_model_comparison_plot_ja.png/.pdf")
    else:
        print(f"エラー: {ex1_results_path} が見つかりません。ex1のプロットをスキップします。")

    # ex2のプロットをテスト
    print("\n--- ex2_performance プロットを生成中 ---")
    ex2_results_path = "ex2_low_interaction_all_results.csv"
    if os.path.exists(ex2_results_path):
        ex2_df = pd.read_csv(ex2_results_path)
        create_and_save_plot(ex2_df, language='en', experiment_name='ex2')
        create_and_save_plot(ex2_df, language='ja', experiment_name='ex2')
        print(f"ex2_performance プロットの生成が完了しました。ファイル: ex2_low_interaction_plot.png/.pdf, ex2_low_interaction_plot_ja.png/.pdf")
    else:
        print(f"エラー: {ex2_results_path} が見つかりません。ex2のプロットをスキップします。")

    # ex3_metricsのプロットをテスト
    print("\n--- ex3_metrics プロットを生成中 ---")
    ex3_results_path = "ex3_extrapolation_all_results.csv" # ex3_metricsはex3_extrapolation_all_results.csvを使用
    if os.path.exists(ex3_results_path):
        ex3_df = pd.read_csv(ex3_results_path)
        create_and_save_plot(ex3_df, language='en', experiment_name='ex3_metrics')
        create_and_save_plot(ex3_df, language='ja', experiment_name='ex3_metrics')
        print(f"ex3_metrics プロットの生成が完了しました。ファイル: ex3_performance_metrics.png/.pdf, ex3_performance_metrics_ja.png/.pdf")
    else:
        print(f"エラー: {ex3_results_path} が見つかりません。ex3_metricsのプロットをスキップします。")

    # ex4_train_testとex4_overfittingのプロットをテスト
    print("\n--- ex4_train_test および ex4_overfitting プロットを生成中 ---")
    ex4_results_path = "ex4_small_sample_all_results.csv"
    if os.path.exists(ex4_results_path):
        ex4_df = pd.read_csv(ex4_results_path)
        create_and_save_plot(ex4_df, language='en', experiment_name='ex4_train_test')
        create_and_save_plot(ex4_df, language='ja', experiment_name='ex4_train_test')
        create_and_save_plot(ex4_df, language='en', experiment_name='ex4_overfitting')
        create_and_save_plot(ex4_df, language='ja', experiment_name='ex4_overfitting')
        print(f"ex4_train_test および ex4_overfitting プロットの生成が完了しました。ファイル: ex4_train_test_comparison.png/.pdf, ex4_train_test_comparison_ja.png/.pdf, ex4_overfitting_metrics.png/.pdf, ex4_overfitting_metrics_ja.png/.pdf")
    else:
        print(f"エラー: {ex4_results_path} が見つかりません。ex4のプロットをスキップします。")

    # ex5のプロットをテスト (既存のコード)
    print("\n--- ex5_performance プロットを生成中 ---")
    ex5_results_path = "ex5_interpretability_all_results.csv"
    if os.path.exists(ex5_results_path):
        ex5_df = pd.read_csv(ex5_results_path)
        create_and_save_plot(ex5_df, language='en', experiment_name='ex5_performance')
        create_and_save_plot(ex5_df, language='ja', experiment_name='ex5_performance')
        print(f"ex5_performance プロットの生成が完了しました。ファイル: ex5_classification_performance.png/.pdf, ex5_classification_performance_ja.png/.pdf")
    else:
        print(f"エラー: {ex5_results_path} が見つかりません。ex5のプロットをスキップします。")

    print("\n=== プロットテストスクリプトを終了します ===")

if __name__ == "__main__":
    main()
