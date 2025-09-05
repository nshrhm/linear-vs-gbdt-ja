import pandas as pd
import os

def main():
    print("=== RMSE (Train) 分布検証スクリプトを開始します ===")

    results_path = "ex4_small_sample_all_results.csv"
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)

        print("\n--- モデルごとのRMSE (Train) の要約統計量 ---")
        # モデルごとにRMSE (Train) の統計量を計算
        stats = df.groupby('Model')['RMSE (Train)'].describe()
        print(stats)

        print("\n--- 各モデルのRMSE (Train) の詳細な値 ---")
        for model_name in df['Model'].unique():
            print(f"\nモデル: {model_name}")
            model_rmse_train = df[df['Model'] == model_name]['RMSE (Train)']
            # 小さい値も確認できるようにソートして表示
            print(model_rmse_train.sort_values().to_string(index=False))

    else:
        print(f"エラー: {results_path} が見つかりません。スクリプトを終了します。")

    print("\n=== RMSE (Train) 分布検証スクリプトを終了します ===")

if __name__ == "__main__":
    main()
