# Contributing Guide

ありがとうございます。以下の最小ルールにご協力ください。

## ブランチ戦略
- `main`: 安定ブランチ（保護）。PR 経由で更新
- `feature/<topic>`: 機能追加やドキュメント更新

## コミットメッセージ
- Conventional Commits を採用
  - 例: `feat: 〜`, `fix: 〜`, `docs: 〜`, `chore: 〜`
- スコープの併用例: `docs(README): 〜`

## PR 手順
1. `feature/*` ブランチを作成
2. 小さく目的を絞って変更
3. CI（構文チェック）が通ることを確認
4. 変更点の要約と背景を PR に記載

## コーディング
- Python 3.10–3.12 を対象
- フォーマット: Black（行長 88）
- 可能な範囲で型ヒント/Docstring を付与

## 公開/非公開ポリシー
- 公開（リポジトリに含める）: `README`, `CONTRIBUTING`, `CHANGELOG`, `.editorconfig`
- 非公開（Git 追跡から除外）: `.vscode/`, `docs/`, `journal/`, `TODO.md`, `PROBLEMS.md`

ご不明点は Issue でお知らせください。

