# KABUPICK X 株価ターミナル

日本株のタイミング検索、4種類の簡易スクリーニング、ウォッチリスト、メモ、ニュース、シナリオ表示をまとめたStreamlitアプリです。

## ローカル起動

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

ブラウザーで `http://localhost:8501` を開きます。

## Renderへの公開

このプロジェクトにはRender用の `render.yaml` を用意しています。GitHubリポジトリにソースを登録してから、Renderで **New → Blueprint** を選び、そのリポジトリを接続してください。Blueprintが無料のWebサービスとして作成されます。公開URLはRenderが割り当てます。

Render無料サービスは15分間アクセスがないと停止し、再アクセス時に起動待ちが発生します。また、無料インスタンスのファイル保存は再起動・停止・再デプロイ後に残りません。この設定ではメモ、ウォッチリスト、通知履歴を利用者ごとのブラウザーセッション内に保持し、利用者間では共有しません。ブラウザーセッション終了後の永続保存は行いません。

## LINE通知

LINE Messaging APIの `LINE_CHANNEL_ACCESS_TOKEN` と `LINE_USER_ID` をRenderの環境変数に登録すると、買いシグナル通知を送れます。未設定時はアプリ内バッジのみ動作します。秘密情報をGitHubのソースに書かないでください。

## データと利用上の注意

株価・財務情報はYahoo!ファイナンス経由で取得し、5分間キャッシュします。ニュース見出しは公開RSSから取得します。データ取得元の遅延や欠損があり得ます。表示される判定・予測・バックテストは参考情報であり、投資助言や利益保証ではありません。

## EXE

EXEは未作成です。後日作成するための `launcher.py` と `build_exe.ps1` を同梱しています。
