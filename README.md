# KABUPICK X 株価ターミナル

Render向けのHTTP版はFastAPI + Jinja2で、通常のHTTPSページ遷移だけで動作します。画面操作用のWebSocketやJavaScriptは使用せず、株価チャートはSVGとしてHTML内に描画します。既存Streamlit版はローカル確認用に `app.py` として残しています。

## ローカル起動

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

ブラウザーで `http://localhost:8501` を開きます。

## Renderへの公開

このプロジェクトにはRender用の `render.yaml` を用意しています。GitHubリポジトリにソースを登録してから、Renderで **New → Blueprint** を選び、そのリポジトリを接続してください。Blueprintが無料のWebサービスとして作成されます。公開URLはRenderが割り当てます。HTTP版を公開するときは、Blueprintの起動コマンドが `uvicorn http_app:app` であることを確認してください。

Render無料サービスは15分間アクセスがないと停止し、再アクセス時に起動待ちが発生します。HTTP版のウォッチリストとメモは利用者ごとのサーバー内セッションに保存し、利用者間では共有しません。無料サービス停止や再デプロイでメモリが消えると、これらのデータも失われます。永続保存は行いません。

## LINE通知

HTTP版ではLINE通知と常時バックグラウンド監視はまだ提供していません。秘密情報をGitHubのソースに書かないでください。

## データと利用上の注意

株価・財務情報はYahoo!ファイナンス経由で取得し、5分間キャッシュします。ニュース見出しは公開RSSから取得します。データ取得元の遅延や欠損があり得ます。表示される判定は参考情報であり、投資助言や利益保証ではありません。

## EXE

EXEは未作成です。後日作成するための `launcher.py` と `build_exe.ps1` を同梱しています。
