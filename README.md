# KABUPICK X 株価ターミナル

Render向けのHTTP版はFastAPI + Jinja2で、通常のHTTPSページ遷移だけで動作します。画面操作用のWebSocketやJavaScriptは使用せず、株価チャートはSVGとしてHTML内に描画します。既存Streamlit版はローカル確認用に `app.py` として残しています。

## ローカル起動

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

ブラウザーで `http://localhost:8501` を開きます。

### 任意のデータ/API設定

Streamlit版は、設定しなくても株価とテクニカル指標で動作します。J-Quantsを使う場合は、端末の環境変数にAPIトークンを設定します。

```powershell
$env:JQUANTS_API_TOKEN = "取得したトークン"
$env:DISCORD_WEBHOOK_URL = "DiscordのWebhook URL"
python -m streamlit run app.py
```

J-Quantsの財務値を優先し、取得できない項目はyfinanceの値、または「—」へフォールバックします。APIキーやWebhook URLはソースコードへ書き込まないでください。

Streamlit版では、ROE・PBR・PER・売上成長率などの条件を画面から変更できます。候補コードは10銘柄以上入力すると比較しやすくなります。

## Renderへの公開

このプロジェクトにはRender用の `render.yaml` を用意しています。GitHubリポジトリにソースを登録してから、Renderで **New → Blueprint** を選び、そのリポジトリを接続してください。Blueprintが無料のWebサービスとして作成されます。公開設定はiPadのSafariでも利用しやすい、WebSocketを使わないHTTP版 `http_app.py` を起動します。Streamlit版 `app.py` はローカル利用・開発用として残しています。

Render無料サービスは15分間アクセスがないと停止し、再アクセス時に起動待ちが発生します。HTTP版のウォッチリストとメモは利用者ごとのサーバー内セッションに保存し、利用者間では共有しません。無料サービス停止や再デプロイでメモリが消えると、これらのデータも失われます。永続保存は行いません。

## 通知

Streamlit版は、監視銘柄の再確認時にDiscord Webhookへ通知できます。LINE通知は既存のローカル用機能として残しています。公開版での常時バックグラウンド監視には、別途Render CronやWorkerなどの常駐実行基盤が必要です。

## データと利用上の注意

株価・財務情報はYahoo!ファイナンス経由で取得し、5分間キャッシュします。ニュース見出しは公開RSSから取得します。データ取得元の遅延や欠損があり得ます。表示される判定は参考情報であり、投資助言や利益保証ではありません。

## EXE

EXEは未作成です。後日作成するための `launcher.py` と `build_exe.ps1` を同梱しています。
