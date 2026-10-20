"""後日EXE化するときの起動入口。今は直接実行しません。"""
import os
import sys

from streamlit.web import cli as stcli


if __name__ == "__main__":
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    sys.argv = ["streamlit", "run", app_path, "--server.headless", "true"]
    raise SystemExit(stcli.main())
