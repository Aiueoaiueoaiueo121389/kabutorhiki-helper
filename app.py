from __future__ import annotations

from datetime import datetime
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf


st.set_page_config(page_title="株の買い時・売り時チェッカー", page_icon="📈", layout="wide")
refresh_count = st_autorefresh(interval=300_000, limit=None, key="kabupick_refresh") or 0
st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: #080d16; color: #e6edf7; }
    html, body, [class*="css"] { font-family: "Yu Gothic UI", "Meiryo", sans-serif; }
    h1, h2, h3 { letter-spacing: .02em; color: #f4f7fc; }
    [data-testid="stHeader"] { background: rgba(8,13,22,.96); }
    [data-testid="stSidebar"] { background: #101a2b; border-right: 1px solid #263b5c; transition: all .28s ease; }
    [data-testid="stSidebar"] button, [data-testid="stSidebar"] input { min-height: 2.8rem; }
    .block-container { padding-top: 1.3rem; max-width: 1500px; }
    .stButton > button[kind="primary"] { background: linear-gradient(110deg,#2563eb,#06b6d4); border:0; color:white; font-weight:750; min-height:3rem; }
    .stButton > button { border-radius:.7rem; transition: transform .16s ease, filter .16s ease; }
    .stButton > button:hover { transform: translateY(-1px); filter:brightness(1.14); }
    [data-testid="stTextInput"] input { font-size:1.25rem !important; font-weight:750 !important; letter-spacing:.12em; background:#0d1727 !important; border-color:#3b82f6 !important; }
    [data-testid="stRadio"] label { font-size:1rem; }
    .signal { padding: 1.2rem; border-radius: .7rem; text-align: center; color: white; margin: .8rem 0 1rem; border: 1px solid #40516d; box-shadow:0 8px 28px rgba(0,0,0,.2); }
    .signal.ss { background: linear-gradient(120deg,#be123c,#7f1d1d); }
    .signal.buy { background: linear-gradient(120deg,#ea580c,#9a3412); }
    .signal.wait { background: linear-gradient(120deg,#ca8a04,#713f12); }
    .signal.sell { background: linear-gradient(120deg,#047857,#064e3b); }
    .signal-score { font-size: 2.1rem; font-weight: 800; }
    .signal-message { font-size: 1.35rem; font-weight: 700; }
    .terminal-card { background:#101a2b; border:1px solid #263b5c; border-radius:.7rem; padding: .85rem 1rem; }
    .terminal-title { color:#9fb2cf; font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; }
    .ticker-row { display:flex; justify-content:space-between; padding:.55rem 0; border-bottom:1px solid #263b5c; font-size:.9rem; }
    .up { color:#22d3ee; } .down { color:#fb7185; }
    .menu-badge { display:inline-block; padding:.12rem .48rem; margin-left:.35rem; border-radius:999px; background:#e11d48; color:white; font-weight:800; font-size:.75rem; }
    @media (prefers-reduced-motion: reduce) { *, *::before, *::after { transition:none !important; animation:none !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


@st.cache_data(ttl=300, show_spinner=False)
def load_data(code: str) -> pd.DataFrame:
    ticker = yf.Ticker(f"{code}.T")
    data = ticker.history(period="1y", interval="1d", auto_adjust=False)
    if data.empty:
        raise ValueError("株価データが見つかりませんでした。コードを確認してください。")
    data = data.rename(columns=str.title)
    close = data["Close"]
    data["SMA25"] = close.rolling(25).mean()
    data["SMA75"] = close.rolling(75).mean()
    data["RSI14"] = rsi(close)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"] = data["MACD"] - data["Signal"]
    return data.dropna(subset=["SMA25", "SMA75", "RSI14", "MACD", "Signal"])


@st.cache_data(ttl=300, show_spinner=False)
def load_fundamentals(code: str) -> dict:
    info = yf.Ticker(f"{code}.T").info
    fields = ["longName", "currentPrice", "marketCap", "trailingPE", "priceToBook", "returnOnEquity", "profitMargins", "debtToEquity", "dividendYield", "revenueGrowth", "earningsGrowth", "operatingMargins", "freeCashflow"]
    return {field: info.get(field) for field in fields}


DATA_DIR = Path(__file__).resolve().parent / "user_data"
WATCHLIST_FILE = DATA_DIR / "watchlist.json"
MEMOS_FILE = DATA_DIR / "memos.json"
SIGNAL_STATE_FILE = DATA_DIR / "signal_state.json"
HOSTED_MODE = os.getenv("KABUPICK_HOSTED", "").lower() in {"1", "true", "yes"}


def read_json(path: Path, default):
    if HOSTED_MODE:
        return st.session_state.get(f"kabupick:{path.name}", default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def write_json(path: Path, value) -> None:
    if HOSTED_MODE:
        st.session_state[f"kabupick:{path.name}"] = value
        return
    DATA_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def send_line_alert(message: str) -> tuple[bool, str]:
    """LINE Messaging API push message. Requires environment variables; never stores tokens in app files."""
    if HOSTED_MODE:
        return False, "公開版では、利用者認証を追加するまでLINE送信を停止しています。"
    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
    recipient = os.getenv("LINE_USER_ID", "").strip()
    if not token or not recipient:
        return False, "LINE_CHANNEL_ACCESS_TOKEN と LINE_USER_ID が未設定です。"
    response = requests.post(
        "https://api.line.me/v2/bot/message/push",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"to": recipient, "messages": [{"type": "text", "text": message[:4500]}]},
        timeout=15,
    )
    if response.ok:
        return True, "LINE通知を送信しました。"
    return False, f"LINE送信に失敗しました（HTTP {response.status_code}）。"


def signal_label(score: int) -> str:
    if score >= 80:
        return "SS級"
    if score >= 65:
        return "買い候補"
    if score <= 35:
        return "売り候補"
    return "様子見"


def refresh_signal_notifications(items: list[str]) -> list[dict]:
    state = read_json(SIGNAL_STATE_FILE, {"signals": {}, "seen": {}})
    notices = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for item in items:
        try:
            frame = load_data(item)
            score, _, _, _ = diagnose(frame)
            label = signal_label(score)
            previous = state.get("signals", {}).get(item)
            if label in ("SS級", "買い候補") and label != previous:
                notices.append({"code": item, "label": label, "score": score, "time": now})
            state.setdefault("signals", {})[item] = label
        except Exception:
            continue
    if notices:
        state.setdefault("unread", []).extend(notices)
        state["unread"] = state["unread"][-50:]
        notification_text = "KABUPICK X シグナル通知\n" + "\n".join(
            f"{entry['code']}: {entry['label']}（{entry['score']}点）" for entry in notices
        )
        try:
            send_line_alert(notification_text)
        except requests.RequestException:
            pass
    write_json(SIGNAL_STATE_FILE, state)
    return state.get("unread", [])


def fmt_value(value, suffix="") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:,.2f}{suffix}"
    return str(value)


def site_links(code: str):
    q = quote(f"{code} 株価")
    return [
        ("Yahoo!ファイナンス", f"https://finance.yahoo.co.jp/quote/{code}.T"),
        ("株探（Kabutan）", f"https://kabutan.jp/stock/?code={code}"),
        ("TDnet 適時開示", f"https://www.release.tdnet.info/inbs/I_main_00.html"),
        ("EDINET", "https://disclosure2dl.edinet-fsa.go.jp/"),
        ("日本経済新聞検索", f"https://www.nikkei.com/search/?query={q}"),
        ("Googleニュース検索", f"https://news.google.com/search?q={q}&hl=ja&gl=JP&ceid=JP%3Aja"),
    ]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(code: str) -> list[dict]:
    """公開RSSから見出しだけを取得。記事本文は各配信元で確認する。"""
    query = quote(f"{code} 株式")
    url = f"https://news.google.com/rss/search?q={query}&hl=ja&gl=JP&ceid=JP:ja"
    response = requests.get(url, timeout=10, headers={"User-Agent": "KABUPICK-X/1.0"})
    response.raise_for_status()
    root = ET.fromstring(response.content)
    items = []
    for item in root.findall("./channel/item")[:12]:
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        published = item.findtext("pubDate") or ""
        source = item.findtext("source") or "Google News"
        if title and link:
            items.append({"title": title, "link": link, "published": published, "source": source})
    return items


def render_menu_page(page: str, code: str) -> None:
    if page == "株価スクリーニング":
        st.subheader("株価スクリーニング")
        st.caption("登録した銘柄を、4つの観点で比較します。全市場を網羅する機能ではありません。")
        mode = st.selectbox("スクリーニング条件", ["成長重視", "割安重視", "高配当", "財務安定"])
        raw = st.text_input("調べる銘柄コード（カンマ区切り）", value="3921,4674,4828,4345")
        codes = [x.strip() for x in raw.split(",") if x.strip().isdigit() and len(x.strip()) == 4]
        rows = []
        with st.spinner("銘柄情報を更新しています…"):
            for item in codes[:20]:
                try:
                    info = load_fundamentals(item)
                    rows.append({"コード": item, "会社名": info.get("longName") or "—", "PER": info.get("trailingPE"), "PBR": info.get("priceToBook"), "ROE": info.get("returnOnEquity"), "売上成長率": info.get("revenueGrowth"), "配当利回り": info.get("dividendYield"), "自己資本比率の代替指標(負債比率)": info.get("debtToEquity")})
                except Exception:
                    continue
        if rows:
            table = pd.DataFrame(rows)
            if mode == "成長重視": table = table.sort_values("売上成長率", ascending=False, na_position="last")
            elif mode == "割安重視": table = table.sort_values("PER", ascending=True, na_position="last")
            elif mode == "高配当": table = table.sort_values("配当利回り", ascending=False, na_position="last")
            else: table = table.sort_values("自己資本比率の代替指標(負債比率)", ascending=True, na_position="last")
            st.dataframe(table, use_container_width=True)
        else:
            st.info("有効な銘柄データがありません。コードと通信状態を確認してください。")
        st.warning("指標は取得時点の参考値です。銘柄選びや売買の最終判断には使わないでください。")
    elif page == "ウォッチリスト":
        st.subheader("ウォッチリスト")
        items = read_json(WATCHLIST_FILE, ["3921", "4674", "4828", "4345"])
        add = st.text_input("銘柄コードを追加", max_chars=4)
        c1, c2 = st.columns(2)
        if c1.button("追加", use_container_width=True) and add.isdigit() and len(add) == 4 and add not in items:
            items.append(add); write_json(WATCHLIST_FILE, items); st.rerun()
        if c2.button("現在の一覧を保存", use_container_width=True):
            write_json(WATCHLIST_FILE, items); st.success("保存しました。")
        rows = []
        for item in items:
            try:
                d = load_data(item); last = d.iloc[-1]; prev = d.iloc[-2]
                rows.append({"コード": item, "終値": float(last.Close), "前日比": float(last.Close - prev.Close), "更新日": d.index[-1].strftime("%Y-%m-%d")})
            except Exception:
                rows.append({"コード": item, "終値": None, "前日比": None, "更新日": "取得失敗"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        for item in items:
            st.link_button(f"{item}.T を開く", f"https://finance.yahoo.co.jp/quote/{item}.T")
    elif page == "メモ":
        st.subheader("投資メモ")
        title = st.text_input("タイトル")
        body = st.text_area("メモ内容", height=160)
        if st.button("メモを保存", type="primary") and body.strip():
            memos = read_json(MEMOS_FILE, [])
            memos.insert(0, {"title": title or "無題", "body": body, "created": datetime.now().strftime("%Y-%m-%d %H:%M")})
            write_json(MEMOS_FILE, memos); st.success("保存しました。"); st.rerun()
        for memo in read_json(MEMOS_FILE, []):
            with st.expander(f"{memo.get('title', '無題')}　{memo.get('created', '')}"):
                st.write(memo.get("body", ""))
    elif page == "株価ニュース":
        st.subheader(f"株価ニュース / {code}.T")
        st.caption("公開RSSの見出しを5分ごとに更新しています。記事本文は配信元で確認してください。")
        try:
            news = fetch_news(code)
            if news:
                for item in news:
                    st.markdown(f"**[{item['title']}]({item['link']})**")
                    st.caption(f"{item['source']}　{item['published']}")
            else:
                st.info("該当するニュースがありませんでした。")
        except Exception as exc:
            st.warning(f"ニュースを取得できませんでした。各サイトのリンクから確認してください。 ({exc})")
        st.divider()
        st.markdown("#### 情報源を直接開く")
        for name, url in site_links(code):
            st.link_button(name, url, use_container_width=True)
    elif page == "情報サイト":
        st.subheader(f"銘柄情報サイト / {code}.T")
        for name, url in site_links(code):
            st.link_button(name, url, use_container_width=True)


def diagnose(data: pd.DataFrame) -> tuple[int, str, str, list[str]]:
    latest = data.iloc[-1]
    previous = data.iloc[-2]
    score = 50
    reasons: list[str] = []
    above_both = latest.Close > latest.SMA25 and latest.Close > latest.SMA75
    below_both = latest.Close < latest.SMA25 and latest.Close < latest.SMA75
    if above_both:
        score += 20
        reasons.append("株価が25日線・75日線の両方を上回っています。")
    elif below_both:
        score -= 20
        reasons.append("株価が25日線・75日線の両方を下回っています。")
    else:
        reasons.append("株価は25日線・75日線の間にあります。")

    if latest.RSI14 <= 30:
        score += 25
        reasons.append(f"RSIは{latest.RSI14:.1f}で、売られすぎの目安（30以下）です。")
    elif latest.RSI14 >= 70:
        score -= 25
        reasons.append(f"RSIは{latest.RSI14:.1f}で、買われすぎの目安（70以上）です。")
    else:
        reasons.append(f"RSIは{latest.RSI14:.1f}で、極端な過熱・売られすぎではありません。")

    golden_cross = previous.MACD <= previous.Signal and latest.MACD > latest.Signal
    if golden_cross:
        score += 25
        reasons.append("MACDが直近でゴールデンクロスしました。")
    elif latest.MACD > latest.Signal:
        reasons.append("MACDはシグナル線を上回っています。")
    else:
        reasons.append("MACDはシグナル線を下回っています。")

    score = max(0, min(100, score))
    if score >= 80:
        return score, "SS級の買いシグナル", "ss", reasons
    if score >= 65:
        return score, "買い候補", "buy", reasons
    if score >= 36:
        return score, "様子見", "wait", reasons
    return score, "売り候補", "sell", reasons


def signal_points(data: pd.DataFrame) -> pd.DataFrame:
    """MACDクロスと移動平均の位置から、チャートに表示する売買ポイントを作る。"""
    points = data.copy()
    points["BUY"] = (points.MACD > points.Signal) & (points.MACD.shift(1) <= points.Signal.shift(1)) & (points.Close > points.SMA25)
    points["SELL"] = (points.MACD < points.Signal) & (points.MACD.shift(1) >= points.Signal.shift(1)) & (points.Close < points.SMA25)
    return points


def forecast_scenario(data: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """過去60営業日の対数リターン平均・変動幅を使う簡易シナリオ推定。"""
    close = data.Close.dropna()
    window = close.tail(60)
    returns = np.log(window / window.shift(1)).dropna()
    drift = float(returns.mean())
    volatility = float(returns.std())
    future_index = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=days)
    steps = np.arange(1, days + 1)
    base = float(close.iloc[-1])
    median = base * np.exp(drift * steps)
    low = base * np.exp((drift - volatility) * steps)
    high = base * np.exp((drift + volatility) * steps)
    return pd.DataFrame({"中央値シナリオ": median, "下振れ目安": low, "上振れ目安": high}, index=future_index)


def backtest_summary(data: pd.DataFrame) -> dict:
    """MACDクロス後10営業日のリターンを集計する簡易検証。"""
    signals = signal_points(data)
    buy_indices = list(signals.index[signals.BUY])
    returns = []
    for idx in buy_indices:
        pos = data.index.get_loc(idx)
        if pos + 10 < len(data):
            returns.append(float(data.Close.iloc[pos + 10] / data.Close.iloc[pos] - 1))
    if not returns:
        return {"件数": 0, "勝率": None, "平均": None, "累積": None}
    return {"件数": len(returns), "勝率": float(np.mean(np.array(returns) > 0)), "平均": float(np.mean(returns)), "累積": float(np.prod(1 + np.array(returns)) - 1)}


st.markdown("<div class='terminal-title'>MARKET TERMINAL / JAPAN EQUITIES</div>", unsafe_allow_html=True)
st.title("株式タイミング・ターミナル")
st.caption("テクニカル指標を一覧化した参考ツールです。自動売買は行いません。")

with st.sidebar:
    st.markdown("## KABUPICK X")
    watch_for_badge = read_json(WATCHLIST_FILE, ["3921", "4674", "4828", "4345"])
    unseen = read_json(SIGNAL_STATE_FILE, {"unread": []}).get("unread", [])
    unread_count = len(unseen)
    if unread_count:
        st.markdown(f"<span class='menu-badge'>● {unread_count}件の未確認通知</span>", unsafe_allow_html=True)
    menu_items = ["タイミング検索", "株価スクリーニング", "ウォッチリスト", "通知", "メモ", "株価ニュース", "情報サイト"]
    page = st.radio(
        "メニュー",
        menu_items,
        index=0,
        key="main_menu",
        format_func=lambda item: f"{item}　🔴{unread_count}" if item == "通知" and unread_count else item,
    )
    st.divider()
    st.markdown("### 銘柄検索")
    code = st.text_input("日本株コード", value=st.session_state.get("input_code", "4783"), max_chars=4, placeholder="例：7203", help="4桁コードを入力してください")
    st.session_state.input_code = code
    run = st.button("この銘柄を分析", type="primary", use_container_width=True)
    st.divider()
    st.caption("株価データ：Yahoo!ファイナンス　｜　更新：5分ごと")
    st.caption("サイドバー上部の矢印でメニューを収納・再表示できます。")

if page == "通知":
    st.subheader("シグナル通知")
    current_items = read_json(WATCHLIST_FILE, ["3921", "4674", "4828", "4345"])
    if st.button("監視銘柄を再確認", type="primary"):
        refresh_signal_notifications(current_items)
        st.rerun()
    signal_state = read_json(SIGNAL_STATE_FILE, {"unread": []})
    notifications = signal_state.get("unread", [])
    st.caption("ウォッチリストを開くたびに最新データで判定を再確認します。価格データ自体は5分間キャッシュされます。")
    if notifications:
        st.markdown(f"未確認の通知 <span class='menu-badge'>{len(notifications)}</span>", unsafe_allow_html=True)
        for notice in reversed(notifications):
            st.info(f"{notice['code']}　{notice['label']}　{notice['score']}点　{notice['time']}")
        if st.button("通知を確認済みにする"):
            signal_state["unread"] = []
            write_json(SIGNAL_STATE_FILE, signal_state)
            st.rerun()
    else:
        st.success("未確認の買いシグナルはありません。")
    st.divider()
    st.markdown("#### LINE通知設定")
    line_ready = bool(not HOSTED_MODE and os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip() and os.getenv("LINE_USER_ID", "").strip())
    st.write("LINE Messaging API：" + ("接続情報あり" if line_ready else "停止中（公開版は認証導入後に利用可能）" if HOSTED_MODE else "未設定（アプリ内バッジは利用できます）"))
    st.caption("公開状態では訪問者から持ち主への通知を防ぐため、LINE送信を停止しています。")
    if line_ready and st.button("LINEへテスト通知を送る"):
        ok, detail = send_line_alert("KABUPICK X のLINE通知テストです。")
        (st.success if ok else st.error)(detail)
    st.stop()

if page != "タイミング検索":
    render_menu_page(page, code)
    if page == "ウォッチリスト":
        latest_unread = refresh_signal_notifications(read_json(WATCHLIST_FILE, ["3921", "4674", "4828", "4345"]))
        if latest_unread:
            st.markdown(f"新着シグナル通知 <span class='menu-badge'>{len(latest_unread)}</span>　サイドバーの「通知」で確認できます。", unsafe_allow_html=True)
    st.caption(f"最終確認：{datetime.now().strftime('%Y-%m-%d %H:%M')}　|　データ取得は5分キャッシュ")
    st.stop()

if run or "data" not in st.session_state or refresh_count > 0 or st.session_state.get("code") != code:
    if not code.isdigit() or len(code) != 4:
        st.error("4桁の数字を入力してください。")
        st.stop()
    try:
        with st.spinner("株価データを取得・計算しています…"):
            data = load_data(code)
        st.session_state.data = data
        st.session_state.code = code
    except Exception as exc:
        st.error(str(exc))
        st.stop()

data = st.session_state.data
code = st.session_state.code
updated_notifications = refresh_signal_notifications(list(dict.fromkeys(read_json(WATCHLIST_FILE, ["3921", "4674", "4828", "4345"]) + [code])))
if len(updated_notifications) > unread_count:
    st.rerun()
latest = data.iloc[-1]
score, message, kind, reasons = diagnose(data)
price = float(latest.Close)
signals = signal_points(data)

st.subheader(f"{code}.T　{data.index[-1].strftime('%Y-%m-%d')} 終値ベース")
st.markdown(
    f'<div class="signal {kind}"><div class="signal-score">{score}点</div><div class="signal-message">{message}</div></div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("現在株価", f"¥{price:,.2f}")
m2.metric("目標株価（+15%）", f"¥{price * 1.15:,.2f}")
m3.metric("損切りライン（-7%）", f"¥{price * 0.93:,.2f}")
m4.metric("1株の必要金額", f"¥{price:,.2f}")

recent = signals.loc[signals.BUY | signals.SELL].tail(8).iloc[::-1]
left, right = st.columns([2.3, 1], gap="medium")
with left:
    st.markdown("<div class='terminal-card'><div class='terminal-title'>PRICE / SIGNAL MAP</div>", unsafe_allow_html=True)
    chart = go.Figure()
    chart.add_trace(go.Candlestick(x=data.index, open=data.Open, high=data.High, low=data.Low, close=data.Close, name="価格", increasing_line_color="#38bdf8", decreasing_line_color="#fb385b"))
    chart.add_trace(go.Scatter(x=data.index, y=data.SMA25, name="SMA25", line=dict(color="#f59e0b", width=1.4)))
    chart.add_trace(go.Scatter(x=data.index, y=data.SMA75, name="SMA75", line=dict(color="#a78bfa", width=1.4)))
    buy = signals[signals.BUY]
    sell = signals[signals.SELL]
    chart.add_trace(go.Scatter(x=buy.index, y=buy.Low * .985, mode="markers+text", text=["BUY"] * len(buy), textposition="bottom center", name="BUY", marker=dict(symbol="triangle-up", size=13, color="#22d3ee"), textfont=dict(color="#22d3ee", size=10)))
    chart.add_trace(go.Scatter(x=sell.index, y=sell.High * 1.015, mode="markers+text", text=["SELL"] * len(sell), textposition="top center", name="SELL", marker=dict(symbol="triangle-down", size=13, color="#fb7185"), textfont=dict(color="#fb7185", size=10)))
    chart.update_layout(height=540, template="plotly_dark", paper_bgcolor="#0d131a", plot_bgcolor="#0d131a", margin=dict(l=10, r=10, t=25, b=10), hovermode="x unified", xaxis_rangeslider_visible=False, yaxis_title="価格（円）", legend=dict(orientation="h", y=1.02))
    st.plotly_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
with right:
    st.markdown("<div class='terminal-card'><div class='terminal-title'>WATCHLIST / SIGNAL LOG</div>", unsafe_allow_html=True)
    if recent.empty:
        st.write("直近のクロスシグナルはありません。")
    else:
        for idx, row in recent.iterrows():
            label = "BUY" if row.BUY else "SELL"
            cls = "up" if label == "BUY" else "down"
            st.markdown(f"<div class='ticker-row'><span>{idx.strftime('%m/%d')}</span><b class='{cls}'>{label}</b><span>¥{row.Close:,.0f}</span></div>", unsafe_allow_html=True)
    st.markdown("<br><div class='terminal-title'>CURRENT INDICATORS</div>", unsafe_allow_html=True)
    st.write(f"RSI14　{latest.RSI14:.1f}")
    st.write(f"MACD　{latest.MACD:.2f}")
    st.write(f"SMA25　¥{latest.SMA25:,.0f}")
    st.markdown("</div>", unsafe_allow_html=True)

tab_reason, tab_chart = st.tabs(["判定の根拠", "シグナル履歴"])
with tab_reason:
    st.markdown("#### 判定に使った材料")
    for reason in reasons:
        st.write(f"- {reason}")
    r1, r2, r3 = st.columns(3)
    r1.metric("RSI（14日）", f"{latest.RSI14:.1f}")
    r2.metric("25日移動平均", f"¥{latest.SMA25:,.2f}")
    r3.metric("75日移動平均", f"¥{latest.SMA75:,.2f}")
    st.warning("これは投資助言ではありません。ニュース、決算、出来高、手数料、保有状況などはスコアに含まれません。")

with tab_chart:
    st.markdown("#### BUY / SELLシグナル履歴")
    if recent.empty:
        st.info("直近1年に条件を満たすクロスシグナルはありません。")
    else:
        history = recent[["Close", "RSI14", "SMA25", "SMA75", "MACD", "Signal", "BUY", "SELL"]].copy()
        history.insert(0, "判定", np.where(history.BUY, "BUY", "SELL"))
        history = history.drop(columns=["BUY", "SELL"]).rename(columns={"Close":"終値", "RSI14":"RSI14", "SMA25":"SMA25", "SMA75":"SMA75", "MACD":"MACD", "Signal":"MACD Signal"})
        st.dataframe(history.style.format({"終値":"¥{:,.0f}", "SMA25":"¥{:,.0f}", "SMA75":"¥{:,.0f}", "RSI14":"{:.1f}", "MACD":"{:.2f}", "MACD Signal":"{:.2f}"}), use_container_width=True)

st.divider()
tab_watch, tab_forecast, tab_verify = st.tabs(["👀 お気に入り一括監視", "🔮 予測・シナリオ", "💡 根拠・バックテスト"])
with tab_watch:
    st.markdown("#### ウォッチリストの一括確認")
    watch_items = read_json(WATCHLIST_FILE, ["3921", "4674", "4828", "4345"])
    watch_rows = []
    for item in watch_items:
        try:
            wd = load_data(item)
            ws, wm, wk, _ = diagnose(wd)
            watch_rows.append({"コード": item, "終値": float(wd.Close.iloc[-1]), "判定": wm, "スコア": ws, "更新日": wd.index[-1].strftime("%Y-%m-%d")})
        except Exception:
            watch_rows.append({"コード": item, "終値": None, "判定": "取得失敗", "スコア": None, "更新日": "—"})
    st.dataframe(pd.DataFrame(watch_rows), use_container_width=True)
    st.caption("ウォッチリストは左メニューの「ウォッチリスト」から編集できます。")

with tab_forecast:
    st.markdown("#### 30営業日の統計シナリオ")
    forecast = forecast_scenario(data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index.tail(90), y=data.Close.tail(90), name="実績", line=dict(color="#38bdf8", width=2)))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast["中央値シナリオ"], name="中央値", line=dict(color="#fbbf24", dash="dot", width=2)))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast["上振れ目安"], name="上振れ目安", line=dict(color="#4ade80", dash="dot")))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast["下振れ目安"], name="下振れ目安", line=dict(color="#fb7185", dash="dot"), fill="tonexty", fillcolor="rgba(148,163,184,.08)"))
    fig.update_layout(height=430, template="plotly_dark", paper_bgcolor="#0d131a", plot_bgcolor="#0d131a", margin=dict(l=10, r=10, t=25, b=10), yaxis_title="価格（円）")
    st.plotly_chart(fig, use_container_width=True)
    current = price
    target = current * 1.10
    probability = float(np.mean(forecast["上振れ目安"] >= target))
    p1, p2, p3 = st.columns(3)
    p1.metric("30日中央値", f"¥{forecast['中央値シナリオ'].iloc[-1]:,.0f}")
    p2.metric("+10%到達の簡易目安", f"{probability:.0%}")
    p3.metric("30日上振れ目安", f"¥{forecast['上振れ目安'].iloc[-1]:,.0f}")
    st.warning("これは過去60営業日の平均リターンと変動幅だけを使った統計的なシナリオです。AIによる確定予測でも、利益を保証するものでもありません。")

with tab_verify:
    st.markdown("#### 判定根拠と簡易バックテスト")
    bt = backtest_summary(data)
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("買いシグナル数", bt["件数"])
    b2.metric("10日後の勝率", "—" if bt["勝率"] is None else f"{bt['勝率']:.0%}")
    b3.metric("平均リターン", "—" if bt["平均"] is None else f"{bt['平均']:.2%}")
    b4.metric("単純累積", "—" if bt["累積"] is None else f"{bt['累積']:.2%}")
    st.write("テクニカル：25日線・75日線・RSI14・MACDクロスを使用")
    st.write("ファンダメンタルズ：スクリーニング画面でPER・PBR・ROE・成長率・配当利回りなどを確認できます。")
    st.warning("バックテストは手数料・税金・スリッページ・約定不能を考慮しない簡易検証です。過去の結果は将来の成績を示しません。")

st.caption(f"最終更新：{datetime.now().strftime('%Y-%m-%d %H:%M')}（この画面の表示時刻）")

