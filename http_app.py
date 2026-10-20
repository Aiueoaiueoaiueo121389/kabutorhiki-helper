from __future__ import annotations

from html import escape
import secrets
from threading import Lock
from time import monotonic
from urllib.parse import quote

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


app = FastAPI(title="KABUPICK X", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_cache_lock = Lock()
_fundamentals_cache: dict[str, tuple[float, dict]] = {}
_sessions: dict[str, tuple[float, dict]] = {}
_sessions_lock = Lock()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    relative = gain / loss.replace(0, np.nan)
    values = 100 - (100 / (1 + relative))
    return values.mask((loss == 0) & (gain > 0), 100).mask((gain == 0) & (loss > 0), 0)


def load_data(code: str) -> pd.DataFrame:
    now = monotonic()
    with _cache_lock:
        cached = _cache.get(code)
        if cached and now - cached[0] < 300:
            return cached[1].copy()
    raw = yf.Ticker(f"{code}.T").history(period="1y", interval="1d", auto_adjust=False)
    if raw.empty:
        raise ValueError("株価データが見つかりません。銘柄コードをご確認ください。")
    data = raw.rename(columns=str.title).copy()
    close = data["Close"]
    data["SMA25"] = close.rolling(25).mean()
    data["SMA75"] = close.rolling(75).mean()
    data["RSI14"] = rsi(close)
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    data["MACD"] = fast - slow
    data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"] = data["MACD"] - data["Signal"]
    data["SMA25_Slope5"] = data["SMA25"] - data["SMA25"].shift(5)
    if "Volume" in data:
        data["VolumeMA20"] = data["Volume"].rolling(20).mean()
        data["VolumeRatio"] = data["Volume"] / data["VolumeMA20"].replace(0, np.nan)
    else:
        data["VolumeRatio"] = np.nan
    data = data.dropna(subset=["SMA25", "SMA75", "RSI14", "MACD", "Signal"])
    with _cache_lock:
        _cache[code] = (now, data.copy())
    return data


def diagnose(data: pd.DataFrame) -> tuple[int, str, str, list[str]]:
    latest = data.iloc[-1]
    trend_up = latest.Close > latest.SMA25 > latest.SMA75 and latest.SMA25_Slope5 > 0
    trend_down = latest.Close < latest.SMA25 < latest.SMA75 and latest.SMA25_Slope5 < 0
    macd_up = latest.MACD > latest.Signal and latest.MACD_Hist > data.MACD_Hist.iloc[-4]
    macd_down = latest.MACD < latest.Signal and latest.MACD_Hist < data.MACD_Hist.iloc[-4]
    rsi_ok = 42 <= latest.RSI14 <= 68
    volume_ok = pd.notna(latest.VolumeRatio) and latest.VolumeRatio >= 1.2
    score = 30 if trend_up else (10 if latest.Close > latest.SMA75 else 0)
    score += 25 if macd_up else (10 if latest.MACD > latest.Signal else 0)
    score += 20 if rsi_ok else (8 if 35 <= latest.RSI14 < 42 or 68 < latest.RSI14 <= 75 else 0)
    score += 15 if volume_ok else (8 if pd.isna(latest.VolumeRatio) or latest.VolumeRatio >= 1 else 0)
    score += 10 if latest.RSI14 > data.RSI14.iloc[-4] else 0
    score = int(max(0, min(100, score)))
    reasons = [
        "上昇トレンド条件成立" if trend_up else "下降トレンド条件成立" if trend_down else "トレンド条件は未成立",
        "MACDの勢いは改善" if macd_up else "MACDの勢いは悪化" if macd_down else "MACDの勢いは明確でない",
        f"RSI14: {latest.RSI14:.1f}（買い条件の目安 42〜68）",
        f"出来高: {latest.VolumeRatio:.2f}倍（20日平均比）" if pd.notna(latest.VolumeRatio) else "出来高確認なし",
    ]
    if trend_up and macd_up and rsi_ok and score >= 90 and volume_ok:
        return score, "買い条件が強く成立", "strong", reasons
    if trend_up and macd_up and rsi_ok and score >= 75:
        return score, "買い条件が複数成立", "buy", reasons
    if trend_down and macd_down and score <= 35:
        return score, "売却検討条件が成立", "sell", reasons
    return score, "条件不一致・様子見", "wait", reasons


def signal_points(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    buy_setup = (
        (result.Close > result.SMA25)
        & (result.SMA25 > result.SMA75)
        & (result.SMA25_Slope5 > 0)
        & (result.MACD > result.Signal)
        & (result.MACD_Hist > result.MACD_Hist.shift(3))
        & result.RSI14.between(42, 68)
    )
    sell_setup = (
        (result.Close < result.SMA25)
        & (result.SMA25 < result.SMA75)
        & (result.SMA25_Slope5 < 0)
        & (result.MACD < result.Signal)
        & (result.MACD_Hist < result.MACD_Hist.shift(3))
    )
    result["BUY"] = buy_setup & ~buy_setup.shift(1, fill_value=False)
    result["SELL"] = sell_setup & ~sell_setup.shift(1, fill_value=False)
    return result


def make_chart(data: pd.DataFrame) -> str:
    rows = data.tail(100)
    width, height = 900, 330
    left, right, top, bottom = 52, 16, 20, 32
    low = float(rows[["Low", "SMA25", "SMA75"]].min().min())
    high = float(rows[["High", "SMA25", "SMA75"]].max().max())
    span = max(high - low, 1e-6)
    x_step = (width - left - right) / max(len(rows) - 1, 1)

    def point(index: int, value: float) -> tuple[float, float]:
        x = left + index * x_step
        y = top + (high - float(value)) / span * (height - top - bottom)
        return x, y

    def line_path(column: str) -> str:
        coords = [point(i, value) for i, value in enumerate(rows[column])]
        return " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(coords))

    candle_svg = []
    body_width = max(2.0, min(7.0, x_step * 0.58))
    marks = signal_points(rows)
    for i, (_, row) in enumerate(rows.iterrows()):
        x, y_low = point(i, row.Low)
        _, y_high = point(i, row.High)
        _, y_open = point(i, row.Open)
        _, y_close = point(i, row.Close)
        color = "#16a085" if row.Close >= row.Open else "#e74c3c"
        top_body = min(y_open, y_close)
        body_height = max(1.5, abs(y_open - y_close))
        candle_svg.append(f'<line x1="{x:.1f}" y1="{y_high:.1f}" x2="{x:.1f}" y2="{y_low:.1f}" stroke="{color}"/>')
        candle_svg.append(f'<rect x="{x-body_width/2:.1f}" y="{top_body:.1f}" width="{body_width:.1f}" height="{body_height:.1f}" fill="{color}"/>')
        if bool(marks.BUY.iloc[i]):
            candle_svg.append(f'<text x="{x:.1f}" y="{max(12,y_high-8):.1f}" text-anchor="middle" class="buy-mark">▲ BUY</text>')
        if bool(marks.SELL.iloc[i]):
            candle_svg.append(f'<text x="{x:.1f}" y="{min(height-4,y_low+18):.1f}" text-anchor="middle" class="sell-mark">▼ SELL</text>')
    grid = []
    for fraction in (0, .25, .5, .75, 1):
        y = top + fraction * (height - top - bottom)
        value = high - fraction * span
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/><text x="{left-8}" y="{y+4:.1f}" text-anchor="end" class="axis">{value:,.0f}</text>')
    first_date = rows.index[0].strftime("%m/%d")
    last_date = rows.index[-1].strftime("%m/%d")
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="株価チャート">'
        + "".join(grid)
        + f'<path d="{line_path("SMA75")}" class="ma75"/><path d="{line_path("SMA25")}" class="ma25"/>'
        + "".join(candle_svg)
        + f'<text x="{left}" y="{height-7}" class="axis">{first_date}</text><text x="{width-right}" y="{height-7}" text-anchor="end" class="axis">{last_date}</text>'
        + "</svg>"
    )


def parse_code(code: str) -> str:
    value = code.strip()
    if not (value.isdigit() and len(value) == 4):
        raise HTTPException(status_code=422, detail="銘柄コードは4桁の数字で入力してください。")
    return value


def load_fundamentals(code: str) -> dict:
    now = monotonic()
    with _cache_lock:
        cached = _fundamentals_cache.get(code)
        if cached and now - cached[0] < 300:
            return cached[1].copy()
    info = yf.Ticker(f"{code}.T").info
    fields = ("longName", "shortName", "trailingPE", "priceToBook", "returnOnEquity", "revenueGrowth", "dividendYield", "debtToEquity")
    result = {field: info.get(field) for field in fields}
    with _cache_lock:
        _fundamentals_cache[code] = (now, result.copy())
    return result


def numeric_or_none(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


@app.get("/healthz", response_class=HTMLResponse)
async def healthz():
    return "ok"


def session_for(request: Request) -> tuple[str, dict]:
    session_id = request.cookies.get("kabupick_session", "")
    with _sessions_lock:
        current = _sessions.get(session_id)
        if current and monotonic() - current[0] < 30 * 24 * 60 * 60:
            _sessions[session_id] = (monotonic(), current[1])
            return session_id, current[1]
        session_id = secrets.token_urlsafe(24)
        value = {"watchlist": [], "memos": [{
            "title": "KABUPICK X スクリーニング条件",
            "body": (
                "【メイン条件】本業の営業利益が中長期で継続成長し、売上も成長傾向。原則黒字で、実際に販売中の製品・サービスがあること。 "
                "ROE 15%以上、自己資本比率50%以上、PBR 3倍以下を基本ラインとし、時価総額100〜1,000億円程度を中心に探す。配当利回り3%以上は加点するが必須ではなく、利益成長を優先。\n\n"
                "【評価の優先順位】過去の営業利益成長→現在の営業利益成長→売上成長→営業利益率→営業CF・FCF→ROE・ROIC→財務健全性→競争優位性（MOAT）→成長余地→企業規模→PER・PBR→配当。 "
                "営業利益が伸びても営業CF/FCFが継続悪化する企業、慢性赤字、借金依存、一時特需・M&A頼み、成長失速、弱いMOATは慎重に評価する。\n\n"
                "【最高優良株の補助条件】長期増収増益、高ROE/ROIC、高営業利益率、強い財務、安定した営業CF、FCF黒字、継続増配、強いMOATを総合確認。条件から少し外れる例外候補は、残す理由を明記する。赤字企業やテーマ・配当・PER・PBR・株価の安さだけで高評価にしない。\n\n"
                "【重要】企業の質と買い場評価は分ける。好業績でも割高なら待つ。テーマは調査の入口に過ぎず、実際の売上・営業利益で確認する。SS+は極めて厳格（長期連続増益、足元・会社予想も成長、ROE20%以上、自己資本比率70%以上、営業利益率20%以上、CF/FCF健全、割高でない、明確なMOAT等）。重大な穴があれば付けない。"
            ),
        }]}
        _sessions[session_id] = (monotonic(), value)
        expired = [key for key, (seen, _) in _sessions.items() if monotonic() - seen > 30 * 24 * 60 * 60]
        for key in expired:
            _sessions.pop(key, None)
        return session_id, value


def page_response(request: Request, name: str, context: dict, session_id: str) -> HTMLResponse:
    response = templates.TemplateResponse(request=request, name=name, context=context)
    response.set_cookie(
        "kabupick_session",
        session_id,
        max_age=30 * 24 * 60 * 60,
        httponly=True,
        secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https",
        samesite="lax",
    )
    if not request.cookies.get("kabupick_startup_seen"):
        response.set_cookie("kabupick_startup_seen", "1", max_age=365 * 24 * 60 * 60, httponly=True, secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https", samesite="lax")
    return response


def safe_return_path(value: str) -> str:
    if value.startswith("/") and not value.startswith("//"):
        return value
    return "/?page=timing"


@app.post("/welcome/ack")
async def acknowledge_welcome(request: Request):
    form = await request.form()
    response = RedirectResponse(safe_return_path(str(form.get("next", "/?page=timing"))), status_code=303)
    response.set_cookie("kabupick_notice_seen", "1", max_age=365 * 24 * 60 * 60, httponly=True, secure=request.headers.get("x-forwarded-proto") == "https", samesite="lax")
    return response


@app.post("/theme")
async def set_theme(request: Request):
    session_id, session = session_for(request)
    form = await request.form()
    session["theme"] = "light" if form.get("theme") == "light" else "dark"
    response = RedirectResponse(safe_return_path(str(form.get("next", "/?page=timing"))), status_code=303)
    response.set_cookie("kabupick_session", session_id, max_age=30 * 24 * 60 * 60, httponly=True, secure=request.headers.get("x-forwarded-proto") == "https", samesite="lax")
    return response


@app.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    page: str = Query("timing"),
    code: str = Query(""),
    codes: str = Query(""),
    sort: str = Query("growth"),
    message: str = Query(""),
    refresh: int = Query(0),
):
    session_id, session = session_for(request)
    context = {
        "request": request,
        "page": page,
        "code": code,
        "codes": codes,
        "error": "",
        "message": message,
        "content": None,
        "session": session,
        "theme": session.get("theme", "dark"),
        "show_welcome": not request.cookies.get("kabupick_notice_seen"),
        "show_startup": not request.cookies.get("kabupick_startup_seen"),
        "next_path": request.url.path + ("?" + request.url.query if request.url.query else ""),
    }
    if page == "screening":
        context["content"] = screening_context(codes, sort) if codes.strip() else {"rows": [], "failures": [], "sort": sort, "searched": False}
        context["sort"] = sort
        return page_response(request, "screening.html", context, session_id)
    if page == "watchlist":
        rows = []
        for item in session["watchlist"]:
            try:
                data = load_data(item)
                last = data.iloc[-1]
                try:
                    profile = load_fundamentals(item)
                    name = profile.get("longName") or profile.get("shortName") or "—"
                except Exception:
                    name = "—"
                rows.append({"code": item, "name": name, "price": float(last.Close), "date": data.index[-1].strftime("%Y-%m-%d"), "error": ""})
            except Exception as exc:
                rows.append({"code": item, "name": "—", "price": None, "date": "—", "error": str(exc)[:160]})
        context["content"] = {"rows": rows}
        return page_response(request, "watchlist.html", context, session_id)
    if page == "memos":
        return page_response(request, "memos.html", context, session_id)
    if page == "news":
        if not code.strip():
            context["content"] = {"items": []}
            return page_response(request, "news.html", context, session_id)
        try:
            import xml.etree.ElementTree as ET
            import requests
            url = f"https://news.google.com/rss/search?q={quote(code + ' 株式')}&hl=ja&gl=JP&ceid=JP:ja"
            response = requests.get(url, timeout=10, headers={"User-Agent": "KABUPICK-X/1.0"})
            response.raise_for_status()
            root = ET.fromstring(response.content)
            news = []
            for item in root.findall("./channel/item")[:12]:
                title, link = item.findtext("title"), item.findtext("link")
                if title and link:
                    news.append({"title": title, "link": link, "date": item.findtext("pubDate") or ""})
            context["content"] = {"items": news}
        except Exception as exc:
            context["error"] = f"ニュースを取得できませんでした: {type(exc).__name__}"
        return page_response(request, "news.html", context, session_id)
    if page == "sources":
        quoted_code = quote(code.strip())
        news_query = quote((code.strip() + " 株式") if code.strip() else "日本株")
        context["content"] = {"sites": [
            ("Yahoo!ファイナンス", f"https://finance.yahoo.co.jp/quote/{quoted_code}.T" if quoted_code else "https://finance.yahoo.co.jp/"),
            ("株探（Kabutan）", f"https://kabutan.jp/stock/?code={quoted_code}" if quoted_code else "https://kabutan.jp/"),
            ("みんかぶ", f"https://minkabu.jp/stock/{quoted_code}" if quoted_code else "https://minkabu.jp/"),
            ("IR BANK", f"https://irbank.net/{quoted_code}" if quoted_code else "https://irbank.net/"),
            ("バフェット・コード", f"https://www.buffett-code.com/company/{quoted_code}/" if quoted_code else "https://www.buffett-code.com/"),
            ("TradingView", f"https://jp.tradingview.com/symbols/TSE-{quoted_code}/" if quoted_code else "https://jp.tradingview.com/markets/stocks-japan/"),
            ("日本経済新聞 検索", f"https://www.nikkei.com/search/?query={quote((code + ' 株価') if code else '日本株') }"),
            ("Googleニュース", f"https://news.google.com/search?q={news_query}&hl=ja&gl=JP&ceid=JP%3Aja"),
            ("TDnet 適時開示（公式）", "https://www.release.tdnet.info/inbs/I_main_00.html"),
            ("EDINET 有価証券報告書（公式）", "https://disclosure.edinet-fsa.go.jp/"),
            ("日本取引所グループ（JPX）", "https://www.jpx.co.jp/"),
            ("企業IR検索", f"https://www.google.com/search?q={quote((code + ' IR 決算') if code else '日本株 企業 IR 決算') }"),
        ]}
        return page_response(request, "sources.html", context, session_id)
    if page == "timing" and not code.strip():
        return page_response(request, "home.html", context, session_id)
    try:
        safe_code = parse_code(code)
        if refresh:
            with _cache_lock:
                _cache.pop(safe_code, None)
        data = load_data(safe_code)
        try:
            profile = load_fundamentals(safe_code)
        except Exception:
            profile = {}
        score, label, kind, reasons = diagnose(data)
        last = data.iloc[-1]
        context["code"] = safe_code
        context["content"] = {
            "code": safe_code,
            "company_name": profile.get("longName") or profile.get("shortName") or "会社名を取得できませんでした",
            "stock_name": profile.get("shortName") or profile.get("longName") or f"{safe_code}.T",
            "is_watched": safe_code in session["watchlist"],
            "score": score,
            "label": label,
            "kind": kind,
            "reasons": reasons,
            "price": float(last.Close),
            "date": data.index[-1].strftime("%Y-%m-%d"),
            "rsi": float(last.RSI14),
            "sma25": float(last.SMA25),
            "sma75": float(last.SMA75),
            "chart": make_chart(data),
            "signals": [
                {"date": idx.strftime("%Y-%m-%d"), "kind": "BUY" if row.BUY else "SELL", "price": float(row.Close)}
                for idx, row in signal_points(data).loc[lambda frame: frame.BUY | frame.SELL].tail(8).iloc[::-1].iterrows()
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        context["error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
    return page_response(request, "home.html", context, session_id)


@app.post("/watchlist/add")
async def add_watchlist(request: Request):
    session_id, session = session_for(request)
    form = await request.form()
    code = str(form.get("code", "")).strip()
    next_path = safe_return_path(str(form.get("next", "")))
    message = ""
    if code.isdigit() and len(code) == 4 and code not in session["watchlist"] and len(session["watchlist"]) < 20:
        session["watchlist"].append(code)
        message = "ウォッチリストに追加しました。"
    elif code in session["watchlist"]:
        message = "この銘柄はすでにウォッチリストにあります。"
    elif code.isdigit() and len(code) == 4:
        message = "ウォッチリストは20銘柄まで追加できます。"
    else:
        message = "銘柄コードは4桁の数字で入力してください。"
    separator = "&" if "?" in next_path else "?"
    response = RedirectResponse(f"{next_path}{separator}message={quote(message)}", status_code=303)
    response.set_cookie("kabupick_session", session_id, max_age=30 * 24 * 60 * 60, httponly=True, secure=request.headers.get("x-forwarded-proto") == "https", samesite="lax")
    return response


@app.post("/watchlist/remove")
async def remove_watchlist(request: Request):
    session_id, session = session_for(request)
    form = await request.form()
    code = str(form.get("code", "")).strip()
    session["watchlist"] = [item for item in session["watchlist"] if item != code]
    response = RedirectResponse("/?page=watchlist", status_code=303)
    response.set_cookie("kabupick_session", session_id, max_age=30 * 24 * 60 * 60, httponly=True, secure=request.headers.get("x-forwarded-proto") == "https", samesite="lax")
    return response


@app.post("/memos/add")
async def add_memo(request: Request):
    session_id, session = session_for(request)
    form = await request.form()
    title = str(form.get("title", "")).strip()[:80] or "無題"
    body = str(form.get("body", "")).strip()[:2000]
    if body:
        session["memos"].insert(0, {"title": title, "body": body})
        del session["memos"][30:]
    response = RedirectResponse("/?page=memos", status_code=303)
    response.set_cookie("kabupick_session", session_id, max_age=30 * 24 * 60 * 60, httponly=True, secure=request.headers.get("x-forwarded-proto") == "https", samesite="lax")
    return response


def screening_context(raw_codes: str, sort: str) -> dict:
    valid = []
    for token in raw_codes.split(",")[:20]:
        token = token.strip()
        if token.isdigit() and len(token) == 4:
            valid.append(token)
    rows = []
    failures = []
    for code in dict.fromkeys(valid):
        try:
            info = load_fundamentals(code)
            rows.append({
                "code": code,
                "name": info.get("longName") or "—",
                "per": numeric_or_none(info.get("trailingPE")),
                "pbr": numeric_or_none(info.get("priceToBook")),
                "roe": numeric_or_none(info.get("returnOnEquity")),
                "growth": numeric_or_none(info.get("revenueGrowth")),
                "dividend": numeric_or_none(info.get("dividendYield")),
                "debt": numeric_or_none(info.get("debtToEquity")),
            })
        except Exception as exc:
            failures.append({"code": code, "reason": f"{type(exc).__name__}: {str(exc)[:160]}"})
    columns = {
        "growth": "growth",
        "value": "per",
        "dividend": "dividend",
        "debt": "debt",
    }
    sort_key = columns.get(sort, "growth")
    descending = sort_key in ("growth", "dividend")
    available = sorted((row for row in rows if row[sort_key] is not None), key=lambda row: row[sort_key], reverse=descending)
    rows = available + [row for row in rows if row[sort_key] is None]
    return {"rows": rows, "failures": failures, "sort": sort, "searched": True}
