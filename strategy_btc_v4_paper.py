# -*- coding: utf-8 -*-
"""
BTC/USDT 单币防过拟合 4H v4 云端模拟盘 (SUB-B 同款 GitHub Actions + Pages)
已内置全球无地区限制多行情源自动容灾 (解决 GitHub Actions 美国服务器 451 限制):
1. 主源: Binance 官方全球行情节点 data-api.binance.vision (含原生主动买入成交量，无 451 封锁)
2. 备源1: OKX 永续合约公开接口 (BTC-USDT-SWAP 4H K线 + 资金费率历史)
3. 备源2: Kraken 美国本土公开接口 (XBTUSD 4H K线，单次 720 根)
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).parent
ONE_WAY_COST = 0.0006
HEADERS = {"User-Agent": "Mozilla/5.0 BTC-Quant-v4-Paper/1.0"}


def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def _rsi(s, p=14):
    d = s.diff()
    up = d.clip(lower=0.0).ewm(alpha=1.0 / p, adjust=False).mean()
    dn = (-d.clip(upper=0.0)).ewm(alpha=1.0 / p, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + up / (dn + 1e-12))


def _atr(df, p=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / p, adjust=False).mean()


def _fetch_binance_vision_4h(limit=600) -> pd.DataFrame:
    """Binance 官方全球公开行情节点 (无美国 IP 451 限制)"""
    url = "https://data-api.binance.vision/api/v3/klines"
    r = requests.get(url, params={"symbol": "BTCUSDT", "interval": "4h", "limit": limit}, headers=HEADERS, timeout=15)
    r.raise_for_status()
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore"
    ]
    df = pd.DataFrame(r.json(), columns=cols)
    df["timestamp"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "taker_buy_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("timestamp").sort_index()[["open", "high", "low", "close", "volume", "taker_buy_volume"]]


def _fetch_okx_swap_4h(limit=600) -> pd.DataFrame:
    """OKX BTC-USDT 永续合约 4H K线备源"""
    rows = []
    after = None
    while len(rows) < limit:
        params = {"instId": "BTC-USDT-SWAP", "bar": "4H", "limit": "300"}
        if after:
            params["after"] = after
        r = requests.get("https://www.okx.com/api/v5/market/candles", params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            break
        rows.extend(data)
        after = data[-1][0]
        if len(data) < 300:
            break
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "volCcy", "volCcyQuote", "confirm"])
    df["timestamp"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 阳线近似估算主动买盘比例用于备源容灾
    df["taker_buy_volume"] = df["volume"] * np.where(df["close"] >= df["open"], 0.55, 0.45)
    return df.drop_duplicates("timestamp").set_index("timestamp").sort_index().tail(limit)[["open", "high", "low", "close", "volume", "taker_buy_volume"]]


def _fetch_kraken_4h() -> pd.DataFrame:
    """Kraken 美国本土公开接口 4H K线备源 (单次返回 720 根)"""
    r = requests.get("https://api.kraken.com/0/public/OHLC", params={"pair": "XBTUSD", "interval": 240}, headers=HEADERS, timeout=15)
    r.raise_for_status()
    res = r.json()["result"]
    pair_key = [k for k in res.keys() if k != "last"][0]
    df = pd.DataFrame(res[pair_key], columns=["ts", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["timestamp"] = pd.to_datetime(df["ts"].astype("int64"), unit="s", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["taker_buy_volume"] = df["volume"] * np.where(df["close"] >= df["open"], 0.55, 0.45)
    return df.set_index("timestamp").sort_index()[["open", "high", "low", "close", "volume", "taker_buy_volume"]]


def _fetch_funding_series(index: pd.DatetimeIndex) -> pd.Series:
    """优先从 OKX 永续拉取 8h 资金费率历史 (无美国 IP 限制)，失败则回退中性费率"""
    try:
        r = requests.get(
            "https://www.okx.com/api/v5/public/funding-rate-history",
            params={"instId": "BTC-USDT-SWAP", "limit": "100"},
            headers=HEADERS,
            timeout=10,
        )
        if r.status_code == 200 and r.json().get("data"):
            fdf = pd.DataFrame(r.json()["data"])
            fdf["ts"] = pd.to_datetime(fdf["fundingTime"].astype("int64"), unit="ms", utc=True)
            s = pd.to_numeric(fdf.set_index("ts")["fundingRate"], errors="coerce").sort_index()
            return s.reindex(index, method="ffill").fillna(0.0001)
    except Exception:
        pass
    return pd.Series(0.0001, index=index, dtype=float)


def fetch_live_4h(limit=600) -> tuple[pd.DataFrame, str]:
    sources = [
        ("Binance Global Data-API", lambda: _fetch_binance_vision_4h(limit)),
        ("OKX BTC-USDT-SWAP", lambda: _fetch_okx_swap_4h(limit)),
        ("Kraken XBTUSD", lambda: _fetch_kraken_4h()),
    ]
    last_err = None
    for name, fn in sources:
        try:
            df = fn()
            if len(df) >= 300:
                df["funding_rate"] = _fetch_funding_series(df.index)
                print(f"[行情源] 成功拉取 {name}: {len(df)} 根 4H K线 (最新: {df.index[-1]})")
                return df, name
        except Exception as e:
            print(f"[行情源容灾] {name} 不可用 ({e})，切换下一备源...")
            last_err = e
    raise RuntimeError(f"所有行情源均不可用: {last_err}")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_1"] = df["close"].pct_change(1)
    df["rvol_30"] = df["ret_1"].rolling(30).std() * np.sqrt(6 * 365)
    df["atr_14"] = _atr(df, 14)
    df["ema_50"], df["ema_120"] = _ema(df["close"], 50), _ema(df["close"], 120)
    for w in [20, 30, 45]:
        df[f"donchian_h_{w}"] = df["high"].rolling(w).max().shift(1)
        df[f"donchian_l_{w}"] = df["low"].rolling(w).min().shift(1)
    df["rsi_14"] = _rsi(df["close"], 14)
    df["vol_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-12)
    df["taker_buy_ratio"] = df["taker_buy_volume"] / (df["volume"] + 1e-12)
    fm = df["funding_rate"].rolling(180, min_periods=30).mean()
    fs = df["funding_rate"].rolling(180, min_periods=30).std() + 1e-8
    df["funding_z"] = ((df["funding_rate"] - fm) / fs).clip(-5, 5)
    d1 = df[["open", "high", "low", "close"]].resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    d1["ma20"], d1["ma60"] = d1["close"].rolling(20).mean(), d1["close"].rolling(60).mean()
    d1["regime_1d"] = np.where((d1["close"] > d1["ma60"]) & (d1["ma20"] > d1["ma60"]), 1, np.where((d1["close"] < d1["ma60"]) & (d1["ma20"] < d1["ma60"]), -1, 0))
    df["regime_1d"] = d1["regime_1d"].shift(1).reindex(df.index, method="ffill")
    return df.dropna()


def render_html(state, last_px, sig, regime, a, source_name):
    trades = state.get("trades", [])
    nt = len(trades)
    wr = (sum(1 for t in trades if t["net_pnl"] > 0) / nt * 100.0) if nt else 0.0
    eq = state["equity"]
    ret = (eq / state["initial_capital"] - 1.0) * 100.0
    dd = (eq / max(state["peak_equity"], 1.0) - 1.0) * 100.0
    reg_txt = "🟢 牛市体制 (做多开启)" if regime == 1 else ("🔴 熊市体制 (空仓防守)" if regime == -1 else "🟡 震荡过渡 (强制空仓)")
    pos = state.get("position")
    pos_txt = "空仓观望 (FLAT)" if not pos else f"{pos['side']} {pos['qty']:.4f} BTC @ ${pos['entry_px']:,.2f} | 止损/吊灯 ${pos['sl_px']:,.1f}"
    rows = "".join(
        f"<tr><td>{t['exit_time'][:16]}</td><td>{t['side']}</td><td>${t['entry_px']:,.1f}</td><td>${t['exit_px']:,.1f}</td>"
        f"<td style='color:{'#10b981' if t['net_pnl']>0 else '#ef4444'}'>{t['net_pnl']:+,.2f}</td><td>{t['exit_reason']}</td></tr>"
        for t in reversed(trades[-25:])
    ) or "<tr><td colspan='6' style='text-align:center;color:#94a3b8'>等待首笔 4H 信号...</td></tr>"

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC/USDT 防过拟合 v4 模拟盘</title>
<style>
body{{background:#0f172a;color:#f8fafc;font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:0;padding:16px;}}
.card{{background:#1e293b;border-radius:12px;padding:16px;margin-bottom:14px;border:1px solid #334155;}}
.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;}}
.lbl{{font-size:12px;color:#94a3b8;}} .val{{font-size:20px;font-weight:700;margin-top:4px;}}
table{{width:100%;border-collapse:collapse;font-size:13px;}} th,td{{padding:8px 6px;border-bottom:1px solid #334155;text-align:left;}}
</style></head><body>
<h2 style="margin:0 0 6px 0;">🛡️ BTC/USDT 单币防过拟合 v4 模拟盘</h2>
<div style="font-size:12px;color:#94a3b8;margin-bottom:14px;">启动日: {state['live_start']} | 更新(UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} | BTC现价: ${last_px:,.2f} | 数据源: {source_name}</div>
<div class="grid">
  <div class="card"><div class="lbl">模拟盘净值</div><div class="val">${eq:,.2f}</div></div>
  <div class="card"><div class="lbl">累计收益率</div><div class="val" style="color:{'#10b981' if ret>=0 else '#ef4444'}">{ret:+.2f}%</div></div>
  <div class="card"><div class="lbl">实盘胜率 / 笔数</div><div class="val">{wr:.1f}% ({nt}笔)</div></div>
  <div class="card"><div class="lbl">当前回撤 / 杠杆上限</div><div class="val">{dd:.2f}% / {a['max_leverage']}x</div></div>
</div>
<div class="card"><div class="lbl">宏观体制与当前持仓</div><div style="font-size:14px;margin-top:6px;">{reg_txt} | 原型: {a['name']}</div>
<div style="font-size:15px;font-weight:600;margin-top:6px;">{pos_txt}</div></div>
<div class="card"><div class="lbl" style="margin-bottom:8px;">近期成交 (最近25笔)</div>
<table><thead><tr><th>平仓时间</th><th>方向</th><th>开仓价</th><th>平仓价</th><th>净盈亏($)</th><th>原因</th></tr></thead><tbody>{rows}</tbody></table></div>
</body></html>"""
    (BASE_DIR / "index.html").write_text(html, encoding="utf-8")


def main():
    a = json.loads((BASE_DIR / "champion_config.json").read_text(encoding="utf-8"))["archetype"]
    state = json.loads((BASE_DIR / "paper_state.json").read_text(encoding="utf-8"))
    raw_df, source_name = fetch_live_4h(600)
    df = build_features(raw_df)
    bar, bar_ts, last_px = df.iloc[-2], df.index[-2].isoformat(), float(df["close"].iloc[-1])

    regime = int(bar["regime_1d"])
    bull = (regime == 1) and (bar["close"] > bar["ema_120"])
    bear = (regime == -1) and (bar["close"] < bar["ema_120"])
    l_sig = bull and bar["close"] > bar[f"donchian_h_{a['donchian_w']}"] and bar["vol_ratio"] >= 1.05 and bar["funding_z"] < 2.5
    if a["use_pullback"]:
        l_sig = l_sig or (bull and bar["ema_50"] > bar["ema_120"] and bar["rsi_14"] <= 38.0 and bar["close"] > bar["open"] and bar["taker_buy_ratio"] > 0.505)
    s_sig = a["allow_short"] and bear and bar["close"] < bar[f"donchian_l_{a['donchian_w']}"] and bar["vol_ratio"] >= 1.05 and bar["funding_z"] > -2.5
    sig = 1 if (l_sig and not s_sig) else (-1 if (s_sig and not l_sig) else 0)

    pos = state.get("position")
    if pos is not None:
        pos["bars_held"] = pos.get("bars_held", 0) + (1 if state.get("last_bar_time") != bar_ts else 0)
        si = 1 if pos["side"] == "LONG" else -1
        er, epx = None, None
        if si == 1:
            pos["best_px"] = max(pos["best_px"], float(bar["high"]))
            if pos["best_px"] - pos["entry_px"] >= 2.5 * pos["entry_atr"]:
                pos["sl_px"] = max(pos["sl_px"], pos["best_px"] - a["trail_dist_atr"] * pos["entry_atr"])
            if bar["low"] <= pos["sl_px"]:
                er, epx = "SL", pos["sl_px"]
            elif regime == -1:
                er, epx = "REGIME_EXIT", last_px
        else:
            pos["best_px"] = min(pos["best_px"], float(bar["low"]))
            if pos["entry_px"] - pos["best_px"] >= 2.0 * pos["entry_atr"]:
                pos["sl_px"] = min(pos["sl_px"], pos["best_px"] + (a["trail_dist_atr"] - 1.0) * pos["entry_atr"])
            tp_s = pos["entry_px"] - a["short_tp_atr"] * pos["entry_atr"]
            if bar["high"] >= pos["sl_px"]:
                er, epx = "SL", pos["sl_px"]
            elif bar["low"] <= tp_s:
                er, epx = "SHORT_TP", tp_s
            elif regime == 1:
                er, epx = "REGIME_EXIT", last_px
        if er is None and pos["bars_held"] >= 90 and si * (last_px - pos["entry_px"]) / pos["entry_px"] < 0.02:
            er, epx = "STAGNANT", last_px
        if er is not None:
            gross = si * (epx - pos["entry_px"]) * pos["qty"]
            fee = abs(pos["qty"] * epx) * ONE_WAY_COST
            net = gross - fee - pos["entry_fee"]
            state["equity"] += (gross - fee)
            state["peak_equity"] = max(state["peak_equity"], state["equity"])
            state["trades"].append({
                "entry_time": pos["entry_time"], "exit_time": datetime.now(timezone.utc).isoformat(),
                "side": pos["side"], "entry_px": pos["entry_px"], "exit_px": epx,
                "net_pnl": net, "exit_reason": er,
            })
            state["position"] = None

    tot_dd = (state["peak_equity"] - state["equity"]) / max(state["peak_equity"], 1.0)
    if state.get("position") is None and state.get("last_bar_time") != bar_ts and sig != 0:
        atr = float(bar["atr_14"])
        dd_s = 0.30 if tot_dd >= 0.13 else (0.60 if tot_dd >= 0.08 else 1.0)
        dir_s = 0.70 if sig == -1 else 1.0
        notional = min(
            (state["equity"] * a["risk_per_trade"] * dd_s * dir_s) / max((a["sl_atr"] * atr) / last_px, 0.005),
            state["equity"] * min(a["target_annual_vol"] / max(float(bar["rvol_30"]), 0.22), a["max_leverage"]) * dd_s * dir_s,
            state["equity"] * a["max_leverage"] * dd_s * dir_s,
        )
        qty, ef = notional / last_px, notional * ONE_WAY_COST
        state["equity"] -= ef
        state["position"] = {
            "side": "LONG" if sig == 1 else "SHORT", "entry_time": datetime.now(timezone.utc).isoformat(),
            "entry_px": last_px, "best_px": last_px, "qty": qty, "entry_atr": atr, "entry_fee": ef,
            "sl_px": last_px - sig * a["sl_atr"] * atr, "bars_held": 0,
        }

    state["last_bar_time"] = bar_ts
    state["equity_history"].append({"time": datetime.now(timezone.utc).isoformat(), "equity": state["equity"]})
    state["equity_history"] = state["equity_history"][-2000:]
    render_html(state, last_px, sig, regime, a, source_name)
    (BASE_DIR / "paper_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[BTC v4 Paper Bot] 完成 | BTC=${last_px:,.2f} | Equity=${state['equity']:,.2f} | Regime={regime} | Sig={sig}")


if __name__ == "__main__":
    main()
