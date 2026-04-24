"""
MON/USDC — RSI(40) × WMA(15) Crossover Strategy
Timeframe : 1-minute
Lookback  : 9 months
SL        : Low of previous candle
TP        : 2 × SL distance
Report    : Telegram message
"""

# ──────────────────────────────────────────────
# DEPENDENCIES — all installed automatically
# ──────────────────────────────────────────────
import subprocess, sys

REQUIRED = ["ccxt", "pandas", "numpy", "requests", "python-dotenv"]

for pkg in REQUIRED:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"[INSTALL] {pkg} …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

# ──────────────────────────────────────────────
import ccxt
import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()  # optional: load TELEGRAM_TOKEN and TELEGRAM_CHAT_ID from .env

# ══════════════════════════════════════════════
# ⚙️  CONFIGURATION  — edit these
# ══════════════════════════════════════════════
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN",   "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

SYMBOL        = "MON/USDC"
TIMEFRAME     = "1m"
RSI_LENGTH    = 40
WMA_LENGTH    = 15
TP_MULTIPLIER = 2.0          # TP = SL distance × 2
MONTHS_BACK   = 9            # script requests 9m; auto-caps to available data
INITIAL_CAP   = 1000.0       # USDC starting capital
RISK_PER_TRADE = 0.01        # 1 % risk per trade

# MON launched Oct 8 2025 — OKX lists MON/USDC spot.
# Fallback chain: okx (MON/USDC) → bybit (MON/USDT) → gate (MON/USDT)
EXCHANGE_ID   = "okx"

# ══════════════════════════════════════════════
# 1. FETCH DATA
# ══════════════════════════════════════════════

def _try_fetch(exchange_id: str, symbol: str, timeframe: str, since_ms: int) -> list:
    """Try fetching all bars from one exchange, paginating as needed."""
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    exchange.load_markets()

    # Normalise symbol — try as-is, then without '/'
    if symbol not in exchange.markets:
        alt = symbol.replace("/", "")
        if alt in exchange.markets:
            symbol = alt
        else:
            raise ccxt.BadSymbol(f"{exchange_id}: symbol {symbol} not found")

    limit    = 1000
    all_bars = []
    cur_since = since_ms

    while True:
        bars = exchange.fetch_ohlcv(symbol, timeframe, since=cur_since, limit=limit)
        if not bars:
            break
        all_bars.extend(bars)
        cur_since = bars[-1][0] + 1
        print(f"  [{exchange_id}] …{len(all_bars):,} bars", end="\r")
        if len(bars) < limit:
            break
        time.sleep(exchange.rateLimit / 1000)

    return all_bars


def fetch_ohlcv(symbol: str, timeframe: str, months: int) -> pd.DataFrame:
    # MON launched Oct 8 2025 — cap lookback to what actually exists
    since_dt  = datetime.now(timezone.utc) - timedelta(days=months * 30)
    since_ms  = int(since_dt.timestamp() * 1000)

    # Fallback chain: preferred exchange first, then alternatives
    candidates = [
        (EXCHANGE_ID, symbol),          # okx  + MON/USDC
        ("bybit",     "MON/USDT"),      # bybit + MON/USDT (deepest liquidity)
        ("gate",      "MON/USDT"),      # gate  + MON/USDT
        ("kucoin",    "MON/USDT"),      # kucoin fallback
    ]

    all_bars = []
    used_pair = symbol

    for ex_id, sym in candidates:
        try:
            print(f"[FETCH] Trying {ex_id} | {sym} | {timeframe} | last {months} months …")
            all_bars = _try_fetch(ex_id, sym, timeframe, since_ms)
            used_pair = sym
            print(f"\n[OK] Fetched from {ex_id} ({sym})")
            break
        except Exception as e:
            print(f"\n[WARN] {ex_id}/{sym} failed: {e}")
            continue

    if not all_bars:
        raise ValueError(
            "Could not fetch MON data from any exchange.\n"
            "Please check your internet connection or try again later."
        )

    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    print(f"[INFO] Symbol traded: {used_pair} | Bars: {len(df):,}")
    return df

# ══════════════════════════════════════════════
# 2. INDICATORS
# ══════════════════════════════════════════════

def calc_rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=length - 1, min_periods=length).mean()
    avg_loss = loss.ewm(com=length - 1, min_periods=length).mean()
    rs   = avg_gain / avg_loss.replace(0, np.nan)
    rsi  = 100 - (100 / (1 + rs))
    return rsi

def calc_wma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1, dtype=float)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"]     = calc_rsi(df["close"], RSI_LENGTH)
    df["rsi_wma"] = calc_wma(df["rsi"],   WMA_LENGTH)
    # Crossover flags
    df["cross_above"] = (df["rsi"] > df["rsi_wma"]) & (df["rsi"].shift(1) <= df["rsi_wma"].shift(1))
    df["cross_below"] = (df["rsi"] < df["rsi_wma"]) & (df["rsi"].shift(1) >= df["rsi_wma"].shift(1))
    return df.dropna()

# ══════════════════════════════════════════════
# 3. BACKTEST ENGINE
# ══════════════════════════════════════════════

def run_backtest(df: pd.DataFrame) -> dict:
    trades      = []
    equity      = INITIAL_CAP
    in_trade    = False
    entry_price = sl = tp = qty = direction = None

    for i in range(1, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]

        # ── manage open trade ──────────────────
        if in_trade:
            hit_sl = hit_tp = False
            pnl = 0.0

            if direction == "long":
                if row["low"] <= sl:
                    hit_sl = True
                    exit_p = sl
                elif row["high"] >= tp:
                    hit_tp = True
                    exit_p = tp
            else:  # short
                if row["high"] >= sl:
                    hit_sl = True
                    exit_p = sl
                elif row["low"] <= tp:
                    hit_tp = True
                    exit_p = tp

            if hit_sl or hit_tp:
                if direction == "long":
                    pnl = (exit_p - entry_price) * qty
                else:
                    pnl = (entry_price - exit_p) * qty

                equity += pnl
                trades.append({
                    "entry_time":  df.index[i - 1],
                    "exit_time":   df.index[i],
                    "direction":   direction,
                    "entry":       entry_price,
                    "exit":        exit_p,
                    "sl":          sl,
                    "tp":          tp,
                    "qty":         qty,
                    "pnl":         round(pnl, 4),
                    "result":      "WIN" if hit_tp else "LOSS",
                    "equity":      round(equity, 4),
                })
                in_trade = False

        # ── look for new entry ─────────────────
        if not in_trade:
            if row["cross_above"]:            # LONG signal
                sl_price = prev["low"]
                sl_dist  = row["close"] - sl_price
                if sl_dist <= 0:
                    continue
                tp_price = row["close"] + sl_dist * TP_MULTIPLIER
                risk_usd = equity * RISK_PER_TRADE
                qty_calc = risk_usd / sl_dist
                if qty_calc * row["close"] > equity:
                    qty_calc = equity / row["close"]
                direction   = "long"
                entry_price = row["close"]
                sl          = sl_price
                tp          = tp_price
                qty         = qty_calc
                in_trade    = True

            elif row["cross_below"]:          # SHORT signal
                sl_price = prev["high"]
                sl_dist  = sl_price - row["close"]
                if sl_dist <= 0:
                    continue
                tp_price = row["close"] - sl_dist * TP_MULTIPLIER
                risk_usd = equity * RISK_PER_TRADE
                qty_calc = risk_usd / sl_dist
                if qty_calc * row["close"] > equity:
                    qty_calc = equity / row["close"]
                direction   = "short"
                entry_price = row["close"]
                sl          = sl_price
                tp          = tp_price
                qty         = qty_calc
                in_trade    = True

    trades_df = pd.DataFrame(trades)
    return trades_df

# ══════════════════════════════════════════════
# 4. STATISTICS
# ══════════════════════════════════════════════

def compute_stats(trades: pd.DataFrame, df: pd.DataFrame) -> dict:
    if trades.empty:
        return {"error": "No trades generated in the backtest period."}

    total        = len(trades)
    wins         = (trades["result"] == "WIN").sum()
    losses       = (trades["result"] == "LOSS").sum()
    win_rate     = wins / total * 100

    gross_profit = trades[trades["pnl"] > 0]["pnl"].sum()
    gross_loss   = trades[trades["pnl"] < 0]["pnl"].sum()
    net_pnl      = trades["pnl"].sum()

    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float("inf")
    avg_win       = trades[trades["pnl"] > 0]["pnl"].mean() if wins else 0
    avg_loss_val  = trades[trades["pnl"] < 0]["pnl"].mean() if losses else 0
    rr_ratio      = abs(avg_win / avg_loss_val) if avg_loss_val != 0 else 0

    # Max drawdown
    equity_curve  = trades["equity"].values
    peak          = np.maximum.accumulate(equity_curve)
    drawdown      = (equity_curve - peak) / peak * 100
    max_dd        = drawdown.min()

    final_equity  = trades["equity"].iloc[-1]
    total_return  = (final_equity - INITIAL_CAP) / INITIAL_CAP * 100

    # Consecutive stats
    results       = trades["result"].tolist()
    max_consec_w  = max_consec_l = cur = 0
    cur_w = cur_l = 0
    for r in results:
        if r == "WIN":
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_consec_w = max(max_consec_w, cur_w)
        max_consec_l = max(max_consec_l, cur_l)

    date_from = df.index[0].strftime("%d %b %Y")
    date_to   = df.index[-1].strftime("%d %b %Y")

    return {
        "date_from":       date_from,
        "date_to":         date_to,
        "total_trades":    total,
        "wins":            int(wins),
        "losses":          int(losses),
        "win_rate":        round(win_rate, 2),
        "gross_profit":    round(gross_profit, 2),
        "gross_loss":      round(gross_loss, 2),
        "net_pnl":         round(net_pnl, 2),
        "profit_factor":   round(profit_factor, 3),
        "avg_win":         round(avg_win, 4),
        "avg_loss":        round(avg_loss_val, 4),
        "rr_ratio":        round(rr_ratio, 2),
        "max_drawdown_pct":round(max_dd, 2),
        "initial_capital": INITIAL_CAP,
        "final_equity":    round(final_equity, 2),
        "total_return_pct":round(total_return, 2),
        "max_consec_wins": max_consec_w,
        "max_consec_loss": max_consec_l,
    }

# ══════════════════════════════════════════════
# 5. TELEGRAM REPORT
# ══════════════════════════════════════════════

def send_telegram(stats: dict):
    if TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("[SKIP] Telegram not configured — printing report to console instead.")
        return

    pnl_emoji   = "🟢" if stats["net_pnl"] >= 0 else "🔴"
    wr_emoji    = "✅" if stats["win_rate"] >= 50 else "⚠️"
    ret_emoji   = "📈" if stats["total_return_pct"] >= 0 else "📉"

    msg = f"""
╔══════════════════════════════╗
║  📊 MON/USDC BACKTEST REPORT  ║
╚══════════════════════════════╝

🕐 *Timeframe* : 1-Minute
📅 *Period*    : {stats['date_from']} → {stats['date_to']} _(9m window; MON launched Oct 8 2025)_
🔢 *Strategy*  : RSI(40) × WMA(15) Crossover
📌 *SL*        : Previous candle low/high
🎯 *TP*        : 2× SL distance

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 *TRADE SUMMARY*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔢 Total Trades     : `{stats['total_trades']}`
✅ Wins             : `{stats['wins']}`
❌ Losses           : `{stats['losses']}`
{wr_emoji} Win Rate         : `{stats['win_rate']}%`
🔁 Max Consec Wins  : `{stats['max_consec_wins']}`
🔁 Max Consec Loss  : `{stats['max_consec_loss']}`

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 *PnL BREAKDOWN*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{pnl_emoji} Net PnL          : `{stats['net_pnl']:+.2f} USDC`
🟢 Gross Profit     : `{stats['gross_profit']:.2f} USDC`
🔴 Gross Loss       : `{stats['gross_loss']:.2f} USDC`
⚖️ Profit Factor    : `{stats['profit_factor']}`
📊 Avg Win          : `{stats['avg_win']:+.4f} USDC`
📊 Avg Loss         : `{stats['avg_loss']:+.4f} USDC`
🎯 Avg R:R Ratio    : `{stats['rr_ratio']}x`

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 *EQUITY CURVE*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💵 Starting Capital : `{stats['initial_capital']:.2f} USDC`
💵 Final Equity     : `{stats['final_equity']:.2f} USDC`
{ret_emoji} Total Return     : `{stats['total_return_pct']:+.2f}%`
📉 Max Drawdown     : `{stats['max_drawdown_pct']:.2f}%`

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚙️ *CONFIG*  Risk/Trade: 1% | TP: 2×SL

_Generated: {datetime.now().strftime('%d %b %Y %H:%M:%S')} UTC_
    """.strip()

    url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       msg,
        "parse_mode": "Markdown",
    }
    try:
        r = requests.post(url, data=data, timeout=10)
        if r.status_code == 200:
            print("[✓] Telegram report sent.")
        else:
            print(f"[!] Telegram error {r.status_code}: {r.text}")
    except Exception as e:
        print(f"[!] Telegram failed: {e}")

# ══════════════════════════════════════════════
# 6. CONSOLE PRINT
# ══════════════════════════════════════════════

def print_report(stats: dict):
    if "error" in stats:
        print(f"\n[ERROR] {stats['error']}")
        return

    print(f"""
╔══════════════════════════════════════════════════════════╗
║            MON/USDC  |  RSI(40) × WMA(15) Crossover      ║
║              1-Minute Backtest  —  9 Month Window         ║
╚══════════════════════════════════════════════════════════╝
  Period         : {stats['date_from']}  →  {stats['date_to']}
  SL             : Previous candle Low/High
  TP             : 2 × SL distance | Risk/trade: 1%
──────────────────────────────────────────────────────────
  Total Trades   : {stats['total_trades']}
  Wins           : {stats['wins']}
  Losses         : {stats['losses']}
  Win Rate       : {stats['win_rate']}%
  Max Consec W   : {stats['max_consec_wins']}
  Max Consec L   : {stats['max_consec_loss']}
──────────────────────────────────────────────────────────
  Net PnL        : {stats['net_pnl']:+.2f} USDC
  Gross Profit   : {stats['gross_profit']:.2f} USDC
  Gross Loss     : {stats['gross_loss']:.2f} USDC
  Profit Factor  : {stats['profit_factor']}
  Avg Win        : {stats['avg_win']:+.4f} USDC
  Avg Loss       : {stats['avg_loss']:+.4f} USDC
  Avg R:R        : {stats['rr_ratio']}x
──────────────────────────────────────────────────────────
  Start Capital  : {stats['initial_capital']:.2f} USDC
  Final Equity   : {stats['final_equity']:.2f} USDC
  Total Return   : {stats['total_return_pct']:+.2f}%
  Max Drawdown   : {stats['max_drawdown_pct']:.2f}%
══════════════════════════════════════════════════════════
""")

# ══════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════

def main():
    # ── Fetch ──────────────────────────────────
    df = fetch_ohlcv(SYMBOL, TIMEFRAME, MONTHS_BACK)
    print(f"[OK] {len(df):,} bars loaded  ({df.index[0]} → {df.index[-1]})")

    # ── Indicators ─────────────────────────────
    print("[CALC] Computing RSI(40) and WMA(15) …")
    df = add_indicators(df)
    print(f"[OK] Indicators ready. {len(df):,} bars after warmup drop.")

    # ── Backtest ────────────────────────────────
    print("[RUN] Running backtest …")
    trades = run_backtest(df)
    print(f"[OK] {len(trades)} trades simulated.")

    # ── Stats ───────────────────────────────────
    stats = compute_stats(trades, df)

    # ── Output ──────────────────────────────────
    print_report(stats)

    # ── Save trades CSV ─────────────────────────
    if not trades.empty:
        out_path = "mon_usdc_trades.csv"
        trades.to_csv(out_path, index=False)
        print(f"[SAVED] Trade log → {out_path}")

    # ── Telegram ────────────────────────────────
    if "error" not in stats:
        send_telegram(stats)

if __name__ == "__main__":
    main()
