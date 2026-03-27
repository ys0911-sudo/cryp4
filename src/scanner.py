"""
Real-time signal scanner — production deployment.

Usage:
    python -m src.scanner                         # WebSocket, top 50, threshold 0.65
    python -m src.scanner -t 0.70 --top 30        # higher threshold, fewer coins
    python -m src.scanner --coins BTC ETH SOL      # specific coins only
    python -m src.scanner --mode poll              # REST polling fallback

Architecture:
    CoinManager  — handles coin list, warmup, refresh, memory lifecycle
    HTFCache     — lazy higher-TF feature cache, refreshes only when needed
    SignalStore   — CSV with monthly rotation, exit tracking, running stats
    Scanner       — WebSocket/polling loop, ties everything together
"""

import sys
import json
import time
import asyncio
import argparse
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import requests

from src.features import build_features, FEATURE_COLUMNS
from src.coin_manager import CoinManager, CoinState, fetch_klines
from src.signal_store import SignalStore

# Force unbuffered output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path(__file__).parent.parent / "output"
MODEL_PATH = OUTPUT_DIR / "signal_model_15m.joblib"

BINANCE_WS = "wss://stream.binance.com:9443"

# ─── Optimized exit config (grid-searched on 15m data) ───
EXIT_CONFIG = {
    'initial_sl_pct': 0.007,
    'trailing_activate_pct': 0.006,
    'trailing_distance_pct': 0.004,
    'tp_pct': 0.02,
    'max_hold_minutes': 480,
}


def load_model():
    if not MODEL_PATH.exists():
        print(f"ERROR: No model at {MODEL_PATH}")
        sys.exit(1)
    bundle = joblib.load(MODEL_PATH)
    return bundle['model'], bundle['feature_cols'], bundle.get('threshold', 0.65)


def predict_signal(
    coin_state: CoinState,
    htf_cache,
    model,
    feature_cols: list[str],
    threshold: float,
) -> dict | None:
    """Check latest candle for a high-confidence signal."""
    df = coin_state.df
    latest = df.iloc[-1]

    if not latest.get('any_bull_signal', False):
        return None

    # Dedup: no repeat signals within 30 min
    if coin_state.last_signal_time:
        gap = (latest['open_time'] - coin_state.last_signal_time).total_seconds()
        if gap < 1800:
            return None

    # Merge higher TF context (lazy — only fetched if needed)
    htf_merge_cols = ['rsi', 'macd_hist', 'ema_50_slope', 'close_vs_ema50', 'ema_50_above_200']
    row_data = df.iloc[[-1]].copy()

    for interval, suffix in [('1h', '_1h'), ('4h', '_4h')]:
        htf_df = htf_cache.ensure_fresh(coin_state.symbol, interval)
        if htf_df is not None:
            available = [c for c in htf_merge_cols if c in htf_df.columns]
            if available:
                htf_sub = htf_df[['open_time'] + available].copy()
                htf_sub = htf_sub.rename(columns={c: f"{c}{suffix}" for c in available})
                htf_sub = htf_sub.sort_values('open_time')
                row_data = row_data.sort_values('open_time')
                row_data = pd.merge_asof(row_data, htf_sub, on='open_time', direction='backward')

    # Prepare features
    avail = [c for c in feature_cols if c in row_data.columns]
    missing = set(feature_cols) - set(avail)
    X = row_data[avail].copy()
    for c in avail:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    for c in missing:
        X[c] = 0
    X = X[feature_cols].fillna(0)

    proba = model.predict_proba(X)[:, 1][0]

    if proba >= threshold:
        entry_price = float(latest['close'])
        limit_entry = round((entry_price + float(latest['low'])) / 2, 6)
        coin_state.last_signal_time = latest['open_time']

        return {
            'symbol': coin_state.symbol,
            'signal_time': latest['open_time'].strftime('%Y-%m-%d %H:%M:%S UTC'),
            'entry_price': entry_price,
            'limit_entry': limit_entry,
            'confidence': round(float(proba), 4),
            'rsi': round(float(latest.get('rsi', 0)), 1),
            'macd_hist': round(float(latest.get('macd_hist', 0)), 6),
            'vol_spike_ratio': round(float(latest.get('vol_spike_ratio', 0)), 2),
            'signal_count': int(latest.get('bull_signal_count', 0)),
            'rsi_cross': bool(latest.get('rsi_cross_up', False)),
            'macd_cross': bool(latest.get('macd_cross_up', False)),
            'ema_cross': bool(latest.get('ema_9_21_cross_up', False)),
            'vol_spike': bool(latest.get('vol_spike', False)),
            'stop_loss': round(entry_price * (1 - EXIT_CONFIG['initial_sl_pct']), 6),
            'trailing_activate': round(entry_price * (1 + EXIT_CONFIG['trailing_activate_pct']), 6),
            'take_profit': round(entry_price * (1 + EXIT_CONFIG['tp_pct']), 6),
            'status': 'open',
            'peak_price': entry_price,
            'exit_price': '',
            'exit_time': '',
            'exit_reason': '',
            'pnl_pct': '',
        }

    return None


def parse_ws_candle(kline: dict) -> dict:
    """Convert WebSocket kline payload to DataFrame-compatible dict."""
    return {
        'open_time': pd.to_datetime(int(kline['t']), unit='ms', utc=True),
        'open': float(kline['o']),
        'high': float(kline['h']),
        'low': float(kline['l']),
        'close': float(kline['c']),
        'volume': float(kline['v']),
        'close_time': pd.to_datetime(int(kline['T']), unit='ms', utc=True),
        'quote_volume': float(kline['q']),
        'num_trades': int(kline['n']),
        'taker_buy_base_vol': float(kline['V']),
        'taker_buy_quote_vol': float(kline['Q']),
    }


def log_signal(signal: dict) -> None:
    """Pretty-print a signal to console."""
    now_str = datetime.now(timezone.utc).strftime('%H:%M:%S')
    print(f"\n  [{now_str}] SIGNAL: {signal['symbol']} @ {signal['entry_price']:.4f}")
    print(f"    Conf: {signal['confidence']:.2f} | "
          f"Signals: {signal['signal_count']} "
          f"(RSI={signal['rsi_cross']} MACD={signal['macd_cross']} "
          f"EMA={signal['ema_cross']} Vol={signal['vol_spike']})")
    print(f"    SL: {signal['stop_loss']:.4f} | "
          f"Trail@: {signal['trailing_activate']:.4f} | "
          f"TP: {signal['take_profit']:.4f} | "
          f"LimitEntry: {signal['limit_entry']:.4f}")


# ─────────────────────────────────────────────────
# WebSocket Mode
# ─────────────────────────────────────────────────

async def run_ws(coin_mgr: CoinManager, store: SignalStore,
                 model, feature_cols, threshold: float):
    """WebSocket event loop with dynamic coin refresh."""
    import websockets

    exit_interval = 60
    coin_refresh_interval = 3600
    last_exit_check = time.time()
    last_coin_refresh = time.time()
    last_status_print = time.time()

    while True:
        # Build stream URL from current coin list
        symbols = coin_mgr.get_active_symbols()
        streams = [f"{s.lower()}@kline_15m" for s in symbols]
        stream_path = "/".join(streams[:200])  # Binance limit: 200 per connection
        url = f"{BINANCE_WS}/stream?streams={stream_path}"

        reconnect = False

        try:
            async with websockets.connect(url, ping_interval=30, ping_timeout=10) as ws:
                print(f"  WebSocket connected ({len(symbols)} coins)")

                while not reconnect:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    except asyncio.TimeoutError:
                        # Periodic maintenance during quiet periods
                        now = time.time()
                        if now - last_exit_check >= exit_interval:
                            closed = store.check_exits()
                            for t in closed:
                                print(f"  EXIT {t['symbol']}: {t['reason']} | "
                                      f"PnL={t['pnl']:+.2f}%")
                            last_exit_check = now

                        if now - last_coin_refresh >= coin_refresh_interval:
                            added, removed = coin_mgr.refresh_coin_list()
                            if added or removed:
                                reconnect = True
                            last_coin_refresh = now

                        if now - last_status_print >= 900:
                            print(f"  [{datetime.now(timezone.utc).strftime('%H:%M')}] "
                                  f"{store.get_stats_summary()}")
                            last_status_print = now
                        continue

                    data = json.loads(msg)
                    if 'data' in data:
                        data = data['data']
                    if data.get('e') != 'kline':
                        continue

                    kline = data['k']
                    if not kline['x']:  # not closed yet
                        continue

                    # ─── Candle closed — process ───
                    sym_lower = kline['s'].lower()
                    state = coin_mgr.coin_states.get(sym_lower)
                    if state is None:
                        continue

                    state.append_candle(parse_ws_candle(kline))

                    signal = predict_signal(
                        state, coin_mgr.htf_cache, model, feature_cols, threshold
                    )
                    if signal:
                        is_new = store.save_signal(signal)
                        if is_new:
                            log_signal(signal)

                    # Periodic checks
                    now = time.time()
                    if now - last_exit_check >= exit_interval:
                        closed = store.check_exits()
                        for t in closed:
                            print(f"  EXIT {t['symbol']}: {t['reason']} | PnL={t['pnl']:+.2f}%")
                        last_exit_check = now

                    if now - last_coin_refresh >= coin_refresh_interval:
                        added, removed = coin_mgr.refresh_coin_list()
                        if added or removed:
                            reconnect = True
                        last_coin_refresh = now

        except (Exception,) as e:
            if not reconnect:
                print(f"  WebSocket error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)
            else:
                print(f"  Reconnecting for coin list update...")
                await asyncio.sleep(1)


# ─────────────────────────────────────────────────
# Polling Mode
# ─────────────────────────────────────────────────

def run_poll(coin_mgr: CoinManager, store: SignalStore,
             model, feature_cols, threshold: float, interval: int = 60):
    """REST polling loop."""
    last_candle_times: dict[str, pd.Timestamp] = {}

    # Initialize last candle times
    for key, state in coin_mgr.coin_states.items():
        last_candle_times[state.symbol] = state.df.iloc[-1]['open_time']

    coin_refresh_at = time.time() + 3600
    htf_refresh_at = time.time() + 900

    while True:
        try:
            t0 = time.time()
            signals_found = 0

            # Refresh coins
            if time.time() >= coin_refresh_at:
                added, removed = coin_mgr.refresh_coin_list()
                for sym in added:
                    if sym.lower() in coin_mgr.coin_states:
                        last_candle_times[sym] = coin_mgr.coin_states[sym.lower()].df.iloc[-1]['open_time']
                for sym in removed:
                    last_candle_times.pop(sym, None)
                coin_refresh_at = time.time() + 3600

            # Scan each coin
            for key, state in list(coin_mgr.coin_states.items()):
                sym = state.symbol
                try:
                    df_new = fetch_klines(sym, '15m', 5)
                    latest_time = df_new.iloc[-1]['open_time']

                    if latest_time <= last_candle_times.get(sym, pd.Timestamp.min.tz_localize('UTC')):
                        continue

                    last_candle_times[sym] = latest_time

                    existing_times = set(state.df['open_time'])
                    new_rows = df_new[~df_new['open_time'].isin(existing_times)]
                    if len(new_rows) > 0:
                        state.df = pd.concat([state.df, new_rows], ignore_index=True)
                        if len(state.df) > 250:
                            state.df = state.df.iloc[-250:].reset_index(drop=True)
                        state.df = build_features(state.df)

                    signal = predict_signal(state, coin_mgr.htf_cache, model, feature_cols, threshold)
                    if signal:
                        is_new = store.save_signal(signal)
                        if is_new:
                            signals_found += 1
                            log_signal(signal)
                except Exception:
                    continue
                time.sleep(0.1)

            # Check exits
            closed = store.check_exits()
            for t in closed:
                print(f"  EXIT {t['symbol']}: {t['reason']} | PnL={t['pnl']:+.2f}%")

            # Refresh HTF
            if time.time() >= htf_refresh_at:
                coin_mgr.htf_cache.refresh_all(coin_mgr.get_active_symbols())
                htf_refresh_at = time.time() + 900

            elapsed = time.time() - t0
            if signals_found or int(time.time()) % 900 < interval:
                print(f"  [{datetime.now(timezone.utc).strftime('%H:%M')}] "
                      f"{store.get_stats_summary()}")

            time.sleep(max(1, interval - elapsed))

        except KeyboardInterrupt:
            print("\nScanner stopped.")
            break


# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Crypto Signal Scanner')
    parser.add_argument('-t', '--threshold', type=float, default=0.65)
    parser.add_argument('--top', type=int, default=50)
    parser.add_argument('--coins', nargs='+', default=None)
    parser.add_argument('--mode', choices=['ws', 'poll'], default='ws')
    parser.add_argument('--poll-interval', type=int, default=60)
    args = parser.parse_args()

    model, feature_cols, _ = load_model()
    store = SignalStore(OUTPUT_DIR, EXIT_CONFIG)
    coin_mgr = CoinManager(top_n=args.top, refresh_interval=3600)

    # Determine coins
    if args.coins:
        symbols = [f"{c.upper()}USDT" if not c.upper().endswith('USDT') else c.upper()
                   for c in args.coins]
    else:
        symbols = None  # CoinManager will fetch from API

    print(f"{'='*60}")
    print(f"  CRYPTO SIGNAL SCANNER")
    print(f"  Mode: {args.mode.upper()} | Threshold: {args.threshold} | Top: {args.top}")
    print(f"  Exit: SL={EXIT_CONFIG['initial_sl_pct']:.1%} "
          f"Trail={EXIT_CONFIG['trailing_distance_pct']:.1%} "
          f"TP={EXIT_CONFIG['tp_pct']:.1%}")
    print(f"  CSV: {store.current_csv}")
    print(f"{'='*60}")

    coin_mgr.warmup(symbols)
    store.cleanup_old_csvs(keep_months=3)

    if args.mode == 'ws':
        try:
            asyncio.run(run_ws(coin_mgr, store, model, feature_cols, args.threshold))
        except KeyboardInterrupt:
            print("\nScanner stopped.")
    else:
        run_poll(coin_mgr, store, model, feature_cols, args.threshold, args.poll_interval)


if __name__ == '__main__':
    main()
