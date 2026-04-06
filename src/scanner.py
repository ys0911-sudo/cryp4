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

from src.features import build_features, compute_htf_features, FEATURE_COLUMNS, HTF_DERIVED_COLUMNS
from src.coin_manager import CoinManager, CoinState, fetch_klines, LARGE_CAP_EXCLUDE
from src.signal_store import SignalStore
from src.sentiment import SentimentScorer

# Force unbuffered output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path(__file__).parent.parent / "output"
MODEL_PATH = OUTPUT_DIR / "signal_model.joblib"   # 1H ATR-relative model

BINANCE_WS = "wss://stream.binance.com:9443"

# ─── Exit config for 1H ATR-relative model ───
# TP and SL are computed per-signal from ATR (stored in signal dict).
# These fallbacks are only used if ATR is unavailable.
# Trailing stop activates when price rises by 1× SL distance above entry.
EXIT_CONFIG = {
    'initial_sl_pct':       0.015,   # fallback: ~1.5% SL if ATR missing
    'trailing_activate_pct': 0.015,  # activate trail after 1× SL move up
    'trailing_distance_pct': 0.010,  # trail by 1% below peak
    'tp_pct':               0.030,   # fallback: ~3% TP if ATR missing
    'max_hold_minutes':     2880,    # 48 hours max hold on 1H entries
}


def load_model():
    """Load model bundle and verify SHA-256 integrity hash."""
    if not MODEL_PATH.exists():
        print(f"ERROR: No model at {MODEL_PATH}")
        sys.exit(1)

    # Integrity check
    hash_path = MODEL_PATH.with_name(MODEL_PATH.name + '.sha256')
    if hash_path.exists():
        import hashlib
        sha = hashlib.sha256()
        with open(MODEL_PATH, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                sha.update(chunk)
        actual = sha.hexdigest()
        expected = hash_path.read_text().strip()
        if actual != expected:
            print(f"ERROR: Model integrity check failed! Expected {expected[:12]}… got {actual[:12]}…")
            sys.exit(1)
        print(f"  Model integrity: OK ({actual[:12]}…)")

    bundle = joblib.load(MODEL_PATH)
    model       = bundle['model']
    feature_cols = bundle['feature_cols']
    threshold   = bundle.get('threshold', 0.65)
    atr_config  = bundle.get('config', {'tp_atr_mult': 2.0, 'sl_atr_mult': 1.0})
    print(f"  Model loaded: {len(feature_cols)} features | threshold={threshold} | "
          f"TP={atr_config['tp_atr_mult']}×ATR SL={atr_config['sl_atr_mult']}×ATR")
    return model, feature_cols, threshold, atr_config


def predict_signal(
    coin_state: CoinState,
    htf_cache,
    store,
    model,
    feature_cols: list[str],
    threshold: float,
    atr_config: dict,
    scorer: SentimentScorer | None = None,
    breadth_pct: float | None = None,
) -> dict | None:
    """Check latest candle for a high-confidence signal.

    No-re-entry: skips if coin already has an open trade in SignalStore.
    ATR-relative TP/SL: computed from current ATR × model multipliers.
    HTF context: 4H and 1D merged into 1H row before prediction.
    Sentiment: if scorer provided, applies adaptive threshold gating.
    """
    sym = coin_state.symbol

    # Skip large-cap coins — model has no edge on efficient markets
    if sym in LARGE_CAP_EXCLUDE:
        return None

    df = coin_state.df
    latest = df.iloc[-1]

    if not latest.get('any_bull_signal', False):
        return None

    # No re-entry: skip if this coin already has an open trade
    if store._open_positions.get(sym):
        return None

    # Dedup: no repeat signals within one 1H candle (3600s)
    if coin_state.last_signal_time:
        gap = (latest['open_time'] - coin_state.last_signal_time).total_seconds()
        if gap < 3600:
            return None

    # Merge 4H and 1D context (backward-looking, no look-ahead bias)
    htf_merge_cols = ['rsi', 'macd_hist', 'ema_50_slope', 'close_vs_ema50', 'ema_50_above_200']
    row_data = df.iloc[[-1]].copy()

    for interval, suffix in [('4h', '_4h'), ('1d', '_1d')]:
        htf_df = htf_cache.ensure_fresh(sym, interval)
        if htf_df is not None:
            available = [c for c in htf_merge_cols if c in htf_df.columns]
            if available:
                htf_sub = htf_df[['open_time'] + available].copy()
                htf_sub = htf_sub.rename(columns={c: f"{c}{suffix}" for c in available})
                htf_sub = htf_sub.sort_values('open_time')
                row_data = row_data.sort_values('open_time')
                row_data = pd.merge_asof(row_data, htf_sub, on='open_time', direction='backward')

    # Cross-TF derived features (htf_trend_aligned, rsi_htf_diff_4h, etc.)
    row_data = compute_htf_features(row_data)

    # Assemble feature vector — fill missing with 0 (graceful degradation)
    X = row_data.copy()
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
        elif X[c].dtype == bool:
            X[c] = X[c].astype(int)
    X = X[feature_cols].fillna(0)

    proba = model.predict_proba(X)[:, 1][0]

    # ── Sentiment-adjusted threshold ─────────────────────────────────
    # Evaluate sentiment BEFORE checking proba so we can gate even
    # high-confidence signals during crash regimes (score ≤ -5 → None).
    sentiment_snapshot: dict = {}
    effective_threshold = threshold

    if scorer is not None:
        adj_threshold, sentiment_snapshot = scorer.evaluate(
            breadth_pct=breadth_pct,
            base_threshold=threshold,
        )
        if adj_threshold is None:
            # Hard gate: market in crash regime — block all new longs
            return None
        # Use whichever is stricter: model default vs sentiment-adjusted
        effective_threshold = max(threshold, adj_threshold)

    if proba >= effective_threshold:
        entry_price = float(latest['close'])
        atr_val     = float(latest.get('atr', 0))

        # ATR-relative TP/SL — same logic as labeling used during training
        if atr_val > 0:
            tp = entry_price + atr_config['tp_atr_mult'] * atr_val
            sl = entry_price - atr_config['sl_atr_mult'] * atr_val
            # Trailing activates when price rises by 1× SL distance (break-even territory)
            trail_activate = entry_price + atr_config['sl_atr_mult'] * atr_val
            trail_distance = atr_val * 0.5   # trail 0.5× ATR below peak
        else:
            # Fallback to fixed percentages if ATR is zero/missing
            tp = entry_price * (1 + EXIT_CONFIG['tp_pct'])
            sl = entry_price * (1 - EXIT_CONFIG['initial_sl_pct'])
            trail_activate = entry_price * (1 + EXIT_CONFIG['trailing_activate_pct'])
            trail_distance = entry_price * EXIT_CONFIG['trailing_distance_pct']

        limit_entry = round((entry_price + float(latest['low'])) / 2, 8)
        coin_state.last_signal_time = latest['open_time']

        signal = {
            'symbol':           sym,
            'signal_time':      latest['open_time'].strftime('%Y-%m-%d %H:%M:%S UTC'),
            'entry_price':      round(entry_price, 8),
            'limit_entry':      round(limit_entry, 8),
            'confidence':       round(float(proba), 4),
            'atr':              round(atr_val, 8),
            'rsi':              round(float(latest.get('rsi', 0)), 1),
            'macd_hist':        round(float(latest.get('macd_hist', 0)), 8),
            'vol_spike_ratio':  round(float(latest.get('vol_spike_ratio', 0)), 2),
            'signal_count':     int(latest.get('bull_signal_count', 0)),
            'rsi_cross':        bool(latest.get('rsi_cross_up', False)),
            'macd_cross':       bool(latest.get('macd_cross_up', False)),
            'ema_cross':        bool(latest.get('ema_9_21_cross_up', False)),
            'vol_spike':        bool(latest.get('vol_spike', False)),
            # ATR-relative exit levels (absolute prices stored per-trade)
            'stop_loss':        round(sl, 8),
            'take_profit':      round(tp, 8),
            'trailing_activate': round(trail_activate, 8),
            'trailing_distance': round(trail_distance, 8),
            'status':           'open',
            'peak_price':       entry_price,
            'exit_price':       '',
            'exit_time':        '',
            'exit_reason':      '',
            'pnl_pct':          '',
        }

        # Attach sentiment snapshot for CSV logging (future logistic regression training)
        if sentiment_snapshot:
            signal.update(sentiment_snapshot)

        return signal

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
                 model, feature_cols, threshold: float, atr_config: dict,
                 scorer: SentimentScorer | None = None):
    """WebSocket event loop with dynamic coin refresh."""
    import websockets

    exit_interval = 60
    coin_refresh_interval = 3600
    sentiment_interval = 300   # print sentiment summary every 5 min
    last_exit_check     = time.time()
    last_coin_refresh   = time.time()
    last_status_print   = time.time()
    last_sentiment_print = time.time()
    _breadth_pct: float | None = None   # shared across candles, updated with ticker refresh

    while True:
        # Build stream URL from current coin list
        symbols = coin_mgr.get_active_symbols()
        streams = [f"{s.lower()}@kline_1h" for s in symbols]
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

                        if scorer and now - last_sentiment_print >= sentiment_interval:
                            _, snap = scorer.evaluate(breadth_pct=_breadth_pct)
                            scorer.print_status(snap)
                            last_sentiment_print = now

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
                    sym_upper = kline['s']

                    # ─── Live exit check on EVERY tick (open or closed candle) ───
                    # Fires SL/TP at the exact trigger price, not 60s later.
                    live_price = float(kline['c'])
                    live_exits = store.check_exits_live(sym_upper, live_price)
                    for t in live_exits:
                        print(f"  LIVE EXIT {t['symbol']}: {t['reason']} | "
                              f"Entry={t['entry']:.6f} | Exit={t['exit']:.6f} | "
                              f"PnL={t['pnl']:+.2f}%")

                    if not kline['x']:  # candle not closed — skip signal processing
                        continue

                    # ─── Candle closed — process signal ───
                    sym_lower = sym_upper.lower()
                    state = coin_mgr.coin_states.get(sym_lower)
                    if state is None:
                        continue

                    state.append_candle(parse_ws_candle(kline))

                    signal = predict_signal(
                        state, coin_mgr.htf_cache, store, model, feature_cols,
                        threshold, atr_config,
                        scorer=scorer, breadth_pct=_breadth_pct,
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

                    if scorer and now - last_sentiment_print >= sentiment_interval:
                        _, snap = scorer.evaluate(breadth_pct=_breadth_pct)
                        scorer.print_status(snap)
                        last_sentiment_print = now

        except (Exception,) as e:
            if not reconnect:
                import traceback, socket
                try:
                    import websockets.exceptions as _wse
                    _transient = (_wse.ConnectionClosedError, _wse.ConnectionClosedOK)
                except Exception:
                    _transient = ()

                is_transient = (
                    isinstance(e, _transient)           # Binance dropped connection
                    or isinstance(e, socket.gaierror)   # DNS blip
                    or isinstance(e, (TimeoutError, ConnectionResetError, OSError))
                )

                if is_transient:
                    # Known network hiccup — one-liner, no traceback noise
                    print(f"  WebSocket: {type(e).__name__} — reconnecting in 5s...")
                else:
                    # Unexpected error — full trace so we can debug
                    print(f"  WebSocket error: {e}. Reconnecting in 5s...")
                    traceback.print_exc()

                await asyncio.sleep(5)
            else:
                print(f"  Reconnecting for coin list update...")
                await asyncio.sleep(1)


# ─────────────────────────────────────────────────
# Polling Mode
# ─────────────────────────────────────────────────

def run_poll(coin_mgr: CoinManager, store: SignalStore,
             model, feature_cols, threshold: float, atr_config: dict,
             interval: int = 60, scorer: SentimentScorer | None = None):
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
                    df_new = fetch_klines(sym, '1h', 5)
                    latest_time = df_new.iloc[-1]['open_time']

                    if latest_time <= last_candle_times.get(sym, pd.Timestamp.min.tz_localize('UTC')):
                        continue

                    last_candle_times[sym] = latest_time

                    existing_times = set(state.df['open_time'])
                    new_rows = df_new[~df_new['open_time'].isin(existing_times)]
                    if len(new_rows) > 0:
                        state.df = pd.concat([state.df, new_rows], ignore_index=True)
                        if len(state.df) > 300:
                            state.df = state.df.iloc[-300:].reset_index(drop=True)
                        state.df = build_features(state.df)

                    signal = predict_signal(
                        state, coin_mgr.htf_cache, store, model, feature_cols,
                        threshold, atr_config, scorer=scorer,
                    )
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

    model, feature_cols, threshold, atr_config = load_model()
    # CLI threshold overrides model default if explicitly set
    if args.threshold != 0.65:
        threshold = args.threshold
    store = SignalStore(OUTPUT_DIR, EXIT_CONFIG)
    coin_mgr = CoinManager(top_n=args.top, refresh_interval=3600)

    # Determine coins
    if args.coins:
        symbols = [f"{c.upper()}USDT" if not c.upper().endswith('USDT') else c.upper()
                   for c in args.coins]
    else:
        symbols = None  # CoinManager will fetch from API

    # Initialise sentiment scorer
    scorer = SentimentScorer()

    # Warm up sentiment on startup (prints initial reading)
    try:
        _, snap = scorer.evaluate()
        scorer.print_status(snap)
    except Exception:
        pass

    print(f"{'='*60}")
    print(f"  CRYPTO SIGNAL SCANNER")
    print(f"  Mode: {args.mode.upper()} | Threshold: {threshold} | Top: {args.top}")
    print(f"  ATR TP={atr_config['tp_atr_mult']}\u00d7 SL={atr_config['sl_atr_mult']}\u00d7 "
          f"| Fallback SL={EXIT_CONFIG['initial_sl_pct']:.1%} TP={EXIT_CONFIG['tp_pct']:.1%}")
    print(f"  Sentiment: adaptive threshold enabled")
    print(f"  CSV: {store.current_csv}")
    print(f"{'='*60}")

    coin_mgr.warmup(symbols)

    if args.mode == 'ws':
        asyncio.run(run_ws(
            coin_mgr, store, model, feature_cols, threshold, atr_config,
            scorer=scorer,
        ))
    else:
        run_poll(
            coin_mgr, store, model, feature_cols, threshold, atr_config,
            interval=args.poll_interval, scorer=scorer,
        )


if __name__ == '__main__':
    main()