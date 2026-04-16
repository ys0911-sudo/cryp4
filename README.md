# Crypto Signal Scanner

A production-grade, ML-powered cryptocurrency trading signal system built on Binance. The scanner monitors up to 50 liquid spot pairs in real-time via WebSocket, runs a LightGBM classifier on each closed 1H candle, applies sentiment-based threshold gating, and pushes actionable alerts to Telegram with ATR-relative take-profit and stop-loss levels.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [How It Works — End-to-End](#how-it-works--end-to-end)
- [Live Scanner (`src/`)](#live-scanner-src)
  - [scanner.py — Main Orchestrator](#scannerpy--main-orchestrator)
  - [coin_manager.py — Dynamic Coin Lifecycle](#coin_managerpy--dynamic-coin-lifecycle)
  - [features.py — Feature Engineering](#featurespy--feature-engineering)
  - [signal_store.py — Signal Persistence & Exit Tracking](#signal_storepy--signal-persistence--exit-tracking)
  - [sentiment.py — Adaptive Threshold Engine](#sentimentpy--adaptive-threshold-engine)
  - [notifier.py — Telegram Alerts](#notifierpy--telegram-alerts)
- [Training Data Program](#training-data-program)
  - [download_binance_ohlcv.py — Historical Data Downloader](#download_binance_ohlcvpy--historical-data-downloader)
  - [src/features.py — Training Feature Pipeline](#srcfeaturespy--training-feature-pipeline)
  - [src/labeling.py — ATR-Relative Labeling](#srclabelingpy--atr-relative-labeling)
  - [src/train.py — LightGBM Training Pipeline](#srctrainpy--lightgbm-training-pipeline)
  - [src/backtest.py — Walk-Forward Backtest Engine](#srcbacktestpy--walk-forward-backtest-engine)
- [Signal Logic & Entry Gates](#signal-logic--entry-gates)
- [Exit Management](#exit-management)
- [Sentiment Gating System](#sentiment-gating-system)
- [Coin Universe & Exclusions](#coin-universe--exclusions)
- [Model Details](#model-details)
- [Installation & Setup](#installation--setup)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [Deployment on AWS](#deployment-on-aws)
- [Output Files](#output-files)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Known Limitations & Future Work](#known-limitations--future-work)

---

## Project Overview

This system is split into two independent programs:

**Live Scanner** (`src/`) — runs continuously on an AWS EC2 instance, consuming Binance WebSocket streams and producing trade signals in real-time.

**Training Data Program** (`Training Data Program/`) — a one-time (or periodic) offline pipeline that downloads historical OHLCV data, engineers features, labels outcomes using ATR-relative TP/SL simulation, trains a LightGBM classifier, and validates it with a walk-forward backtest before saving the model.

The two share an identical feature engineering specification so that the signal seen live is mathematically identical to the signal the model was trained on — eliminating the most common source of training-to-production skew.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING DATA PROGRAM                        │
│                                                                 │
│  download_binance_ohlcv.py                                      │
│    └─ Downloads 18 months of 15m/1h/4h/1d OHLCV               │
│       for all active USDT pairs                                 │
│                                                                 │
│  src/train.py                                                   │
│    ├─ data_loader.py  → loads + filters mature coins           │
│    ├─ features.py     → indicators, crossovers, HTF merge       │
│    ├─ labeling.py     → ATR-relative TP/SL forward labels      │
│    ├─ LightGBM 5-fold TimeSeriesSplit CV                        │
│    ├─ auto-threshold selection (max precision on holdout)       │
│    ├─ backtest.py     → walk-forward simulation                 │
│    └─ saves signal_model.joblib + .sha256 integrity hash        │
└─────────────────────────────────────────────────────────────────┘
                            │
                     model artifact
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LIVE SCANNER                               │
│                                                                 │
│  CoinManager.warmup()                                           │
│    └─ fetches top 50 liquid USDT coins                          │
│    └─ loads 300 × 1H candles per coin                           │
│    └─ warms HTFCache (4H + 1D per coin)                         │
│    └─ seeds LiveEMAState from last 50 × 1m candles              │
│                                                                 │
│  WebSocket — 50 coins × (1H + 1m) = 100 concurrent streams     │
│                                                                 │
│  On 1H candle close:                                            │
│    features.build_features()                                    │
│      → HTF merge (4H + 1D backward-looking merge_asof)          │
│      → compute_htf_features() (cross-TF derived)               │
│      → model.predict_proba()                                    │
│      → SentimentScorer.evaluate() (adaptive threshold)          │
│      → entry timing gates (SL cap, EMA freshness, cooldown)     │
│      → SignalStore.save_signal()                                │
│      → TelegramNotifier.send_signal()                           │
│                                                                 │
│  On 1m tick:                                                    │
│    → LiveEMAState.update() (EMA9/EMA21 rolling)                 │
│    → SignalStore.check_exits_live() (in-memory, zero I/O)       │
│                                                                 │
│  Every 60s:                                                     │
│    → SignalStore.check_exits() (REST batch price fetch)         │
│                                                                 │
│  Every 5 min:                                                   │
│    → SentimentScorer.evaluate() + print_status()               │
│                                                                 │
│  Every 60 min:                                                  │
│    → CoinManager.refresh_coin_list() (add/remove coins)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
crypto-scanner/
│
├── src/                          # Live scanner (production)
│   ├── __main__.py               # Entry point: python -m src.scanner
│   ├── scanner.py                # Main WebSocket/polling loop + signal logic
│   ├── coin_manager.py           # CoinState, LiveEMAState, HTFCache, CoinManager
│   ├── features.py               # Feature engineering (synced with training)
│   ├── signal_store.py           # CSV persistence, exit tracking, stats
│   ├── sentiment.py              # Adaptive sentiment threshold engine
│   └── notifier.py               # Telegram alert system (background queue)
│
├── output/                       # Runtime artifacts
│   ├── signal_model.joblib       # Trained 1H LightGBM model bundle
│   ├── signal_model.joblib.sha256# Integrity hash (checked on startup)
│   ├── signal_model_4h.joblib    # Optional 4H model
│   ├── signals_2026-04.csv       # Monthly signal log (auto-rotated)
│   └── signals_latest.csv        # Always points to current month's CSV
│
├── Training Data Program/
│   ├── download_binance_ohlcv.py # Historical data downloader
│   ├── requirements.txt          # Training dependencies
│   ├── data/
│   │   ├── 15m/  <sym>.csv       # Raw OHLCV (15-minute)
│   │   ├── 1h/   <sym>.csv       # Raw OHLCV (1-hour)
│   │   ├── 4h/   <sym>.csv       # Raw OHLCV (4-hour)
│   │   ├── 1d/   <sym>.csv       # Raw OHLCV (daily)
│   │   └── metadata.csv          # Per-coin quality report
│   ├── output/
│   │   ├── signal_model.joblib   # Trained model (copy to ../output/ to deploy)
│   │   ├── signal_model_4h.joblib
│   │   └── feature_importance.png
│   └── src/
│       ├── data_loader.py        # Load / filter / merge OHLCV from CSVs
│       ├── features.py           # Same feature spec as live scanner
│       ├── labeling.py           # ATR-relative TP/SL forward labeling
│       ├── train.py              # 1H model training pipeline
│       ├── train_4h.py           # 4H model training pipeline
│       ├── backtest.py           # Walk-forward backtest engine
│       ├── scanner.py            # Offline realtime scanner variant
│       └── eda.py                # Exploratory data analysis helpers
│
├── requirements.txt              # Live scanner dependencies
├── DEPLOY.md                     # Step-by-step AWS deployment guide
└── README.md                     # This file
```

---

## How It Works — End-to-End

### Phase 1: Data Collection (Offline)

Run once (or periodically to refresh data):

```bash
cd "Training Data Program"
python download_binance_ohlcv.py
```

This downloads 18 months of OHLCV history for every active USDT spot pair across four timeframes (15m, 1h, 4h, 1d) with 10 parallel threads. It also runs data quality validation and writes `data/metadata.csv` so the training pipeline can skip immature or low-quality coins.

### Phase 2: Model Training (Offline)

```bash
cd "Training Data Program"
python -m src.train
```

The pipeline:
1. Identifies the top 50 mature liquid coins (≥300 candles of 1H history)
2. Computes all technical indicators and crossover signals on 1H data
3. Merges 4H and 1D features backward-looking using `merge_asof` to prevent look-ahead bias
4. Computes cross-TF derived features (HTF trend alignment, RSI spreads, etc.)
5. Labels each row using ATR-relative forward simulation (TP = 2×ATR, SL = 1×ATR)
6. Trains a LightGBM classifier using 5-fold `TimeSeriesSplit` cross-validation
7. Runs a 4-month holdout evaluation and auto-selects the optimal confidence threshold
8. Runs a walk-forward backtest on the holdout set with realistic fees and no-re-entry
9. Saves the model bundle with SHA-256 integrity hash

### Phase 3: Live Scanning (Production)

```bash
python -m src.scanner
```

The scanner:
1. Verifies model integrity via SHA-256 hash
2. Warms up 50 coins (300 × 1H + 4H + 1D history + 50 × 1m for EMA seeding)
3. Opens a single Binance WebSocket with 100 streams (50 coins × 1H + 1m)
4. On every 1H candle close: runs the full prediction pipeline
5. On every 1m tick: updates live EMA state and checks exit conditions in memory
6. Pushes signals and exits to Telegram instantly

---

## Live Scanner (`src/`)

### scanner.py — Main Orchestrator

The central module. Handles the WebSocket event loop and orchestrates all other components.

**WebSocket mode** (default): Opens a single multiplexed Binance WebSocket subscribing to `{symbol}@kline_1h` and `{symbol}@kline_1m` for all active coins simultaneously. Binance allows up to 200 streams per connection; 50 coins × 2 intervals = 100 streams, leaving headroom.

**Polling mode** (fallback): Hits the Binance REST API (`/api/v3/klines`) for each coin every N seconds. More resilient to network issues but higher latency.

**Signal prediction flow** (`predict_signal`):

```
coin_state.df (last 250 × 1H candles with features)
  │
  ├─ any_bull_signal? (RSI/MACD/EMA crossover or vol spike)    ← early exit
  ├─ open position for this coin?                              ← no re-entry
  ├─ SL cooldown active?                                       ← early exit
  ├─ signal too soon after last?  (< 3600s)                   ← dedup
  │
  ├─ HTFCache.ensure_fresh(4H)                                 ← merge 4H context
  ├─ HTFCache.ensure_fresh(1D)                                 ← merge 1D context
  ├─ compute_htf_features()                                    ← cross-TF derived
  │
  ├─ model.predict_proba()                                     ← LightGBM inference
  │   └─ proba < threshold → return None
  │
  ├─ SentimentScorer.evaluate()                                ← adaptive threshold
  │   └─ hard_gate? → blocked signal (logged but not traded)
  │   └─ proba < adj_threshold? → blocked signal
  │
  ├─ SL cap gate (max_sl_pct = 5%)                            ← entry timing
  ├─ LiveEMA freshness gate (max extension = 0.4% above EMA21) ← entry timing
  │
  └─ emit signal dict with TP/SL/trail levels
```

**Exit configuration:**

| Parameter | Value | Description |
|---|---|---|
| `tp_atr_mult` | 2.0× | Take profit = entry + 2×ATR |
| `sl_atr_mult` | 1.0× | Stop loss = entry − 1×ATR |
| `max_sl_pct` | 5% | Cap on SL distance (prevents catastrophic losses on illiquid coins) |
| `live_ema_max_ext` | 0.4% | Skip signal if price >0.4% above 1m EMA21 (prevents late entries) |
| `sl_cooldown_hours` | 6 | Hours to block re-entry after a stop loss hit |
| `max_hold_minutes` | 2880 | Time exit after 48 hours |

---

### coin_manager.py — Dynamic Coin Lifecycle

**`fetch_top_coins(top_n)`** — Queries Binance `/api/v3/ticker/24hr`, filters for USDT pairs with ≥$5M daily quote volume, sorts by volume, returns top N symbols. Excludes stablecoins, fiat-pegged tokens, and commodity tokens (gold, silver).

**`CoinState`** — Per-coin rolling buffer. Stores the last 250 candles with computed features. On each new candle append, trims to max_rows and recomputes all features via `build_features()`. Reports memory usage in KB.

**`LiveEMAState`** — Maintains EMA9 and EMA21 on 1-minute closes using incremental Wilder smoothing (no array recomputation). Pre-seeded from the last 50 REST-fetched 1m candles at startup. After warmup, fed exclusively from the WebSocket 1m stream — zero REST I/O. Used by `predict_signal` to check whether price is still close enough to EMA21 for a clean entry. Blocked when `(price - EMA21) / EMA21 > 0.4%`.

**`HTFCache`** — Lazy cache for 4H and 1D feature data. Only fetches when a signal fires for a given coin (not on every cycle), refreshing if the cached data is older than 1 hour. Supports a full refresh sweep every 15 minutes. Tracks memory usage across all cached DataFrames.

**`CoinManager`** — Wraps all three of the above with a full lifecycle: warmup, add, remove, and hourly refresh. On refresh, adds newly top-ranked coins and removes dropped ones, triggering a WebSocket reconnect when the stream list changes.

---

### features.py — Feature Engineering

The feature pipeline is kept strictly identical between training and live execution. Any divergence would introduce silent prediction bias.

**`compute_indicators(df)`** — Single-TF indicator computation:

| Indicator | Parameters | Columns produced |
|---|---|---|
| RSI | 14-period | `rsi` |
| MACD | (12, 26, 9) | `macd_line`, `macd_signal`, `macd_hist` |
| EMA | 9, 21, 50, 200 | `ema_9`, `ema_21`, `ema_50`, `ema_200` |
| Volume SMA | 20-period | `vol_sma_20`, `vol_spike_ratio` |
| ATR | 14-period | `atr`, `atr_pct` |
| Bollinger Bands | (20, 2σ) | `bb_upper`, `bb_lower`, `bb_width` |

**`compute_crossover_signals(df)`** — Event detection columns (all boolean → int):

- `rsi_cross_up` — RSI crosses above 50
- `rsi_cross_30_up` — RSI crosses above 30 (oversold exit)
- `macd_cross_up` — MACD line crosses above signal line
- `ema_9_21_cross_up` — EMA9 crosses above EMA21
- `vol_spike` — volume > 2× 20-period average
- `bull_signal_count` — sum of the above (0–4)
- `any_bull_signal` — `bull_signal_count >= 1` (pre-filter before model inference)

**`compute_context_features(df)`** — Contextual and structural features:

- Trend: `ema_50_slope`, `ema_200_slope`, `close_vs_ema50`, `close_vs_ema200`, `ema_50_above_200`
- Momentum: `ret_1`, `ret_5`, `ret_10`, `ret_20`
- Candle shape: `body_pct`, `upper_wick`, `lower_wick`
- Order flow: `taker_buy_ratio` (key Binance-specific edge — taker buys as % of quote volume), `trades_spike`
- Bollinger position: `bb_position` (0 = at lower band, 1 = at upper band)
- Bullish divergence proxy: `bull_divergence` (price at 10-candle low but RSI >5 points above its own 10-candle low)
- Temporal: `hour_of_day`, `day_of_week`, `coin_age_days`, `week_high_proximity`

**`compute_htf_features(df)`** — Cross-TF derived features (called after HTF column merge):

- `htf_trend_aligned` — count of TFs (1H/4H/1D) where EMA50 > EMA200 (0–3)
- `rsi_htf_diff_4h` — 1H RSI minus 4H RSI (positive = 1H leading higher TF)
- `rsi_htf_diff_1d` — 1H RSI minus daily RSI
- `rsi_4h_overbought` — 4H RSI > 70 (risk of rejection on higher TF)
- `rsi_4h_oversold` — 4H RSI < 30
- `macd_hist_4h_pos` — 4H MACD histogram positive (trend confirmation)
- `daily_trend_rising` — daily EMA50 slope positive

**Total feature set: 47 features** (30 primary TF + 10 HTF suffix columns + 7 cross-TF derived).

---

### signal_store.py — Signal Persistence & Exit Tracking

Manages the full lifecycle of every signal from emission to exit.

**CSV rotation** — Signals are written to `output/signals_YYYY-MM.csv` (monthly). A symlink-equivalent `signals_latest.csv` is kept in sync. Old files are pruned after 3 months. The current month's file is always the source of truth.

**Deduplication** — Before writing, checks that no existing row matches `(symbol, signal_time)`. Prevents double-firing on scanner restart.

**Live exit checking** (`check_exits_live`) — Called on every WebSocket price tick (both 1H and 1m streams). Runs entirely in memory from `_open_positions` dict — no disk I/O unless an exit triggers. Checks four conditions in priority order:

1. **Stop loss** — `price ≤ pos['stop_loss']`
2. **Trailing stop** — `peak ≥ trail_activate` AND `price ≤ peak − trail_distance`
3. **Take profit** — `price ≥ pos['take_profit']`
4. **Time exit** — position held ≥ 48 hours

All exit prices use the exact ATR-computed trigger level (not the current price at poll time), preventing negative slippage in the log.

**REST fallback exit checker** (`check_exits`) — Runs every 60 seconds as a safety net. Batch-fetches all prices from Binance `/api/v3/ticker/price` and calls `check_exits_live` for each open position. Also re-syncs in-memory positions from CSV on scanner restart.

**SL cooldown** — Tracks `_last_stop_loss[symbol]` timestamps. After a stop loss, `is_in_cooldown()` returns `True` for 6 hours, blocking re-entry on that coin.

---

### sentiment.py — Adaptive Threshold Engine

A three-tier market regime filter that adjusts the minimum model confidence dynamically based on macro conditions.

**Data sources** (all public Binance endpoints — no API key required):

| Signal | Endpoint | Interpretation |
|---|---|---|
| BTC Perpetual Basis | `fapi.binance.com/fapi/v1/premiumIndex` | `(markPrice − indexPrice) / indexPrice`. Positive = contango (bullish); Negative = backwardation (bearish/panic) |
| Long/Short Ratio | `fapi.binance.com/futures/data/globalLongShortAccountRatio` | Fraction of accounts net-long. >0.72 = crowded; <0.45 = capitulation |
| OI Delta | `fapi.binance.com/fapi/v1/openInterest` | % change vs 5-min baseline. Sharp drop = liquidations (contrarian bullish) |
| Market Breadth | `api.binance.com/api/v3/ticker/24hr` | % of top USDT pairs up in 24h |
| BTC 4H RSI | `api.binance.com/api/v3/klines` | RSI-14 on 4H candles (computed in-house with Wilder smoothing) |

**Scoring logic** — Each factor contributes ±1 to ±3 to a composite integer score:

| Condition | Points |
|---|---|
| BTC 4H RSI < 35 (crash territory) | −3 |
| BTC 4H RSI < 45 (weak) | −1 |
| BTC 4H RSI > 60 (healthy momentum) | +1 |
| BTC 4H RSI > 70 (overbought) | −1 |
| Basis < −0.05% (strong backwardation) | −3 |
| Basis slightly negative | −1 |
| Basis > +0.05% (healthy contango) | +1 |
| L/S ratio > 0.72 (crowded longs) | −2 |
| L/S ratio < 0.45 (capitulation) | +1 |
| OI delta < −5% (liquidation cascade) | +1 |
| OI delta > +5% (leverage building) | −1 |
| Breadth < 30% (broad sell-off) | −3 |
| Breadth < 45% (weak market) | −1 |
| Breadth > 60% (broad rally) | +1 |

**Score → threshold mapping:**

| Score | Regime | Min Confidence |
|---|---|---|
| ≤ −5 | 🔴 Crash — block all longs | None (hard gate) |
| ≤ −2 | 🟠 Bearish — very selective | 0.90 |
| ≤ 0 | 🟡 Cautious | 0.82 |
| ≤ 2 | 🟢 Neutral | 0.75 |
| > 2 | 💚 Bullish | 0.70 |

Note: the model's raw training threshold (0.65) is deliberately not used as a floor in live trading. The minimum live threshold is always 0.70.

**Blocked signals** are still saved to CSV with `status='blocked'` and `block_reason='...'`. This preserves them as negative training examples for future logistic regression weight fitting once 300+ closed trades accumulate.

**Cache TTLs:** Basis and breadth refresh every 60s; BTC RSI and L/S ratio every 5 minutes.

---

### notifier.py — Telegram Alerts

Thread-safe Telegram notification system using a background daemon thread and a bounded queue (max 200 messages). The WebSocket event loop is never blocked waiting for Telegram's API.

**Setup** (one-time, 2 minutes):
1. Open Telegram → search @BotFather → `/newbot` → copy the token
2. Send `/start` to your new bot
3. Visit `https://api.telegram.org/bot<TOKEN>/getUpdates` → note `chat.id`
4. Set environment variables (see below)

**Alert types:**

| Type | Trigger | Emoji |
|---|---|---|
| SIGNAL | New tradeable signal approved by model + sentiment | 🟢 |
| BLOCKED | Model fired but sentiment gated it (optional, off by default) | 🔴 |
| EXIT | Trade closed (SL / TP / trailing / time) | 📤 |
| SENTIMENT | Periodic market regime summary (every 30 min) | 📊 |
| STARTUP | Scanner restarted | 🚀 |

**Signal alert format:**
```
🟢 SIGNAL: ONDOUSDT
━━━━━━━━━━━━━━━━
💰 Entry: 0.8420  |  Limit: 0.8390
📈 TP: 0.8672  (+2.99%)
🛡 SL: 0.8278  (-1.69%)
🔔 Trail activates @ 0.8562
━━━━━━━━━━━━━━━━
Conf: 0.73  |  RSI: 52.1  |  Indicators: RSI · MACD · Vol
Sentiment: score=+1  thr=0.75
⏰ 17 Apr 14:32 IST
```

Rate limit handling: on HTTP 429, backs off by `retry_after` seconds returned by Telegram. Up to 2 retries per message before silently dropping.

---

## Training Data Program

### download_binance_ohlcv.py — Historical Data Downloader

Downloads 18 months of OHLCV data for all active USDT spot pairs across four timeframes (15m, 1h, 4h, 1d).

**Key features:**
- Concurrent downloads with `ThreadPoolExecutor` (10 workers, configurable)
- Pagination: handles the Binance 1000-candle API limit transparently
- Rate limit handling: respects 429 and 418 (IP ban) responses with exponential backoff
- Atomic CSV writes (write to `.tmp` then `os.replace`) — no partial files on interruption
- Resume support: `SKIP_EXISTING=True` skips already-downloaded files
- Post-download data quality validation for each file:
  - NaN / negative price detection
  - Duplicate timestamp removal
  - Gap detection (flags if any gap > 3× interval duration)
  - Completeness check (flags if < 70% of expected candles present)
- Writes `data/metadata.csv` with per-coin quality report; training pipeline reads this to exclude immature coins (< 300 × 1H candles) and quality-flagged coins

---

### src/features.py — Training Feature Pipeline

Identical specification to the live `src/features.py`. Both files are kept in sync manually — any feature change must be applied to both.

---

### src/labeling.py — ATR-Relative Labeling

**`label_tp_sl_atr(df, tp_atr_mult=2.0, sl_atr_mult=1.0)`** — The primary labeling function used by the 1H model.

For each row, simulates walking forward candle by candle:
- `TP = entry_close + 2 × ATR` → label = 1 (win)
- `SL = entry_close − 1 × ATR` → label = 0 (loss)
- Neither within 200 candles → label = 0 (timeout)

Using ATR-relative levels instead of fixed percentages ensures that:
- Volatile coins (e.g. PEPE with 4% ATR on 1H) get proportionally wider stops
- Stable large-caps (BTC with 0.5% ATR) get tighter stops
- The R:R ratio (tp/sl = 2:1) remains constant across all coins

Per-row absolute TP and SL prices (`tp_price_atr`, `sl_price_atr`) are also stored for the backtest engine to use directly.

**`ATR_LABELING_CONFIGS`:**

| Config | TP mult | SL mult | Use case |
|---|---|---|---|
| `atr_2to1` | 2.0 | 1.0 | Default 1H model |
| `atr_3to1` | 3.0 | 1.0 | Higher R:R, fewer wins |
| `atr_scalp` | 1.5 | 1.0 | Scalp style |
| `atr_4h_swing` | 3.0 | 1.0 | 4H model |

---

### src/train.py — LightGBM Training Pipeline

**Configuration:**
- Primary timeframe: 1H
- Higher timeframes merged: 4H, 1D
- Top 50 mature liquid coins
- Labeling: ATR 2:1 R:R
- Holdout: 4 months
- Time-series CV: 5 folds (`TimeSeriesSplit`)

**Model:**
```python
LGBMClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    num_leaves=31, min_child_samples=50,
    subsample=0.8, colsample_bytree=0.8,
    class_weight='balanced', random_state=42,
)
```

`class_weight='balanced'` is critical — winning trades are a minority class (typically ~40–55% of signals that fire).

**Threshold selection** — Instead of a hardcoded threshold, the training script sweeps candidates `[0.40, 0.45, ..., 0.75]` on the holdout test set and picks the one that maximizes precision while keeping at least `max(10, 2% of test rows)` trades. This prevents the model from picking a threshold so high it never fires.

**Model bundle saved** (`signal_model.joblib`):
```python
{
    'model':        final_model,          # fitted LGBMClassifier
    'feature_cols': feature_cols,         # ordered list (must match live pipeline)
    'label_style':  'atr_2to1',
    'threshold':    best_threshold,       # auto-selected
    'config':       {'tp_atr_mult': 2.0, 'sl_atr_mult': 1.0},
}
```

A SHA-256 hash is written alongside the `.joblib` file. The scanner verifies this on every startup — if the hash mismatches (corrupt download, accidental overwrite), it exits immediately rather than running with a bad model.

---

### src/backtest.py — Walk-Forward Backtest Engine

Simulates live trading on the holdout test set.

**Key correctness fixes from original:**
- **Entry price = next candle open** (not signal candle close) — matches real execution
- **Full OHLCV walk-forward** — uses the complete consecutive candle series, not jumps between signal rows
- **No re-entry while in trade** — once a coin has an open trade, new signals for that coin are skipped until the trade closes
- **Compound PnL** — `(1+r1) × (1+r2) × ... − 1` instead of additive sum
- **Spot fee model** — 0.1% per side (0.2% round-trip), matching Binance spot taker rate

**Metrics reported:**
- Total trades, win rate, avg win %, avg loss %
- Expectancy per trade, profit factor
- Compound total PnL %
- Max consecutive losses, max drawdown %
- TP wins / SL losses / timeouts
- Average hold duration (in candles)

---

## Signal Logic & Entry Gates

A signal fires when all of the following conditions are satisfied in sequence:

1. **Bullish crossover pre-filter** — At least one of RSI/MACD/EMA crossover or volume spike occurred on the latest closed 1H candle. This dramatically reduces the number of candles passed to the model.

2. **No open position** — The coin has no currently open trade in `SignalStore`. One trade per coin at a time.

3. **SL cooldown cleared** — At least 6 hours have elapsed since the last stop loss on this coin.

4. **Candle dedup** — No signal for this coin within the same 1H period (3600s window).

5. **HTF context merge** — 4H and 1D features merged backward-looking via `merge_asof`.

6. **Model confidence ≥ base threshold** — LightGBM `predict_proba` ≥ threshold (default 0.65, typically auto-selected during training as 0.65–0.70).

7. **Sentiment gate** — The composite sentiment score must not trigger a hard block, and model confidence must exceed the sentiment-adjusted threshold (0.70–0.90 depending on regime).

8. **SL cap** — If ATR implies a stop loss wider than 5%, the SL is clamped to 5%. The TP is kept at the original ATR level, which actually improves the R:R ratio.

9. **Live EMA freshness** — The current price must be no more than 0.4% above the 1-minute EMA21. If price has already run significantly, the signal is skipped (late entry prevention).

---

## Exit Management

Each signal stores per-trade absolute exit prices computed at signal time:

| Level | Formula |
|---|---|
| `stop_loss` | `entry − 1 × ATR` (capped at −5% max) |
| `take_profit` | `entry + 2 × ATR` |
| `trailing_activate` | `entry + 1 × ATR` (break-even territory) |
| `trailing_distance` | `0.5 × ATR` (absolute price offset from peak) |

Exit conditions are evaluated on every price tick (live) and every 60 seconds (REST fallback):

1. **Stop loss hit** → exit at exact `stop_loss` price
2. **Trailing stop triggered** → price rose above `trailing_activate`, then fell `trailing_distance` below peak
3. **Take profit hit** → exit at exact `take_profit` price
4. **Time exit** → 48 hours elapsed → exit at live market price

---

## Sentiment Gating System

The sentiment module runs as a read-only overlay — it can only raise the confidence threshold or block signals entirely. It never fires a signal by itself.

Blocked signals are still logged to CSV with `status='blocked'` so they can be used as negative training examples for future supervised threshold calibration. The goal is to eventually replace the hand-tuned point system with logistic regression coefficients derived from 300+ closed trades.

Every signal row in the CSV includes a sentiment snapshot: `sent_btc_4h_rsi`, `sent_basis`, `sent_ls_ratio`, `sent_oi_delta`, `sent_breadth`, `sent_score`, `sent_threshold`.

---

## Coin Universe & Exclusions

**Excluded from universe entirely:**
- USD stablecoins: USDC, FDUSD, TUSD, BUSD, USDP, USDT, USD1, RLUS, USDE, DAI
- Fiat-pegged: EUR, AEUR, GBP
- Commodity tokens: PAXG (gold), XAUT (gold), XAGU (silver)

**Kept in universe but excluded from signals** (`LARGE_CAP_EXCLUDE`):
- BTC, ETH, BNB, XRP, SOL, ADA, DOGE, TRX, LTC, BCH, LINK, DOT, XLM, AVAX, UNI

These large-caps are used for HTF context (their market structure informs the multi-TF features) but are not traded — the 1H ATR-relative TP/SL targets are too tight relative to spread and fees on highly efficient markets.

**Minimum liquidity filter**: ≥$5M daily quote volume (applied in `CoinManager`).

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | LightGBM (gradient boosted trees) |
| Primary timeframe | 1H |
| HTF context | 4H + 1D (merged backward-looking) |
| Features | 47 total (30 primary + 10 HTF suffix + 7 cross-TF derived) |
| Label | ATR-relative TP/SL forward simulation (2:1 R:R) |
| Training window | ~14+ months (varies by coin history) |
| Holdout | 4 months |
| CV strategy | 5-fold TimeSeriesSplit |
| Threshold selection | Maximize precision on holdout (auto) |
| Class balancing | `class_weight='balanced'` |
| Integrity | SHA-256 hash verified on scanner startup |
| Training universe | Top 50 mature liquid USDT coins |

---

## Installation & Setup

### Live Scanner

```bash
# Clone or pull the repo
cd crypto-scanner

# Install dependencies
pip install -r requirements.txt
# or on Ubuntu system Python:
pip install -r requirements.txt --break-system-packages

# Ensure the model is present
ls output/signal_model.joblib
ls output/signal_model.joblib.sha256
```

### Training Data Program

```bash
cd "Training Data Program"

# Install training dependencies
pip install -r requirements.txt --break-system-packages

# Download 18 months of historical data (takes 20–60 min depending on connection)
python download_binance_ohlcv.py

# Train the 1H model
python -m src.train

# Train the 4H model (optional)
python -m src.train_4h

# Copy trained models to live scanner output directory
cp output/signal_model.joblib ../output/
cp output/signal_model.joblib.sha256 ../output/
```

---

## Environment Variables

The scanner reads secrets exclusively from environment variables. Never hardcode credentials.

| Variable | Required | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Optional | BotFather token (e.g. `7412356789:AAFxxx...`) |
| `TELEGRAM_CHAT_ID` | Optional | Your personal/group chat ID (e.g. `123456789`) |
| `TELEGRAM_SEND_BLOCKED` | Optional | `true` to also receive sentiment-blocked alerts (default: `false`) |

If `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are not set, the scanner runs normally but sends no Telegram notifications.

**Setting env vars on Linux/AWS:**

```bash
# Option 1: export in shell session
export TELEGRAM_BOT_TOKEN="7412356789:AAFxxx..."
export TELEGRAM_CHAT_ID="123456789"

# Option 2: add to /etc/environment (persists across reboots)
echo 'TELEGRAM_BOT_TOKEN=7412356789:AAFxxx...' | sudo tee -a /etc/environment
echo 'TELEGRAM_CHAT_ID=123456789' | sudo tee -a /etc/environment

# Option 3: set in systemd service file (see Deployment section)
```

---

## Usage

```bash
# Default: WebSocket mode, top 50 coins, threshold from model
python -m src.scanner

# Higher threshold, fewer coins
python -m src.scanner -t 0.70 --top 30

# Specific coins only
python -m src.scanner --coins BTC ETH SOL ONDO

# REST polling fallback (no WebSocket)
python -m src.scanner --mode poll

# Polling with custom interval (seconds)
python -m src.scanner --mode poll --poll-interval 120
```

**Startup output:**
```
============================================================
  CRYPTO SIGNAL SCANNER
  Mode: WS | Threshold: 0.65 | Top: 50
  ATR TP=2.0× SL=1.0× | Fallback SL=1.5% TP=3.0%
  Sentiment: adaptive threshold enabled
  CSV: /path/to/output/signals_2026-04.csv
============================================================
  [Telegram] Bot ready — chat_id=123456789 | blocked_alerts=off
  [Sentiment] score=+1 thr=0.75 | BTC4hRSI=58 basis=+0.021% L/S=0.52 OIΔ=+0.3% breadth=61%
  Model integrity: OK (a3f9b12c4e...)
  Model loaded: 47 features | threshold=0.65 | TP=2.0×ATR SL=1.0×ATR
  Warming up 50 coins...
    10/50 done
    ...
  Ready: 50 coins, ~42.3 MB memory
  WebSocket connected (50 coins, 1H + 1m streams)
```

**Live signal output:**
```
  [14:32:17 IST] 🟢 SIGNAL: ONDOUSDT @ 0.8420
    Conf: 0.73 | Signals: 3 (RSI=True MACD=True EMA=False Vol=True)
    SL: 0.8278 | Trail@: 0.8562 | TP: 0.8672 | LimitEntry: 0.8371
    Sentiment: score=+1 | thr=0.75

  LIVE EXIT ONDOUSDT: take_profit | Entry=0.842000 | Exit=0.867200 | PnL=+2.99%
```

---

## Deployment on AWS

See `DEPLOY.md` for the full step-by-step guide. Summary:

**Recommended: systemd service**

```ini
# /etc/systemd/system/crypto-scanner.service
[Unit]
Description=Crypto Signal Scanner
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/crypto-scanner
ExecStart=/usr/bin/python3 -m src.scanner
Restart=always
RestartSec=10
Environment="TELEGRAM_BOT_TOKEN=your_token_here"
Environment="TELEGRAM_CHAT_ID=your_chat_id_here"

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable crypto-scanner
sudo systemctl start crypto-scanner
sudo journalctl -u crypto-scanner -f   # tail live logs
```

**Alternative: tmux**

```bash
tmux new -s crypto
python -m src.scanner
# Ctrl+A D to detach
```

**Rollback:**

```bash
git log --oneline -5
git checkout <previous-hash>
sudo systemctl restart crypto-scanner
```

---

## Output Files

| File | Description |
|---|---|
| `output/signal_model.joblib` | Trained LightGBM model bundle |
| `output/signal_model.joblib.sha256` | SHA-256 integrity hash |
| `output/signals_YYYY-MM.csv` | Monthly signal log (auto-rotated) |
| `output/signals_latest.csv` | Alias for current month's CSV |
| `Training Data Program/data/metadata.csv` | Per-coin data quality report |
| `Training Data Program/output/feature_importance.png` | Feature importance bar chart |
| `Training Data Program/output/backtest_results.csv` | Per-coin holdout backtest metrics |

**Signal CSV columns:**

| Column | Description |
|---|---|
| `symbol` | e.g. `ONDOUSDT` |
| `signal_time` | IST timestamp of the 1H candle close |
| `entry_price` | Close of the signal candle |
| `limit_entry` | `(entry + low) / 2` — suggested limit order price |
| `confidence` | Model `predict_proba` score (0–1) |
| `atr` | ATR at signal time (absolute price) |
| `rsi` | RSI value at signal time |
| `macd_hist` | MACD histogram value |
| `vol_spike_ratio` | Volume / 20-period volume SMA |
| `signal_count` | Number of bullish crossovers that fired |
| `rsi_cross`, `macd_cross`, `ema_cross`, `vol_spike` | Which crossovers fired |
| `stop_loss` | Per-trade SL price (entry − 1×ATR, capped at −5%) |
| `take_profit` | Per-trade TP price (entry + 2×ATR) |
| `trailing_activate` | Price above which trailing stop activates |
| `trailing_distance` | Absolute trail offset from peak (0.5×ATR) |
| `status` | `open` / `closed` / `blocked` |
| `block_reason` | Populated when `status='blocked'` |
| `peak_price` | Highest price seen since entry |
| `exit_price` | Exact exit trigger price |
| `exit_time` | IST timestamp of exit |
| `exit_reason` | `stop_loss` / `take_profit` / `trailing_stop` / `time_exit` |
| `pnl_pct` | `(exit_price / entry_price − 1) × 100` |
| `sent_btc_4h_rsi` | BTC 4H RSI at signal time |
| `sent_basis` | BTC perpetual basis at signal time |
| `sent_ls_ratio` | L/S account ratio at signal time |
| `sent_oi_delta` | OI % change at signal time |
| `sent_breadth` | Market breadth % at signal time |
| `sent_score` | Composite sentiment score |
| `sent_threshold` | Adjusted threshold applied |

---

## Key Engineering Decisions

**Why 1H instead of 15m?**
The 1H timeframe closes 4× less frequently, generating fewer but higher-quality signals. The ATR-relative TP is also large enough to absorb spread and fees at a 2:1 R:R, which was not reliably the case on 15m with tight targets.

**Why ATR-relative TP/SL instead of fixed percentages?**
Fixed percentages produce inconsistent R:R across coins. A 1.5% stop on BTC (low ATR) is appropriate; the same 1.5% stop on PEPE (high ATR) will be hit by routine intraday noise. ATR-normalization keeps R:R constant regardless of coin volatility.

**Why exclude large-caps from signals?**
BTC, ETH and other tier-1 assets are priced by institutions and high-frequency traders globally. The 1H ATR on BTC is ~0.5–1%, making TP targets of 1–2% very tight after fees. Smaller alts with 2–5% ATR offer far more room. Large-caps are still used for HTF context (their EMA slopes and RSI inform the `htf_trend_aligned` feature).

**Why adaptive sentiment threshold?**
A model trained on historical data has no way of knowing whether the current market is in crash mode, ranging, or a bull run. The sentiment layer provides this regime context in real-time without retraining.

**Why no API keys for Binance?**
All data consumed by the live scanner uses public endpoints: WebSocket kline streams (no auth required), `/api/v3/ticker/24hr` (public), Futures premium index (public), L/S ratio (public), OI (public). This eliminates the most common operational failure mode: expired or rate-limited API keys.

**Why background thread for Telegram?**
Telegram's HTTP API can be slow (>1s response times under load). Blocking the WebSocket event loop for Telegram would cause missed candles. A bounded `queue.Queue` drains asynchronously via a daemon thread.

---

## Known Limitations & Future Work

**Current limitations:**
- Long-only — no short signals. The model is trained exclusively on bullish crossovers. A separate bearish model would need different labeling and feature selection.
- No position sizing — all signals are treated equally regardless of confidence or volatility. Kelly criterion or a fixed fractional model would improve capital efficiency.
- Single exchange — Binance spot only. Cross-exchange arbitrage and funding rate strategies are out of scope.
- No order execution — the scanner generates signals but does not place orders. Integration with a broker API (Binance, CCXT) is a natural next step.
- Sentiment model is hand-tuned — point weights are based on domain intuition, not fitted coefficients. Once 300+ closed trades are available, logistic regression on the `sent_*` snapshot columns will replace this.

**Future improvements:**
- Fit sentiment weights via logistic regression on accumulated signal CSV data
- Add 4H model as a second opinion layer (signal fires only when both 1H and 4H agree)
- Backfill blocked signals as negative labels for retraining
- Add short signals using bearish crossover pre-filter + inverted labeling
- Portfolio-level position sizing (Kelly or fixed fractional)
- Broker API integration for automated order placement
