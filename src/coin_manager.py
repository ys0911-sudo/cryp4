"""
Dynamic coin list manager.

Fetches top liquid coins from Binance, refreshes periodically,
handles additions/removals cleanly.
"""

import time
import requests
import pandas as pd
from src.features import build_features

BINANCE_REST = "https://api.binance.com"

EXCLUDE_COINS = {
    # USD stablecoins
    'USDCUSDT', 'FDUSDUSDT', 'TUSDUSDT', 'BUSDUSDT', 'USDPUSDT',
    'USDTUSDT', 'USD1USDT', 'RLUSDUSDT', 'USDEUSDT', 'DAIUSDT',
    # Fiat-pegged
    'EURUSDT', 'AEURUSDT', 'GBPUSDT',
    # Gold / commodity tokens — tracks metals, not crypto sentiment
    'PAXGUSDT', 'XAUTUSDT', 'XAGUUSDT',
}

# Large-cap coins: kept in universe for HTF context but excluded from signals.
# Highly efficient markets — 1H ATR targets are too tight vs. spread + fees.
LARGE_CAP_EXCLUDE = {
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT',
    'ADAUSDT', 'DOGEUSDT', 'TRXUSDT', 'LTCUSDT', 'BCHUSDT',
    'LINKUSDT', 'DOTUSDT', 'XLMUSDT', 'AVAXUSDT', 'UNIUSDT',
}

MIN_QUOTE_VOLUME = 5_000_000  # $5M daily


def fetch_top_coins(top_n: int = 50) -> list[str]:
    """Fetch current top N liquid USDT coins from Binance."""
    try:
        resp = requests.get(f"{BINANCE_REST}/api/v3/ticker/24hr", timeout=15)
        resp.raise_for_status()
        pairs = []
        for d in resp.json():
            sym = d['symbol']
            if not sym.endswith('USDT') or sym in EXCLUDE_COINS:
                continue
            vol = float(d.get('quoteVolume', 0))
            if vol >= MIN_QUOTE_VOLUME:
                pairs.append((sym, vol))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in pairs[:top_n]]
    except Exception as e:
        print(f"  [!] Coin list fetch failed: {e}")
        return []


def fetch_klines(symbol: str, interval: str = '15m', limit: int = 250) -> pd.DataFrame:
    """Fetch klines from Binance REST API."""
    url = f"{BINANCE_REST}/api/v3/klines"
    resp = requests.get(url, params={
        'symbol': symbol, 'interval': interval, 'limit': limit
    }, timeout=15)
    resp.raise_for_status()

    df = pd.DataFrame(resp.json(), columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'num_trades',
        'taker_buy_base_vol', 'taker_buy_quote_vol', '_ignore'
    ]).drop(columns=['_ignore'])

    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                'taker_buy_base_vol', 'taker_buy_quote_vol']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['num_trades'] = pd.to_numeric(df['num_trades'], errors='coerce')
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
    return df


class CoinState:
    """Holds rolling candle buffer + computed features for one coin.

    Memory management:
    - Keeps max `max_rows` candles (default 250)
    - Older candles are dropped on every update
    - Features are recomputed in-place (no copy)
    """

    __slots__ = ['symbol', 'df', 'last_signal_time', 'max_rows']

    def __init__(self, symbol: str, df: pd.DataFrame, max_rows: int = 250):
        self.symbol = symbol
        self.max_rows = max_rows
        self.df = df.tail(max_rows).reset_index(drop=True)
        self.last_signal_time = None

    def append_candle(self, candle_data: dict) -> None:
        """Append one closed candle and trim + recompute."""
        new_row = pd.DataFrame([candle_data])
        self.df = pd.concat([self.df, new_row], ignore_index=True)

        # Trim to max_rows — this is the key memory control
        if len(self.df) > self.max_rows:
            self.df = self.df.iloc[-self.max_rows:].reset_index(drop=True)

        # Recompute features on the trimmed buffer
        self.df = build_features(self.df)

    def memory_kb(self) -> float:
        return self.df.memory_usage(deep=True).sum() / 1024


class LiveEMAState:
    """
    Rolling 1-minute candle EMA tracker for live entry-timing checks.

    Fed by the WebSocket 1m kline stream (zero REST I/O after warmup).
    Used in predict_signal to detect whether price has already run too far
    above EMA21 before committing to an entry — the root cause of late entries
    on high-positive-sentiment signals.
    """

    __slots__ = ['symbol', 'closes', 'ema9', 'ema21', 'max_candles']

    def __init__(self, symbol: str, max_candles: int = 100):
        self.symbol     = symbol
        self.closes: list[float] = []
        self.ema9:   float | None = None
        self.ema21:  float | None = None
        self.max_candles = max_candles

    def update(self, close: float) -> None:
        """Ingest one 1m close and update EMA9 / EMA21 incrementally."""
        self.closes.append(close)
        if len(self.closes) > self.max_candles:
            self.closes.pop(0)

        k9  = 2 / (9  + 1)
        k21 = 2 / (21 + 1)

        if self.ema9 is None:
            if len(self.closes) >= 9:
                self.ema9 = sum(self.closes[-9:]) / 9
        else:
            self.ema9 = close * k9 + self.ema9 * (1 - k9)

        if self.ema21 is None:
            if len(self.closes) >= 21:
                self.ema21 = sum(self.closes[-21:]) / 21
        else:
            self.ema21 = close * k21 + self.ema21 * (1 - k21)

    def entry_is_fresh(self, current_price: float, max_extension: float = 0.004) -> bool:
        """
        True if price is still close enough to EMA21 for a clean entry.

        Returns False (block) when price is more than `max_extension` above
        EMA21 — meaning momentum already played out and we'd be entering late.
        Returns True (allow) when EMA21 is not yet warmed up (< 21 candles).
        """
        if self.ema21 is None:
            return True   # not enough 1m data yet — don't block
        extension = (current_price - self.ema21) / self.ema21
        return extension <= max_extension


class HTFCache:
    """Higher timeframe feature cache with lazy refresh.

    Only refreshes a coin's HTF data when it's actually needed
    (i.e., when a signal fires on that coin), not on every cycle.

    Periodic full refresh happens every `full_refresh_interval` seconds.
    """

    def __init__(self, full_refresh_interval: int = 3600):
        self._cache: dict[str, pd.DataFrame] = {}  # key: "BTCUSDT_1h"
        self._last_update: dict[str, float] = {}
        self._stale_after: int = 3600  # stale after 1H (matches primary TF candle close)
        self.full_refresh_interval = full_refresh_interval
        self._last_full_refresh = 0.0

    def get(self, symbol: str, interval: str) -> pd.DataFrame | None:
        """Get cached HTF features. Returns None if not cached."""
        key = f"{symbol}_{interval}"
        return self._cache.get(key)

    def refresh_one(self, symbol: str, interval: str) -> pd.DataFrame | None:
        """Fetch and cache HTF data for one symbol+interval."""
        key = f"{symbol}_{interval}"
        try:
            df = fetch_klines(symbol, interval, limit=250)
            df = build_features(df)
            self._cache[key] = df
            self._last_update[key] = time.time()
            return df
        except Exception:
            return self._cache.get(key)  # return stale if fetch fails

    def ensure_fresh(self, symbol: str, interval: str) -> pd.DataFrame | None:
        """Get HTF data, refreshing if stale."""
        key = f"{symbol}_{interval}"
        last = self._last_update.get(key, 0)
        if time.time() - last > self._stale_after:
            return self.refresh_one(symbol, interval)
        return self._cache.get(key)

    def refresh_all(self, symbols: list[str]) -> None:
        """Full refresh of all symbols. Called periodically."""
        now = time.time()
        if now - self._last_full_refresh < self.full_refresh_interval:
            return

        for sym in symbols:
            for interval in ['4h', '1d']:
                self.refresh_one(sym, interval)
                time.sleep(0.1)  # rate limit

        self._last_full_refresh = now

    def drop(self, symbol: str) -> None:
        """Remove a coin from cache."""
        for interval in ['4h', '1d']:
            key = f"{symbol}_{interval}"
            self._cache.pop(key, None)
            self._last_update.pop(key, None)

    def memory_kb(self) -> float:
        total = 0
        for df in self._cache.values():
            total += df.memory_usage(deep=True).sum()
        return total / 1024


class CoinManager:
    """Manages the full lifecycle of monitored coins.

    - Startup: warmup top N coins
    - Periodic: refresh coin list, add new, remove dropped
    - Provides coin_states and htf_cache to scanner
    """

    def __init__(self, top_n: int = 50, refresh_interval: int = 3600):
        self.top_n = top_n
        self.refresh_interval = refresh_interval
        self.coin_states: dict[str, CoinState] = {}  # key: symbol lowercase
        self.live_ema_states: dict[str, LiveEMAState] = {}  # key: symbol lowercase
        self.htf_cache = HTFCache(full_refresh_interval=900)
        self._last_coin_refresh = 0.0
        self._active_symbols: set[str] = set()

    def warmup(self, symbols: list[str] | None = None) -> None:
        """Initial warmup: fetch history + features for all coins."""
        if symbols is None:
            symbols = fetch_top_coins(self.top_n)
            if not symbols:
                print("  [!] Could not fetch coin list, using fallback")
                symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT']

        print(f"  Warming up {len(symbols)} coins...")
        for i, sym in enumerate(symbols):
            self._add_coin(sym)
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(symbols)} done")
                time.sleep(0.3)

        self._active_symbols = set(sym.lower() for sym in symbols if sym.lower() in self.coin_states)
        print(f"  Ready: {len(self.coin_states)} coins, "
              f"~{self._total_memory_mb():.1f} MB memory")

    def _add_coin(self, symbol: str) -> bool:
        """Add one coin: fetch 1H history + HTF context."""
        key = symbol.lower()
        if key in self.coin_states:
            return True
        try:
            # 1H is now the primary TF — fetch 300 candles (~12.5 days)
            df = fetch_klines(symbol, '1h', 300)
            df = build_features(df)
            self.coin_states[key] = CoinState(symbol, df)

            # Warm HTF cache: 4H and 1D context for the 1H model
            for interval in ['4h', '1d']:
                self.htf_cache.refresh_one(symbol, interval)

            # Pre-warm 1m EMA state
            self._init_live_ema(symbol)

            return True
        except Exception as e:
            print(f"    [!] {symbol}: {e}")
            return False

    def _init_live_ema(self, symbol: str) -> None:
        """Pre-warm LiveEMAState with the last 50 1m candles via REST."""
        key = symbol.lower()
        state = LiveEMAState(symbol)
        try:
            df_1m = fetch_klines(symbol, '1m', 50)
            for close in df_1m['close']:
                state.update(float(close))
        except Exception as e:
            print(f"    [!] LiveEMA warmup failed for {symbol}: {e}")
        self.live_ema_states[key] = state

    def _remove_coin(self, symbol: str) -> None:
        """Remove coin and free its memory."""
        key = symbol.lower()
        if key in self.coin_states:
            del self.coin_states[key]
        self.live_ema_states.pop(key, None)
        self.htf_cache.drop(symbol)
        self._active_symbols.discard(key)

    def refresh_coin_list(self) -> tuple[list[str], list[str]]:
        """Check if coin list has changed. Returns (added, removed).

        Called periodically. Only fetches new data for newly added coins.
        """
        now = time.time()
        if now - self._last_coin_refresh < self.refresh_interval:
            return [], []

        self._last_coin_refresh = now
        new_list = fetch_top_coins(self.top_n)
        if not new_list:
            return [], []

        new_set = set(s.lower() for s in new_list)
        added = new_set - self._active_symbols
        removed = self._active_symbols - new_set

        for sym in added:
            sym_upper = sym.upper()
            if self._add_coin(sym_upper):
                print(f"  [+] Added {sym_upper}")

        for sym in removed:
            sym_upper = self.coin_states[sym].symbol if sym in self.coin_states else sym.upper()
            self._remove_coin(sym_upper)
            print(f"  [-] Removed {sym_upper}")

        self._active_symbols = new_set & set(self.coin_states.keys())

        if added or removed:
            print(f"  Monitoring {len(self._active_symbols)} coins, "
                  f"~{self._total_memory_mb():.1f} MB")

        return [s.upper() for s in added], [s.upper() for s in removed]

    def get_active_symbols(self) -> list[str]:
        """Return current active symbol list (uppercase)."""
        return [self.coin_states[k].symbol for k in sorted(self._active_symbols)
                if k in self.coin_states]

    def _total_memory_mb(self) -> float:
        coin_kb = sum(s.memory_kb() for s in self.coin_states.values())
        htf_kb = self.htf_cache.memory_kb()
        return (coin_kb + htf_kb) / 1024
