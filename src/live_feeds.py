"""
Live market microstructure feeds — CVD and Liquidation trackers.

Architecture
────────────
Two asyncio tasks run alongside the main WebSocket loop in run_ws():

  CVDTracker        → wss://stream.binance.com:9443/ws/btcusdt@aggTrade
                      Cumulative Volume Delta: signed sum of taker buy/sell
                      volume over rolling 15m and 1H windows.
                      Normalised ratio (CVD / total_volume) gives a [-1, +1]
                      scale comparable across different market volumes.

  LiquidationTracker → wss://fstream.binance.com/ws/!forceOrder@arr
                       Forced liquidation USD across ALL futures pairs.
                       Spike in 5m window = capitulation → contrarian bullish.
                       Separates long liquidations vs short liquidations.

Design principles
─────────────────
  • Fault-tolerant: reconnects silently on any WebSocket error.
  • Memory-bounded: entries older than 1H are pruned on every new event.
  • Caller-safe: snapshot() always returns a dict even during warmup/outage
    (all values 0.0), so the scorer never raises on missing data.
  • No API key required for either stream.

Logged columns (attached to every signal row for future model training)
───────────────────────────────────────────────────────────────────────
  cvd_15m          raw CVD in BTC over last 15 minutes (signed)
  cvd_1h           raw CVD in BTC over last 1 hour (signed)
  cvd_ratio_15m    cvd_15m / total_volume_15m  (−1 to +1)
  cvd_ratio_1h     cvd_1h  / total_volume_1h   (−1 to +1)
  cvd_slope        cvd_15m / 15 (BTC per minute, rate of change)
  liq_5m_usd       total liquidation USD across all pairs, last 5 minutes
  liq_5m_long_usd  long-position liquidations only, last 5 minutes
  liq_5m_short_usd short-position liquidations only, last 5 minutes
  liq_1h_usd       total liquidation USD, last 1 hour
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque

BINANCE_WS_SPOT    = "wss://stream.binance.com:9443/ws"
BINANCE_WS_FUTURES = "wss://fstream.binance.com/ws"

_RECONNECT_DELAY = 5     # seconds between reconnect attempts
_MAX_WINDOW      = 3600  # 1H — maximum data retention in deque


# ─────────────────────────────────────────────────────────────────
# CVD Tracker
# ─────────────────────────────────────────────────────────────────

class CVDTracker:
    """
    Tracks BTC Cumulative Volume Delta from the spot aggTrade stream.

    CVD = Σ signed_qty where:
        signed_qty = +qty  if taker is buyer  (isBuyerMaker = False)
        signed_qty = -qty  if taker is seller (isBuyerMaker = True)

    The normalised ratio (CVD / total_volume) is the primary gate signal
    because it is volume-independent — a ratio of +0.20 means buyers
    won 60% of aggressor volume regardless of whether it was a quiet
    or active session.
    """

    def __init__(self) -> None:
        # Each entry: (unix_timestamp, signed_qty, abs_qty)
        self._trades: deque[tuple[float, float, float]] = deque()
        self._lock         = asyncio.Lock()
        self._trade_count  = 0   # total received since startup (warmup guard)

    # ── Public API ────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True once we have at least 5 minutes of accumulated data."""
        return self._trade_count > 0 and self._oldest_age() >= 300

    def snapshot(self) -> dict:
        """
        Return CVD metrics dict.
        All values are 0.0 during warmup or if stream is down.
        """
        if not self._trade_count:
            return {
                'cvd_15m':       0.0,
                'cvd_1h':        0.0,
                'cvd_ratio_15m': 0.0,
                'cvd_ratio_1h':  0.0,
                'cvd_slope':     0.0,
            }

        cvd_15m, vol_15m = self._window_sum(900)
        cvd_1h,  vol_1h  = self._window_sum(3600)

        ratio_15m = cvd_15m / vol_15m if vol_15m > 0 else 0.0
        ratio_1h  = cvd_1h  / vol_1h  if vol_1h  > 0 else 0.0
        slope     = cvd_15m / 15.0    # BTC per minute

        return {
            'cvd_15m':       round(cvd_15m,   4),
            'cvd_1h':        round(cvd_1h,    4),
            'cvd_ratio_15m': round(ratio_15m, 4),
            'cvd_ratio_1h':  round(ratio_1h,  4),
            'cvd_slope':     round(slope,      4),
        }

    # ── Async runner ──────────────────────────────────────────────

    async def run(self) -> None:
        """
        Persistent WebSocket loop. Reconnects automatically on any failure.
        Designed to be started with asyncio.create_task() inside run_ws().
        """
        import websockets

        url = f"{BINANCE_WS_SPOT}/btcusdt@aggTrade"
        while True:
            try:
                async with websockets.connect(
                    url, ping_interval=30, ping_timeout=10
                ) as ws:
                    async for raw in ws:
                        msg = json.loads(raw)
                        qty    = float(msg['q'])
                        # isBuyerMaker=True → taker is SELLER → negative delta
                        signed = -qty if msg['m'] else qty
                        ts     = msg['T'] / 1000.0   # ms → seconds
                        await self._append(ts, signed, qty)
                        self._trade_count += 1
            except Exception:
                pass   # reconnect silently — don't pollute terminal
            await asyncio.sleep(_RECONNECT_DELAY)

    # ── Internal ──────────────────────────────────────────────────

    async def _append(self, ts: float, signed_qty: float, abs_qty: float) -> None:
        cutoff = time.time() - _MAX_WINDOW
        async with self._lock:
            self._trades.append((ts, signed_qty, abs_qty))
            while self._trades and self._trades[0][0] < cutoff:
                self._trades.popleft()

    def _window_sum(self, seconds: int) -> tuple[float, float]:
        """Return (cvd, total_volume) for the last `seconds` window."""
        cutoff = time.time() - seconds
        cvd = vol = 0.0
        for ts, sq, aq in self._trades:
            if ts >= cutoff:
                cvd += sq
                vol += aq
        return cvd, vol

    def _oldest_age(self) -> float:
        """Age of the oldest retained trade in seconds."""
        if not self._trades:
            return 0.0
        return time.time() - self._trades[0][0]


# ─────────────────────────────────────────────────────────────────
# Liquidation Tracker
# ─────────────────────────────────────────────────────────────────

class LiquidationTracker:
    """
    Tracks forced liquidation volume (USD) from the futures forceOrder stream.

    The global stream (!forceOrder@arr) includes ALL futures pairs so the
    USD total reflects market-wide stress, not just BTC.

    Interpretation:
        liq_5m_usd > $20M   → notable liquidation pressure
        liq_5m_usd > $50M   → significant capitulation
        liq_5m_usd > $100M  → extreme capitulation — high-probability bounce zone

    Side breakdown:
        long_liq  = SELL-side force order (long position liquidated)
        short_liq = BUY-side force order (short position liquidated)

    A spike in long_liq >> short_liq = forced long selling → contrarian bullish.
    A spike in short_liq >> long_liq = short squeeze territory → momentum bullish.
    """

    def __init__(self) -> None:
        # Each entry: (unix_timestamp, usd_value, side)
        # side: 'long_liq' | 'short_liq'
        self._events: deque[tuple[float, float, str]] = deque()
        self._lock        = asyncio.Lock()
        self._event_count = 0

    # ── Public API ────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True once we have received at least one liquidation event."""
        return self._event_count > 0

    def snapshot(self) -> dict:
        """
        Return liquidation USD totals for 5m and 1H windows.
        All values are 0.0 during startup or if stream is down.
        """
        total_5m, long_5m, short_5m = self._window_sum(300)
        total_1h, _,       _        = self._window_sum(3600)

        return {
            'liq_5m_usd':       round(total_5m, 0),
            'liq_5m_long_usd':  round(long_5m,  0),
            'liq_5m_short_usd': round(short_5m, 0),
            'liq_1h_usd':       round(total_1h, 0),
        }

    # ── Async runner ──────────────────────────────────────────────

    async def run(self) -> None:
        """
        Persistent WebSocket loop for the futures liquidation stream.
        Reconnects automatically on any failure.
        """
        import websockets

        url = f"{BINANCE_WS_FUTURES}/!forceOrder@arr"
        while True:
            try:
                async with websockets.connect(
                    url, ping_interval=30, ping_timeout=10
                ) as ws:
                    async for raw in ws:
                        msg = json.loads(raw)
                        if msg.get('e') != 'forceOrder':
                            continue
                        o = msg['o']
                        # Use cumulative filled qty (z) with avg fill price (ap)
                        qty   = float(o.get('z') or o.get('q', 0))
                        price = float(o.get('ap') or o.get('p', 0))
                        usd   = qty * price
                        if usd <= 0:
                            continue
                        # SELL order = long position being forcibly closed
                        side = 'long_liq' if o['S'] == 'SELL' else 'short_liq'
                        ts   = msg['E'] / 1000.0
                        await self._append(ts, usd, side)
                        self._event_count += 1
            except Exception:
                pass   # reconnect silently
            await asyncio.sleep(_RECONNECT_DELAY)

    # ── Internal ──────────────────────────────────────────────────

    async def _append(self, ts: float, usd: float, side: str) -> None:
        cutoff = time.time() - _MAX_WINDOW
        async with self._lock:
            self._events.append((ts, usd, side))
            while self._events and self._events[0][0] < cutoff:
                self._events.popleft()

    def _window_sum(self, seconds: int) -> tuple[float, float, float]:
        """Return (total_usd, long_liq_usd, short_liq_usd) for the window."""
        cutoff = time.time() - seconds
        total = long_liq = short_liq = 0.0
        for ts, usd, side in self._events:
            if ts >= cutoff:
                total += usd
                if side == 'long_liq':
                    long_liq += usd
                else:
                    short_liq += usd
        return total, long_liq, short_liq
