"""
Market sentiment scorer for adaptive threshold adjustment.

Architecture
────────────
Three-tier system:

  Tier 1 – Hard gates   : block ALL longs regardless of confidence
  Tier 2 – Points score : adjust threshold up/down based on market conditions
  Tier 3 – Snapshot log : every signal records sentiment state for future
                          logistic regression weight derivation (300+ trades)

Data sources (no API key required for any of them)
────────────────────────────────────────────────────
  Basis        → fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT
                 (markPrice - indexPrice) / indexPrice
                 Real-time leading indicator; funding rate is a lagged 8H
                 settlement of basis — so we use basis directly.

  L/S ratio    → fapi.binance.com/futures/data/globalLongShortAccountRatio
                 Ratio of accounts net-long vs net-short (0-1 scale).
                 > 0.72 = crowded longs (mean-reverting risk up)
                 < 0.45 = capitulation (potential bounce)

  OI delta     → fapi.binance.com/fapi/v1/openInterest  (two polls, 5m apart)
                 % change in open interest.
                 Sharp drop (< -5%) = forced liquidations / capitulation → bullish
                 Sharp rise (>  5%) = over-leveraged → bearish / fragile

  Market breadth → passed in from CoinManager's fetch_top_coins() result.
                   % of top-N USDT pairs up in 24h.
                   < 30% = broad sell-off   > 60% = broad rally

  BTC 4H RSI   → Fetched from Binance REST klines (4H candles, RSI-14).
                  < 35 = deeply oversold (crash territory)
                  > 70 = overbought (risk-off for new longs)

Scoring
───────
  Each factor contributes ±1 to ±3 to a composite integer score.
  Score → threshold mapping:

    score ≤ -5  →  None    (block all longs — hard crash gate)
    score ≤ -2  →  0.90    (very selective — strong signals only)
    score ≤  0  →  0.82    (cautious — raised bar)
    score ≤  2  →  0.75    (neutral — slight raise)
    score  >  2  →  0.70    (bullish — slightly relaxed)

  Model default threshold (0.65 from training) is intentionally NOT used
  as the floor — we never go below 0.70 in live trading.

Future weighting
────────────────
  Once 300+ closed trades are available, fit a logistic regression:
    P(win) ~ btc_4h_rsi + basis + ls_ratio + oi_delta + breadth
  Use coefficients as feature weights to replace the hand-tuned points above.
  The snapshot columns logged with every signal provide the training data.
"""

from __future__ import annotations

import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

BINANCE_REST  = "https://api.binance.com"
BINANCE_FAPI  = "https://fapi.binance.com"
COINGECKO_API = "https://api.coingecko.com"

# Cache TTLs (seconds)
_FAST_TTL   = 60    # basis, breadth (update every candle open)
_MEDIUM_TTL = 300   # BTC 4H RSI, L/S ratio (5 min)
_SLOW_TTL   = 3600  # stablecoin market cap % (rarely changes)

# OI baseline for delta computation
_oi_baseline: dict = {'value': None, 'timestamp': 0.0}
_oi_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────
# Individual data fetchers
# ─────────────────────────────────────────────────────────────────

def _fetch_btc_basis() -> Optional[float]:
    """
    Return BTC perpetual basis as a fraction.
    basis = (markPrice - indexPrice) / indexPrice
    Positive = contango (longs pay shorts) — bullish
    Negative = backwardation (shorts pay longs) — bearish
    """
    try:
        r = requests.get(
            f"{BINANCE_FAPI}/fapi/v1/premiumIndex",
            params={"symbol": "BTCUSDT"},
            timeout=5,
        )
        r.raise_for_status()
        d = r.json()
        mark  = float(d["markPrice"])
        index = float(d["indexPrice"])
        if index == 0:
            return None
        return (mark - index) / index
    except Exception:
        return None


def _fetch_btc_ls_ratio() -> Optional[float]:
    """
    Return BTC global long/short account ratio (latest 1H period).
    Value is in [0, 1]: e.g. 0.55 means 55% of accounts are net long.
    """
    try:
        r = requests.get(
            f"{BINANCE_FAPI}/futures/data/globalLongShortAccountRatio",
            params={"symbol": "BTCUSDT", "period": "1h", "limit": 1},
            timeout=5,
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        # longShortRatio is the ratio long/short; convert to long fraction
        ls = float(data[0]["longShortRatio"])
        return ls / (1 + ls)   # → fraction of accounts that are long
    except Exception:
        return None


def _fetch_btc_oi_delta() -> Optional[float]:
    """
    Return BTC open interest % change vs 5-minute-ago baseline.
    Positive = OI growing (more leverage) → bearish risk
    Negative = OI falling (deleveraging / liquidations) → bullish signal
    """
    global _oi_baseline
    try:
        r = requests.get(
            f"{BINANCE_FAPI}/fapi/v1/openInterest",
            params={"symbol": "BTCUSDT"},
            timeout=5,
        )
        r.raise_for_status()
        current_oi = float(r.json()["openInterest"])
        now = time.time()

        with _oi_lock:
            baseline = _oi_baseline.copy()

        if baseline['value'] is None or (now - baseline['timestamp']) > 3600:
            # First read or stale baseline — reset, no delta yet
            with _oi_lock:
                _oi_baseline = {'value': current_oi, 'timestamp': now}
            return 0.0

        delta_pct = (current_oi - baseline['value']) / baseline['value'] * 100

        # Refresh baseline every 5 minutes
        if (now - baseline['timestamp']) >= 300:
            with _oi_lock:
                _oi_baseline = {'value': current_oi, 'timestamp': now}

        return delta_pct
    except Exception:
        return None


def _fetch_btc_4h_rsi() -> Optional[float]:
    """Return BTC RSI-14 on the 4H timeframe."""
    try:
        r = requests.get(
            f"{BINANCE_REST}/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "4h", "limit": 50},
            timeout=8,
        )
        r.raise_for_status()
        closes = np.array([float(c[4]) for c in r.json()])
        if len(closes) < 15:
            return None
        deltas = np.diff(closes)
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        # Wilder smoothing (EMA with alpha=1/14)
        avg_gain = np.mean(gains[:14])
        avg_loss = np.mean(losses[:14])
        for i in range(14, len(gains)):
            avg_gain = (avg_gain * 13 + gains[i]) / 14
            avg_loss = (avg_loss * 13 + losses[i]) / 14
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100 - 100 / (1 + rs), 1)
    except Exception:
        return None


def _compute_market_breadth(ticker_data: list[dict]) -> Optional[float]:
    """
    Compute % of top USDT pairs up in 24h from the already-fetched ticker list.
    ticker_data: list of dicts with 'symbol' and 'priceChangePercent' keys.
    Returns float in [0, 100] or None.
    """
    if not ticker_data:
        return None
    try:
        usdt_pairs = [
            t for t in ticker_data
            if t['symbol'].endswith('USDT')
            and not t['symbol'].endswith('DOWNUSDT')
            and not t['symbol'].endswith('UPUSDT')
        ]
        if not usdt_pairs:
            return None
        up = sum(1 for t in usdt_pairs if float(t['priceChangePercent']) > 0)
        return round(up / len(usdt_pairs) * 100, 1)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────
# SentimentScorer — cached, thread-safe
# ─────────────────────────────────────────────────────────────────

class SentimentScorer:
    """
    Fetches and caches market sentiment data, computes composite score,
    and returns an adjusted signal threshold.

    Usage
    ─────
        scorer = SentimentScorer()

        # In predict_signal() before emitting a signal:
        threshold, snapshot = scorer.evaluate(
            breadth_pct=breadth,            # from CoinManager ticker data
            base_threshold=model_threshold,
        )
        if threshold is None:
            return None   # hard gate — block signal
        if proba < threshold:
            return None   # sentiment-adjusted threshold not met

        # Attach snapshot to signal dict for CSV logging
        signal.update(snapshot)
    """

    def __init__(self):
        self._cache: dict = {}
        self._lock  = threading.Lock()

    # ── Cache helpers ──────────────────────────────────────────

    def _cached(self, key: str, ttl: int, fetcher):
        """Return cached value if fresh, else call fetcher and cache result."""
        with self._lock:
            entry = self._cache.get(key)
        now = time.time()
        if entry and (now - entry['ts']) < ttl and entry['value'] is not None:
            return entry['value']
        value = fetcher()
        with self._lock:
            self._cache[key] = {'value': value, 'ts': now}
        return value

    # ── Public API ─────────────────────────────────────────────

    def get_snapshot(self, breadth_pct: Optional[float] = None) -> dict:
        """
        Fetch (or return cached) all sentiment metrics.
        breadth_pct: pass in from CoinManager ticker if available (avoids extra REST call).
        Returns a dict of raw metric values.
        """
        btc_4h_rsi = self._cached('btc_4h_rsi', _MEDIUM_TTL, _fetch_btc_4h_rsi)
        basis      = self._cached('basis',       _FAST_TTL,   _fetch_btc_basis)
        ls_ratio   = self._cached('ls_ratio',    _MEDIUM_TTL, _fetch_btc_ls_ratio)
        oi_delta   = self._cached('oi_delta',    _FAST_TTL,   _fetch_btc_oi_delta)

        if breadth_pct is None:
            # Fallback: fetch breadth ourselves (slower path)
            breadth_pct = self._cached('breadth', _FAST_TTL, self._fetch_breadth_fallback)

        return {
            'sent_btc_4h_rsi': btc_4h_rsi,
            'sent_basis':      basis,
            'sent_ls_ratio':   ls_ratio,
            'sent_oi_delta':   oi_delta,
            'sent_breadth':    breadth_pct,
        }

    def evaluate(
        self,
        breadth_pct: Optional[float] = None,
        base_threshold: float = 0.70,
    ) -> tuple[Optional[float], dict]:
        """
        Evaluate market sentiment and return (adjusted_threshold, snapshot).

        adjusted_threshold:
            None  → hard gate, block all longs
            float → minimum model confidence required

        snapshot:
            Dict of sentiment columns to attach to every signal row.
            Keys: sent_btc_4h_rsi, sent_basis, sent_ls_ratio,
                  sent_oi_delta, sent_breadth, sent_score, sent_threshold
        """
        snap = self.get_snapshot(breadth_pct)
        score = self._compute_score(snap)
        threshold = self._score_to_threshold(score)

        snap['sent_score']     = score
        snap['sent_threshold'] = threshold if threshold is not None else 'blocked'

        return threshold, snap

    # ── Scoring logic ──────────────────────────────────────────

    def _compute_score(self, snap: dict) -> int:
        """
        Points-based composite score.
        Range roughly -10 to +6 in practice.
        See module docstring for rationale.
        """
        score = 0

        # ── BTC 4H RSI ──────────────────────────────────────
        rsi = snap.get('sent_btc_4h_rsi')
        if rsi is not None:
            if rsi < 35:
                score -= 3   # crash territory
            elif rsi < 45:
                score -= 1   # weak
            elif rsi > 70:
                score -= 1   # overbought — risk for new longs
            elif rsi > 60:
                score += 1   # healthy momentum

        # ── Basis (perpetual premium) ────────────────────────
        basis = snap.get('sent_basis')
        if basis is not None:
            basis_pct = basis * 100
            if basis_pct < -0.05:
                score -= 3   # strong backwardation — panic / forced selling
            elif basis_pct < 0:
                score -= 1   # mild negative — caution
            elif basis_pct > 0.05:
                score += 1   # healthy contango — market paying to be long

        # ── Long/Short ratio ─────────────────────────────────
        ls = snap.get('sent_ls_ratio')
        if ls is not None:
            if ls > 0.72:
                score -= 2   # crowded longs → likely to mean-revert down
            elif ls < 0.45:
                score += 1   # capitulation — contrarian bullish
            # 0.45–0.72 = neutral, no adjustment

        # ── OI delta ─────────────────────────────────────────
        oi = snap.get('sent_oi_delta')
        if oi is not None:
            if oi < -5.0:
                score += 1   # liquidation cascade → capitulation, bounce likely
            elif oi > 5.0:
                score -= 1   # leverage building fast → fragile

        # ── Market breadth ───────────────────────────────────
        breadth = snap.get('sent_breadth')
        if breadth is not None:
            if breadth < 30:
                score -= 3   # broad sell-off — avoid new longs
            elif breadth < 45:
                score -= 1   # weak market
            elif breadth > 60:
                score += 1   # broad rally — supportive

        return score

    @staticmethod
    def _score_to_threshold(score: int) -> Optional[float]:
        """Map composite score to minimum confidence threshold."""
        if score <= -5:
            return None    # hard gate — block all longs
        elif score <= -2:
            return 0.90    # very selective
        elif score <= 0:
            return 0.82    # cautious
        elif score <= 2:
            return 0.75    # neutral-positive
        else:
            return 0.70    # broadly bullish

    # ── Fallback breadth fetcher ───────────────────────────────

    @staticmethod
    def _fetch_breadth_fallback() -> Optional[float]:
        """Fetch market breadth independently (used only if not provided by caller)."""
        try:
            r = requests.get(
                f"{BINANCE_REST}/api/v3/ticker/24hr",
                timeout=10,
            )
            r.raise_for_status()
            return _compute_market_breadth(r.json())
        except Exception:
            return None

    # ── Convenience printer ────────────────────────────────────

    def print_status(self, snap: dict) -> None:
        """Print one-line sentiment summary to console."""
        rsi   = snap.get('sent_btc_4h_rsi')
        basis = snap.get('sent_basis')
        ls    = snap.get('sent_ls_ratio')
        oi    = snap.get('sent_oi_delta')
        brd   = snap.get('sent_breadth')
        score = snap.get('sent_score', '?')
        thr   = snap.get('sent_threshold', '?')

        rsi_s   = f"{rsi:.0f}"          if rsi   is not None else "N/A"
        basis_s = f"{basis * 100:+.3f}%" if basis is not None else "N/A"
        ls_s    = f"{ls:.2f}"            if ls    is not None else "N/A"
        oi_s    = f"{oi:+.1f}%"          if oi    is not None else "N/A"
        brd_s   = f"{brd:.0f}%"          if brd   is not None else "N/A"

        print(
            f"  [Sentiment] score={score:+} thr={thr} | "
            f"BTC4hRSI={rsi_s} basis={basis_s} L/S={ls_s} "
            f"OI\u0394={oi_s} breadth={brd_s}"
        )