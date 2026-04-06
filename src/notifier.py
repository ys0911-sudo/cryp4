"""
Telegram notification system for crypto scanner alerts.

Setup (one-time, takes 2 minutes)
──────────────────────────────────
1. Open Telegram → search @BotFather → send /newbot
2. Follow prompts → BotFather gives you a token like:
      7412356789:AAFxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
3. Send ANY message (e.g. /start) to your new bot
4. Visit this URL in your browser (replace <TOKEN>):
      https://api.telegram.org/bot<TOKEN>/getUpdates
   Look for "chat":{"id": 123456789} — that number is your CHAT_ID

5. Set environment variables (or add to your .env / AWS Parameter Store):
      TELEGRAM_BOT_TOKEN=7412356789:AAFxxx...
      TELEGRAM_CHAT_ID=123456789
      TELEGRAM_SEND_BLOCKED=false   # optional: set true to receive blocked alerts too

That's it. The scanner reads these on startup and sends alerts automatically.

Architecture
─────────────
All sends go through a background thread queue so the WebSocket loop is
never blocked waiting for Telegram's API. If Telegram is slow or down,
alerts queue up and drain automatically. No signal is ever missed.

Alert types
────────────
  🟢 SIGNAL  — tradeable alert (sentiment approved)
  🔴 BLOCKED — model fired but sentiment gated it (optional, off by default)
  📤 EXIT    — trade closed (SL / TP / trailing / time)
  📊 SENTIMENT — periodic market regime update (every 30 min)
"""

from __future__ import annotations

import os
import queue
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

IST = timezone(timedelta(hours=5, minutes=30))

TELEGRAM_API = "https://api.telegram.org"
_SEND_TIMEOUT = 8    # seconds per API call
_RETRY_DELAY  = 3    # seconds between retries
_MAX_RETRIES  = 2


class TelegramNotifier:
    """
    Thread-safe Telegram alert sender with background queue.

    Usage
    ─────
        notifier = TelegramNotifier()          # reads env vars
        if notifier.enabled:
            notifier.send_signal(signal)
            notifier.send_exit(trade_dict)
    """

    def __init__(
        self,
        token:        Optional[str] = None,
        chat_id:      Optional[str] = None,
        send_blocked: Optional[bool] = None,
    ):
        self.token    = token    or os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.chat_id  = chat_id  or os.getenv('TELEGRAM_CHAT_ID', '')

        # Whether to forward sentiment-blocked signals (off by default — can be noisy)
        env_blocked = os.getenv('TELEGRAM_SEND_BLOCKED', 'false').lower() == 'true'
        self.send_blocked_alerts = send_blocked if send_blocked is not None else env_blocked

        self.enabled = bool(self.token and self.chat_id)

        # Background sender thread + queue
        self._queue: queue.Queue = queue.Queue(maxsize=200)
        self._thread = threading.Thread(target=self._worker, daemon=True, name='telegram-sender')
        if self.enabled:
            self._thread.start()
            print(f"  [Telegram] Bot ready — chat_id={self.chat_id} | "
                  f"blocked_alerts={'on' if self.send_blocked_alerts else 'off'}")
        else:
            print("  [Telegram] Disabled (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID to enable)")

    # ─────────────────────────────────────────────────
    # Public send methods
    # ─────────────────────────────────────────────────

    def send_signal(self, signal: dict) -> None:
        """Queue a tradeable signal alert."""
        if not self.enabled:
            return
        self._queue.put(('signal', signal))

    def send_blocked(self, signal: dict) -> None:
        """Queue a sentiment-blocked signal alert (only if enabled)."""
        if not self.enabled or not self.send_blocked_alerts:
            return
        self._queue.put(('blocked', signal))

    def send_exit(self, trade: dict) -> None:
        """Queue a trade exit alert."""
        if not self.enabled:
            return
        self._queue.put(('exit', trade))

    def send_sentiment(self, snap: dict) -> None:
        """Queue a periodic sentiment update."""
        if not self.enabled:
            return
        self._queue.put(('sentiment', snap))

    def send_startup(self, threshold: float, top_n: int, mode: str) -> None:
        """Send a startup notification so you know the scanner is live."""
        if not self.enabled:
            return
        self._queue.put(('startup', {
            'threshold': threshold,
            'top_n':     top_n,
            'mode':      mode,
        }))

    # ─────────────────────────────────────────────────
    # Message formatters
    # ─────────────────────────────────────────────────

    @staticmethod
    def _fmt_signal(signal: dict) -> str:
        entry  = float(signal['entry_price'])
        tp     = float(signal['take_profit'])
        sl     = float(signal['stop_loss'])
        tp_pct = (tp / entry - 1) * 100
        sl_pct = (sl / entry - 1) * 100
        conf   = float(signal['confidence'])

        crossers = []
        if signal.get('rsi_cross'):  crossers.append('RSI')
        if signal.get('macd_cross'): crossers.append('MACD')
        if signal.get('ema_cross'):  crossers.append('EMA')
        if signal.get('vol_spike'):  crossers.append('Vol')
        cross_str = ' · '.join(crossers) if crossers else 'none'

        score = signal.get('sent_score', 'N/A')
        thr   = signal.get('sent_threshold', 'N/A')
        now   = datetime.now(IST).strftime('%d %b %H:%M IST')

        return (
            f"🟢 <b>SIGNAL: {signal['symbol']}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"💰 Entry: <b>{entry:.4f}</b>  |  Limit: {signal['limit_entry']:.4f}\n"
            f"📈 TP: <b>{tp:.4f}</b>  (<code>+{tp_pct:.2f}%</code>)\n"
            f"🛡 SL: <b>{sl:.4f}</b>  (<code>{sl_pct:.2f}%</code>)\n"
            f"🔔 Trail activates @ {signal['trailing_activate']:.4f}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Conf: <b>{conf:.2f}</b>  |  RSI: {signal.get('rsi', 'N/A')}  |  Indicators: {cross_str}\n"
            f"Sentiment: score={score}  thr={thr}\n"
            f"⏰ {now}"
        )

    @staticmethod
    def _fmt_blocked(signal: dict) -> str:
        entry  = float(signal['entry_price'])
        conf   = float(signal['confidence'])
        reason = signal.get('block_reason', '?')
        score  = signal.get('sent_score', 'N/A')
        now    = datetime.now(IST).strftime('%d %b %H:%M IST')
        return (
            f"🔴 <b>BLOCKED: {signal['symbol']}</b>\n"
            f"Entry: {entry:.4f}  |  Conf: {conf:.2f}\n"
            f"Reason: <code>{reason}</code>\n"
            f"Sentiment score: {score}\n"
            f"<i>(Logged for training, not traded)</i>\n"
            f"⏰ {now}"
        )

    @staticmethod
    def _fmt_exit(trade: dict) -> str:
        pnl    = float(trade.get('pnl', 0))
        reason = trade.get('reason', '?')
        entry  = trade.get('entry', 0)
        exit_p = trade.get('exit', 0)
        emoji  = '✅' if pnl > 0 else '❌'
        reason_labels = {
            'take_profit':   'Take Profit ✅',
            'stop_loss':     'Stop Loss ❌',
            'trailing_stop': 'Trailing Stop 🔔',
            'time_exit':     'Time Exit ⏳',
        }
        reason_str = reason_labels.get(reason, reason)
        now = datetime.now(IST).strftime('%d %b %H:%M IST')
        return (
            f"📤 <b>EXIT: {trade['symbol']}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Reason: <b>{reason_str}</b>\n"
            f"Entry: {entry:.4f}  →  Exit: {exit_p:.4f}\n"
            f"PnL: {emoji} <b><code>{pnl:+.2f}%</code></b>\n"
            f"⏰ {now}"
        )

    @staticmethod
    def _fmt_sentiment(snap: dict) -> str:
        rsi   = snap.get('sent_btc_4h_rsi')
        basis = snap.get('sent_basis')
        ls    = snap.get('sent_ls_ratio')
        oi    = snap.get('sent_oi_delta')
        brd   = snap.get('sent_breadth')
        score = snap.get('sent_score', '?')
        thr   = snap.get('sent_threshold', '?')

        rsi_s   = f"{rsi:.0f}"           if rsi   is not None else "N/A"
        basis_s = f"{basis * 100:+.3f}%" if basis is not None else "N/A"
        ls_s    = f"{ls:.2f}"            if ls    is not None else "N/A"
        oi_s    = f"{oi:+.1f}%"          if oi    is not None else "N/A"
        brd_s   = f"{brd:.0f}%"          if brd   is not None else "N/A"

        score_int = int(score) if isinstance(score, (int, float)) else 0
        if score_int <= -5:
            regime = "🔴 CRASH REGIME — all longs blocked"
        elif score_int <= -2:
            regime = "🟠 BEARISH — threshold raised to 0.90"
        elif score_int <= 0:
            regime = "🟡 CAUTIOUS — threshold 0.82"
        elif score_int <= 2:
            regime = "🟢 NEUTRAL — threshold 0.75"
        else:
            regime = "💚 BULLISH — threshold 0.70"

        now = datetime.now(IST).strftime('%d %b %H:%M IST')
        return (
            f"📊 <b>Market Sentiment</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"{regime}\n"
            f"Score: <b>{score}</b>  |  Threshold: <b>{thr}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"BTC 4H RSI: {rsi_s}\n"
            f"Basis: {basis_s}\n"
            f"L/S Ratio: {ls_s}\n"
            f"OI Δ: {oi_s}\n"
            f"Breadth: {brd_s}\n"
            f"⏰ {now}"
        )

    @staticmethod
    def _fmt_startup(data: dict) -> str:
        now = datetime.now(IST).strftime('%d %b %Y %H:%M IST')
        return (
            f"🚀 <b>Scanner Started</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Mode: {data['mode'].upper()}  |  Top: {data['top_n']} coins\n"
            f"Base threshold: {data['threshold']}\n"
            f"Sentiment: adaptive gating ON\n"
            f"⏰ {now}"
        )

    # ─────────────────────────────────────────────────
    # Background worker
    # ─────────────────────────────────────────────────

    def _worker(self) -> None:
        """Drain the queue and send messages in a background thread."""
        while True:
            try:
                item = self._queue.get(timeout=5)
                kind, data = item

                if kind == 'signal':
                    text = self._fmt_signal(data)
                elif kind == 'blocked':
                    text = self._fmt_blocked(data)
                elif kind == 'exit':
                    text = self._fmt_exit(data)
                elif kind == 'sentiment':
                    text = self._fmt_sentiment(data)
                elif kind == 'startup':
                    text = self._fmt_startup(data)
                else:
                    continue

                self._send_with_retry(text)
                self._queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"  [Telegram] Worker error: {e}")

    def _send_with_retry(self, text: str) -> None:
        """POST message to Telegram with up to _MAX_RETRIES attempts."""
        url = f"{TELEGRAM_API}/bot{self.token}/sendMessage"
        payload = {
            'chat_id':    self.chat_id,
            'text':       text,
            'parse_mode': 'HTML',
        }
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                r = requests.post(url, json=payload, timeout=_SEND_TIMEOUT)
                if r.status_code == 200:
                    return
                # 429 = rate limit — back off
                if r.status_code == 429:
                    retry_after = r.json().get('parameters', {}).get('retry_after', 5)
                    time.sleep(retry_after)
                else:
                    print(f"  [Telegram] Send failed ({r.status_code}): {r.text[:80]}")
                    return
            except requests.exceptions.RequestException as e:
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)
                else:
                    print(f"  [Telegram] Send error after {_MAX_RETRIES} attempts: {e}")
