"""
Signal storage with automatic CSV rotation and summary stats.

Handles:
- Writing new signals to CSV
- Deduplication
- Exit tracking (updates open -> closed)
  - LIVE exit checking on every WebSocket price tick (no I/O, in-memory)
  - Fallback REST-based exit checking every 60s as safety net
- Monthly CSV rotation (so files don't grow forever)
- Running stats (win rate, PnL) without reading full CSV

ATR-relative exits (1H model):
  Each signal now stores per-trade absolute exit levels computed from ATR:
    stop_loss         → entry - sl_atr_mult × ATR    (e.g. entry - 1×ATR)
    take_profit       → entry + tp_atr_mult × ATR    (e.g. entry + 2×ATR)
    trailing_activate → entry + sl_atr_mult × ATR    (break-even territory)
    trailing_distance → ATR × 0.5                    (0.5×ATR trailing width)

  These per-trade prices take priority over global EXIT_CONFIG percentages.
  EXIT_CONFIG percentages are only used as fallback when ATR was unavailable
  at signal time (e.g. very new coins with insufficient history).

Exit price accuracy:
  Exit prices are the EXACT trigger level, not the polled market price:
    stop_loss     → pos['stop_loss']              exact ATR-computed level
    take_profit   → pos['take_profit']            exact ATR-computed level
    trailing_stop → peak - pos['trailing_distance']  exact trail level
    time_exit     → live market price             (no trigger level exists)
"""

from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))   # UTC+5:30
from pathlib import Path

import pandas as pd
import numpy as np
import requests

BINANCE_REST = "https://api.binance.com"


class SignalStore:
    """Manages signal CSV with rotation and lifecycle tracking."""

    def __init__(self, output_dir: Path, exit_config: dict):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.exit_config = exit_config

        # Running stats (in-memory, survives CSV rotation)
        self.total_signals = 0
        self.total_closed = 0
        self.total_wins = 0
        self.total_pnl = 0.0

        # In-memory open positions for live (zero I/O) exit checking.
        # key: symbol (UPPERCASE), value: list of position dicts.
        self._open_positions: dict[str, list[dict]] = {}

        # Tracks positions closed by live checker so check_exits() REST fallback
        # doesn't re-add them from CSV if the CSV write temporarily failed.
        # key: "SYMBOL|signal_time"
        self._recently_closed: set[str] = set()

        self._load_open_positions()

    # ─────────────────────────────────────────────────
    # Startup
    # ─────────────────────────────────────────────────

    def _load_open_positions(self) -> None:
        """Load any open positions from CSV into memory on startup."""
        csv_path = self.current_csv
        if not csv_path.exists():
            return
        try:
            df = pd.read_csv(csv_path)
            open_df = df[df['status'] == 'open']
            for _, row in open_df.iterrows():
                sym = row['symbol']
                pos = row.to_dict()
                # Ensure peak_price is valid
                if pd.isna(pos.get('peak_price')):
                    pos['peak_price'] = pos['entry_price']
                self._open_positions.setdefault(sym, []).append(pos)
            total = sum(len(v) for v in self._open_positions.values())
            if total:
                print(f"  [store] Resumed {total} open position(s) from CSV")
        except Exception as e:
            print(f"  [store] Warning: could not load open positions: {e}")

    # ─────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────

    @property
    def current_csv(self) -> Path:
        """Current month's CSV file. Rotates monthly."""
        month_str = datetime.now(IST).strftime('%Y-%m')
        return self.output_dir / f"signals_{month_str}.csv"

    @property
    def latest_csv(self) -> Path:
        """Always-current pointer to the latest CSV."""
        return self.output_dir / "signals_latest.csv"

    # ─────────────────────────────────────────────────
    # Signal saving
    # ─────────────────────────────────────────────────

    def save_signal(self, signal: dict) -> bool:
        """Save a new signal. Returns True if new (not duplicate)."""
        csv_path = self.current_csv
        df_new = pd.DataFrame([signal])

        if csv_path.exists():
            existing = pd.read_csv(csv_path)
            dup = (
                (existing['symbol'] == signal['symbol']) &
                (existing['signal_time'] == signal['signal_time'])
            )
            if dup.any():
                return False
            df_out = pd.concat([existing, df_new], ignore_index=True)
        else:
            df_out = df_new

        df_out.to_csv(csv_path, index=False)
        df_out.to_csv(self.latest_csv, index=False)

        # Blocked signals are logged to CSV for training data but not tracked
        # as open positions (we never entered the trade)
        if signal.get('status') != 'blocked':
            pos = dict(signal)
            pos['peak_price'] = pos.get('peak_price') or pos['entry_price']
            self._open_positions.setdefault(signal['symbol'], []).append(pos)

        self.total_signals += 1
        return True

    # ─────────────────────────────────────────────────
    # LIVE exit checking (called on every WebSocket tick)
    # ─────────────────────────────────────────────────

    def check_exits_live(self, symbol: str, current_price: float) -> list[dict]:
        """
        Check exit conditions for one symbol using a live WebSocket price.

        Called on EVERY price update — must be fast (pure in-memory, no I/O).
        CSV is only written when a position actually closes.

        Returns a list of closed trade dicts (usually empty or one item).
        """
        positions = self._open_positions.get(symbol)
        if not positions:
            return []

        now = datetime.now(timezone.utc)
        closed_trades = []
        still_open = []

        for pos in positions:
            entry = float(pos['entry_price'])

            # Track peak price with each live tick
            old_peak = float(pos.get('peak_price') or entry)
            peak = max(old_peak, current_price)
            pos['peak_price'] = peak

            exit_reason = self._check_exit_conditions(
                entry, current_price, peak, pos, now
            )

            if exit_reason:
                # ── KEY FIX: use the exact trigger price, not current_price ──
                exit_price = self._get_exact_exit_price(
                    entry, peak, current_price, exit_reason, pos
                )
                pnl = round((exit_price / entry - 1) * 100, 3)

                pos.update({
                    'status':      'closed',
                    'exit_price':  round(exit_price, 8),
                    'exit_time':   now.astimezone(IST).strftime('%Y-%m-%d %H:%M:%S IST'),
                    'exit_reason': exit_reason,
                    'pnl_pct':     pnl,
                    'peak_price':  round(peak, 8),
                })

                # Register as closed BEFORE writing CSV so the REST fallback
                # check_exits() won't re-add this position if the CSV write fails.
                closed_key = f"{symbol}|{pos['signal_time']}"
                self._recently_closed.add(closed_key)

                self._write_closed_position(pos)

                self.total_closed += 1
                if pnl > 0:
                    self.total_wins += 1
                self.total_pnl += pnl

                closed_trades.append({
                    'symbol': symbol,
                    'reason': exit_reason,
                    'entry':  entry,
                    'exit':   exit_price,
                    'pnl':    pnl,
                })
            else:
                still_open.append(pos)

        self._open_positions[symbol] = still_open
        return closed_trades

    def _get_exact_exit_price(
        self,
        entry: float,
        peak: float,
        current_price: float,
        exit_reason: str,
        pos: dict,
    ) -> float:
        """
        Return the precise price at which the exit condition was triggered.
        Prevents recording a worse price caused by polling lag.

        Per-trade ATR-relative prices (stored in pos dict) take priority.
        EXIT_CONFIG percentages are fallback only.
        """
        cfg = self.exit_config
        if exit_reason == 'stop_loss':
            # Use per-trade absolute SL if available (ATR-relative model)
            if 'stop_loss' in pos and pd.notna(pos.get('stop_loss')):
                return float(pos['stop_loss'])
            return entry * (1 - cfg['initial_sl_pct'])
        elif exit_reason == 'take_profit':
            # Use per-trade absolute TP if available (ATR-relative model)
            if 'take_profit' in pos and pd.notna(pos.get('take_profit')):
                return float(pos['take_profit'])
            return entry * (1 + cfg['tp_pct'])
        elif exit_reason == 'trailing_stop':
            # Trail distance is stored as absolute price offset (e.g. 0.5 × ATR)
            if 'trailing_distance' in pos and pd.notna(pos.get('trailing_distance')):
                return peak - float(pos['trailing_distance'])
            return peak * (1 - cfg['trailing_distance_pct'])
        else:
            # time_exit has no specific trigger price — use live market price
            return current_price

    def _write_closed_position(self, pos: dict) -> None:
        """Update the CSV row for a newly closed position."""
        csv_path = self.current_csv
        if not csv_path.exists():
            return
        try:
            # Read ALL columns as str to avoid float64 dtype inference on
            # empty columns (exit_time, exit_reason, status start as '').
            # pandas converts '' → NaN → float64 by default, which then
            # rejects string assignments even after astype(object) in 2.x.
            df = pd.read_csv(csv_path, dtype=str)

            mask = (
                (df['symbol'] == str(pos['symbol'])) &
                (df['signal_time'] == str(pos['signal_time']))
            )
            if mask.any():
                idx = df.index[mask][0]
                # df.at is the safest scalar setter — no dtype coercion issues
                df.at[idx, 'status']      = str(pos.get('status', 'closed'))
                df.at[idx, 'exit_price']  = str(pos.get('exit_price', ''))
                df.at[idx, 'exit_time']   = str(pos.get('exit_time', ''))
                df.at[idx, 'exit_reason'] = str(pos.get('exit_reason', ''))
                df.at[idx, 'pnl_pct']     = str(pos.get('pnl_pct', ''))
                df.at[idx, 'peak_price']  = str(pos.get('peak_price', ''))
                df.to_csv(csv_path, index=False)
                df.to_csv(self.latest_csv, index=False)
        except Exception as e:
            print(f"  [store] Warning: CSV write failed for {pos.get('symbol')}: {e}")

    # ─────────────────────────────────────────────────
    # Fallback REST-based exit check (every 60s)
    # ─────────────────────────────────────────────────

    def check_exits(self) -> list[dict]:
        """
        Fallback exit checker using REST price fetch.

        Runs every 60s as a safety net for any positions the live checker
        may have missed (e.g. scanner restart, coins not in WebSocket stream).
        Also syncs in-memory state with CSV on restart.
        """
        csv_path = self.current_csv
        if not csv_path.exists():
            return []

        signals = pd.read_csv(csv_path)
        if 'status' not in signals.columns:
            signals['status'] = 'open'
            signals['peak_price'] = signals['entry_price']
            signals['exit_price'] = np.nan
            signals['exit_time'] = ''
            signals['exit_reason'] = ''
            signals['pnl_pct'] = np.nan

        open_signals = signals[signals['status'] == 'open']
        if len(open_signals) == 0:
            return []

        # Re-sync in-memory positions with CSV
        # (catches positions opened before a scanner restart)
        for _, row in open_signals.iterrows():
            sym = row['symbol']
            sig_time = row['signal_time']
            closed_key = f"{sym}|{sig_time}"

            # Skip positions already closed by live checker even if CSV
            # hasn't been updated yet (prevents double-exit on CSV write failure)
            if closed_key in self._recently_closed:
                continue

            already_tracked = any(
                p.get('signal_time') == sig_time
                for p in self._open_positions.get(sym, [])
            )
            if not already_tracked:
                pos = row.to_dict()
                if pd.isna(pos.get('peak_price')):
                    pos['peak_price'] = pos['entry_price']
                self._open_positions.setdefault(sym, []).append(pos)

        prices = self._fetch_all_prices()
        if not prices:
            return []

        all_closed = []
        for sym, price in prices.items():
            if self._open_positions.get(sym):
                closed = self.check_exits_live(sym, price)
                all_closed.extend(closed)

        return all_closed

    # ─────────────────────────────────────────────────
    # Exit condition logic (shared by live + REST paths)
    # ─────────────────────────────────────────────────

    def _check_exit_conditions(
        self, entry, price, peak, pos, now
    ) -> str | None:
        """Evaluate exit conditions. Returns reason string or None.

        Per-trade ATR-relative absolute prices (stored in pos dict) take
        priority over global EXIT_CONFIG percentages. This ensures each trade
        uses the TP/SL levels that were computed from ATR at signal time,
        matching the backtest's labeling logic exactly.
        """
        cfg = self.exit_config

        # 1. Hard stop loss — use per-trade absolute level if available
        sl_price = (float(pos['stop_loss'])
                    if 'stop_loss' in pos and pd.notna(pos.get('stop_loss'))
                    else entry * (1 - cfg['initial_sl_pct']))
        if price <= sl_price:
            return 'stop_loss'

        # 2. Trailing stop — use per-trade absolute activate/distance if available
        trail_activate = (float(pos['trailing_activate'])
                          if 'trailing_activate' in pos and pd.notna(pos.get('trailing_activate'))
                          else entry * (1 + cfg['trailing_activate_pct']))
        if peak >= trail_activate:
            trail_distance = (float(pos['trailing_distance'])
                              if 'trailing_distance' in pos and pd.notna(pos.get('trailing_distance'))
                              else peak * cfg['trailing_distance_pct'])
            trail_sl = peak - trail_distance
            if price <= trail_sl:
                return 'trailing_stop'

        # 3. Take profit — use per-trade absolute level if available
        tp_price = (float(pos['take_profit'])
                    if 'take_profit' in pos and pd.notna(pos.get('take_profit'))
                    else entry * (1 + cfg['tp_pct']))
        if price >= tp_price:
            return 'take_profit'

        # 4. Time exit
        raw_st = str(pos['signal_time']).replace(' IST', '').replace(' UTC', '')
        signal_time = pd.to_datetime(raw_st)
        if signal_time.tzinfo is None:
            # IST-formatted strings: subtract 5:30 to get UTC for hold duration math
            signal_time = signal_time.replace(tzinfo=IST)
        mins_held = (now - signal_time).total_seconds() / 60
        if mins_held >= cfg['max_hold_minutes']:
            return 'time_exit'

        return None

    def _fetch_all_prices(self) -> dict[str, float]:
        """Batch fetch all current prices from Binance REST."""
        try:
            resp = requests.get(
                f"{BINANCE_REST}/api/v3/ticker/price", timeout=10
            )
            resp.raise_for_status()
            return {d['symbol']: float(d['price']) for d in resp.json()}
        except Exception:
            return {}

    # ─────────────────────────────────────────────────
    # Stats & housekeeping
    # ─────────────────────────────────────────────────

    def get_open_count(self) -> int:
        return sum(len(v) for v in self._open_positions.values())

    def get_open_symbols(self) -> set[str]:
        """Return set of symbols (UPPERCASE) that currently have an open trade."""
        return {sym for sym, positions in self._open_positions.items() if positions}

    def get_stats_summary(self) -> str:
        """One-line summary of running stats."""
        if self.total_closed == 0:
            return (f"Signals: {self.total_signals} | "
                    f"Open: {self.get_open_count()} | No closed trades yet")
        wr = self.total_wins / self.total_closed
        avg_pnl = self.total_pnl / self.total_closed
        return (
            f"Signals: {self.total_signals} | Open: {self.get_open_count()} | "
            f"Closed: {self.total_closed} | WR: {wr:.0%} | Avg PnL: {avg_pnl:+.2f}%"
        )

    def cleanup_old_csvs(self, keep_months: int = 3) -> None:
        """Delete CSV files older than `keep_months`."""
        now = datetime.now(timezone.utc)
        for f in self.output_dir.glob("signals_*.csv"):
            if f.name == 'signals_latest.csv':
                continue
            try:
                parts = f.stem.split('_')   # signals_2026-04 → ['signals', '2026-04']
                if len(parts) < 2:
                    continue
                file_month = datetime.strptime(parts[1], '%Y-%m').replace(tzinfo=timezone.utc)
                months_old = (now.year - file_month.year) * 12 + (now.month - file_month.month)
                if months_old > keep_months:
                    f.unlink()
                    print(f"  [store] Deleted old CSV: {f.name}")
            except Exception:
                continue