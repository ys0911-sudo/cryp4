"""
Signal storage with automatic CSV rotation and summary stats.

Handles:
- Writing new signals to CSV
- Deduplication
- Exit tracking (updates open -> closed)
- Monthly CSV rotation (so files don't grow forever)
- Running stats (win rate, PnL) without reading full CSV
"""

import os
from datetime import datetime, timezone
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

    @property
    def current_csv(self) -> Path:
        """Current month's CSV file. Rotates monthly."""
        month_str = datetime.now(timezone.utc).strftime('%Y-%m')
        return self.output_dir / f"signals_{month_str}.csv"

    @property
    def latest_csv(self) -> Path:
        """Symlink-like pointer to current CSV for easy checking."""
        return self.output_dir / "signals_latest.csv"

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

        # Also write/overwrite latest pointer
        df_out.to_csv(self.latest_csv, index=False)

        self.total_signals += 1
        return True

    def check_exits(self) -> list[dict]:
        """Check all open signals for exit conditions. Returns list of closed trades."""
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

        # Batch fetch all prices in one call
        prices = self._fetch_all_prices()
        if not prices:
            return []

        now = datetime.now(timezone.utc)
        closed_trades = []
        changed = False

        for idx, row in open_signals.iterrows():
            symbol = row['symbol']
            entry = row['entry_price']
            current_price = prices.get(symbol)
            if current_price is None:
                continue

            peak = max(row.get('peak_price', entry), current_price)
            signals.at[idx, 'peak_price'] = peak

            exit_reason = self._check_exit_conditions(entry, current_price, peak, row, now)

            if exit_reason:
                pnl = (current_price / entry - 1) * 100
                signals.at[idx, 'status'] = 'closed'
                signals.at[idx, 'exit_price'] = round(current_price, 6)
                signals.at[idx, 'exit_time'] = now.strftime('%Y-%m-%d %H:%M:%S UTC')
                signals.at[idx, 'exit_reason'] = exit_reason
                signals.at[idx, 'pnl_pct'] = round(pnl, 3)
                changed = True

                # Update running stats
                self.total_closed += 1
                if pnl > 0:
                    self.total_wins += 1
                self.total_pnl += pnl

                closed_trades.append({
                    'symbol': symbol, 'reason': exit_reason,
                    'entry': entry, 'exit': current_price, 'pnl': pnl
                })

        if changed:
            signals.to_csv(csv_path, index=False)
            signals.to_csv(self.latest_csv, index=False)

        return closed_trades

    def _check_exit_conditions(self, entry, price, peak, row, now) -> str | None:
        """Evaluate exit conditions. Returns reason string or None."""
        cfg = self.exit_config

        # 1. Hard stop loss
        if price <= entry * (1 - cfg['initial_sl_pct']):
            return 'stop_loss'

        # 2. Trailing stop
        if peak >= entry * (1 + cfg['trailing_activate_pct']):
            trail_sl = peak * (1 - cfg['trailing_distance_pct'])
            if price <= trail_sl:
                return 'trailing_stop'

        # 3. Take profit
        if price >= entry * (1 + cfg['tp_pct']):
            return 'take_profit'

        # 4. Time exit
        signal_time = pd.to_datetime(row['signal_time'])
        if signal_time.tzinfo is None:
            signal_time = signal_time.replace(tzinfo=timezone.utc)
        mins_held = (now - signal_time).total_seconds() / 60
        if mins_held >= cfg['max_hold_minutes']:
            return 'time_exit'

        return None

    def _fetch_all_prices(self) -> dict[str, float]:
        """Batch fetch all current prices from Binance."""
        try:
            resp = requests.get(f"{BINANCE_REST}/api/v3/ticker/price", timeout=10)
            resp.raise_for_status()
            return {d['symbol']: float(d['price']) for d in resp.json()}
        except Exception:
            return {}

    def get_open_count(self) -> int:
        csv_path = self.current_csv
        if not csv_path.exists():
            return 0
        try:
            df = pd.read_csv(csv_path)
            return int((df.get('status', pd.Series()) == 'open').sum())
        except Exception:
            return 0

    def get_stats_summary(self) -> str:
        """One-line summary of running stats."""
        if self.total_closed == 0:
            return f"Signals: {self.total_signals} | Open: {self.get_open_count()} | No closed trades yet"
        wr = self.total_wins / self.total_closed
        avg_pnl = self.total_pnl / self.total_closed
        return (f"Signals: {self.total_signals} | Open: {self.get_open_count()} | "
                f"Closed: {self.total_closed} | WR: {wr:.0%} | Avg PnL: {avg_pnl:+.2f}%")

    def cleanup_old_csvs(self, keep_months: int = 3) -> None:
        """Delete CSV files older than `keep_months`."""
        now = datetime.now(timezone.utc)
        for f in self.output_dir.glob("signals_*.csv"):
            if f.name == 'signals_latest.csv':
                continue
            try:
                # Parse YYYY-MM from filename
                parts = f.stem.split('_')
                if len(parts) >= 2:
                    file_date = datetime.strptime(parts[1], '%Y-%m').replace(tzinfo=timezone.utc)
                    age_days = (now - file_date).days
                    if age_days > keep_months * 31:
                        f.unlink()
                        print(f"  [cleanup] Deleted old CSV: {f.name}")
            except Exception:
                continue
