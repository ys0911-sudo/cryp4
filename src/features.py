"""
Feature engineering — optimized for incremental computation.

Instead of recomputing all 57 columns on 300 rows every candle,
we keep a rolling buffer and only recompute the tail when possible.
For indicators with long lookback (EMA 200), we still need full history
but avoid copying the entire DataFrame.
"""

import pandas as pd
import numpy as np
import ta


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicator columns."""
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    df['rsi'] = ta.momentum.RSIIndicator(c, window=14).rsi()

    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    df['macd_line'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    df['ema_9'] = c.ewm(span=9, adjust=False).mean()
    df['ema_21'] = c.ewm(span=21, adjust=False).mean()
    df['ema_50'] = c.ewm(span=50, adjust=False).mean()
    df['ema_200'] = c.ewm(span=200, adjust=False).mean()

    df['vol_sma_20'] = v.rolling(20).mean()
    df['vol_spike_ratio'] = v / df['vol_sma_20']

    df['atr'] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    df['atr_pct'] = df['atr'] / c

    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / c

    return df


def compute_crossover_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Detect crossover events."""
    df['rsi_cross_up'] = (df['rsi'] > 50) & (df['rsi'].shift(1) <= 50)
    df['rsi_cross_down'] = (df['rsi'] < 50) & (df['rsi'].shift(1) >= 50)
    df['rsi_cross_30_up'] = (df['rsi'] > 30) & (df['rsi'].shift(1) <= 30)

    df['macd_cross_up'] = (
        (df['macd_line'] > df['macd_signal']) &
        (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))
    )
    df['macd_cross_down'] = (
        (df['macd_line'] < df['macd_signal']) &
        (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))
    )

    df['ema_9_21_cross_up'] = (
        (df['ema_9'] > df['ema_21']) & (df['ema_9'].shift(1) <= df['ema_21'].shift(1))
    )
    df['ema_9_21_cross_down'] = (
        (df['ema_9'] < df['ema_21']) & (df['ema_9'].shift(1) >= df['ema_21'].shift(1))
    )

    df['vol_spike'] = df['vol_spike_ratio'] > 2.0

    df['bull_signal_count'] = (
        df['rsi_cross_up'].astype(int) + df['macd_cross_up'].astype(int) +
        df['ema_9_21_cross_up'].astype(int) + df['vol_spike'].astype(int)
    )
    df['any_bull_signal'] = df['bull_signal_count'] >= 1

    return df


def compute_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add contextual features."""
    df['ema_50_slope'] = df['ema_50'].pct_change(5)
    df['ema_200_slope'] = df['ema_200'].pct_change(10)
    df['close_vs_ema50'] = df['close'] / df['ema_50'] - 1
    df['close_vs_ema200'] = df['close'] / df['ema_200'] - 1
    df['ema_50_above_200'] = (df['ema_50'] > df['ema_200']).astype(int)

    df['ret_1'] = df['close'].pct_change(1)
    df['ret_5'] = df['close'].pct_change(5)
    df['ret_10'] = df['close'].pct_change(10)
    df['ret_20'] = df['close'].pct_change(20)

    df['body_pct'] = (df['close'] - df['open']) / df['open']
    df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']

    df['taker_buy_ratio'] = df['taker_buy_quote_vol'] / df['quote_volume'].replace(0, np.nan)

    df['trades_sma_20'] = df['num_trades'].rolling(20).mean()
    df['trades_spike'] = df['num_trades'] / df['trades_sma_20'].replace(0, np.nan)

    bb_range = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
    df['bb_position'] = (df['close'] - df['bb_lower']) / bb_range

    df['price_low_10'] = df['close'].rolling(10).min()
    df['rsi_low_10'] = df['rsi'].rolling(10).min()
    df['price_at_10low'] = (df['close'] <= df['price_low_10'] * 1.001).astype(int)
    df['rsi_above_10low'] = (df['rsi'] > df['rsi_low_10'] + 2).astype(int)
    df['bull_divergence'] = df['price_at_10low'] & df['rsi_above_10low']

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline."""
    df = df.copy()
    df = compute_indicators(df)
    df = compute_crossover_signals(df)
    df = compute_context_features(df)
    return df


FEATURE_COLUMNS = [
    'rsi', 'macd_hist', 'atr_pct', 'bb_width', 'bb_position',
    'vol_spike_ratio',
    'ema_50_slope', 'ema_200_slope', 'close_vs_ema50', 'close_vs_ema200',
    'ema_50_above_200',
    'ret_1', 'ret_5', 'ret_10', 'ret_20',
    'body_pct', 'upper_wick', 'lower_wick',
    'taker_buy_ratio', 'trades_spike',
    'rsi_cross_up', 'macd_cross_up', 'ema_9_21_cross_up', 'vol_spike',
    'bull_signal_count',
    'bull_divergence',
]
