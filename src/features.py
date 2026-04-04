"""
Feature engineering — synced with Training Data Program pipeline.

Changes from original
─────────────────────
FIX #12  safe_vol_sma guards against division-by-zero in vol_spike_ratio.
FIX #16  rsi_above_10low threshold raised from +2 to +5 (reduces false divergence).
NEW      Temporal features: hour_of_day, day_of_week, coin_age_days, week_high_proximity.
NEW      compute_htf_features(): cross-TF derived features after HTF merge.
NEW      HTF_DERIVED_COLUMNS list for feature_cols assembly in scanner.
"""

import pandas as pd
import numpy as np
import ta


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicator columns."""
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    df['rsi'] = ta.momentum.RSIIndicator(c, window=14).rsi()

    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    df['macd_line']   = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist']   = macd.macd_diff()

    df['ema_9']   = c.ewm(span=9,   adjust=False).mean()
    df['ema_21']  = c.ewm(span=21,  adjust=False).mean()
    df['ema_50']  = c.ewm(span=50,  adjust=False).mean()
    df['ema_200'] = c.ewm(span=200, adjust=False).mean()

    # FIX #12: guard against zero vol_sma before dividing
    df['vol_sma_20']      = v.rolling(20).mean()
    safe_vol_sma          = df['vol_sma_20'].replace(0, np.nan)
    df['vol_spike_ratio'] = v / safe_vol_sma

    df['atr']     = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    df['atr_pct'] = df['atr'] / c

    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / c

    return df


def compute_crossover_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Detect crossover events."""
    df['rsi_cross_up']    = (df['rsi'] > 50) & (df['rsi'].shift(1) <= 50)
    df['rsi_cross_down']  = (df['rsi'] < 50) & (df['rsi'].shift(1) >= 50)
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
    """Add contextual and temporal features."""
    df['ema_50_slope']    = df['ema_50'].pct_change(5)
    df['ema_200_slope']   = df['ema_200'].pct_change(10)
    df['close_vs_ema50']  = df['close'] / df['ema_50'] - 1
    df['close_vs_ema200'] = df['close'] / df['ema_200'] - 1
    df['ema_50_above_200'] = (df['ema_50'] > df['ema_200']).astype(int)

    df['ret_1']  = df['close'].pct_change(1)
    df['ret_5']  = df['close'].pct_change(5)
    df['ret_10'] = df['close'].pct_change(10)
    df['ret_20'] = df['close'].pct_change(20)

    df['body_pct']   = (df['close'] - df['open']) / df['open']
    df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']

    df['taker_buy_ratio'] = (
        df['taker_buy_quote_vol'] / df['quote_volume'].replace(0, np.nan)
    )

    df['trades_sma_20'] = df['num_trades'].rolling(20).mean()
    df['trades_spike']  = df['num_trades'] / df['trades_sma_20'].replace(0, np.nan)

    bb_range = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
    df['bb_position'] = (df['close'] - df['bb_lower']) / bb_range

    df['price_low_10']    = df['close'].rolling(10).min()
    df['rsi_low_10']      = df['rsi'].rolling(10).min()
    df['price_at_10low']  = (df['close'] <= df['price_low_10'] * 1.001).astype(int)
    # FIX #16: raised from +2 to +5 to reduce false bullish divergence signals
    df['rsi_above_10low'] = (df['rsi'] > df['rsi_low_10'] + 5).astype(int)
    df['bull_divergence'] = df['price_at_10low'] & df['rsi_above_10low']

    # NEW: temporal features
    if 'open_time' in df.columns:
        times = pd.to_datetime(df['open_time'], utc=True)
        df['hour_of_day']  = times.dt.hour.astype(float)
        df['day_of_week']  = times.dt.dayofweek.astype(float)
        # 7-day high proximity: 0 = at the high, negative = below it
        rolling_high = df['close'].rolling(168, min_periods=1).max()  # 168 × 1H = 7 days
        df['week_high_proximity'] = df['close'] / rolling_high.replace(0, np.nan) - 1
        # Days since first candle in buffer (large value = mature coin)
        first_time = times.iloc[0]
        df['coin_age_days'] = (times - first_time).dt.total_seconds() / 86400
    else:
        df['hour_of_day']         = 12.0
        df['day_of_week']         = 2.0
        df['week_high_proximity'] = 0.0
        df['coin_age_days']       = 365.0

    return df


def compute_htf_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-TF derived features — call AFTER HTF columns are merged in.

    Works on whatever HTF suffix columns are present (_4h, _1d) so it is
    compatible with both the 1H scanner (has _4h + _1d) and a future
    4H scanner (has only _1d).
    """
    df = df.copy()

    # How many TFs agree EMA_50 > EMA_200 (0, 1, 2, or 3)
    alignment_cols = [
        c for c in ['ema_50_above_200', 'ema_50_above_200_4h', 'ema_50_above_200_1d']
        if c in df.columns
    ]
    if alignment_cols:
        df['htf_trend_aligned'] = (
            df[alignment_cols].fillna(0).sum(axis=1).astype(int)
        )

    # RSI spread between primary TF and higher TFs
    if 'rsi' in df.columns and 'rsi_4h' in df.columns:
        df['rsi_htf_diff_4h'] = df['rsi'] - df['rsi_4h']
    if 'rsi' in df.columns and 'rsi_1d' in df.columns:
        df['rsi_htf_diff_1d'] = df['rsi'] - df['rsi_1d']

    # 4H extremes
    if 'rsi_4h' in df.columns:
        df['rsi_4h_overbought'] = (df['rsi_4h'] > 70).astype(int)
        df['rsi_4h_oversold']   = (df['rsi_4h'] < 30).astype(int)

    # 4H MACD momentum
    if 'macd_hist_4h' in df.columns:
        df['macd_hist_4h_pos'] = (df['macd_hist_4h'] > 0).astype(int)

    # Daily trend direction
    if 'ema_50_slope_1d' in df.columns:
        df['daily_trend_rising'] = (df['ema_50_slope_1d'] > 0).astype(int)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline (primary TF only — HTF merge happens in scanner)."""
    df = df.copy()
    df = compute_indicators(df)
    df = compute_crossover_signals(df)
    df = compute_context_features(df)
    return df


FEATURE_COLUMNS = [
    # Core indicators
    'rsi', 'macd_hist', 'atr_pct', 'bb_width', 'bb_position',
    'vol_spike_ratio',
    # Trend
    'ema_50_slope', 'ema_200_slope', 'close_vs_ema50', 'close_vs_ema200',
    'ema_50_above_200',
    # Returns
    'ret_1', 'ret_5', 'ret_10', 'ret_20',
    # Candle structure
    'body_pct', 'upper_wick', 'lower_wick',
    # Order flow
    'taker_buy_ratio', 'trades_spike',
    # Signal flags
    'rsi_cross_up', 'macd_cross_up', 'ema_9_21_cross_up', 'vol_spike',
    'bull_signal_count', 'bull_divergence',
    # Temporal (NEW)
    'hour_of_day', 'day_of_week', 'coin_age_days', 'week_high_proximity',
]

# Cross-TF derived features added after HTF merge (included in model feature set)
HTF_DERIVED_COLUMNS = [
    'htf_trend_aligned',
    'rsi_htf_diff_4h',
    'rsi_htf_diff_1d',
    'rsi_4h_overbought',
    'rsi_4h_oversold',
    'macd_hist_4h_pos',
    'daily_trend_rising',
]
