# Deployment Guide — Push & Update on AWS Ubuntu Server

## What changed in this update

- **Primary timeframe**: 15m → **1H** (WebSocket now subscribes to `@kline_1h`)
- **Model**: `signal_model_15m.joblib` → **`signal_model.joblib`** (ATR-relative 2:1 R:R)
- **TP/SL**: Fixed percentages → **per-trade ATR-relative absolute prices** (stored in every signal)
- **No re-entry**: Scanner skips coins that already have an open trade
- **Large-cap exclusion**: BTC, ETH, BNB, XRP, SOL, ADA, DOGE, TRX, LTC, BCH, LINK, DOT, XLM, AVAX, UNI excluded from signals
- **Multi-TF context**: 4H and 1D features merged into every 1H prediction row
- **Cross-TF features**: `htf_trend_aligned`, `rsi_htf_diff_4h`, `daily_trend_rising`, etc.
- **Gold/stablecoin exclusion**: PAXG, XAUT, XAGU excluded from coin universe

---

## Step 1 — Commit and push locally (your machine)

```bash
# Check what changed
git status
git diff --stat

# Stage everything EXCEPT the "Training Data Program" folder
git add .
git reset HEAD "Training Data Program/"

# Commit
git commit -m "Apply 1H model: ATR TP/SL, no-re-entry, HTF context, kline_1h stream"

# Push to your active branch (replace 'your-branch' with the actual branch name)
git push origin your-branch
```

> **Note**: The model `.joblib` files are binary and may be large (~5–20 MB). If Git LFS is configured, they'll be tracked through that automatically. If not and the push is slow, that's normal.

---

## Step 2 — SSH into your AWS Ubuntu server

```bash
ssh ubuntu@your-ec2-ip
# or
ssh -i ~/.ssh/your-key.pem ubuntu@your-ec2-ip
```

---

## Step 3 — Pull and restart the scanner

```bash
# Navigate to your project directory
cd ~/crypto-scanner    # adjust path to where you cloned the repo

# Pull latest changes from your branch
git pull origin your-branch

# Verify the model file is present
ls -lh output/signal_model.joblib
ls -lh output/signal_model.joblib.sha256

# Check which systemd service runs the scanner
sudo systemctl status crypto-scanner   # or whatever your service is named
```

### If running via systemd (recommended):
```bash
# Restart the service to pick up new code and model
sudo systemctl restart crypto-scanner

# Watch live logs to confirm it starts correctly
sudo journalctl -u crypto-scanner -f
```

You should see output like:
```
  Model integrity: OK (a3f9b12c4...)
  Model loaded: 47 features | threshold=0.65 | TP=2.0×ATR SL=1.0×ATR
  Warming up 50 coins...
  WebSocket connected (50 coins)
```

### If running via screen/tmux:
```bash
# Kill the old session
screen -ls                    # find session name
screen -X -S crypto quit      # kill it (replace 'crypto' with session name)

# Start fresh
screen -S crypto
python -m src.scanner
# Ctrl+A then D to detach
```

### If running via nohup:
```bash
# Kill the old process
ps aux | grep scanner
kill <PID>

# Start fresh
nohup python -m src.scanner > logs/scanner.log 2>&1 &
tail -f logs/scanner.log
```

---

## Step 4 — Verify it's working

Check the live log for these indicators:

| What to look for | Means |
|---|---|
| `Model integrity: OK` | SHA-256 hash verified — correct model loaded |
| `TP=2.0×ATR SL=1.0×ATR` | ATR-relative exits active |
| `WebSocket connected (50 coins)` | 1H stream subscribed correctly |
| `SIGNAL: XYZUSDT @ 0.1234` | Live signal firing |
| `EXIT XYZUSDT: stop_loss` | Exit tracking working |

**No longer expect** to see `@kline_15m` in logs — it's now `@kline_1h`.

---

## Step 5 — Optional: retrain on the server (if you have data downloaded)

If you have the OHLCV training data on the server and want to retrain there:

```bash
cd ~/crypto-scanner/Training\ Data\ Program

# Retrain 1H model
python -m src.train

# Retrain 4H model
python -m src.train_4h

# Copy fresh models to scanner output
cp output/signal_model.joblib ../output/
cp output/signal_model.joblib.sha256 ../output/
cp output/signal_model_4h.joblib ../output/
cp output/signal_model_4h.joblib.sha256 ../output/

# Restart scanner to pick up new model
sudo systemctl restart crypto-scanner
```

---

## Rollback (if something breaks)

```bash
# On the server — revert to previous commit
git log --oneline -5          # find the previous commit hash
git checkout <previous-hash>  # or git revert HEAD
sudo systemctl restart crypto-scanner
```

---

## Common issues

**`ERROR: No model at output/signal_model.joblib`**
The model wasn't committed or wasn't pulled. Run `git pull` and check `ls output/`.

**`ERROR: Model integrity check failed!`**
The `.sha256` file doesn't match the `.joblib`. Retrain or re-copy both files together — never copy one without the other.

**`NameError` or `ImportError` on start**
Run `python -c "from src.scanner import main"` on the server to see the exact error. Usually means a dependency needs installing: `pip install lightgbm ta websockets --break-system-packages`.

**Scanner starts but no signals fire**
Normal — the 1H model fires much less frequently than the old 15m model (hourly candles close 4× less often). Check `signals_latest.csv` after a few hours. If still nothing after 24h, lower threshold: `python -m src.scanner -t 0.60`.
