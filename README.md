# Binance USDⓈ-M Futures News Trading Bot

A Python-based news-driven trading system for Binance USDⓈ-M perpetual
futures. This repository currently contains **Stage 1**: project structure,
secure configuration, and the verified-library foundation modules
(exchange connectivity, async notifications, and standardized logging).

## Modules

| Module             | Stage | Responsibility                                                        |
| ------------------ | :---: | --------------------------------------------------------------------- |
| `config.py`        |   1   | Load & validate secrets/settings from `.env` via `pydantic-settings`. |
| `logger.py`        |   1   | Standardized console + rotating-file logging, error reporting helper. |
| `exchange.py`      |   1   | Async `ccxt` Binance USDⓈ-M futures client init (testnet supported).  |
| `notifier.py`      |   1   | Async Telegram alerts via `python-telegram-bot`.                      |
| `news_analyzer.py` |   2   | RSS news ingestion (16+ feeds) + FinBERT sentiment scoring.             |
| `trading_engine.py`|   3   | `pandas_ta` RSI/ATR + slope, marketable-limit Long/Short order engine.|
| `strategy.py`      |   4   | Dynamic trailing-stop / fixed-stop / time-exit (Long/Short symmetric).|
| `bot.py`           |   4   | Core trading loop: news + indicators + strategy orchestration.        |
| `state.py`         |   4   | Thread-safe shared state bridging the bot loop and the GUI.           |
| `app.py`           |   4   | Streamlit web dashboard (balance, positions, charts, settings, logs). |
| `healthcheck.py`   |   1   | Smoke test that wires the Stage-1 modules together.                   |

## Stage 2: news ingestion + financial NLP

`news_analyzer.py` provides three pieces:

- **`NewsCollector`** — asynchronously polls news sources:
  - **RSS** (default **16+ free feeds**: Cointelegraph, CoinDesk, Decrypt, Bitcoin
    Magazine, The Block, Blockworks, NewsBTC, AMBCrypto, CryptoPotato, CoinJournal,
    Crypto.news, Bitcoinist, CryptoSlate, U.Today, BeInCrypto, …). Override with
    comma-separated `NEWS_RSS_FEEDS`.
  - **CryptoPanic API** (optional token) — when `CRYPTOPANIC_API_TOKEN` is set,
    it is used instead of RSS.
  - URL/title de-duplication across feeds.
- **`SentimentAnalyzer`** — loads the institutional finance-tuned model
  `ProsusAI/finbert` and scores English text from **-1.0 (very negative)** to
  **+1.0 (very positive)** as `P(positive) - P(negative)`. CPU inference is
  optimized via thread tuning, **INT8 dynamic quantization**, and
  `torch.inference_mode`.
- **`NewsAnalyzer`** — orchestrates a **1-minute polling loop** (configurable
  via `NEWS_POLL_INTERVAL`) that collects fresh news, scores it, and invokes a
  callback per analyzed headline.

### Recommended `.env` for maximum headline coverage (free)

```env
NEWS_POLL_INTERVAL=30
# NEWS_RSS_FEEDS=   # optional override; blank uses built-in 16+ feeds
```

```python
import asyncio
from news_analyzer import NewsAnalyzer

async def on_news(item):
    print(f"{item.score:+.2f} [{item.label}] {item.title}")

asyncio.run(NewsAnalyzer().start(on_news))   # polls every minute
```

> The first run downloads the FinBERT model (~400 MB) to the Hugging Face
> cache; subsequent runs load it locally.

## Stage 3: indicators + marketable-limit order engine

`trading_engine.py` provides:

- **Technical indicators (`pandas_ta`)** — real-time **RSI(14)** and **ATR(14)**
  on the **15m** timeframe for BTC/ETH/SOL/XRP, plus a **linear-regression
  slope** over the last 5 closes to measure trend direction & strength
  (positive = up, negative = down). Pure computation is exposed via
  `compute_indicators_from_df()` / `linreg_slope()` for easy testing.
- **Marketable Limit Order engine** — to maximize fill rate and avoid slippage,
  entries cross the spread with an **IOC/FOK limit order**:
  - **Long**: limit **buy at the best Ask**.
  - **Short**: limit **sell at the best Bid**.
  If any quantity is left unfilled, the engine logs the **specific reason** and
  **cancels immediately**.
- **Concurrent position cap** — a lock-protected counter limits simultaneous
  positions to **`MAX_POSITIONS`** (default 2); further entries are rejected.

```python
import asyncio
from exchange import create_exchange, close_exchange, load_markets_safe
from trading_engine import TradingEngine

async def main():
    ex = create_exchange()
    await load_markets_safe(ex)
    engine = TradingEngine(ex)
    ind = await engine.compute_indicators("BTC/USDT")
    print(ind)                       # RSI / ATR / slope / direction
    if engine.can_open():
        print(await engine.enter_position("BTC/USDT", "long"))
    await close_exchange(ex)

asyncio.run(main())
```

### Stage 3 environment variables

| Variable              | Description                                    |
| --------------------- | ---------------------------------------------- |
| `TRADE_SYMBOLS`       | Comma list, e.g. `BTC,ETH,SOL,XRP`.            |
| `TIMEFRAME`           | Candle timeframe for indicators (`15m`).       |
| `MAX_POSITIONS`       | Max simultaneous positions (default `2`).      |
| `ORDER_TIME_IN_FORCE` | `IOC` or `FOK` for marketable-limit orders.    |
| `POSITION_SIZE_USDT`  | Notional size per entry in USDT.               |
| `LEVERAGE`            | Leverage multiplier applied per symbol.        |

## Stage 4: dynamic exits + Streamlit dashboard

### Strategy (`strategy.py`) — Long/Short symmetric

- **Fixed stop-loss**: immediate **market** close when price moves
  `STOP_LOSS_PCT`% against the entry.
- **Dynamic trailing stop**: take-profit line starts at **ATR x 3.0** and
  ratchets in the favorable direction. It **tightens to ATR x 1.5** (snapping
  the line right next to price) when:
  - **Long**: slope > 0 **and** news score > `0.7`, **or** RSI strongly crosses
    above 50;
  - **Short**: slope < 0 **and** news score < `-0.7`, **or** RSI strongly
    crosses below 50.
- **Time exit**: if there is no clear trend (no tightening) and the position is
  held beyond `TIME_EXIT_HOURS` (7h), it is fully closed via a
  **marketable-limit** order.

### Core loop (`bot.py`)

Runs the real news pipeline (RSS + FinBERT) in all modes. Strong sentiment news
mentioning a target coin, confirmed by indicator direction, triggers a
marketable-limit entry; positions are then managed by `strategy.Position`.

- **LIVE** mode (real credentials): trades via `ccxt` on Binance USDⓈ-M.
- **SIM** mode (placeholder credentials): paper-trades on a synthetic market so
  the dashboard is fully demonstrable without keys.

### Telegram alerts

Position open/close messages include **news content / news score / position
size / coin / entry & exit price** (see `notifier.send_position_open/close`).

### Web dashboard (`app.py`)

```bash
streamlit run app.py
```

- Live **account balance (USDT)** and **open positions** table.
- **Two candlestick charts** overlaying the dynamic **take-profit / stop-loss /
  entry** lines.
- **Settings panel**: margin mode (Isolated/Cross), leverage, investment size,
  stop-loss %, trailing ATR multiple, time-exit hours, plus Start/Stop.
- Scrolling **news + sentiment scores** and **entry/exit + order-failure logs**.

The bot runs in a background thread inside the Streamlit process, sharing data
through the thread-safe `state.STATE` singleton — a single, remote-deployable
process.

## Architecture & libraries

- **Configuration**: `pydantic-settings` (12-factor, fail-fast validation,
  secrets wrapped in `SecretStr` so they never leak into logs).
- **Exchange**: `ccxt` async client (`ccxt.async_support.binance`) configured
  for `defaultType=future` with rate limiting and testnet/sandbox support.
- **Notifications**: `python-telegram-bot` (async `Bot.send_message`).
- **Logging**: stdlib `logging` with a `RotatingFileHandler` (10 MB × 5) and a
  shared format across console and file; `log_exception()` emits a standardized
  failure line (context, exception type, error code, message, traceback).

## Setup

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # Windows PowerShell
# source .venv/bin/activate         # macOS/Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. configure secrets
copy .env.example .env             # Windows
# cp .env.example .env               # macOS/Linux
# then edit .env and fill in your real keys
```

### Required environment variables

| Variable             | Description                                  |
| -------------------- | -------------------------------------------- |
| `BINANCE_API_KEY`    | Binance API key (Futures-enabled).           |
| `BINANCE_SECRET_KEY` | Binance API secret.                          |
| `TELEGRAM_TOKEN`     | Telegram bot token from @BotFather.          |
| `TELEGRAM_CHAT_ID`   | Target chat/channel ID for alerts.           |
| `BINANCE_TESTNET`    | `true`/`false` — use Futures testnet.        |
| `LOG_LEVEL`          | `DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL`. |
| `LOG_DIR`            | Directory for log files (default `logs`).    |

## Verify the installation

```bash
python healthcheck.py
```

With placeholder credentials, the live network steps are skipped (the client
objects still initialize), and everything is logged to the console and to
`logs/trading_bot.log`. Once you fill in real keys, the health check will load
markets from Binance and send a test Telegram message.

## Security notes

- `.env` is git-ignored; **never commit real credentials**.
- Secrets are stored as `SecretStr` and only unwrapped at the exact point of
  use, so they don't appear in logs or tracebacks.
- Start with `BINANCE_TESTNET=true` until the full strategy is validated.
