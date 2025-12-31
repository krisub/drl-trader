import yfinance as yf
import pandas as pd
from ta.trend import MACD, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator

# 30 Tickers from Table I.
# GOOGL is used (Class A), not GOOG.
TICKERS = [
    "AXP",
    "AMGN",
    "AAPL",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "JNJ",
    "KO",
    "JPM",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "UNH",
    "CRM",
    "VZ",
    "V",
    "WMT",
    "DIS",
    "GOOGL",
    "NVDA",
    "AMZN",
]

# More recent date ranges.
TRAIN_START = "2012-07-27"
TRAIN_END = "2018-12-31"
TEST_START = "2019-01-01"
TEST_END = "2023-01-24"


def fetch_and_process_data(tickers, start_date, end_date):
    print(f"Fetching data for {len(tickers)} stocks from {start_date} to {end_date}...")

    # auto_adjust=True: handle stock splits/dividends automatically
    raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    # yfinance returns a MultiIndex (Price Type, Ticker).
    # Stacking to '(Date, Ticker) -> Features' requires processing per ticker to add indictors.
    processed_frames = []

    for ticker in tickers:
        try:
            df = (
                raw_data.xs(ticker, level=1, axis=1).copy()
                if isinstance(raw_data.columns, pd.MultiIndex)
                else raw_data.copy()
            )

            if df.empty:
                print(f"No data for {ticker}")
                continue

            # Indicators: Adjusted Close, MACD, MACD Signal, MACD Histogram, CCI, RSI, ADX
            macd = MACD(
                close=df["Close"], window_slow=26, window_fast=12, window_sign=9
            )
            df["MACD"] = macd.macd()
            df["MACD_signal"] = macd.macd_signal()
            df["MACD_hist"] = macd.macd_diff()
            df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
            df["CCI"] = CCIIndicator(
                high=df["High"], low=df["Low"], close=df["Close"], window=14
            ).cci()
            adx = ADXIndicator(
                high=df["High"], low=df["Low"], close=df["Close"], window=14
            )
            df["ADX"] = adx.adx()

            # TODO: Rename columns to match paper's expected format if needed
            # ex. MACD_12_26_9 -> macd

            df["ticker"] = ticker
            processed_frames.append(df)

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    full_df = pd.concat(processed_frames)

    # "252 trading days": yfinance already returns only trading days (excludes weekends/holidays).
    full_df.dropna(inplace=True)

    return full_df


market_data = fetch_and_process_data(TICKERS, TRAIN_START, TEST_END)
print(f"Data shape: {market_data.shape}")
print(market_data.head())
