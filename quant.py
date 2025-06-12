# imports
import numpy as np
import pandas as pd
import requests
import logging

# finance
import yfinance as yf
import pandas_datareader as pdr
import talib as ta

# visualisation
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# time
import time
from datetime import date, datetime, timedelta 


# downloading crypto from binance
def get_coins():
        # set up logging to display info and error messages
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        # download sol
        coin = ["SOLUSDT"]

        def fetch_crypto_ohlc(coin, interval="1d"):
            url = "https://api.binance.com/api/v1/klines"
        
            # 180 days back from today as it is max for binance api
            start_time = datetime.now() - timedelta(days=180)
            end_time = datetime.now()

            all_data = []

            while start_time < end_time:
                # define request parameters for binance api
                params = {
                    "symbol": coin,
                    "interval": interval,
                    "startTime": int(start_time.timestamp() * 1000),
                    "endTime": int(
                        (start_time + timedelta(days=90)).timestamp() * 1000
                    ),  # fetching 90 days at a time
                }

                response = requests.get(url, params=params)

                # check if request was successful
                if response.status_code != 200:
                    logging.error(f"Error fetching data for {coin}: {response.status_code}")
                    break

                data = response.json()

                # check if data is returned
                if not data:
                    logging.warning(f"No OHLC data found for {coin}.")
                    break

                all_data.extend(data)

                # update start time for next request
                start_time = pd.to_datetime(data[-1][0], unit="ms") + timedelta(
                    milliseconds=1
                )

                logging.info(f"Fetched data for {coin} up to {start_time}")
                time.sleep(0.5)  # pause to avoid api limits

            if not all_data:
                return pd.DataFrame()

            # convert data into df with appropriate column names
            ohlc_df = pd.DataFrame(
                all_data,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base_asset_volume",
                    "taker_buy_quote_asset_volume",
                    "ignore",
                ],
            )

            # columns
            ohlc_df["ticker"] = coin

            ohlc_df["timestamp"] = pd.to_datetime(ohlc_df["timestamp"], unit="ms")
            ohlc_df.set_index("timestamp", inplace=True)

            # convert price and volume to float
            ohlc_df[["open", "high", "low", "close", "volume"]] = ohlc_df[["open", "high", "low", "close", "volume"]].astype(float)
            ohlc_df["price_change"] = (ohlc_df["close"] - ohlc_df["open"]) / ohlc_df["open"] * 100
            
            return ohlc_df[
                [
                    "ticker",
                    "open",
                    "high",
                    "low",
                    "close",
                    "price_change",
                    "volume"
                ]
            ]

        def generate_datetime_features(df):

            # generate additional datetime features
            df["year"] = df.index.year.astype(int)
            df["month"] = df.index.month.astype(int)
            df["day"] = df.index.dayofweek.astype(int)
            df["hour"] = df.index.hour.astype(int)

            return df

        # main script
        all_data = pd.DataFrame()

        for coin in coin:
            logging.info(f"Fetching data for {coin}")
            df = fetch_crypto_ohlc(coin)

            if not df.empty:
                df = generate_datetime_features(df)
                all_data = pd.concat([all_data, df])
            else:
                logging.warning(f"No data fetched for {coin}")

        # reorder columns to have datetime features at the beginning
        datetime_features = ["year", "month", "day", "hour", "ticker"]
        other_columns = [col for col in all_data.columns if col not in datetime_features]
        all_data = all_data[datetime_features + other_columns]

        all_data['ticker'] = all_data['ticker'].astype('category')

        # save df to csv
        all_data.to_csv('sol.csv', index=True)
        logging.info("Data fetching and processing complete. Data saved to crypto.csv")

        all_data.info()

        all_data.describe()

if __name__ == "__main__":
        get_coins()


# daily returns

# calculate daily returns based on close price
df = pd.read_csv("crypto.csv", parse_dates=True, index_col=0)
df["return"] = df["close"].pct_change()

mean_return = df["return"].mean()
volatility = df["return"].std()
skewness = df["return"].skew()
kurtosis = df["return"].kurt()

print("Mean return:", mean_return)
print("Volatility:", volatility)
print("Skewness:", skewness)
print("Kurtosis:", kurtosis)

sns.histplot(df["return"].dropna(), bins=50, kde=True)
plt.title("Return Distribution of SOL")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.show()

df["rolling_vol"] = df["return"].rolling(window=20).std()

df["rolling_vol"].plot(figsize=(12, 4), title="20-Day Rolling Volatility")

