import yfinance as yf
import pandas as pd
import numpy as np

def fetch_prices(tickers, start_date, end_date):
    '''Fetches adjusted close prices for the given tickers and date range.'''

    if isinstance(tickers, str):
        tickers = [tickers]
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    return data['Close']

def calculate_returns(prices):
    '''Calculates daily returns from price data.'''

    returns = prices.pct_change().dropna()
    return returns


if __name__ == "__main__":
    prices = fetch_prices(['AAPL', 'MSFT', 'GOOGL'], '2023-01-01', '2024-01-01')
    print(prices.head())
    
    returns = calculate_returns(prices)
    print(returns.head())




