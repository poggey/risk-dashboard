import numpy as np
import pandas as pd

def portfolio_returns(returns, weights):
    '''Calculates expected return of a portfolio with given weights and asset returns.'''
    weights = np.array(weights).flatten()
    return returns @ weights
    ## preserves dates as index

def annualised_return(returns):
    '''annualised return'''
    return np.mean(returns) * 252

def annualised_volatility(returns):
    '''annualised volatility'''
    return np.std(returns) * np.sqrt(252)

def sharpe_ratio(returns, rfr=0.04):
    '''Calculates the Sharpe Ratio of a portfolio.'''
    return (annualised_return(returns) - rfr) / annualised_volatility(returns)

def max_drawdown(returns):
    '''Returns the worst peak-to-trough loss the portfolio would have experienced over the period'''
    cumulative_wealth = (1 + returns).cumprod()
    peak = cumulative_wealth.cummax()
    drawdown = (cumulative_wealth - peak) / peak
    return drawdown.min()
    ## note the minimum is the maximum drawdown since is negative

def drawdown_series(returns):
    '''Returns a series of drawdowns over time.'''
    cumulative_wealth = (1 + returns).cumprod()
    peak = cumulative_wealth.cummax()
    drawdown = (cumulative_wealth - peak) / peak
    return drawdown
    ## identical but returns the whole series

def runup_series(returns):
    '''Returns a series of runups over time.'''
    cumulative_wealth = (1 + returns).cumprod()
    trough = cumulative_wealth.cummin()
    runup = (cumulative_wealth - trough) / trough
    return runup

def value_at_risk(returns, confidence_level=0.95):
    '''estimates the worst expected loss over the time period at the given confidence level'''
    return np.percentile(returns, (1 - confidence_level) * 100)
    ## this works backwards, its the value below which only bottom 5% of the returns fall

def conditional_value_at_risk(returns, confidence_level=0.95):
    '''estimates the expected loss given that the loss is beyond the VaR confidence level'''
    return returns[returns <= value_at_risk(returns, confidence_level)].mean()
    ## also called expected shortfall, its the average of the worst 5% 

def sortino_ratio(returns, rfr, target_return=0):
    '''sortino ratiob is the sharpe ratio but only considers volatility from returns below the target return which is set as 0 here'''
    down_returns = returns[returns < target_return]
    ratio = (annualised_return(returns) - rfr) / annualised_volatility(down_returns)
    return ratio

def beta(portfolio_returns, benchmark_returns):
    '''beta is a measure of the portfolio's sensitivity to market movements'''
    if isinstance(portfolio_returns, pd.DataFrame):
        portfolio_returns = portfolio_returns.iloc[:, 0]
    if isinstance(benchmark_returns, pd.DataFrame):
        benchmark_returns = benchmark_returns.iloc[:, 0]
    
    # Align on dates and drop missing so only compare returns on same dates
    combined = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    p, b = combined.iloc[:, 0], combined.iloc[:, 1]
    
    covariance = np.cov(p, b)[0, 1]
    variance = np.var(b)
    return covariance / variance

def rolling_sharpe_ratio(returns, window=252, rfr=0.04):
    '''calculates a rolling sharpe ratio over a year'''
    rolling_returns_mean = returns.rolling(window).mean() * 252
    rolling_stddev = returns.rolling(window).std() * np.sqrt(252)
    return (rolling_returns_mean - rfr) / rolling_stddev


if __name__ == "__main__":
    # Test the functions
    from data import fetch_prices, calculate_returns
    
    prices = fetch_prices(['AAPL', 'MSFT', 'GOOGL'], '2020-01-01', '2024-01-01')
    returns = calculate_returns(prices)
    weights = np.array([1/3, 1/3, 1/3])
    
    port_ret = portfolio_returns(returns, weights)
    
    print(f"Annualised return: {annualised_return(port_ret):.4f}")
    print(f"Annualised volatility: {annualised_volatility(port_ret):.4f}")
    print(f"Sharpe ratio: {sharpe_ratio(port_ret, 0.04):.4f}")
    print(f"Max drawdown: {max_drawdown(port_ret):.4f}")
    print(f"95% VaR: {value_at_risk(port_ret):.4f}")
    print(f"95% CVaR: {conditional_value_at_risk(port_ret):.4f}")
    print(f"Sortino: {sortino_ratio(port_ret, 0.04):.4f}")
    print(f"Beta (vs SPY): {beta(port_ret, calculate_returns(fetch_prices('SPY', '2020-01-01', '2024-01-01'))):.4f}")
    print(f"Rolling Sharpe Ratio (last value): {rolling_sharpe_ratio(port_ret).iloc[-1]:.4f}")



