import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy import stats

def plot_cumulative_asset_returns(returns, benchmark_returns=None):
    fig = go.Figure()
    cumulative_returns = (1 + returns).cumprod()
    
    for col in cumulative_returns.columns:
        fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[col], mode='lines', name=col))

    if benchmark_returns is not None:
        benchmark_cum = (1 + benchmark_returns).cumprod()
        fig.add_trace(go.Scatter(x=benchmark_cum.index, y=benchmark_cum, mode='lines', name='Benchmark'))

    fig.update_layout(title='Cumulative Asset Performance',
                  xaxis_title='Date',
                  yaxis_title='Growth of £1',
                  template='plotly_white')
    return fig

def plot_cumulative_portfolio_returns(portfolio_returns, benchmark_returns=None):
    fig = go.Figure()
    cumulative_returns = (1 + portfolio_returns).cumprod()
    fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns, mode='lines', name='Portfolio'))
    
    if benchmark_returns is not None:
        benchmark_cum = (1 + benchmark_returns).cumprod()
        fig.add_trace(go.Scatter(x=benchmark_cum.index, y=benchmark_cum, mode='lines', name='Benchmark'))

    fig.update_layout(title='Cumulative Portfolio Performance',
                xaxis_title='Date',
                yaxis_title='Growth of £1',
                template='plotly_white')
    return fig

def plot_drawdown(portfolio_returns):
    from metrics import drawdown_series

    df = drawdown_series(portfolio_returns)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df.values, fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.3)', line=dict(color='red'), name='Drawdown'))
    fig.update_layout(title='Drawdown through time',
                xaxis_title='Date',
                yaxis_title='Drawdown',
                yaxis_tickformat='.0%',
                hovermode='x unified',
                template='plotly_white')
    return fig

def plot_runup(portfolio_returns):
    from metrics import runup_series

    df = runup_series(portfolio_returns)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df.values, fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.3)', line=dict(color='green'), name='Run-up'))
    fig.update_layout(title='Run-up through time',
                xaxis_title='Date',
                yaxis_title='Run-up',
                yaxis_tickformat='.0%',
                hovermode='x unified',
                template='plotly_white')
    return fig

def plot_return_distribution(portfolio_returns, confidence_level=0.95):
    from metrics import value_at_risk

    var = value_at_risk(portfolio_returns, confidence_level=confidence_level)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=portfolio_returns, nbinsx=50, name='Daily Returns', marker_color='steelblue', histnorm='probability density'))
    fig.add_vline(x=var, line_dash='dash', line_color='red', annotation_text=f'{int(confidence_level*100)}% VaR: {var:.2%}', annotation_position='top')
    fig.add_vline(x=-var, line_dash='dash', line_color='green', annotation_text=f'{int((1-confidence_level)*100)}% Upper Tail: {-var:.2%}', annotation_position='top')

    x_range = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 500)
    
    # Normal distribution curve
    mu, sigma = portfolio_returns.mean(), portfolio_returns.std()
    normal_pdf = stats.norm.pdf(x_range, mu, sigma)
    fig.add_trace(go.Scatter(
        x=x_range, y=normal_pdf,
        mode='lines',
        name='Normal fit',
        line=dict(color='black', dash='dash', width=2)))
    
    fig.update_layout(
        title='Distribution of Daily Returns',
        xaxis_title='Daily Return',
        yaxis_title='Density',
        xaxis_tickformat='.1%',
        template='plotly_white',
        showlegend=False)

    return fig

def plot_rolling_sharpe(portfolio_returns, window=252):
    from metrics import rolling_sharpe_ratio

    fig = go.Figure()
    df = rolling_sharpe_ratio(portfolio_returns, window=window)
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df.values,
        mode='lines',
        name='Rolling Sharpe',
        line=dict(color='black', width=2)))

    fig.add_hline(y=0, line_dash='dash', line_color='grey', opacity=0.5)
    fig.add_hline(y=1, line_dash='dot', line_color='green', opacity=0.5,
                  annotation_text='Sharpe = 1', annotation_position='right')
    
    fig.update_layout(
        title=f'Rolling Sharpe Ratio ({window}-day window)',
        xaxis_title='Date',
        yaxis_title='Sharpe Ratio',
        template='plotly_white',
        showlegend=False)
    
    return fig

def plot_correlation_heatmap(asset_returns):
    corr = asset_returns.corr()

    fig = px.imshow(
        corr,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title='Asset Correlation Matrix')
    
    fig.update_layout(template='plotly_white')

    return fig

def plot_weights(tickers, weights):
    fig = px.pie(values=weights,
            names=tickers,
            title='Portfolio Composition',
            template='plotly_white')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig




if __name__ == "__main__":
    from data import fetch_prices, calculate_returns
    from metrics import portfolio_returns
    import numpy as np
    
    # Set up test data
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    prices = fetch_prices(tickers, '2020-01-01', '2024-01-01')
    asset_rets = calculate_returns(prices)
    weights = np.array([1/3, 1/3, 1/3])
    port_ret = portfolio_returns(asset_rets, weights)
    
    # Fetch a benchmark
    spy_prices = fetch_prices('SPY', '2020-01-01', '2024-01-01')
    bench_ret = calculate_returns(spy_prices).iloc[:, 0]
    
    plot_weights(tickers, weights).show()