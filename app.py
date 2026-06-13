import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data import fetch_prices, calculate_returns
from metrics import (portfolio_returns, annualised_return, annualised_volatility,
                     sharpe_ratio, max_drawdown, drawdown_series, runup_series,
                     value_at_risk, conditional_value_at_risk, sortino_ratio,
                     beta, rolling_sharpe_ratio)
from visualisations import (plot_cumulative_asset_returns, plot_cumulative_portfolio_returns,  # type: ignore
                            plot_drawdown, plot_runup, plot_return_distribution,
                            plot_rolling_sharpe, plot_correlation_heatmap, plot_weights)

st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide")

## this is just setup

st.title("Portfolio Risk Analytics Dashboard")
st.markdown("Enter your portfolio details in the sidebar and click **Analyse** to see the full risk and performance breakdown.")


## Popular tickers should be listed (organised)
TICKER_PRESETS = {
    "US Tech": ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "TSLA"],
    "US Banks": ["JPM", "BAC", "GS", "MS", "WFC", "C"],
    "UK FTSE 100": ["ISF.L", "BARC.L", "HSBA.L", "SHEL.L", "AZN.L", "GSK.L", "ULVR.L"],
    "Global ETFs": ["VWRL.L", "SPY", "QQQ", "VTI", "EFA", "EEM"],
    "Bonds & Gold": ["IGLT.L", "IBTM.L", "SGLN.L", "TLT", "GLD"],
    "Defensives": ["JNJ", "PG", "KO", "WMT", "PEP"],
}
ALL_PRESETS = sorted(set([t for sublist in TICKER_PRESETS.values() for t in sublist]))

## sidebar
with st.sidebar:
    st.subheader("Quick Start")
    preset_cols = st.columns(3)

    def apply_preset(preset_tickers):
        st.session_state.ticker_multiselect = preset_tickers
        st.rerun()

    with preset_cols[0]:
        if st.button("4 US Tech", use_container_width=True):
            apply_preset(TICKER_PRESETS["US Tech"][:4])
    with preset_cols[1]:
        if st.button("4 UK Blue Chips", use_container_width=True):
            apply_preset(TICKER_PRESETS["UK FTSE 100"][:4])
    with preset_cols[2]:
        if st.button("Multi-Asset", use_container_width=True):
            apply_preset(["VWRL.L", "IGLT.L", "SGLN.L", "IUKP.L"])

    st.divider()

    ## Ticker selection
    st.subheader("Select Tickers")

    # if this is the first run start a session to record the current state
    if "ticker_multiselect" not in st.session_state:
        st.session_state.ticker_multiselect = ["AAPL", "MSFT", "GOOGL"]

    selected = st.multiselect(
        "Choose from popular tickers or type to add custom",
        options=ALL_PRESETS,
        key="ticker_multiselect"
    )
    
    # Custom ticker entry
    custom_ticker = st.text_input(
        "Add custom ticker", 
        placeholder="e.g. NVDA, BP.L",
        help="Press Enter to add. Use Yahoo Finance format."
    )
    if custom_ticker and custom_ticker.upper() not in selected:
        selected = selected + [custom_ticker.upper().strip()]
    ## this makes user typed upper case, checks not already selected then adds to selected and strips white space
    
    st.session_state.selected_tickers = selected
    tickers = selected
    
    st.divider()
    
    # Weight allocation
    st.subheader("Allocation")

    if len(tickers) == 0:
        st.warning("Select at least one ticker to continue")
        st.stop()

    weight_mode = st.radio(
        "Method",
        ["Equal weight", "Custom percentages", "By £ amount"],
        horizontal=True
    )

    if weight_mode == "Equal weight":
        weights = np.array([1/len(tickers)] * len(tickers))
        ## this is the simple equal weights

    elif weight_mode == "Custom percentages":
        st.caption("Drag any slider — others auto-rebalance to keep total at 100%")
        
        # State key tied to current ticker set
        state_key = f"pct_weights_{'_'.join(tickers)}"
        
        # weights for this ticker set
        if state_key not in st.session_state:
            st.session_state[state_key] = {t: 100.0/len(tickers) for t in tickers}
        
        # assign session state value for every ticker slider
        for ticker in tickers:
            slider_key = f"slider_{ticker}"
            if slider_key not in st.session_state:
                st.session_state[slider_key] = st.session_state[state_key][ticker]
        
        # when a slider changes, rebalance the others
        def rebalance(changed_ticker, ticker_list):
            old_value = st.session_state[state_key][changed_ticker]
            new_value = st.session_state[f"slider_{changed_ticker}"]
            delta = new_value - old_value
            
            other_total = sum(
                st.session_state[state_key][t] 
                for t in ticker_list if t != changed_ticker
            )
            
            if other_total > 0.01:
                for t in ticker_list:
                    if t != changed_ticker:
                        old_t = st.session_state[state_key][t]
                        new_t = max(0, old_t - delta * (old_t / other_total))
                        st.session_state[state_key][t] = new_t
                        st.session_state[f"slider_{t}"] = new_t
            
            st.session_state[state_key][changed_ticker] = new_value
            ## this is similar to normalising:
            # 1. see how much the slider has changed
            # 2. adjust the other two sliders by that same amount (delta) however adjust 
            #    to be proportional to the current weight of the other two sliders
        
        # now must visually show the process
        for ticker in tickers:
            st.slider(
                ticker,
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                format="%.0f%%",
                key=f"slider_{ticker}",
                on_change=rebalance,
                args=(ticker, tickers)
            )
        ## for each ticker, call the rebalance function when there is a change, passing in the ticker name and ticker list
        
        ## Read final values from session state
        raw = np.array([st.session_state[state_key][t] for t in tickers])
        weights = raw / 100

    else:  ## By £ amount - this is self explanatory
        st.caption("Enter the amount you'd allocate to each asset")
        
        total_investment = st.number_input(
            "Total portfolio value (£)",
            value=10000.0,
            step=1000.0,
            min_value=0.0
        )
        
        amounts = {}
        for ticker in tickers:
            amounts[ticker] = st.number_input(
                f"{ticker}",
                value=total_investment / len(tickers),
                step=100.0,
                min_value=0.0,
                key=f"amount_{ticker}"
            )
        
        total_alloc = sum(amounts.values())
        
        if total_alloc > 0:
            weights = np.array([amounts[t] / total_alloc for t in tickers])
        else:
            weights = np.array([1/len(tickers)] * len(tickers))
        
        if abs(total_alloc - total_investment) > 1:
            st.caption(f"Allocated: £{total_alloc:,.0f} / £{total_investment:,.0f}")

    # Live pie chart - updating
    st.divider()
    st.caption("**Current Allocation**")

    pie_fig = plot_weights(tickers, weights)
    st.plotly_chart(pie_fig, use_container_width=True)

    ## since streamlit reruns at every interaction, no need for fancy piechart, it will just rerun and update

    
    # Date range  
    st.subheader("Time Period")
    period = st.select_slider(
        "Lookback",
        options=["6M", "1Y", "2Y", "3Y", "5Y", "10Y"],
        value="3Y"
    )
    period_days = {"6M": 180, "1Y": 365, "2Y": 730, "3Y": 1095, 
                   "5Y": 1825, "10Y": 3650}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days[period])
    
    st.caption(f"{start_date.strftime('%d %b %Y')} → {end_date.strftime('%d %b %Y')}")
    
    st.divider()
    
    # Benchmark and risk-free rate
    st.subheader("Settings")
    benchmark = st.selectbox(
        "Benchmark",
        options=["SPY", "QQQ", "ISF.L", "VWRL.L", "EFA"],
        index=0
    )
    
    risk_free_rate = st.number_input(
        "Risk-free rate", 
        value=0.04, 
        step=0.005, 
        format="%.3f",
        help="Annualised, e.g. 0.04 = 4%"
    )
    
    st.divider()
    
    analyse_button = st.button(
        "Analyse Portfolio",
        type="primary",
        use_container_width=True
    )

## sidebar end

tab1, tab2, tab3 = st.tabs(["Risk Dashboard", "Sleep Test", "Scenarios"])

if analyse_button:
    with st.spinner("Fetching data..."):
        try:
            prices = fetch_prices(tickers, start_date, end_date)
            returns = calculate_returns(prices)
            port_ret = portfolio_returns(returns, weights)

            bench_prices = fetch_prices(benchmark, start_date, end_date)
            bench_ret = calculate_returns(bench_prices).iloc[:, 0]

            # Store in session state so data persists across reruns
            st.session_state.port_ret = port_ret
            st.session_state.returns = returns
            st.session_state.bench_ret = bench_ret
            st.session_state.analysis_weights = weights
            st.session_state.analysis_tickers = tickers
            st.session_state.analysis_done = True
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()

# Check if we have analysis data (either just computed or from previous run)
if st.session_state.get('analysis_done', False):
    port_ret = st.session_state.port_ret
    returns = st.session_state.returns
    bench_ret = st.session_state.bench_ret

    with tab1:
        ## Performance metrics
        st.header("Key Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Annualised Return", f"{annualised_return(port_ret):.2%}")
        c2.metric("Annualised Volatility", f"{annualised_volatility(port_ret):.2%}")
        c3.metric("Sharpe Ratio", f"{sharpe_ratio(port_ret, risk_free_rate):.2f}")
        c4.metric("Max Drawdown", f"{max_drawdown(port_ret):.2%}")
        
        ## Risk metrics row - same as above
        st.header("Risk Metrics")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("95% VaR (daily)", f"{value_at_risk(port_ret):.2%}")
        c6.metric("95% CVaR (daily)", f"{conditional_value_at_risk(port_ret):.2%}")
        c7.metric("Sortino Ratio", f"{sortino_ratio(port_ret, risk_free_rate):.2f}")
        c8.metric("Beta vs Benchmark", f"{beta(port_ret, bench_ret):.2f}")
        
        ## Performance chart
        st.header("Performance")
        st.plotly_chart(plot_cumulative_portfolio_returns(port_ret, bench_ret), use_container_width=True)
        ## use_container_width=True tells streamlit to fill available space

        ## Risk charts
        st.header("Risk Analysis")
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(plot_drawdown(port_ret), use_container_width=True)
        with col_b:
            st.plotly_chart(plot_return_distribution(port_ret), use_container_width=True)
        
        ## Rolling Sharpe
        st.plotly_chart(plot_rolling_sharpe(port_ret), use_container_width=True)
        
        ## Composition charts side by side
        st.header("Portfolio Composition")
        col_c, col_d = st.columns(2)
        with col_c:
            st.plotly_chart(plot_weights(tickers, weights), use_container_width=True)
        with col_d:
            st.plotly_chart(plot_correlation_heatmap(returns), use_container_width=True)

    with tab2:
        st.header("Sleep Test")
        st.caption("How emotionally tough would this portfolio be to hold?")

        # Sliders
        col1, col2 = st.columns(2)
        with col1:
            dd_tolerance = st.slider("Max drawdown you'd accept", 5, 50, 20, format="%d%%")
            recovery_tolerance = st.slider("Max recovery period (months)", 1, 36, 12)
            day_shock_tolerance = st.slider("Worst single day you'd accept", 1, 15, 5, format="%d%%")
        with col2:
            frequency_tolerance = st.slider("Panic moments per year you'd tolerate", 0, 10, 2)
            loss_aversion = st.slider("Loss aversion factor", 1.0, 3.0, 2.0, step=0.1)

        dd = drawdown_series(port_ret)
        actual_max_dd = abs(dd.min()) * 100
        actual_worst_day = abs(port_ret.min()) * 100

        # Compute longest recovery: count consecutive days in drawdown
        in_drawdown = (dd < 0).astype(int)
        drawdown_groups = (in_drawdown != in_drawdown.shift()).cumsum()
        longest_dd_days = in_drawdown.groupby(drawdown_groups).sum().max()
        actual_recovery_months = longest_dd_days / 21  # ~21 trading days per month

        # Count panic moments: days with returns below -3%
        panic_days = (port_ret < -0.03).sum()
        years_of_data = len(port_ret) / 252
        panic_moments_per_year = panic_days / years_of_data if years_of_data > 0 else 0

        # Score each dimension as 0-100 (how comfortable the user would be)
        # Use max() to avoid division by zero
        dd_score = min(100, (dd_tolerance / max(actual_max_dd, 0.01)) * 100)
        day_score = min(100, (day_shock_tolerance / max(actual_worst_day, 0.01)) * 100)
        recovery_score = min(100, (recovery_tolerance / max(actual_recovery_months, 0.01)) * 100)
        frequency_score = min(100, (frequency_tolerance / max(panic_moments_per_year, 0.01)) * 100)

        # Loss aversion: higher = losses feel worse = harder to sleep
        # Linear scale from 1.0 (score=100) to 3.0 (score=33)
        loss_aversion_score = 100 - (loss_aversion - 1.0) * 33.33

        composite_score = (dd_score + day_score + recovery_score + frequency_score + loss_aversion_score) / 5

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=composite_score,
            title={'text': "Sleep Test Score"},
            gauge={'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 33], 'color': "lightcoral"},
                    {'range': [33, 67], 'color': "lightyellow"},
                    {'range': [67, 100], 'color': "lightgreen"}]}))
        st.plotly_chart(fig, use_container_width=True)

        # Radar chart
        categories = ['Drawdown', 'Recovery', 'Daily Shock', 'Frequency', 'Loss Aversion']
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[dd_score, recovery_score, day_score, frequency_score, loss_aversion_score],
            theta=categories,
            fill='toself',
            name='Your Portfolio'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[100, 100, 100, 100, 100],
            theta=categories,
            fill='toself',
            name='Ideal',
            opacity=0.3
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
        st.plotly_chart(fig_radar, use_container_width=True)