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
    st.caption("↓ Scroll down to customise portfolio and re-analyse")
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

    st.caption("Click below to update with your selections")
    analyse_button = st.button(
        "Analyse Portfolio",
        type="primary",
        use_container_width=True
    )

## sidebar end

tab1, tab2, tab3 = st.tabs(["Risk Dashboard", "Sleep Test", "Scenarios"])

## Auto-analyse on first load, or when button is clicked
first_load = 'analysis_done' not in st.session_state

if analyse_button or first_load:
    with st.spinner("Fetching data..." if not first_load else "Loading default portfolio..."):
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

        ## Sliders
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

        ## Compute longest recovery: count consecutive days in drawdown
        ## this probably should have been done in metrics
        in_drawdown = (dd < 0).astype(int)
        drawdown_groups = (in_drawdown != in_drawdown.shift()).cumsum()
        longest_dd_days = in_drawdown.groupby(drawdown_groups).sum().max()
        actual_recovery_months = longest_dd_days / 21  # ~21 trading days per month

        ## Count panic moments: days with returns below -3% (arbitrary value)
        panic_days = (port_ret < -0.03).sum()
        years_of_data = len(port_ret) / 252
        panic_moments_per_year = panic_days / years_of_data if years_of_data > 0 else 0

        # Score each dimension as 0-100 - this is just a ratio
        ## Use max() to avoid division by zero
        dd_score = min(100, (dd_tolerance / max(actual_max_dd, 0.01)) * 100)
        day_score = min(100, (day_shock_tolerance / max(actual_worst_day, 0.01)) * 100)
        recovery_score = min(100, (recovery_tolerance / max(actual_recovery_months, 0.01)) * 100)
        frequency_score = min(100, (frequency_tolerance / max(panic_moments_per_year, 0.01)) * 100)

        # Linear scale from 1.0 (score=100) to 3.0 (score=33)
        ## this is possibly the least logical - could be removed
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

        ## chart
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

    with tab3:
        st.header("Stress Test Scenarios")
        st.caption("Simulate factor shocks and see how your portfolio would behave through a crisis")

        ## Factor Shocks Section
        st.subheader("1. Define Factor Shocks")
        st.caption("Shock the underlying market factors - impacts propagate to assets based on their sensitivities")

        factor_cols = st.columns(3)
        with factor_cols[0]:
            market_shock = st.slider("Market Shock", -60, 0, -30, format="%d%%",
                                     help="Broad equity market decline (e.g., -30% = 2008-style)")
        with factor_cols[1]:
            rate_shock = st.slider("Interest Rate Shock (bps)", 0, 300, 100,
                                   help="Basis points increase (e.g., 200bps = aggressive Fed)")
        with factor_cols[2]:
            vix_spike = st.slider("Volatility Spike (VIX)", 20, 80, 40,
                                  help="VIX level during crisis (normal ~15, 2008 peak ~80)")

        st.divider()

        ## Drawdown Shape Section - to mimic old crises
        st.subheader("2. Select Crisis Shape")
        st.caption("How does the crisis unfold over time?")

        shape = st.radio(
            "Scenario Shape",
            ["V-Shape (Fast crash, fast recovery)",
             "U-Shape (Crash, long bottom, slow recovery)",
             "L-Shape (Crash, no recovery)"],
            horizontal=True
        )

        ## Set durations based on shape - historic , as array
        if "V-Shape" in shape:
            crash_weeks, bottom_weeks, recovery_weeks = 4, 0, 12
            shape_desc = "Sharp decline followed by quick recovery (e.g., COVID March 2020)"
        elif "U-Shape" in shape:
            crash_weeks, bottom_weeks, recovery_weeks = 8, 16, 24
            shape_desc = "Decline, extended bottom, gradual recovery (e.g., 2008 Financial Crisis)"
        else:  # L-Shape
            crash_weeks, bottom_weeks, recovery_weeks = 8, 52, 0
            shape_desc = "Decline with no meaningful recovery (e.g., Japan 1990s)"

        st.caption(f"*{shape_desc}*")

        # Show timeline preview based on selection
        st.caption(f"Timeline: {crash_weeks}wk crash → {bottom_weeks}wk bottom → {recovery_weeks}wk recovery")

        st.divider()

        ## Calculate factor exposures and simulate
        st.subheader("3. Scenario Results")

        ## Calculate market beta for each asset using beta func
        asset_betas = {}
        asset_shocks = {}

        for ticker in st.session_state.analysis_tickers:
            if ticker in returns.columns:
                asset_ret = returns[ticker].dropna()

                ## call beta function
                asset_betas[ticker] = beta(asset_ret, bench_ret)

                ## Estimate rate sensitivity (simplified: higher vol = more rate sensitive)
                vol = asset_ret.std() * np.sqrt(252)
                rate_sensitivity = -vol * 0.5

                ## Calculate total shock for this asset
                market_impact = asset_betas[ticker] * (market_shock / 100)
                rate_impact = rate_sensitivity * (rate_shock / 10000)
                vix_impact = -0.002 * (vix_spike - 20)

                asset_shocks[ticker] = (market_impact + rate_impact + vix_impact) * 100

        ## Portfolio-level shock (weighted by portfolio weights)
        analysis_weights = st.session_state.analysis_weights
        portfolio_shock = sum(
            asset_shocks.get(ticker, market_shock) * weight
            for ticker, weight in zip(st.session_state.analysis_tickers, analysis_weights)
        )

        ## Display asset-level impacts
        st.markdown("**Factor Exposures & Estimated Impacts**")

        impact_data = []
        for ticker, weight in zip(st.session_state.analysis_tickers, analysis_weights):
            impact_data.append({
                "Asset": ticker,
                "Weight": f"{weight*100:.1f}%",
                "Beta": f"{asset_betas.get(ticker, 1.0):.2f}",
                "Est. Shock": f"{asset_shocks.get(ticker, market_shock):.1f}%",
                "Contribution": f"{asset_shocks.get(ticker, market_shock) * weight:.1f}%"
            })

        st.dataframe(pd.DataFrame(impact_data), use_container_width=True, hide_index=True)

        ## Key stats
        stat_cols = st.columns(4)
        stat_cols[0].metric("Portfolio Max Drawdown", f"{portfolio_shock:.1f}%")
        stat_cols[1].metric("Time to Bottom", f"{crash_weeks} weeks")
        stat_cols[2].metric("Time at Bottom", f"{bottom_weeks} weeks")
        stat_cols[3].metric("Recovery Time", f"{recovery_weeks} weeks" if recovery_weeks > 0 else "No recovery")

        total_underwater = crash_weeks + bottom_weeks + recovery_weeks
        st.caption(f"**Total time underwater: {total_underwater} weeks ({total_underwater/4:.0f} months)**")

        st.divider()

        ## Build and plot the scenario timeline
        st.subheader("4. Scenario Timeline")

        # Get last 3 months of actual data as "pre-shock" period
        pre_shock_days = min(63, len(port_ret))  # ~3 months
        pre_shock_returns = port_ret.iloc[-pre_shock_days:]
        pre_shock_cumulative = (1 + pre_shock_returns).cumprod()
        pre_shock_cumulative = pre_shock_cumulative / pre_shock_cumulative.iloc[0]  # Normalize to start at 1

        # Build scenario phases (in weekly steps, convert to daily for smoother chart)
        days_per_week = 5

        # Crash phase: linear decline to max drawdown
        crash_days = crash_weeks * days_per_week
        crash_values = np.linspace(1.0, 1.0 + portfolio_shock/100, crash_days)

        # Bottom phase: oscillate around the bottom with some volatility if not V
        bottom_days = bottom_weeks * days_per_week
        if bottom_days > 0:
            np.random.seed(42)  # Fixed seed for reproducibility
            noise = np.random.normal(0, 0.005, bottom_days)
            bottom_base = 1.0 + portfolio_shock/100
            bottom_values = bottom_base + np.cumsum(noise)
            # Keep it oscillating around the bottom, not drifting
            bottom_values = bottom_values - (bottom_values - bottom_base).mean()
        else:
            bottom_values = np.array([])

        ## Recovery phase: gradual return toward original value
        recovery_days = recovery_weeks * days_per_week
        if recovery_days > 0:
            recovery_start = bottom_values[-1] if len(bottom_values) > 0 else (1.0 + portfolio_shock/100)
            recovery_end = 1.0  # Full recovery
            recovery_values = np.linspace(recovery_start, recovery_end, recovery_days)
        else:
            recovery_values = np.array([])

        # Combine
        scenario_values = np.concatenate([crash_values, bottom_values, recovery_values])

        # Scale scenario to connect with pre-shock data
        scenario_scaled = scenario_values * pre_shock_cumulative.iloc[-1]

        # Create date index for scenario (continuing from last date)
        last_date = pre_shock_cumulative.index[-1]
        scenario_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                        periods=len(scenario_scaled), freq='B')

        fig_scenario = go.Figure()

        # Pre-shock actual data
        fig_scenario.add_trace(go.Scatter(
            x=pre_shock_cumulative.index,
            y=pre_shock_cumulative.values,
            mode='lines',
            name='Actual (Pre-Shock)',
            line=dict(color='blue', width=2)
        ))

        # Scenario projection
        fig_scenario.add_trace(go.Scatter(
            x=scenario_dates,
            y=scenario_scaled,
            mode='lines',
            name='Scenario Projection',
            line=dict(color='red', width=2, dash='dot')
        ))

        ## Add annotations
        crash_end_date = scenario_dates[crash_days-1] if crash_days > 0 else last_date
        bottom_end_idx = crash_days + bottom_days - 1
        bottom_end_date = scenario_dates[bottom_end_idx] if bottom_days > 0 else crash_end_date

        ## Shade the crash phase
        fig_scenario.add_vrect(
            x0=last_date, x1=crash_end_date,
            fillcolor="red", opacity=0.1,
            annotation_text="Crash", annotation_position="top left"
        )

        if bottom_days > 0:
            fig_scenario.add_vrect(
                x0=crash_end_date, x1=bottom_end_date,
                fillcolor="orange", opacity=0.1,
                annotation_text="Bottom", annotation_position="top left"
            )

        if recovery_days > 0:
            fig_scenario.add_vrect(
                x0=bottom_end_date, x1=scenario_dates[-1],
                fillcolor="green", opacity=0.1,
                annotation_text="Recovery", annotation_position="top left"
            )

        fig_scenario.update_layout(
            title="Portfolio Value Through Simulated Crisis",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (Normalized)",
            template="plotly_white",
            hovermode="x unified",
            showlegend=True
        )

        st.plotly_chart(fig_scenario, use_container_width=True)

        ## Summary interpretation
        st.markdown("---")
        st.markdown("**Interpretation**")

        if portfolio_shock > -20:
            severity = "moderate"
        elif portfolio_shock > -40:
            severity = "significant"
        else:
            severity = "severe"

        st.markdown(f"""
        Under this scenario, your portfolio would experience a **{severity}** drawdown of **{portfolio_shock:.1f}%**.

        - **Crash phase**: {crash_weeks} weeks of decline
        - **Bottom phase**: {bottom_weeks} weeks of consolidation
        - **Recovery phase**: {recovery_weeks} weeks to recover {"(full recovery)" if recovery_weeks > 0 else "(no recovery modeled)"}

        The assets most affected would be those with higher market beta. Consider whether you could
        stay invested through {total_underwater} weeks ({total_underwater//4} months) of being underwater.
        """)