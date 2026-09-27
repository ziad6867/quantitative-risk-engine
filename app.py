from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

import edhec_risk_kit_129 as erk


st.set_page_config(
    page_title="Interactive LDI and Dynamic Risk Engine",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.12), rgba(255, 0, 127, 0.10));
        border: 1px solid rgba(0, 229, 255, 0.38);
        border-radius: 14px;
        padding: 0.8rem 1rem;
        box-shadow: 0 0 18px rgba(0, 229, 255, 0.10);
    }
    [data-testid="stMetricValue"] {
        color: #00e5ff;
        text-shadow: 0 0 10px rgba(0, 229, 255, 0.45);
    }
    [data-testid="stMetricLabel"] {
        color: #f4f7fb;
        font-weight: 700;
    }
    [data-testid="stMetricDelta"] {
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Stress-testing your portfolio so it does not get completely knocked out in a ranked match against the market."
)


@st.cache_data
def download_prices(tickers, start_date, end_date):
    """Download adjusted close prices for the requested tickers."""
    ticker_list = [ticker.strip().upper() for ticker in tickers.split(",") if ticker.strip()]
    if not ticker_list:
        return pd.DataFrame()

    prices = yf.download(
        ticker_list,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if prices.empty or "Adj Close" not in prices:
        return pd.DataFrame()

    prices = prices["Adj Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=ticker_list[0])

    return prices.dropna(how="any")


with st.sidebar:
    st.header("Market Data")
    ticker_input = st.text_input("Tickers", value="SPY, TLT, GLD")
    start_date = st.date_input("Start Date", value=date(2015, 1, 1))
    end_date = st.date_input("End Date", value=date.today())
    risk_free_rate = st.number_input(
        "Risk-Free Rate (annual)",
        min_value=0.0,
        max_value=1.0,
        value=0.03,
        step=0.005,
        format="%.3f",
    )


tabs = st.tabs(["📈 Efficient Frontier", "🎢 LDI Simulator", "🩸 Tail Risk", "🛡️ CPPI Armor"])

with tabs[0]:
    st.title("Efficient Frontier")

    if start_date >= end_date:
        st.error("Start Date must be earlier than End Date.")
    else:
        prices = download_prices(ticker_input, start_date, end_date)

        if prices.empty or prices.shape[1] < 1:
            st.warning("No adjusted close data was returned for the selected inputs.")
        else:
            daily_returns = prices.pct_change().dropna(how="any")

            if daily_returns.empty:
                st.warning("Not enough price history is available to calculate returns.")
            else:
                annualized_returns = (1 + daily_returns).prod() ** (
                    252 / len(daily_returns)
                ) - 1
                covariance_matrix = daily_returns.cov() * 252

                rng = np.random.default_rng(42)
                weights = rng.dirichlet(
                    np.ones(len(annualized_returns)), size=500
                )
                portfolio_returns = weights @ annualized_returns.to_numpy()
                portfolio_volatility = np.sqrt(
                    np.einsum(
                        "ij,jk,ik->i",
                        weights,
                        covariance_matrix.to_numpy(),
                        weights,
                    )
                )
                sharpe_ratios = np.divide(
                    portfolio_returns - risk_free_rate,
                    portfolio_volatility,
                    out=np.zeros_like(portfolio_returns),
                    where=portfolio_volatility != 0,
                )

                frontier_data = pd.DataFrame(
                    {
                        "Volatility": portfolio_volatility,
                        "Return": portfolio_returns,
                        "Sharpe Ratio": sharpe_ratios,
                    }
                )
                fig = px.scatter(
                    frontier_data,
                    x="Volatility",
                    y="Return",
                    color="Sharpe Ratio",
                    color_continuous_scale=["#ff007f", "#00e5ff", "#39ff14"],
                    hover_data={
                        "Volatility": ":.2%",
                        "Return": ":.2%",
                        "Sharpe Ratio": ":.3f",
                    },
                    title="500 Random Portfolios",
                )
                fig.update_layout(
                    template="plotly_dark",
                    xaxis_title="Volatility",
                    yaxis_title="Annualized Return",
                )

                er = annualized_returns.to_numpy()
                cov = covariance_matrix.to_numpy()
                msr_weights = erk.msr(risk_free_rate, er, cov)
                gmv_weights = erk.gmv(cov)

                msr_return = float(erk.portfolio_return(msr_weights, er))
                msr_volatility = float(erk.portfolio_vol(msr_weights, cov))
                gmv_return = float(erk.portfolio_return(gmv_weights, er))
                gmv_volatility = float(erk.portfolio_vol(gmv_weights, cov))

                fig.add_scatter(
                    x=[gmv_volatility],
                    y=[gmv_return],
                    mode="markers",
                    name="GMV",
                    marker={
                        "color": "red",
                        "size": 18,
                        "symbol": "star",
                        "line": {"color": "white", "width": 1},
                    },
                    hovertemplate="GMV<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
                )
                fig.add_scatter(
                    x=[msr_volatility],
                    y=[msr_return],
                    mode="markers",
                    name="MSR",
                    marker={
                        "color": "gold",
                        "size": 18,
                        "symbol": "star",
                        "line": {"color": "white", "width": 1},
                    },
                    hovertemplate="MSR<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Five hundred possible futures, lined up like tiny portfolio contenders.")

                msr_data = pd.DataFrame(
                    {"Asset": annualized_returns.index, "Weight": msr_weights}
                ).set_index("Asset")
                gmv_data = pd.DataFrame(
                    {"Asset": annualized_returns.index, "Weight": gmv_weights}
                ).set_index("Asset")

                msr_column, gmv_column = st.columns(2)
                with msr_column:
                    st.subheader("Maximum Sharpe Ratio (MSR)")
                    st.metric("Expected Return", f"{msr_return:.2%}")
                    st.metric("Volatility", f"{msr_volatility:.2%}")
                    st.dataframe(
                        msr_data.style.format("{:.4%}"),
                        use_container_width=True,
                    )
                with gmv_column:
                    st.subheader("Global Minimum Variance (GMV)")
                    st.metric("Expected Return", f"{gmv_return:.2%}")
                    st.metric("Volatility", f"{gmv_volatility:.2%}")
                    st.dataframe(
                        gmv_data.style.format("{:.4%}"),
                        use_container_width=True,
                    )

with tabs[1]:
    st.title("Liability-Driven Investing Simulator")

    ldi_tickers = [
        ticker.strip().upper()
        for ticker in ticker_input.split(",")
        if ticker.strip()
    ]

    if start_date >= end_date:
        st.error("Start Date must be earlier than End Date.")
    elif len(ldi_tickers) < 2:
        st.warning("Enter at least two tickers: one growth asset and one hedging asset.")
    else:
        ldi_prices = download_prices(ticker_input, start_date, end_date)
        ldi_assets = [ticker for ticker in ldi_tickers if ticker in ldi_prices.columns]

        if ldi_prices.empty or len(ldi_assets) < 2:
            st.warning("No usable adjusted close data was returned for the first two tickers.")
        else:
            ldi_returns = ldi_prices[ldi_assets[:2]].pct_change().dropna(how="any")

            if ldi_returns.empty:
                st.warning("Not enough price history is available to run the LDI simulation.")
            else:
                starting_value = 1_000_000
                simulated_values = (1 + ldi_returns).cumprod() * starting_value
                simulated_values.columns = ["Asset Value", "Liability Value"]
                simulated_values.index.name = "Date"
                funding_ratio = simulated_values["Asset Value"] / simulated_values[
                    "Liability Value"
                ]

                current_asset_value = simulated_values["Asset Value"].iloc[-1]
                current_liability_value = simulated_values["Liability Value"].iloc[-1]
                current_funding_ratio = funding_ratio.iloc[-1]

                asset_metric, liability_metric, funding_metric = st.columns(3)
                with asset_metric:
                    st.metric("Current Asset Value", f"${current_asset_value:,.0f}")
                with liability_metric:
                    st.metric("Current Liability Value", f"${current_liability_value:,.0f}")
                with funding_metric:
                    st.metric(
                        "Live Funding Ratio",
                        f"{current_funding_ratio:.2%}",
                        delta=f"{current_funding_ratio - 1:.2%}",
                        delta_color="normal",
                    )

                value_data = simulated_values.reset_index().melt(
                    id_vars="Date",
                    var_name="Series",
                    value_name="Value",
                )
                value_fig = px.line(
                    value_data,
                    x="Date",
                    y="Value",
                    color="Series",
                    color_discrete_map={
                        "Asset Value": "#00e5ff",
                        "Liability Value": "#ff007f",
                    },
                    title="Simulated Asset and Liability Values",
                )
                value_fig.update_layout(
                    template="plotly_dark",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    legend_title="",
                )
                st.plotly_chart(value_fig, use_container_width=True)
                st.caption("Blue is your growth engine; pink is the liability shadow keeping score.")

                funding_data = funding_ratio.rename("Funding Ratio").reset_index()
                funding_fig = px.line(
                    funding_data,
                    x="Date",
                    y="Funding Ratio",
                    title="Historical Funding Ratio",
                )
                funding_fig.add_hline(
                    y=1.0,
                    line_color="#ff0000",
                    line_dash="dash",
                    annotation_text="Fully Funded (100%)",
                    annotation_position="top left",
                )
                funding_fig.update_layout(
                    template="plotly_dark",
                    xaxis_title="Date",
                    yaxis_title="Funding Ratio",
                )
                st.plotly_chart(funding_fig, use_container_width=True)
                st.caption("Above the red line is fully funded. Below it, the portfolio starts sweating.")

with tabs[2]:
    st.title("Tail Risk and Drawdowns")

    tail_tickers = [
        ticker.strip().upper()
        for ticker in ticker_input.split(",")
        if ticker.strip()
    ]

    if start_date >= end_date:
        st.error("Start Date must be earlier than End Date.")
    elif not tail_tickers:
        st.warning("Enter at least one ticker to analyze tail risk.")
    else:
        tail_prices = download_prices(ticker_input, start_date, end_date)
        growth_ticker = tail_tickers[0]

        if tail_prices.empty or growth_ticker not in tail_prices.columns:
            st.warning("No usable adjusted close data was returned for the growth asset.")
        else:
            growth_returns = tail_prices[growth_ticker].pct_change().dropna()

            if growth_returns.empty:
                st.warning("Not enough price history is available for tail risk analysis.")
            else:
                historic_var = erk.var_historic(growth_returns, level=5)
                cornish_fisher_var = erk.var_gaussian(
                    growth_returns,
                    level=5,
                    modified=True,
                )
                historic_cvar = erk.cvar_historic(growth_returns, level=5)

                historic_var_column, cornish_fisher_column, cvar_column = st.columns(3)
                with historic_var_column:
                    st.metric("5% Historic VaR", f"{historic_var:.2%}")
                with cornish_fisher_column:
                    st.metric("5% Cornish-Fisher VaR", f"{cornish_fisher_var:.2%}")
                with cvar_column:
                    st.metric("5% Historic CVaR", f"{historic_cvar:.2%}")

                wealth_index = (1 + growth_returns).cumprod()
                previous_peaks = wealth_index.cummax()
                drawdowns = (wealth_index - previous_peaks) / previous_peaks
                drawdown_data = drawdowns.rename("Drawdown").reset_index()
                drawdown_data.columns = ["Date", "Drawdown"]

                drawdown_fig = px.area(
                    drawdown_data,
                    x="Date",
                    y="Drawdown",
                    title=f"{growth_ticker} Underwater Plot",
                )
                drawdown_fig.update_traces(
                    fillcolor="rgba(255, 0, 0, 0.55)",
                    line_color="#ff0000",
                )
                drawdown_fig.update_layout(
                    template="plotly_dark",
                    xaxis_title="Date",
                    yaxis_title="Drawdown",
                    yaxis_tickformat=".0%",
                )
                st.plotly_chart(drawdown_fig, use_container_width=True)
                st.caption("Every time the asset hit the deck hard. Ouch.")

                return_data = growth_returns.rename("Daily Return").reset_index(drop=True)
                histogram_fig = px.histogram(
                    return_data,
                    x="Daily Return",
                    nbins=75,
                    title=f"{growth_ticker} Daily Return Distribution",
                )
                histogram_fig.update_layout(
                    template="plotly_dark",
                    xaxis_title="Daily Return",
                    yaxis_title="Frequency",
                )
                histogram_fig.update_traces(marker_color="#00e5ff")
                st.plotly_chart(histogram_fig, use_container_width=True)
                st.caption("The tails are where the market keeps its jump scares.")

with tabs[3]:
    st.title("CPPI Engine")

    multiplier = st.slider(
        "Multiplier (m)",
        min_value=1.0,
        max_value=7.0,
        value=3.0,
        step=0.1,
    )
    floor = st.slider(
        "Floor",
        min_value=0.50,
        max_value=0.99,
        value=0.80,
        step=0.01,
        format="%.2f",
    )
    trend_enabled = st.toggle("Enable Trend-Following Signal (SMA 50/200)")

    cppi_tickers = [
        ticker.strip().upper()
        for ticker in ticker_input.split(",")
        if ticker.strip()
    ]

    if start_date >= end_date:
        st.error("Start Date must be earlier than End Date.")
    elif len(cppi_tickers) < 2:
        st.warning("Enter at least two tickers: one growth asset and one hedging asset.")
    else:
        cppi_prices = download_prices(ticker_input, start_date, end_date)
        cppi_assets = [ticker for ticker in cppi_tickers if ticker in cppi_prices.columns]

        if cppi_prices.empty or len(cppi_assets) < 2:
            st.warning("No usable adjusted close data was returned for the CPPI assets.")
        else:
            returns = cppi_prices[cppi_assets[:2]].pct_change()

            # 1. Grab only the first two columns and drop any row with a missing value.
            # This guarantees perfect index alignment and the exact same length.
            cppi_data = returns.iloc[:, :2].dropna(how="any")

            # 2. Keep both inputs as 2D DataFrames for the EDHEC engine.
            # Both columns must share the same label because run_cppi combines
            # their one-row values into a single-column history.
            risky_r = cppi_data.iloc[:, [0]].copy()
            safe_r = cppi_data.iloc[:, [1]].copy()
            risky_r.columns = ["R"]
            safe_r.columns = ["R"]
            starting_wealth = 1_000_000

            if cppi_data.empty:
                st.warning("Not enough price history is available to run the CPPI engine.")
            elif hasattr(erk, "run_cppi") and not trend_enabled:
                cppi_result = erk.run_cppi(
                    risky_r,
                    safe_r=safe_r,
                    m=multiplier,
                    start=starting_wealth,
                    floor=floor,
                )
                cppi_wealth = cppi_result["Wealth"].iloc[:, 0]
                risky_wealth = cppi_result["Risky Wealth"].iloc[:, 0]
                floor_value = starting_wealth * floor

                wealth_data = pd.DataFrame(
                    {
                        "CPPI Strategy Wealth": cppi_wealth,
                        "100% Risky Asset Wealth": risky_wealth,
                    }
                )
                wealth_data.index.name = "Date"
                wealth_data = wealth_data.reset_index().melt(
                    id_vars="Date",
                    var_name="Strategy",
                    value_name="Wealth",
                )
                wealth_fig = px.line(
                    wealth_data,
                    x="Date",
                    y="Wealth",
                    color="Strategy",
                    color_discrete_map={
                        "CPPI Strategy Wealth": "#39ff14",
                        "100% Risky Asset Wealth": "#ff007f",
                    },
                    title="CPPI Strategy vs. 100% Risky Asset",
                )
                wealth_fig.add_hline(
                    y=floor_value,
                    line_color="yellow",
                    line_dash="dash",
                    annotation_text=f"Floor (${floor_value:,.0f})",
                    annotation_position="top left",
                )
                wealth_fig.update_layout(
                    template="plotly_dark",
                    xaxis_title="Date",
                    yaxis_title="Wealth ($)",
                    legend_title="",
                )
                st.plotly_chart(wealth_fig, use_container_width=True)
                st.caption("Armor on: the strategy leans into upside while keeping one eye on the floor.")
            else:
                account_value = starting_wealth
                floor_value = starting_wealth * floor
                cppi_history = []
                risky_series = risky_r.iloc[:, 0]
                safe_series = safe_r.iloc[:, 0]

                if trend_enabled:
                    risky_price = (1 + risky_r).cumprod()
                    sma50 = risky_price.rolling(50).mean()
                    sma200 = risky_price.rolling(200).mean()

                for i, timestamp in enumerate(risky_series.index):
                    if trend_enabled:
                        if i == 0:
                            signal_multiplier = multiplier
                        else:
                            previous_sma50 = sma50.iloc[i - 1, 0]
                            previous_sma200 = sma200.iloc[i - 1, 0]
                            signal_multiplier = (
                                5.0
                                if pd.notna(previous_sma50)
                                and pd.notna(previous_sma200)
                                and previous_sma50 >= previous_sma200
                                else 1.0
                            )
                    else:
                        signal_multiplier = multiplier

                    cushion = (account_value - floor_value) / account_value
                    risky_weight = min(max(signal_multiplier * cushion, 0.0), 1.0)
                    safe_weight = 1.0 - risky_weight
                    risky_return = float(risky_r.iloc[i, 0])
                    safe_return = float(safe_r.iloc[i, 0])
                    account_value = account_value * (
                        risky_weight * (1 + risky_return)
                        + safe_weight * (1 + safe_return)
                    )
                    cppi_history.append(account_value)

                cppi_wealth = pd.Series(cppi_history, index=risky_series.index)
                risky_wealth = starting_wealth * (1 + risky_series).cumprod()
                wealth_data = pd.DataFrame(
                    {
                        "CPPI Strategy Wealth": cppi_wealth,
                        "100% Risky Asset Wealth": risky_wealth,
                    }
                )
                wealth_data.index.name = "Date"
                wealth_data = wealth_data.reset_index().melt(
                    id_vars="Date",
                    var_name="Strategy",
                    value_name="Wealth",
                )
                wealth_fig = px.line(
                    wealth_data,
                    x="Date",
                    y="Wealth",
                    color="Strategy",
                    color_discrete_map={
                        "CPPI Strategy Wealth": "#39ff14",
                        "100% Risky Asset Wealth": "#ff007f",
                    },
                    title="CPPI Strategy vs. 100% Risky Asset",
                )
                wealth_fig.add_hline(
                    y=floor_value,
                    line_color="yellow",
                    line_dash="dash",
                    annotation_text=f"Floor (${floor_value:,.0f})",
                    annotation_position="top left",
                )
                wealth_fig.update_layout(
                    template="plotly_dark",
                    xaxis_title="Date",
                    yaxis_title="Wealth ($)",
                    legend_title="",
                )
                st.plotly_chart(wealth_fig, use_container_width=True)
                st.caption("Signal mode engaged: the armor changes stance when the trend gets moody.")
