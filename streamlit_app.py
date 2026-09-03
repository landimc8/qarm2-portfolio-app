"""
QARM II - Example Portfolio Optimization App
==============================================
This demonstrates the core theory from Lecture Notes 1-2 (Theory of
portfolio optimization / efficient frontier) as an interactive Streamlit app.

Concepts used (straight from the slides):
    - Expected returns (mu) and covariance matrix (Sigma) of asset returns
    - Markowitz mean-variance optimization:
          min  x' Sigma x        (portfolio variance)
          s.t. x' mu = target_return
               sum(x) = 1
               (optionally: x >= 0, i.e. long-only / constrained)
    - The efficient frontier: sweeping target_return and re-solving
    - The Sharpe ratio and the max-Sharpe (tangency) portfolio

This is a STARTER template - swap the tickers / add your client's
constraints and data source to turn it into your actual project.
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import minimize

st.set_page_config(page_title="QARM II Portfolio Optimizer", layout="wide")
st.title("QARM II — Mean-Variance Portfolio Optimizer")
st.caption("Markowitz efficient frontier, built on the theory from LN1-2")

# ---------------------------------------------------------------
# 1. Sidebar inputs (this is where your "client" constraints go)
# ---------------------------------------------------------------
st.sidebar.header("Settings")

tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated)", value="AAPL, MSFT, JPM, XOM, GLD"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime("today"))

risk_free_rate = st.sidebar.number_input(
    "Risk-free rate (annual, %)", value=2.0, step=0.1
) / 100

long_only = st.sidebar.checkbox("Long-only (no short-selling)", value=True)

n_points = st.sidebar.slider("Number of frontier points", 10, 100, 40)

# ---------------------------------------------------------------
# 2. Pull data and compute mu / Sigma
# ---------------------------------------------------------------
@st.cache_data
def load_data(tickers, start, end):
    prices = yf.download(tickers, start=start, end=end)["Close"]
    returns = prices.pct_change().dropna()
    return prices, returns

if len(tickers) < 2:
    st.warning("Enter at least two tickers to build a portfolio.")
    st.stop()

with st.spinner("Downloading price data..."):
    prices, returns = load_data(tickers, start_date, end_date)

mu = returns.mean() * 252            # annualized expected returns
Sigma = returns.cov() * 252          # annualized covariance matrix
n_assets = len(tickers)

# ---------------------------------------------------------------
# 3. Markowitz optimization (the quadratic problem from the slides)
# ---------------------------------------------------------------
def portfolio_variance(w, Sigma):
    return w @ Sigma @ w

def solve_min_variance(target_return, mu, Sigma, long_only):
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: w @ mu - target_return},
    ]
    bounds = [(0, 1) for _ in range(n_assets)] if long_only else None
    w0 = np.ones(n_assets) / n_assets
    result = minimize(
        portfolio_variance, w0, args=(Sigma,),
        method="SLSQP", bounds=bounds, constraints=constraints,
    )
    return result.x if result.success else None

# Sweep target returns to trace the efficient frontier
target_returns = np.linspace(mu.min(), mu.max(), n_points)
frontier_vol, frontier_ret, frontier_weights = [], [], []

for target in target_returns:
    w = solve_min_variance(target, mu.values, Sigma.values, long_only)
    if w is not None:
        frontier_vol.append(np.sqrt(portfolio_variance(w, Sigma.values)))
        frontier_ret.append(target)
        frontier_weights.append(w)

frontier_vol = np.array(frontier_vol)
frontier_ret = np.array(frontier_ret)

# Max Sharpe ratio portfolio (tangency portfolio)
sharpe_ratios = (frontier_ret - risk_free_rate) / frontier_vol
max_sharpe_idx = np.argmax(sharpe_ratios)

# Minimum variance portfolio
min_var_idx = np.argmin(frontier_vol)

# ---------------------------------------------------------------
# 4. Display: efficient frontier plot
# ---------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier_vol, y=frontier_ret, mode="lines",
        name="Efficient Frontier", line=dict(color="royalblue", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=[frontier_vol[max_sharpe_idx]], y=[frontier_ret[max_sharpe_idx]],
        mode="markers", name="Max Sharpe Portfolio",
        marker=dict(color="red", size=12, symbol="star"),
    ))
    fig.add_trace(go.Scatter(
        x=[frontier_vol[min_var_idx]], y=[frontier_ret[min_var_idx]],
        mode="markers", name="Min Variance Portfolio",
        marker=dict(color="green", size=10),
    ))
    # individual assets for reference
    asset_vol = np.sqrt(np.diag(Sigma))
    fig.add_trace(go.Scatter(
        x=asset_vol, y=mu, mode="markers+text", name="Individual Assets",
        text=tickers, textposition="top center",
        marker=dict(color="grey", size=8),
    ))
    fig.update_layout(
        xaxis_title="Volatility (annualized)",
        yaxis_title="Expected Return (annualized)",
        height=550,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Max Sharpe Portfolio")
    st.metric("Expected Return", f"{frontier_ret[max_sharpe_idx]:.2%}")
    st.metric("Volatility", f"{frontier_vol[max_sharpe_idx]:.2%}")
    st.metric("Sharpe Ratio", f"{sharpe_ratios[max_sharpe_idx]:.2f}")

    weights_df = pd.DataFrame({
        "Ticker": tickers,
        "Weight": frontier_weights[max_sharpe_idx],
    }).sort_values("Weight", ascending=False)
    st.dataframe(weights_df.style.format({"Weight": "{:.1%}"}), hide_index=True)

st.divider()
st.caption(
    "Next steps for your project: replace this generic asset universe with "
    "your client's actual data, add their specific constraints (e.g. sector "
    "caps, ESG limits), and pick the optimization technique that fits their "
    "profile (constrained MVO, Black-Litterman, risk budgeting, etc.)."
)