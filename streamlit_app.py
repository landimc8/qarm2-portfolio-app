"""
streamlit_app.py

Wires data_loader.py, constraints.py, and optimizer.py into something
the procurement director can actually click around in. Layout: header
-> sidebar constraints -> frontier plot -> allocation -> risk budgeting
-> raw numbers.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import streamlit as st

from data_loader import (
    PORTFOLIO_TICKERS, WASTE_BASED, TICKERS,
    get_returns_for_optimizer, fetch_prices, clean_prices, sanity_check, compute_crack_spread,
)
import pandas as pd
from optimizer import (
    compute_constrained_frontier,
    compute_unconstrained_frontier,
    min_variance_portfolio,
    max_sharpe_equivalent_portfolio,
    risk_contributions,
    sustainability_share,
    pairwise_correlation,
    bucket_split,
)

st.set_page_config(page_title="Rotterdam Biofuels -- Feedstock Hedging", layout="wide")

# tickers alone mean nothing on a chart -- pair each with the real feedstock
# short codes go on the bars; full names live in the key below (see
# render_ticker_key()) so the charts don't get crowded with two-line labels
TICKER_LABELS = {
    "ZL=F": "Soybean Oil",
    "ZS=F": "Soybean (rapeseed proxy)",
    "HO=F": "Diesel (UCO proxy)",
    "LE=F": "Live Cattle (tallow proxy)",
    "DAR": "Darling Ingredients (tallow/UCO proxy)",
}

# fully distinct colors per ticker (not shades of one hue -- too easy to
# confuse at a glance). Bucket grouping (the RED2 story) is carried by the
# feedstock key below, not by the bars themselves -- solid full-color bars
# read better than hatched/textured ones.
TICKER_COLORS = {
    "ZL=F": "#2CA02C",   # green
    "ZS=F": "#1F77B4",   # blue
    "HO=F": "#D62728",   # red
    "LE=F": "#9467BD",   # purple
    "DAR": "#FF7F0E",    # orange
}


TICKER_MD_COLORS = {"ZL=F": "green", "ZS=F": "blue", "HO=F": "red", "LE=F": "violet", "DAR": "orange"}


def render_ticker_key():
    """Color-coded legend matching the chart bars exactly; bucket noted per ticker in words."""
    st.markdown(
        "**Feedstock key** (\u2020 = waste-based, RED2-eligible): "
        + "  \u2003".join(
            f":{TICKER_MD_COLORS[t]}[\u25cf] **{t}**{' \u2020' if t in WASTE_BASED else ''} \u2014 {TICKER_LABELS[t]}"
            for t in PORTFOLIO_TICKERS
        )
    )


# --- header ------------------------------------------------------------

st.title("Rotterdam Biofuels SA -- Feedstock Cost-Risk Hedging")
st.markdown(
    """
    Rotterdam Biofuels SA buys a basket of feedstocks -- used cooking oil
    (UCO), tallow, soybean oil, and rapeseed oil -- and refines them into
    biodiesel and renewable diesel (HVO). Feedstock costs swing significantly
    month to month, squeezing margins. This tool builds a feedstock
    allocation that minimizes cost volatility while respecting three
    business realities: the EU RED2 sustainability minimum for
    waste-based feedstocks, a concentration cap so no single feedstock
    dominates supply, and the procurement budget.
    """
)

with st.expander("A note on the data (basis risk)"):
    st.markdown(
        """
        Public daily price data for UCO and tallow doesn't exist outside
        paid industry services (Argus, Fastmarkets). This tool proxies
        them with liquid, publicly available futures and equities instead:

        - **UCO** is proxied by **HO=F** (NY Harbor ULSD/diesel futures),
          since UCO-based fuel competes directly with diesel.
        - **Tallow** is proxied by **LE=F** (Live Cattle futures) and
          **DAR** (Darling Ingredients equity), since tallow is a cattle
          rendering byproduct and Darling is the largest global processor
          of rendered fats into fuel.
        - The second crop-based feedstock uses **ZS=F** (soybean futures)
          rather than canola/rapeseed futures (RS=F) -- Yahoo Finance
          returned "possibly delisted; no price data found" for RS=F on
          every attempt, confirmed across repeated pulls rather than a
          one-off network issue. This tracks with canola/rapeseed trading
          on ICE Futures Canada rather than a CME Group exchange, where
          Yahoo's futures coverage is noticeably thinner. ZS=F steps in as
          a reliably available, still crop-based substitute.

        This is a deliberate modeling choice, not an oversight -- the gap
        between a hedging proxy and the true underlying exposure is a
        known, accepted limitation in real-world hedging called **basis
        risk**. The optimizer below minimizes variance of the proxy
        basket; true UCO/tallow cost variance would be somewhat higher.

        **One limitation worth flagging directly:** ZL=F and ZS=F are
        both part of the CBOT soybean complex, so the crop-based leg is
        more correlated than a soybean-oil/rapeseed-oil pairing was
        originally intended to be. The live correlation figure below the
        frontier chart reports exactly how correlated the two actually
        are on the current data -- there's no reliably sourced figure for
        what a genuine soybean-oil/rapeseed-oil pair would score, so this
        is reported on its own terms rather than framed as worse than
        that pairing would be.
        """
    )


# --- sidebar -------------------------------------------------------------

st.sidebar.header("Procurement constraints")

sustainability_min = st.sidebar.slider(
    "Minimum waste-based share (RED2 double-counting)",
    min_value=0.0, max_value=0.90, value=0.40, step=0.05,
    help="Combined minimum weight for HO=F, LE=F, and DAR (the UCO/tallow proxies).",
)
concentration_cap = st.sidebar.slider(
    "Maximum weight per feedstock (concentration cap)",
    min_value=0.20, max_value=1.00, value=0.35, step=0.05,
    help="No single feedstock may exceed this share of the procurement budget.",
)
years_lookback = st.sidebar.slider("Years of historical data", 1, 10, 5, 1)
n_frontier_points = st.sidebar.slider("Frontier resolution (points)", 10, 100, 50, 10)
procurement_budget = st.sidebar.number_input(
    "Procurement budget ($)",
    min_value=100_000, max_value=100_000_000, value=10_000_000, step=100_000,
    help="For converting weights into dollar amounts below -- doesn't change the optimization itself.",
)
refresh_clicked = st.sidebar.button("Force-refresh price data")


# --- data + optimization (cached) ---------------------------------------

# st.button() is only True on the click's own rerun, so wiring it straight
# into the cache key would get silently ignored the rerun after. A counter
# that only increases makes every click permanently change the cache key.
if "refresh_counter" not in st.session_state:
    st.session_state.refresh_counter = 0
if refresh_clicked:
    st.session_state.refresh_counter += 1


@st.cache_data(show_spinner="Pulling feedstock price data...")
def load_returns(years, refresh_counter):
    return get_returns_for_optimizer(years=years, force_refresh=refresh_counter > 0)


@st.cache_data(show_spinner="Checking data provenance...")
def load_raw_prices(years, refresh_counter):
    # separate from load_returns() because this keeps CL=F (excluded from
    # PORTFOLIO_TICKERS) around for the crack-spread diagnostic, and keeps
    # the pre-cleaning picture for sanity_check() to report on
    return fetch_prices(years=years, force_refresh=refresh_counter > 0)


@st.cache_data(show_spinner="Solving the constrained frontier...")
def load_constrained_frontier(mean_returns, cov_matrix, sustainability_min,
                               concentration_cap, n_points):
    return compute_constrained_frontier(
        mean_returns, cov_matrix, sustainability_min, concentration_cap, n_points
    )


@st.cache_data(show_spinner="Solving the unconstrained frontier...")
def load_unconstrained_frontier(mean_returns, cov_matrix, n_points):
    # separate cache from the constrained one -- doesn't depend on either
    # slider, so bundling them would recompute this for no reason
    return compute_unconstrained_frontier(mean_returns, cov_matrix, n_points)


try:
    returns = load_returns(years_lookback, st.session_state.refresh_counter)
except Exception as exc:
    st.error(
        f"Couldn't pull price data from Yahoo Finance: {exc}\n\n"
        "Check your internet connection, or try the force-refresh button "
        "in case the local cache is stale or corrupted."
    )
    st.stop()

mean_returns = returns.mean().values
cov_matrix = returns.cov().values

raw_prices = load_raw_prices(years_lookback, st.session_state.refresh_counter)
provenance = sanity_check(raw_prices)

with st.expander("Data provenance -- where this comes from, exactly"):
    st.markdown(
        f"""
        All prices come from **Yahoo Finance**, pulled through the
        [`yfinance`](https://pypi.org/project/yfinance/) Python library --
        this mirrors Yahoo's own end-of-day data rather than a separate
        paid vendor.

        - **Price field:** `Close`, with `auto_adjust=True` -- Yahoo's
          *adjusted* close (splits/dividends folded in), not the raw print.
        - **Frequency:** daily.
        - **Date range requested:** last {years_lookback} year(s) as of
          today, resolving to **{raw_prices.index.min().date()} through
          {raw_prices.index.max().date()}** in what was actually returned.
        - **Day-mismatch handling:** the six tickers don't all trade on
          exactly the same calendar days (holiday and exchange-specific
          closures differ between CBOT/NYMEX futures and the DAR equity).
          Small per-ticker gaps are forward- then back-filled
          (`ffill().bfill()`) before computing returns, which leaves
          nothing to drop -- a handful of days (e.g. 23 Nov 2023, 9 Jan
          2025, and 4 Jul 2025 on the cached data) carry a repeated price
          for whichever ticker was closed instead, which registers as a
          zero return for that ticker on that day rather than removing
          the day from the dataset.
        """
    )
    st.dataframe(pd.DataFrame(provenance).T)

with st.expander("Crack spread diagnostic (HO=F minus CL=F)"):
    st.markdown(
        "The HO-CL crack spread is a standard refining-margin diagnostic. "
        "**CL=F is never given a portfolio weight** (see the basis-risk note "
        "above) -- it only feeds this chart, as a rough read on diesel "
        "refining economics alongside the UCO proxy, HO=F."
    )
    cleaned_for_spread = clean_prices(raw_prices)
    crack_spread = compute_crack_spread(cleaned_for_spread)
    fig_spread, ax_spread = plt.subplots(figsize=(9, 3))
    ax_spread.plot(crack_spread.index, crack_spread.values, color="#555555", linewidth=1)
    ax_spread.set_ylabel("HO=F - CL=F ($)")
    ax_spread.set_title("Diesel-crude crack spread over the lookback window")
    st.pyplot(fig_spread)

constrained_frontier = load_constrained_frontier(
    mean_returns, cov_matrix, sustainability_min, concentration_cap, n_frontier_points
)
unconstrained_frontier = load_unconstrained_frontier(mean_returns, cov_matrix, n_frontier_points)

if constrained_frontier["success"].sum() == 0:
    st.error(
        "No feasible portfolio found for the current constraint settings. "
        "Try lowering the sustainability minimum, or raising the concentration "
        "cap -- a low cap can leave too little room to also satisfy the budget "
        "and sustainability rules at once."
    )
    st.stop()


# --- frontier plot ---------------------------------------------------------

st.header("Efficient frontier: constrained vs. unconstrained")

fig, ax = plt.subplots(figsize=(9, 5))
valid_u = unconstrained_frontier["success"]
ax.plot(unconstrained_frontier["volatilities"][valid_u], unconstrained_frontier["returns"][valid_u],
        label="Unconstrained (budget + long-only only)", linestyle="--", color="gray")
valid_c = constrained_frontier["success"]
ax.plot(constrained_frontier["volatilities"][valid_c], constrained_frontier["returns"][valid_c],
        label="Constrained (RED2 + concentration cap)", color="steelblue", linewidth=2)
ax.set_xlabel("Daily cost volatility (std. dev. of returns)")
ax.set_ylabel("Expected daily return")
ax.set_title("Cost of the client's real-world constraints")
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
ax.legend()
st.pyplot(fig)

st.caption(
    "The gap between the two curves is what the sustainability minimum and "
    "concentration cap 'cost' the client in expected return per unit of risk "
    "-- the price of staying RED2-eligible and supply-diversified."
)

# live evidence for the diversification note in the basis-risk expander
# above -- computed from whatever the app just pulled, so it's an actual
# measurement rather than a claim the reader has to take on faith. This
# sits below the chart on purpose, not above it: the pitch script has the
# presenter pointing down at it right after the frontier plot.
crop_correlation = pairwise_correlation(cov_matrix, PORTFOLIO_TICKERS, "ZL=F", "ZS=F")
st.caption(
    f"**Diversification check** -- over this lookback window, ZL=F and ZS=F are "
    f"{crop_correlation:.0%} correlated. This is the live number behind the "
    f"basis-risk note above. There is no reliably sourced figure for what a "
    f"genuine soybean-oil/rapeseed-oil pair would score, so this number is "
    f"reported on its own terms rather than framed as 'more correlated than "
    f"it should be.' It will move if the lookback window or the market "
    f"itself changes."
)


# --- recommended allocation ------------------------------------------------

st.header("Recommended allocation")

pick_method = st.radio(
    "Pick a point on the constrained frontier:",
    ["Minimum variance (most risk-averse)", "Best return-per-risk (Sharpe-equivalent)"],
    horizontal=True,
)
recommended = (
    min_variance_portfolio(constrained_frontier) if pick_method.startswith("Minimum")
    else max_sharpe_equivalent_portfolio(constrained_frontier)
)
weights = recommended["weights"]
bar_colors = [TICKER_COLORS[t] for t in PORTFOLIO_TICKERS]

render_ticker_key()

col1, col2 = st.columns([2, 1])

with col1:
    # grouped by RED2 bucket (waste-based, then crop-based) instead of
    # one flat row in PORTFOLIO_TICKERS order -- this is what the
    # roadmap's "split into waste-based vs. crop-based buckets" line
    # actually asked for
    waste_t, waste_w, crop_t, crop_w = bucket_split(weights, PORTFOLIO_TICKERS, WASTE_BASED)
    ordered_tickers = waste_t + crop_t
    ordered_weights = np.array(waste_w + crop_w)
    ordered_colors = [TICKER_COLORS[t] for t in ordered_tickers]

    gap = 1  # extra x-slot between the two buckets, purely visual
    x_positions = list(range(len(waste_t))) + [i + len(waste_t) + gap for i in range(len(crop_t))]

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    bars2 = ax2.bar(x_positions, ordered_weights, color=ordered_colors, width=0.6)
    ax2.bar_label(bars2, labels=[f"{w:.1%}" for w in ordered_weights], padding=3, fontsize=9)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(ordered_tickers)

    # dashed divider through the gap slot, plus a bucket label centered
    # over each group so the RED2 split reads at a glance
    divider_x = len(waste_t) - 0.5 + gap / 2
    ax2.axvline(divider_x, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    y_top = max(concentration_cap, ordered_weights.max()) * 1.15
    ax2.text((len(waste_t) - 1) / 2, y_top, "Waste-based (RED2)",
              ha="center", fontsize=8, color="dimgray")
    ax2.text(len(waste_t) + gap + (len(crop_t) - 1) / 2, y_top, "Crop-based",
              ha="center", fontsize=8, color="dimgray")

    ax2.set_ylabel("Portfolio weight")
    ax2.set_title("Feedstock allocation")
    ax2.set_ylim(0, max(concentration_cap, ordered_weights.max()) * 1.3)  # headroom for labels + bucket text
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    st.pyplot(fig2)

with col2:
    st.metric("Expected daily cost volatility", f"{recommended['volatility']:.4%}")
    st.metric("Expected daily return", f"{recommended['return']:.4%}")

    sustainability_achieved = sustainability_share(weights)
    sustainability_pass = sustainability_achieved >= sustainability_min - 1e-9
    st.metric("Sustainability share achieved", f"{sustainability_achieved:.1%}",
              help=f"Target minimum: {sustainability_min:.0%}")
    st.markdown(f"{'✅ **PASS**' if sustainability_pass else '❌ **FAIL**'} -- RED2 sustainability minimum")

    concentration_pass = weights.max() <= concentration_cap + 1e-9
    st.metric("Largest single feedstock weight", f"{weights.max():.1%}")
    st.markdown(f"{'✅ **PASS**' if concentration_pass else '❌ **FAIL**'} -- concentration cap")

    if "sharpe_equivalent" in recommended:  # only the Sharpe pick has this key
        st.metric("Return-per-risk (Sharpe-equivalent)", f"{recommended['sharpe_equivalent']:.4f}")

    st.caption(f"At a ${procurement_budget:,.0f} budget:")
    st.write({t: f"${w * procurement_budget:,.0f}" for t, w in zip(PORTFOLIO_TICKERS, weights)})


# --- risk budgeting ----------------------------------------------------

st.header("Risk budgeting: where the risk actually comes from")

rc = risk_contributions(weights, cov_matrix)
fig3, ax3 = plt.subplots(figsize=(7, 4))
bars3 = ax3.bar(PORTFOLIO_TICKERS, rc, color=bar_colors)
ax3.bar_label(bars3, labels=[f"{v:.1%}" for v in rc], padding=3, fontsize=9)
ax3.set_ylabel("% contribution to total portfolio variance")
ax3.set_title("Risk contribution per feedstock")
ax3.set_ylim(0, rc.max() * 1.25)  # headroom for the labels
ax3.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
st.pyplot(fig3)

st.caption(
    "A feedstock's risk contribution isn't always proportional to its "
    "allocation weight -- a smaller position in a volatile or highly "
    "correlated feedstock can still dominate overall portfolio risk. "
    "This is the risk-budgeting lens layered on top of the mean-variance result."
)


# --- raw numbers, for anyone checking the math -----------------------------

with st.expander("Show raw weights and risk contributions"):
    st.write({
        ticker: {
            "weight": round(float(w), 4),
            "risk_contribution": round(float(r), 4),
            "bucket": "waste-based" if ticker in WASTE_BASED else "crop-based",
        }
        for ticker, w, r in zip(PORTFOLIO_TICKERS, weights, rc)
    })
