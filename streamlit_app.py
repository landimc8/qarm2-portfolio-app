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
    You buy four feedstocks (used cooking oil, tallow, soybean oil and rapeseed oil) 
    and turn them into biodiesel and renewable diesel (HVO). Their prices move a lot 
    from month to month, and as a buyer, rising prices mean higher costs and a smaller 
    margin. This tool finds the mix of feedstocks whose total cost moves the least, 
    while following your three rules: the EU RED2 minimum for waste-based feedstocks, 
    a cap so no single feedstock dominates your supply, and your procurement budget. 
    It also shows a stable-margin mix if your selling price follows diesel, 
    how much profit a bad month could take, and it can run on your own prices.
    """
)

with st.expander("A note on the data (basis risk)"):
    st.markdown(
        """
        Public daily prices for UCO and tallow are not free: Argus and
        Fastmarkets sell them. So this tool uses free prices that move in
        a similar way (proxies):

        - **UCO → HO=F** (New York diesel futures), because UCO-based fuel
          competes with diesel.
        - **Tallow → LE=F** (live cattle futures) and **DAR** (Darling
          Ingredients), because tallow comes from cattle and Darling is the
          biggest processor of animal fats into fuel.
        - **Rapeseed → ZS=F** (US soybean futures), because Yahoo Finance
          has no canola/rapeseed price. ZS=F is a seed, not an oil, so it
          adds two gaps: a different crop, and seed versus oil.

        The gap between a proxy and your real price is called **basis
        risk**. We measured it in the two checks at the bottom of the page:
        **good for UCO** (correlation 0.89 with Rotterdam gasoil) and
        **weak for rapeseed** (0.44 with Euronext rapeseed). Soybean oil
        (ZL=F) and soybeans (ZS=F) also come from the same Chicago market,
        so they move together more than real soybean oil and rapeseed oil
        would: the correlation under the frontier chart shows how much.

        **To remove most of this gap, upload your real prices** (Argus or
        your invoices) in the sidebar under **"Add your own data"**: the
        tool then runs the same optimization on them.
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
    min_value=0.25, max_value=1.00, value=0.35, step=0.05,
    help="No single feedstock may exceed this share of the procurement budget. "
         "Tallow counts once: LE=F + DAR together. Minimum 25%, because 4 real feedstocks x 25% = 100%.",
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
        All portfolio prices come from **Yahoo Finance**, pulled through the
        [`yfinance`](https://pypi.org/project/yfinance/) Python library --
        this mirrors Yahoo's own end-of-day data rather than a separate
        paid vendor. There are two exceptions, both used only in the
        basis-risk checks at the bottom: the Rotterdam gasoil price and the
        Euronext rapeseed price, which come from Investing.com (Yahoo does
        not carry them). If you upload your own prices in the sidebar, they
        are used only in the "Your own data" box.

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
    prov_df = pd.DataFrame(provenance).T
    prov_df["first_date"] = pd.to_datetime(prov_df["first_date"]).dt.date
    prov_df["last_date"] = pd.to_datetime(prov_df["last_date"]).dt.date
    st.dataframe(prov_df)

with st.expander("Crack spread diagnostic (HO=F x 42 minus CL=F, $/bbl)"):
    st.markdown(
        "This chart shows the gap between the diesel price and the crude oil price: "
        "how much a refinery earns per barrel. We use diesel (HO=F) as the proxy for "
        "used cooking oil, because fuel made from UCO is sold in the same market as diesel. "
        "When this gap is large, diesel is expensive, and used cooking oil often gets "
        "more expensive too. For you, this matters because your biodiesel selling price "
        "often follows diesel. **Crude oil (CL=F) is only used for this chart. It is not "
        "part of the portfolio.**"
    )
    cleaned_for_spread = clean_prices(raw_prices)
    crack_spread = compute_crack_spread(cleaned_for_spread)
    diesel_bbl = cleaned_for_spread["HO=F"] * 42      # $/gal -> $/bbl
    crude_bbl = cleaned_for_spread["CL=F"]
    fig_spread, (ax_px, ax_spread) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                                                  gridspec_kw={"height_ratios": [2, 1]})
    ax_px.plot(diesel_bbl.index, diesel_bbl.values, color="#eb6834", linewidth=1, label="Diesel (HO=F x 42)")
    ax_px.plot(crude_bbl.index, crude_bbl.values, color="#2a78d6", linewidth=1, linestyle="--",
               label="WTI crude (CL=F)")
    ax_px.set_ylabel("$ per barrel")
    ax_px.set_title("Diesel and crude prices, and the crack spread between them")
    ax_px.legend()
    ax_spread.plot(crack_spread.index, crack_spread.values, color="#555555", linewidth=1)
    ax_spread.set_ylabel("Crack ($/bbl)")
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
        label="Unconstrained (budget + long-only)", linestyle="--", color="gray")
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
    "The gap between the grey and blue lines is the cost of the client's rules: for the same "
    "risk, the blue line gives less return. At the default settings, the safest mix already "
    "holds more than 40% waste, so most of this cost comes from the 35% cap per feedstock."
)

# live evidence for the diversification note in the basis-risk expander
# above -- computed from whatever the app just pulled, so it's an actual
# measurement rather than a claim the reader has to take on faith. This
# sits below the chart on purpose, not above it: the pitch script has the
# presenter pointing down at it right after the frontier plot.
crop_correlation = pairwise_correlation(cov_matrix, PORTFOLIO_TICKERS, "ZL=F", "ZS=F")
st.caption(
    f"**Diversification check:** over this period, ZL=F and ZS=F are "
    f"**{crop_correlation:.0%}** correlated: they often move together, but not always. "
    f"Both come from the Chicago soybean market, so our two crop feedstocks are less "
    f"diversified than planned. This number updates when you change the years slider."
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

    _real = {t: w for t, w in zip(PORTFOLIO_TICKERS, weights) if t not in ("LE=F", "DAR")}
    _real["Tallow (LE=F + DAR)"] = sum(w for t, w in zip(PORTFOLIO_TICKERS, weights) if t in ("LE=F", "DAR"))
    largest_real = max(_real.values())
    concentration_pass = largest_real <= concentration_cap + 1e-9
    st.metric("Largest single feedstock weight", f"{largest_real:.1%}",
              help="Tallow counts as one feedstock: LE=F + DAR together.")
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


# --- global minimum variance (LN1 closed form, no client rules) -------------

with st.expander("Compare: global minimum variance (no rules) vs our recommended mix"):
    # 1) Professor's closed form (LN1): budget only, short selling allowed
    ones = np.ones(len(PORTFOLIO_TICKERS))
    inv_cov = np.linalg.inv(cov_matrix)
    w_gmv = inv_cov @ ones / (ones @ inv_cov @ ones)          # Sigma^-1 1 / 1' Sigma^-1 1
    vol_gmv = np.sqrt(w_gmv @ cov_matrix @ w_gmv)

    # 2) Our optimizer: lowest-risk point of the grey line (budget + long-only)
    grey_mv = min_variance_portfolio(unconstrained_frontier)
    w_grey = grey_mv["weights"]
    vol_grey = grey_mv["volatility"]

    st.markdown(
        "Global minimum variance: "
        "$x_{gmv} = \\Sigma^{-1}\\mathbf{1} / \\mathbf{1}^\\top\\Sigma^{-1}\\mathbf{1}$ "
        "(only the budget rule, short selling allowed)."
    )
    st.dataframe(pd.DataFrame({
        "Analytical solution (budget only)": [f"{w:.1%}" for w in w_gmv] + [f"{vol_gmv:.2%}"],
        "Grey line minimum (budget + long-only)": [f"{w:.1%}" for w in w_grey] + [f"{vol_grey:.2%}"],
        "Our recommended mix (+ RED2 + cap)": [f"{w:.1%}" for w in weights] + [f"{recommended['volatility']:.2%}"],
    }, index=PORTFOLIO_TICKERS + ["Daily risk"]))
    st.caption(
        "The analytical solution has no negative weights on our data, so it matches the grey line's "
        "safest point (small differences come from the 50-point grid). It puts about two-thirds of the "
        "money in tallow (live cattle + Darling). Your 35% cap allows only 35%, so our mix is riskier: "
        "that is the cost of your rules."
    )


# --- basis-risk check: Rotterdam gasoil vs our UCO proxy (HO=F) -------------

GASOIL_FILE = "data/gasoil_ice.csv"   # ICE London Gas Oil, Investing.com export ($/tonne)

with st.expander("Basis-risk check: Rotterdam gasoil vs our UCO proxy (HO=F)"):
    try:
        gas = pd.read_csv(GASOIL_FILE, thousands=",")
        gas["Date"] = pd.to_datetime(gas["Date"], format="%m/%d/%Y")
        gas = gas.set_index("Date")["Price"].sort_index() / 7.45      # $/tonne -> $/bbl
        ho = clean_prices(raw_prices)["HO=F"] * 42                    # $/gal -> $/bbl
        both = pd.concat({"Gasoil (Rotterdam)": gas, "HO=F (New York)": ho}, axis=1, sort=True).dropna()
        weekly = both.resample("W-FRI").last().dropna()
        premium = weekly["HO=F (New York)"] - weekly["Gasoil (Rotterdam)"]
        corr_w = weekly.pct_change().dropna().corr().iloc[0, 1]

        m1, m2, m3 = st.columns(3)
        m1.metric("Weekly return correlation", f"{corr_w:.0%}")
        m2.metric("Average NY premium", f"{premium.mean():+.1f} $/bbl")
        m3.metric("Largest gap", f"${premium.abs().max():.0f}/bbl")

        fig_b, (ax_p, ax_s) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                                           gridspec_kw={"height_ratios": [2, 1]})
        ax_p.plot(weekly.index, weekly["Gasoil (Rotterdam)"], color="#1baf7a", label="ICE gasoil (/ 7.45)")
        ax_p.plot(weekly.index, weekly["HO=F (New York)"], color="#eb6834", linestyle="--",
                  label="HO=F (x 42), our UCO proxy")
        ax_p.set_ylabel("$ per barrel")
        ax_p.legend()
        ax_s.fill_between(premium.index, premium, 0, color="gray", alpha=0.2)
        ax_s.plot(premium.index, premium, color="gray", linewidth=1)
        ax_s.axhline(0, color="black", linewidth=0.5)
        ax_s.set_ylabel("NY premium ($/bbl)")
        st.pyplot(fig_b)

        st.caption(
            "You buy in Rotterdam, but our proxy for used cooking oil is the New York diesel price. "
            "Here we compare it with the Rotterdam diesel price (ICE gasoil). Most weeks, the two "
            "prices move together (89%). But in a crisis, like 2022 or spring 2026, the gap between "
            "them can jump. This checks the New York vs Rotterdam gap only: it does not show how "
            "closely diesel follows your real UCO price, which only your own UCO prices can tell. "
            "Gasoil prices come from Investing.com, converted from $ per tonne to $ per barrel."
        )
    except FileNotFoundError:
        st.info("Add the ICE gasoil CSV as data/gasoil_ice.csv to show this check.")


# --- why minimum variance: side-by-side comparison --------------------------

with st.expander("Why we recommend minimum variance (not Sharpe)"):
    mv_pick = min_variance_portfolio(constrained_frontier)
    sr_pick = max_sharpe_equivalent_portfolio(constrained_frontier)
    horizon = car_horizon if "car_horizon" in globals() else 21
    conf = car_conf if "car_conf" in globals() else "95%"
    z_val = 1.645 if conf == "95%" else 2.326

    def _why_row(p):
        rc_p = risk_contributions(p["weights"], cov_matrix)
        top = int(np.argmax(rc_p))
        return [
            f"{p['volatility']:.2%}",
            f"{p['return']:.3%}",
            f"{(1 + p['return']) ** 252 - 1:+.1%}",
            f"${procurement_budget * p['volatility'] * np.sqrt(horizon) * z_val:,.0f}",
            f"{p['return'] / p['volatility']:.3f}",
            f"{PORTFOLIO_TICKERS[top]} ({rc_p[top]:.0%})",
            f"{sustainability_share(p['weights']):.0%}",
        ]

    st.markdown(
                        "You are a **buyer**: a higher 'return' means feedstock prices going up, "
        "so higher costs. Your goal is **stable costs**, which is what minimum variance gives."
    )
    st.dataframe(pd.DataFrame({
        "Minimum variance (recommended)": _why_row(mv_pick),
        "Sharpe-equivalent": _why_row(sr_pick),
    }, index=[
        "Daily risk: σ(x) = √(xᵀΣx)",
        "Daily 'return' = price increase: μ(x) = xᵀμ",
        "Price increase over a year: (1 + μ)^252 − 1",
        f"Cost-at-Risk ({conf}, {horizon} days): B × σ × √h × z",
        "Return per risk: μ / σ",
        "Main risk source: RC*ᵢ",
        "Waste share (RED2)",
    ]))
    _car_mv = procurement_budget * mv_pick["volatility"] * np.sqrt(horizon) * z_val
    st.caption(
        f"**What is the Cost-at-Risk?** In a bad month (about 1 month in 20), your feedstock bill "
        f"could rise by about \\${_car_mv:,.0f} with our mix. In the other 19 months, it rises by "
        f"less, or falls. Formula: budget × daily risk × √{horizon} days × {z_val}. We recommend "
        f"you follow this number every month: it shows your risk in money, so you can decide how "
        f"much to hedge. It assumes normal price moves, so real crises like 2022 can be worse. "
        f"Minimum variance always has the lowest risk by definition. We recommend it because "
        f"stable costs are the right goal for a buyer."
    )


# --- stable margin: minimize tracking error vs the selling price (LN2) -------

with st.expander("Stable costs or stable margin? (tracking error vs diesel)"):
    from scipy.optimize import minimize as _minimize
    from constraints import build_all_constraints as _bac, get_bounds as _gb

    # Benchmark b = 100% diesel (HO=F): our proxy for the biodiesel selling price
    b = np.array([1.0 if t == "HO=F" else 0.0 for t in PORTFOLIO_TICKERS])

    def _margin_var(x):
        d = x - b
        return d @ cov_matrix @ d          # (x - b)' Sigma (x - b) = variance of the margin

    _res = _minimize(
        _margin_var, np.ones(len(PORTFOLIO_TICKERS)) / len(PORTFOLIO_TICKERS),
        method="SLSQP",
        bounds=_gb(concentration_cap=concentration_cap, n_assets=len(PORTFOLIO_TICKERS)),
        constraints=_bac(sustainability_min=sustainability_min, concentration_cap=concentration_cap),
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w_margin = _res.x
    w_cost = min_variance_portfolio(constrained_frontier)["weights"]

    def _cost_risk(x):
        return np.sqrt(x @ cov_matrix @ x)

    def _margin_risk(x):
        return np.sqrt(_margin_var(x))

        st.markdown(
        "If you sell your biodiesel at a price linked to diesel, your **margin** is roughly "
        "*selling price change − feedstock cost change*. Minimizing the variance of the margin is "
        "the same as minimizing the **tracking error** against a benchmark *b* = 100% diesel: "
        "$\\min_x \\,(x-b)^\\top\\Sigma\\,(x-b)$, with the same three rules."
    )
    st.dataframe(pd.DataFrame({
        "Stable costs (our recommendation)": [f"{w:.1%}" for w in w_cost]
            + [f"{_cost_risk(w_cost):.2%}", f"{_margin_risk(w_cost):.2%}", f"{sustainability_share(w_cost):.0%}"],
        "Stable margin (tracking error vs diesel)": [f"{w:.1%}" for w in w_margin]
            + [f"{_cost_risk(w_margin):.2%}", f"{_margin_risk(w_margin):.2%}", f"{sustainability_share(w_margin):.0%}"],
    }, index=PORTFOLIO_TICKERS + ["Daily cost risk", "Daily margin risk", "Waste share (RED2)"]))
    st.caption(
        "The stable-margin mix buys more diesel-linked UCO and soybean oil, because their costs move with "
        "the selling price. Its costs move more, but its margin moves less. Limit: here diesel (HO=F) is "
        "both our UCO proxy and the selling price, so the gain looks larger than it really is. With real "
        "prices (your own data box), the gain is usually smaller. European biodiesel follows Rotterdam "
        "gasoil; we use HO=F as a free proxy."
    )


# --- rapeseed basis-risk check: ZS=F (US soybeans) vs Euronext rapeseed -------

with st.expander("Basis-risk check: US soybeans (ZS=F) vs European rapeseed (Euronext)"):
    _rs = pd.read_csv("data/rapeseed_euronext.csv", thousands=",")
    _rs["Date"] = pd.to_datetime(_rs["Date"], format="%m/%d/%Y")
    _rs = _rs.set_index("Date").sort_index()["Price"].astype(float).rename("Rapeseed (Euronext)")

    # find the app's price table (the DataFrame with ZS=F prices, not returns)
    _px = next(v for v in list(globals().values())
               if isinstance(v, pd.DataFrame) and {"ZS=F", "ZL=F"} <= set(v.columns)
               and v["ZS=F"].dropna().min() > 100)

    _d = pd.concat([_px[["ZS=F", "ZL=F"]], _rs], axis=1, sort=True).dropna()
    _wk = _d.resample("W-FRI").last().dropna()
    _ret = np.log(_wk).diff().dropna()

    # Euronext contracts expire Feb/May/Aug/Nov: the continuous series jumps at each roll.
    # Remove the weeks that contain a roll, so we measure prices, not contract changes.
    _roll_days = _rs.index[_rs.index.month.isin([2, 5, 8, 11]) & (_rs.index.day <= 3)]
    _roll_weeks = set(_roll_days.to_period("W-FRI").end_time.normalize())
    _ret = _ret[~_ret.index.normalize().isin(_roll_weeks)]

    _c_zs = _ret["ZS=F"].corr(_ret["Rapeseed (Euronext)"])
    _c_zl = _ret["ZL=F"].corr(_ret["Rapeseed (Euronext)"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Correlation ZS=F vs rapeseed", f"{_c_zs:.2f}")
    c2.metric("Correlation ZL=F vs rapeseed", f"{_c_zl:.2f}")
    c3.metric("Weeks compared", f"{len(_ret)}")

    st.line_chart((_wk / _wk.iloc[0] * 100).rename(columns={
        "ZS=F": "US soybeans (ZS=F)", "ZL=F": "US soybean oil (ZL=F)"}))
    st.caption(
        "Prices indexed to 100 at the start. Correlation of weekly returns; roll weeks removed. "
        "Rapeseed is in €/tonne and ZS=F in US cents per bushel, so we compare price changes, "
        "not price levels. A correlation of about 0.4 means US soybeans explain only a small "
        "part of European rapeseed moves (R² = corr² ≈ 0.2): the basis risk is large. "
        "Soybean oil (ZL=F) follows rapeseed a little better than soybeans. "
        "Euronext rapeseed is the seed and you buy rapeseed oil, so a small seed-vs-oil gap remains. "
        "For your rapeseed, we recommend hedging with the Euronext contract directly. "
        "Source: Euronext rapeseed futures (COMc1), Investing.com."
    )


# --- does the mix really reduce cost swings? (diversification proof) ---------

with st.expander("How much does our mix reduce cost swings?"):
    _w_mix = np.asarray(min_variance_portfolio(constrained_frontier)["weights"])
    _r = returns[PORTFOLIO_TICKERS]
    _monthly = (1 + _r).resample("ME").prod() - 1          # monthly price change of each proxy
    _cov = np.asarray(cov_matrix)

    _names = {"ZL=F": "Only soybean oil (ZL=F)", "ZS=F": "Only soybeans (ZS=F)",
              "HO=F": "Only UCO (diesel, HO=F)", "LE=F": "Only live cattle (LE=F)",
              "DAR": "Only Darling (DAR)"}
    _options = {_names.get(t, "Only " + t): np.eye(len(PORTFOLIO_TICKERS))[i]
                for i, t in enumerate(PORTFOLIO_TICKERS)}
    _options["Equal split (20% each)"] = np.ones(len(PORTFOLIO_TICKERS)) / len(PORTFOLIO_TICKERS)
    _options["Our recommended mix"] = _w_mix

    _rows = []
    for _label, _w in _options.items():
        _m = float((_monthly.values @ _w).std())             # typical monthly cost change
        _rows.append([_label, f"{np.sqrt(_w @ _cov @ _w):.2%}", f"{_m:.1%}", f"${procurement_budget * _m:,.0f}"])
    _tab = pd.DataFrame(_rows, columns=["What you buy", "Daily cost risk",
                                        "Typical monthly cost change", "In money (monthly)"]).set_index("What you buy")
    st.dataframe(_tab)

    _m_mix = float((_monthly.values @ _w_mix).std())
    _m_eq = float((_monthly.values @ _options["Equal split (20% each)"]).std())
    _m_max = max(float(_monthly[t].values.std()) for t in PORTFOLIO_TICKERS)
    st.markdown(
        f"Our mix moves **{_m_max / _m_mix:.1f} times less** than the riskiest single feedstock, "
        f"and **{1 - _m_mix / _m_eq:.0%} less** than a simple equal split."
    )
    st.caption(
        "Typical monthly cost change = standard deviation of the monthly price change of each option, "
        "on the same data as the model. This is diversification: the feedstocks don't all move together, "
        "so the cost of the mix moves less. In-sample: the weights come from the same data. "
        "The equal split is only a comparison: it breaks the tallow cap (LE=F + DAR = 40%)."
    )


# --- profit-at-risk calculator: what a bad month does to the client's profit --

with st.expander("Profit-at-Risk calculator: how much profit could you lose?"):
    from scipy.optimize import minimize as _pminimize
    from constraints import build_all_constraints as _pbac, get_bounds as _pgb

    st.markdown(
        "Enter how you sell your biodiesel. The app shows your expected monthly profit "
        "and how much of it could disappear in a bad month, with each mix."
    )
    _c1, _c2 = st.columns(2)
    with _c1:
        _sales = st.radio("How do you sell your biodiesel?",
                          ["Fixed selling price", "Selling price follows diesel"], key="pc_sales")
        _price = st.number_input("Selling price ($ per tonne of biodiesel)", min_value=0.0,
                                 value=1500.0, step=50.0, key="pc_price")
    with _c2:
        _tonnes = st.number_input("Tonnes sold per month", min_value=0.0,
                                  value=10000.0, step=500.0, key="pc_tonnes")
        _other = st.number_input("Other costs ($ per tonne: energy, staff, transport)", min_value=0.0,
                                 value=300.0, step=25.0, key="pc_other")
    _share = st.slider("If the price follows diesel: share of the selling price that moves with diesel", 0, 100, 67, 1,
                       key="pc_share", format="%d%%",
                       help="Example: biodiesel sold at the diesel price + a fixed premium of $500/t. "
                            "At $1,500/t, the diesel part is $1,000, so 67% of the price moves with diesel.") / 100

    _h = car_horizon if "car_horizon" in globals() else 21
    _lvl = car_conf if "car_conf" in globals() else "95%"
    _z = 1.645 if _lvl == "95%" else 2.326
    _cov = np.asarray(cov_matrix)
    _n = len(PORTFOLIO_TICKERS)
    _b = np.array([1.0 if t == "HO=F" else 0.0 for t in PORTFOLIO_TICKERS])   # diesel

    # the two mixes: main (stable costs) and stable margin (tracking error vs diesel, LN2)
    _w_main = np.asarray(min_variance_portfolio(constrained_frontier)["weights"])
    _res = _pminimize(lambda x: (x - _b) @ _cov @ (x - _b), np.ones(_n) / _n, method="SLSQP",
                      bounds=_pgb(concentration_cap=concentration_cap, n_assets=_n),
                      constraints=_pbac(sustainability_min=sustainability_min,
                                        concentration_cap=concentration_cap),
                      options={"maxiter": 500, "ftol": 1e-12})
    _w_margin = _res.x

    _revenue = _price * _tonnes
    _feed = procurement_budget                     # feedstock bill = the budget in the sidebar
    _other_total = _other * _tonnes
    _profit = _revenue - _feed - _other_total

    def _par(w, follows_diesel):
        # dollar exposure to each price: revenue moves with diesel if the price follows diesel,
        # feedstock costs move with the mix. Profit change = revenue change - cost change.
        d = (_revenue * _share * _b if follows_diesel else 0.0) - _feed * w
        return float(np.sqrt(d @ _cov @ d) * np.sqrt(_h) * _z)

    def _usd(v):
        return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"

    _follows = _sales == "Selling price follows diesel"
    _par_main, _par_margin = _par(_w_main, _follows), _par(_w_margin, _follows)
    _best = "main mix (stable costs)" if _par_main <= _par_margin else "stable-margin mix"
    _par_best = min(_par_main, _par_margin)

    m1, m2, m3 = st.columns(3)
    m1.metric("Expected monthly profit", _usd(_profit),
              help="Sales - feedstock budget - other costs")
    m2.metric(f"Profit-at-Risk ({_lvl}, {_h} days)", f"${_par_best:,.0f}",
              help=f"With the {_best}")
    m3.metric("Profit in a bad month", _usd(_profit - _par_best))

    st.dataframe(pd.DataFrame({
        "Main mix (stable costs)": [f"${_par(_w_main, False):,.0f}", f"${_par(_w_main, True):,.0f}"],
        "Stable-margin mix": [f"${_par(_w_margin, False):,.0f}", f"${_par(_w_margin, True):,.0f}"],
    }, index=["Fixed selling price", "Selling price follows diesel"]))

    st.markdown(
        f"Because your {'selling price follows diesel' if _follows else 'selling price is fixed'}, "
        f"the **{_best}** protects the profit best: "
        f"in a bad month, it could lose about **{_usd(_par_best)}** of its **{_usd(_profit)}** profit. "
        f"The other mix would put **{_usd(abs(_par_main - _par_margin))} more** at risk.".replace("$", "\\$")
    )
    if _profit < 0:
        st.error("With these inputs the client loses money even in a normal month: sales are lower than "
                 "feedstock + other costs. Check the selling price, tonnes and the budget in the sidebar.")
    elif _profit - _par_best < 0:
        st.warning("In a bad month, the profit could turn into a loss. The client may also need futures to lock prices.")
    st.caption(
        "Profit-at-Risk = √(dᵀΣd) × √days × z, where d = dollar exposure: + the diesel-linked part of sales "
        "(only if the price follows diesel), − feedstock budget × mix weights. Same parametric formula as the "
        "Cost-at-Risk (normal returns, not from the course). Selling price, tonnes and other costs are your "
        "inputs; only the risks come from the model. Two limits: the feedstock bill is the budget in the "
        "sidebar, so keep it consistent with your tonnes; and diesel (HO=F) is both our UCO proxy and the "
        "selling price, so the stable-margin gain looks larger than with real prices."
    )


# --- recommended data sources for the next version ---------------------------

with st.expander("Recommended data sources (next version of your tool)"):
    st.markdown(
        "Today the tool uses free proxies. With professional data, it would use **your real prices** "
        "and **your real rules** instead of our assumptions."
    )
    st.table(pd.DataFrame({
        "Source we recommend": [
            "Argus",
            "Argus",
            "SquareCommodities (regulatory data)",
            "ICE (gasoil), CME (soybean oil), Euronext (rapeseed)",
            "Nabisy",
            "EIA, IEA, JODI (oil and diesel), USDA (oilseeds)",
        ],
        "What it replaces or adds in the tool": [
            "Replaces our proxies (HO=F, LE=F, DAR, ZS=F), so most of the basis risk disappears",
            "Replaces diesel (HO=F) as the benchmark for the stable-margin mix",
            "Replaces our assumed 40% RED2 minimum with the real mandate level",
            "With Argus prices, gives the hedge ratio to lock your prices with futures",
            "Confirms which volumes count as waste-based for RED2",
            "Helps you decide when and how much to hedge (supply, demand, stocks)",
        ],
    }, index=[
        "Your feedstock prices (UCO, tallow, rapeseed oil, soybean oil)",
        "Your selling price (FAME, HVO)",
        "The real mandate levels",
        "Futures for the hedge",
        "Sustainability proofs",
        "Market fundamentals",
    ]))
    st.caption(
        "Prices (Argus and the exchanges) go into the model. Fundamentals (EIA, IEA, JODI, USDA) and "
        "regulatory data (SquareCommodities, Nabisy) set the rules and explain why prices move. "
        "Your own purchase prices, supplier quotes and sales contracts would improve the tool even more."
    )


# --- add your own data: run the model on the client's real prices ------------

with st.sidebar:
    st.markdown("---")
    st.subheader("Add your own data (optional)")
    st.caption("Replace our free proxies with your real prices: Argus prices or your own invoices.")
    _own_template = (
        "Date,UCO,Tallow,Rapeseed oil,Soybean oil,Selling price\n"
        "2026-01-05,1100,1050,1020,1000,1500\n"
        "2026-01-06,1105,1048,1025,1004,1508\n"
    )
    st.download_button("Download the template (CSV)", _own_template,
                       file_name="your_prices_template.csv", mime="text/csv")
    _own_file = st.file_uploader("Upload your feedstock prices (CSV)", type="csv", key="own_prices",
                                 help="One column per feedstock (UCO, Tallow, Rapeseed oil, Soybean oil), "
                                      "same units over time. Optional: a 'Selling price' column (Argus FAME/HVO) "
                                      "for the stable-margin mix.")
    st.caption("Real mandate level: set the RED2 slider above to the level from SquareCommodities’ "
               "regulatory data.")

with st.expander("Your own data: the recommended mix on your real prices"):
    if _own_file is None:
        st.info("Upload your prices in the sidebar (‘Add your own data’). The tool will then run the same "
                "optimization, with the same rules, on your real prices instead of our proxies.")
    else:
        from scipy.optimize import minimize as _ominimize
        try:
            _od = pd.read_csv(_own_file)
            _od["Date"] = pd.to_datetime(_od["Date"])
            _od = _od.set_index("Date").sort_index()
        except Exception as _e:
            st.error(f"Could not read the file. Use the template: a 'Date' column and one column per price. ({_e})")
            st.stop()

        _feeds = [c for c in ["UCO", "Tallow", "Rapeseed oil", "Soybean oil"] if c in _od.columns]
        _waste = [c for c in _feeds if c in ("UCO", "Tallow")]
        _has_sell = "Selling price" in _od.columns
        _cols = _feeds + (["Selling price"] if _has_sell else [])
        _r = _od[_cols].apply(pd.to_numeric, errors="coerce").dropna().pct_change().dropna()

        if len(_feeds) < 2 or len(_r) < 20:
            st.error("We need at least 2 feedstock columns and about 20 dates of prices.")
        elif concentration_cap * len(_feeds) < 1 or concentration_cap * len(_waste) < sustainability_min:
            st.error("With these columns, your rules can’t all hold at once: lower the RED2 minimum or raise the cap.")
        else:
            _n = len(_feeds)
            _S = _r[_feeds].cov().values
            _widx = [_feeds.index(c) for c in _waste]
            _cons = [{"type": "eq", "fun": lambda w: w.sum() - 1},
                     {"type": "ineq", "fun": lambda w: sum(w[i] for i in _widx) - sustainability_min}]
            _bnds = [(0.0, concentration_cap)] * _n
            _res = _ominimize(lambda w: w @ _S @ w, np.ones(_n) / _n, method="SLSQP",
                              bounds=_bnds, constraints=_cons, options={"maxiter": 500, "ftol": 1e-12})
            _w = _res.x

            # how many trading days one row of your data covers (daily, weekly or monthly files)
            _gap = max(float(pd.Series(_od.index).diff().dt.days.median()) * 5 / 7, 1.0)
            _per_month = np.sqrt(21 / _gap)
            _sig = float(np.sqrt(_w @ _S @ _w))
            _car = procurement_budget * _sig * _per_month * 1.645

            # our proxy mix, regrouped by real feedstock, for comparison
            _wp = dict(zip(PORTFOLIO_TICKERS, np.asarray(min_variance_portfolio(constrained_frontier)["weights"])))
            _proxy_share = {"UCO": _wp.get("HO=F", 0), "Tallow": _wp.get("LE=F", 0) + _wp.get("DAR", 0),
                            "Rapeseed oil": _wp.get("ZS=F", 0), "Soybean oil": _wp.get("ZL=F", 0)}

            st.markdown(f"Using **{len(_r)} price changes** from your file "
                        f"({_r.index.min():%d %b %Y} to {_r.index.max():%d %b %Y}).")
            st.dataframe(pd.DataFrame({
                "Mix on your real prices": [f"{x:.1%}" for x in _w],
                "Money": [f"${procurement_budget * x:,.0f}" for x in _w],
                "Our proxy mix (for comparison)": [f"{_proxy_share[c]:.1%}" for c in _feeds],
            }, index=_feeds))
            # daily equivalent, so it compares with the main app (which uses daily prices)
            _sig_day = _sig / np.sqrt(_gap)
            _sig_proxy = float(min_variance_portfolio(constrained_frontier)["volatility"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Waste share (RED2)", f"{sum(_w[i] for i in _widx):.1%}")
            c2.metric("Risk per row of your data", f"{_sig:.2%}",
                      help=f"One row of your file covers about {_gap:.0f} trading day(s).")
            c3.metric("Expected daily cost volatility", f"{_sig_day:.2%}",
                      delta=f"{(_sig_day - _sig_proxy) * 100:+.2f} pts vs proxy mix ({_sig_proxy:.2%})",
                      delta_color="inverse",
                      help=f"Risk per row ÷ √{_gap:.0f} trading days. Same unit as the main app’s "
                           f"‘Expected daily cost volatility’.")
            c4.metric("Cost-at-Risk (95%, 1 month)", f"${_car:,.0f}")

            if _has_sell:
                # stable margin: minimise the variance of (your costs - your selling price)
                _C = _r[_feeds + ["Selling price"]].cov().values
                def _mvar(w):
                    v = np.append(w, -1.0)
                    return v @ _C @ v
                _rm = _ominimize(_mvar, np.ones(_n) / _n, method="SLSQP", bounds=_bnds,
                                 constraints=_cons, options={"maxiter": 500, "ftol": 1e-12})
                st.markdown("**Stable-margin mix on your real selling price** (tracking error):")
                st.dataframe(pd.DataFrame({
                    "Stable-margin mix": [f"{x:.1%}" for x in _rm.x],
                    "Money": [f"${procurement_budget * x:,.0f}" for x in _rm.x],
                }, index=_feeds))
                _mm, _ms = np.sqrt(_mvar(_w)), np.sqrt(_mvar(_rm.x))
                st.caption(f"How much your margin moves per row: main mix {_mm:.2%}, stable-margin mix "
                           f"{_ms:.2%}, that is {1 - _ms / _mm:.0%} calmer. A small gain means none of "
                           f"your feedstocks follows your selling price closely: to protect your margin, "
                           f"a futures hedge on the selling side works better.")

            st.caption("Same method and same rules as the main app (minimum variance, RED2 minimum, cap per "
                       "feedstock), but on your real prices, so there is no basis risk from proxies. "
                       "Waste-based feedstocks: UCO and Tallow. Cost-at-Risk assumes normal returns.")


# --- what if you change your rules? (sensitivity table) ----------------------

with st.expander("What if you change your rules? (cap and waste minimum)"):
    from scipy.optimize import minimize as _wminimize
    from constraints import build_all_constraints as _wbac, get_bounds as _wgb

    @st.cache_data(show_spinner="Testing other rules...")
    def _whatif(cov, red, cap, n):
        # the calmest mix for one set of rules (same rules as the main app)
        res = _wminimize(lambda w: w @ cov @ w, np.ones(n) / n, method="SLSQP",
                         bounds=_wgb(concentration_cap=cap, n_assets=n),
                         constraints=_wbac(sustainability_min=red, concentration_cap=cap),
                         options={"maxiter": 500, "ftol": 1e-12})
        if not res.success:
            return None
        w = res.x
        return float(np.sqrt(w @ cov @ w)), float(sustainability_share(w)), w

    _wcov = np.asarray(cov_matrix)
    _wn = len(PORTFOLIO_TICKERS)
    _which = st.radio("Which rule do you want to test?",
                      ["The cap per feedstock", "The waste minimum (RED2)"], horizontal=True, key="wi_rule")

    if _which == "The cap per feedstock":
        _vals = [0.25, 0.30, 0.35, 0.40, 0.50, 1.00]
        _runs = [(sustainability_min, v) for v in _vals]
        _labels = ["No cap" if v == 1.0 else f"{v:.0%}" for v in _vals]
        _now = concentration_cap
        st.markdown(f"Your waste minimum stays at **{sustainability_min:.0%}**. Only the cap changes.")
    else:
        _vals = [0.0, 0.25, 0.40, 0.50, 0.60, 0.70, 0.75]
        _runs = [(v, concentration_cap) for v in _vals]
        _labels = [f"{v:.0%}" for v in _vals]
        _now = sustainability_min
        st.markdown(f"Your cap stays at **{concentration_cap:.0%}**. Only the waste minimum changes.")

    _base = _whatif(_wcov, sustainability_min, concentration_cap, _wn)
    _rows = []
    for (_red, _cap), _lab, _v in zip(_runs, _labels, _vals):
        _out = _whatif(_wcov, _red, _cap, _wn)
        _mark = "  ← your setting" if abs(_v - _now) < 1e-9 else ""
        if _out is None:
            _rows.append([_lab + _mark, "no feasible mix", "—", "—", "—", "Your rules can’t all hold at once"])
            continue
        _vol, _waste, _w = _out
        _car = procurement_budget * _vol * np.sqrt(21) * 1.645
        _diff = "—" if (_mark or _base is None) else f"{(_vol - _base[0]) * 100:+.2f} pts"
        _real = {"Soybean oil": _w[PORTFOLIO_TICKERS.index("ZL=F")],
                 "Rapeseed (soybeans)": _w[PORTFOLIO_TICKERS.index("ZS=F")],
                 "UCO (diesel)": _w[PORTFOLIO_TICKERS.index("HO=F")],
                 "Tallow (cattle + Darling)": _w[PORTFOLIO_TICKERS.index("LE=F")]
                                              + _w[PORTFOLIO_TICKERS.index("DAR")]}
        _top = max(_real.values())
        _biggest = (f"{_top:.0%} each" if max(_real.values()) - min(_real.values()) < 0.005
                    else f"{max(_real, key=_real.get)} {_top:.0%}")
        _rows.append([_lab + _mark, f"{_vol:.2%}", _diff, f"${_car:,.0f}", f"{_waste:.0%}",
                      _biggest])

    st.table(pd.DataFrame(_rows, columns=[
        "Setting", "Daily cost volatility", "vs your setting", "Cost-at-Risk (95%, 1 month)",
        "Waste share", "Biggest feedstock"]).set_index("Setting"))

    if _which == "The cap per feedstock":
        st.caption("A looser cap lowers your risk, but you depend more on one feedstock (see 'Biggest feedstock'). "
                   "A tighter cap spreads your supply, but your costs move more.")
    else:
        st.caption("A higher waste minimum is greener. It costs little while the calmest mix already has that much "
                   "waste, then more and more. Above 2 × the cap (UCO + tallow), no mix is possible.")
    st.caption("Each row is the calmest mix for those rules, on the same data as the main app. "
               "Cost-at-Risk uses your budget in the sidebar, 95% and 21 days.")
