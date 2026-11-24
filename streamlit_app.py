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

        **The extended universe already closes part of this gap:** choose
        "Extended universe (15 feedstocks)" under Recommended allocation to use
        real exchange prices for UCO (Rotterdam gasoil), rapeseed (Euronext)
        and canola (ICE).
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
        paid vendor. The exceptions come from Investing.com files (Yahoo does
        not carry them): the Rotterdam gasoil and Euronext rapeseed prices
        (basis-risk checks and extended universe), ICE canola and Bursa
        Malaysia palm oil (extended universe), and the Bloomberg Commodity and
        Bloomberg Agriculture indexes (benchmarks; downloaded from Yahoo when
        the files are missing). If you upload your own prices in the sidebar,
        they are used only in the "Your own data" box.

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
        - **CL=F (crude oil):** still downloaded with the others, but no
          longer used (the crack spread box was removed).
        - **Extended universe (15 feedstocks):** weekly prices (Friday
          close), so that different exchange holidays and monthly contract
          rolls do not create false jumps; weeks with a Euronext rapeseed
          roll are removed. Each price stays in its own currency (EUR, CAD,
          MYR, USD). The table below covers the main 4-feedstock data only.
        """
    )
    prov_df = pd.DataFrame(provenance).T
    prov_df["first_date"] = pd.to_datetime(prov_df["first_date"]).dt.date
    prov_df["last_date"] = pd.to_datetime(prov_df["last_date"]).dt.date
    st.dataframe(prov_df)

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


# --- 9-feedstock mix for the "4 / 9 feedstocks" switch (same method as the extended box) ---

_X9_FEEDS = {  # name: (source, file or ticker, waste-based?, tallow group?)
    "UCO (Rotterdam gasoil)": ("csv", "data/gasoil_ice.csv", True, False),
    "Tallow (live cattle)": ("yahoo", "LE=F", True, True),
    "Tallow (Darling)": ("yahoo", "DAR", True, True),
    "Pork fat / white grease (lean hogs)": ("yahoo", "HE=F", True, False),
    "Soybean oil": ("yahoo", "ZL=F", False, False),
    "Rapeseed (Euronext)": ("csv", "data/rapeseed_euronext.csv", False, False),
    "Canola (ICE)": ("csv", "data/canola_ice.csv", False, False),
    "Palm oil (Bursa Malaysia)": ("csv", "data/palm_bursa.csv", False, False),
    "Distillers corn oil (corn)": ("yahoo", "ZC=F", False, False),
    "Soybeans (CBOT)": ("yahoo", "ZS=F", False, False),
    "UCO (heating oil)": ("yahoo", "HO=F", True, False),
    "Tallow (feeder cattle)": ("yahoo", "GF=F", True, True),
    "Neste (company proxy)": ("yahoo", "NESTE.HE", False, False),
    "Verbio (company proxy)": ("yahoo", "VBK.DE", False, False),
    "Bunge (company proxy)": ("yahoo", "BG", False, False),
}
_X9_SHORT = {
    "UCO (Rotterdam gasoil)": "UCO\n(gasoil)", "Tallow (live cattle)": "Tallow\n(cattle)",
    "Tallow (Darling)": "Tallow\n(Darling)", "Pork fat / white grease (lean hogs)": "Pork fat\n(hogs)",
    "Soybean oil": "Soybean\noil", "Rapeseed (Euronext)": "Rapeseed\n(Euronext)", "Canola (ICE)": "Canola\n(ICE)",
    "Palm oil (Bursa Malaysia)": "Palm oil\n(Bursa)", "Distillers corn oil (corn)": "Corn oil\n(corn)",
    "Soybeans (CBOT)": "Soybeans\n(CBOT)", "UCO (heating oil)": "UCO\n(heat. oil)",
    "Neste (company proxy)": "Neste\n(stock)", "Verbio (company proxy)": "Verbio\n(stock)",
    "Bunge (company proxy)": "Bunge\n(stock)",
    "Tallow (feeder cattle)": "Tallow\n(feeder)", "Rapeseed oil (ZCE China)": "Rapes. oil\n(China)",
    "Yellow grease (USDA)": "Yellow\ngrease",
}
_X9_COLORS = {
    "UCO (Rotterdam gasoil)": "#C0392B", "Tallow (live cattle)": "#8E5FB0", "Tallow (Darling)": "#8B4513",
    "Pork fat / white grease (lean hogs)": "#D98880", "Soybean oil": "#4A9A34", "Rapeseed (Euronext)": "#3A6BA5",
    "Canola (ICE)": "#D4AC0D", "Palm oil (Bursa Malaysia)": "#E67E22", "Distillers corn oil (corn)": "#7F8C8D",
    "Soybeans (CBOT)": "#1F4E79", "UCO (heating oil)": "#E74C3C", "Tallow (feeder cattle)": "#5B2C6F",
    "Rapeseed oil (ZCE China)": "#5DADE2", "Yellow grease (USDA)": "#F4D03F",
    "Neste (company proxy)": "#16A085", "Verbio (company proxy)": "#A3C639", "Bunge (company proxy)": "#B9770E",
}


@st.cache_data(ttl=86400, show_spinner="Preparing the 9-feedstock mix...")
def _x9_mix(years, red, cap, feeds_key=(), goal="minvar"):
    import yfinance as yf
    from scipy.optimize import minimize as _min
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)
    prices, missing = {}, []
    for name, (src, ref, _w, _t) in _X9_FEEDS.items():
        sr = None
        try:
            if src == "csv":
                d = pd.read_csv(ref, thousands=",")
                d["Date"] = pd.to_datetime(d["Date"], format="%m/%d/%Y")
                sr = d.set_index("Date").sort_index()["Price"].astype(float)
                sr = sr[sr.index >= sr.index.max() - pd.DateOffset(years=years)]
            else:
                d = yf.download(ref, start=start, end=end, auto_adjust=True, progress=False, threads=False)
                if d is not None and not d.empty:
                    c = d["Close"]
                    sr = (c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c).dropna()
        except Exception:
            sr = None
        if sr is None or len(sr) < 250:
            missing.append(name)
        else:
            prices[name] = sr
    if len(prices) < 4:
        return None
    px = pd.DataFrame(prices).sort_index()
    px = px[px.index >= px.dropna(how="any").index.min()].ffill(limit=5).dropna()
    r = px.resample("W-FRI").last().pct_change().dropna()
    if "Rapeseed (Euronext)" in r.columns:
        r = r[~(r.index.month.isin([2, 5, 8, 11]) & (r.index.day <= 9))]
    names = list(r.columns)
    n = len(names)
    X = r.values - r.values.mean(axis=0)
    T = X.shape[0]
    emp = X.T @ X / T
    mu = np.trace(emp) / n
    delta = ((emp - mu * np.eye(n)) ** 2).sum() / n
    beta = min(((X ** 2).T @ (X ** 2) / T - emp ** 2).sum() / (n * T), delta)
    shrink = float(beta / delta) if delta > 0 else 0.0
    S_shrunk = (1 - shrink) * emp + shrink * mu * np.eye(n)
    waste = np.array([1.0 if _X9_FEEDS[k][2] else 0.0 for k in names])
    tallow = np.array([1.0 if _X9_FEEDS[k][3] else 0.0 for k in names])
    cons = [{"type": "eq", "fun": lambda x: x.sum() - 1},
            {"type": "ineq", "fun": lambda x: x @ waste - red},
            {"type": "ineq", "fun": lambda x: cap - x @ tallow}]
    res = _min(lambda x: x @ S_shrunk @ x, np.ones(n) / n, method="SLSQP", bounds=[(0.0, cap)] * n,
               constraints=cons, options={"maxiter": 1000, "ftol": 1e-14})
    if not res.success:
        return None
    w = np.clip(res.x, 0, None)
    mu = r.mean().values  # average weekly price change of each feedstock
    used = "minvar"
    if goal == "sharpe":
        # Sharpe-equivalent (rf = 0): highest average price change per unit of risk, same rules, shrunk covariance
        def _neg_sr(x):
            return -(x @ mu) / np.sqrt(max(x @ S_shrunk @ x, 1e-18))
        best = None
        for x0 in (w / w.sum(), np.ones(n) / n):
            rs = _min(_neg_sr, x0, method="SLSQP", bounds=[(0.0, cap)] * n, constraints=cons,
                      options={"maxiter": 1000, "ftol": 1e-12})
            if rs.success and (best is None or rs.fun < best.fun):
                best = rs
        if best is not None and -best.fun > 0:
            w = np.clip(best.x, 0, None)
            used = "sharpe"
    return {"names": names, "w": w / w.sum(), "S": r.cov().values, "waste": waste, "tallow": tallow,
            "missing": missing, "weeks": T, "shrink": shrink, "mu": mu, "goal": used, "Ss": S_shrunk, "r": r}

@st.cache_data(ttl=86400, show_spinner="Building the 15-feedstock frontier...")
def _x15_frontier(years, red, cap, feeds_key=(), npts=50):
    # same data as the 15-feedstock mix: weekly returns, Ledoit-Wolf shrinkage (LN 3)
    import yfinance as yf
    from scipy.optimize import minimize as _min, linprog as _lp
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)
    prices = {}
    for name, (src, ref, _w, _t) in _X9_FEEDS.items():
        try:
            if src == "csv":
                d = pd.read_csv(ref, thousands=",")
                d["Date"] = pd.to_datetime(d["Date"], format="%m/%d/%Y")
                sr = d.set_index("Date").sort_index()["Price"].astype(float)
                sr = sr[sr.index >= sr.index.max() - pd.DateOffset(years=years)]
            else:
                d = yf.download(ref, start=start, end=end, auto_adjust=True, progress=False, threads=False)
                c = d["Close"]
                sr = (c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c).dropna()
            if len(sr) >= 250:
                prices[name] = sr
        except Exception:
            pass
    if len(prices) < 4:
        return None
    px = pd.DataFrame(prices).sort_index()
    px = px[px.index >= px.dropna(how="any").index.min()].ffill(limit=5).dropna()
    r = px.resample("W-FRI").last().pct_change().dropna()
    if "Rapeseed (Euronext)" in r.columns:
        r = r[~(r.index.month.isin([2, 5, 8, 11]) & (r.index.day <= 9))]
    names = list(r.columns)
    n = len(names)
    X = r.values - r.values.mean(axis=0)
    T = X.shape[0]
    emp = X.T @ X / T
    m = np.trace(emp) / n
    delta = ((emp - m * np.eye(n)) ** 2).sum() / n
    beta = min(((X ** 2).T @ (X ** 2) / T - emp ** 2).sum() / (n * T), delta)
    sh = float(beta / delta) if delta > 0 else 0.0
    S = (1 - sh) * emp + sh * m * np.eye(n)
    S_raw = r.cov().values
    mu = r.mean().values
    waste = np.array([1.0 if _X9_FEEDS[k][2] else 0.0 for k in names])
    tallow = np.array([1.0 if _X9_FEEDS[k][3] else 0.0 for k in names])

    def frontier(rules):
        bnds = [(0.0, cap)] * n if rules else [(0.0, 1.0)] * n
        cons = [{"type": "eq", "fun": lambda x: x.sum() - 1}]
        A_ub, b_ub = None, None
        if rules:
            cons += [{"type": "ineq", "fun": lambda x: x @ waste - red},
                     {"type": "ineq", "fun": lambda x: cap - x @ tallow}]
            A_ub, b_ub = np.vstack([-waste, tallow]), np.array([-red, cap])
        lo = _lp(mu, A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, n)), b_eq=[1], bounds=bnds, method="highs")
        hi = _lp(-mu, A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, n)), b_eq=[1], bounds=bnds, method="highs")
        if not (lo.success and hi.success):
            return None
        vols, rets, x0 = [], [], np.ones(n) / n
        for t in np.linspace(mu @ lo.x, mu @ hi.x, npts):
            res = _min(lambda x: x @ S @ x, x0, method="SLSQP", bounds=bnds,
                       constraints=cons + [{"type": "eq", "fun": lambda x, t=t: x @ mu - t}],
                       options={"maxiter": 500, "ftol": 1e-12})
            if res.success:
                vols.append(np.sqrt(res.x @ S_raw @ res.x / 5)); rets.append(t / 5); x0 = res.x
        mv = _min(lambda x: x @ S @ x, np.ones(n) / n, method="SLSQP", bounds=bnds, constraints=cons,
                  options={"maxiter": 1000, "ftol": 1e-14})
        mvp = (float(np.sqrt(mv.x @ S_raw @ mv.x / 5)), float(mu @ mv.x / 5)) if mv.success else None
        return {"vols": np.array(vols), "rets": np.array(rets), "mv": mvp}

    return {"rules": frontier(True), "free": frontier(False), "n": n, "shrink": sh}


def _beautify_frontier(ax):
    # one clean style for both frontier charts
    ax.grid(True, color="#E9E9E9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9A9A9A")
    ax.tick_params(colors="#444444", labelsize=9)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=3))
    ax.xaxis.label.set_color("#333333")
    ax.yaxis.label.set_color("#333333")
    ax.title.set_fontsize(13)
    ax.title.set_color("#222222")
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.95, edgecolor="#DDDDDD")
    ax.figure.tight_layout()


# --- frontier plot ---------------------------------------------------------

st.header("Efficient frontier: constrained vs. unconstrained")

_front15 = st.radio("Feedstocks in this chart:", ["4 feedstocks (main recommendation)", "Extended universe (15 feedstocks)"],
                    horizontal=True, key="front15")
_f15 = None
if _front15.startswith("Extended"):
    _f15 = _x15_frontier(int(years_lookback), float(sustainability_min), float(concentration_cap),
                         tuple(_X9_FEEDS), int(n_frontier_points))
    if _f15 is None or _f15["rules"] is None or _f15["free"] is None:
        st.warning("The 15-feedstock frontier could not be built right now. Showing the 4-feedstock frontier.")
        _f15 = None

if _f15 is not None:
    _fr, _ff = _f15["rules"], _f15["free"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(_ff["vols"], _ff["rets"], label="Unconstrained (budget + long-only)", linestyle="--", color="#9A9A9A", linewidth=1.6)
    ax.plot(_fr["vols"], _fr["rets"], label="Constrained (RED2 + concentration cap)", color="#1F5FA8", linewidth=2.6)
    if _ff["mv"]:
        ax.scatter([_ff["mv"][0]], [_ff["mv"][1]], color="gray", s=45, zorder=5, label="Safest mix with no rules")
    if _fr["mv"]:
        ax.scatter([_fr["mv"][0]], [_fr["mv"][1]], color="#D62728", s=90, zorder=6, edgecolor="black",
                   label="Safest mix with your rules (15 feedstocks)")
        ax.annotate(f"Safest mix with your rules\n{_fr['mv'][0]:.2%} daily risk", (_fr["mv"][0], _fr["mv"][1]),
                    xytext=(36, 42), textcoords="offset points", arrowprops=dict(arrowstyle="-", color="#D62728", lw=0.8), bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#F2C4C4", alpha=0.95), fontsize=9, color="#D62728", va="center")
    _mv4 = min_variance_portfolio(constrained_frontier)
    ax.scatter([_mv4["volatility"]], [_mv4["return"]], marker="x", color="#D62728", s=70, zorder=6,
               label=f"Our 4-feedstock recommendation ({_mv4['volatility']:.2%})")
    ax.set_xlabel("Daily cost volatility (how much your costs move each day)")
    ax.set_ylabel("Expected daily price change")
    ax.set_title(f"Cost of your real-world rules ({_f15['n']} feedstocks)")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
    _beautify_frontier(ax)
    st.pyplot(fig)
    st.caption(
        f"The same chart with {_f15['n']} feedstocks (weekly data ÷ 5 to compare with daily numbers, Ledoit-Wolf "
        f"shrinkage {_f15['shrink']:.0%}). Both curves move to the left: with more feedstocks to choose from, the "
        "safest mix is calmer, with or without your rules. The red cross is our 4-feedstock recommendation. "
        "Average price changes over a few years are noisy, so the shape of the upper and lower parts can move a "
        "lot when the data changes; the left end (the safest mix) is the reliable part."
    )
else:
    fig, ax = plt.subplots(figsize=(9, 5))
    valid_u = unconstrained_frontier["success"]
    ax.plot(unconstrained_frontier["volatilities"][valid_u], unconstrained_frontier["returns"][valid_u],
            label="Unconstrained (budget + long-only)", linestyle="--", color="#9A9A9A", linewidth=1.6)
    valid_c = constrained_frontier["success"]
    ax.plot(constrained_frontier["volatilities"][valid_c], constrained_frontier["returns"][valid_c],
            label="Constrained (RED2 + concentration cap)", color="#1F5FA8", linewidth=2.6)
    _mv_c = min_variance_portfolio(constrained_frontier)
    _mv_u = min_variance_portfolio(unconstrained_frontier)
    ax.scatter([_mv_u["volatility"]], [_mv_u["return"]], color="gray", s=45, zorder=5,
               label="Safest mix with no rules")
    ax.scatter([_mv_c["volatility"]], [_mv_c["return"]], color="#D62728", s=90, zorder=6, edgecolor="black",
               label="Our recommendation: safest mix with your rules")
    ax.annotate(f"Our recommendation\n{_mv_c['volatility']:.2%} daily risk",
                (_mv_c["volatility"], _mv_c["return"]), xytext=(30, 28), textcoords="offset points", arrowprops=dict(arrowstyle="-", color="#D62728", lw=0.8), bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#F2C4C4", alpha=0.95),
                fontsize=9, color="#D62728", va="center")
    ax.set_xlabel("Daily cost volatility (how much your costs move each day)")
    ax.set_ylabel("Expected daily price change")
    ax.set_title("Cost of your real-world rules")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
    _beautify_frontier(ax)
    st.pyplot(fig)

st.caption(
    "The red dot is our recommendation: the safest mix that follows your rules. The grey dot is the "
    "safest mix with no rules. The distance between them is the cost of your rules. At the default settings, the safest mix already holds more than 40% waste, "
    "so most of this cost comes from the 35% cap per feedstock. For an investor, the upper half of "
    "the curve is efficient. For you, as a buyer, a higher point means prices rising faster, so if "
    "anything the bottom half would be better. But we don't bet on past average price changes: we "
    "take the leftmost point of the blue line, where your costs move the least."
)

st.caption(
    "**Why is the grey dot calmer than ours?** Diversification lowers risk compared with any single "
    "feedstock or an equal split. The grey dot is even calmer because it concentrates on the calmest "
    "feedstock. Your cap trades a little of that for not depending on one supplier: diversification of "
    "supply, not just of price risk."
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


# --- explore the curves point by point -----------------------------------------

with st.expander("Explore the curves point by point"):
    _ex_names = {"ZL=F": "Soybean oil", "ZS=F": "Rapeseed (soybeans)", "HO=F": "UCO (diesel)",
                 "LE=F": "Tallow (live cattle)", "DAR": "Tallow (Darling)"}
    _ex_short = {"ZL=F": "Soybean oil", "ZS=F": "Soybeans", "HO=F": "UCO", "LE=F": "Live cattle", "DAR": "Darling"}
    _ex_lines = ["The blue line (your rules)", "The grey line (no rules)"]
    _sh_pick = max_sharpe_equivalent_portfolio(constrained_frontier)

    def _ex_blue_pos(target):
        _f = constrained_frontier
        _ids = sorted([i for i in range(len(_f["success"])) if _f["success"][i]], key=lambda i: -_f["returns"][i])
        return _ids.index(int(target)) if int(target) in _ids else 0

    def _ex_go(target):
        st.session_state["ex_line"] = _ex_lines[0]
        st.session_state["ex_pos_blue"] = _ex_blue_pos(target)

    _xb1, _xb2, _xb3 = st.columns([1, 1, 2])
    _xb1.button("Go to the red dot", key="ex_go_red",
                on_click=_ex_go, args=(min_variance_portfolio(constrained_frontier)["index"],))
    _xb2.button("Go to the Sharpe point", key="ex_go_sharpe", on_click=_ex_go, args=(_sh_pick["index"],))
    if "ex_line" not in st.session_state:
        st.session_state["ex_line"] = _ex_lines[0]
    _ex_line = st.radio("Slide along:", _ex_lines, horizontal=True, key="ex_line")
    _ex_is_grey = _ex_line.startswith("The grey")
    _ex_f = unconstrained_frontier if _ex_is_grey else constrained_frontier
    _ex_idx = sorted([i for i in range(len(_ex_f["success"])) if _ex_f["success"][i]],
                     key=lambda i: -_ex_f["returns"][i])
    _ex_mv = int(min_variance_portfolio(_ex_f)["index"])
    _ex_start = _ex_idx.index(_ex_mv) if _ex_mv in _ex_idx else 0
    _ex_key = "ex_pos_grey" if _ex_is_grey else "ex_pos_blue"
    if _ex_key not in st.session_state or st.session_state[_ex_key] > len(_ex_idx) - 1:
        st.session_state[_ex_key] = _ex_start
    _ex_pos = st.slider("Move along the line (left = top of the curve, right = bottom of the curve)",
                        0, len(_ex_idx) - 1, key=_ex_key)
    _ex_i = _ex_idx[_ex_pos]
    _ex_w = np.asarray(_ex_f["weights"][_ex_i], dtype=float)
    _ex_vol, _ex_ret = float(_ex_f["volatilities"][_ex_i]), float(_ex_f["returns"][_ex_i])
    _sd = np.sqrt(np.diag(np.asarray(cov_matrix)))
    _mu = np.asarray(mean_returns)

    fig_ex, ax_ex = plt.subplots(figsize=(9, 5))
    _vu, _vc = unconstrained_frontier["success"], constrained_frontier["success"]
    ax_ex.plot(unconstrained_frontier["volatilities"][_vu], unconstrained_frontier["returns"][_vu],
               linestyle="--", color="gray", label="Grey line: no rules")
    ax_ex.plot(constrained_frontier["volatilities"][_vc], constrained_frontier["returns"][_vc],
               color="steelblue", linewidth=2, label="Blue line: your rules")
    for _k, _t in enumerate(PORTFOLIO_TICKERS):
        _on = _ex_w[_k] > 0.005
        ax_ex.scatter(_sd[_k], _mu[_k], marker="s", s=70, color=TICKER_COLORS[_t], edgecolor="black",
                      alpha=1.0 if _on else 0.3, zorder=4)
        ax_ex.annotate(_ex_short[_t], (_sd[_k], _mu[_k]), xytext=(7, 5), textcoords="offset points", fontsize=9)
    _g_mv, _r_mv = min_variance_portfolio(unconstrained_frontier), min_variance_portfolio(constrained_frontier)
    ax_ex.scatter(_g_mv["volatility"], _g_mv["return"], color="gray", s=45, zorder=5, label="Grey dot: safest, no rules")
    ax_ex.scatter(_r_mv["volatility"], _r_mv["return"], color="#D62728", s=80, edgecolor="black", zorder=6,
                  label="Red dot: our recommendation")
    _sh_slope = _sh_pick["return"] / _sh_pick["volatility"]
    _x_line = np.array([0.0, max(_sd) * 1.02])
    ax_ex.plot(_x_line, _sh_slope * _x_line, linestyle=":", color="#7F77DD", linewidth=1.5,
               label="Steepest line from zero")
    ax_ex.scatter(_sh_pick["volatility"], _sh_pick["return"], marker="D", color="#7F77DD", s=80, edgecolor="black",
                  zorder=6, label="Sharpe-equivalent: best price change per risk")
    ax_ex.scatter(_ex_vol, _ex_ret, s=260, facecolors="none", edgecolors="black", linewidths=2.2, zorder=7,
                  label="You are here")
    _xl = ax_ex.get_xlim()
    ax_ex.set_xlim(max(_xl[0], min(_sd) * 0.6), _xl[1])
    ax_ex.set_ylim(min(0.0, float(min(_mu))) - 0.00002, float(max(_mu)) * 1.12)
    ax_ex.set_xlabel("Daily cost volatility (how much your costs move each day)")
    ax_ex.set_ylabel("Expected daily price change")
    ax_ex.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
    ax_ex.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
    ax_ex.legend(fontsize=8, loc="best")
    st.pyplot(fig_ex)

    _ex_waste = float(sum(_ex_w[PORTFOLIO_TICKERS.index(t)] for t in WASTE_BASED if t in PORTFOLIO_TICKERS))
    _ex_tallow = _ex_w[PORTFOLIO_TICKERS.index("LE=F")] + _ex_w[PORTFOLIO_TICKERS.index("DAR")]
    _ex_big = max([_ex_w[PORTFOLIO_TICKERS.index(t)] for t in PORTFOLIO_TICKERS if t not in ("LE=F", "DAR")]
                  + [_ex_tallow])
    _ex_ok_w = _ex_waste >= sustainability_min - 1e-3
    _ex_ok_c = _ex_big <= concentration_cap + 1e-3
    e1, e2, e5, e3, e4 = st.columns(5)
    e1.metric("Daily risk", f"{_ex_vol:.2%}")
    e2.metric("Expected daily price change", f"{_ex_ret:.3%}")
    e5.metric("Price change per risk", f"{_ex_ret / _ex_vol:.4f}",
              help="Expected daily price change ÷ daily risk: the Sharpe ratio with a risk-free rate of 0. "
                   "The Sharpe-equivalent point has the highest value on the blue line.")
    e3.metric(f"Waste share (min {sustainability_min:.0%})", f"{_ex_waste:.0%} {'✅' if _ex_ok_w else '❌'}")
    e4.metric(f"Largest feedstock (cap {concentration_cap:.0%})", f"{_ex_big:.0%} {'✅' if _ex_ok_c else '❌'}",
              help="Tallow counts once: live cattle + Darling.")

    if _ex_i == _ex_mv:
        _ex_where = ("Grey dot: the safest mix with no rules." if _ex_is_grey
                     else "Red dot: our recommendation, the safest mix that follows your rules.")
    elif _ex_ret > _ex_f["returns"][_ex_mv]:
        _ex_where = "Upper half of the curve: prices rose faster, and your costs move more."
    else:
        _ex_where = "Bottom half of the curve: prices rose a little less, but your costs move more."
    if not _ex_is_grey and _ex_i == int(_sh_pick["index"]):
        _ex_where = ("Sharpe-equivalent: the highest price change per risk on the blue line. Good for an "
                     "investor, but for you it means more risk and faster-rising prices.")
    if _ex_is_grey and _ex_w[PORTFOLIO_TICKERS.index("DAR")] > 0.995:
        _ex_where = "Right end of the grey line: 100% Darling."
    st.markdown(f"**Where you are:** {_ex_where}")

    st.dataframe(pd.DataFrame({
        "Share of your budget": [round(float(x) * 100, 1) for x in _ex_w],
        "Daily risk alone": [f"{_sd[k]:.2%}" for k in range(len(PORTFOLIO_TICKERS))],
        "Average daily price change alone": [f"{_mu[k]:.3%}" for k in range(len(PORTFOLIO_TICKERS))],
    }, index=[_ex_names[t] for t in PORTFOLIO_TICKERS]),
        column_config={"Share of your budget": st.column_config.ProgressColumn(
            "Share of your budget", min_value=0, max_value=100, format="%.0f%%")})
    st.caption("The squares are each feedstock alone; faded squares are not in the mix at this point. "
               "On the blue line, moving away from the red dot only adds risk. The grey dot breaks your cap: "
               "it puts most of your money in live cattle. The flat end of the grey line is the mix sliding "
               "toward 100% Darling, which your cap forbids.")


# --- recommended allocation ------------------------------------------------

st.header("Recommended allocation")

_view9 = st.radio(
    "Feedstocks shown here and in risk budgeting:",
    ["4 feedstocks (main recommendation)", "Extended universe (15 feedstocks)"],
    horizontal=True, key="view9",
    help="The 9-feedstock view uses real European prices for UCO and rapeseed, adds canola, palm oil, pork fat "
         "and corn oil, and uses shrinkage (LN 3). Hedging, Black-Litterman and the other boxes stay on 4 feedstocks.",
)
_x9 = None
if not _view9.startswith("4"):
    _x9 = _x9_mix(int(years_lookback), float(sustainability_min), float(concentration_cap), tuple(_X9_FEEDS))
    if _x9 is None:
        st.warning("The 9-feedstock mix could not be built right now (data missing or no mix follows your "
                   "rules). Showing the 4-feedstock recommendation.")

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

_x9_want = "sharpe" if pick_method.startswith("Best") else "minvar"
if _x9 is not None and _x9_want == "sharpe":
    _x9 = _x9_mix(int(years_lookback), float(sustainability_min), float(concentration_cap), tuple(_X9_FEEDS),
                  "sharpe") or _x9

if _x9 is not None:
    _n9, _w9, _S9 = _x9["names"], _x9["w"], _x9["S"]
    _o9 = [k for k in _n9 if _X9_FEEDS[k][2]] + [k for k in _n9 if not _X9_FEEDS[k][2]]
    _i9 = [_n9.index(k) for k in _o9]
    _nw9 = sum(1 for k in _o9 if _X9_FEEDS[k][2])
    if _x9.get("goal") == "sharpe":
        st.caption("Best return-per-risk (Sharpe-equivalent) mix of the extended universe: weekly data, shrinkage, "
                   "same rules. Average price changes over a few years are noisy (and include 3 company share "
                   "prices), so this mix can move a lot when the data changes: treat it as illustrative.")
    else:
        st.caption("Minimum-variance mix of the extended universe (weekly data, shrinkage)."
                   + (" No mix with a positive return-per-risk was found, so the minimum-variance mix is shown."
                      if _x9_want == "sharpe" else ""))
    if _x9["missing"]:
        st.warning("Not available right now (skipped): " + ", ".join(_x9["missing"]) + ".")
    c9a, c9b = st.columns([2, 1])
    with c9a:
        fig9, ax9 = plt.subplots(figsize=(10, 4.6))
        _b9 = ax9.bar([_X9_SHORT.get(k, k) for k in _o9], _w9[_i9], color=[_X9_COLORS.get(k, "#999999") for k in _o9])
        ax9.bar_label(_b9, labels=[f"{v:.1%}" for v in _w9[_i9]], padding=3, fontsize=8)
        ax9.axvline(_nw9 - 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        _top9 = max(concentration_cap, float(_w9.max())) * 1.3
        ax9.set_ylim(0, _top9)
        ax9.text((_nw9 - 1) / 2, _top9 * 0.89, "Waste-based (RED2)", ha="center", fontsize=8, color="dimgray")
        ax9.text(_nw9 + (len(_o9) - _nw9 - 1) / 2, _top9 * 0.89, "Crop-based", ha="center", fontsize=8,
                 color="dimgray")
        ax9.set_ylabel("Portfolio weight")
        ax9.set_title(f"Feedstock allocation ({len(_n9)} feedstocks)")
        ax9.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax9.tick_params(axis="x", labelsize=7)
        fig9.tight_layout()
        st.pyplot(fig9)
    with c9b:
        st.metric("Expected daily cost volatility", f"{np.sqrt(_w9 @ _S9 @ _w9 / 5):.2%}",
                  help="Weekly risk ÷ √5, so it compares with the 4-feedstock daily number.")
        _ret9 = float(_x9["mu"] @ _w9) / 5
        st.metric("Expected daily price change", f"{_ret9:.4%}", help="Average weekly price change ÷ 5.")
        if _x9.get("goal") == "sharpe":
            st.metric("Return-per-risk (Sharpe-equivalent)", f"{_ret9 / np.sqrt(_w9 @ _S9 @ _w9 / 5):.4f}")
        _ws9 = float(_w9 @ _x9["waste"])
        st.metric("Sustainability share achieved", f"{_ws9:.1%}", help=f"Target minimum: {sustainability_min:.0%}")
        st.markdown(f"{'✅ **PASS**' if _ws9 >= sustainability_min - 1e-6 else '❌ **FAIL**'} -- RED2 sustainability minimum")
        _r9 = {k: w for k, w in zip(_n9, _w9) if not _X9_FEEDS[k][3]}
        _r9["Tallow"] = float(_w9 @ _x9["tallow"])
        _big9 = max(_r9.values())
        st.metric("Largest single feedstock weight", f"{_big9:.1%}",
                  help="Tallow counts as one feedstock: live cattle + Darling together.")
        st.markdown(f"{'✅ **PASS**' if _big9 <= concentration_cap + 1e-6 else '❌ **FAIL**'} -- concentration cap")
        st.caption(f"At a ${procurement_budget:,.0f} budget:")
        st.dataframe(pd.DataFrame({
            "Share": [f"{_w9[i]:.1%}" for i in _i9],
            "Money": [f"${_w9[i] * procurement_budget:,.0f}" for i in _i9],
        }, index=_o9))
else:
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
        st.metric("Expected daily cost volatility", f"{recommended['volatility']:.2%}")
        st.metric("Expected daily price change", f"{recommended['return']:.4%}")

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
        st.dataframe(pd.DataFrame({
            "Share": [f"{w:.1%}" for w in weights],
            "Money": [f"${w * procurement_budget:,.0f}" for w in weights],
        }, index=[TICKER_LABELS[t] for t in PORTFOLIO_TICKERS]))


# --- risk budgeting ----------------------------------------------------

st.header("Risk budgeting: where the risk actually comes from")

if _x9 is not None:
    _rc9 = _w9 * (_S9 @ _w9) / (_w9 @ _S9 @ _w9)
    fig9r, ax9r = plt.subplots(figsize=(10, 4.2))
    _b9r = ax9r.bar([_X9_SHORT.get(k, k) for k in _o9], _rc9[_i9], color=[_X9_COLORS.get(k, "#999999") for k in _o9])
    ax9r.bar_label(_b9r, labels=[f"{v:.1%}" for v in _rc9[_i9]], padding=3, fontsize=8)
    ax9r.set_ylim(min(0.0, float(_rc9.min()) * 1.25), float(_rc9.max()) * 1.25)
    ax9r.set_ylabel("Share of your cost risk")
    ax9r.set_title(f"Risk contribution per feedstock ({len(_n9)} feedstocks)")
    ax9r.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax9r.tick_params(axis="x", labelsize=7)
    fig9r.tight_layout()
    st.pyplot(fig9r)
    # caption written from the numbers, so it stays true at any setting
    _held9 = [i for i in range(len(_n9)) if _w9[i] >= 0.02]
    _gap9 = {i: _rc9[i] - _w9[i] for i in _held9}
    _hi9, _lo9 = max(_gap9, key=_gap9.get), min(_gap9, key=_gap9.get)
    _tfull9 = float(_w9 @ _x9["tallow"]) >= concentration_cap - 1e-3
    _locap9 = _w9[_lo9] >= concentration_cap - 1e-3 or (_X9_FEEDS[_n9[_lo9]][3] and _tfull9)
    _m9 = (f"Each bar shows how much of your cost risk comes from each feedstock. This is not the same as how much "
           f"you buy: {_n9[_hi9]} is {_w9[_hi9]:.0%} of your budget but {_rc9[_hi9]:.0%} of your risk, because its "
           f"price moves more or moves with the others.")
    if _lo9 != _hi9:
        _m9 += (f" {_n9[_lo9]} is {_w9[_lo9]:.0%} of your budget but only {_rc9[_lo9]:.0%} of your risk: "
                + ("it is your calmest feedstock, and the best mix would buy more of it if your cap allowed."
                   if _locap9 else "it calms the mix because it moves differently from the others."))
    st.caption(_m9)
else:
    rc = risk_contributions(weights, cov_matrix)
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    _short = {"ZL=F": "Soybean oil", "ZS=F": "Rapeseed\n(soybeans)", "HO=F": "UCO\n(diesel)",
              "LE=F": "Tallow\n(live cattle)", "DAR": "Tallow\n(Darling)"}
    bars3 = ax3.bar([_short.get(t, t) for t in PORTFOLIO_TICKERS], rc, color=bar_colors)
    ax3.bar_label(bars3, labels=[f"{v:.1%}" for v in rc], padding=3, fontsize=9)
    ax3.set_ylabel("Share of your cost risk")
    ax3.set_title("Risk contribution per feedstock")
    ax3.set_ylim(0, rc.max() * 1.25)  # headroom for the labels
    ax3.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    st.pyplot(fig3)

    _i_zl, _i_le = PORTFOLIO_TICKERS.index("ZL=F"), PORTFOLIO_TICKERS.index("LE=F")
    st.caption(
        f"Each bar shows how much of your cost risk comes from each feedstock. This is not the same as how "
        f"much you buy: soybean oil is {weights[_i_zl]:.0%} of your budget but {rc[_i_zl]:.0%} of your risk, "
        f"because its price jumps more and moves with your rapeseed. Live cattle (tallow) is "
        f"{weights[_i_le]:.0%} of your budget but only {rc[_i_le]:.0%} of your risk: it is your calmest "
        f"feedstock, and the best mix would buy more of it if your cap allowed."
    )


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
    }, index=[TICKER_LABELS[t] for t in PORTFOLIO_TICKERS] + ["Daily risk"]))
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
    horizon = car_horizon if "car_horizon" in globals() else 21
    conf = car_conf if "car_conf" in globals() else "95%"
    z_val = 1.645 if conf == "95%" else 2.326

    _why_view = st.radio("Feedstocks in this comparison:",
                         ["4 feedstocks (main recommendation)", "Extended universe (15 feedstocks)"],
                         horizontal=True, key="why15")

    def _pick4(p):
        rc_p = risk_contributions(p["weights"], cov_matrix)
        top = int(np.argmax(rc_p))
        return {"vol": p["volatility"], "ret": p["return"],
                "top": f"{TICKER_LABELS[PORTFOLIO_TICKERS[top]]} ({rc_p[top]:.0%})",
                "waste": sustainability_share(p["weights"])}

    def _pick15(x):
        w, S = x["w"], x["S"]
        rc_p = w * (S @ w) / (w @ S @ w)
        top = int(np.argmax(rc_p))
        return {"vol": float(np.sqrt(w @ S @ w / 5)), "ret": float(x["mu"] @ w / 5),
                "top": f"{x['names'][top]} ({rc_p[top]:.0%})", "waste": float(w @ x["waste"])}

    mv_row = sr_row = None
    if _why_view.startswith("Extended"):
        _a = _x9_mix(int(years_lookback), float(sustainability_min), float(concentration_cap), tuple(_X9_FEEDS))
        _b = _x9_mix(int(years_lookback), float(sustainability_min), float(concentration_cap), tuple(_X9_FEEDS),
                     "sharpe")
        if _a is not None and _b is not None and _b.get("goal") == "sharpe":
            mv_row, sr_row = _pick15(_a), _pick15(_b)
        else:
            st.warning("The 15-feedstock comparison could not be built right now. Showing 4 feedstocks.")
    if mv_row is None:
        mv_row = _pick4(min_variance_portfolio(constrained_frontier))
        sr_row = _pick4(max_sharpe_equivalent_portfolio(constrained_frontier))

    def _why_row(r):
        return [
            f"{r['vol']:.2%}",
            f"{r['ret']:.3%}",
            f"{(1 + r['ret']) ** 252 - 1:+.1%}",
            f"${procurement_budget * r['vol'] * np.sqrt(horizon) * z_val:,.0f}",
            f"{r['ret'] / r['vol']:.3f}",
            r["top"],
            f"{r['waste']:.0%}",
        ]

    st.markdown(
        "You are a **buyer**: a higher 'return' means feedstock prices going up, "
        "so higher costs. Your goal is **stable costs**, which is what minimum variance gives."
    )
    st.dataframe(pd.DataFrame({
        "Minimum variance (recommended)": _why_row(mv_row),
        "Sharpe-equivalent": _why_row(sr_row),
    }, index=[
        "Daily risk: σ(x) = √(xᵀΣx)",
        "Daily 'return' = price increase: μ(x) = xᵀμ",
        "Price increase over a year: (1 + μ)^252 − 1",
        f"Cost-at-Risk ({conf}, {horizon} days): B × σ × √h × z",
        "Price increase per risk: μ / σ (lower is better for you)",
        "Main risk source (risk contribution)",
        "Waste share (RED2)",
    ]))
    _car_mv = procurement_budget * mv_row["vol"] * np.sqrt(horizon) * z_val
    _n_why = "15 feedstocks (weekly data ÷ 5, shrinkage)" if _why_view.startswith("Extended") else "your 4 main feedstocks"
    st.caption(
        f"Computed with {_n_why}. "
        f"**What is the Cost-at-Risk?** In a bad month (about 1 month in 20), your feedstock bill "
        f"could rise by about \\${_car_mv:,.0f} with our mix. In the other 19 months, it rises by "
        f"less, or falls. Formula: budget × daily risk × √{horizon} days × {z_val}. We recommend "
        f"you follow this number every month: it shows your risk in money, so you can decide how "
        f"much to hedge. It assumes normal price moves, so real crises like 2022 can be worse. "
        f"Minimum variance always has the lowest risk by definition. The Sharpe pick is worse for you "
        f"twice: your costs move more ({sr_row['vol']:.2%} vs {mv_row['vol']:.2%} a day), "
        f"and it picks the feedstocks whose prices rose fastest in the past "
        f"({(1 + sr_row['ret']) ** 252 - 1:+.1%} a year vs {(1 + mv_row['ret']) ** 252 - 1:+.1%}). "
        f"That is why we recommend minimum variance: stable costs are the right goal for a buyer."
    )


# --- your benchmark: stay close to what you already buy (tracking error) -------

with st.expander("Your benchmark: stay close to what you already buy (tracking error)"):
    from scipy.optimize import minimize as _bminimize
    from constraints import build_all_constraints as _bbac, get_bounds as _bgb

    st.markdown(
        "Our main mix only looks at risk. Here **you** choose a benchmark to stay close to, and how much it "
        "matters to you. The tool then finds the mix that balances **stable costs** and **staying close to your "
        "benchmark**, always with your three rules."
    )
    _bm_view = st.radio("Feedstocks the tool can buy:",
                        ["4 feedstocks (main recommendation)", "Extended universe (15 feedstocks)"],
                        horizontal=True, key="bm15")
    _bm_choice = st.radio("Your benchmark:", ["Your current purchasing mix", "Your selling price (diesel)",
                                             "A greener target mix"], horizontal=True, key="bm_choice")
    if _bm_choice == "Your current purchasing mix":
        st.caption("Enter what you buy today, in % of your budget.")
        _bc1, _bc2, _bc3, _bc4 = st.columns(4)
        _bu = _bc1.number_input("UCO %", 0.0, 100.0, 20.0, 5.0, key="bm_uco")
        _bt = _bc2.number_input("Tallow %", 0.0, 100.0, 20.0, 5.0, key="bm_tallow")
        _br = _bc3.number_input("Rapeseed oil %", 0.0, 100.0, 30.0, 5.0, key="bm_rape")
        _bs = _bc4.number_input("Soybean oil %", 0.0, 100.0, 30.0, 5.0, key="bm_soy")
        _btot = _bu + _bt + _br + _bs
        if _btot <= 0:
            st.warning("Your shares add up to 0%: we use an equal split instead.")
            _bu = _bt = _br = _bs = 25.0
        elif abs(_btot - 100) > 0.5:
            st.warning(f"Your shares add up to {_btot:.0f}%: we rescale them to 100%.")
        _bmap4 = {"HO=F": _bu, "LE=F": _bt, "ZS=F": _br, "ZL=F": _bs}
        _bmap15 = {"UCO (Rotterdam gasoil)": _bu, "Tallow (live cattle)": _bt, "Rapeseed (Euronext)": _br,
                   "Soybean oil": _bs}
    elif _bm_choice == "Your selling price (diesel)":
        st.caption("Benchmark: 100% diesel, our proxy for your selling price. Staying close to it keeps your margin steady.")
        _bmap4 = {"HO=F": 100.0}
        _bmap15 = {"UCO (heating oil)": 100.0}
    else:
        st.caption("Target: 35% UCO, 35% tallow and 30% rapeseed oil: 70% waste and no soybean oil, "
                   "close to the Dutch 2026 rules.")
        _bmap4 = {"HO=F": 35.0, "LE=F": 35.0, "ZS=F": 30.0}
        _bmap15 = {"UCO (Rotterdam gasoil)": 35.0, "Tallow (live cattle)": 35.0, "Rapeseed (Euronext)": 30.0}

    _lam = st.slider("How much does staying close to your benchmark matter to you?", 0, 100, 50, 10,
                     format="%d%%", key="bm_lambda",
                     help="0% = only stable costs (our main mix). 100% = as close as possible to your "
                          "benchmark (minimum tracking error).") / 100

    @st.cache_data(show_spinner=False)
    def _bm_solve(cov, b, lam, red, cap, n):
        # minimize (1 - lambda) x'Sx + lambda (x - b)'S(x - b), with the same three rules
        res = _bminimize(lambda x: (1 - lam) * (x @ cov @ x) + lam * ((x - b) @ cov @ (x - b)),
                         np.ones(n) / n, method="SLSQP",
                         bounds=_bgb(concentration_cap=cap, n_assets=n),
                         constraints=_bbac(sustainability_min=red, concentration_cap=cap),
                         options={"maxiter": 500, "ftol": 1e-12})
        return res.x if res.success else None

    @st.cache_data(show_spinner=False)
    def _bm_solve15(cov, b, lam, red, cap, waste, tallow):
        # same objective with 15 feedstocks: tallow proxies share one cap, waste-based share above the minimum
        n = len(b)
        cons = [{"type": "eq", "fun": lambda x: x.sum() - 1},
                {"type": "ineq", "fun": lambda x: x @ waste - red},
                {"type": "ineq", "fun": lambda x: cap - x @ tallow}]
        res = _bminimize(lambda x: (1 - lam) * (x @ cov @ x) + lam * ((x - b) @ cov @ (x - b)),
                         np.ones(n) / n, method="SLSQP", bounds=[(0.0, cap)] * n, constraints=cons,
                         options={"maxiter": 1000, "ftol": 1e-14})
        return res.x if res.success else None

    _x15 = None
    if _bm_view.startswith("Extended"):
        _x15 = _x9_mix(int(years_lookback), float(sustainability_min), float(concentration_cap), tuple(_X9_FEEDS))
        if _x15 is None or "Ss" not in _x15:
            st.warning("The 15-feedstock universe could not be built right now. Showing 4 feedstocks.")
            _x15 = None

    if _x15 is not None:
        _names = list(_x15["names"])
        _bcov = np.asarray(_x15["S"]) / 5            # daily risk, measured like the other 15-feedstock views
        _bopt = np.asarray(_x15["Ss"]) / 5           # shrunk covariance (LN 3) for the optimisation
        _wv, _tv = np.asarray(_x15["waste"]), np.asarray(_x15["tallow"])
        _b = np.array([_bmap15.get(k, 0.0) for k in _names], dtype=float)
        _b = _b / _b.sum()

        def _solve(l):
            return _bm_solve15(_bopt, _b, l, sustainability_min, concentration_cap, _wv, _tv)

        _bx0 = _solve(0.0)
        _labels = _names
        _bwaste = lambda x: float(x @ _wv)
        _b_tallow = float(_b @ _tv)
        _b_big = max([_b[i] for i in range(len(_b)) if _tv[i] == 0] + [_b_tallow])
    else:
        _tk = list(PORTFOLIO_TICKERS)
        _bcov = np.asarray(cov_matrix)
        _b = np.array([_bmap4.get(t, 0.0) for t in _tk], dtype=float)
        _b = _b / _b.sum()

        def _solve(l):
            return _bm_solve(_bcov, _b, l, sustainability_min, concentration_cap, len(_tk))

        _bx0 = np.asarray(min_variance_portfolio(constrained_frontier)["weights"])
        _labels = [TICKER_LABELS[t] for t in _tk]
        _bwaste = lambda x: float(sum(x[_tk.index(t)] for t in WASTE_BASED if t in _tk))
        _b_tallow = _b[_tk.index("LE=F")] + _b[_tk.index("DAR")]
        _b_big = max([_b[_tk.index(t)] for t in _tk if t not in ("LE=F", "DAR")] + [_b_tallow])

    _bx = _solve(_lam)

    if _bx is None or _bx0 is None:
        st.error("No mix follows your rules here. Try a lower waste minimum or a higher cap.")
    else:
        def _brisk(x):
            return float(np.sqrt(x @ _bcov @ x))

        def _bte(x):
            return float(np.sqrt((x - _b) @ _bcov @ (x - _b)))

        def _bcar(x):
            return procurement_budget * _brisk(x) * np.sqrt(21) * 1.645

        bm1, bm2, bm3 = st.columns(3)
        bm1.metric("Daily cost risk", f"{_brisk(_bx):.2%}",
                   delta=f"{(_brisk(_bx) - _brisk(_bx0)) * 100:+.2f} pts vs our main mix", delta_color="inverse")
        bm2.metric("Distance from your benchmark", f"{_bte(_bx):.2%}",
                   delta=f"{(_bte(_bx) - _bte(_bx0)) * 100:+.2f} pts vs our main mix", delta_color="inverse",
                   help="Tracking error: how much your mix's cost moves differently from your benchmark each day.")
        bm3.metric("Cost-at-Risk (95%, 1 month)", f"${_bcar(_bx):,.0f}")

        def _bcol(x, te):
            return [f"{max(v, 0):.1%}" for v in x] + [f"{_brisk(x):.2%}", te, f"{_bwaste(x):.0%}"]

        st.dataframe(pd.DataFrame({
            "Your benchmark": _bcol(_b, "0.00%"),
            "Mix for your choice": _bcol(_bx, f"{_bte(_bx):.2%}"),
            "Our main mix (stable costs)": _bcol(_bx0, f"{_bte(_bx0):.2%}"),
        }, index=list(_labels) + ["Daily cost risk", "Distance from your benchmark", "Waste share (RED2)"]))

        _b_broken = []
        if _bwaste(_b) < sustainability_min - 1e-3:
            _b_broken.append(f"waste share {_bwaste(_b):.0%} below your {sustainability_min:.0%} minimum")
        if _b_big > concentration_cap + 1e-3:
            _b_broken.append(f"a feedstock at {_b_big:.0%}, above your {concentration_cap:.0%} cap")
        if _b_broken:
            st.info("Your benchmark breaks your rules (" + "; ".join(_b_broken) + "), so the tool can't copy it "
                    "exactly: it finds the closest mix that follows your rules.")

        _bgrid = [i / 10 for i in range(11)]
        _bpts = [(l, _solve(l)) for l in _bgrid]
        _bpts = [(l, p) for l, p in _bpts if p is not None]
        fig_bm, ax_bm = plt.subplots(figsize=(7, 3.6))
        ax_bm.plot([_bte(p) for _, p in _bpts], [_brisk(p) for _, p in _bpts], marker="o", color="steelblue",
                   label="Every setting of the slider")
        for l, p in _bpts:
            if l in (0.0, 0.5, 1.0):
                ax_bm.annotate(f"{l:.0%}", (_bte(p), _brisk(p)), xytext=(6, 4), textcoords="offset points",
                               fontsize=8)
        ax_bm.scatter([_bte(_bx)], [_brisk(_bx)], s=170, facecolors="none", edgecolors="#D62728", linewidths=2,
                      zorder=5, label="Your choice")
        ax_bm.set_xlabel("Distance from your benchmark (daily tracking error)")
        ax_bm.set_ylabel("Daily cost risk")
        ax_bm.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
        ax_bm.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
        ax_bm.set_title("Your trade-off: stable costs vs staying close to your benchmark"
                        + (f" ({len(_labels)} feedstocks)" if _x15 is not None else ""))
        ax_bm.legend(fontsize=8)
        st.pyplot(fig_bm)

        st.caption(
            "Each dot is one setting of the slider (0% = our main mix, 100% = closest to your benchmark). Moving "
            "toward your benchmark makes your costs move more; staying with our main mix moves you away from "
            "your benchmark. You choose the balance. Method: minimum tracking error (LN 2), combined with minimum "
            "variance: minimize (1 − λ)·xᵀΣx + λ·(x − b)ᵀΣ(x − b), with your three rules."
            + (" With 15 feedstocks, your benchmark uses the closest series: UCO = Rotterdam gasoil, tallow = live "
               "cattle, rapeseed oil = Euronext rapeseed, diesel = heating oil; weekly data ÷ 5 and shrinkage "
               "(LN 3) for the optimisation." if _x15 is not None else "")
        )


# --- your price views: Black-Litterman (LN 4) ------------------------------------

with st.expander("Your price views: Black-Litterman (tell us what you expect)"):
    from scipy.optimize import minimize as _vminimize
    from constraints import build_all_constraints as _vbac, get_bounds as _vgb

    st.markdown(
        "Our main mix ignores price forecasts. Here **you** can add your own views, for example *“UCO will rise "
        "with the Dutch 2026 rules”*, and say how sure you are. The **Black-Litterman** model blends your views "
        "with a neutral starting point, and the tool then avoids the feedstocks you expect to get expensive, "
        "always with your three rules."
    )
    _v_view = st.radio("Feedstocks the tool can buy:",
                       ["4 feedstocks (main recommendation)", "Extended universe (15 feedstocks)"],
                       horizontal=True, key="bl15")
    _x15v = None
    if _v_view.startswith("Extended"):
        _x15v = _x9_mix(int(years_lookback), float(sustainability_min), float(concentration_cap), tuple(_X9_FEEDS))
        if _x15v is None or "Ss" not in _x15v:
            st.warning("The 15-feedstock universe could not be built right now. Showing 4 feedstocks.")
            _x15v = None

    if _x15v is not None:
        _vtk = list(_x15v["names"])
        _vfeed = {"No view": None,
                  "UCO": ["UCO (Rotterdam gasoil)", "UCO (heating oil)"],
                  "Tallow": ["Tallow (live cattle)", "Tallow (Darling)", "Tallow (feeder cattle)"],
                  "Rapeseed oil": ["Rapeseed (Euronext)"], "Soybean oil": ["Soybean oil"],
                  "Pork fat / white grease": ["Pork fat / white grease (lean hogs)"], "Canola": ["Canola (ICE)"],
                  "Palm oil": ["Palm oil (Bursa Malaysia)"], "Corn oil": ["Distillers corn oil (corn)"]}
        _vfeed = {k: (None if v is None else [t for t in v if t in _vtk]) for k, v in _vfeed.items()}
        _vfeed = {k: v for k, v in _vfeed.items() if v is None or v}
        _vsuf = "15"
    else:
        _vtk = list(PORTFOLIO_TICKERS)
        _vfeed = {"No view": None, "UCO": ["HO=F"], "Tallow": ["LE=F", "DAR"], "Rapeseed oil": ["ZS=F"],
                  "Soybean oil": ["ZL=F"]}
        _vsuf = ""

    _vc1, _vc2 = st.columns(2)
    with _vc1:
        _v1f = st.selectbox("View 1: feedstock", list(_vfeed), index=1, key="v1_feed" + _vsuf)
        _v1q = st.number_input("View 1: expected price change over the next year (%)", -50.0, 50.0, 20.0, 5.0,
                               key="v1_q")
    with _vc2:
        _v2f = st.selectbox("View 2: feedstock", list(_vfeed), index=0, key="v2_feed" + _vsuf)
        _v2q = st.number_input("View 2: expected price change over the next year (%)", -50.0, 50.0, 0.0, 5.0,
                               key="v2_q")
    _vconf = st.slider("How sure are you about your views?", 10, 95, 50, 5, format="%d%%", key="v_conf") / 100
    _vweight = st.select_slider("How much should your views count against stable costs?",
                                ["A little", "Medium", "A lot"], value="Medium", key="v_weight")
    _vgamma = {"A little": 20.0, "Medium": 5.0, "A lot": 2.0}[_vweight]

    _vn = len(_vtk)
    _vtau = 0.05
    if _x15v is not None:
        _vS = np.asarray(_x15v["S"]) / 5        # daily risk, measured like the other 15-feedstock views
        _vSo = np.asarray(_x15v["Ss"]) / 5      # shrunk covariance (LN 3) for Black-Litterman and the optimisation
        _vpi = np.full(_vn, float(np.mean(np.asarray(_x15v["mu"]) / 5)))
    else:
        _vS = _vSo = np.asarray(cov_matrix)
        _vpi = np.full(_vn, float(np.mean(np.asarray(mean_returns))))
    # neutral prior: every feedstock is expected to move by the same average amount (no opinion),
    # so with no views the answer is exactly our minimum-variance mix
    _vchosen = [(f, q) for f, q in ((_v1f, _v1q), (_v2f, _v2q)) if _vfeed[f] is not None]
    if len({f for f, _ in _vchosen}) < len(_vchosen):
        st.warning("Your two views are on the same feedstock: we only use view 1.")
        _vchosen = _vchosen[:1]
    _views = [(t, q) for f, q in _vchosen for t in _vfeed[f]]
    if _views:
        _vP = np.array([np.eye(_vn)[_vtk.index(t)] for t, _ in _views])
        _vQ = np.array([q / 100 / 252 for _, q in _views])
        _vOm = np.diag(np.diag(_vP @ (_vtau * _vSo) @ _vP.T)) * (1 - _vconf) / _vconf
        _vA = np.linalg.inv(np.linalg.inv(_vtau * _vSo) + _vP.T @ np.linalg.inv(_vOm) @ _vP)
        _vmu = _vA @ (np.linalg.inv(_vtau * _vSo) @ _vpi + _vP.T @ np.linalg.inv(_vOm) @ _vQ)
    else:
        _vmu = _vpi.copy()

    @st.cache_data(show_spinner=False)
    def _v_solve(cov, mu, gamma, red, cap, n):
        # a buyer wants low expected price increases AND stable costs: minimize x'mu + gamma/2 x'Sx
        res = _vminimize(lambda x: x @ mu + gamma / 2 * (x @ cov @ x), np.ones(n) / n, method="SLSQP",
                         bounds=_vgb(concentration_cap=cap, n_assets=n),
                         constraints=_vbac(sustainability_min=red, concentration_cap=cap),
                         options={"maxiter": 500, "ftol": 1e-14})
        return res.x if res.success else None

    @st.cache_data(show_spinner=False)
    def _v_solve15(cov, mu, gamma, red, cap, waste, tallow):
        n = len(mu)
        cons = [{"type": "eq", "fun": lambda x: x.sum() - 1},
                {"type": "ineq", "fun": lambda x: x @ waste - red},
                {"type": "ineq", "fun": lambda x: cap - x @ tallow}]
        res = _vminimize(lambda x: x @ mu + gamma / 2 * (x @ cov @ x), np.ones(n) / n, method="SLSQP",
                         bounds=[(0.0, cap)] * n, constraints=cons, options={"maxiter": 1000, "ftol": 1e-15})
        return res.x if res.success else None

    if _x15v is not None:
        _vw, _vt = np.asarray(_x15v["waste"]), np.asarray(_x15v["tallow"])
        _vx = _v_solve15(_vSo, _vmu, _vgamma, sustainability_min, concentration_cap, _vw, _vt)
        _vx0 = _v_solve15(_vSo, _vpi, _vgamma, sustainability_min, concentration_cap, _vw, _vt)
        _vlabels = _vtk
    else:
        _vx = _v_solve(_vS, _vmu, _vgamma, sustainability_min, concentration_cap, _vn)
        _vx0 = np.asarray(min_variance_portfolio(constrained_frontier)["weights"])
        _vlabels = [TICKER_LABELS[t] for t in _vtk]
    if _vx is None or _vx0 is None:
        st.error("No mix follows your rules here. Try a lower waste minimum or a higher cap.")
    else:
        def _vrisk(x):
            return float(np.sqrt(x @ _vS @ x))

        def _vyear(x, m):
            return float(x @ m) * 252

        vm1, vm2, vm3 = st.columns(3)
        vm1.metric("Expected price change of your mix (1 year)", f"{_vyear(_vx, _vmu):+.1%}",
                   delta=f"{(_vyear(_vx, _vmu) - _vyear(_vx0, _vmu)) * 100:+.1f} pts vs our main mix",
                   delta_color="inverse", help="Using the blended (Black-Litterman) expected price changes.")
        vm2.metric("Daily cost risk", f"{_vrisk(_vx):.2%}",
                   delta=f"{(_vrisk(_vx) - _vrisk(_vx0)) * 100:+.2f} pts vs our main mix", delta_color="inverse")
        vm3.metric("Cost-at-Risk (95%, 1 month)", f"${procurement_budget * _vrisk(_vx) * np.sqrt(21) * 1.645:,.0f}")

        st.dataframe(pd.DataFrame({
            "Neutral expected change (1 year)": [f"{v * 252:+.1%}" for v in _vpi],
            "After your views (1 year)": [f"{v * 252:+.1%}" for v in _vmu],
            "Our main mix": [f"{max(v, 0):.1%}" for v in _vx0],
            "Mix with your views": [f"{max(v, 0):.1%}" for v in _vx],
        }, index=list(_vlabels)))
        st.caption(
            "How it works: we start from a neutral view (every feedstock is expected to move by the same average "
            "amount, so with no views you get our main mix). Black-Litterman then blends in your views, weighted "
            "by how sure you are: μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ [(τΣ)⁻¹π + PᵀΩ⁻¹Q], with τ = 0.05. A view on one "
            "feedstock also moves the feedstocks that move with it (through the covariance matrix). As a buyer, "
            "we then minimize expected price change + risk. Your rules still hold: for example, the waste "
            "minimum keeps some UCO even if you expect it to rise. "
            + ("With 15 feedstocks, a view applies to every series of that feedstock (UCO: Rotterdam gasoil and "
               "heating oil; tallow: live cattle, Darling and feeder cattle); weekly data ÷ 5 and shrinkage (LN 3). "
               if _x15v is not None else
               "A tallow view applies to both tallow proxies (live cattle and Darling). ")
            + "Views are opinions, not forecasts we make."
        )


# --- backtest: does the strategy work out of sample? -------------------------------

with st.expander("Backtest: does our strategy work on prices it has not seen?"):
    from scipy.optimize import minimize as _tminimize
    from constraints import build_all_constraints as _tbac, get_bounds as _tgb

    st.markdown(
        "Our main mix is computed on five years of prices and judged on the same prices (in-sample). Here we "
        "do it the honest way: **every month**, we recompute the mix using **only the past data**, buy it for "
        "the next month, and record how much your costs really moved. We compare it with a simple equal split."
    )
    _bt_view = st.radio("Feedstocks the tool can buy:",
                        ["4 feedstocks (main recommendation)", "Extended universe (15 feedstocks)"],
                        horizontal=True, key="bt15")

    def _bt_mixchart(wdf, colors, labels):
        # one bar per month, stacked to 100%, so the change in the mix is easy to read
        wdf = wdf.clip(lower=0)
        wdf = wdf.loc[:, wdf.max() > 0.005]
        fig_w, ax_w = plt.subplots(figsize=(9, 3.4))
        bottom = np.zeros(len(wdf))
        xs = np.arange(len(wdf))
        for c in wdf.columns:
            ax_w.bar(xs, wdf[c].values, bottom=bottom, width=0.9, color=colors.get(c, "#999999"),
                     label=labels.get(c, c), edgecolor="white", linewidth=0.3)
            bottom += wdf[c].values
        step = max(1, len(wdf) // 8)
        ax_w.set_xticks(xs[::step])
        ax_w.set_xticklabels([d.strftime("%b %Y") for d in wdf.index[::step]], fontsize=8)
        ax_w.set_ylim(0, 1)
        ax_w.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax_w.set_ylabel("Share of your budget")
        for side in ("top", "right"):
            ax_w.spines[side].set_visible(False)
        ax_w.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
        fig_w.tight_layout()
        st.pyplot(fig_w)

    def _bt_linechart(series, title):
        fig_bt, ax_bt = plt.subplots(figsize=(9, 3.6))
        styles = [("#D62728", 1.6, "-"), ("#1F5FA8", 1.3, "-"), ("gray", 1.2, "--")]
        for (name, vals), (col, lw, ls) in zip(series.items(), styles):
            ax_bt.plot(np.arange(len(vals)), np.cumprod(1 + vals) * 100, color=col, linewidth=lw, linestyle=ls,
                       label=name)
        ax_bt.set_ylabel("Cost index (start = 100)")
        ax_bt.set_title(title)
        ax_bt.legend(fontsize=8)
        for side in ("top", "right"):
            ax_bt.spines[side].set_visible(False)
        return fig_bt, ax_bt

    _x15b = None
    if _bt_view.startswith("Extended"):
        _x15b = _x9_mix(int(years_lookback), float(sustainability_min), float(concentration_cap), tuple(_X9_FEEDS))
        if _x15b is None or "r" not in _x15b:
            st.warning("The 15-feedstock universe could not be built right now. Showing 4 feedstocks.")
            _x15b = None

    if _x15b is None:
        @st.cache_data(show_spinner="Running the monthly backtest...")
        def _bt_run(rets, red, cap, window):
            tk = list(rets.columns)
            n = len(tk)
            month_ends = rets.resample("ME").last().index
            out_mv, out_eq, weights = [], [], []
            eq = np.array([0.25 if t in ("ZL=F", "ZS=F", "HO=F", "LE=F") else 0.0 for t in tk])
            for i in range(len(month_ends) - 1):
                start, end = month_ends[i], month_ends[i + 1]
                past = rets.loc[:start].tail(window)
                if len(past) < window * 0.9:
                    continue
                cov = past.cov().values
                res = _tminimize(lambda x: x @ cov @ x, np.ones(n) / n, method="SLSQP",
                                 bounds=_tgb(concentration_cap=cap, n_assets=n),
                                 constraints=_tbac(sustainability_min=red, concentration_cap=cap),
                                 options={"maxiter": 500, "ftol": 1e-12})
                if not res.success:
                    continue
                nxt = rets.loc[(rets.index > start) & (rets.index <= end)]
                out_mv.append(nxt.values @ res.x)
                out_eq.append(nxt.values @ eq)
                weights.append(pd.Series(res.x, index=tk, name=end))
            if not out_mv:
                return None
            return np.concatenate(out_mv), np.concatenate(out_eq), pd.DataFrame(weights)

        _bt_window = 252
        _bt = _bt_run(returns[PORTFOLIO_TICKERS], sustainability_min, concentration_cap, _bt_window)
        if _bt is None:
            st.info("Not enough data for a backtest: choose at least 2 years in the sidebar.")
        else:
            _bt_mv, _bt_ew, _bt_w = _bt
            _bt_vol_mv, _bt_vol_ew = float(np.std(_bt_mv)), float(np.std(_bt_ew))
            _bt_in = float(min_variance_portfolio(constrained_frontier)["volatility"])
            bt1, bt2, bt3 = st.columns(3)
            bt1.metric("Our strategy, out of sample", f"{_bt_vol_mv:.2%} a day",
                       help="Daily cost volatility actually realized, month after month, with the mix recomputed "
                            "each month on the past year only.")
            bt2.metric("Equal split (25% each)", f"{_bt_vol_ew:.2%} a day",
                       delta=f"{(_bt_vol_ew / _bt_vol_mv - 1):+.0%} more movement", delta_color="inverse")
            bt3.metric("In-sample estimate (main mix)", f"{_bt_in:.2%} a day",
                       help="The number the main chart shows, computed on the same data it is judged on.")

            fig_bt, ax_bt = _bt_linechart({"Our strategy (rebalanced monthly)": _bt_mv, "Equal split": _bt_ew},
                                          "Your feedstock cost, month after month, out of sample")
            ax_bt.set_xlabel("Trading days in the test (out of sample)")
            fig_bt.tight_layout()
            st.pyplot(fig_bt)

            st.markdown("**How the mix changed each month** (recomputed on the past year only):")
            _bt_mixchart(_bt_w, TICKER_COLORS, TICKER_LABELS)
            st.caption(
                f"Test: {len(_bt_w)} monthly rebalances, each using only the previous {_bt_window} trading days. "
                "A lower line is not the goal (we can't control price trends); what matters is how much the cost "
                "moves. If the out-of-sample volatility is close to the in-sample estimate, the strategy is not "
                "just fitted to the past. The equal split uses 25% of each real feedstock, which follows your rules "
                "at the default settings."
            )
    else:
        @st.cache_data(show_spinner="Running the 15-feedstock backtest...")
        def _bt_run15(rets, red, cap, window, waste, tallow):
            n = rets.shape[1]
            cons = [{"type": "eq", "fun": lambda x: x.sum() - 1},
                    {"type": "ineq", "fun": lambda x: x @ waste - red},
                    {"type": "ineq", "fun": lambda x: cap - x @ tallow}]

            def solve(cov):
                res = _tminimize(lambda x: x @ cov @ x, np.ones(n) / n, method="SLSQP",
                                 bounds=[(0.0, cap)] * n, constraints=cons,
                                 options={"maxiter": 1000, "ftol": 1e-14})
                return res.x if res.success else None

            def shrink(X):
                X = X - X.mean(axis=0)
                T = X.shape[0]
                emp = X.T @ X / T
                m = np.trace(emp) / n
                d = ((emp - m * np.eye(n)) ** 2).sum() / n
                bt = min(((X ** 2).T @ (X ** 2) / T - emp ** 2).sum() / (n * T), d)
                sh = bt / d if d > 0 else 0.0
                return (1 - sh) * emp + sh * m * np.eye(n)

            month_ends = rets.resample("ME").last().index
            out_s, out_r, out_e, weights = [], [], [], []
            for i in range(len(month_ends) - 1):
                start, end = month_ends[i], month_ends[i + 1]
                past = rets.loc[:start].tail(window)
                if len(past) < window * 0.9:
                    continue
                ws, wr = solve(shrink(past.values)), solve(past.cov().values)
                if ws is None or wr is None:
                    continue
                nxt = rets.loc[(rets.index > start) & (rets.index <= end)].values
                out_s.append(nxt @ ws)
                out_r.append(nxt @ wr)
                out_e.append(nxt @ (np.ones(n) / n))
                weights.append(pd.Series(ws, index=rets.columns, name=end))
            if not out_s:
                return None
            return np.concatenate(out_s), np.concatenate(out_r), np.concatenate(out_e), pd.DataFrame(weights)

        _bt_window = 104  # two years of weekly prices
        _bt = _bt_run15(_x15b["r"], float(sustainability_min), float(concentration_cap), _bt_window,
                        np.asarray(_x15b["waste"]), np.asarray(_x15b["tallow"]))
        if _bt is None:
            st.info("Not enough data for a 15-feedstock backtest: choose at least 4 years in the sidebar.")
        else:
            _bs, _br, _be, _bt_w = _bt
            _d = np.sqrt(5)  # weekly volatility ÷ √5 = daily
            _vs, _vr, _ve = (float(np.std(v)) / _d for v in (_bs, _br, _be))
            _vin = float(np.sqrt(_x15b["w"] @ _x15b["S"] @ _x15b["w"] / 5))
            bt1, bt2, bt3, bt4 = st.columns(4)
            bt1.metric("With shrinkage, out of sample", f"{_vs:.2%} a day",
                       help="Mix recomputed each month on the past two years, with Ledoit-Wolf shrinkage (LN 3).")
            bt2.metric("Without shrinkage", f"{_vr:.2%} a day",
                       delta=f"{(_vr / _vs - 1):+.0%} vs with shrinkage", delta_color="inverse",
                       help="Same test with the raw covariance matrix.")
            bt3.metric(f"Equal split (1/{len(_x15b['names'])} each)", f"{_ve:.2%} a day",
                       delta=f"{(_ve / _vs - 1):+.0%} vs with shrinkage", delta_color="inverse")
            bt4.metric("In-sample estimate", f"{_vin:.2%} a day",
                       help="The 15-feedstock minimum-variance mix, judged on the same data it was built on.")

            fig_bt, ax_bt = _bt_linechart({"With shrinkage (rebalanced monthly)": _bs, "Without shrinkage": _br,
                                           "Equal split": _be},
                                          f"Your feedstock cost with {len(_x15b['names'])} feedstocks, out of sample")
            ax_bt.set_xlabel("Weeks in the test (out of sample)")
            fig_bt.tight_layout()
            st.pyplot(fig_bt)

            st.markdown("**How the mix changed each month** (with shrinkage, recomputed on the past two years):")
            _bt_mixchart(_bt_w, _X9_COLORS, {})
            st.caption(
                f"Test: {len(_bt_w)} monthly rebalances, each using only the previous {_bt_window} weeks (weekly "
                "prices; volatility ÷ √5 to compare with daily numbers). With 15 feedstocks the tool estimates 105 "
                "correlations from two years of data, so we also test shrinkage (LN 3). "
                + (f"Here shrinkage lowers the out-of-sample risk by {(1 - _vs / _vr):.0%}. " if _vs < _vr * 0.98 else
                   "Here shrinkage changes almost nothing: with two years of weekly data the raw estimate is already "
                   "stable enough, so shrinkage is a safety net rather than a gain. ")
                + (f"The out-of-sample risk ({_vs:.2%}) is close to or below the in-sample estimate ({_vin:.2%}), so "
                   "the extended mix is not just fitted to the past. " if _vs <= _vin * 1.1 else
                   f"The out-of-sample risk ({_vs:.2%}) is above the in-sample estimate ({_vin:.2%}): part of the "
                   "in-sample gain comes from fitting the past. ")
                + f"The equal split moves {(_ve / _vs - 1):.0%} more: choosing the mix matters more than the method."
            )


# --- other risk strategies: minimum CVaR and risk parity ----------------------------

with st.expander("Other strategies: minimum CVaR and risk parity (for comparison)"):
    from scipy.optimize import minimize as _cminimize, linprog as _clinprog

    st.markdown(
        "Two well-known alternatives, with your same three rules. **Minimum CVaR** protects you against the "
        "worst days (crises like 2022) instead of average swings. **Risk parity** (equal risk contribution) "
        "makes each feedstock bring the same share of your risk."
    )
    _cs_view = st.radio("Feedstocks the tool can buy:",
                        ["4 feedstocks (main recommendation)", "Extended universe (15 feedstocks)"],
                        horizontal=True, key="cs15")
    _x15c = None
    if _cs_view.startswith("Extended"):
        _x15c = _x9_mix(int(years_lookback), float(sustainability_min), float(concentration_cap), tuple(_X9_FEEDS))
        if _x15c is None or "r" not in _x15c:
            st.warning("The 15-feedstock universe could not be built right now. Showing 4 feedstocks.")
            _x15c = None
    _cbeta = 0.95

    if _x15c is not None:
        _cnames = list(_x15c["names"])
        _cR = _x15c["r"].values                  # weekly returns
        _cS = np.asarray(_x15c["S"])             # weekly covariance
        _cwaste, _ctallow = np.asarray(_x15c["waste"]), np.asarray(_x15c["tallow"])
        _cmv = np.asarray(_x15c["w"])
        _clabels = _cnames
        _cper, _cscale = "week", 5.0
        _crp_free = list(range(len(_cnames)))   # every feedstock gets the same share of risk
    else:
        _ctk = list(PORTFOLIO_TICKERS)
        _cR = returns[PORTFOLIO_TICKERS].values
        _cS = np.asarray(cov_matrix)
        _cwaste = np.array([1.0 if t in WASTE_BASED else 0.0 for t in _ctk])
        _ctallow = np.array([1.0 if t in ("LE=F", "DAR") else 0.0 for t in _ctk])
        _cmv = np.asarray(min_variance_portfolio(constrained_frontier)["weights"])
        _clabels = [TICKER_LABELS[t] for t in _ctk]
        _cper, _cscale = "day", 1.0
        _crp_free = [i for i, t in enumerate(_ctk) if t != "DAR"]   # Darling kept at 0 (tallow = live cattle)
    _cn = len(_clabels)

    @st.cache_data(show_spinner="Solving minimum CVaR...")
    def _c_min_cvar(R, red, cap, beta, waste, tallow):
        # Rockafellar-Uryasev linear program; loss = cost increase = x'r (a buyer loses when prices rise)
        T, n = R.shape
        c = np.concatenate([np.zeros(n), [1.0], np.full(T, 1.0 / ((1 - beta) * T))])
        A_ub = np.hstack([R, -np.ones((T, 1)), -np.eye(T)])
        b_ub = np.zeros(T)
        A_ub = np.vstack([A_ub, np.concatenate([-waste, [0.0], np.zeros(T)]),
                          np.concatenate([tallow, [0.0], np.zeros(T)])])
        b_ub = np.concatenate([b_ub, [-red, cap]])
        A_eq = np.concatenate([np.ones(n), [0.0], np.zeros(T)])[None, :]
        bounds = [(0, cap)] * n + [(None, None)] + [(0, None)] * T
        res = _clinprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=bounds, method="highs")
        return res.x[:n] if res.success else None

    @st.cache_data(show_spinner="Solving risk parity...")
    def _c_erc(cov, red, cap, waste, tallow, live):
        # equal risk contribution among the chosen feedstocks, with your rules
        n = cov.shape[0]
        live = list(live)

        def obj(x):
            v = x @ cov @ x
            rc = x * (cov @ x) / v
            return float(sum((rc[i] - 1 / len(live)) ** 2 for i in live))
        bnds = [(0.002, cap) if i in live else (0.0, 0.0) for i in range(n)]
        x0 = np.array([1 / len(live) if i in live else 0.0 for i in range(n)])
        cons = [{"type": "eq", "fun": lambda x: x.sum() - 1},
                {"type": "ineq", "fun": lambda x: x @ waste - red},
                {"type": "ineq", "fun": lambda x: cap - x @ tallow}]
        res = _cminimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons,
                         options={"maxiter": 2000, "ftol": 1e-14})
        x = res.x
        ok = abs(x.sum() - 1) < 1e-4 and x @ waste >= red - 1e-4 and x @ tallow <= cap + 1e-4
        return x if (res.success or ok) else None

    _ccv = _c_min_cvar(_cR, float(sustainability_min), float(concentration_cap), _cbeta, _cwaste, _ctallow)
    _crp = _c_erc(_cS, float(sustainability_min), float(concentration_cap), _cwaste, _ctallow, tuple(_crp_free))
    if _x15c is not None:
        _ceq = np.ones(_cn) / _cn
        _ceq_name = f"Equal split (1/{_cn} each)"
    else:
        _ceq = np.array([0.25 if t in ("ZL=F", "ZS=F", "HO=F", "LE=F") else 0.0 for t in _ctk])
        _ceq_name = "Equal split"

    def _c_stats(x):
        if x is None:
            return ["—"] * (_cn + 4)
        x = np.clip(x, 0, None)
        pnl = _cR @ x
        var = np.quantile(pnl, _cbeta)
        cvar = pnl[pnl >= var].mean()
        rc = x * (_cS @ x) / (x @ _cS @ x)
        return ([f"{v:.1%}" for v in x] + [f"{np.sqrt(x @ _cS @ x / _cscale):.2%}", f"{cvar:.2%}",
                f"${procurement_budget * cvar:,.0f}", f"{rc.max():.0%}"])

    st.dataframe(pd.DataFrame({
        "Minimum variance (ours)": _c_stats(_cmv),
        "Minimum CVaR": _c_stats(_ccv),
        "Risk parity": _c_stats(_crp),
        _ceq_name: _c_stats(_ceq),
    }, index=list(_clabels) + ["Daily cost risk", f"CVaR 95% (average of the worst 5% {_cper}s)",
                               f"CVaR in money (1 bad {_cper})", "Largest risk contribution"]))

    def _cv(x):
        x = np.clip(x, 0, None)
        return float(np.sqrt(x @ _cS @ x / _cscale))

    _cok = _ccv is not None and _crp is not None
    if _cok:
        _cmsg = (f"Daily risk: minimum variance {_cv(_cmv):.2%}, minimum CVaR {_cv(_ccv):.2%}, risk parity "
                 f"{_cv(_crp):.2%}, equal split {_cv(_ceq):.2%}. ")
    else:
        _cmsg = ""
    st.caption(
        f"CVaR 95% = the average cost increase on the worst 5% of {_cper}s (historical, no normal assumption), "
        "minimized with the Rockafellar-Uryasev linear program. "
        + ("Risk parity gives every feedstock the same share of risk; weekly prices, risk ÷ √5 for daily numbers. "
           if _x15c is not None else
           "Risk parity gives each real feedstock the same share of risk (Darling kept at 0, as tallow is already "
           "covered by live cattle). ")
        + _cmsg
        + "Minimum variance has the lowest daily risk by definition; risk parity sits between it and the equal "
        "split. We keep minimum variance as our recommendation: these two strategies don't use any input from "
        "you, and on your data they give similar or higher risk."
    )


# --- extended universe: 15 feedstocks, shrinkage (LN 3) and a commodity benchmark -----

with st.expander("Extended universe: 15 feedstocks, shrinkage and a commodity benchmark"):
    from scipy.optimize import minimize as _xminimize

    st.markdown(
        "A proof-of-concept extension. We widen your feedstock universe from 4 to **15 feedstocks**, use **real "
        "exchange prices** for UCO (Rotterdam gasoil), rapeseed (Euronext), canola (ICE) and palm oil (Bursa "
        "Malaysia), and stabilise the covariance "
        "matrix with **shrinkage** (LN 3). Then we compare your mix with a **commodity market benchmark**."
    )

    # feedstock: (source, ticker or file, waste-based?, tallow group?)
    _XFEED = {
        "UCO (Rotterdam gasoil)": ("csv", "data/gasoil_ice.csv", True, False),
        "Tallow (live cattle)": ("yahoo", "LE=F", True, True),
        "Tallow (Darling)": ("yahoo", "DAR", True, True),
        "Pork fat / white grease (lean hogs)": ("yahoo", "HE=F", True, False),
        "Soybean oil": ("yahoo", "ZL=F", False, False),
        "Rapeseed (Euronext)": ("csv", "data/rapeseed_euronext.csv", False, False),
        "Canola (ICE)": ("csv", "data/canola_ice.csv", False, False),
        "Palm oil (Bursa Malaysia)": ("csv", "data/palm_bursa.csv", False, False),
        "Distillers corn oil (corn)": ("yahoo", "ZC=F", False, False),
        "Soybeans (CBOT)": ("yahoo", "ZS=F", False, False),
        "UCO (heating oil)": ("yahoo", "HO=F", True, False),
        "Tallow (feeder cattle)": ("yahoo", "GF=F", True, True),
        "Neste (company proxy)": ("yahoo", "NESTE.HE", False, False),
        "Verbio (company proxy)": ("yahoo", "VBK.DE", False, False),
        "Bunge (company proxy)": ("yahoo", "BG", False, False),
    }
    # benchmark: (source, file or ticker, Yahoo ticker to try when the file is not here)
    _XBENCH = {"Bloomberg Commodity Index (BCOM)": ("csv", "data/bcom.csv", "^BCOM"),
               "Bloomberg Agriculture Index (BCOMAG)": ("csv", "data/bcomag.csv", "^BCOMAG"),
               "Agriculture basket ETF (DBA)": ("yahoo", "DBA", None),
               "Broad commodities ETF (S&P GSCI, GSG)": ("yahoo", "GSG", None)}

    @st.cache_data(ttl=86400, show_spinner="Downloading the extra prices...")
    def _x_yahoo(tickers, years):
        import yfinance as yf
        end = pd.Timestamp.today().normalize()
        start = end - pd.DateOffset(years=years)
        out = {}
        for t in tickers:
            try:
                d = yf.download(t, start=start, end=end, auto_adjust=True, progress=False, threads=False)
                if d is None or d.empty:
                    continue
                c = d["Close"]
                if isinstance(c, pd.DataFrame):
                    c = c.iloc[:, 0]
                out[t] = c.dropna()
            except Exception:
                continue
        return out

    def _x_csv(path, years):
        d = pd.read_csv(path, thousands=",")
        d["Date"] = pd.to_datetime(d["Date"], format="%m/%d/%Y")
        sr = d.set_index("Date").sort_index()["Price"].astype(float)
        return sr[sr.index >= sr.index.max() - pd.DateOffset(years=years)]

    import os as _xos

    def _xb_where(k):
        # ("csv", path) if the file is here, ("yahoo", ticker) otherwise, None if there is no source
        src, ref, backup = _XBENCH[k]
        if src == "csv" and _xos.path.exists(ref):
            return ("csv", ref)
        if src == "yahoo":
            return ("yahoo", ref)
        return ("yahoo", backup) if backup else None

    _xyahoo = tuple(sorted({v[1] for v in _XFEED.values() if v[0] == "yahoo"}
                          | {w[1] for w in map(_xb_where, _XBENCH) if w and w[0] == "yahoo"}))
    _xdl = _x_yahoo(_xyahoo, int(years_lookback))
    # only offer the benchmarks this copy of the app can read: your files first, Yahoo when a file is missing
    _xbench_ok = [k for k in _XBENCH if _xb_where(k) and (_xb_where(k)[0] == "csv" or _xb_where(k)[1] in _xdl)]
    _xbench_name = st.selectbox("Commodity benchmark", _xbench_ok, key="x_bench")

    _xprices, _xmissing = {}, []
    for _name, (_src, _ref, _w, _tg) in _XFEED.items():
        try:
            _sr = _x_csv(_ref, int(years_lookback)) if _src == "csv" else _xdl.get(_ref)
        except FileNotFoundError:
            _sr = None
        if _sr is None or len(_sr) < 250:
            _xmissing.append(_name)
        else:
            _xprices[_name] = _sr
    _xbsrc, _xbref = _xb_where(_xbench_name)
    try:
        _xb_ser = _x_csv(_xbref, int(years_lookback)) if _xbsrc == "csv" else _xdl.get(_xbref)
    except FileNotFoundError:
        _xb_ser = None

    if _xmissing:
        st.warning("Not available right now (skipped): " + ", ".join(_xmissing) + ".")
    if len(_xprices) < 4:
        st.error("Too few feedstocks could be loaded to run the extended mix. Try the force-refresh button later.")
    else:
        _xp = pd.DataFrame(_xprices).sort_index()
        _xp = _xp[_xp.index >= _xp.dropna(how="any").index.min()].ffill(limit=5).dropna()
        # weekly returns (Fridays), as in the basis-risk checks: different exchange holidays and the monthly
        # futures rolls of the Investing.com series would add fake jumps to daily returns
        _xr = _xp.resample("W-FRI").last().pct_change().dropna()
        if "Rapeseed (Euronext)" in _xr.columns:
            # Euronext rolls in Feb/May/Aug/Nov: drop the weeks that contain a roll
            _xr = _xr[~(_xr.index.month.isin([2, 5, 8, 11]) & (_xr.index.day <= 9))]
        _xnames = list(_xr.columns)
        _xn = len(_xnames)
        _xS = _xr.cov().values

        # Ledoit-Wolf shrinkage towards a scaled identity (the estimator used in scikit-learn)
        _X = _xr.values - _xr.values.mean(axis=0)
        _T = _X.shape[0]
        _emp = _X.T @ _X / _T
        _mu_t = np.trace(_emp) / _xn
        _delta = ((_emp - _mu_t * np.eye(_xn)) ** 2).sum() / _xn
        _beta = min(((_X ** 2).T @ (_X ** 2) / _T - _emp ** 2).sum() / (_xn * _T), _delta)
        _xshrink = float(_beta / _delta) if _delta > 0 else 0.0
        _xSs = (1 - _xshrink) * _emp + _xshrink * _mu_t * np.eye(_xn)

        _xwaste = np.array([1.0 if _XFEED[n][2] else 0.0 for n in _xnames])
        _xtallow = np.array([1.0 if _XFEED[n][3] else 0.0 for n in _xnames])
        _xcons = [{"type": "eq", "fun": lambda x: x.sum() - 1},
                  {"type": "ineq", "fun": lambda x: x @ _xwaste - sustainability_min},
                  {"type": "ineq", "fun": lambda x: concentration_cap - x @ _xtallow}]
        _xbnds = [(0.0, concentration_cap)] * _xn

        def _x_minvar(cov):
            res = _xminimize(lambda x: x @ cov @ x, np.ones(_xn) / _xn, method="SLSQP", bounds=_xbnds,
                             constraints=_xcons, options={"maxiter": 1000, "ftol": 1e-14})
            return res.x if res.success else None

        _xw_sample = _x_minvar(_xS)
        _xw_shrunk = _x_minvar(_xSs)
        if _xw_sample is None or _xw_shrunk is None:
            st.error("No mix follows your rules with these feedstocks. Try a lower waste minimum or a higher cap.")
        else:
            _xmain = float(min_variance_portfolio(constrained_frontier)["volatility"])
            xm1, xm2, xm3 = st.columns(3)
            xm1.metric("Feedstocks used", f"{_xn}", help="Out of the 8 in the extended universe.")
            _xday = float(np.sqrt(_xw_shrunk @ _xS @ _xw_shrunk / 5))
            xm2.metric("Daily cost risk (extended mix)", f"{_xday:.2%}",
                       delta=f"{(_xday - _xmain) * 100:+.2f} pts vs main mix (4 feedstocks)", delta_color="inverse",
                       help="Weekly risk ÷ √5, so it compares with the main app's daily numbers.")
            xm3.metric("Shrinkage intensity", f"{_xshrink:.0%}",
                       help="How much the covariance matrix is pulled towards a simple, stable target (Ledoit-Wolf). "
                            "0% = raw estimate, 100% = target only.")

            st.dataframe(pd.DataFrame({
                "Waste-based": ["yes" if _XFEED[n][2] else "no" for n in _xnames],
                "Daily risk alone": [f"{np.sqrt(_xS[i, i] / 5):.2%}" for i in range(_xn)],
                "Mix (raw covariance)": [f"{max(v, 0):.1%}" for v in _xw_sample],
                "Mix (with shrinkage)": [f"{max(v, 0):.1%}" for v in _xw_shrunk],
            }, index=_xnames))
            st.caption(
                f"{_T} weeks of returns (weekly risk ÷ √5 gives daily). With {_xn} feedstocks we estimate {_xn * (_xn - 1) // 2} correlations from the same "
                f"history. Shrinkage (LN 3) blends the raw estimate with a stable target; at {_xshrink:.0%} it only "
                "adjusts the mix slightly (compare the last two columns), and the backtest shows it changes almost "
                "nothing out of sample: a safety net that matters more with shorter histories. The mix with "
                "shrinkage is our extended recommendation. Waste-based: UCO and animal fats (tallow, pork fat). Corn is a rough proxy for "
                "distillers corn oil, lean hogs for white grease, feeder cattle and heating oil are extra proxies for tallow and UCO. Neste (renewable diesel, a large buyer of UCO and fats), Verbio (biodiesel) and Bunge (oilseed crushing) are company share prices used as proxies: they also move with each company's profits and the stock market, and they never count as waste-based. Prices stay in their own currency: rapeseed in euros, canola in "
                "Canadian dollars, palm oil in ringgit, Neste and Verbio in euros, the others in dollars. Palm oil is high-ILUC: under "
                "the Dutch 2026 rules a real buyer could use almost none, so treat its weight as a comparison only. "
                "Weeks with a rapeseed contract roll are removed."
            )

            st.markdown("To see this 15-feedstock mix as a chart, with its risk budgeting, choose "
                        "**Extended universe (15 feedstocks)** in the switch under Recommended allocation.")

            # ---- commodity benchmark: tracking error and beta ----
            if _xb_ser is None or len(_xb_ser) < 250:
                st.info("The commodity benchmark could not be downloaded right now.")
            else:
                _xall = pd.concat([_xp, _xb_ser.rename("Benchmark")], axis=1, sort=True).ffill(limit=5).dropna()
                _xra = _xall.resample("W-FRI").last().pct_change().dropna()
                if "Rapeseed (Euronext)" in _xra.columns:
                    _xra = _xra[~(_xra.index.month.isin([2, 5, 8, 11]) & (_xra.index.day <= 9))]
                _rb = _xra["Benchmark"].values
                _RX = _xra[_xnames].values
                _cov_full = np.cov(np.column_stack([_RX, _rb]), rowvar=False)
                _Sx, _sxb, _vb = _cov_full[:_xn, :_xn], _cov_full[:_xn, -1], _cov_full[-1, -1]

                def _x_te(x):
                    return float(np.sqrt(max(x @ _Sx @ x - 2 * x @ _sxb + _vb, 0.0) / 5))

                def _x_beta(x):
                    return float((x @ _sxb) / _vb)

                _res_te = _xminimize(lambda x: x @ _Sx @ x - 2 * x @ _sxb, np.ones(_xn) / _xn, method="SLSQP",
                                     bounds=_xbnds, constraints=_xcons, options={"maxiter": 1000, "ftol": 1e-14})
                _xw_te = _res_te.x if _res_te.success else None

                st.markdown(f"**Your feedstock costs vs the commodity market ({_xbench_name})**")
                xb1, xb2, xb3 = st.columns(3)
                xb1.metric("Tracking error (daily)", f"{_x_te(_xw_shrunk):.2%}",
                           help="How much your extended mix's cost moves differently from the benchmark each day "
                                "(weekly tracking error ÷ √5).")
                xb2.metric("Beta to the benchmark", f"{_x_beta(_xw_shrunk):.2f}",
                           help="1.0 = your costs move one-for-one with the commodity market.")
                xb3.metric("Correlation", f"{_x_beta(_xw_shrunk) * np.sqrt(_vb) / np.sqrt(_xw_shrunk @ _Sx @ _xw_shrunk):.0%}")

                if _xw_te is not None:
                    st.dataframe(pd.DataFrame({
                        "Extended mix (stable costs)": [f"{max(v, 0):.1%}" for v in _xw_shrunk]
                            + [f"{np.sqrt(_xw_shrunk @ _Sx @ _xw_shrunk / 5):.2%}", f"{_x_te(_xw_shrunk):.2%}"],
                        "Mix closest to the benchmark": [f"{max(v, 0):.1%}" for v in _xw_te]
                            + [f"{np.sqrt(_xw_te @ _Sx @ _xw_te / 5):.2%}", f"{_x_te(_xw_te):.2%}"],
                    }, index=_xnames + ["Daily cost risk", "Tracking error vs benchmark (daily)"]))

                _xidx = pd.DataFrame({"Your extended mix": np.cumprod(1 + _RX @ _xw_shrunk) * 100,
                                      _xbench_name: np.cumprod(1 + _rb) * 100}, index=_xra.index)
                st.line_chart(_xidx)
                _xcorr = _x_beta(_xw_shrunk) * np.sqrt(_vb) / np.sqrt(_xw_shrunk @ _Sx @ _xw_shrunk)
                if _xcorr < 0.3:
                    _xmsg = ("Your costs barely follow the commodity market. They have their own drivers (fats, UCO, "
                             "European rapeseed, biofuel rules), so a commodity index is not a hedge for you.")
                elif _xcorr < 0.7:
                    _xmsg = (f"Your costs follow the commodity market only partly: the market explains about "
                             f"{_xcorr ** 2:.0%} of your cost swings (correlation²). The shared part comes from corn, "
                             f"soybeans, cattle and hogs, which are also inside the index. The rest comes from your own "
                             f"drivers (UCO, European rapeseed, biofuel rules). A commodity index is only a partial "
                             f"hedge; your own feedstock futures are more precise.")
                else:
                    _xmsg = ("Your costs move closely with the commodity market, so a commodity index would already "
                             "hedge a large part of your risk.")
                if _xw_te is not None:
                    _xmsg += (f" Even the mix closest to the benchmark keeps a tracking error of {_x_te(_xw_te):.2%} a "
                              f"day: with your rules, you cannot copy the market.")
                st.caption(
                    "Tracking error = √[xᵀΣx − 2xᵀσ_b + σ_b²]: the daily volatility of the gap between your cost and "
                    "the benchmark (LN 2). " + _xmsg
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

    _idx = _wk / _wk.iloc[0] * 100
    fig_rs, ax_rs = plt.subplots(figsize=(9, 4))
    ax_rs.plot(_idx.index, _idx["Rapeseed (Euronext)"], color="black", linewidth=1.5,
               label="European rapeseed (Euronext): your real price")
    ax_rs.plot(_idx.index, _idx["ZS=F"], color=TICKER_COLORS["ZS=F"], linewidth=1,
               label="US soybeans (ZS=F): our rapeseed proxy")
    ax_rs.plot(_idx.index, _idx["ZL=F"], color=TICKER_COLORS["ZL=F"], linewidth=1, linestyle="--",
               label="US soybean oil (ZL=F)")
    ax_rs.set_ylabel("Price index (start = 100)")
    ax_rs.legend()
    st.pyplot(fig_rs)
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

    _names = {"ZL=F": "Only soybean oil (ZL=F)", "ZS=F": "Only rapeseed (soybeans, ZS=F)",
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
        "The equal split is only a comparison: it breaks the tallow cap (LE=F + DAR = 40%). "
        "Live cattle alone looks as calm as our mix, but it breaks your cap (100% in one feedstock), "
        "and it is only a proxy: real tallow prices can move differently from cattle prices, and one feedstock "
        "cannot cover all your supply."
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
                                 value=1180.0, step=10.0, key="pc_price")
    with _c2:
        _tonnes = st.number_input("Tonnes sold per month", min_value=0.0,
                                  value=10000.0, step=500.0, key="pc_tonnes")
        _other = st.number_input("Other costs ($ per tonne: energy, staff, transport)", min_value=0.0,
                                 value=100.0, step=10.0, key="pc_other")
    _share = st.slider("If the price follows diesel: share of the selling price that moves with diesel", 0, 100, 67, 1,
                       key="pc_share", format="%d%%",
                       help="Example: biodiesel sold at the diesel price + a fixed premium of $400/t. "
                            "At $1,180/t, the diesel part is $780, so 66% of the price moves with diesel.") / 100

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
    if _follows and _par_margin < _par_main:
        st.info("On our free data, the stable-margin mix looks better here, but the gain is overstated: diesel "
                "(HO=F) is both our UCO proxy and your selling price. Our recommendation stays the stable-cost "
                "mix, plus a gasoil futures hedge on your selling price. Check the gain on your own prices first.")
    if _profit < 0:
        st.error("With these inputs you lose money even in a normal month: sales are lower than "
                 "feedstock + other costs. Check the selling price, tonnes and the budget in the sidebar.")
    elif _profit - _par_best < 0:
        st.warning("In a bad month, the profit could turn into a loss. You may also need futures to lock your prices.")
    _breakeven = (_feed + _other_total) / _tonnes if _tonnes > 0 else 0.0
    st.markdown(
        f"**Your break-even selling price: ${_breakeven:,.0f} per tonne** "
        f"(feedstock + other costs, divided by tonnes). Below it you lose money, above it you make a profit."
        .replace("$", "\\$")
    )
    if _profit - _par_best < 0:
        st.markdown(
            "**What if the market pays less than your costs?** FAME is a market price (ICE gasoil + a premium), "
            "so you can't simply raise your selling price. You have three choices:\n"
            "1. **Lower your costs:** cheaper or waste-based feedstocks, better buying, lower processing costs.\n"
            "2. **Sell a better-paid product:** UCOME usually sells above FAME 0°, because waste-based biodiesel "
            "counts double for the RED mandates.\n"
            "3. **Produce less, or pause the plant,** until margins come back."
        )
    st.caption(
        "Profit-at-Risk = √(dᵀΣd) × √days × z, where d = dollar exposure: + the diesel-linked part of sales "
        "(only if the price follows diesel), − feedstock budget × mix weights. Same parametric formula as the "
        "Cost-at-Risk (normal returns, not from the course). Selling price, tonnes and other costs are your "
        "inputs; only the risks come from the model. Two limits: the feedstock bill is the budget in the "
        "sidebar, so keep it consistent with your tonnes; and diesel (HO=F) is both our UCO proxy and the "
        "selling price, so the stable-margin gain looks larger than with real prices."
    )


# --- hedging: lock your margin with futures and forward contracts ------------

with st.expander("Hedging: lock your margin (futures and forward contracts)"):
    st.markdown(
        "Our mix makes your costs as **stable** as possible. A hedge goes one step further: it **locks** a "
        "price today for the coming months. Choose how much you want to lock. The numbers use the inputs of "
        "the Profit-at-Risk calculator above."
    )
    st.markdown(
        "**Our recommended hedge:** lock **100% of your crop oils** with futures (they create most of your "
        "risk and their futures are easy to trade), and **50% of your UCO and tallow** with forward contracts "
        "(you keep some room if prices fall). If your selling price follows diesel, also lock the selling "
        "price with gasoil futures."
    )

    def _hg_recommended():
        st.session_state["hg_crop"] = 100
        st.session_state["hg_waste"] = 50
        st.session_state["hg_sell"] = 100 if st.session_state.get("pc_sales") == "Selling price follows diesel" else 0

    def _hg_reset():
        for _k in ("hg_crop", "hg_waste", "hg_sell"):
            st.session_state[_k] = 0

    _hb1, _hb2, _hb3 = st.columns([2, 1, 3])
    _hb1.button("Use our recommended hedge", on_click=_hg_recommended, key="hg_btn")
    _hb2.button("No hedge", on_click=_hg_reset, key="hg_reset")

    _hc1, _hc2, _hc3 = st.columns(3)
    with _hc1:
        _h_crop = st.slider("Crop oils locked with futures", 0, 100, step=10, format="%d%%", key="hg_crop",
                            help="Soybean oil: CBOT soybean oil futures. Rapeseed oil: Euronext rapeseed futures.") / 100
    with _hc2:
        _h_waste = st.slider("UCO and tallow locked with forward contracts", 0, 100, step=10, format="%d%%",
                             key="hg_waste",
                             help="There are no liquid futures for UCO and tallow: you fix the price with your "
                                  "suppliers for the coming months.") / 100
    with _hc3:
        _h_sell = st.slider("Selling price locked with gasoil futures", 0, 100, step=10, format="%d%%",
                            key="hg_sell",
                            help="Only used if your selling price follows diesel (choose it in the calculator "
                                 "above): you sell ICE gasoil futures.") / 100

    _hv = np.array([_h_crop if t in ("ZL=F", "ZS=F") else _h_waste for t in PORTFOLIO_TICKERS])

    def _par_hedged(hv, hs):
        # dollar exposure left after the hedges: the locked part of each price no longer moves
        d = (_revenue * _share * (1 - hs) * _b if _follows else 0.0) - _feed * _w_main * (1 - hv)
        return float(np.sqrt(d @ _cov @ d) * np.sqrt(_h) * _z)

    _p_none = _par_hedged(np.zeros(len(PORTFOLIO_TICKERS)), 0.0)
    _p_hedged = _par_hedged(_hv, _h_sell if _follows else 0.0)

    hm1, hm2, hm3 = st.columns(3)
    hm1.metric("Profit-at-Risk without hedge", _usd(_p_none),
               help="Our mix (stable costs), with the inputs of the calculator above.")
    hm2.metric("Profit-at-Risk with your hedges", _usd(_p_hedged),
               delta=f"{(_p_hedged - _p_none) / _p_none:+.0%}" if _p_hedged < _p_none else None, delta_color="inverse")
    hm3.metric("Profit in a bad month", _usd(_profit - _p_hedged),
               delta=f"{_usd(_p_none - _p_hedged)} more than without hedge" if _p_hedged < _p_none else None)

    _tn = {t: _tonnes * w for t, w in zip(PORTFOLIO_TICKERS, _w_main)}
    _hrows = [
        ["Soybean oil", "CBOT soybean oil futures (about 27 t per contract)", f"{_tn['ZL=F'] * _h_crop:,.0f} t"],
        ["Rapeseed oil", "Euronext rapeseed futures (50 t of seed per contract)", f"{_tn['ZS=F'] * _h_crop:,.0f} t"],
        ["UCO", "Fixed-price forward contracts with your suppliers", f"{_tn['HO=F'] * _h_waste:,.0f} t"],
        ["Tallow", "Fixed-price forward contracts with your suppliers",
         f"{(_tn['LE=F'] + _tn['DAR']) * _h_waste:,.0f} t"],
    ]
    if _follows:
        _hrows.append(["Selling price", "Sell ICE gasoil futures (100 t per contract)",
                       f"{_tonnes * _share * _h_sell:,.0f} t"])
    _per_contract = {"Soybean oil": 27.2, "Rapeseed oil": 50 / 2.4, "Selling price": 100.0}
    for _r in _hrows:
        _tq = float(_r[2].replace(",", "").replace(" t", ""))
        _r.append(f"about {_tq / _per_contract[_r[0]]:,.0f}" if _r[0] in _per_contract and _tq > 0 else "—")
    st.table(pd.DataFrame(_hrows, columns=["What you lock", "How", "Tonnes per month", "Futures contracts"])
             .set_index("What you lock"))
    _locked = float(sum(_tn[t] * h for t, h in zip(PORTFOLIO_TICKERS, _hv)))
    _open = max(_tonnes - _locked, 0.0)
    st.markdown(
        f"**In total:** you lock about **{_locked:,.0f} t** of your {_tonnes:,.0f} t a month "
        f"({_locked / _tonnes:.0%}). About **{_open:,.0f} t stay at market price**: for these, you pay "
        f"whatever the market asks that month." if _tonnes > 0 else ""
    )

    if _profit <= 0:
        st.warning("Your margin is negative at these prices: a hedge would only lock in this loss. "
                   "Look at the three choices in the Profit-at-Risk calculator first.")

    st.caption(
        "A hedge does not make your feedstocks cheaper: it fixes the price. You are protected if prices rise, "
        "but you don't gain if they fall, and futures need cash for margin calls. Best practice: when you sign "
        "a fixed selling price, lock your feedstock costs on the same day (back-to-back), so your margin is "
        "locked too. Limits: the numbers assume each hedge follows your real price perfectly. In reality, basis "
        "risk remains, for example between Euronext rapeseed (a seed) and your rapeseed oil. Tonnes assume "
        "about 1 tonne of feedstock per tonne of biodiesel. Rapeseed contracts use about 2.4 t of seed per tonne "
        "of oil: a broker would set the exact hedge ratio."
    )


# --- recommended data sources for the next version ---------------------------

with st.expander("Recommended data sources (next version of your tool)"):
    st.markdown(
        "Today the tool uses free proxies. With professional data, it would use **your real prices** "
        "and **your real rules** instead of our assumptions."
    )
    st.table(pd.DataFrame({
        "Source we recommend": [
            "Argus or SquareCommodities (spot prices)",
            "Argus or SquareCommodities (FAME 0°, UCOME)",
            "SquareCommodities (regulatory data)",
            "ICE (gasoil, FAME 0°), CBOT (soybean oil), Euronext (rapeseed), your suppliers (UCO, tallow)",
            "Nabisy",
            "EIA, IEA, JODI (oil and diesel), USDA (oilseeds)",
        ],
        "What it replaces or adds in the tool": [
            "Replaces our proxies (HO=F, LE=F, DAR, ZS=F), so most of the basis risk disappears",
            "Replaces diesel (HO=F) in the stable-margin mix, and gives the real selling price for the Profit-at-Risk calculator and your break-even",
            "Replaces our assumed 40% RED2 minimum with the real mandate level",
            "With your real prices, turns our rough contract numbers in the hedging box into exact hedge ratios",
            "Confirms which volumes count as waste-based for RED2",
            "Helps you decide when and how much to hedge (supply, demand, stocks)",
        ],
    }, index=[
        "Your feedstock prices (UCO, tallow, rapeseed oil, soybean oil)",
        "Your selling price (FAME 0°, UCOME)",
        "The real mandate levels",
        "Futures for the hedge",
        "Sustainability proofs",
        "Market fundamentals",
    ]))
    st.markdown(
        "**Links:** "
        "[Argus](https://www.argusmedia.com/en/commodities/biofuels-and-feedstocks) · "
        "[SquareCommodities](https://www.squarecommodities.com) · "
        "[ICE](https://www.ice.com) · "
        "[CBOT / CME Group](https://www.cmegroup.com) · "
        "[Euronext](https://live.euronext.com) · "
        "[Nabisy](https://nabisy.ble.de/app/start) · "
        "[EIA](https://www.eia.gov) · "
        "[IEA](https://www.iea.org) · "
        "[JODI](https://www.jodidata.org) · "
        "[USDA FAS](https://fas.usda.gov)"
    )
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
                _gain = 1 - _ms / _mm
                st.caption(f"How much your margin moves per row: main mix {_mm:.2%}, stable-margin mix "
                           f"{_ms:.2%}, that is {_gain:.0%} calmer. "
                           + ("A small gain like this means none of your feedstocks follows your selling "
                              "price closely: to protect your margin, a futures hedge on the selling side "
                              "works better." if _gain < 0.2 else
                              "A large gain: some of your feedstocks follow your selling price, so the "
                              "stable-margin mix is worth considering, together with a futures hedge."))

            st.caption("Same method and same rules as the main app (minimum variance, RED2 minimum, cap per "
                       "feedstock), but on your real prices, so the basis risk from our proxies disappears. "
                       "Some basis risk remains when you hedge, between your prices and the futures. "
                       "Waste-based feedstocks: UCO and tallow. Cost-at-Risk assumes normal returns.")


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
