"""
data_loader.py

Pulls feedstock proxy prices from yfinance, cleans them, converts to
daily returns for the optimizer.

    ZL=F  Soybean oil futures      -> crop-based
    ZS=F  Soybean futures          -> crop-based (originally RS=F/canola,
                                       swapped -- Yahoo doesn't carry that
                                       ticker, see README basis-risk section)
    HO=F  NY Harbor ULSD (diesel)  -> UCO proxy, waste-based
    LE=F  Live cattle futures      -> tallow proxy, waste-based
    DAR   Darling Ingredients      -> tallow/UCO proxy, waste-based
    CL=F  WTI crude futures        -> crack-spread input only, not a weight
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

TICKERS = ["ZL=F", "ZS=F", "HO=F", "LE=F", "DAR", "CL=F"]
PORTFOLIO_TICKERS = ["ZL=F", "ZS=F", "HO=F", "LE=F", "DAR"]  # CL=F excluded, see above
WASTE_BASED = ["HO=F", "LE=F", "DAR"]
CROP_BASED = ["ZL=F", "ZS=F"]

CACHE_DIR = "data_cache"
CACHE_MAX_AGE_HOURS = 24


def _cache_file_for(years):
    """
    One cache file per lookback window, e.g. data_cache/feedstock_prices_5y.csv.

    Previously there was a single CACHE_FILE shared across every value of
    `years`. As long as *any* cache existed and was under 24h old, every
    call -- 1 year, 5 years, 10 years -- returned that same file's data,
    so the "Years of historical data" slider silently did nothing after
    the first pull of the day. Keying the filename on years fixes that:
    each lookback window gets (and reuses) its own cache.
    """
    return os.path.join(CACHE_DIR, f"feedstock_prices_{years}y.csv")


def _cache_is_fresh(cache_file):
    if not os.path.exists(cache_file):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
    return age < timedelta(hours=CACHE_MAX_AGE_HOURS)


def fetch_prices(years=5, force_refresh=False):
    """Daily close prices for all TICKERS, cached locally per `years` value for a day at a time."""
    cache_file = _cache_file_for(years)

    if not force_refresh and _cache_is_fresh(cache_file):
        try:
            prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not prices.empty:
                return prices
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
            pass  # corrupt cache -- fall through and re-pull

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    raw = yf.download(
        TICKERS,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=False,  # threads=True hammers yfinance's sqlite cache
                         # concurrently -> "database is locked" on Windows
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()  # shouldn't hit this with 6 tickers
        prices.columns = TICKERS[:1]

    prices = prices[TICKERS]
    os.makedirs(CACHE_DIR, exist_ok=True)
    prices.to_csv(cache_file)
    return prices


def sanity_check(prices):
    """Per-ticker date range / observation count / missing %, before trusting the data."""
    report = {}
    for ticker in TICKERS:
        col = prices[ticker]
        report[ticker] = {
            "first_date": col.first_valid_index(),
            "last_date": col.last_valid_index(),
            "n_observations": col.notna().sum(),
            "n_missing": col.isna().sum(),
            "pct_missing": round(100 * col.isna().mean(), 2),
        }
    return report


def clean_prices(prices):
    """
    Forward/back-fill small gaps (holiday mismatches, staggered start
    dates), then drop whatever's left. Futures rollover on the =F
    tickers can still cause small jumps -- known limitation, noted in
    the README rather than engineered around here.
    """
    fully_missing = [t for t in prices.columns if prices[t].isna().all()]
    if fully_missing:
        # catch this before dropna(), which would otherwise silently
        # empty the whole frame with no error to explain why
        raise ValueError(
            f"These tickers came back with zero data, probably a failed "
            f"download: {fully_missing}. Re-run fetch_prices(force_refresh=True)."
        )
    return prices.ffill().bfill().dropna()


def compute_returns(prices):
    return prices.pct_change().dropna()


def compute_crack_spread(prices):
    """HO=F minus CL=F -- diagnostic only, not a portfolio weight."""
    return prices["HO=F"] - prices["CL=F"]


def get_returns_for_optimizer(years=5, force_refresh=False):
    """Fetch -> clean -> returns, PORTFOLIO_TICKERS only. What the app should call."""
    prices = clean_prices(fetch_prices(years=years, force_refresh=force_refresh))
    return compute_returns(prices)[PORTFOLIO_TICKERS]


if __name__ == "__main__":
    print("Pulling feedstock proxy price data...")
    prices = fetch_prices(years=5)

    print("\nSanity check:")
    for ticker, stats in sanity_check(prices).items():
        print(f"  {ticker:6s}  {stats['n_observations']:5d} obs  "
              f"({stats['first_date']} -> {stats['last_date']})  "
              f"missing: {stats['pct_missing']}%")

    cleaned = clean_prices(prices)
    returns = compute_returns(cleaned)

    print(f"\nCleaned price rows: {len(cleaned)}")
    print(f"Return rows:         {len(returns)}")
    print("\nSample returns:")
    print(returns.tail())
