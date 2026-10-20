"""
optimizer.py

The actual quant logic for the project:
    1. Constrained mean-variance optimization (SLSQP), swept across the
       *feasible* range of target returns given the active bounds and
       constraints, to trace the constrained efficient frontier -- this
       is the main method from the brief.
    2. A truly unconstrained frontier (budget + long-only, nothing
       else) computed the same way, so the app can show what the
       sustainability/concentration rules actually cost the client.
    3. Risk budgeting (the secondary lens): % contribution of each
       feedstock to total portfolio variance at a given point on the
       frontier.
    4. A pairwise correlation helper, used to back the README's
       crop-based diversification note with a live number instead of
       an assertion.
    5. A bucket-split helper, used to group any per-ticker array
       (weights, risk contributions) into waste-based vs. crop-based
       subtotals for the app's charts.

Everything here works in daily returns -- annualizing for display is
left to streamlit_app.py so this file stays focused on the math.
"""

import numpy as np
from scipy.optimize import minimize

from constraints import build_all_constraints, get_bounds, budget_constraint
from data_loader import PORTFOLIO_TICKERS, WASTE_BASED


# ---------------------------------------------------------------------------
# Basic portfolio math
# ---------------------------------------------------------------------------

def portfolio_return(weights, mean_returns):
    """w' * mu"""
    return np.dot(weights, mean_returns)


def portfolio_variance(weights, cov_matrix):
    """w' * Sigma * w -- what SLSQP is actually minimizing."""
    return np.dot(weights.T, np.dot(cov_matrix, weights))


def portfolio_volatility(weights, cov_matrix):
    """sqrt of variance -- the risk number that's actually readable."""
    return np.sqrt(portfolio_variance(weights, cov_matrix))


# ---------------------------------------------------------------------------
# Single-point optimization: minimize variance for a given target return
# ---------------------------------------------------------------------------

def _target_return_constraint(target_return, mean_returns):
    """Pins the portfolio's expected return to a specific target."""
    def _fn(weights):
        return portfolio_return(weights, mean_returns) - target_return
    return {"type": "eq", "fun": _fn}


def minimize_variance_for_target(target_return, mean_returns, cov_matrix,
                                  extra_constraints, bounds, n_assets):
    """
    Solve: minimize variance subject to hitting target_return exactly,
    plus whatever's in extra_constraints, plus bounds.

    Returns the raw scipy OptimizeResult -- caller decides what to do
    if it didn't converge, rather than this function guessing.
    """
    all_constraints = list(extra_constraints) + [
        _target_return_constraint(target_return, mean_returns)
    ]

    # equal-weight start -- simple and usually good enough for SLSQP to
    # converge from, no need to get clever with the initial guess here
    x0 = np.ones(n_assets) / n_assets

    return minimize(
        portfolio_variance,
        x0,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=all_constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )


# ---------------------------------------------------------------------------
# Frontier sweeps
# ---------------------------------------------------------------------------

def _feasible_return_range(mean_returns, extra_constraints, bounds, n_assets):
    """
    What compute_frontier() actually sweeps between -- the min and max
    portfolio return that's reachable given the *active* bounds and
    constraints, not the raw per-asset min/max.

    Why this matters: mean_returns.min()/.max() describes what a single
    asset could return on its own, completely ignoring the concentration
    cap and sustainability floor. A tight cap can collapse the feasible
    set down to one point or a narrow sliver -- e.g. a 0.20 cap across 5
    assets leaves exactly one feasible portfolio (5 x 0.20 = 1.0, no
    slack at all). Sweeping the old unconstrained range in that case
    means almost every swept target return misses the one point that's
    actually achievable, and every solve in compute_frontier() fails.

    This solves two small optimizations -- minimize and maximize
    portfolio_return -- subject to the same bounds/constraints the real
    frontier sweep will use, so the sweep range is exactly what's
    reachable.
    """
    x0 = np.ones(n_assets) / n_assets

    lo_result = minimize(
        lambda w: portfolio_return(w, mean_returns), x0,
        method="SLSQP", bounds=bounds, constraints=extra_constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    hi_result = minimize(
        lambda w: -portfolio_return(w, mean_returns), x0,
        method="SLSQP", bounds=bounds, constraints=extra_constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )

    if not (lo_result.success and hi_result.success):
        # couldn't even solve for the feasible range -- fall back to the
        # old unconstrained range rather than crash outright; the main
        # sweep below will then fail loudly the same way it used to,
        # which is at least no worse than before this fix
        return mean_returns.min(), mean_returns.max()

    return portfolio_return(lo_result.x, mean_returns), portfolio_return(hi_result.x, mean_returns)


def compute_frontier(mean_returns, cov_matrix, extra_constraints, bounds,
                      n_points=50):
    """
    Sweep target return across what's actually feasible given
    extra_constraints and bounds, solving minimize_variance_for_target at
    each point, and stitch the results into a frontier.

    Returns a dict of parallel arrays -- returns, volatilities,
    weights, plus a success mask so the caller can drop any points
    that didn't converge (some points near the extremes may genuinely
    be infeasible once the sustainability/concentration constraints
    are in play).
    """
    n_assets = len(mean_returns)
    lo_return, hi_return = _feasible_return_range(mean_returns, extra_constraints, bounds, n_assets)

    # tiny inward margin so the swept endpoints don't sit exactly on the
    # boundary of feasibility, where SLSQP can fail purely on floating-
    # point precision rather than genuine infeasibility. When lo_return
    # == hi_return (the single-feasible-point case, like the 0.20 cap
    # above), margin is 0 and every swept target is that one point --
    # which is correct, since that's the only portfolio that exists.
    margin = (hi_return - lo_return) * 1e-6
    target_returns = np.linspace(lo_return + margin, hi_return - margin, n_points)

    achieved_returns, achieved_vols, achieved_weights, success_flags = [], [], [], []

    for target in target_returns:
        result = minimize_variance_for_target(
            target, mean_returns, cov_matrix, extra_constraints, bounds, n_assets
        )
        success_flags.append(result.success)
        if result.success:
            w = result.x
            achieved_returns.append(portfolio_return(w, mean_returns))
            achieved_vols.append(portfolio_volatility(w, cov_matrix))
            achieved_weights.append(w)
        else:
            # keep a placeholder so every array stays the same length as
            # target_returns -- streamlit_app.py filters these out before plotting
            achieved_returns.append(np.nan)
            achieved_vols.append(np.nan)
            achieved_weights.append(np.full(n_assets, np.nan))

    return {
        "target_returns": target_returns,
        "returns": np.array(achieved_returns),
        "volatilities": np.array(achieved_vols),
        "weights": np.array(achieved_weights),
        "success": np.array(success_flags),
    }


def compute_constrained_frontier(mean_returns, cov_matrix,
                                  sustainability_min, concentration_cap,
                                  n_points=50):
    """
    The frontier the client actually has to live with: sustainability
    floor + concentration cap + budget + long-only, all switched on.

    NOTE: concentration_cap goes into get_bounds(), not
    build_all_constraints() -- that's just where it lives now that the
    cap is handled as a bound rather than an SLSQP constraint (see
    constraints.py). Forgetting to pass it here was a bug in an
    earlier version of this file -- the concentration slider would've
    silently done nothing.
    """
    extra_constraints = build_all_constraints(sustainability_min=sustainability_min)
    bounds = get_bounds(concentration_cap=concentration_cap, n_assets=len(mean_returns))
    return compute_frontier(mean_returns, cov_matrix, extra_constraints, bounds, n_points)


def compute_unconstrained_frontier(mean_returns, cov_matrix, n_points=50):
    """
    Budget + long-only ONLY -- no sustainability floor, no
    concentration cap. This is what gets plotted next to the
    constrained frontier so the client can see what the RED2 and
    concentration rules cost them in return-per-unit-risk.

    Important: this deliberately does NOT call get_bounds(), because
    get_bounds() defaults to the 0.35 concentration cap -- using it
    here would sneak the cap back in and make this frontier not
    actually unconstrained. Bounds are plain (0, 1) instead.
    """
    extra_constraints = [budget_constraint()]
    n_assets = len(mean_returns)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    return compute_frontier(mean_returns, cov_matrix, extra_constraints, bounds, n_points)


# ---------------------------------------------------------------------------
# Picking a single "the" recommended portfolio off the frontier
# ---------------------------------------------------------------------------

def min_variance_portfolio(frontier):
    """Lowest-volatility point on an already-computed frontier -- the safe default pick."""
    vols_masked = np.where(frontier["success"], frontier["volatilities"], np.inf)
    idx = np.argmin(vols_masked)
    return {
        "index": idx,
        "weights": frontier["weights"][idx],
        "return": frontier["returns"][idx],
        "volatility": frontier["volatilities"][idx],
    }


def max_sharpe_equivalent_portfolio(frontier, risk_free_rate=0.0):
    """
    Highest return/risk ratio on the frontier. Calling it
    "sharpe-equivalent" rather than a literal Sharpe ratio since we're
    working with cost/return premia on a feedstock basket, not excess
    equity returns over a risk-free rate in the usual sense.
    """
    returns, vols, valid = frontier["returns"], frontier["volatilities"], frontier["success"]

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(valid & (vols > 0), (returns - risk_free_rate) / vols, -np.inf)

    idx = np.argmax(ratios)
    return {
        "index": idx,
        "weights": frontier["weights"][idx],
        "return": frontier["returns"][idx],
        "volatility": frontier["volatilities"][idx],
        "sharpe_equivalent": ratios[idx],
    }


# ---------------------------------------------------------------------------
# Risk budgeting (secondary lens from the brief)
# ---------------------------------------------------------------------------

def risk_contributions(weights, cov_matrix):
    """
    RC_i = w_i * (Sigma @ w)_i / (w' Sigma w)

    These sum to 1.0 -- 100% of portfolio variance accounted for,
    split out by feedstock. This is the risk-budgeting bar chart the
    app shows alongside the main frontier.
    """
    total_variance = portfolio_variance(weights, cov_matrix)
    marginal_contributions = cov_matrix @ weights
    raw_contributions = weights * marginal_contributions
    return raw_contributions / total_variance


def sustainability_share(weights, tickers=PORTFOLIO_TICKERS):
    """Combined weight on the waste-based feedstocks, for the RED2 metric in the app."""
    return sum(w for t, w in zip(tickers, weights) if t in WASTE_BASED)


def pairwise_correlation(cov_matrix, tickers, ticker_a, ticker_b):
    """
    corr(a, b) = cov(a, b) / (std(a) * std(b)) -- read straight off the
    covariance matrix, no separate correlation matrix needed.

    This backs the "diversification note" in the README/app: rather than
    just asserting ZL=F and ZS=F move together because they're both CBOT
    soybean complex, this computes it from whatever data the app actually
    pulled, so the number is live and moves with the lookback window
    instead of being a claim someone has to take on faith.
    """
    i, j = tickers.index(ticker_a), tickers.index(ticker_b)
    std_a = np.sqrt(cov_matrix[i, i])
    std_b = np.sqrt(cov_matrix[j, j])
    return cov_matrix[i, j] / (std_a * std_b)


def bucket_split(values, tickers, waste_based):
    """
    Regroup a per-ticker array -- weights or risk contributions, either
    works -- into waste-based tickers first, then crop-based, instead of
    whatever order PORTFOLIO_TICKERS happens to be in.

    This is what lets streamlit_app.py draw the allocation chart as two
    visibly separate groups (RED2-eligible vs. not) instead of five flat
    bars with no bucket structure, which is what the roadmap actually
    asked for.
    """
    waste_tickers, waste_values, crop_tickers, crop_values = [], [], [], []
    for t, v in zip(tickers, values):
        if t in waste_based:
            waste_tickers.append(t)
            waste_values.append(v)
        else:
            crop_tickers.append(t)
            crop_values.append(v)
    return waste_tickers, waste_values, crop_tickers, crop_values


# ---------------------------------------------------------------------------
# Manual test run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # small synthetic dataset just to prove the optimizer runs end to
    # end and actually respects every constraint, before plugging in
    # real yfinance returns
    np.random.seed(0)
    n_assets = len(PORTFOLIO_TICKERS)
    mean_returns = np.array([0.0004, 0.0003, 0.0005, 0.0002, 0.0006])
    A = np.random.randn(n_assets, n_assets) * 0.01
    cov_matrix = A @ A.T  # guarantees this is a valid (PSD) covariance matrix

    print("Tickers:", PORTFOLIO_TICKERS)

    print("\n--- constrained frontier (10 points) ---")
    constrained = compute_constrained_frontier(
        mean_returns, cov_matrix,
        sustainability_min=0.40, concentration_cap=0.35,
        n_points=10,
    )
    print(f"Converged: {constrained['success'].sum()} / {len(constrained['success'])}")

    min_var = min_variance_portfolio(constrained)
    print(f"Min-variance weights: {np.round(min_var['weights'], 3)}  "
          f"(sum={min_var['weights'].sum():.4f}, max={min_var['weights'].max():.3f})")
    print(f"Return: {min_var['return']:.6f}  Volatility: {min_var['volatility']:.6f}")
    print(f"Sustainability share: {sustainability_share(min_var['weights']):.3f}  "
          f"(needs to be >= 0.40)")

    rc = risk_contributions(min_var["weights"], cov_matrix)
    print(f"Risk contributions: {np.round(rc, 3)}  sum={rc.sum():.4f}")

    print("\n--- unconstrained frontier (10 points), for comparison ---")
    unconstrained = compute_unconstrained_frontier(mean_returns, cov_matrix, n_points=10)
    uc_min_var = min_variance_portfolio(unconstrained)
    print(f"Min-variance weights: {np.round(uc_min_var['weights'], 3)}  "
          f"(max={uc_min_var['weights'].max():.3f} -- should be free to exceed 0.35 here)")
