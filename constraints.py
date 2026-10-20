"""
constraints.py

Real-world constraints the procurement director cares about, kept
separate from optimizer.py so the optimization logic doesn't get
buried under constraint-building code.

    1. Sustainability minimum -- RED2 needs a minimum waste-based
       share (HO=F, LE=F, DAR). SLSQP inequality constraint.
    2. Full investment -- weights sum to 1. SLSQP equality constraint.
    3. Long-only + concentration cap -- no shorting, no single
       feedstock dominating. Just bounds, not SLSQP constraints
       (see get_bounds()).

scipy wants inequality functions to return >= 0 when satisfied, and
equality ones to return exactly 0 -- easy to flip a sign by accident,
so noting it here.
"""

import numpy as np

from data_loader import WASTE_BASED, PORTFOLIO_TICKERS

DEFAULT_SUSTAINABILITY_MIN = 0.40   # at least 40% waste-based, for RED2
DEFAULT_CONCENTRATION_CAP = 0.35    # no feedstock above 35% of the mix


def _waste_based_indices(tickers=PORTFOLIO_TICKERS):
    """Where the waste-based tickers sit in the weight vector."""
    return [tickers.index(t) for t in WASTE_BASED if t in tickers]


def sustainability_constraint(sustainability_min=DEFAULT_SUSTAINABILITY_MIN,
                               tickers=PORTFOLIO_TICKERS):
    """Combined HO=F + LE=F + DAR weight must clear sustainability_min for RED2 eligibility."""
    waste_idx = _waste_based_indices(tickers)

    def _constraint_fn(weights):
        return sum(weights[i] for i in waste_idx) - sustainability_min  # >= 0 is fine

    return {"type": "ineq", "fun": _constraint_fn}


def budget_constraint():
    """Weights sum to 1 -- spend the whole procurement budget, nothing left over."""
    return {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}


def get_bounds(concentration_cap=DEFAULT_CONCENTRATION_CAP, n_assets=len(PORTFOLIO_TICKERS)):
    """
    Long-only, capped at concentration_cap instead of 1.0.

    The cap used to be its own set of SLSQP inequality constraints,
    one per asset -- overkill. A cap on a single weight is just a box
    bound, and scipy's bounds= handles that natively and cheaper than
    a pile of constraint functions doing the same job.
    """
    return tuple((0.0, concentration_cap) for _ in range(n_assets))


def build_all_constraints(sustainability_min=DEFAULT_SUSTAINABILITY_MIN, tickers=PORTFOLIO_TICKERS):
    """
    What optimizer.py passes to minimize() as constraints=. Just
    sustainability + budget -- concentration cap and long-only live in
    get_bounds(), passed separately as bounds=.
    """
    return [sustainability_constraint(sustainability_min, tickers), budget_constraint()]


if __name__ == "__main__":
    # equal weight should pass everything at the defaults: 5 assets at
    # 20% each, under the 35% cap, waste-based trio sums to 60% (>= 40%)
    test_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    constraints = build_all_constraints()
    bounds = get_bounds()

    print(f"Testing equal-weight portfolio: {test_weights}")
    print(f"Ticker order: {PORTFOLIO_TICKERS}\n")

    for i, c in enumerate(constraints):
        value = c["fun"](test_weights)
        ok = (value >= -1e-8) if c["type"] == "ineq" else abs(value) < 1e-8
        print(f"  Constraint {i} ({c['type']}): value = {value:.4f}  {'OK' if ok else 'VIOLATED'}")

    print(f"\nBounds (cap = {DEFAULT_CONCENTRATION_CAP}): {bounds}")
    for w, (lo, hi) in zip(test_weights, bounds):
        ok = lo <= w <= hi
        print(f"  weight={w:.2f}  range=({lo}, {hi})  {'OK' if ok else 'VIOLATED'}")
