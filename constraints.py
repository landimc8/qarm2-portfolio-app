"""Client rules for the feedstock procurement portfolio.

Rotterdam Biofuels SA has three rules:
  1. Waste share: at least `min_waste_share` of volume must come from
     waste-based feedstocks (EU RED II).
  2. Concentration cap: no single feedstock above `max_weight`.
  3. Budget: the weighted average price must not exceed `max_avg_price`
     (optional, expressed in the same unit as the price data).

Weights are shares of total procurement volume. They are non-negative
and sum to 1 (fully invested, no short selling).

This module has no optimizer logic. It only describes the rules and turns
them into the format scipy.optimize.minimize (SLSQP) expects, so that
optimizer.py and streamlit_app.py share a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Default feedstock universe. Adjust once data_loader.py defines the final tickers.
DEFAULT_FEEDSTOCKS = ["UCO", "Tallow", "Soybean oil", "Rapeseed oil", "Palm oil"]

# Waste-based / advanced feedstocks under RED II (Annex IX).
DEFAULT_WASTE_BASED = {"UCO", "Tallow"}


@dataclass
class Constraints:
    """Client rules. Every field can be set from a Streamlit widget."""

    feedstocks: list[str] = field(default_factory=lambda: list(DEFAULT_FEEDSTOCKS))
    waste_based: set[str] = field(default_factory=lambda: set(DEFAULT_WASTE_BASED))
    min_waste_share: float = 0.30       # RED II minimum share of waste-based feedstock
    max_weight: float = 0.40            # concentration cap per feedstock
    max_avg_price: float | None = None  # budget: ceiling on the weighted average price

    # ------------------------------------------------------------------ helpers
    @property
    def n(self) -> int:
        return len(self.feedstocks)

    @property
    def waste_mask(self) -> np.ndarray:
        """1.0 for waste-based feedstocks, 0.0 otherwise (same order as feedstocks)."""
        return np.array([1.0 if f in self.waste_based else 0.0 for f in self.feedstocks])

    def validate(self) -> list[str]:
        """Return a list of human-readable problems. Empty list means the rules are feasible
        (as far as can be told without prices)."""
        problems = []
        if self.n == 0:
            return ["No feedstock selected."]
        unknown = self.waste_based - set(self.feedstocks)
        if unknown:
            problems.append(f"Waste-based feedstocks not in the universe: {sorted(unknown)}.")
        if not 0 <= self.min_waste_share <= 1:
            problems.append("Minimum waste share must be between 0% and 100%.")
        if not 0 < self.max_weight <= 1:
            problems.append("Concentration cap must be between 0% and 100%.")
        if self.max_weight * self.n < 1 - 1e-9:
            problems.append(
                f"The cap is too low: {self.n} feedstocks at {self.max_weight:.0%} each "
                "cannot reach 100% of the volume."
            )
        n_waste = int(self.waste_mask.sum())
        if n_waste * self.max_weight < self.min_waste_share - 1e-9:
            problems.append(
                f"Waste target unreachable: {n_waste} waste-based feedstock(s) capped at "
                f"{self.max_weight:.0%} give at most {n_waste * self.max_weight:.0%}, "
                f"below the {self.min_waste_share:.0%} required."
            )
        return problems

    def check_prices(self, prices: np.ndarray) -> list[str]:
        """Extra feasibility check once prices are known: can the budget be met at all?
        Cheapest feasible portfolio is found with a simple greedy fill."""
        if self.max_avg_price is None:
            return []
        prices = np.asarray(prices, dtype=float)
        cheapest = _min_avg_price(prices, self)
        if cheapest is not None and cheapest > self.max_avg_price + 1e-9:
            return [
                f"Budget too tight: the cheapest compliant portfolio costs "
                f"{cheapest:,.2f}, above the ceiling of {self.max_avg_price:,.2f}."
            ]
        return []

    # ------------------------------------------------------- scipy-ready output
    def bounds(self) -> list[tuple[float, float]]:
        """Per-feedstock bounds: no short selling, capped concentration."""
        return [(0.0, self.max_weight)] * self.n

    def scipy_constraints(self, prices: np.ndarray | None = None) -> list[dict]:
        """Constraint list for scipy.optimize.minimize(method="SLSQP").

        'eq'  means fun(w) == 0.
        'ineq' means fun(w) >= 0.
        """
        cons = [
            # Fully invested: weights sum to 1
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)},
            # Waste share >= minimum
            {
                "type": "ineq",
                "fun": lambda w, m=self.waste_mask: m @ w - self.min_waste_share,
                "jac": lambda w, m=self.waste_mask: m,
            },
        ]
        if self.max_avg_price is not None:
            if prices is None:
                raise ValueError("A budget is set, so prices are needed to build the constraint.")
            p = np.asarray(prices, dtype=float)
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda w, p=p: self.max_avg_price - p @ w,
                    "jac": lambda w, p=p: -p,
                }
            )
        return cons

    def starting_point(self) -> np.ndarray:
        """Equal weights nudged towards feasibility; a reasonable SLSQP starting guess."""
        w = np.full(self.n, 1.0 / self.n)
        mask = self.waste_mask
        if mask.sum() > 0 and mask @ w < self.min_waste_share:
            # Shift volume from non-waste to waste feedstocks, respecting the cap.
            w = np.where(mask == 1, min(self.max_weight, self.min_waste_share / mask.sum()), 0.0)
            rest = 1.0 - w.sum()
            non_waste = np.where(mask == 0)[0]
            if len(non_waste):
                w[non_waste] = rest / len(non_waste)
        return np.clip(w, 0.0, self.max_weight)

    def check_solution(self, w: np.ndarray, prices: np.ndarray | None = None, tol: float = 1e-6) -> dict:
        """Report which rules a given weight vector respects (useful for the Results tab)."""
        w = np.asarray(w, dtype=float)
        out = {
            "fully_invested": abs(w.sum() - 1) <= tol,
            "no_short": bool((w >= -tol).all()),
            "cap_respected": bool((w <= self.max_weight + tol).all()),
            "waste_share": float(self.waste_mask @ w),
            "waste_ok": bool(self.waste_mask @ w >= self.min_waste_share - tol),
        }
        if self.max_avg_price is not None and prices is not None:
            avg = float(np.asarray(prices, dtype=float) @ w)
            out["avg_price"] = avg
            out["budget_ok"] = avg <= self.max_avg_price + tol
        return out


def _min_avg_price(prices: np.ndarray, c: Constraints) -> float | None:
    """Cheapest weighted average price that satisfies cap and waste share (greedy LP)."""
    mask = c.waste_mask.astype(bool)
    w = np.zeros(c.n)
    # 1) Meet the waste share with the cheapest waste feedstocks
    need = c.min_waste_share
    for i in sorted(np.where(mask)[0], key=lambda i: prices[i]):
        take = min(c.max_weight, need)
        w[i] = take
        need -= take
        if need <= 1e-12:
            break
    if need > 1e-9:
        return None
    # 2) Fill the remaining volume with the cheapest feedstocks overall
    remaining = 1.0 - w.sum()
    for i in np.argsort(prices):
        room = c.max_weight - w[i]
        add = min(room, remaining)
        w[i] += add
        remaining -= add
        if remaining <= 1e-12:
            break
    return float(prices @ w) if remaining <= 1e-9 else None
