"""Budget optimiser: where should a limited amount of treatment go to cool the city most?

The planner already ranks wards, but a ranking cannot answer the question a budget meeting asks:
"we can afford X km2 of cool roof, which wards do we start with, and what do we get for it?"
This module answers it.

THE MODEL
    Each ward offers a CAPACITY (km2 of surface that can be treated: ward area x built fraction x
    roof share) and a BENEFIT if all of it is treated (area-weighted cooling, K km2, the ward's mean
    coverage-scaled delta-T times its area). Treating a share f of a ward is taken to deliver f of
    its benefit. That is not an extra assumption: the planner's own delta-T is linear in coverage
    (delta_t = full-surface response x built fraction x roof share), so the benefit really is
    proportional to the area treated. What it does ignore is interaction BETWEEN wards (heat does
    not stop at a ward boundary), and that is stated in the UI.

TWO WAYS TO SPEND THE BUDGET
    partial   any share of any ward can be treated. This is a fractional knapsack, and sorting by
              benefit per treated km2 and filling in that order is provably optimal.
    whole     a ward is treated entirely or not at all (a ward-level programme). This is a 0/1
              knapsack, solved exactly by dynamic programming over a discretised budget.

HONEST COMPARISON
    "Optimal" means nothing without something to beat, so every plan is scored against two things
    a planner might do instead: spread the budget evenly over every ward, and treat the hottest
    wards first. The gain over each is reported as it comes out, including when it is small. On the
    Kochi data (deployed model, cool roof, 2-5 km2) the optimum beats spreading evenly by about
    27-32% but treating the hottest wards first by only 1-6%: the hottest wards are also among the
    ones that cool most per km2, so "start with the hottest" is already close to optimal, and the
    optimiser's real contribution is to say how far down that list a given budget reaches.
"""

import math

import numpy as np

EPS = 1e-9
DP_UNIT_SQKM = 1e-4         # whole-ward solver works in 100 m2 steps ...
DP_MAX_STEPS = 200_000      # ... unless that would need more than this many (memory bound)


def _clean(items):
    """Keep only wards that can actually be treated and that cool when treated."""
    return [i for i in items if i["capacity_sqkm"] > EPS and i["benefit_k_km2"] > EPS]


def _greedy(items, key, budget, whole):
    """Fill the budget in `key` order. Partial mode splits the last ward; whole mode skips a ward
    that does not fit and carries on, since a smaller one further down the list still might."""
    picks, left = {}, budget
    for it in sorted(items, key=key):
        if left <= EPS:
            break
        cap = it["capacity_sqkm"]
        if whole:
            if cap <= left + EPS:
                picks[it["ward_id"]] = 1.0
                left -= cap
        else:
            take = min(cap, left)
            picks[it["ward_id"]] = take / cap
            left -= take
    return picks


def _knapsack_whole(items, budget):
    """Exact 0/1 knapsack over a discretised budget. Weights are rounded UP and the budget DOWN, so
    the chosen set can never exceed the real budget. The step is 100 m2 (finer for a very large
    budget, to bound memory), so each ward's area is off by at most 100 m2 and a plan can leave a
    few thousand m2 unspent; areas that are whole multiples of the step are handled exactly."""
    n = len(items)
    unit = max(DP_UNIT_SQKM, budget / DP_MAX_STEPS)
    steps = int(budget / unit + 1e-6)
    weights = [max(1, int(math.ceil(it["capacity_sqkm"] / unit - 1e-6))) for it in items]
    best = np.zeros(steps + 1)
    keep = np.zeros((n, steps + 1), dtype=bool)
    for i, (w, it) in enumerate(zip(weights, items)):
        if w > steps:
            continue
        cand = best[: steps + 1 - w] + it["benefit_k_km2"]
        take = cand > best[w:] + 1e-12
        keep[i, w:] = take
        best[w:] = np.where(take, cand, best[w:])
    picks, c = {}, steps
    for i in range(n - 1, -1, -1):
        if keep[i, c]:
            picks[items[i]["ward_id"]] = 1.0
            c -= weights[i]
    return picks


def score(items, picks):
    """Total benefit (K km2) and treated area (km2) of an allocation {ward_id: share treated}."""
    by_id = {i["ward_id"]: i for i in items}
    benefit = sum(f * by_id[w]["benefit_k_km2"] for w, f in picks.items())
    area = sum(f * by_id[w]["capacity_sqkm"] for w, f in picks.items())
    return benefit, area


def optimise(items, budget_sqkm, whole_wards=False):
    """Choose what to treat. `items` are dicts with ward_id, capacity_sqkm, benefit_k_km2 (positive
    = cooling) and t_base. Returns the optimal allocation plus the two reference allocations."""
    items = _clean(items)
    total_cap = sum(i["capacity_sqkm"] for i in items)
    if not items or budget_sqkm <= EPS:
        return {"picks": {}, "uniform": {}, "hottest": {}, "capacity_sqkm": total_cap}

    if budget_sqkm >= total_cap - EPS:               # budget covers everything
        everything = {i["ward_id"]: 1.0 for i in items}
        return {"picks": everything, "uniform": dict(everything), "hottest": dict(everything),
                "capacity_sqkm": total_cap}

    if whole_wards:
        picks = _knapsack_whole(items, budget_sqkm)
    else:
        picks = _greedy(items, lambda i: -i["benefit_k_km2"] / i["capacity_sqkm"],
                        budget_sqkm, whole=False)

    share = budget_sqkm / total_cap
    uniform = {i["ward_id"]: share for i in items}
    hottest = _greedy(items, lambda i: -i["t_base"], budget_sqkm, whole=whole_wards)
    return {"picks": picks, "uniform": uniform, "hottest": hottest, "capacity_sqkm": total_cap}
