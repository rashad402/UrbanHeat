"""Tests for the budget optimiser (api/optimiser.py).

Run:  python tests/test_optimiser.py

Offline: numpy only. An optimiser that returns a plausible plan is easy to write and hard to
notice being wrong, because every plan it produces looks reasonable. So these checks do not just
confirm that the output is sane; they compare it against an independent answer. The whole-ward
solver is checked against brute force over every subset, and the partial solver against a
hand-worked case and against hundreds of random alternative allocations that must never beat it.
"""

import itertools
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api import optimiser as O  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   [{detail}]" if detail else ""))


def ward(wid, cap, benefit, t=38.0):
    return {"ward_id": wid, "capacity_sqkm": cap, "benefit_k_km2": benefit, "t_base": t}


def random_wards(n, seed):
    rng = random.Random(seed)
    # capacities are multiples of 0.1 so the discretised solver is exact, not just close
    return [ward(f"w{i}", rng.randint(1, 15) / 10, rng.uniform(0.2, 3.0), rng.uniform(34, 42))
            for i in range(n)]


def main():
    # ───────────── hand-worked partial case ─────────────
    items = [ward("A", 1.0, 5.0), ward("B", 1.0, 2.0)]
    r = O.optimise(items, 1.5)
    b, a = O.score(items, r["picks"])
    check("partial: fills the best ward, then part of the next",
          abs(r["picks"]["A"] - 1.0) < 1e-9 and abs(r["picks"]["B"] - 0.5) < 1e-9, str(r["picks"]))
    check("partial: benefit is 5 + 0.5 x 2 = 6", abs(b - 6.0) < 1e-9, f"{b:.3f}")
    check("partial: spends exactly the budget", abs(a - 1.5) < 1e-9, f"{a:.3f} km2")

    # density, not size, decides the order
    items = [ward("big", 4.0, 4.0), ward("dense", 1.0, 3.0)]
    r = O.optimise(items, 1.0)
    check("partial: picks the densest ward, not the largest", list(r["picks"]) == ["dense"])

    # ───────────── partial is never beaten ─────────────
    for seed in range(5):
        items = random_wards(20, seed)
        budget = sum(i["capacity_sqkm"] for i in items) * random.Random(seed).uniform(0.1, 0.8)
        r = O.optimise(items, budget)
        best, area = O.score(items, r["picks"])
        rng = random.Random(100 + seed)
        worst_gap = float("inf")
        for _ in range(300):
            w = [rng.random() for _ in items]
            scale = budget / sum(wi * i["capacity_sqkm"] for wi, i in zip(w, items))
            alt = {i["ward_id"]: min(1.0, wi * scale) for wi, i in zip(w, items)}
            alt_benefit, alt_area = O.score(items, alt)
            if alt_area <= budget + 1e-9:
                worst_gap = min(worst_gap, best - alt_benefit)
        check(f"partial (seed {seed}): no random allocation beats it",
              worst_gap >= -1e-9, f"smallest margin {worst_gap:.4f}")
        check(f"partial (seed {seed}): never exceeds the budget", area <= budget + 1e-9,
              f"{area:.4f} <= {budget:.4f}")

    # ───────────── whole wards vs brute force ─────────────
    for seed in range(6):
        items = random_wards(10, 40 + seed)
        budget = round(sum(i["capacity_sqkm"] for i in items) * 0.4, 1)
        r = O.optimise(items, budget, whole_wards=True)
        got, area = O.score(items, r["picks"])
        best = 0.0
        for k in range(len(items) + 1):
            for combo in itertools.combinations(items, k):
                if sum(c["capacity_sqkm"] for c in combo) <= budget + 1e-9:
                    best = max(best, sum(c["benefit_k_km2"] for c in combo))
        check(f"whole (seed {seed}): matches brute force over all 1024 subsets",
              abs(got - best) < 1e-6, f"dp {got:.4f} vs brute {best:.4f}")
        check(f"whole (seed {seed}): every treated ward is treated in full, within budget",
              all(f == 1.0 for f in r["picks"].values()) and area <= budget + 1e-9,
              f"{area:.3f} <= {budget:.3f}")

    # ───────────── the comparisons are honest ─────────────
    items = random_wards(30, 7)
    budget = sum(i["capacity_sqkm"] for i in items) * 0.25
    r = O.optimise(items, budget)
    best, _ = O.score(items, r["picks"])
    uni, uni_area = O.score(items, r["uniform"])
    hot, hot_area = O.score(items, r["hottest"])
    check("spreading evenly uses the same budget as the optimum",
          abs(uni_area - budget) < 1e-6, f"{uni_area:.4f} vs {budget:.4f}")
    check("hottest-first uses the same budget as the optimum",
          abs(hot_area - budget) < 1e-6, f"{hot_area:.4f} vs {budget:.4f}")
    check("optimum is at least as good as spreading evenly and as hottest-first",
          best >= uni - 1e-9 and best >= hot - 1e-9, f"{best:.3f} vs {uni:.3f} / {hot:.3f}")

    # ───────────── edges ─────────────
    items = [ward("A", 1.0, 2.0), ward("B", 2.0, 3.0)]
    r = O.optimise(items, 50.0)
    check("a budget above total capacity treats everything",
          r["picks"] == {"A": 1.0, "B": 1.0} and abs(r["capacity_sqkm"] - 3.0) < 1e-9)
    check("a zero budget treats nothing", O.optimise(items, 0.0)["picks"] == {})
    r = O.optimise([ward("dead", 0.0, 1.0), ward("warm", 1.0, -0.5), ward("ok", 1.0, 1.0)], 5.0)
    check("wards that cannot be treated, or that would warm, are ignored",
          list(r["picks"]) == ["ok"], str(r["picks"]))
    r = O.optimise([ward("huge", 10.0, 9.0), ward("small", 1.0, 0.5)], 2.0, whole_wards=True)
    check("whole mode skips a ward that does not fit and takes the one that does",
          list(r["picks"]) == ["small"], str(r["picks"]))
    check("no wards at all is fine", O.optimise([], 3.0)["picks"] == {})

    print("=" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
