import random
from typing import List, Sequence, Tuple


def uniform_bounded_sum(
    total: int,
    bounds: Sequence[Tuple[int, int]],
    *,
    rng: random.Random | None = None,
) -> List[int]:
    """
    Draw one solution uniformly at random from the set
        { x ∈ ℤⁿ : Σ x_i = total  and  L_i ≤ x_i ≤ H_i }.

    Parameters
    ----------
    total   : int
        The required sum S.
    bounds  : iterable of (Li, Hi)
        Inclusive lower/upper bounds for each variable.
    rng     : random.Random, optional
        RNG instance for reproducibility.

    Returns
    -------
    list[int]  -- one uniformly-sampled solution.

    Raises
    ------
    ValueError  if the constraint set is empty.

    Notes
    -----
    1. Each variable is shifted  y_i = x_i − L_i  so that 0 ≤ y_i ≤ d_i,
       with d_i = H_i − L_i and Σ y_i = S' (= total − Σ L_i).
    2. Dynamic programming pre-computes  N(i, t) = #ways variables i..n−1
       can realise residual t.  During sampling, we walk left→right and
       choose y_i with probability  N(i+1, t−y_i) / N(i, t).
    """
    rng = rng or random.Random()
    n = len(bounds)
    L = [lo for lo, _ in bounds]
    D = [hi - lo for lo, hi in bounds]  # widths
    residual = total - sum(L)

    if residual < 0 or residual > sum(D):
        raise ValueError("No integer solutions with the given bounds.")

    # ── DP: N[i][t] = #ways using variables i..n−1 to sum to t ─────────────────
    R = residual
    N = [[0] * (R + 1) for _ in range(n + 1)]
    N[n][0] = 1  # base case

    # Build suffix counts with cumulative‐sum optimisation (O(n·R))
    for i in range(n - 1, -1, -1):
        d = D[i]
        # prefix sums of next row for O(1) range-sum queries
        pref = [0] * (R + 2)
        for t in range(R + 1):
            pref[t + 1] = pref[t] + N[i + 1][t]

        for t in range(R + 1):
            hi = min(d, t)
            # sum_{k=0..hi} N[i+1][t−k]  via prefix sums
            N[i][t] = pref[t + 1] - pref[t - hi]

    # ── Random walk to build a sample ──────────────────────────────────────────
    sol, rem = [], residual
    for i in range(n):
        d = D[i]
        r = rng.randrange(N[i][rem])  # 0 … N[i][rem]−1
        cum = 0
        for k in range(min(d, rem) + 1):
            cnt = N[i + 1][rem - k]
            cum += cnt
            if r < cum:  # choose y_i = k
                sol.append(k + L[i])  # un-shift
                rem -= k
                break

    # Sanity check (should always hold)
    assert rem == 0 and sum(sol) == total
    return sol


# Verify
if __name__ == "__main__":
    S = 17
    bounds = [(0, 10), (0, 8), (4, 9), (0, 6)]  # four variables

    for _ in range(5):
        res = uniform_bounded_sum(S, bounds)
        print(res, sum(res))
