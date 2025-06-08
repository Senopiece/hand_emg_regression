import math
import random
from typing import List, Sequence, Tuple


def print_bars(label, arr, width=150) -> None:
    """
    Visualize positions in a fixed-width 'ascii bar' of length `width`.
    """
    print(label, end=" ")
    norm = [x / arr[-1] for x in arr]
    for i in range(width + 1):
        print("|" if any(abs(i - p * width) < 0.5 for p in norm) else "-", end="")
    print()


def to_segmentation_array(segments):
    pos = 0
    arr = [0]
    for seg in segments:
        pos += len(seg.couples)
        arr.append(pos)
    return arr


def metric(arr: List[float]) -> float:
    """
    Compute the variance of the consecutive-interval lengths.
    Lower variance means the intervals are more “even.”
    """
    intervals = [b - a for a, b in zip(arr[:-1], arr[1:])]
    mean = sum(intervals) / len(intervals)
    return -sum((d - mean) ** 2 for d in intervals)


def simulated_annealing(
    base: Sequence[float | int],
    n: int,
    iterations: int = 10_000,
    init_temp: float = 1.0,
    cooling_rate: float = 0.995,
) -> Tuple[List[float], float]:
    """
    Insert n points into `base` so as to maximize `metric` on the full sorted array,
    using simulated annealing to explore the search space efficiently.
    Returns the best full array and its metric.
    """
    base = list(base)
    lo, hi = base[0], base[-1]

    # start with a random feasible insertion
    current = sorted(random.uniform(lo, hi) for _ in range(n))
    best = current.copy()
    best_metric = metric(sorted(base + current))
    current_metric = best_metric
    T = init_temp

    for _ in range(iterations):
        # pick one point to randomly move within its neighbors
        idx = random.randrange(n)
        left = base[0] if idx == 0 else current[idx - 1]
        right = base[-1] if idx == n - 1 else current[idx + 1]

        candidate = current.copy()
        candidate[idx] = random.uniform(left, right)
        candidate.sort()

        full = sorted(base + candidate)
        m_new = metric(full)
        delta = m_new - current_metric

        # accept if better, or with prob exp(Δ/T)
        if delta > 0 or math.exp(delta / T) > random.random():
            current, current_metric = candidate, m_new
            if m_new > best_metric:
                best, best_metric = candidate.copy(), m_new

        T *= cooling_rate

    return sorted(base + best), best_metric
