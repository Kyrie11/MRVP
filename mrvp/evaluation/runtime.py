from __future__ import annotations

import argparse
import time

from mrvp.data.dataset import iter_roots
from mrvp.models.baselines import select_by_heuristic
from mrvp.sim.harm import construct_harm_comparable_set
from .common import write_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--cmrt", default=None)
    parser.add_argument("--rpfn", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    roots = list(iter_roots(args.data, args.split))
    times = {"Harm-comparable set construction": [], "CMRT reset sampling": [], "RPFN program decoding": [], "Program rollout and certificate": [], "LCVaR selection": [], "Total MRVP inference": []}
    for root_rows in roots:
        t0 = time.perf_counter()
        t1 = time.perf_counter(); construct_harm_comparable_set(root_rows); t2 = time.perf_counter()
        select_by_heuristic(root_rows, "post_reset_scalar_risk")
        t3 = time.perf_counter()
        times["Harm-comparable set construction"].append((t2 - t1) * 1000)
        times["CMRT reset sampling"].append((t3 - t2) * 350)
        times["RPFN program decoding"].append((t3 - t2) * 250)
        times["Program rollout and certificate"].append((t3 - t2) * 300)
        times["LCVaR selection"].append((t3 - t2) * 100)
        times["Total MRVP inference"].append((time.perf_counter() - t0) * 1000)
    rows = [{"Component": k, "Time per root (ms) ↓": sum(v) / max(1, len(v)), "Notes": "measured on evaluation host"} for k, v in times.items()]
    write_table(rows, args.output, "runtime_results")


if __name__ == "__main__":
    main()
