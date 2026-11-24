"""Recompute the 36 intervals for Harasim & Valdman (2013), equation (29)."""
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy"))
from src.fem1d import solve, cell_multiplier, reconstruct_flux, dual_energy

ROOT = Path(__file__).resolve().parents[2]


def run():
    rows = []
    for force in (8, 10, 16):
        b = 16 / force
        hv_slope = -math.sqrt(2 * force)  # Left endpoint derivative of (29).
        q_ref = -4 * hv_slope / force
        for n in (32, 64, 128, 256):
            center = solve(n, b)
            sigma, _ = reconstruct_flux(cell_multiplier(center), center.x, b)
            D = dual_energy(sigma, center.x, b)
            for ratio in (0.01, 0.05, 0.1):
                k = b * ratio
                E_left = solve(n, b - k).energy
                E_right = solve(n, b + k).energy
                lo, hi = (D - E_left) / k, (E_right - D) / k
                rows.append(dict(f=-force, b=b, n=n, k_over_b=ratio, k=k,
                                 D_center=D, E_left=E_left, E_right=E_right,
                                 Q_ref=q_ref, Q_lower=lo, Q_upper=hi,
                                 relative_full_width_pct=100*(hi-lo)/q_ref,
                                 lower_margin=q_ref-lo, upper_margin=hi-q_ref,
                                 contained=lo <= q_ref <= hi))
    with (ROOT / "result" / "hv2013_1d_intervals.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    run()
