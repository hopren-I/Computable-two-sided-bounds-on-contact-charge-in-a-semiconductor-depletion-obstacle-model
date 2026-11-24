"""Recompute Harasim & Valdman (2014), benchmark II, on inscribed Q1 meshes."""
import csv
from math import log, pi, sqrt
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy"))
from src.fem1d import solve_obstacle
from src.fem2d import Mesh2D, assemble_q1, cell_nodes

LOAD = -10.0
OBSTACLE = -1.0


def disk_reference():
    # Equation (25) and the radial solution in the free annulus.
    radius = brentq(lambda r: r*r*(1 - 2*log(r)) - (1 - 4*OBSTACLE/LOAD),
                    1e-10, 1.0)
    def energy_density(r):
        u = LOAD * ((1 - r*r)/4 + radius**2 * log(r)/2)
        du = -LOAD * (r - radius**2/r)/2
        return 2*pi*r*(du**2/2 - LOAD*u)
    energy = -LOAD*OBSTACLE*pi*radius**2 + quad(energy_density, radius, 1)[0]
    if abs(radius - 0.5024744) > 5e-8:
        raise ValueError("contact radius disagrees with published benchmark II")
    return radius, energy


def discrete_disk(n):
    # Use the enclosing square [-1,1]^2. Exterior and staircase-boundary
    # nodes carry the same prescribed value, equivalent to a constant extension.
    mesh = Mesh2D(n, n, 2., 2.)
    K, mass = assemble_q1(mesh)
    x, y = np.meshgrid(mesh.xs()-1, mesh.ys()-1, indexing="ij")
    nodes = cell_nodes(mesh)
    inside = (x*x + y*y <= 1).ravel()
    cells = nodes[np.all(inside[nodes], axis=1)]
    adjacent = np.bincount(cells.ravel(), minlength=mesh.n_nodes())
    fixed = np.flatnonzero(adjacent < 4)
    b = -OBSTACLE/abs(LOAD)
    w, _, _ = solve_obstacle(K, mass, fixed, np.full(len(fixed), b),
                             poisson_start=True)
    u = abs(LOAD)*w + OBSTACLE
    # u=0 throughout all excluded cells, so square and staircase integrals agree.
    energy = float(0.5*u @ (K @ u) - LOAD*(mass @ u))
    contact_area = np.sum(np.all(w[cells] <= 1e-10, axis=1))*mesh.hx**2
    return sqrt(contact_area/pi), energy


if __name__ == "__main__":
    radius, energy = disk_reference()
    rows = []
    for n in (32, 64, 128, 256):
        rh, eh = discrete_disk(n)
        rows.append((2/n, rh, radius, abs(rh-radius), eh, energy, eh-energy))
        if eh < energy - 1e-10:
            raise ValueError("primal energy below disk minimum")
    path = Path(__file__).resolve().parents[2] / "result/hv2014_disk.csv"
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("h", "R_h", "R_ref", "abs_radius_error",
                         "J_h", "J_ref", "energy_error"))
        writer.writerows(rows)
