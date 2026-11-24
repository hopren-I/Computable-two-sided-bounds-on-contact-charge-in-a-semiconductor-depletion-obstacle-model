"""One-dimensional experiments for the depletion complementarity model.

M1  mesh convergence, primal-dual gap and effectivity
M2  interface interval, charge intervals, capacitance intervals
S2  partial/full depletion branches and the transition at b = ell^2/2
M5  Poisson--Boltzmann versus abrupt depletion
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analytic1d import analytic
from .fem1d import (cell_multiplier, charge_balance, charge_direct,
                    diagnostics, dual_energy, interface_interval,
                    reconstruct_flux, solve, exact_l2_errors, kkt_residuals)

from .paths import ROOT, TABLES, FIGS


plt.rcParams.update({
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 160,
    "savefig.bbox": "tight",
})


def _write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    path = TABLES / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def _l2_errors(sol, b: float, ell: float) -> tuple[float, float]:
    return exact_l2_errors(sol, b, ell)


def run_m1(ns=(16, 32, 64, 128, 256, 512), b=0.5, ell=2.0) -> list[dict]:
    rows = []
    for n in ns:
        d = diagnostics(n, b, ell)
        l2u, l2g = _l2_errors(solve(n, b, ell), b, ell)
        theta = d["gap"] / d["energy_error"] if d["energy_error"] > 1e-15 else np.nan
        rows.append({
            "n": n,
            "h": d["h"], "b": b,
            "energy_primal": d["energy"],
            "energy_exact": d["energy_exact"],
            "energy_error": d["energy_error"],
            "dual_energy": d["dual_energy"],
            "gap": d["gap"],
            "effectivity_gap_over_energy_error": theta,
            "L2_u_error": l2u,
            "L2_grad_error": l2g,
            "Q_direct": d["Q_direct"],
            "Q_balance": d["Q_balance"],
            "Q_exact": d["Q_exact"],
            "Q_direct_error": d["Q_direct"] - d["Q_exact"],
            "Q_balance_error": d["Q_balance"] - d["Q_exact"],
            "interface_lo": d["interval_lo"],
            "interface_hi": d["interval_hi"],
            "interface_width": d["interval_hi"] - d["interval_lo"],
            "contains_interface": d["contains_width"],
            **{k:d[k] for k in ("gap_identity_residual", "charge_flux_residual",
                "primal_violation", "boundary_residual", "stationarity_residual",
                "dual_violation", "complementarity_residual")},
        })
    _write_csv("m1_convergence.csv", rows)

    h = np.array([r["h"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for key, lab, marker in [
        ("gap", "primal-dual gap", "o"),
        ("energy_error", "energy error", "s"),
        ("L2_grad_error", "gradient L2 error", "^"),
        ("L2_u_error", "potential L2 error", "d"),
    ]:
        ax[0].loglog(h, [r[key] for r in rows], marker + "-", label=lab)
    ax[0].set_xlabel("mesh size h")
    ax[0].set_ylabel("error / gap")
    ax[0].legend(fontsize=7)
    ax[0].set_title("(a) 1D convergence")
    ax[1].semilogx(h, [r["effectivity_gap_over_energy_error"] for r in rows], "o-")
    ax[1].set_xlabel("mesh size h")
    ax[1].set_ylabel(r"$G/(E_h-E)$")
    ax[1].set_title("(b) gap effectivity")
    fig.tight_layout()
    fig.savefig(FIGS / "m1_convergence.png")
    fig.savefig(FIGS / "m1_convergence.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.loglog(h, np.abs([r["Q_direct_error"] for r in rows]), "o-", label="direct derivative")
    ax.loglog(h, np.abs([r["Q_balance_error"] for r in rows]), "s-", label="balance/harmonic")
    ax.set_xlabel("mesh size h")
    ax.set_ylabel("charge error")
    ax.legend()
    ax.set_title("charge recovery order")
    fig.tight_layout()
    fig.savefig(FIGS / "m1_charge_order.png")
    fig.savefig(FIGS / "m1_charge_order.pdf")
    plt.close(fig)
    return rows


def _bounds(n: int, b: float, ell: float) -> tuple[float, float]:
    d = diagnostics(n, b, ell)
    return d["energy"], d["dual_energy"]


def _cache_bounds(n: int, ell: float):
    cache: dict[float, tuple[float, float]] = {}

    def get(b: float) -> tuple[float, float]:
        key = round(float(b), 12)
        if key not in cache:
            cache[key] = _bounds(n, key, ell)
        return cache[key]

    return get


def _charge_bounds(n: int, b: float, k: float, ell: float, get) -> tuple[float, float]:
    if k <= 0 or b-k < 0:
        raise ValueError("positive secant step and nonnegative bias samples required")
    ep_b, em_b = get(b)
    ep_bm, em_bm = get(b - k)
    ep_bp, em_bp = get(b + k)
    q_minus = (em_b - ep_bm) / k
    q_plus = (ep_bp - em_b) / k
    return q_minus, q_plus


def run_m2(n=256, b=0.5, ell=2.0) -> dict:
    d = diagnostics(n, b, ell)
    ana = analytic(np.array([b]), b, ell)
    get = _cache_bounds(n, ell)

    # interface interval
    rows_if = [{
        "n": n, "b": b, "gap": d["gap"], "delta": d["delta"],
        "x_delta": d["x_delta"], "lower": d["interval_lo"],
        "upper": d["interval_hi"], "width": d["interval_hi"] - d["interval_lo"],
        "exact_interface": d["width_exact"],
        "contains": d["contains_width"],
        "width_over_delta_sqrt": (d["interval_hi"] - d["interval_lo"]) / np.sqrt(d["delta"]),
    }]
    _write_csv("m2_interface.csv", rows_if)

    # charge secants
    hs = [0.0001, 0.00025, 0.0005, 0.001, 0.002, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1]
    rows_ch = []
    for hstep in hs:
        qm, qp = _charge_bounds(n, b, hstep, ell, get)
        rows_ch.append({
            "b": b, "h": hstep, "Q_minus": qm, "Q_plus": qp,
            "width": qp - qm, "Q_exact": ana["Q"],
            "contains": bool(qm <= ana["Q"] <= qp),
            "width_over_h": (qp - qm) / hstep,
        })
    _write_csv("m2_charge_intervals.csv", rows_ch)

    # capacitance pairs (h, k)
    pairs = [(0.025, 0.025), (0.05, 0.05), (0.1, 0.1),
             (0.05, 0.005), (0.1, 0.005)]
    rows_cap = []
    for hstep, kstep in pairs:
        qm_ph, qp_ph = _charge_bounds(n, b + hstep, kstep, ell, get)
        qm_mh, qp_mh = _charge_bounds(n, b - hstep, kstep, ell, get)
        c_minus = (qm_ph - qp_mh) / (2 * hstep)
        c_plus = (qp_ph - qm_mh) / (2 * hstep)
        # analytic average capacitance on [b-h, b+h]
        c_avg = (analytic([0.],b+hstep,ell)["Q"]-analytic([0.],b-hstep,ell)["Q"])/(2*hstep)
        if b-hstep <= 0 or b-hstep < ell**2/2 <= b+hstep:
            raise ValueError("point capacitance remainder requires a smooth branch interval")
        M = 3.0 * (2.0 * (b - hstep)) ** (-2.5) if b+hstep < ell**2/2 else 0.0
        comp = M * hstep**2 / 6.0
        rows_cap.append({
            "h": hstep, "k": kstep,
            "C_minus_average": c_minus, "C_plus_average": c_plus,
            "average_width": c_plus - c_minus,
            "C_average_exact": c_avg,
            "contains_average": bool(c_minus <= c_avg <= c_plus),
            "C_minus_point": c_minus - comp,
            "C_plus_point": c_plus + comp,
            "point_width": (c_plus + comp) - (c_minus - comp),
            "C_point_exact": ana["C"],
            "contains_point": bool(c_minus - comp <= ana["C"] <= c_plus + comp),
            "M": M, "compensation": comp,
        })
    _write_csv("m2_capacitance_intervals.csv", rows_cap)

    # plots
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ax[0].loglog([r["h"] for r in rows_ch], [r["width"] for r in rows_ch], "o-")
    ax[0].set_xlabel("secant step h")
    ax[0].set_ylabel("charge interval width")
    ax[0].set_title("(a) charge interval")
    for i, r in enumerate(rows_cap):
        ax[1].plot([i], [r["average_width"]], "o", color="C0", label="average" if i==0 else None)
        ax[1].plot([i], [r["point_width"]], "s", color="C1", mfc="none", label="pointwise" if i==0 else None)
    ax[1].set_xticks(range(len(rows_cap)))
    ax[1].set_xticklabels([f"{r['h']}/{r['k']}" for r in rows_cap], fontsize=7)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("h / k")
    ax[1].set_ylabel("capacitance interval width")
    ax[1].set_title("(b) capacitance interval")
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGS / "m2_intervals.png")
    fig.savefig(FIGS / "m2_intervals.pdf")
    plt.close(fig)

    # interface figure: potential zoom
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(d["x"], d["u"], label="P1", lw=1.2)
    ax.plot(d["x"], d["u_exact"], "--", label="analytic", lw=1.2)
    ax.axvline(d["width_exact"], color="k", lw=0.6, ls=":")
    ax.axvspan(d["interval_lo"], d["interval_hi"], color="C2", alpha=0.15,
               label="interface interval")
    ax.set_xlim(0.6, 1.4)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend(fontsize=7)
    ax.set_title("interface inclusion")
    fig.tight_layout()
    fig.savefig(FIGS / "m2_interface.png")
    fig.savefig(FIGS / "m2_interface.pdf")
    plt.close(fig)
    return {"interface": rows_if, "charge": rows_ch, "capacitance": rows_cap}


def run_s2(bs=None, n=256, ell=2.0) -> list[dict]:
    if bs is None:
        bs = np.concatenate([np.linspace(0.1, 1.9, 25),
                             np.linspace(1.95, 2.05, 5),
                             np.linspace(2.1, 3.0, 15)])
    rows = []
    for b in bs:
        b = float(b)
        d = diagnostics(n, b, ell)
        ana = analytic(np.array([b]), b, ell)
        rows.append({
            "b": b, "branch": ana["branch"],
            "Q_exact": ana["Q"], "Q_direct": d["Q_direct"],
            "Q_balance": d["Q_balance"],
            "Q_direct_error": d["Q_direct"] - ana["Q"],
            "Q_balance_error": d["Q_balance"] - ana["Q"],
            "C_exact": ana["C"], "width_exact": ana["width"],
            "u_right_derivative": (d["u"][-1] - d["u"][-2]) / d["h"],
            "energy_exact": ana["energy"], "energy": d["energy"],
            "right_derivative_exact": min(0., ell/2-b/ell),
            "right_derivative_flux": d["sigma"][-1],
        })
    # numerical capacitance from the balance charge
    qb = np.array([r["Q_balance"] for r in rows])
    bb = np.array([r["b"] for r in rows])
    cnum = np.gradient(qb, bb)
    for r, c in zip(rows, cnum):
        r["C_numeric"] = float(c)
    _write_csv("s2_branches.csv", rows)

    b = np.array([r["b"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ax[0].plot(b, [r["Q_exact"] for r in rows], "-", label="analytic Q")
    ax[0].plot(b, [r["Q_balance"] for r in rows], "o", ms=3, label="Q balance")
    ax[0].axvline(ell**2 / 2, color="k", ls=":", lw=0.8)
    ax[0].set_xlabel("b")
    ax[0].set_ylabel("Q")
    ax[0].legend(fontsize=7)
    ax[0].set_title("(a) charge across branches")
    ax[1].plot(b, [r["C_exact"] for r in rows], "-", label="analytic C")
    ax[1].plot(b, [r["C_numeric"] for r in rows], "o", ms=3,
               label="finite-difference C")
    ax[1].axvline(ell**2 / 2, color="k", ls=":", lw=0.8)
    ax[1].set_ylim(0, 2.5)
    ax[1].set_xlabel("b")
    ax[1].set_ylabel("C")
    ax[1].legend(fontsize=7)
    ax[1].set_title("(b) capacitance across branches")
    fig.tight_layout()
    fig.savefig(FIGS / "s2_branches.png")
    fig.savefig(FIGS / "s2_branches.pdf")
    plt.close(fig)
    return rows


def run_m5(b=0.5, ell=2.0) -> list[dict]:
    from scipy.integrate import solve_bvp

    if b <= 0 or ell <= 0:
        raise ValueError("PB comparison requires b > 0 and ell > 0")
    qdep = analytic([0.], b, ell)["Q"]
    x0 = np.linspace(0, ell, 400)
    u0 = analytic(x0, b, ell)["u"]
    y = np.vstack([u0, np.gradient(u0, x0)])
    x = x0
    rows = []
    for th in (0.1, 0.05, 0.025, 0.01, 0.005):
        def ode(xx, yy):
            return np.vstack([yy[1], -np.expm1(np.clip(-yy[0]/th, -700, 700))])

        def bc(ya, yb):
            return np.array([ya[0] - b, yb[0] - 0.0])

        sol = solve_bvp(ode, bc, x, y, tol=1e-8, max_nodes=50000)
        if sol.status != 0:
            raise RuntimeError(sol.message)
        qpb = float(-sol.y[1][0])
        rows.append({
            "theta": th,
            "Q_poisson_boltzmann": qpb,
            "b": b, "ell": ell, "Q_depletion": qdep,
            "relative_difference_percent": 100.0 * abs(qpb - qdep)/qdep,
            "right_derivative": float(sol.y[1][-1]),
            "min_u": float(sol.y[0].min()),
            "bvp_residual": float(np.max(sol.rms_residuals)),
            "first_integral_residual": float(qpb**2-sol.y[1][-1]**2-2*(b+th*np.expm1(-b/th))),
        })
        x = np.linspace(0, ell, max(400, len(sol.x)))
        y = np.vstack([np.interp(x, sol.x, sol.y[0]),
                       np.interp(x, sol.x, sol.y[1])])
    _write_csv("m5_poisson_boltzmann.csv", rows)

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    th = np.array([r["theta"] for r in rows])
    err = np.array([r["relative_difference_percent"] for r in rows])
    ax.loglog(th, err, "o-")
    ax.set_xlabel(r"$\vartheta=V_T/V_0$")
    ax.set_ylabel("charge difference (%)")
    ax.set_title("abrupt depletion vs Poisson--Boltzmann")
    fig.tight_layout()
    fig.savefig(FIGS / "m5_poisson_boltzmann.png")
    fig.savefig(FIGS / "m5_poisson_boltzmann.pdf")
    plt.close(fig)
    return rows


def run_validation_1d():
    rows = []
    for n in (16,32,64,128,256):
        for b in (0.,0.00390625,0.05,0.37,0.5,1.3,1.99,2.,2.01,2.5):
            d=diagnostics(n,b)
            rows.append({k:v for k,v in d.items() if np.isscalar(v)})
    _write_csv("validation_1d.csv",rows)
    parameter = []
    for n in (64,128,256):
        get = _cache_bounds(n,2.)
        for b in (.05,.37,.5,1.3,1.99,2.01,2.5):
            h,k = min(.01,b/4), min(.002,b/8)
            qm,qp = _charge_bounds(n,b,k,2.,get)
            qml,qpl = _charge_bounds(n,b-h,k,2.,get)
            qmr,qpr = _charge_bounds(n,b+h,k,2.,get)
            cm,cp = (qmr-qpl)/(2*h),(qpr-qml)/(2*h)
            exact=(analytic([0],b+h)["Q"]-analytic([0],b-h)["Q"])/(2*h)
            smooth = b-h>0 and not (b-h<2<=b+h)
            M=3*(2*(b-h))**(-2.5) if b+h<2 else 0.
            compensation = M*h*h/6 if smooth else np.nan
            parameter.append(dict(n=n,b=b,h=h,k=k,Q_minus=qm,Q_plus=qp,
                Q_exact=analytic([0],b)["Q"],C_average_minus=cm,C_average_plus=cp,
                C_average_exact=exact,point_applicable=smooth,
                C_point_minus=cm-compensation,C_point_plus=cp+compensation,
                C_point_exact=analytic([0],b)["C"]))
    _write_csv("m2_parameter_sweep.csv",parameter)
    # A genuinely nonmonotone nonnegative P1 candidate with the correct trace.
    from dataclasses import replace
    base=solve(256,.5)
    v=base.u + .01*np.maximum(1-np.abs(base.x-1.5)/.15,0)
    candidate=replace(base,u=v)
    lam=cell_multiplier(base); sig,_=reconstruct_flux(lam,base.x,.5)
    gap=candidate.energy-dual_energy(sig,base.x,.5)
    delta,xd,lo,hi=interface_interval(candidate,gap)
    nonmono=[dict(n=256,b=.5,gap=gap,delta=delta,lower=lo,upper=hi,
                 exact_interface=1.,positive_increments=int(np.sum(np.diff(v)>1e-10)),
                 min_u=float(v.min()),boundary_residual=float(max(abs(v[0]-.5),abs(v[-1]))))]
    _write_csv("m2_nonmonotone.csv",nonmono)
    fig,ax=plt.subplots(figsize=(6,3))
    ax.plot(base.x,base.u,label="obstacle solution")
    ax.plot(base.x,v,label="nonmonotone feasible candidate")
    ax.axvspan(lo,hi,alpha=.15,color="C2",label="interface interval")
    ax.axvline(1,color="k",ls=":",lw=.8);ax.set(xlabel="x",ylabel="u")
    ax.legend(fontsize=8);fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(FIGS/f"m2_nonmonotone.{ext}")
    plt.close(fig)
    return dict(cases=len(rows),parameter_cases=len(parameter),nonmonotone=nonmono)


def main() -> dict:
    out = {}
    out["M1"] = run_m1()
    out["M2"] = run_m2()
    out["S2"] = run_s2()
    out["M5"] = run_m5()
    out["validation"] = run_validation_1d()
    summary = {
        "M1_last": out["M1"][-1],
        "M2_charge": out["M2"]["charge"],
        "M2_capacitance": out["M2"]["capacitance"],
        "S2_transition": [r for r in out["S2"]
                           if abs(r["b"] - 2.0) < 0.06],
        "M5": out["M5"],
    }
    (TABLES / "summary_1d.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    main()
