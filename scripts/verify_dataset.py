"""Verifica un dataset generado por scripts/generate_data.py.

No entrena nada ni modifica nada: solo comprueba que los datos son físicamente
correctos antes de gastar horas entrenando sobre ellos.

Uso:
    python scripts/verify_dataset.py
    python scripts/verify_dataset.py --path /tmp/pilot.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

BETA_CRIT = 3.0 * np.sqrt(3.0) / 2.0
# Límite de deflexión fuerte (Bozza): delta_phi -> -ln(beta/beta_c - 1) + C
BOZZA_C = np.log(216.0 * (7.0 - 4.0 * np.sqrt(3.0))) - np.pi  # ~ -0.400232

_results: list[tuple[bool, str]] = []


def brentq_p(sol_dense, phi_end):
    """phi del periastro (raíz de du/dphi); phi_end si el rayo cae."""
    from scipy.optimize import brentq
    if float(sol_dense(0.0)[1]) * float(sol_dense(phi_end)[1]) < 0.0:
        return float(brentq(lambda f: sol_dense(f)[1], 0.0, phi_end, xtol=1e-13))
    return phi_end


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((ok, name))
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=Path,
                   default=ROOT / "data" / "processed" / "geodesics_dataset.npz")
    args = p.parse_args()

    d = np.load(args.path, allow_pickle=False)
    cfg = json.loads(str(d["config_json"]))
    r0 = cfg["r0_over_rs"]

    beta, phi, u = d["beta"], d["phi"], d["u_tilde"]
    rb, st, dphi = d["ray_beta"], d["ray_status"], d["ray_delta_phi"]
    rmin, phi_end = d["ray_r_min_over_rs"], d["ray_phi_end"]
    esc, cap, orb = st == 0, st == 1, st == 2

    print(f"\n{args.path.name}: {rb.size} rayos, {beta.size} muestras")
    print(f"beta_crit = {BETA_CRIT:.6f}   r0 = {r0} r_s   seed = {cfg['seed']}")

    # ---------------------------------------------------------------- 1
    print("\n1. Integridad")
    check(beta.size == phi.size == u.size, "arrays de muestra alineados")
    check(np.isfinite(beta).all() and np.isfinite(phi).all() and np.isfinite(u).all(),
          "sin NaN/inf en las trayectorias")
    check(np.isnan(dphi[~esc]).all() and np.isfinite(dphi[esc]).all(),
          "delta_phi = NaN exactamente en los no-escapados",
          f"(escaped={esc.sum()}, captured={cap.sum()}, orbit={orb.sum()})")
    check(beta.dtype == np.float64 and phi.dtype == np.float64,
          "float64 (necesario cerca del crítico)")
    check((u > 0).all() and (u <= 1.0 + 1e-9).all(),
          "0 < u_tilde <= 1 (nunca dentro del horizonte)",
          f"max u_tilde = {u.max():.6f}")

    # ---------------------------------------------------------------- 2
    print("\n2. Origen de phi en el periastro")
    idx = d["ray_index"]
    first = np.concatenate(([0], np.flatnonzero(np.diff(idx)) + 1))
    starts = np.concatenate((first, [idx.size]))
    u_max_grid = np.array([u[a:b].max() for a, b in zip(starts[:-1], starts[1:])])
    phi_at_max = np.array([phi[a:b][np.argmax(u[a:b])]
                           for a, b in zip(starts[:-1], starts[1:])])
    check(np.abs(phi_at_max[esc]).max() < 1e-6,
          "phi = 0 coincide con el máximo de u_tilde (escapados)",
          f"max |phi_peri| = {np.abs(phi_at_max[esc]).max():.2e}")
    check(np.allclose(u_max_grid, d["ray_u_max"], rtol=1e-12),
          "u_max de la malla == u_max guardado")

    # Simetría: el periastro es punto de retorno, luego u(-phi) = u(+phi).
    sym_err = []
    for a, b_ in zip(starts[:-1], starts[1:]):
        pp, uu = phi[a:b_], u[a:b_]
        neg, pos = pp < 0, pp > 0
        if neg.sum() < 5 or pos.sum() < 5:
            continue
        # Spline cúbico, no interpolación lineal: con la malla por curvatura los
        # puntos cerca del periastro son escasos y el error O(h^2) de np.interp
        # (~3e-4) enmascaraba la simetría real.
        lo = min(-pp[neg].min(), pp[pos].max())
        t = np.linspace(0.05 * lo, 0.95 * lo, 40)
        left = CubicSpline(-pp[neg][::-1], uu[neg][::-1])(t)
        right = CubicSpline(pp[pos], uu[pos])(t)
        sym_err.append(np.max(np.abs(left - right) / np.maximum(right, 1e-6)))
    check(max(sym_err) < 1e-4 if sym_err else True,
          "u_tilde(-phi) = u_tilde(+phi): trayectoria simétrica",
          f"peor err.rel = {max(sym_err):.2e} en {len(sym_err)} rayos"
          if sym_err else "")

    # Independencia de r0: la razón de ser del cambio de origen.
    from physics.schwarzschild import SMetric            # noqa: E402
    from physics.integrator import GeodesicIntegrator    # noqa: E402
    metric = SMetric()
    ig = GeodesicIntegrator(metric)
    worst_r0 = 0.0
    for beta_t in (2.7, 4.0, 9.0):
        sols = []
        for rr in (80.0, 500.0):
            res = ig.integrate_ray(b=beta_t * metric.r_s, r0=rr * metric.r_s,
                                   rtol=1e-10, atol=1e-13, dense_output=True)
            pe = res["phi"][-1]
            sols.append((res["sol_dense"], pe, brentq_p(res["sol_dense"], pe)))
        # Rango COMÚN medido desde el periastro: con r0 distinto la pierna de
        # salida tiene longitudes distintas, y comparar rangos diferentes daba
        # un falso fallo.
        span = 0.9 * min(pe - pp for _, pe, pp in sols)
        g = np.linspace(0.0, span, 60)
        cur = [metric.r_s * sd(g + pp)[0] for sd, _, pp in sols]
        worst_r0 = max(worst_r0, float(np.max(np.abs(cur[0] - cur[1]) / cur[1])))
    check(worst_r0 < 1e-6,
          "u_tilde(beta, phi) NO depende de r0 con el origen en el periastro",
          f"peor err.rel = {worst_r0:.2e}")

    # ---------------------------------------------------------------- 3
    print("\n3. Frontera de captura")
    if cap.any() and esc.any():
        hi_cap, lo_esc = rb[cap].max(), rb[esc].min()
        check(hi_cap < BETA_CRIT < lo_esc, "capturados < beta_crit < escapados",
              f"({hi_cap:.8f} < {BETA_CRIT:.8f} < {lo_esc:.8f})")
    check(bool((rmin[esc] > 1.5).all()),
          "escapados tienen r_min > 1.5 r_s (fuera de la esfera de fotones)",
          f"min = {rmin[esc].min():.6f}")

    # ---------------------------------------------------------------- 4
    print("\n4. Límite de campo débil   delta_phi = 2/beta + 15pi/(16 beta^2)")
    far = esc & (rb > 8.0)
    if far.sum() >= 5:
        series = 2.0 / rb[far] + 15.0 * np.pi / (16.0 * rb[far] ** 2)
        rel = np.abs(dphi[far] - series) / series
        b_hi = rb[far] > 12.0
        check(rel[b_hi].max() < 0.02 if b_hi.any() else rel.max() < 0.05,
              "converge a la serie post-newtoniana",
              f"err.rel max = {rel.max():.2%} (beta>12: {rel[b_hi].max():.2%})"
              if b_hi.any() else f"err.rel max = {rel.max():.2%}")
    else:
        check(True, "campo débil omitido (pocos rayos lejanos)")

    # ---------------------------------------------------------------- 5
    print(f"\n5. Límite de campo fuerte   residuo -> {BOZZA_C:.6f}")
    eps = rb / BETA_CRIT - 1.0
    near = esc & (eps > 0) & (eps < 1e-3)
    if near.sum() >= 5:
        resid = dphi[near] + np.log(eps[near])
        check(abs(np.median(resid) - BOZZA_C) < 5e-3,
              "constante de Bozza reproducida",
              f"mediana = {np.median(resid):.6f}  sigma = {resid.std():.2e}  n = {near.sum()}")
        s = -np.log(eps[near])
        slope = np.polyfit(s, dphi[near], 1)[0]
        check(abs(slope - 1.0) < 0.02, "pendiente d(delta_phi)/d(-ln eps) = 1",
              f"{slope:.5f}")
    else:
        check(True, "campo fuerte omitido (pocos rayos cerca del crítico)")

    # ---------------------------------------------------------------- 6
    print("\n6. Monotonía de delta_phi(beta)")
    o = np.argsort(rb[esc])
    dv = np.diff(dphi[esc][o])
    bad = int((dv > 1e-9).sum())
    check(bad == 0, "delta_phi estrictamente decreciente en beta",
          f"{bad} violaciones de {dv.size}")

    # ---------------------------------------------------------------- 7
    print("\n7. Residuo de la ODE sobre las trayectorias guardadas")
    print("   d2u/dphi2 + u - 1.5 u^2 = 0   (esto valida lo que consume la red)")
    worst, checked = 0.0, 0
    rng = np.random.default_rng(0)
    for i in rng.choice(rb.size, size=min(200, rb.size), replace=False):
        a, b_ = starts[i], starts[i + 1]
        pp, uu = phi[a:b_], u[a:b_]
        if pp.size < 12:
            continue
        cs = CubicSpline(pp, uu)
        m = pp[3:-3]
        res = cs(m, 2) + cs(m) - 1.5 * cs(m) ** 2
        scale = np.maximum(np.abs(cs(m)), 1e-3)
        worst = max(worst, float(np.max(np.abs(res) / scale)))
        checked += 1
    check(worst < 5e-2, "residuo relativo pequeño en 200 rayos al azar",
          f"peor = {worst:.2e} ({checked} rayos)")

    # ---------------------------------------------------------------- 8
    print("\n8. Balance del dataset")
    fc = cap.mean()
    check(0.15 < fc < 0.65, "fracción capturada razonable", f"{fc:.1%}")
    check(orb.sum() / rb.size < 0.02,
          "pocos 'orbit' (truncados por max_revolutions)",
          f"{orb.sum()} de {rb.size}")
    q = np.nanpercentile(dphi[esc], [0, 25, 50, 75, 100])
    check(q[3] > 1.0, "delta_phi bien repartido, no aplastado cerca de 0",
          f"percentiles 0/25/50/75/100 = " + "/".join(f"{v:.2f}" for v in q))
    check(phi_end.max() < cfg["max_revolutions"] - 1e-6,
          "ningún rayo topa con max_revolutions")

    # ---------------------------------------------------------------- resumen
    n_fail = sum(1 for ok, _ in _results if not ok)
    print("\n" + "=" * 62)
    if n_fail:
        print(f"{n_fail} de {len(_results)} comprobaciones FALLARON")
        for ok, name in _results:
            if not ok:
                print(f"   - {name}")
    else:
        print(f"Las {len(_results)} comprobaciones pasaron. Dataset utilizable.")
    print("=" * 62)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
