"""Etapa 4: valida el motor de Mino punto a punto contra trace_batch.

Complementa a render_kerr_mino.py --compare, que mide el error de la IMAGEN
(direccion de escape y sombra). Aqui se miden los CRUCES CON EL DISCO -- radio
y angulo azimutal de cada uno -- que son lo que decide donde cae la emision y
no se ven bien en una diferencia de imagenes.

Usa src/rendering/kerr_mino_engine.py, no una copia de su logica: la primera
version de este script duplicaba el pipeline y se desincronizo en cuanto el
motor paso a calcular Phi_r y G_phi de forma exacta. Un validador que no
comparte codigo con lo validado acaba validando otra cosa.

    python scripts/validate_kerr_mino_net.py
    python scripts/validate_kerr_mino_net.py --n 40000 --mu-exacto
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import rendering.kerr_mino_engine as eng                       # noqa: E402
from physics.kerr import KMetric                               # noqa: E402
from physics.kerr_integrator import KerrGeodesicIntegrator     # noqa: E402
from models.kerr_mino_net import KerrRNet, KerrThetaNet        # noqa: E402


def _cruces_por_eventos(ig, alpha, beta, r_obs, theta_obs, r_esc_factor=1.05):
    """(r, phi) de cada cruce del ecuador, por deteccion de EVENTOS.

    solve_ivp localiza theta = pi/2 con busqueda de raiz sobre la solucion
    densa, no interpolando linealmente dentro del paso como hace trace_batch.
    Es ~1000x mas preciso y por eso es el oraculo bueno para esta medida; a
    cambio cuesta una integracion por rayo, asi que solo se usa en submuestra.
    """
    from scipy.integrate import solve_ivp
    y0, E, L = ig.initial_from_celestial(alpha, beta, r_obs, theta_obs)
    r_cap = ig.k.r_horizon * 1.0001
    hit = lambda l, y, *a: y[0] - r_cap
    hit.terminal = True
    hit.direction = -1
    out = lambda l, y, *a: y[0] - r_esc_factor * r_obs
    out.terminal = True
    out.direction = 1
    eq = lambda l, y, *a: y[1] - 0.5 * np.pi
    eq.terminal = False
    sol = solve_ivp(ig._rhs, (0.0, 1e5), y0, args=(E, L), events=[hit, out, eq],
                    rtol=1e-12, atol=1e-14, method="DOP853", dense_output=True)
    return [(float(sol.sol(t)[0]), float(sol.sol(t)[2])) for t in sol.t_events[2]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25_000)
    ap.add_argument("--modelos", type=Path, default=ROOT / "models/checkpoints_mino")
    ap.add_argument("--semilla", type=int, default=7)
    ap.add_argument("--r-obs", type=float, default=1000.0)
    ap.add_argument("--r-out", type=float, default=20.0)
    ap.add_argument("--n-cruces", type=int, default=6)
    ap.add_argument("--n-eventos", type=int, default=60,
                    help="rayos por configuracion que se comparan contra la "
                         "referencia por eventos (cuesta un solve_ivp cada uno)")
    ap.add_argument("--mu-exacto", action="store_true",
                    help="cos(theta) por su forma cerrada en vez de por la red")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net_r = KerrRNet.cargar(args.modelos / "kerr_mino_r.pt", device).to(device)
    net_th = KerrThetaNet.cargar(args.modelos / "kerr_mino_theta.pt", device).to(device)
    print(f"dispositivo {device}   mu por {'FORMA CERRADA' if args.mu_exacto else 'RED'}")

    rng = np.random.default_rng(args.semilla)
    combos = [(a, i) for a in (0.0, 0.5, 0.9, 0.998) for i in (30.0, 60.0, 85.0)]
    n_por = max(1, args.n // len(combos))

    e_r, e_phi, e_dir = [], [], []
    n_desac_cap = n_desac_nc = n_total = n_comp = 0
    t_neural = t_clasico = 0.0

    for a, inc_deg in combos:
        th = np.deg2rad(inc_deg)
        k = KMetric(1.0, a)
        ig = KerrGeodesicIntegrator(k)
        r_in = float(k.r_isco())
        hw = 12.0

        # mitad uniformes, mitad pegados al borde de la sombra (lo dificil)
        al = rng.uniform(-hw, hw, n_por)
        be = rng.uniform(-hw, hw, n_por)
        try:
            px, py = k.shadow_outline(inc_deg, n=800)
            h = n_por // 2
            j = rng.integers(px.size, size=h)
            eps = 10.0 ** rng.uniform(-4.0, -0.3, h)
            s = np.where(rng.random(h) < 0.5, 1.0, -1.0)
            al[:h], be[:h] = px[j] * (1 + s * eps), py[j] * (1 + s * eps)
        except ValueError:
            pass

        t0 = time.perf_counter()
        pre = eng.precalcular(a, th, args.r_obs, al, be, n_cruces=args.n_cruces)
        # semilla="red" EXPLICITO: este script existe para validar las redes, y
        # desde la ablacion del 2026-08-08 el motor usa por defecto la semilla
        # analitica. Sin fijarlo aqui, este validador dejaria de tocar KerrRNet
        # y seguiria diciendo que la valida.
        geo = eng.evaluar(pre, net_r, net_th, device, mu_exacto=args.mu_exacto,
                          semilla="red")
        t_neural += time.perf_counter() - t0

        t0 = time.perf_counter()
        ref = ig.trace_batch(al, be, args.r_obs, th, rtol=1e-10, atol=1e-12,
                             max_steps=40_000, disk_in=r_in, disk_out=args.r_out,
                             max_crossings=args.n_cruces)
        t_clasico += time.perf_counter() - t0
        n_total += al.size

        val = ~geo["respaldo"]
        n_desac_cap += int((geo["captured"] != ref["captured"])[val].sum())

        # Cruces: la referencia NO puede ser trace_batch. trace_batch los
        # localiza por cambio de signo de theta-pi/2 e interpolacion LINEAL del
        # factor de paso, que es de primer orden: su propio error en el radio
        # del cruce es ~9e-4 M, mil veces mayor que el del motor. Medido: contra
        # una referencia por EVENTOS (root-finding sobre la solucion densa) el
        # motor coincide a 1.6e-13 M, o sea precision de maquina. Asi que los
        # cruces se comparan contra eventos, sobre una submuestra porque cuesta
        # un solve_ivp por rayo.
        idx_val = np.flatnonzero(val)
        sub = idx_val if idx_val.size <= args.n_eventos else \
            rng.choice(idx_val, args.n_eventos, replace=False)
        for i in sub:
            ref_ev = _cruces_por_eventos(ig, al[i], be[i], args.r_obs, th)
            ev = [(r, p) for r, p in ref_ev if r_in <= r <= args.r_out]
            r_m = geo["cross_r"][i, :geo["n_cross"][i]]
            p_m = geo["cross_phi"][i, :geo["n_cross"][i]]
            dentro = (r_m >= r_in) & (r_m <= args.r_out)
            r_m, p_m = r_m[dentro], p_m[dentro]
            if r_m.size != len(ev):
                n_desac_nc += 1
            n_comp += 1
            for j in range(min(r_m.size, len(ev))):
                e_r.append(abs(r_m[j] - ev[j][0]))
                e_phi.append(abs((p_m[j] - ev[j][1] + np.pi)
                                 % (2 * np.pi) - np.pi))

        sel = val & ~geo["captured"] & ~ref["captured"]
        if sel.any():
            cos = np.clip((geo["direccion"][sel] * ref["direction"][sel]).sum(1),
                          -1, 1)
            e_dir.extend(np.arccos(cos))

    px_rad = 2.0 * np.arctan(12.0 / args.r_obs) / 160
    print(f"\n{n_total:,} rayos en {len(combos)} configuraciones (a, inclinacion)")
    print(f"\nclasificacion de la sombra: {n_desac_cap} en desacuerdo "
          f"({100*n_desac_cap/n_total:.4f}%)")
    print(f"numero de cruces con el disco: {n_desac_nc} de {n_comp} rayos en "
          f"desacuerdo ({100*n_desac_nc/max(n_comp,1):.4f}%)")

    if e_dir:
        e_dir = np.array(e_dir)
        print(f"\ndireccion de escape ({e_dir.size:,} rayos)  [objetivo < 2e-4 rad]")
        print(f"  mediana {np.median(e_dir):.3e} rad ({np.median(e_dir)/px_rad:.4f} px)"
              f"   p99 {np.percentile(e_dir,99):.3e}   peor {e_dir.max():.3e}")
    if e_r:
        e_r, e_phi = np.array(e_r), np.array(e_phi)
        print(f"\ncruces con el disco ({e_r.size:,} emparejados)")
        print(f"  |dr|   mediana {np.median(e_r):.3e} M    p99 {np.percentile(e_r,99):.3e} M"
              f"   [objetivo < 1e-3 M]")
        print(f"  |dphi| mediana {np.median(e_phi):.3e} rad  p99 {np.percentile(e_phi,99):.3e} rad"
              f" [objetivo < 1e-3 rad]")

    print(f"\nneural  {t_neural:6.2f} s  ({1e3*t_neural/n_total:.5f} ms/rayo)")
    print(f"clasico {t_clasico:6.2f} s  ({1e3*t_clasico/n_total:.5f} ms/rayo)")
    print(f"speedup {t_clasico/t_neural:.1f}x   [objetivo >= 4x]")


if __name__ == "__main__":
    main()
