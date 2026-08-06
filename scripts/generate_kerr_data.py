"""Genera el dataset de geodesicas de Kerr para entrenar el modelo con espin.

Que se guarda y por que
-----------------------
En Schwarzschild la red aprende la TRAYECTORIA u(phi), porque con simetria
esferica basta beta para determinarla y el renderer la muestrea donde quiera.

En Kerr eso no compensa. El movimiento es 3D (el arrastre saca al foton de su
plano) y el renderer solo usa TRES cosas de cada rayo:

    1. si cae al agujero      -> se sabe ANALITICAMENTE (KMetric.shadow_outline)
    2. la direccion asintotica al cielo
    3. los cruces con el plano ecuatorial, donde esta el disco

Asi que se guardan directamente esas salidas, no la geodesica entera.
Reconstruir la trayectoria para luego tirar casi todo seria pagar de mas.

El punto 1 no se aprende: el borde de la sombra se conoce en forma cerrada y es
exacto. Aprenderlo solo meteria error justo en el borde, que es lo mas visible.

Muestreo
--------
Igual que en Schwarzschild, uniforme NO sirve: toda la estructura interesante
esta pegada al borde de la sombra, donde el rayo da muchas vueltas. Aqui el
borde no es un numero (beta_crit) sino una CURVA que depende de (a*, i), asi
que se mide la distancia a esa curva y se muestrea logaritmicamente en ella.

    python scripts/generate_kerr_data.py --n-config 40 --n-rayos 4000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from physics.kerr import KMetric                              # noqa: E402
from physics.kerr_integrator import KerrGeodesicIntegrator    # noqa: E402

MAX_CRUCES = 6


def contorno(k: KMetric, inc_deg: float, n: int = 4000):
    """Borde de la sombra como poligono cerrado, en (alpha, beta)."""
    al, be = k.shadow_outline(inc_deg, n=n)
    return np.stack([al, be], axis=1)


def dist_al_borde(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distancia de cada punto al poligono del borde. Positiva siempre.

    Se aproxima por la distancia al vertice mas cercano, que con 4000 vertices
    basta de sobra: el error es menor que el espaciado del propio contorno.
    """
    d = np.linalg.norm(pts[:, None, :] - poly[None, :, :], axis=2)
    return d.min(axis=1)


def muestrear_config(k: KMetric, inc_deg: float, n_rayos: int, hw: float,
                     rng: np.random.Generator):
    """(alpha, beta) para una configuracion, con densidad junto al borde.

    Tres poblaciones:
      - 40% pegadas al borde, a distancia logaritmica (dentro y fuera)
      - 40% uniformes en el disco de radio hw, para cubrir el campo lejano
      - 20% en el anillo intermedio, para que no haya un hueco entre ambas
    """
    poly = contorno(k, inc_deg)
    n_b = int(0.4 * n_rayos)
    n_u = int(0.4 * n_rayos)
    n_m = n_rayos - n_b - n_u

    # --- pegadas al borde: se coge un vertice al azar y se desplaza a lo largo
    # de la normal aproximada una distancia log-espaciada
    idx = rng.integers(0, poly.shape[0], n_b)
    p0 = poly[idx]
    centro = poly.mean(axis=0)
    nrm = p0 - centro
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    d = 10.0 ** rng.uniform(-5.0, 0.7, n_b)          # de 1e-5 a ~5 M
    signo = rng.choice([-1.0, 1.0], n_b)
    borde = p0 + nrm * (d * signo)[:, None]

    # --- uniformes en el disco (sqrt para que sea uniforme en area)
    r = hw * np.sqrt(rng.uniform(0, 1, n_u))
    t = rng.uniform(0, 2 * np.pi, n_u)
    unif = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)

    # --- anillo intermedio
    r = rng.uniform(1.0, 12.0, n_m)
    t = rng.uniform(0, 2 * np.pi, n_m)
    medio = centro + np.stack([r * np.cos(t), r * np.sin(t)], axis=1)

    pts = np.concatenate([borde, unif, medio], axis=0)
    return pts, dist_al_borde(pts, poly)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-config", type=int, default=40,
                   help="cuantos pares (espin, inclinacion)")
    p.add_argument("--n-rayos", type=int, default=4000, help="rayos por config")
    p.add_argument("--half-width", type=float, default=90.0,
                   help="alcance en alpha/beta, en M. Conviene que cubra el "
                        "radio de Einstein sqrt(4 M r_obs)")
    p.add_argument("--r-obs", type=float, default=1000.0)
    p.add_argument("--disk-out", type=float, default=20.0)
    p.add_argument("--a-max", type=float, default=0.998)
    p.add_argument("--inc-min", type=float, default=5.0)
    p.add_argument("--inc-max", type=float, default=89.0)
    p.add_argument("--seed", type=int, default=20260805)
    p.add_argument("--out", type=Path,
                   default=ROOT / "data" / "processed" / "kerr_dataset.npz")
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    # rejilla cuasi-aleatoria en (espin, inclinacion): estratificada para que no
    # queden huecos, con jitter para no aprender la rejilla
    ejes = int(np.ceil(np.sqrt(a.n_config)))
    gi, gj = np.meshgrid(np.arange(ejes), np.arange(ejes))
    u1 = (gi.ravel() + rng.uniform(0, 1, ejes**2)) / ejes
    u2 = (gj.ravel() + rng.uniform(0, 1, ejes**2)) / ejes
    orden = rng.permutation(ejes**2)[:a.n_config]
    espines = (a.a_max * u1[orden])
    incs = a.inc_min + (a.inc_max - a.inc_min) * u2[orden]

    A, B, ESP, INC, DBORDE = [], [], [], [], []
    CAP, DIR = [], []
    CR, CPHI, NCR = [], [], []

    t0 = time.perf_counter()
    for c, (sp, inc) in enumerate(zip(espines, incs), 1):
        k = KMetric(M=1.0, a=float(sp))
        ig = KerrGeodesicIntegrator(k)
        pts, dist = muestrear_config(k, float(inc), a.n_rayos, a.half_width, rng)
        out = ig.trace_batch(pts[:, 0], pts[:, 1], a.r_obs, np.deg2rad(inc),
                             disk_in=k.r_isco(), disk_out=a.disk_out,
                             max_crossings=MAX_CRUCES)
        A.append(pts[:, 0]); B.append(pts[:, 1])
        ESP.append(np.full(pts.shape[0], sp)); INC.append(np.full(pts.shape[0], inc))
        DBORDE.append(dist)
        CAP.append(out["captured"]); DIR.append(out["direction"])
        CR.append(out["cross_r"]); CPHI.append(out["cross_phi"])
        NCR.append(out["n_cross"])
        if c % 5 == 0 or c == a.n_config:
            el = time.perf_counter() - t0
            print(f"  {c}/{a.n_config} configs  ({el:.0f} s, "
                  f"{el/c*(a.n_config-c):.0f} s restantes)")

    d = dict(alpha=np.concatenate(A).astype(np.float64),
             beta=np.concatenate(B).astype(np.float64),
             spin=np.concatenate(ESP).astype(np.float64),
             inc_deg=np.concatenate(INC).astype(np.float64),
             dist_borde=np.concatenate(DBORDE).astype(np.float64),
             captured=np.concatenate(CAP),
             direction=np.concatenate(DIR).astype(np.float32),
             cross_r=np.concatenate(CR), cross_phi=np.concatenate(CPHI),
             n_cross=np.concatenate(NCR),
             config_json=json.dumps(vars(a), default=str),
             elapsed_s=time.perf_counter() - t0)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **d)
    n = d["alpha"].size
    print(f"\nGuardado en {a.out}")
    print(f"  rayos     : {n:,}  ({a.n_config} configuraciones)")
    print(f"  capturados: {d['captured'].sum():,} "
          f"({d['captured'].mean()*100:.1f}%)")
    print(f"  con cruces: {(d['n_cross'] > 0).sum():,}  "
          f"(media {d['n_cross'][d['n_cross']>0].mean():.2f} cruces)")
    print(f"  espin     : [{espines.min():.3f}, {espines.max():.3f}]")
    print(f"  inclinacion: [{incs.min():.1f}, {incs.max():.1f}] grados")
    print(f"  tiempo    : {d['elapsed_s']:.1f} s")
    print(f"  tamaño    : {a.out.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
