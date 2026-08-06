"""Genera los dos datasets de tiempo de Mino que entrenan Net_theta y Net_r.

Por que dos datasets separados
-------------------------------
Las ecuaciones (dr/dlambda)^2 = R(r) y (dtheta/dlambda)^2 = Theta(theta) estan
desacopladas (ver src/physics/kerr_mino.py), asi que se entrenan dos redes
independientes en vez de una que reciba la camara entera.

Dataset theta -- data/processed/kerr_mino_theta.npz
----------------------------------------------------
Por el colapso demostrado en validate_kerr_mino.py (test 6), la forma
mu(lambda)/sqrt(u_mas) depende SOLO de (x, m) con x = lambda/Lam_theta. Eso
significa que para generar los datos no hace falta ningun rayo fisico real:
basta elegir directamente los pares objetivo (u_mas, m) y fabricar un (xi, eta)
cualquiera que los reproduzca, con un espin auxiliar a_gen = 1.0 fijo que solo
actua de "engranaje" para invertir la relacion -- no es el espin del agujero,
y Net_theta nunca lo ve. La inversion es cerrada:

    Q_menos = -a_gen^2 u_mas / m          (m < 0; m=0 aparte, ver abajo)
    eta     = u_mas Q_menos
    xi^2    = (1 - u_mas)(Q_menos + a_gen^2)

(se deduce invirtiendo Q_menos = 1/2(B + sqrt(B^2+4a^2 eta)) de kerr_mino.py).

Dataset r -- data/processed/kerr_mino_r.npz
---------------------------------------------
Aqui si hace falta un rayo real: se muestrea (alpha, beta, inclinacion, a) como
DISPOSITIVO para caer en (xi, eta, a) con la densidad que interesa cerca de la
curva critica, se clasifica con kerr_mino.clasificar (exacto, sin integrar), y
se traza r(lambda) desde el anclaje con la EDO de segundo orden. La
inclinacion no se guarda en ningun sitio: es solo el mecanismo de muestreo.

Malla en la variable independiente
-----------------------------------
r y lambda son monotonos entre el anclaje y r_max en los dos ramos (escapa o
cae), asi que la malla se construye en el espacio de r -- donde la curvatura
tiene forma cerrada -- y se pasa a lambda con la propia cuadratura de Carlson,
sin ODE. La ODE (cara) se integra una sola vez por rayo, solo para leer
(u, Phi_r) en los puntos ya elegidos.

Uso:
    python scripts/generate_kerr_mino_data.py --etapa theta --n-tracks 50000
    python scripts/generate_kerr_mino_data.py --etapa r --n-tracks 160000 -j 12
    python scripts/generate_kerr_mino_data.py --etapa theta --n-tracks 500 --out /tmp/pilot.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import physics.kerr_mino as km          # noqa: E402
from physics.kerr import KMetric        # noqa: E402

STATUS_ESCAPED, STATUS_CAPTURED = 0, 1


def _progress(it, total: int):
    try:
        from tqdm import tqdm
        yield from tqdm(it, total=total, unit="rayo", ncols=80)
    except ImportError:
        step = max(1, total // 20)
        for i, x in enumerate(it):
            if i % step == 0:
                print(f"  {i}/{total}", flush=True)
            yield x


# ===========================================================================
# Dataset theta
# ===========================================================================
@dataclass(frozen=True)
class ConfigTheta:
    n_tracks: int = 50_000
    n_x: int = 64                    # puntos por track, en x en [0, 1/2]

    frac_uniforme: float = 0.60
    frac_polar: float = 0.20         # u_mas -> 1 (rayos casi polares)
    frac_critico: float = 0.20       # m_tilde -> 1 (raiz de theta casi doble)

    u_mas_min: float = 1e-3
    m_tilde_max: float = 1.0 - 1e-7  # m -> -infinito
    a_gen: float = 1.0               # espin auxiliar, no es el del agujero

    curvature_floor: float = 0.12
    n_fine: int = 2048

    seed: int = 20260805


def _muestrear_theta(cfg: ConfigTheta, rng: np.random.Generator):
    n = cfg.n_tracks
    n_u = int(round(cfg.frac_uniforme * n))
    n_p = int(round(cfg.frac_polar * n))
    n_c = n - n_u - n_p

    u_mas = np.concatenate([
        rng.uniform(cfg.u_mas_min, 1.0 - 1e-6, n_u),
        1.0 - 10.0 ** rng.uniform(-6.0, -0.3, n_p),                  # -> 1
        rng.uniform(cfg.u_mas_min, 1.0 - 1e-6, n_c),
    ])
    m_tilde = np.concatenate([
        rng.uniform(0.0, cfg.m_tilde_max, n_u),
        rng.uniform(0.0, cfg.m_tilde_max, n_p),
        1.0 - 10.0 ** rng.uniform(-7.0, -0.3, n_c),                  # -> 1
    ])
    perm = rng.permutation(n)
    return u_mas[perm], m_tilde[perm]


def _xi_eta_desde_target(u_mas: float, m: float, a_gen: float):
    """Invierte (u_mas, m) -> (xi, eta) con espin auxiliar fijo. Ver docstring."""
    if abs(m) < 1e-13:                        # m=0: Q_menos es gauge libre
        Q_menos = 1.0
    else:
        Q_menos = -(a_gen * a_gen) * u_mas / m
    eta = u_mas * Q_menos
    xi2 = (1.0 - u_mas) * (Q_menos + a_gen * a_gen)
    return float(np.sqrt(max(xi2, 0.0))), float(eta)


def _grid_theta_por_curvatura(u_mas: float, m: float, n: int,
                              floor: float, n_fine: int) -> np.ndarray:
    """Malla en x in [0, 1/2] ponderada por |d^2 mu/dx^2|, forma cerrada.

    mu''(lambda) = W'(mu)/2, y d^2mu/dx^2 = Lam_theta^2 mu''(lambda): el factor
    Lam_theta^2 es una constante para el track, asi que no afecta al peso
    normalizado. No hace falta ninguna EDO para construir la malla.
    """
    xf = np.linspace(0.0, 0.5, n_fine)
    mu = km.mu_de_x(xf, u_mas, m)
    # dW_dmu no depende de xi/eta/a por separado en esta parametrizacion de
    # generacion: se recalcula via el par (u_mas, u_menos) directamente.
    u_menos = u_mas / m if abs(m) > 1e-13 else -1e18
    dW = -4.0 * mu**3 + 2.0 * (u_mas + u_menos) * mu   # W=-(mu^2-u+)(mu^2-u-)
    kappa = np.abs(dW)
    kmax = kappa.max()
    w = (kappa / kmax + floor) if kmax > 0 else np.ones_like(kappa)
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(xf))))
    cdf /= cdf[-1]
    return np.interp(np.linspace(0.0, 1.0, n), cdf, xf)


def _proceso_theta(args):
    u_mas, m_tilde, cfg = args
    m = m_tilde / (m_tilde - 1.0) if m_tilde < 1.0 else -1e18
    xi, eta = _xi_eta_desde_target(u_mas, m, cfg.a_gen)

    grid_x = _grid_theta_por_curvatura(u_mas, m, cfg.n_x, cfg.curvature_floor,
                                       cfg.n_fine)
    grid_x[0] = 0.0
    grid_x.sort()

    y_theta = km.mu_de_x(grid_x, u_mas, m) / np.sqrt(u_mas)
    nu = 1.0 if abs(m) < 1e-13 else np.sqrt(-(cfg.a_gen**2) * u_mas / m)
    g = km.G_phi_semiperiodo(u_mas, m, nu)
    ratio_g = np.array([
        km.G_phi_exacto(np.array([x]), u_mas, m, nu)[0] / g if g > 0 else 0.0
        for x in grid_x
    ])

    return grid_x, y_theta, ratio_g, u_mas, m, g


def generate_theta(cfg: ConfigTheta, out_path: Path, workers: int) -> dict:
    rng = np.random.default_rng(cfg.seed)
    u_mas, m_tilde = _muestrear_theta(cfg, rng)
    args = [(float(u_mas[i]), float(m_tilde[i]), cfg) for i in range(cfg.n_tracks)]

    t0 = time.perf_counter()
    if workers > 1:
        with Pool(workers) as pool:
            results = list(_progress(pool.imap(_proceso_theta, args, chunksize=64),
                                     cfg.n_tracks))
    else:
        results = list(_progress((_proceso_theta(a) for a in args), cfg.n_tracks))
    elapsed = time.perf_counter() - t0

    x_flat = np.concatenate([r[0] for r in results])
    y_flat = np.concatenate([r[1] for r in results])
    g_flat = np.concatenate([r[2] for r in results])
    ray_index = np.repeat(np.arange(len(results), dtype=np.int32),
                          [r[0].size for r in results])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x=x_flat, y_theta=y_flat, ratio_g=g_flat, ray_index=ray_index,
        track_u_mas=np.array([r[3] for r in results]),
        track_m=np.array([r[4] for r in results]),
        track_g_semiperiodo=np.array([r[5] for r in results]),
        config_json=json.dumps(asdict(cfg)), elapsed_s=elapsed,
    )
    return {"tracks": len(results), "samples": x_flat.size,
           "elapsed_s": elapsed, "mb": out_path.stat().st_size / 1e6}


# ===========================================================================
# Dataset r
# ===========================================================================
@dataclass(frozen=True)
class ConfigR:
    n_tracks: int = 160_000
    n_lam: int = 96

    frac_critica_esc: float = 0.40
    frac_critica_cap: float = 0.20
    frac_medio_lejano: float = 0.25
    frac_captura_profunda: float = 0.15

    s_min: float = -8.0
    s_max: float = -0.5
    half_width_medio: float = 30.0
    a_max: float = 0.998
    frac_a_alto: float = 0.10        # masa extra en a > 0.9

    r_max: float = 2000.0
    rtol: float = 1e-11
    atol: float = 1e-13

    curvature_floor: float = 0.12
    n_fine: int = 3000

    seed: int = 20260805


def _muestrear_a(n: int, cfg: ConfigR, rng: np.random.Generator) -> np.ndarray:
    n_alto = int(round(cfg.frac_a_alto * n))
    a = np.concatenate([rng.uniform(0.0, cfg.a_max, n - n_alto),
                        rng.uniform(0.9, cfg.a_max, n_alto)])
    rng.shuffle(a)
    return a


def _pixel_desde_contorno(a: float, rng: np.random.Generator, eps_lo: float,
                          eps_hi: float, signo: float | None = None):
    """Un pixel cerca de la curva critica, perturbado una fraccion eps."""
    inc_deg = rng.uniform(2.0, 89.0)
    k = KMetric(M=1.0, a=max(a, 1e-6))
    try:
        px, py = k.shadow_outline(inc_deg, n=500)
    except ValueError:
        inc_deg = 45.0
        px, py = k.shadow_outline(inc_deg, n=500)
    j = rng.integers(px.size)
    eps = 10.0 ** rng.uniform(np.log10(eps_lo), np.log10(eps_hi))
    s = signo if signo is not None else (1.0 if rng.random() < 0.5 else -1.0)
    return px[j] * (1.0 + s * eps), py[j] * (1.0 + s * eps), np.deg2rad(inc_deg)


def _muestrear_r(cfg: ConfigR, rng: np.random.Generator):
    """Devuelve (xi, eta, a) por track, con el reparto de bandas del plan."""
    n = cfg.n_tracks
    counts = {
        "esc": int(round(cfg.frac_critica_esc * n)),
        "cap": int(round(cfg.frac_critica_cap * n)),
        "far": int(round(cfg.frac_medio_lejano * n)),
    }
    counts["deep"] = n - sum(counts.values())

    xis, etas, a_s = [], [], []
    for _ in range(counts["esc"]):
        a = float(_muestrear_a(1, cfg, rng)[0])
        al, be, th = _pixel_desde_contorno(a, rng, 10 ** cfg.s_min, 10 ** cfg.s_max,
                                           signo=+1.0)
        xi, eta = km.xi_eta_desde_pixel(al, be, th, a)
        xis.append(xi); etas.append(eta); a_s.append(a)
    for _ in range(counts["cap"]):
        a = float(_muestrear_a(1, cfg, rng)[0])
        al, be, th = _pixel_desde_contorno(a, rng, 10 ** cfg.s_min, 10 ** cfg.s_max,
                                           signo=-1.0)
        xi, eta = km.xi_eta_desde_pixel(al, be, th, a)
        xis.append(xi); etas.append(eta); a_s.append(a)
    for _ in range(counts["far"]):
        a = float(_muestrear_a(1, cfg, rng)[0])
        hw = cfg.half_width_medio
        al = rng.uniform(-hw, hw); be = rng.uniform(-hw, hw)
        th = np.deg2rad(rng.uniform(2.0, 89.0))
        xi, eta = km.xi_eta_desde_pixel(al, be, th, a)
        xis.append(xi); etas.append(eta); a_s.append(a)
    for _ in range(counts["deep"]):
        a = float(_muestrear_a(1, cfg, rng)[0])
        # bien adentro del contorno: factor de escala chico desde el centroide
        inc_deg = rng.uniform(2.0, 89.0)
        k = KMetric(M=1.0, a=max(a, 1e-6))
        try:
            px, py = k.shadow_outline(inc_deg, n=500)
        except ValueError:
            inc_deg = 45.0
            px, py = k.shadow_outline(inc_deg, n=500)
        cx, cy = float(px.mean()), float(py.mean())
        j = rng.integers(px.size)
        f = rng.uniform(0.05, 0.6)
        al = cx + f * (px[j] - cx); be = cy + f * (py[j] - cy)
        xi, eta = km.xi_eta_desde_pixel(al, be, np.deg2rad(inc_deg), a)
        xis.append(xi); etas.append(eta); a_s.append(a)

    xi = np.array(xis); eta = np.array(etas); a = np.array(a_s)
    perm = rng.permutation(n)
    return xi[perm], eta[perm], a[perm]


def _grid_r_por_curvatura(cls: dict, xi: float, eta: float, a: float, n: int,
                          r_max: float, floor: float, n_fine: int):
    """Malla en r por curvatura de u(lambda) en forma cerrada, y su lambda.

    u''(lambda) = 2 R(r)/r^3 - R'(r)/(2r^2), sin ninguna EDO: se puede evaluar
    en una malla fina de r y pasarla a lambda con Carlson antes de decidir
    donde muestrear. Igual que generate_data.py pero con la curvatura exacta
    en vez de la de Binet.
    """
    r_ancla = cls["r_ancla"]
    R = cls["raices"]

    # concentra la malla fina cerca del anclaje (ahi la curvatura es alta) con
    # un cambio de variable log en (r - r_ancla)
    t = np.linspace(0.0, 1.0, n_fine)
    r_fine = r_ancla + (r_max - r_ancla) * t ** 3

    Rval = km.potencial_R(r_fine, xi, eta, a)
    dRval = km.dR_dr(r_fine, xi, eta, a)
    Rval = np.maximum(Rval, 0.0)
    u_pp = 2.0 * Rval / r_fine**3 - dRval / (2.0 * r_fine**2)
    kappa = np.abs(u_pp)
    kmax = kappa.max()
    w = (kappa / kmax + floor) if kmax > 0 else np.ones_like(kappa)

    lam_fine = km.integral_mino_radial(np.full(n_fine, r_ancla), r_fine, R)
    lam_fine[0] = 0.0

    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(lam_fine))))
    if cdf[-1] <= 0:
        return np.linspace(0.0, lam_fine[-1], n)
    cdf /= cdf[-1]
    return np.interp(np.linspace(0.0, 1.0, n), cdf, lam_fine)


def _proceso_r(args):
    xi, eta, a, cfg = args
    k_h = 1.0 + np.sqrt(max(1.0 - a * a, 0.0))
    xi_v, eta_v = np.array([xi]), np.array([eta])
    cls_v = km.clasificar(xi_v, eta_v, a, k_h)
    cls = {"raices": cls_v["raices"][:, 0:1], "escapa": bool(cls_v["escapa"][0]),
          "delta_gap": float(cls_v["delta_gap"][0]),
          "r_ancla": float(cls_v["r_ancla"][0]),
          "r_plateau": float(cls_v["r_plateau"][0])}

    sol = km.trazar_mino_r(xi, eta, a, cls["escapa"], cls["r_ancla"],
                           r_max=cfg.r_max, rtol=cfg.rtol, atol=cfg.atol)
    lam_end = float(sol.t[-1])
    if lam_end <= 0:
        return None

    grid_lam = _grid_r_por_curvatura(cls, xi, eta, a, cfg.n_lam, cfg.r_max,
                                     cfg.curvature_floor, cfg.n_fine)
    grid_lam[0] = 0.0
    grid_lam = np.clip(grid_lam, 0.0, lam_end)
    grid_lam.sort()

    r_g, _, Phi_g = sol.sol(grid_lam)
    u_g = 1.0 / r_g

    # invariante RELATIVO: R(r) ~ r^4, y r_max=2000 lo lleva a ~1e13, asi que
    # comparar |rp^2 - R(r)| en absoluto rechazaria buenos rayos por escala,
    # no por error. Se normaliza por el propio tamano de R en cada punto.
    R_traza = km.potencial_R(sol.y[0], xi, eta, a)
    invariante = float(np.max(np.abs(sol.y[1] ** 2 - R_traza)
                              / np.maximum(1.0, np.abs(R_traza))))
    lam_inf = float(km.integral_mino_radial_inf(np.array([cls["r_ancla"]]),
                                                 cls_v["raices"])[0])

    return (grid_lam, u_g, Phi_g, xi, eta, a, 1.0 if cls["escapa"] else -1.0,
           cls["delta_gap"], cls["r_ancla"], cls["r_plateau"], lam_inf,
           invariante, cls["escapa"])


def generate_r(cfg: ConfigR, out_path: Path, workers: int) -> dict:
    rng = np.random.default_rng(cfg.seed)
    xi, eta, a = _muestrear_r(cfg, rng)
    val = eta > km.ETA_MIN
    xi, eta, a = xi[val], eta[val], a[val]
    args = [(float(xi[i]), float(eta[i]), float(a[i]), cfg) for i in range(xi.size)]

    t0 = time.perf_counter()
    if workers > 1:
        with Pool(workers) as pool:
            results = list(_progress(pool.imap(_proceso_r, args, chunksize=32),
                                     len(args)))
    else:
        results = list(_progress((_proceso_r(x) for x in args), len(args)))
    elapsed = time.perf_counter() - t0

    UMBRAL_INVARIANTE = 1e-3    # relativo; ver comentario en _proceso_r
    n_rechazo = sum(1 for r in results if r is None)
    results = [r for r in results if r is not None]
    n_malos = sum(1 for r in results if r[11] > UMBRAL_INVARIANTE)
    results = [r for r in results if r[11] <= UMBRAL_INVARIANTE]

    lam_flat = np.concatenate([r[0] for r in results]).astype(np.float64)
    u_flat = np.concatenate([r[1] for r in results]).astype(np.float64)
    phi_flat = np.concatenate([r[2] for r in results]).astype(np.float64)
    ray_index = np.repeat(np.arange(len(results), dtype=np.int32),
                          [r[0].size for r in results])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        lam=lam_flat, u=u_flat, Phi_r=phi_flat, ray_index=ray_index,
        track_xi=np.array([r[3] for r in results]),
        track_eta=np.array([r[4] for r in results]),
        track_a=np.array([r[5] for r in results]),
        track_lado=np.array([r[6] for r in results]),
        track_delta_gap=np.array([r[7] for r in results]),
        track_r_ancla=np.array([r[8] for r in results]),
        track_r_plateau=np.array([r[9] for r in results]),
        track_lambda_inf=np.array([r[10] for r in results]),
        track_escapa=np.array([r[12] for r in results], dtype=bool),
        config_json=json.dumps(asdict(cfg)), elapsed_s=elapsed,
    )
    return {"tracks": len(results), "samples": lam_flat.size,
           "rechazados_lam0": n_rechazo, "rechazados_invariante": n_malos,
           "elapsed_s": elapsed, "mb": out_path.stat().st_size / 1e6}


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa", choices=["theta", "r", "ambos"], default="ambos")
    ap.add_argument("--n-tracks", type=int, default=None)
    ap.add_argument("-j", "--workers", type=int, default=8)
    ap.add_argument("--out-theta", type=Path,
                    default=ROOT / "data/processed/kerr_mino_theta.npz")
    ap.add_argument("--out-r", type=Path,
                    default=ROOT / "data/processed/kerr_mino_r.npz")
    args = ap.parse_args()

    if args.etapa in ("theta", "ambos"):
        cfg = ConfigTheta(n_tracks=args.n_tracks or ConfigTheta.n_tracks)
        print(f"Generando dataset theta: {cfg.n_tracks} tracks x {cfg.n_x} pts")
        info = generate_theta(cfg, args.out_theta, args.workers)
        print(f"  -> {args.out_theta}  {info}")

    if args.etapa in ("r", "ambos"):
        cfg = ConfigR(n_tracks=args.n_tracks or ConfigR.n_tracks)
        print(f"Generando dataset r: {cfg.n_tracks} tracks x {cfg.n_lam} pts")
        info = generate_r(cfg, args.out_r, args.workers)
        print(f"  -> {args.out_r}  {info}")


if __name__ == "__main__":
    main()
