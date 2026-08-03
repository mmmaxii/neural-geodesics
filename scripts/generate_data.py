"""Genera el dataset de trayectorias de geodésicas nulas en Schwarzschild.

Todo se guarda ADIMENSIONAL, en unidades del radio de Schwarzschild:

    beta    = b / r_s          parámetro de impacto
    u_tilde = r_s / r          coordenada radial inversa
    phi                        ángulo azimutal (ya adimensional)

La ecuación de Binet en estas variables es

    d²u~/dphi² + u~ = (3/2) u~²

que NO contiene la masa. Por eso un único dataset sirve para cualquier agujero
negro de Schwarzschild: para una masa M concreta basta con r_s = 2GM/c².

Salida: data/processed/geodesics_dataset.npz

Uso:
    python scripts/generate_data.py                      # 20k rayos
    python scripts/generate_data.py --n-rays 100000 -j 8
    python scripts/generate_data.py --n-rays 500 --out /tmp/pilot.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict, field
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

# Permite ejecutar el script sin haber hecho `pip install -e .`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from physics.schwarzschild import SMetric          # noqa: E402
from physics.integrator import GeodesicIntegrator  # noqa: E402

# Códigos de estado. Se guarda el entero, no un booleano: perder la distinción
# entre 'captured' y 'orbit' hace imposible depurar el dataset después.
STATUS_ESCAPED = 0
STATUS_CAPTURED = 1
STATUS_ORBIT = 2
STATUS_NAMES = {STATUS_ESCAPED: "escaped", STATUS_CAPTURED: "captured", STATUS_ORBIT: "orbit"}


# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    n_rays: int = 20_000
    n_phi: int = 128                # puntos por trayectoria

    # --- muestreo en beta ---
    # Las bandas "crítica" se parametrizan por s = log10(|beta/beta_c - 1|).
    # Uniforme en s <=> uniforme en delta_phi, porque delta_phi ~ -ln(beta/beta_c - 1).
    # Eso equilibra el target de regresión, que es lo que importa para entrenar.
    s_min: float = -8.0             # más cerca que esto choca con max_revolutions
    s_max: float = -1.0
    beta_max: float = 15.0          # cubre el FOV por defecto (DEFAULT_FOV_RS)
    beta_deep_min: float = 0.02

    frac_outer: float = 0.40        # beta > beta_c, cerca del crítico
    frac_inner: float = 0.20        # beta < beta_c, cerca del crítico
    frac_far: float = 0.30          # campo lejano, log-espaciado
    frac_deep: float = 0.10         # captura profunda

    # --- integración ---
    r0_over_rs: float = 500.0
    max_revolutions: float = 8.0 * np.pi
    rtol: float = 1e-9
    atol: float = 1e-12

    # --- malla en phi ---
    # Los puntos se reparten por CURVATURA de la solución, no por distancia al
    # periastro. La ODE nos da la curvatura gratis: d2u/dphi2 = -u + 1.5 u^2,
    # que se anula exactamente en u = 2/3 (la esfera de fotones, punto fijo
    # inestable). Para rayos casi críticos u(phi) tiene una meseta larguísima
    # justo ahí, y agrupar en el periastro malgastaba medio dataset en una
    # recta. El peso es |curvatura| normalizada + un suelo para no dejar huecos.
    curvature_floor: float = 0.15
    n_fine: int = 4096

    seed: int = 20260802

    def counts(self) -> dict:
        """Reparte n_rays entre bandas garantizando que la suma sea exacta."""
        fr = {
            "outer": self.frac_outer,
            "inner": self.frac_inner,
            "far": self.frac_far,
            "deep": self.frac_deep,
        }
        total = sum(fr.values())
        out = {k: int(np.floor(self.n_rays * v / total)) for k, v in fr.items()}
        out["outer"] += self.n_rays - sum(out.values())  # el resto al banco principal
        return out


BETA_CRIT = 3.0 * np.sqrt(3.0) / 2.0  # = b_crit / r_s, exacto e independiente de M


# --------------------------------------------------------------------------
# Muestreo de beta
# --------------------------------------------------------------------------
def sample_beta(cfg: Config, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (beta, band_id) ya mezclados."""
    n = cfg.counts()
    bc = BETA_CRIT
    parts, bands = [], []

    # Banda 0: exterior crítica -> la divergencia logarítmica de delta_phi
    s = rng.uniform(cfg.s_min, cfg.s_max, n["outer"])
    parts.append(bc * (1.0 + 10.0**s))
    bands.append(np.zeros(n["outer"], np.int8))

    # Banda 1: interior crítica -> la frontera vista por dentro
    s = rng.uniform(cfg.s_min, cfg.s_max, n["inner"])
    parts.append(bc * (1.0 - 10.0**s))
    bands.append(np.ones(n["inner"], np.int8))

    # Banda 2: campo lejano, log-espaciado
    t = rng.uniform(np.log10(1.1 * bc), np.log10(cfg.beta_max), n["far"])
    parts.append(10.0**t)
    bands.append(np.full(n["far"], 2, np.int8))

    # Banda 3: captura profunda
    parts.append(rng.uniform(cfg.beta_deep_min, bc, n["deep"]))
    bands.append(np.full(n["deep"], 3, np.int8))

    beta = np.concatenate(parts)
    band = np.concatenate(bands)
    perm = rng.permutation(beta.size)  # sin sesgo de orden al entrenar
    return beta[perm], band[perm]


# --------------------------------------------------------------------------
# Malla en phi
# --------------------------------------------------------------------------
def phi_grid(sol_dense, phi_end: float, r_s: float, n: int,
             floor: float, n_fine: int) -> np.ndarray:
    """Malla en [0, phi_end] con densidad proporcional a la curvatura de u(phi).

    Peso  w(phi) = |−u~ + 1.5 u~²| / max(...) + floor,  que es exactamente
    |d²u~/dphi²| según la ecuación de Binet: no hay que estimarla numéricamente.
    Los puntos salen de invertir la CDF de w, así que quedan densos en los
    hombros de la trayectoria (donde el rayo entra y sale de la zona curvada) y
    ralos en la meseta de la esfera de fotones, que es una recta.
    """
    if not np.isfinite(phi_end) or phi_end <= 0.0:
        return np.zeros(1)

    fine = np.linspace(0.0, phi_end, n_fine)
    u = r_s * sol_dense(fine)[0]
    kappa = np.abs(-u + 1.5 * u**2)
    kmax = kappa.max()
    w = (kappa / kmax + floor) if kmax > 0 else np.ones_like(kappa)

    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(fine))))
    cdf /= cdf[-1]
    # cdf es estrictamente creciente (w > 0 por el suelo), así que se invierte bien.
    return np.interp(np.linspace(0.0, 1.0, n), cdf, fine)


def find_periastron(sol_dense, phi_end: float) -> float:
    """phi del periastro: la raíz de p = du/dphi.

    Se busca la raíz en vez de usar argmax porque cerca del crítico u(phi) tiene
    una meseta plana y el argmax es numéricamente frágil; p = 0 sigue siendo un
    cruce limpio. Para rayos capturados p nunca cambia de signo (u crece
    monótonamente hasta el horizonte) y se devuelve el final: el origen queda
    entonces en el horizonte, que es igual de intrínseco.
    """
    p0 = float(sol_dense(0.0)[1])
    pe = float(sol_dense(phi_end)[1])
    if p0 * pe < 0.0:
        return float(brentq(lambda f: sol_dense(f)[1], 0.0, phi_end, xtol=1e-13))
    return phi_end


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------
_W: dict = {}


def _init_worker(cfg_dict: dict) -> None:
    cfg = Config(**cfg_dict)
    metric = SMetric()
    _W["cfg"] = cfg
    _W["metric"] = metric
    _W["integ"] = GeodesicIntegrator(metric)


def _process(beta: float) -> tuple:
    cfg: Config = _W["cfg"]
    metric: SMetric = _W["metric"]
    integ: GeodesicIntegrator = _W["integ"]
    r_s = metric.r_s

    res = integ.integrate_ray(
        b=beta * r_s,
        r0=cfg.r0_over_rs * r_s,
        max_revolutions=cfg.max_revolutions,
        rtol=cfg.rtol,
        atol=cfg.atol,
        dense_output=True,
    )

    status = {"escaped": STATUS_ESCAPED,
              "captured": STATUS_CAPTURED,
              "orbit": STATUS_ORBIT}[res["status"]]

    sol_dense = res["sol_dense"]
    phi_end = float(res["phi"][-1])

    grid = phi_grid(sol_dense, phi_end, r_s, cfg.n_phi,
                    cfg.curvature_floor, cfg.n_fine)

    # ORIGEN DE PHI EN EL PERIASTRO. Con phi=0 en r0 la trayectoria dependía de
    # dónde arrancaste a integrar, y mover la cámara obligaba a buscar una raíz
    # por píxel y por fotograma. Referida al periastro, u~(beta, phi) es
    # intrínseca: misma curva para cualquier r0, y además SIMÉTRICA en phi,
    # porque el periastro es un punto de retorno.
    phi_peri = find_periastron(sol_dense, phi_end)

    # El periastro es un punto singular de la parametrización (du/dphi = 0) y
    # el eje de simetría: se fuerza dentro de la malla sustituyendo el punto más
    # cercano, para que la red vea phi = 0 exacto en todos los rayos.
    grid[int(np.argmin(np.abs(grid - phi_peri)))] = phi_peri
    grid.sort()
    u_tilde = r_s * sol_dense(grid)[0]
    grid = grid - phi_peri

    # Un rayo capturado también aporta trayectoria válida (r0 -> horizonte):
    # es dato de entrenamiento perfectamente bueno para la red u(beta, phi).
    # Lo que NO es válido es su delta_phi, que no existe -> NaN, nunca 0.0.
    delta_phi = float(res["delta_phi"]) if status == STATUS_ESCAPED else np.nan

    return (
        np.full(grid.size, beta),
        grid,
        u_tilde,
        beta,
        status,
        phi_end,
        float(np.max(u_tilde)),   # = r_s / r_min
        delta_phi,
        phi_peri,                 # phi desde r0 hasta el periastro
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def generate(cfg: Config, out_path: Path, workers: int) -> dict:
    rng = np.random.default_rng(cfg.seed)
    beta, band = sample_beta(cfg, rng)

    t0 = time.perf_counter()
    chunk = max(1, min(256, cfg.n_rays // (workers * 4) or 1))

    if workers > 1:
        with Pool(workers, initializer=_init_worker, initargs=(asdict(cfg),)) as pool:
            results = list(_progress(pool.imap(_process, beta, chunksize=chunk), cfg.n_rays))
    else:
        _init_worker(asdict(cfg))
        results = list(_progress((_process(b) for b in beta), cfg.n_rays))

    elapsed = time.perf_counter() - t0

    beta_flat = np.concatenate([r[0] for r in results])
    phi_flat = np.concatenate([r[1] for r in results])
    u_flat = np.concatenate([r[2] for r in results])
    ray_index = np.repeat(np.arange(len(results), dtype=np.int32),
                          [r[0].size for r in results])

    ray_beta = np.array([r[3] for r in results])
    ray_status = np.array([r[4] for r in results], dtype=np.int8)
    ray_phi_end = np.array([r[5] for r in results])
    ray_u_max = np.array([r[6] for r in results])
    ray_delta_phi = np.array([r[7] for r in results])
    ray_phi_peri = np.array([r[8] for r in results])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        # --- por muestra (lo que consume la red (beta, phi) -> u_tilde) ---
        beta=beta_flat,
        phi=phi_flat,
        u_tilde=u_flat,
        ray_index=ray_index,
        # --- por rayo ---
        ray_beta=ray_beta,
        ray_band=band,
        ray_status=ray_status,
        ray_phi_end=ray_phi_end,
        ray_phi_peri=ray_phi_peri,
        ray_u_max=ray_u_max,
        ray_r_min_over_rs=1.0 / ray_u_max,
        ray_delta_phi=ray_delta_phi,
        # --- metadata (sin esto el dataset no es reproducible) ---
        beta_crit=BETA_CRIT,
        config_json=json.dumps(asdict(cfg)),
        elapsed_s=elapsed,
    )

    return {
        "rays": len(results),
        "samples": beta_flat.size,
        "elapsed_s": elapsed,
        "mb": out_path.stat().st_size / 1e6,
        "status_counts": {STATUS_NAMES[k]: int((ray_status == k).sum())
                          for k in (0, 1, 2)},
    }


def _progress(it, total: int):
    try:
        from tqdm import tqdm
        yield from tqdm(it, total=total, unit="ray", ncols=80)
    except ImportError:
        step = max(1, total // 20)
        for i, v in enumerate(it):
            if i % step == 0:
                print(f"\r  {i}/{total} rayos", end="", flush=True)
            yield v
        print(f"\r  {total}/{total} rayos")


def main() -> None:
    cfg_default = Config()
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-rays", type=int, default=cfg_default.n_rays)
    p.add_argument("--n-phi", type=int, default=cfg_default.n_phi)
    p.add_argument("--seed", type=int, default=cfg_default.seed)
    p.add_argument("-j", "--workers", type=int, default=max(1, cpu_count() - 1))
    p.add_argument("--out", type=Path,
                   default=ROOT / "data" / "processed" / "geodesics_dataset.npz")
    args = p.parse_args()

    cfg = Config(n_rays=args.n_rays, n_phi=args.n_phi, seed=args.seed)

    print(f"beta_crit = {BETA_CRIT:.6f} r_s   (independiente de M)")
    print(f"rayos={cfg.n_rays}  phi/rayo={cfg.n_phi}  workers={args.workers}")
    print(f"bandas: {cfg.counts()}\n")

    info = generate(cfg, args.out, args.workers)

    print(f"\nGuardado en {args.out}")
    print(f"  rayos     : {info['rays']}")
    print(f"  muestras  : {info['samples']}")
    print(f"  estados   : {info['status_counts']}")
    print(f"  tiempo    : {info['elapsed_s']:.1f} s")
    print(f"  tamaño    : {info['mb']:.1f} MB")


if __name__ == "__main__":
    main()
