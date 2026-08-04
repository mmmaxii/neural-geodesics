"""Renderiza el campo de estrellas lensado por el agujero negro.

A diferencia de render_classical.py --grid (que valida el mapeo de direcciones)
o el cielo procedural (que no tiene estrellas reales), esto proyecta un
catalogo de estrellas puntuales -- Gaia real o sintetico -- a traves de la
lente ya trazada, con las imagenes multiples y la magnificacion.

Uso:
    python scripts/render_starfield.py
    python scripts/render_starfield.py --catalog data/raw/gaia_stars.npz --ra 83.8 --dec -5.4
    python scripts/render_starfield.py --synthetic --seed 1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rendering.classical_renderer import Camera, ClassicalRenderer, bloom, tonemap  # noqa: E402
from rendering.starfield import LensedStarfield  # noqa: E402


def sky_orientation(cam: Camera, ra_deg: float, dec_deg: float,
                     roll_deg: float = 0.0) -> np.ndarray:
    """Rotacion ICRS -> marco de la camara que pone (ra, dec) en el eje optico.

    Fija hacia donde en el cielo real "mira" el agujero negro: la direccion
    (ra, dec) se manda al eje n de la camara (psi = 0, el centro de la sombra).
    roll gira el resto del cielo alrededor de ese eje: no hay una orientacion
    natural, el disco no impone ninguna referencia en el cielo real.
    """
    ra, dec = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    t = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    up = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u1 = up - (up @ t) * t
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(t, u1)
    r = np.deg2rad(roll_deg)
    u1, u2 = np.cos(r) * u1 + np.sin(r) * u2, -np.sin(r) * u1 + np.cos(r) * u2

    n, e1, e2 = cam.basis()
    A = np.stack([t, u1, u2])       # filas: base ICRS con t como primer eje
    B = np.stack([n, e1, e2])       # filas: base de la camara
    return B.T @ A                  # R tal que R@t=n, R@u1=e1, R@u2=e2


def random_orientation(seed: int) -> np.ndarray:
    """Rotacion aleatoria uniforme, para previsualizar sin fijar un punto del cielo."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.diag(r))    # signo para que sea uniforme en SO(3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--r-cam", type=float, default=30.0, help="en r_s")
    p.add_argument("--inc", type=float, default=80.0, help="0 de cara, 90 de canto")
    p.add_argument("--fov", type=float, default=42.0)
    p.add_argument("-W", "--width", type=int, default=640)
    p.add_argument("-H", "--height", type=int, default=360)
    p.add_argument("--n-radial", type=int, default=1500)

    p.add_argument("--catalog", type=Path,
                    default=ROOT / "data" / "raw" / "gaia_stars.npz")
    p.add_argument("--synthetic", action="store_true",
                    help="ignora --catalog y usa un cielo sintetico")
    p.add_argument("--n-synthetic", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--ra", type=float, default=None,
                    help="ascension recta del punto del cielo al que apunta la "
                         "camara, en grados. Sin esto, orientacion aleatoria (--seed)")
    p.add_argument("--dec", type=float, default=None, help="declinacion, en grados")
    p.add_argument("--roll", type=float, default=0.0)

    p.add_argument("--n-images", type=int, default=6,
                    help="cuantas imagenes repetidas por estrella trazar")
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--psf-sigma", type=float, default=0.9)
    p.add_argument("--max-mu", type=float, default=400.0,
                    help="tope de magnificacion cerca del anillo de fotones")

    p.add_argument("--exposure", type=float, default=1.2)
    p.add_argument("--bloom", type=float, default=0.8, help="umbral; 0 lo desactiva")
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "figures" / "render_estrellas.png")
    a = p.parse_args()

    cam = Camera(r_cam=a.r_cam, inclination_deg=a.inc, fov_deg=a.fov,
                 width=a.width, height=a.height)
    rend = ClassicalRenderer(cam, disk=None, n_radial=a.n_radial)

    print(f"camara a {a.r_cam} r_s, inclinacion {a.inc} grados, {a.width}x{a.height}")
    t0 = time.perf_counter()
    rend.build_table()
    print(f"trazado: {rend.psi_tab.size} rayos integrados en "
          f"{time.perf_counter() - t0:.1f} s")

    if a.ra is not None and a.dec is not None:
        orientation = sky_orientation(cam, a.ra, a.dec, a.roll)
        print(f"orientacion: (ra, dec) = ({a.ra}, {a.dec}) grados al centro de la sombra")
    else:
        orientation = random_orientation(a.seed)
        print(f"orientacion: aleatoria (seed={a.seed})")

    t0 = time.perf_counter()
    if a.synthetic:
        sf = LensedStarfield.synthetic(rend, n=a.n_synthetic, seed=a.seed,
                                        orientation=orientation)
        print(f"catalogo: sintetico, {a.n_synthetic} estrellas")
    else:
        sf = LensedStarfield.from_file(rend, a.catalog, orientation=orientation)
        print(f"catalogo: {a.catalog.name}, {sf.catalog[0].shape[0]} estrellas")

    img = sf.render(n_images=a.n_images, gain=a.gain, psf_sigma=a.psf_sigma,
                     max_mu=a.max_mu)
    print(f"proyeccion: {time.perf_counter() - t0:.2f} s")

    if a.bloom > 0:
        img = bloom(img, threshold=a.bloom)
    out = tonemap(img, exposure=a.exposure)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(a.out, out)
    print(f"guardado en {a.out}")


if __name__ == "__main__":
    main()
