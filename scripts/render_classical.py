"""Renderiza el agujero negro con el metodo clasico (RK45).

Uso:
    python scripts/render_classical.py
    python scripts/render_classical.py --inc 60 --r-cam 15 -W 960 -H 540
    python scripts/render_classical.py --no-disk --grid   # validar el fondo
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

from rendering.classical_renderer import (      # noqa: E402
    Camera, ClassicalRenderer, DiskParams, bloom, load_sky_image, tonemap)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--r-cam", type=float, default=30.0, help="en r_s")
    p.add_argument("--inc", type=float, default=80.0, help="0 de cara, 90 de canto")
    p.add_argument("--fov", type=float, default=42.0)
    p.add_argument("-W", "--width", type=int, default=640)
    p.add_argument("-H", "--height", type=int, default=360)
    p.add_argument("--no-disk", action="store_true")
    p.add_argument("--no-sky", action="store_true")
    p.add_argument("--grid", action="store_true", help="rejilla en vez de estrellas")
    p.add_argument("--sky-image", type=Path, default=None,
                   help="panorama equirectangular 2:1 del cielo")
    p.add_argument("--sky-brightness", type=float, default=1.0)
    p.add_argument("--x-out", type=float, default=14.0)
    p.add_argument("--exposure", type=float, default=1.2)
    p.add_argument("--bloom", type=float, default=0.8, help="umbral; 0 lo desactiva")
    p.add_argument("--turbulence", type=float, default=0.85)
    p.add_argument("--ss", type=int, default=1, help="supersampling (2 = 4x pixeles)")
    p.add_argument("--n-radial", type=int, default=1500,
                   help="rayos en la tabla. Bajo = bandas concentricas en el disco, "
                        "porque la trayectoria se busca por vecino mas cercano en beta")
    p.add_argument("--n-phi-tab", type=int, default=3072,
                   help="muestras de phi por rayo. Bajo = borde de la sombra punteado")
    p.add_argument("--n-images", type=int, default=5,
                   help="cuantas veces se deja que el rayo cruce el disco")
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "figures" / "render_clasico.png")
    a = p.parse_args()

    cam = Camera(r_cam=a.r_cam, inclination_deg=a.inc, fov_deg=a.fov,
                 width=a.width * a.ss, height=a.height * a.ss)
    disk = None if a.no_disk else DiskParams(x_out=a.x_out,
                                             turbulence=a.turbulence,
                                             n_images=a.n_images)
    rend = ClassicalRenderer(cam, disk=disk, n_radial=a.n_radial,
                             n_phi_tab=a.n_phi_tab)

    print(f"camara a {a.r_cam} r_s, inclinacion {a.inc} grados, {a.width}x{a.height}")
    print(f"sombra: radio angular exacto {np.rad2deg(cam.shadow_angle()):.4f} grados")

    t0 = time.perf_counter()
    rend.build_table()
    t_trace = time.perf_counter() - t0
    print(f"trazado: {rend.psi_tab.size} rayos integrados en {t_trace:.1f} s")

    t0 = time.perf_counter()
    tex = None
    if a.sky_image is not None:
        tex = load_sky_image(a.sky_image)
        print(f"cielo: {a.sky_image.name} {tex.shape[1]}x{tex.shape[0]}"
              f"  (relacion {tex.shape[1]/tex.shape[0]:.2f}, deberia ser 2.00)")
    mode = "grid" if a.grid else ("image" if tex is not None else "stars")
    img = rend.render(sky=not a.no_sky, sky_mode=mode, sky_tex=tex,
                      sky_brightness=a.sky_brightness)
    t_shade = time.perf_counter() - t0
    px = a.width * a.height
    print(f"sombreado: {px} pixeles en {t_shade:.2f} s")
    print(f"total: {t_trace + t_shade:.1f} s por fotograma")
    print(f"  (sin la simetria radial serian {px} integraciones, "
          f"~{px * t_trace / rend.psi_tab.size / 60:.0f} min)")

    if a.bloom > 0:
        img = bloom(img, threshold=a.bloom * a.ss)
    out = tonemap(img, exposure=a.exposure)
    if a.ss > 1:                      # promediar bloques de ss x ss
        s = a.ss
        out = out.reshape(a.height, s, a.width, s, 3).mean(axis=(1, 3))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(a.out, out)
    print(f"guardado en {a.out}")


if __name__ == "__main__":
    main()
