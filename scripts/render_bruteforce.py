"""Render por FUERZA BRUTA: una integracion RK45 por pixel, sin usar simetria.

Por que existe
--------------
El renderer normal aprovecha que Schwarzschild tiene simetria esferica: la
trayectoria depende SOLO del parametro de impacto beta, asi que integra una
tabla de ~10^4 rayos y todos los pixeles la consultan. Es exacto y rapidisimo.

Este script hace lo contrario: integra un rayo propio para CADA pixel, sin
tabla y sin interpolar en beta. Sirve para dos cosas:

1. Medir el error que introduce la tabla. Con --compare renderiza las dos
   versiones y saca el mapa de diferencias. Si el error es despreciable, la
   optimizacion esta justificada; si no, hay que subir --n-radial.

2. Ser el esqueleto para KERR. Con espin el movimiento ya no es plano y la
   trayectoria no depende solo de beta, asi que la tabla 1D deja de servir y
   hay que integrar por pixel. Este es ese camino de codigo, ya montado.

Es LENTO a proposito: el coste crece con el numero de pixeles, no con la
resolucion de la tabla. Usar resoluciones bajas.

    python scripts/render_bruteforce.py --compare -W 160 -H 90
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

from physics import disk as D                                    # noqa: E402
from physics.integrator import GeodesicIntegrator                # noqa: E402
from physics.schwarzschild import SMetric                        # noqa: E402
from rendering.classical_renderer import (                       # noqa: E402
    BETA_CRIT, Camera, ClassicalRenderer, DiskParams, blackbody_rgb,
    disk_turbulence, load_sky_image, sample_equirect, star_texture, tonemap)


def render_bruteforce(cam: Camera, disk: DiskParams | None, sky_tex=None,
                      sky_brightness: float = 1.0, rtol: float = 1e-8,
                      atol: float = 1e-11, progress: bool = True) -> np.ndarray:
    """Una integracion por pixel. Sin tabla, sin interpolar en beta."""
    metric = SMetric()
    integ = GeodesicIntegrator(metric)
    rs = metric.r_s

    psi, alpha = cam.pixel_angles()
    beta = cam.impact_parameter(psi)
    inc = np.deg2rad(cam.inclination_deg)
    n_hat, e1, e2 = cam.basis()
    phi0_all = D.first_crossing_phi(alpha, inc)

    img = np.zeros(psi.shape + (3,), np.float32)
    h, w = psi.shape
    t0 = time.perf_counter()

    for iy in range(h):
        for ix in range(w):
            b = float(beta[iy, ix])
            a = float(alpha[iy, ix])

            # ---- integracion propia de ESTE pixel
            if b <= 1e-9:
                continue
            res = integ.integrate_ray(b=b * rs, r0=cam.r_cam * rs,
                                      rtol=rtol, atol=atol, dense_output=True)
            sol, phi_end = res["sol_dense"], res["phi"][-1]
            escaped = res["status"] == "escaped"

            # ---- fondo
            if escaped and sky_tex is not None:
                r_end = 1.05 * cam.r_cam
                phi_inf = phi_end + np.arcsin(np.clip(b / r_end, -1.0, 1.0))
                e_hat = np.cos(a) * e1 + np.sin(a) * e2
                d = np.cos(phi_inf) * n_hat + np.sin(phi_inf) * e_hat
                theta_s = np.arccos(np.clip(d[2], -1.0, 1.0))
                lon_s = np.arctan2(d[1], d[0])
                if isinstance(sky_tex, np.ndarray):
                    col = sample_equirect(sky_tex,
                                          np.array([theta_s]), np.array([lon_s]),
                                          sky_brightness)[0]
                else:
                    col = star_texture(np.array([theta_s]), np.array([lon_s]))[0]
                img[iy, ix] += col

            # ---- disco: misma fisica que ClassicalRenderer._add_disk
            if disk is not None:
                phi0 = float(phi0_all[iy, ix])
                for k in range(disk.n_images):
                    phi_k = phi0 + k * np.pi
                    if phi_k > phi_end:
                        break
                    u = float(sol(phi_k)[0]) * rs
                    if u <= 0:
                        continue
                    x = 1.0 / u
                    if not D.disk_hit(x, disk.x_in, disk.x_out):
                        continue
                    g = D.redshift_g(x, D.image_to_bz(b * np.sin(a), inc))
                    prof = D.temperature_profile(x, disk.x_in)
                    t_obs = max(g * disk.display_temp_K * prof, 800.0)
                    bright = max(g, 0.0) ** 4 * prof ** 4
                    if disk.turbulence > 0.0:
                        px = np.cos(phi_k) * np.sin(inc) - np.sin(phi_k) * np.cos(a) * np.cos(inc)
                        py = -np.sin(phi_k) * np.sin(a)
                        th_disk = np.arctan2(py, px)
                        bright *= float(disk_turbulence(np.array([x]),
                                                        np.array([th_disk]),
                                                        disk.turbulence)[0])
                    img[iy, ix] += blackbody_rgb(np.array([t_obs]))[0] * bright

        if progress and (iy % 10 == 0 or iy == h - 1):
            done = (iy + 1) * w
            el = time.perf_counter() - t0
            print(f"\r  fila {iy+1}/{h}  ({done} rayos, {el:.0f} s, "
                  f"{el/max(done,1)*1000:.2f} ms/rayo)", end="", flush=True)
    if progress:
        print()
    return np.clip(img, 0.0, None)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--r-cam", type=float, default=30.0)
    p.add_argument("--inc", type=float, default=85.0)
    p.add_argument("--fov", type=float, default=40.0)
    p.add_argument("-W", "--width", type=int, default=160)
    p.add_argument("-H", "--height", type=int, default=90)
    p.add_argument("--x-out", type=float, default=12.0)
    p.add_argument("--turbulence", type=float, default=0.0,
                   help="0 por defecto: para comparar conviene un disco liso")
    p.add_argument("--n-images", type=int, default=4)
    p.add_argument("--no-disk", action="store_true")
    p.add_argument("--sky-image", type=Path, default=None)
    p.add_argument("--sky-brightness", type=float, default=1.0)
    p.add_argument("--exposure", type=float, default=1.2)
    p.add_argument("--compare", action="store_true",
                   help="renderiza tambien con tabla y saca el mapa de error")
    p.add_argument("--n-radial", type=int, default=1500,
                   help="resolucion de la tabla con la que comparar")
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "figures" / "render_bruteforce.png")
    a = p.parse_args()

    cam = Camera(r_cam=a.r_cam, inclination_deg=a.inc, fov_deg=a.fov,
                 width=a.width, height=a.height)
    disk = None if a.no_disk else DiskParams(x_out=a.x_out,
                                             turbulence=a.turbulence,
                                             n_images=a.n_images)
    sky_tex = load_sky_image(a.sky_image) if a.sky_image else None

    npx = a.width * a.height
    print(f"FUERZA BRUTA: {a.width}x{a.height} = {npx} integraciones RK45")
    t0 = time.perf_counter()
    bf = render_bruteforce(cam, disk, sky_tex, a.sky_brightness)
    t_bf = time.perf_counter() - t0
    print(f"  total {t_bf:.1f} s")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(a.out, tonemap(bf, exposure=a.exposure))
    print(f"guardado en {a.out}")

    if not a.compare:
        return

    print(f"\nCON TABLA (simetria radial, n_radial={a.n_radial}):")
    t0 = time.perf_counter()
    rend = ClassicalRenderer(cam, disk=disk, n_radial=a.n_radial)
    tb = rend.render(sky=sky_tex is not None,
                     sky_mode="image" if sky_tex is not None else "stars",
                     sky_tex=sky_tex, sky_brightness=a.sky_brightness)
    t_tab = time.perf_counter() - t0
    print(f"  total {t_tab:.1f} s   ->  {t_bf/max(t_tab,1e-9):.1f}x mas rapido")

    out_tab = a.out.with_name(a.out.stem + "_tabla.png")
    plt.imsave(out_tab, tonemap(tb, exposure=a.exposure))

    # ---- diferencia, medida en la imagen ya revelada (que es lo que se ve)
    A = tonemap(bf, exposure=a.exposure)
    B = tonemap(tb, exposure=a.exposure)
    d = np.abs(A - B).max(-1)
    print("\nDIFERENCIA (sobre la imagen final, 0-1):")
    print(f"  media   {d.mean():.5f}   ({d.mean()*255:.2f} niveles de 255)")
    print(f"  p99     {np.percentile(d,99):.5f}")
    print(f"  maxima  {d.max():.5f}   ({d.max()*255:.2f} niveles)")
    print(f"  pixeles con dif > 1/255: {(d > 1/255).sum()} de {d.size} "
          f"({(d > 1/255).mean()*100:.2f}%)")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].imshow(A); ax[0].set_title(f"fuerza bruta ({t_bf:.0f} s)")
    ax[1].imshow(B); ax[1].set_title(f"tabla n={a.n_radial} ({t_tab:.1f} s)")
    im = ax[2].imshow(d, cmap="inferno"); ax[2].set_title("|diferencia|")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    for k in ax:
        k.axis("off")
    fig.tight_layout()
    cmp_path = a.out.with_name(a.out.stem + "_comparacion.png")
    fig.savefig(cmp_path, dpi=110)
    print(f"\ncomparacion -> {cmp_path}")


if __name__ == "__main__":
    main()
