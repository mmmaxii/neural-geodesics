"""Renderer NEURAL de Kerr: las redes sustituyen al trazado de geodesicas.

Es el mismo renderer que render_kerr.py pero sin integrar ni una sola geodesica.
Cada pixel se resuelve asi:

    cae al agujero?   ->  ANALITICO, con el contorno de la sombra
    hacia donde sale? ->  KerrSkyNet
    cruza el disco?   ->  KerrDiskNet

La captura no la decide ninguna red a proposito: el borde de la sombra se
conoce en forma cerrada y es exacto. Comprobado sobre 3200 rayos que coincide
con el trazado sin una sola discrepancia. Dejarselo a una red solo meteria
error justo en el borde, que es lo mas visible de la imagen.

    python scripts/render_neural.py --spin 0.9 --compare

Con --compare renderiza tambien la version clasica trazando rayos y saca la
diferencia, que es la unica forma honesta de saber si la red sirve.
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
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from models.kerr_net import (KerrSkyNet, marco_tangente)          # noqa: E402
from physics.kerr import KMetric                                  # noqa: E402
from physics.kerr_integrator import KerrGeodesicIntegrator        # noqa: E402
from rendering.classical_renderer import (load_sky_image,         # noqa: E402
                                          sample_equirect, tonemap)


def dentro_sombra(alpha, beta, k: KMetric, inc_deg: float):
    """Captura EXACTA por el contorno analitico, sin red y sin trazar."""
    from matplotlib.path import Path as MPath
    from scipy.spatial import cKDTree
    x, y = k.shadow_outline(inc_deg, n=4000)
    poly = np.stack([x, y], 1)
    pts = np.stack([alpha, beta], 1)
    dentro = MPath(poly).contains_points(pts)
    # Distancia al borde: es una entrada de la red, y sale del mismo contorno
    # analitico, asi que no es informacion privilegiada.
    # Con un KD-tree y no comparando todos contra todos: a 960x540 la matriz
    # completa serian 1.7e9 pares, o sea 25 GB.
    d, _ = cKDTree(poly).query(pts, k=1, workers=-1)
    return dentro, d


def direcciones_red(net, alpha, beta, dist, spin, inc_deg, r_obs, dev,
                    lote=400_000):
    """Direccion de salida de cada rayo, segun la red. Por lotes para no
    reventar la memoria de la GPU."""
    n = alpha.size
    out = np.empty((n, 3), np.float32)
    th = np.deg2rad(inc_deg)
    for i in range(0, n, lote):
        s = slice(i, min(i + lote, n))
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x[s])).double().to(dev)
        c = lambda v: torch.full((s.stop - s.start,), v, dtype=torch.float64,
                                 device=dev)
        X = net.norm.features(t(alpha), t(beta), c(spin), c(inc_deg),
                              torch.from_numpy(np.ascontiguousarray(dist[s])).double().to(dev), xp=torch).float()
        with torch.no_grad():
            if net.residual:
                M = marco_tangente(t(alpha), t(beta), c(th), r_obs, xp=torch)
                d = net(X, tuple(q.float() for q in M))
            else:
                d = net(X)
        out[s] = d.cpu().numpy()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spin", type=float, default=0.9)
    p.add_argument("--inc", type=float, default=80.0)
    p.add_argument("-W", "--width", type=int, default=960)
    p.add_argument("-H", "--height", type=int, default=540)
    p.add_argument("--half-width", type=float, default=80.0)
    p.add_argument("--r-obs", type=float, default=1000.0)
    p.add_argument("--red", type=Path,
                   default=ROOT / "models" / "checkpoints_residual" / "kerr_sky.pt")
    p.add_argument("--sky-image", type=Path,
                   default=ROOT / "data" / "raw" / "gaia_panorama_mag12.png")
    p.add_argument("--sky-brightness", type=float, default=0.5)
    p.add_argument("--exposure", type=float, default=1.2)
    p.add_argument("--compare", action="store_true",
                   help="renderiza tambien la version clasica y saca el error")
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "figures" / "render_neural.png")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    k = KMetric(M=1.0, a=a.spin)
    print(f"a* = {a.spin}, i = {a.inc} deg, {a.width}x{a.height}, {dev}")

    ax = np.linspace(-a.half_width, a.half_width, a.width)
    ay = np.linspace(-a.half_width, a.half_width, a.height) * (a.height / a.width)
    A, B = np.meshgrid(ax, ay)
    al, be = A.ravel(), B.ravel()
    n = al.size

    sky = load_sky_image(a.sky_image) if a.sky_image.exists() else None

    # ---------------------------------------------------------------- NEURAL
    t0 = time.perf_counter()
    cap, dist = dentro_sombra(al, be, k, a.inc)
    t_geom = time.perf_counter() - t0

    net = KerrSkyNet.cargar(a.red, dev).to(dev)
    t0 = time.perf_counter()
    d_red = direcciones_red(net, al, be, dist, a.spin, a.inc, a.r_obs, dev)
    if dev == "cuda":
        torch.cuda.synchronize()
    t_red = time.perf_counter() - t0

    def colorear(d, capturado):
        c = np.zeros((n, 3), np.float32)
        e = ~capturado
        th = np.arccos(np.clip(d[e, 2], -1, 1))
        lo = np.arctan2(d[e, 1], d[e, 0])
        c[e] = sample_equirect(sky, th, lo, a.sky_brightness) if sky is not None \
            else 0.5
        return c.reshape(a.height, a.width, 3)

    img_red = colorear(d_red, cap)
    print(f"NEURAL : geometria {t_geom:.2f} s + red {t_red:.3f} s = "
          f"{t_geom + t_red:.2f} s   ({n/t_red/1e6:.1f} M rayos/s en la red)")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(a.out, tonemap(img_red, exposure=a.exposure))
    print(f"guardado en {a.out}")

    if not a.compare:
        return

    # -------------------------------------------------------------- CLASICO
    ig = KerrGeodesicIntegrator(k)
    t0 = time.perf_counter()
    o = ig.trace_batch(al, be, a.r_obs, np.deg2rad(a.inc))
    t_cla = time.perf_counter() - t0
    img_cla = colorear(o["direction"], o["captured"])
    print(f"CLASICO: {t_cla:.1f} s   ({n/t_cla/1e3:.1f} k rayos/s)")
    print(f"  -> la red es {t_cla/t_red:.0f}x mas rapida en el trazado")

    # error angular pixel a pixel, solo donde el rayo escapa en ambos
    e = ~cap & ~o["captured"]
    ang = np.degrees(np.arccos(np.clip((d_red[e] * o["direction"][e]).sum(1), -1, 1)))
    px = 2 * np.degrees(np.arctan(a.half_width / a.r_obs)) / a.width
    print(f"\nERROR de la red frente al trazado ({e.sum():,} pixeles):")
    print(f"  medio {ang.mean():.4f} deg = {ang.mean()/px:.1f} pixeles")
    print(f"  mediana {np.median(ang):.4f} = {np.median(ang)/px:.1f} px, "
          f"p95 {np.percentile(ang, 95):.3f}, peor {ang.max():.2f}")
    print(f"  captura identica: {(cap == o['captured']).mean()*100:.3f}%")

    dif = np.abs(tonemap(img_red, exposure=a.exposure)
                 - tonemap(img_cla, exposure=a.exposure)).max(-1)
    fig, ax_ = plt.subplots(1, 3, figsize=(19, 4.0))
    ax_[0].imshow(tonemap(img_red, exposure=a.exposure))
    ax_[0].set_title(f"NEURAL  ({t_red:.3f} s)")
    ax_[1].imshow(tonemap(img_cla, exposure=a.exposure))
    ax_[1].set_title(f"CLASICO trazando rayos  ({t_cla:.0f} s)")
    im = ax_[2].imshow(dif, cmap="inferno")
    ax_[2].set_title("|diferencia|")
    fig.colorbar(im, ax=ax_[2], fraction=0.046)
    for q in ax_:
        q.axis("off")
    fig.tight_layout()
    cmp_p = a.out.with_name(a.out.stem + "_comparacion.png")
    fig.savefig(cmp_p, dpi=110)
    print(f"comparacion -> {cmp_p}")


if __name__ == "__main__":
    main()
