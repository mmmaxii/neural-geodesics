"""Render de Kerr con el surrogate en tiempo de Mino.

Espejo de scripts/render_kerr.py: mismos flags, mismo sombreado, misma
convencion de imagen. Lo unico que cambia es COMO se obtiene la geometria de
cada rayo -- por las dos redes de tiempo de Mino mas algebra exacta, en vez de
integrando una EDO de cinco variables por pixel.

    python scripts/render_kerr_mino.py --spin 0.9 --inc 85 -W 480 -H 270
    python scripts/render_kerr_mino.py --spin 0.9 --inc 85 --compare

--compare renderiza la MISMA escena por los dos caminos y mide la diferencia en
pixeles, que es la unidad que importa: un error angular pequeno puede ser
enorme si el encuadre es cerrado.
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

import physics.disk as D                                       # noqa: E402
import rendering.kerr_mino_engine as eng                       # noqa: E402
from physics.kerr import KMetric                               # noqa: E402
from physics.kerr_integrator import KerrGeodesicIntegrator     # noqa: E402
from models.kerr_mino_net import KerrRNet, KerrThetaNet        # noqa: E402
from rendering.classical_renderer import (                     # noqa: E402
    blackbody_rgb, bloom, disk_turbulence, load_sky_image, sample_equirect,
    star_texture, tonemap)


def malla_imagen(width, height, half_width):
    """Plano imagen en coordenadas de Bardeen, igual que render_kerr.py:265-277.

    OJO con el signo de beta: la fila 0 (arriba) es beta NEGATIVO. beta entra
    como p_theta y theta crece hacia el sur, asi que beta > 0 pone la fuente al
    SUR. Invertir esto espeja el render entero de arriba abajo -- fue un bug
    real, documentado en render_kerr.py.
    """
    ax = np.linspace(-half_width, half_width, width)
    ay = np.linspace(-half_width, half_width, height) * (height / width)
    A, B = np.meshgrid(ax, ay)
    return A.ravel(), B.ravel()


def sombrear(geo, cfg, k, sky_tex):
    """Geometria -> color. Mismo bloque que render_kerr.py:140-192.

    Se copia en vez de importarse porque render_kerr.py lo tiene dentro de un
    worker de multiprocessing con estado global; el resultado tiene que ser
    identico para que --compare mida la geometria y no dos sombreados distintos.
    """
    n = geo["captured"].size
    col = np.zeros((n, 3), np.float32)
    dk = cfg.get("disk")

    # ---------- fondo
    esc = ~geo["captured"] & ~geo["respaldo"]
    if esc.any():
        d = geo["direccion"][esc]
        R = cfg.get("R_cielo")
        if R is not None:
            d = d @ R.T
        th_s = np.arccos(np.clip(d[:, 2], -1.0, 1.0))
        lo_s = np.arctan2(d[:, 1], d[:, 0])
        if sky_tex is not None:
            col[esc] = sample_equirect(sky_tex, th_s, lo_s, cfg["sky_brightness"])
        else:
            col[esc] = star_texture(th_s, lo_s)

    # ---------- disco (opticamente fino: los cruces SE SUMAN)
    if dk:
        cr, cp, nc = geo["cross_r"], geo["cross_phi"], geo["n_cross"]
        xi = geo["xi"]
        for j in range(cr.shape[1]):
            m = (nc > j) & (cr[:, j] >= dk["r_in"]) & (cr[:, j] <= dk["r_out"])
            if not m.any():
                continue
            r_e, ph_e = cr[m, j], cp[m, j]
            g = k.redshift_g(r_e, xi[m], prograde=dk["prograde"])
            prof = D.temperature_profile(r_e, dk["r_in"])
            if dk["norm_r"]:
                prof = prof / max(D.temperature_profile(dk["norm_r"], dk["r_in"]),
                                  1e-9)
            t_obs = np.clip(g * dk["temp_K"] * prof, 800.0, None)
            bright = np.clip(g, 0.0, None) ** 4 * prof ** 4
            if dk["turbulence"] > 0.0:
                bright = bright * disk_turbulence(r_e, ph_e, dk["turbulence"])
            col[np.flatnonzero(m)] += (blackbody_rgb(t_obs)
                                       * bright[:, None]).astype(np.float32)
    return col


def geometria_clasica(ig, al, be, cfg, n_cruces):
    """La misma geometria por el trazador de siempre, para --compare."""
    dk = cfg.get("disk")
    out = ig.trace_batch(al, be, cfg["r_obs"], cfg["theta_obs"],
                         rtol=1e-8, atol=1e-10,
                         disk_in=(dk["r_in"] if dk else None),
                         disk_out=(dk["r_out"] if dk else None),
                         max_crossings=n_cruces)
    n = al.size
    cr = np.zeros((n, n_cruces)); cp = np.zeros((n, n_cruces))
    if dk:
        m = min(n_cruces, out["cross_r"].shape[1])
        cr[:, :m] = out["cross_r"][:, :m]
        cp[:, :m] = out["cross_phi"][:, :m]
    dirs = np.zeros((n, 3))
    dirs[~out["captured"]] = out["direction"][~out["captured"]]
    return {"cross_r": cr, "cross_phi": cp,
            "n_cross": out.get("n_cross", np.zeros(n, np.int32)).astype(np.int32),
            "direccion": dirs, "captured": out["captured"],
            "respaldo": np.zeros(n, bool), "xi": out["L"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spin", type=float, default=0.9)
    p.add_argument("-W", "--width", type=int, default=480)
    p.add_argument("-H", "--height", type=int, default=270)
    p.add_argument("--half-width", type=float, default=11.0)
    p.add_argument("--r-obs", type=float, default=1000.0)
    p.add_argument("--inc", type=float, default=85.0)
    p.add_argument("--no-disk", action="store_true")
    p.add_argument("--r-out", type=float, default=20.0)
    p.add_argument("--n-images", type=int, default=6)
    p.add_argument("--turbulence", type=float, default=0.45)
    p.add_argument("--temp-K", type=float, default=3800.0)
    p.add_argument("--norm-r", type=float, default=None)
    p.add_argument("--retrograde", action="store_true")
    p.add_argument("--sky-image", type=Path,
                   default=ROOT / "data" / "raw" / "gaia_panorama.png")
    p.add_argument("--sky-brightness", type=float, default=1.0)
    p.add_argument("--modelos", type=Path, default=ROOT / "models/checkpoints_mino")
    p.add_argument("--mu-exacto", action="store_true",
                   help="usa la forma cerrada de cos(theta) (Jacobi sn) en vez "
                        "de la red. Mas preciso; la red se queda en ~2.5e-3 y "
                        "no baja de ahi ni con 6x parametros")
    p.add_argument("--compare", action="store_true",
                   help="renderiza tambien por el trazador clasico y mide la "
                        "diferencia en PIXELES")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    k = KMetric(M=1.0, a=a.spin)
    theta_obs = np.deg2rad(a.inc)
    cfg = {"r_obs": a.r_obs, "theta_obs": theta_obs, "spin": a.spin,
           "sky_brightness": a.sky_brightness}
    if not a.no_disk:
        cfg["disk"] = {"r_in": float(k.r_isco(prograde=not a.retrograde)),
                       "r_out": a.r_out, "n_images": a.n_images,
                       "turbulence": a.turbulence, "temp_K": a.temp_K,
                       "norm_r": a.norm_r, "prograde": not a.retrograde}

    sky_tex = load_sky_image(a.sky_image) if a.sky_image.exists() else None
    al, be = malla_imagen(a.width, a.height, a.half_width)
    print(f"dispositivo {device}   {al.size:,} pixeles   a={a.spin}  inc={a.inc}")

    net_r = KerrRNet.cargar(a.modelos / "kerr_mino_r.pt", device).to(device)
    net_th = KerrThetaNet.cargar(a.modelos / "kerr_mino_theta.pt", device).to(device)

    t0 = time.perf_counter()
    pre = eng.precalcular(a.spin, theta_obs, a.r_obs, al, be,
                          n_cruces=a.n_images)
    t_pre = time.perf_counter() - t0
    t0 = time.perf_counter()
    geo = eng.evaluar(pre, net_r, net_th, device, mu_exacto=a.mu_exacto)
    t_red = time.perf_counter() - t0
    t_neural = t_pre + t_red
    print(f"neural: {t_neural:.2f} s  (precalculo exacto {t_pre:.2f} s + "
          f"redes {t_red:.2f} s)   {1e3*t_neural/al.size:.5f} ms/rayo")
    if geo["respaldo"].any():
        print(f"  {geo['respaldo'].sum():,} pixeles al trazador clasico "
              f"(eta<=0 o eje): {100*geo['respaldo'].mean():.3f}%")

    img = sombrear(geo, cfg, k, sky_tex).reshape(a.height, a.width, 3)
    salida = a.out or (ROOT / "results" / "figures" /
                       f"kerr_mino_a{a.spin}_i{a.inc:.0f}.png")
    salida.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(salida, np.clip(tonemap(bloom(img)), 0, 1))
    print(f"-> {salida}")

    if not a.compare:
        return

    print("\n=== comparacion contra el trazador clasico ===")
    ig = KerrGeodesicIntegrator(k)
    t0 = time.perf_counter()
    geo_c = geometria_clasica(ig, al, be, cfg, a.n_images)
    t_clasico = time.perf_counter() - t0
    print(f"clasico: {t_clasico:.2f} s   {1e3*t_clasico/al.size:.5f} ms/rayo")
    print(f"speedup: {t_clasico/t_neural:.1f}x")

    # --- clasificacion
    dif_cap = (geo["captured"] != geo_c["captured"]) & ~geo["respaldo"]
    print(f"captura identica en el {100*(1-dif_cap.mean()):.3f}% de los pixeles "
          f"({dif_cap.sum():,} distintos)")

    # --- direccion, en PIXELES
    sel = ~geo["captured"] & ~geo_c["captured"] & ~geo["respaldo"]
    if sel.any():
        cos = np.clip((geo["direccion"][sel] * geo_c["direccion"][sel]).sum(1),
                      -1, 1)
        ang = np.arccos(cos)
        px_rad = 2.0 * np.arctan(a.half_width / a.r_obs) / a.width
        px = ang / px_rad
        print(f"direccion de escape: media {px.mean():.3f} px, "
              f"mediana {np.median(px):.3f} px, p95 {np.percentile(px,95):.3f} px, "
              f"peor {px.max():.2f} px")

    img_c = sombrear(geo_c, cfg, k, sky_tex).reshape(a.height, a.width, 3)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for e, (im, t) in enumerate([(img, "neural (Mino)"), (img_c, "clasico"),
                                 (np.abs(img - img_c), "|diferencia|")]):
        ax[e].imshow(np.clip(tonemap(bloom(im)) if e < 2 else im * 8, 0, 1))
        ax[e].set_title(t); ax[e].axis("off")
    fig.tight_layout()
    cmp_path = salida.with_name(salida.stem + "_compare.png")
    fig.savefig(cmp_path, dpi=110)
    print(f"-> {cmp_path}")


if __name__ == "__main__":
    main()
