"""Render de Kerr con el surrogate en tiempo de Mino.

Espejo de scripts/render_kerr.py: mismos flags, mismo sombreado, misma
convencion de imagen. Lo unico que cambia es COMO se obtiene la geometria de
cada rayo -- por las dos redes de tiempo de Mino mas algebra exacta, en vez de
integrando una EDO de cinco variables por pixel.

    python scripts/render_kerr_mino.py --spin 0.9 --inc 85 -W 480 -H 270
    python scripts/render_kerr_mino.py --spin 0.9 --inc 85 --compare
    python scripts/render_kerr_mino.py --spin 0 --grid --no-disk --fov-deg 60 --r-obs 20

--compare renderiza la MISMA escena por los dos caminos y mide la diferencia en
pixeles, que es la unidad que importa: un error angular pequeno puede ser
enorme si el encuadre es cerrado.

La camara
---------
El encuadre lo fija --fov-deg, un campo de vision angular de verdad, y la
distancia --r-obs va aparte. Antes el encuadre era --half-width, en unidades de
M: eso es un parametro de impacto, no un angulo, y como la sombra mide siempre
~5 M en esas coordenadas la consecuencia era que al mover la distancia la
sombra NO cambiaba de tamano y lo unico que se movia era la escala del fondo.
Parecia que el agujero estaba clavado y que se alejaba el cielo. Ahora
acercarse agranda la sombra y el anillo, y el fondo se queda donde estaba.
La conversion pixel -> rayo la hace physics/kerr_camera.py por la tetrada del
observador local; --half-width sigue aceptandose para reproducir encuadres
viejos. Para MIRAR si el lensing esta bien, usar --grid: un campo de estrellas
no delata un mapeo espejado y una rejilla si.
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
import physics.kerr_camera as kc                                # noqa: E402
import rendering.kerr_mino_engine as eng                       # noqa: E402
from physics.kerr import KMetric                               # noqa: E402
from physics.kerr_integrator import KerrGeodesicIntegrator     # noqa: E402
from models.kerr_mino_net import KerrRNet, KerrThetaNet        # noqa: E402
from rendering.classical_renderer import (                     # noqa: E402
    blackbody_rgb, bloom, disk_turbulence, grid_texture, load_sky_image,
    sample_equirect, star_texture, tonemap)


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
        if cfg.get("grid"):
            # rejilla de coordenadas: si el mapeo pixel -> direccion esta
            # girado o espejado, una rejilla lo canta y un campo de estrellas
            # no. Es la referencia visual que usa el renderer de Schwarzschild.
            col[esc] = grid_texture(th_s, lo_s)
        elif sky_tex is not None:
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
                         max_crossings=n_cruces,
                         r_escape=cfg["r_esc"])
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
    p.add_argument("--fov-deg", type=float, default=1.259,
                   help="campo de vision HORIZONTAL en grados. Es el encuadre: "
                        "es independiente de --r-obs, asi que acercar la camara "
                        "agranda el agujero en vez de hacer zoom al fondo. El "
                        "valor por defecto es el que daba --half-width 11 desde "
                        "r_obs = 1000 M")
    p.add_argument("--half-width", type=float, default=None,
                   help="COMPATIBILIDAD: semiancho del plano imagen en M, como "
                        "antes. Se convierte al FOV equivalente para el --r-obs "
                        "dado y sustituye a --fov-deg. Sirve para reproducir "
                        "encuadres viejos; para trabajar, usa --fov-deg")
    p.add_argument("--r-obs", type=float, default=1000.0)
    p.add_argument("--r-esc", type=float, default=None,
                   help="radio al que se lee la direccion de escape. Por "
                        "defecto max(1e4, 20 r_obs); ver kerr_camera.radio_escape")
    p.add_argument("--grid", action="store_true",
                   help="rejilla de coordenadas como fondo en vez del cielo. Es "
                        "lo que hay que mirar para comprobar el lensing: un "
                        "campo de estrellas no delata un mapeo espejado")
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
    fov = (kc.fov_desde_half_width(a.r_obs, a.half_width)
           if a.half_width is not None else a.fov_deg)
    r_esc = a.r_esc if a.r_esc is not None else kc.radio_escape(a.r_obs)
    cfg = {"r_obs": a.r_obs, "theta_obs": theta_obs, "spin": a.spin,
           "sky_brightness": a.sky_brightness, "grid": a.grid,
           "r_esc": r_esc}
    if not a.no_disk:
        cfg["disk"] = {"r_in": float(k.r_isco(prograde=not a.retrograde)),
                       "r_out": a.r_out, "n_images": a.n_images,
                       "turbulence": a.turbulence, "temp_K": a.temp_K,
                       "norm_r": a.norm_r, "prograde": not a.retrograde}

    sky_tex = None if a.grid else (load_sky_image(a.sky_image)
                                   if a.sky_image.exists() else None)
    al, be = kc.malla_celeste(k, a.r_obs, theta_obs, fov, a.width, a.height)
    print(f"dispositivo {device}   {al.size:,} pixeles   a={a.spin}  inc={a.inc}")
    print(f"camara: FOV {fov:.4f} deg   r_obs {a.r_obs:.0f} M   "
          f"r_esc {r_esc:.0f} M   |alpha| max {np.abs(al).max():.2f} M")
    if a.spin == 0.0:
        print(f"  radio angular exacto de la sombra: "
              f"{np.rad2deg(kc.radio_angular_sombra_schwarzschild(a.r_obs)):.4f} deg "
              f"= {np.tan(kc.radio_angular_sombra_schwarzschild(a.r_obs))/kc.rad_por_pixel(fov, a.width):.1f} px")

    net_r = KerrRNet.cargar(a.modelos / "kerr_mino_r.pt", device).to(device)
    net_th = KerrThetaNet.cargar(a.modelos / "kerr_mino_theta.pt", device).to(device)

    t0 = time.perf_counter()
    pre = eng.precalcular(a.spin, theta_obs, a.r_obs, al, be,
                          n_cruces=a.n_images, r_esc=r_esc)
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
    # Los cruces que salen no finitos los descarta despues la mascara del disco
    # sin decir nada (nan compara False), asi que se cuentan aqui: un fallo
    # silencioso es peor que uno ruidoso. Vienen de refinar_r en rayos pegados
    # a la curva critica; es previo a la camara angular y sale igual con el
    # encuadre viejo (medido: 25 con la camara vieja, 28 con la nueva, sobre
    # ~41k cruces). Sigue sin resolver.
    n_nan = int((~np.isfinite(geo["cross_r"])).sum())
    if n_nan:
        print(f"  {n_nan:,} cruces no finitos descartados "
              f"({100*n_nan/max(geo['cross_r'].size, 1):.4f}% de la tabla)")

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
        px = ang / kc.rad_por_pixel(fov, a.width)
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
