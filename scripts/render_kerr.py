"""Primera simulacion de Kerr: sombra y lente sobre el cielo real de Gaia.

Trazado por fuerza bruta, un rayo por pixel. No hay atajo posible: con espin
el movimiento deja de ser plano y la trayectoria ya no depende solo de beta,
asi que la tabla 1D del renderer de Schwarzschild no sirve.

El plano imagen usa las coordenadas celestes de Bardeen (alpha, beta) en
unidades de M, las mismas que KMetric.shadow_outline(), de forma que la sombra
trazada y la analitica se pueden superponer para validar.

El eje de giro del agujero se toma alineado con el eje z del catalogo, que es
el polo norte celeste ICRS. Es una eleccion arbitraria: no hay ninguna
relacion fisica entre el espin y las coordenadas del cielo.

    python scripts/render_kerr.py --spin 0.998 -W 160 -H 90
    python scripts/render_kerr.py --compare-spin 0.0 0.9 0.998
"""
from __future__ import annotations

import argparse
import json
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

from physics import disk as D                                 # noqa: E402
from physics.kerr import KMetric                              # noqa: E402
from physics.kerr_integrator import KerrGeodesicIntegrator    # noqa: E402
from rendering.classical_renderer import (                    # noqa: E402
    blackbody_rgb, bloom, disk_turbulence, load_sky_image, sample_equirect,
    star_texture, tonemap)


_G = {}          # estado por proceso hijo, ver _init_worker


def _init_worker(sky_tex, cfg, stars=None):
    """Carga una sola vez por proceso lo que es caro de pasar por pickle.

    El panorama del cielo son decenas de MB; mandarlo con cada bloque de
    pixeles costaria mas que trazar los rayos.
    """
    _G["sky"] = sky_tex
    _G["cfg"] = cfg
    _G["metric"] = KMetric(M=1.0, a=cfg["spin"])
    _G["integ"] = KerrGeodesicIntegrator(_G["metric"])
    _G["stars"] = stars
    if stars is not None:
        from scipy.spatial import cKDTree
        _G["tree"] = cKDTree(stars["direction"])


def _stars_color(dirs, cfg):
    """Color del fondo con las estrellas como FUENTES PUNTUALES.

    Por que hace falta esto y no basta el panorama
    ----------------------------------------------
    Un panorama tiene una resolucion angular FIJA. En un primer plano el
    encuadre abarca una porcion minuscula del cielo (el render heroe cubre
    1/8240 de la esfera), asi que el panorama aporta unos pocos cientos de
    pixeles propios que hay que estirar a millones: salen manchas.

    Una estrella como PUNTO no tiene resolucion: es una coordenada. Da igual
    cuanto se acerque la camara, sale nitida. Son los mismos datos de Gaia,
    solo que sin pasar por una imagen intermedia.

    Cada estrella se trata como un disco pequeño en el PLANO FUENTE, con un
    perfil gaussiano de radio angular `star_sigma`. Eso es lo que hace que la
    magnificacion salga sola: donde la lente amplia, mas pixeles caen dentro
    del mismo disco y la estrella se ve mas grande y mas brillante, sin tener
    que calcular el jacobiano a mano.
    """
    tree, st = _G["tree"], _G["stars"]
    sig = cfg["star_sigma"]                 # radio angular en radianes
    # radio de busqueda en distancia de cuerda; 3 sigmas cubre la gaussiana
    rad = 2.0 * np.sin(min(3.0 * sig, np.pi / 2) / 2.0)

    k = cfg["star_neighbors"]
    d2, idx = tree.query(dirs, k=k, distance_upper_bound=rad,
                         workers=1)
    d2 = np.atleast_2d(d2)
    idx = np.atleast_2d(idx)

    col = np.zeros((dirs.shape[0], 3), np.float32)
    n_st = st["direction"].shape[0]
    for j in range(d2.shape[1]):
        m = idx[:, j] < n_st                      # los que no encontro vienen con n
        if not m.any():
            continue
        ii = idx[m, j]
        # cuerda -> angulo, y perfil gaussiano en el plano fuente
        ang = 2.0 * np.arcsin(np.clip(d2[m, j] / 2.0, 0.0, 1.0))
        w = np.exp(-0.5 * (ang / sig) ** 2)
        col[m] += (st["rgb"][ii] * (st["flux"][ii] * w)[:, None]).astype(np.float32)
    return col * (cfg["sky_brightness"] * cfg["star_gain"])


def _shade_chunk(args):
    """Traza y sombrea un bloque de pixeles. Devuelve RGB en luz lineal.

    Se sombrea aqui dentro, en el hijo, para no devolver al padre las tablas
    de cruces (que con millones de rayos pesan mas que la propia imagen).
    """
    alphas, betas = args
    cfg, k, ig = _G["cfg"], _G["metric"], _G["integ"]
    dk = cfg["disk"]

    out = ig.trace_batch(
        alphas, betas, cfg["r_obs"], cfg["theta_obs"],
        rtol=cfg["rtol"], atol=cfg["atol"],
        disk_in=(dk["r_in"] if dk else None),
        disk_out=(dk["r_out"] if dk else None),
        max_crossings=(dk["n_images"] if dk else 6))

    col = np.zeros((alphas.size, 3), np.float32)

    # ---------- fondo
    esc = ~out["captured"]
    if esc.any():
        d = out["direction"][esc]
        if _G.get("stars") is not None:
            col[esc] = _stars_color(d, cfg)
        else:
            th_s = np.arccos(np.clip(d[:, 2], -1.0, 1.0))
            lo_s = np.arctan2(d[:, 1], d[:, 0])
            if _G["sky"] is not None:
                col[esc] = sample_equirect(_G["sky"], th_s, lo_s,
                                           cfg["sky_brightness"])
            else:
                col[esc] = star_texture(th_s, lo_s)

    # ---------- disco (opticamente fino: los cruces SE SUMAN)
    if dk:
        cr, cp, nc = out["cross_r"], out["cross_phi"], out["n_cross"]
        xi = out["L"]                       # L/E del foton, constante del rayo
        for j in range(cr.shape[1]):
            m = nc > j
            if not m.any():
                continue
            r_e = cr[m, j].astype(np.float64)
            ph_e = cp[m, j].astype(np.float64)
            g = k.redshift_g(r_e, xi[m], prograde=dk["prograde"])
            prof = D.temperature_profile(r_e, dk["r_in"])
            if dk["norm_r"]:
                # OJO: temperature_profile normaliza a MAXIMO 1, y el maximo
                # esta en (49/36) x_in. Al subir el espin el ISCO baja, el pico
                # se mete hacia dentro y TODO el resto del disco queda pequeño
                # en relacion a el. Comparar espines asi hace creer que un
                # agujero que gira rapido tiene el disco mas apagado, cuando es
                # justo al reves: su eficiencia radiativa pasa del 5.7% al 32%.
                # Renormalizando en un radio FIJO, el disco exterior queda igual
                # entre espines y las diferencias se ven donde de verdad estan.
                prof = prof / max(D.temperature_profile(dk["norm_r"],
                                                        dk["r_in"]), 1e-9)
            t_obs = np.clip(g * dk["temp_K"] * prof, 800.0, None)
            # g^4 por invariancia de Liouville, y el perfil radial de emision
            bright = np.clip(g, 0.0, None) ** 4 * prof ** 4
            if dk["turbulence"] > 0.0:
                bright = bright * disk_turbulence(r_e, ph_e, dk["turbulence"])
            col[np.flatnonzero(m)] += (blackbody_rgb(t_obs)
                                       * bright[:, None]).astype(np.float32)
    return col


def load_star_catalog(path: Path, gmax: float | None = None) -> dict:
    """Carga el catalogo y le precalcula el color de cada estrella.

    El color sale del indice bp_rp por la misma via que en el panorama, asi
    que las dos representaciones del cielo son consistentes entre si.
    """
    from rendering.starfield import bp_rp_to_temp
    d = np.load(path, allow_pickle=False)
    gmag = d["gmag"]
    m = np.ones(gmag.size, bool) if gmax is None else (gmag <= gmax)
    return {"direction": np.ascontiguousarray(d["direction"][m], np.float64),
            "flux": d["flux"][m].astype(np.float64),
            "rgb": blackbody_rgb(bp_rp_to_temp(d["bp_rp"][m])).astype(np.float64)}


def render(cfg: dict, width: int, height: int, half_width: float,
           sky_tex, workers: int, ss: int = 1,
           stars: dict | None = None) -> tuple[np.ndarray, dict]:
    """Imagen (height, width, 3) en luz lineal, mas un diccionario de tiempos.

    Con ss>1 se traza a ss veces la resolucion en cada eje y se promedia. El
    coste sube como ss^2, pero es la unica forma de quitar el dentado del
    borde de la sombra, que es una discontinuidad de verdad.
    """
    W, H = width * ss, height * ss
    ax = np.linspace(-half_width, half_width, W)
    ay = np.linspace(-half_width, half_width, H) * (height / width)
    A, B = np.meshgrid(ax, ay[::-1])            # fila 0 arriba
    flat_a, flat_b = A.ravel(), B.ravel()
    n = flat_a.size

    # Bloques intercalados: el coste por rayo varia muchisimo (los que caen
    # cerca del anillo de fotones dan muchas vueltas). Intercalar reparte esos
    # rayos caros entre todos los procesos en vez de amontonarlos en uno.
    chunks = [(flat_a[i::workers], flat_b[i::workers]) for i in range(workers)]

    t0 = time.perf_counter()
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers, initializer=_init_worker,
                  initargs=(sky_tex, cfg, stars)) as pool:
            parts = pool.map(_shade_chunk, chunks)
    else:
        _init_worker(sky_tex, cfg, stars)
        parts = [_shade_chunk(c) for c in chunks]
    t_trace = time.perf_counter() - t0

    col = np.zeros((n, 3), np.float32)
    for i, p in enumerate(parts):
        col[i::workers] = p
    img = col.reshape(H, W, 3)
    if ss > 1:
        img = img.reshape(height, ss, width, ss, 3).mean(axis=(1, 3))

    return img, {"n_rays": n, "t_trace": t_trace,
                 "ms_per_ray": t_trace / n * 1000.0}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spin", type=float, default=0.9, help="a/M, entre -1 y 1")
    p.add_argument("--compare-spin", type=float, nargs="+", default=None,
                   help="renderiza varios espines y los pone en una figura")
    p.add_argument("-W", "--width", type=int, default=160)
    p.add_argument("-H", "--height", type=int, default=90)
    p.add_argument("--ss", type=int, default=1, help="supersampling (2 = 4x rayos)")
    p.add_argument("--half-width", type=float, default=11.0,
                   help="semiancho del plano imagen, en M")
    p.add_argument("--r-obs", type=float, default=1000.0, help="en M")
    p.add_argument("--inc", type=float, default=85.0,
                   help="0 = de frente al eje de giro, 90 = de canto")

    p.add_argument("--no-disk", action="store_true")
    p.add_argument("--r-out", type=float, default=20.0,
                   help="borde exterior del disco, en M. El interior es el ISCO, "
                        "que lo fija el espin")
    p.add_argument("--n-images", type=int, default=6,
                   help="cuantos cruces del disco se acumulan por rayo")
    p.add_argument("--turbulence", type=float, default=0.45)
    p.add_argument("--temp-K", type=float, default=3800.0,
                   help="temperatura de PRESENTACION, ver DiskParams")
    p.add_argument("--norm-r", type=float, default=None,
                   help="renormaliza el perfil del disco en este radio (en M) "
                        "en vez de en su maximo. IMPRESCINDIBLE para comparar "
                        "espines: sin esto el ISCO mueve el pico y falsea el "
                        "brillo relativo del disco exterior")
    p.add_argument("--retrograde", action="store_true",
                   help="disco contrarrotante respecto al agujero")

    p.add_argument("--sky-image", type=Path,
                   default=ROOT / "data" / "raw" / "gaia_panorama.png")
    p.add_argument("--stars", type=Path, default=None,
                   help="catalogo npz para pintar el cielo con estrellas "
                        "PUNTUALES en vez del panorama. Sin limite de "
                        "resolucion, imprescindible en primeros planos")
    p.add_argument("--star-sigma-arcsec", type=float, default=None,
                   help="radio angular de cada estrella en el plano fuente. "
                        "Por defecto se toma ~1.1 pixeles del render: una "
                        "estrella mas pequeña que un pixel cae entre centros y "
                        "desaparece, igual que pasaria sin PSF en un telescopio")
    p.add_argument("--star-gain", type=float, default=400.0,
                   help="ganancia del cielo puntual. Hace falta bastante: el "
                        "flujo es 10^(-0.4 g), y en g=12 vale 1.6e-5")
    p.add_argument("--star-neighbors", type=int, default=4,
                   help="cuantas estrellas cercanas suma por pixel")
    p.add_argument("--gmax", type=float, default=None,
                   help="recorta el catalogo a magnitud <= gmax")
    p.add_argument("--sky-brightness", type=float, default=0.6)
    p.add_argument("--exposure", type=float, default=1.3)
    p.add_argument("--bloom", type=float, default=0.0, help="umbral; 0 lo apaga")
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--workers", type=int, default=0, help="0 = todos los nucleos")
    p.add_argument("--outline", action="store_true",
                   help="superpone la sombra analitica para validar")
    p.add_argument("--bench", type=Path, default=None,
                   help="guarda los tiempos en un JSON")
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "figures" / "render_kerr.png")
    a = p.parse_args()

    if a.workers <= 0:
        import os
        a.workers = max(1, (os.cpu_count() or 2))

    stars = None
    if a.stars:
        stars = load_star_catalog(a.stars, a.gmax)
        print(f"cielo: {stars['direction'].shape[0]:,} estrellas PUNTUALES "
              f"de {a.stars.name}")
        sky = None
    else:
        sky = (load_sky_image(a.sky_image)
               if a.sky_image and a.sky_image.exists() else None)
        if sky is None:
            print(f"aviso: no encuentro {a.sky_image}, uso cielo procedural")
        else:
            print(f"cielo: panorama {a.sky_image.name} "
                  f"{sky.shape[1]}x{sky.shape[0]}")

    spins = a.compare_spin if a.compare_spin else [a.spin]
    imgs, bench = [], []
    t_all = time.perf_counter()

    for s in spins:
        k = KMetric(M=1.0, a=s)
        disk = None if a.no_disk else {
            "r_in": k.r_isco(prograde=not a.retrograde),
            "r_out": a.r_out, "n_images": a.n_images,
            "turbulence": a.turbulence, "temp_K": a.temp_K,
            "prograde": not a.retrograde, "norm_r": a.norm_r}
        # tamaño angular de un pixel del render, en radianes
        px_rad = 2.0 * np.arctan(a.half_width / a.r_obs) / (a.width * a.ss)
        sigma = (np.deg2rad(a.star_sigma_arcsec / 3600.0)
                 if a.star_sigma_arcsec else 1.1 * px_rad)
        cfg = {"spin": s, "r_obs": a.r_obs,
               "theta_obs": np.deg2rad(a.inc),
               "rtol": a.rtol, "atol": a.atol,
               "sky_brightness": a.sky_brightness, "disk": disk,
               "star_sigma": sigma,
               "star_gain": a.star_gain,
               "star_neighbors": a.star_neighbors}
        if stars is not None and s == spins[0]:
            print(f"  pixel = {np.degrees(px_rad)*3600:.2f} arcsec, "
                  f"sigma estrella = {np.degrees(sigma)*3600:.2f} arcsec")

        print(f"a* = {s}   ISCO = {k.r_isco():.4f} M   "
              f"horizonte = {k.r_horizon:.4f} M")
        img, st = render(cfg, a.width, a.height, a.half_width, sky,
                         a.workers, a.ss, stars)
        print(f"  {st['n_rays']} rayos en {st['t_trace']:.1f} s   "
              f"({st['ms_per_ray']:.4f} ms/rayo, {a.workers} procesos)")
        st.update(spin=s, width=a.width, height=a.height, ss=a.ss,
                  workers=a.workers, isco=k.r_isco(), rtol=a.rtol)
        bench.append(st)

        if a.bloom > 0:
            img = bloom(img, threshold=a.bloom)
        imgs.append(tonemap(img, exposure=a.exposure))

    total = time.perf_counter() - t_all
    print(f"total: {total:.1f} s")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    ext = [-a.half_width, a.half_width,
           -a.half_width * a.height / a.width, a.half_width * a.height / a.width]

    if len(imgs) == 1 and not a.outline:
        plt.imsave(a.out, imgs[0])
    else:
        fig, axs = plt.subplots(len(imgs), 1, figsize=(11, 6.4 * len(imgs)),
                                squeeze=False)
        for ax_, im, s in zip(axs[:, 0], imgs, spins):
            ax_.imshow(im, extent=ext)
            if a.outline:
                al, be = KMetric(1.0, s).shadow_outline(a.inc, n=3000)
                ax_.plot(np.append(al, al[0]), np.append(be, be[0]),
                         "--", color="#ff4444", lw=1.1, label="sombra analitica")
                ax_.legend(loc="upper right", fontsize=8)
            ax_.set_title(f"a* = {s}", fontsize=11)
            ax_.set_xlabel("alpha [M]")
            ax_.set_ylabel("beta [M]")
        fig.tight_layout()
        fig.savefig(a.out, dpi=115)
    print(f"guardado en {a.out}")

    if a.bench:
        a.bench.parent.mkdir(parents=True, exist_ok=True)
        a.bench.write_text(json.dumps(
            {"total_s": total, "runs": bench}, indent=2), encoding="utf-8")
        print(f"tiempos en {a.bench}")


if __name__ == "__main__":
    main()
