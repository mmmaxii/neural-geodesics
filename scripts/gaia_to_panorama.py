"""Convierte el catalogo real de Gaia en un panorama equirectangular.

Por que hace falta
------------------
Una fuente PUNTUAL no muestra deformacion: un punto lensado sigue siendo un
punto, solo cambia de sitio y de brillo. Por muchas estrellas que haya, el
alabeo del espacio no se ve. Para que la lente se note hace falta una fuente
EXTENDIDA y continua, donde cada trozo de estructura se pueda estirar.

Este script construye esa fuente continua a partir de las MISMAS estrellas
reales: acumula sus flujos en una malla (theta, lon) y suaviza un poco. La
banda de la Via Lactea sale sola, porque es densidad real de estrellas, y es
justo la estructura de gran escala que la lente deforma de forma visible.

Sigue siendo dato real: posiciones ICRS reales, flujos reales, colores reales
de bp_rp. Lo unico que cambia es la representacion (mapa en vez de puntos).

    python scripts/gaia_to_panorama.py
    python scripts/gaia_to_panorama.py --height 1536 --blur 1.2

La salida la consume el renderer:

    python scripts/render_classical.py --sky-image data/raw/gaia_panorama.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rendering.classical_renderer import blackbody_rgb   # noqa: E402
from rendering.starfield import bp_rp_to_temp            # noqa: E402


def build(direction, flux, bp_rp, height: int, blur: float,
          clip_pct: float) -> np.ndarray:
    """Acumula las estrellas en una malla equirectangular 2:1.

    La convencion (theta desde +z, lon = atan2(y, x)) es la MISMA que usa
    sample_equirect() en el renderer. Si no coincidiera, el cielo saldria
    girado o espejado respecto a las direcciones de salida de los rayos.
    """
    h, w = height, 2 * height
    d = direction / np.linalg.norm(direction, axis=1, keepdims=True)
    theta = np.arccos(np.clip(d[:, 2], -1.0, 1.0))     # 0 en el polo norte
    lon = np.arctan2(d[:, 1], d[:, 0])

    i = np.clip((theta / np.pi * (h - 1)).astype(np.int32), 0, h - 1)
    j = (((lon + np.pi) / (2 * np.pi)) % 1.0 * w).astype(np.int32) % w

    # color real de cada estrella a partir de su indice bp_rp
    rgb_star = blackbody_rgb(bp_rp_to_temp(bp_rp))

    img = np.zeros((h, w, 3), np.float64)
    for c in range(3):
        np.add.at(img[..., c], (i, j), flux * rgb_star[:, c])

    lat = (np.arange(h) + 0.5) / h * np.pi
    sin_lat = np.maximum(np.sin(lat), 1e-3)

    if blur > 0:
        from scipy.ndimage import gaussian_filter, gaussian_filter1d
        # Suavizado en LONGITUD dependiente de la latitud. Un sigma fijo en
        # pixeles no vale: cerca de los polos los pixeles cubren muchisimo menos
        # cielo en longitud, asi que para suavizar la MISMA escala angular hace
        # falta sigma / sin(theta) pixeles. Sin esto los polos quedan negros:
        # las pocas estrellas que caen ahi se reparten entre 2048 columnas.
        sig = np.clip(blur / sin_lat, blur, w / 8.0)
        for r in range(h):
            img[r] = gaussian_filter1d(img[r], sig[r], axis=0, mode="wrap")
        # y en latitud, donde el paso angular si es uniforme
        img = gaussian_filter(img, (blur, 0, 0), mode="nearest")

    # Compensacion de area: cerca de los polos las celdas de la malla cubren
    # MENOS cielo, asi que el mismo flujo repartido en menos cielo daria un
    # brillo superficial artificialmente alto.
    img /= sin_lat[:, None, None]

    # Normalizacion robusta: unas pocas estrellas muy brillantes se llevarian
    # todo el rango dinamico y el resto del cielo saldria negro.
    scale = np.percentile(img[img > 0], clip_pct) if (img > 0).any() else 1.0
    return np.clip(img / max(scale, 1e-12), 0.0, 1.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", type=Path,
                   default=ROOT / "data" / "raw" / "gaia_stars.npz")
    p.add_argument("--height", type=int, default=1024,
                   help="alto del panorama; el ancho es el doble")
    p.add_argument("--blur", type=float, default=1.0,
                   help="suavizado en pixeles; convierte los puntos en campo continuo")
    p.add_argument("--clip-pct", type=float, default=99.9,
                   help="percentil que se mapea a blanco")
    p.add_argument("--out", type=Path,
                   default=ROOT / "data" / "raw" / "gaia_panorama.png")
    a = p.parse_args()

    d = np.load(a.catalog, allow_pickle=False)
    direction = d["direction"].astype(np.float64)
    flux = d["flux"].astype(np.float64)
    bp_rp = d["bp_rp"].astype(np.float64)
    print(f"catalogo: {a.catalog.name}, {direction.shape[0]} estrellas reales")

    img = build(direction, flux, bp_rp, a.height, a.blur, a.clip_pct)

    # a sRGB: el renderer deshace esta gamma al cargar (load_sky_image)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(a.out, img ** (1.0 / 2.2))
    print(f"panorama {img.shape[1]}x{img.shape[0]} -> {a.out}")


if __name__ == "__main__":
    main()
