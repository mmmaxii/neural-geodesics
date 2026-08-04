"""Estrellas puntuales vistas a traves de la lente del agujero negro.

A diferencia de una textura, aqui cada estrella es un punto exacto, y eso hace
aparecer el fenomeno de verdad: UNA MISMA ESTRELLA SE VE VARIAS VECES, a un lado
y al otro de la sombra, y las imagenes de orden alto se amontonan pegadas al
anillo de fotones.

La ecuacion de la lente
-----------------------
El rayo que sale de la camara con parametro de impacto beta barre un angulo
total Phi(beta) hasta el infinito, y acaba apuntando a

    d = cos(Phi) n + sin(Phi) e

Si una estrella esta en direccion s, forma con la camara el angulo

    Theta = arccos(s . n)

y sus imagenes son todos los beta que cumplen

    Phi(beta) = Theta,  2pi - Theta,  2pi + Theta,  4pi - Theta, ...

Las de indice par salen del mismo lado que la estrella; las impares del lado
contrario. Como Phi diverge cuando beta tiende al critico, hay infinitas
imagenes, cada vez mas pegadas al borde de la sombra y mas debiles.

Comprobacion sin gravedad: Phi = pi - psi, luego la primera solucion da
psi = pi - Theta, que es exactamente la separacion angular real. Sin desviacion,
la estrella se ve donde esta.

Magnificacion
-------------
Para una lente con simetria circular,

    mu = sin(psi) / sin(theta_fuente) * |dpsi/dbeta| / |dPhi/dbeta|

El |dPhi/dbeta| del denominador es la razon de que las imagenes de orden alto
sean tan tenues: alli Phi cambia muy deprisa con beta.

NOTA PARA LA FASE NEURONAL: esto necesita Phi(beta) Y SU DERIVADA. Con RK45 la
derivada sale por diferencias finitas, con ruido. Una red la da exacta por
diferenciacion automatica, asi que este renderizado es mejor con red que sin
ella, no solo mas rapido.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .classical_renderer import blackbody_rgb


def bp_rp_to_temp(bp_rp):
    """Indice de color de Gaia -> temperatura aproximada, en kelvin.

    bp_rp es la diferencia de brillo entre el filtro azul y el rojo: negativo
    significa estrella azul y caliente, y por encima de 1.5 roja y fria.
    Formula de Ballesteros, pensada para B-V y aqui usada como aproximacion.
    """
    c = np.clip(np.asarray(bp_rp, float), -0.4, 3.0)
    return 4600.0 * (1.0 / (0.92 * c + 1.70) + 1.0 / (0.92 * c + 0.62))


def load_catalog(path):
    d = np.load(Path(path), allow_pickle=False)
    return (d["direction"].astype(np.float64), d["flux"].astype(np.float64),
            d["bp_rp"].astype(np.float64))


def synthetic_catalog(n=2_000_000, seed=0, band_fraction=0.55):
    """Catalogo falso para probar la tuberia sin depender de la descarga.

    Mezcla estrellas repartidas por todo el cielo con una concentracion en una
    banda, imitando el plano galactico.
    """
    rng = np.random.default_rng(seed)
    n_band = int(n * band_fraction)
    d = rng.normal(size=(n - n_band, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)

    pole = np.array([0.34, 0.47, 0.81]); pole /= np.linalg.norm(pole)
    u1 = np.cross(pole, [0, 0, 1.0]); u1 /= np.linalg.norm(u1)
    u2 = np.cross(pole, u1)
    lo = rng.uniform(0, 2 * np.pi, n_band)
    lat = rng.normal(0, 0.12, n_band)
    db = (np.cos(lat)[:, None] * (np.cos(lo)[:, None] * u1 + np.sin(lo)[:, None] * u2)
          + np.sin(lat)[:, None] * pole)
    direction = np.vstack([d, db])

    gmag = 21.0 - 9.0 * rng.power(0.45, direction.shape[0])   # muchas debiles
    flux = 10.0 ** (-0.4 * gmag)
    bp_rp = rng.normal(0.9, 0.5, direction.shape[0])
    return direction, flux, bp_rp


class LensedStarfield:
    """Proyecta un catalogo de estrellas a traves de la lente ya trazada."""

    def __init__(self, renderer, orientation: np.ndarray | None = None):
        self.r = renderer
        if not hasattr(renderer, "phi_inf"):
            renderer.build_table()
        # SOLO rayos que escapan: los capturados no llegan nunca al infinito, asi
        # que su Phi no corresponde a ninguna direccion del cielo. Meterlos en la
        # tabla de inversion la corrompe entera.
        esc = ~renderer.captured_tab
        # Phi decrece con beta (mas lejos = menos desviacion), asi que para
        # invertir por interpolacion hay que darle la vuelta.
        order = np.argsort(renderer.phi_inf[esc])
        self.phi_sorted = renderer.phi_inf[esc][order]
        self.beta_sorted = renderer.beta_tab[esc][order]
        self.psi_sorted = renderer.psi_tab[esc][order]
        keep = np.concatenate(([True], np.diff(self.phi_sorted) > 1e-12))
        self.phi_sorted = self.phi_sorted[keep]
        self.beta_sorted = self.beta_sorted[keep]
        self.psi_sorted = self.psi_sorted[keep]
        # derivadas para la magnificacion
        self.dpsi = np.gradient(self.psi_sorted, self.beta_sorted)
        self.dphi = np.gradient(self.phi_sorted, self.beta_sorted)
        self.orientation = orientation

    # ------------------------------------------------------------------
    def _invert(self, phi):
        """Phi -> (beta, psi, |dpsi/dbeta|, |dPhi/dbeta|). NaN fuera de rango."""
        lo, hi = self.phi_sorted[0], self.phi_sorted[-1]
        ok = (phi >= lo) & (phi <= hi)
        out = [np.full(phi.shape, np.nan) for _ in range(4)]
        if not ok.any():
            return (*out, ok)
        p = phi[ok]
        for arr, tgt in zip((self.beta_sorted, self.psi_sorted,
                             self.dpsi, self.dphi), out):
            tgt[ok] = np.interp(p, self.phi_sorted, arr)
        return (*out, ok)

    # ------------------------------------------------------------------
    def render(self, n_images: int = 4, gain: float = 1.0, psf_sigma: float = 0.9,
               max_mu: float = 400.0) -> np.ndarray:
        cam = self.r.cam
        n, e1, e2 = cam.basis()
        s, flux, bp_rp = self.catalog
        if self.orientation is not None:
            s = s @ self.orientation.T

        cos_t = np.clip(s @ n, -1.0, 1.0)
        Theta = np.arccos(cos_t)
        perp = s - cos_t[:, None] * n
        alpha_s = np.arctan2(perp @ e2, perp @ e1)

        rgb_star = blackbody_rgb(bp_rp_to_temp(bp_rp))
        img = np.zeros((cam.height, cam.width, 3), np.float64)
        half = np.tan(np.deg2rad(cam.fov_deg) / 2.0)
        aspect = cam.height / cam.width

        for k in range(n_images):
            m = k // 2
            if k % 2 == 0:                       # mismo lado que la estrella
                phi_t, alpha = Theta + 2 * np.pi * m, alpha_s
            else:                                # lado contrario
                phi_t, alpha = 2 * np.pi * (m + 1) - Theta, alpha_s + np.pi

            beta, psi, dpsi, dphi, ok = self._invert(phi_t)
            if not ok.any():
                continue
            sin_src = np.abs(np.sin(np.pi - phi_t))
            mu = (np.sin(psi) * np.abs(dpsi)
                  / np.maximum(sin_src * np.abs(dphi), 1e-12))
            mu = np.clip(np.nan_to_num(mu), 0.0, max_mu)

            # (psi, alpha) -> pixel. Inverso exacto de Camera.pixel_angles.
            t = np.tan(psi)
            tx, ty = t * np.sin(alpha), -t * np.cos(alpha)
            ix = (tx / half + 1.0) * cam.width / 2.0 - 0.5
            iy = (ty / (half * aspect) + 1.0) * cam.height / 2.0 - 0.5

            vis = (ok & np.isfinite(ix) & np.isfinite(iy)
                   & (ix >= 0) & (ix < cam.width - 1)
                   & (iy >= 0) & (iy < cam.height - 1) & (mu > 0))
            if not vis.any():
                continue
            w = (flux[vis] * mu[vis] * gain)[:, None] * rgb_star[vis]
            np.add.at(img, (iy[vis].astype(np.int32), ix[vis].astype(np.int32)), w)

        if psf_sigma > 0:
            from scipy.ndimage import gaussian_filter
            img = gaussian_filter(img, (psf_sigma, psf_sigma, 0))
        return img.astype(np.float32)

    # ------------------------------------------------------------------
    catalog: tuple

    @classmethod
    def from_file(cls, renderer, path, orientation=None):
        obj = cls(renderer, orientation)
        obj.catalog = load_catalog(path)
        return obj

    @classmethod
    def synthetic(cls, renderer, n=2_000_000, seed=0, orientation=None):
        obj = cls(renderer, orientation)
        obj.catalog = synthetic_catalog(n, seed)
        return obj
