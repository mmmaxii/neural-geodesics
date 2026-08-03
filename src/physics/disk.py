"""Disco de acreción delgado alrededor de un agujero negro de Schwarzschild.

Todo lo geométrico y lo radiativo relativo es ADIMENSIONAL en x = r / r_s:

    x_horizonte = 0.5 * r_s / r_s ... (r_s)      -> x = 1
    x_fotones   = 1.5 M -> 3M / r_s              -> x = 1.5
    x_isco      = 6M / r_s                       -> x = 3

La ÚNICA cantidad de este módulo que depende de la masa es la escala absoluta
de temperatura (`temperature_scale_K`). El perfil radial, el factor de
corrimiento g, el patrón de brillo y la geometría de los cruces son idénticos
para cualquier M. Por eso la red neuronal nunca necesita ver la masa: se
entrena en variables adimensionales y la masa entra al final, como un simple
cambio de escala espacial y un mapa de color.

Convenciones angulares
----------------------
    inc   : ángulo entre la línea de visión y la NORMAL del disco.
            inc = 0    -> disco de cara
            inc = pi/2 -> disco de canto
    alpha : ángulo polar en el plano imagen, medido desde la proyección del
            eje de rotación del disco.
"""
from __future__ import annotations

import numpy as np

# Constantes físicas (SI) — solo se usan para la escala absoluta de temperatura.
G_SI = 6.674_30e-11        # m^3 kg^-1 s^-2
C_SI = 2.997_924_58e8      # m s^-1
SIGMA_SB = 5.670_374_419e-8  # W m^-2 K^-4
M_SUN = 1.988_47e30        # kg

# Radios característicos en unidades de r_s (exactos, independientes de M)
X_HORIZON = 1.0
X_PHOTON = 1.5
X_ISCO = 3.0
BETA_CRIT = 3.0 * np.sqrt(3.0) / 2.0  # = b_crit / r_s ~ 2.598076


# --------------------------------------------------------------------------
# Cinemática de órbitas circulares
# --------------------------------------------------------------------------
def omega_rs(x):
    """Velocidad angular orbital adimensional Omega * r_s (c = 1).

    Kepler relativista: Omega = sqrt(M / r^3). Con r = x r_s y M = r_s / 2,
        Omega r_s = 1 / sqrt(2 x^3)
    """
    x = np.asarray(x, dtype=float)
    return 1.0 / np.sqrt(2.0 * x**3)


def orbital_velocity(x):
    """Velocidad orbital v/c medida por un observador estático local.

        v = sqrt(M/r) / sqrt(1 - 2M/r) = (1/sqrt(2x)) / sqrt(1 - 1/x)

    En el ISCO (x=3) vale 0.5 exacto, y alcanza exactamente c en la esfera de
    fotones (x=1.5): ahí la órbita circular solo es posible para la luz.
    """
    x = np.asarray(x, dtype=float)
    return (1.0 / np.sqrt(2.0 * x)) / np.sqrt(1.0 - 1.0 / x)


def lorentz_gamma(x):
    return 1.0 / np.sqrt(1.0 - orbital_velocity(x) ** 2)


def time_dilation(x):
    """Factor sqrt(1 - 3M/r) = sqrt(1 - 1.5/x): dilatación total (gravedad + movimiento).

    Se anula en la esfera de fotones (x = 1.5), que es justo donde las órbitas
    circulares dejan de existir para partículas masivas.
    """
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.clip(1.0 - 1.5 / x, 0.0, None))


# --------------------------------------------------------------------------
# Corrimiento al rojo / azul
# --------------------------------------------------------------------------
def redshift_g(x, xi):
    """Factor g = nu_obs / nu_emitida para un elemento del disco.

        g = sqrt(1 - 3M/r) / (1 - Omega * b_z)

    donde b_z = L_z / E es el momento angular del fotón respecto al eje de
    rotación del disco, por unidad de energía. En el plano imagen, b_z es
    simplemente la coordenada perpendicular al eje proyectado, multiplicada
    por sin(inc) — eso es lo que hay que pasar como `xi` (ya en unidades de r_s
    y ya con el sin(inc) incluido; ver `image_to_bz`).

    El numerador (gravedad + dilatación) siempre oscurece; el denominador es el
    Doppler, y es lo que hace que un lado del disco salga mucho más brillante
    que el otro. Es el efecto dominante en las imágenes conocidas.
    """
    x = np.asarray(x, dtype=float)
    xi = np.asarray(xi, dtype=float)
    return time_dilation(x) / (1.0 - omega_rs(x) * xi)


def image_to_bz(x_img, inc):
    """Componente b_z (en r_s) a partir de la coordenada del plano imagen.

    x_img es la coordenada perpendicular a la proyección del eje de rotación,
    en unidades de r_s. Con el disco de cara (inc=0) no hay componente Doppler
    a lo largo de la línea de visión y b_z se anula: el disco sale simétrico.
    """
    return np.asarray(x_img, dtype=float) * np.sin(inc)


def beaming_factor(g):
    """Amplificación de intensidad específica integrada: I_obs / I_em = g^4.

    Tres potencias vienen de la invariancia de I_nu / nu^3 (Liouville) y una
    de integrar sobre frecuencia.
    """
    return np.asarray(g, dtype=float) ** 4


# --------------------------------------------------------------------------
# Perfil de temperatura (Shakura-Sunyaev, disco delgado)
# --------------------------------------------------------------------------
def temperature_profile(x, x_in=X_ISCO):
    """Forma radial adimensional de la temperatura, normalizada a máximo 1.

        T(x) ∝ [ x^-3 (1 - sqrt(x_in/x)) ]^(1/4)

    Es cero en el borde interno (condición de torque nulo), sube hasta un
    máximo en x = (49/36) x_in y luego cae como x^(-3/4).
    """
    x = np.asarray(x, dtype=float)
    inner = np.clip(1.0 - np.sqrt(x_in / np.maximum(x, 1e-12)), 0.0, None)
    prof = (inner / np.maximum(x, 1e-12) ** 3) ** 0.25
    x_peak = (49.0 / 36.0) * x_in
    peak = ((1.0 - np.sqrt(x_in / x_peak)) / x_peak**3) ** 0.25
    return np.where(x >= x_in, prof / peak, 0.0)


def temperature_scale_K(mass_solar, mdot_kg_s):
    """Prefactor absoluto de temperatura, en kelvin. ES LO ÚNICO QUE VE LA MASA.

        T^4 = 3 Mdot c^6 / (64 pi sigma G^2 M^2) * x^-3 (1 - sqrt(x_in/x))

    de donde T ∝ (Mdot / M^2)^(1/4). Con acreción de Eddington (Mdot ∝ M) queda
    T ∝ M^(-1/4): los agujeros negros supermasivos tienen discos MÁS FRÍOS
    (ultravioleta) que los estelares (rayos X), pese a ser mucho más luminosos.
    """
    M = mass_solar * M_SUN
    t4 = 3.0 * mdot_kg_s * C_SI**6 / (64.0 * np.pi * SIGMA_SB * G_SI**2 * M**2)
    return t4**0.25


M_PROTON = 1.672_621_9e-27   # kg
SIGMA_THOMSON = 6.652_458_7e-29  # m^2


def eddington_luminosity(mass_solar):
    """L_Edd = 4 pi G M m_p c / sigma_T, en watts. ~1.26e31 W por masa solar."""
    return (4.0 * np.pi * G_SI * mass_solar * M_SUN * M_PROTON * C_SI
            / SIGMA_THOMSON)


def eddington_mdot(mass_solar, efficiency=0.1):
    """Tasa de acreción de Eddington en kg/s: Mdot = L_Edd / (eta c^2)."""
    return eddington_luminosity(mass_solar) / (efficiency * C_SI**2)


def temperature_K(x, mass_solar, mdot_kg_s, x_in=X_ISCO):
    """Temperatura física del disco en kelvin (perfil x escala)."""
    x = np.asarray(x, dtype=float)
    inner = np.clip(1.0 - np.sqrt(x_in / np.maximum(x, 1e-12)), 0.0, None)
    shape = (inner / np.maximum(x, 1e-12) ** 3) ** 0.25
    return temperature_scale_K(mass_solar, mdot_kg_s) * np.where(x >= x_in, shape, 0.0)


# --------------------------------------------------------------------------
# Geometría: dónde cruza un rayo el plano ecuatorial
# --------------------------------------------------------------------------
def first_crossing_phi(alpha, inc):
    """Ángulo phi del primer cruce del rayo con el plano ecuatorial.

    El rayo es plano (simetría esférica). Escribiendo su dirección como

        r_hat(phi) = cos(phi) n_hat + sin(phi) e_hat

    con n_hat hacia el observador y e_hat en el plano imagen en la dirección
    del píxel, la condición de cruce r_hat . z_hat = 0 da

        cos(phi) cos(inc) + sin(phi) sin(inc) cos(alpha) = 0
        =>  tan(phi) = -cot(inc) / cos(alpha)

    Comprobaciones: de cara (inc=0) devuelve pi/2, porque el rayo tiene que
    girar 90 grados para llegar al plano ecuatorial. De canto (inc=pi/2)
    devuelve pi, porque el observador YA está en el plano y el siguiente cruce
    llega tras media vuelta.

    De canto el observador YA está en el plano ecuatorial, así que phi=0 es
    técnicamente un cruce, pero es el observador mismo; se descarta y se
    devuelve el siguiente. Por eso el resultado se envuelve en (0, pi], no en
    [0, pi).

    Los cruces siguientes están en phi_0 + k*pi (ver `crossing_angles`).
    """
    alpha = np.asarray(alpha, dtype=float)
    phi = np.mod(np.arctan2(-np.cos(inc), np.sin(inc) * np.cos(alpha)), np.pi)
    return np.where(phi <= 1e-12, np.pi, phi)


def crossing_angles(alpha, inc, n_images=3):
    """Ángulos phi de los n primeros cruces con el ecuador.

    k=0 es la imagen directa del disco.
    k=1 es la imagen secundaria: el disco visto POR DETRÁS del agujero, el arco
        que parece curvarse por encima. Es lo que hace icónicas estas imágenes.
    k>=2 es el anillo de fotones.

    Todas salen del mismo forward pass de la red, evaluando u(beta, phi_k).
    """
    phi0 = first_crossing_phi(alpha, inc)
    k = np.arange(n_images).reshape((-1,) + (1,) * np.ndim(phi0))
    return phi0 + k * np.pi


def disk_hit(x_cross, x_in=X_ISCO, x_out=20.0):
    """True donde el radio de cruce cae dentro del disco."""
    x_cross = np.asarray(x_cross, dtype=float)
    return np.isfinite(x_cross) & (x_cross >= x_in) & (x_cross <= x_out)
