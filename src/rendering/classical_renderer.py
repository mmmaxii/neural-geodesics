"""Renderer clasico: traza rayos con RK45 y produce la imagen del agujero negro.

Es la VERDAD-TERRENO. Lento a proposito: la red neuronal tendra que reproducir
exactamente esta imagen, mucho mas rapido.

Todo en unidades de r_s (radio de Schwarzschild), asi que la imagen es la misma
para cualquier masa. La masa solo entra al final, en el color del disco.

Geometria
---------
La camara esta a R = r_cam / r_s del centro, formando angulo `inclination` con
la NORMAL del disco (0 = de cara, 90 grados = de canto).

Para cada pixel se calculan dos numeros:

    psi   : angulo entre el rayo y la direccion radial (que tan de refilon pasa)
    alpha : angulo polar en el plano imagen, medido desde el eje de rotacion
            proyectado

y de psi sale el parametro de impacto, con el factor exacto de la metrica:

    beta = R sin(psi) / sqrt(1 - 1/R)

El rayo se mueve en el plano que contienen la camara y su direccion, y su
posicion angular es

    r_hat(phi) = cos(phi) n_hat + sin(phi) e_hat

con phi = 0 en la camara. Eso es justo la parametrizacion que usa
physics/disk.py para los cruces con el disco.

Optimizacion clave
------------------
El espaciotiempo es esfericamente simetrico, asi que la trayectoria depende SOLO
de beta, no de alpha. Todos los pixeles a la misma distancia del centro comparten
trayectoria. Se integra una tabla de ~1500 valores de beta en vez de un rayo por
pixel, y los pixeles la consultan. Es exacto, no es una aproximacion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics import disk as D
from physics.integrator import GeodesicIntegrator
from physics.schwarzschild import SMetric

BETA_CRIT = D.BETA_CRIT


# --------------------------------------------------------------------------
@dataclass
class Camera:
    """Camara pinhole apuntando al agujero negro."""

    r_cam: float = 30.0            # en r_s
    inclination_deg: float = 80.0  # 0 = de cara al disco, 90 = de canto
    fov_deg: float = 28.0          # campo de vision horizontal
    width: int = 480
    height: int = 270

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(n, e1, e2): hacia la camara, 'arriba' en la imagen, y el tercero.

        e1 es la proyeccion del eje de rotacion del disco sobre el plano imagen,
        que es la referencia desde la que disk.py mide el angulo alpha.
        """
        i = np.deg2rad(self.inclination_deg)
        n = np.array([np.sin(i), 0.0, np.cos(i)])
        e1 = np.array([-np.cos(i), 0.0, np.sin(i)])
        e2 = np.cross(n, e1)
        return n, e1, e2

    def pixel_angles(self) -> tuple[np.ndarray, np.ndarray]:
        """(psi, alpha) para cada pixel, en radianes."""
        half = np.tan(np.deg2rad(self.fov_deg) / 2.0)
        tx = (2.0 * (np.arange(self.width) + 0.5) / self.width - 1.0) * half
        ty = (2.0 * (np.arange(self.height) + 0.5) / self.height - 1.0) * half \
            * (self.height / self.width)
        TX, TY = np.meshgrid(tx, -ty)          # -ty: fila 0 arriba
        psi = np.arctan(np.hypot(TX, TY))
        alpha = np.arctan2(TX, TY)             # e_hat = cos(a) e1 + sin(a) e2
        return psi, alpha

    def impact_parameter(self, psi):
        """beta = R sin(psi) / sqrt(1 - 1/R).

        El sqrt es g_tt de la metrica (f_aux en SMetric). Sin el, la sombra sale
        del tamano equivocado, y el error crece cuanto mas cerca este la camara.
        """
        return self.r_cam * np.sin(psi) / np.sqrt(1.0 - 1.0 / self.r_cam)

    def shadow_angle(self) -> float:
        """Radio angular exacto de la sombra. La prueba de fuego del mapeo."""
        s = BETA_CRIT / self.r_cam * np.sqrt(1.0 - 1.0 / self.r_cam)
        return float(np.arcsin(np.clip(s, -1.0, 1.0)))


# --------------------------------------------------------------------------
@dataclass
class DiskParams:
    x_in: float = D.X_ISCO
    x_out: float = 14.0
    n_images: int = 5        # 0 directa, 1 secundaria, 2+ anillo de fotones
    turbulence: float = 0.85  # 0 = disco liso analitico
    mass_solar: float = 1e7
    eddington_ratio: float = 0.1

    # Temperatura de PRESENTACION, no fisica. Un disco real de 1e7 masas solares
    # tiene su pico cerca de 6e5 K y emite en ultravioleta extremo: el ojo no lo
    # veria. Se mapea el perfil al visible haciendo que el pico sin corrimiento
    # caiga aqui, de forma que los colores muestran los desplazamientos
    # RELATIVOS (Doppler y gravitatorio), que es la informacion fisica real.
    # La temperatura absoluta se sigue pudiendo consultar con
    # physics.disk.temperature_K.
    display_temp_K: float = 3800.0


# --------------------------------------------------------------------------
class ClassicalRenderer:
    def __init__(self, camera: Camera, disk: DiskParams | None = None,
                 n_radial: int = 1500, n_phi_tab: int = 3072,
                 rtol: float = 1e-8, atol: float = 1e-11):
        self.cam = camera
        self.disk = disk
        self.n_radial = n_radial
        self.n_phi_tab = n_phi_tab
        self.rtol, self.atol = rtol, atol
        self.metric = SMetric()
        self.integ = GeodesicIntegrator(self.metric)

    # ---------------------------------------------------------------- tabla
    def _sample_psi(self) -> np.ndarray:
        """Valores de psi a integrar, con extra densidad junto al borde de la sombra.

        Ahi la trayectoria cambia muchisimo entre rayos vecinos (es donde el rayo
        decide si cae o escapa), asi que un muestreo uniforme dejaria el anillo de
        fotones pixelado.
        """
        psi_max = np.arctan(np.tan(np.deg2rad(self.cam.fov_deg) / 2.0) * np.sqrt(2))
        psi_sh = self.cam.shadow_angle()
        base = np.linspace(0.0, psi_max, self.n_radial // 2)
        # denso a ambos lados del borde, en escala logaritmica de la distancia
        d = psi_sh * np.logspace(-7, -1.2, self.n_radial // 4)
        near = np.concatenate([psi_sh - d, psi_sh + d])
        return np.unique(np.clip(np.concatenate([base, near]), 0.0, psi_max))

    def build_table(self) -> None:
        """Integra un rayo por cada psi y guarda u(phi) en una malla comun."""
        cam = self.cam
        self.psi_tab = self._sample_psi()
        self.beta_tab = cam.impact_parameter(self.psi_tab)
        rs = self.metric.r_s
        n = self.psi_tab.size

        sols, phi_end, status = [], np.zeros(n), np.zeros(n, np.int8)
        for k, beta in enumerate(self.beta_tab):
            if beta <= 1e-9:                      # rayo radial: cae seguro
                sols.append(None); phi_end[k] = 0.0; status[k] = 1
                continue
            res = self.integ.integrate_ray(
                b=beta * rs, r0=cam.r_cam * rs, rtol=self.rtol, atol=self.atol,
                dense_output=True)
            sols.append(res["sol_dense"])
            phi_end[k] = res["phi"][-1]
            status[k] = 0 if res["status"] == "escaped" else 1

        self.phi_end = phi_end
        # Clasificacion EXACTA, no interpolada: el borde de la sombra es
        # beta = beta_crit y se conoce en forma cerrada. Interpolarlo daria un
        # borde dentado.
        self.captured_tab = self.beta_tab < BETA_CRIT

        # Tabla u(beta, phi) sobre una malla comun de phi -> consulta vectorizada.
        self.phi_grid = np.linspace(0.0, float(phi_end.max()), self.n_phi_tab)
        self.u_tab = np.full((n, self.n_phi_tab), np.nan, np.float32)
        for k, sd in enumerate(sols):
            if sd is None:
                continue
            m = self.phi_grid <= phi_end[k]
            self.u_tab[k, m] = rs * sd(self.phi_grid[m])[0]

        # Direccion asintotica de salida: phi_end solo llega a 1.05*r_cam, y el
        # tramo que falta hasta el infinito barre arcsin(b/r) mas.
        r_end = np.where(phi_end > 0, 1.05 * cam.r_cam, np.inf)
        self.phi_inf = phi_end + np.arcsin(
            np.clip(self.beta_tab / r_end, -1.0, 1.0))

    # ------------------------------------------------------------- consulta
    def _u_at(self, idx: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """u~ interpolada en la tabla. NaN si el rayo ya termino."""
        dphi = self.phi_grid[1] - self.phi_grid[0]
        t = np.clip(phi / dphi, 0.0, self.n_phi_tab - 1.0000001)
        j = t.astype(np.int32)
        f = (t - j).astype(np.float32)
        a = self.u_tab[idx, j]
        b = self.u_tab[idx, np.minimum(j + 1, self.n_phi_tab - 1)]
        out = a * (1.0 - f) + b * f
        return np.where(np.isnan(b), a, out)

    # -------------------------------------------------------------- render
    def render(self, sky: bool = True, sky_mode: str = "stars",
               sky_tex=None, sky_brightness: float = 1.0) -> np.ndarray:
        """sky_mode: 'stars' (procedural), 'grid' (validacion) o 'image'.

        Con 'image' hay que pasar en sky_tex un panorama equirectangular ya
        cargado con load_sky_image().
        """
        self.sky_mode = sky_mode
        self.sky_tex = sky_tex
        self.sky_brightness = sky_brightness
        return self._render(sky)

    def _render(self, sky: bool = True) -> np.ndarray:
        if not hasattr(self, "u_tab"):
            self.build_table()
        cam = self.cam
        psi, alpha = cam.pixel_angles()
        beta = cam.impact_parameter(psi)
        idx = np.searchsorted(self.psi_tab, psi).clip(0, self.psi_tab.size - 1)

        img = np.zeros(psi.shape + (3,), np.float32)
        captured = beta < BETA_CRIT
        alive = ~captured

        if sky:
            img[alive] = self._sky(psi, alpha, idx, alive)
        if self.disk is not None:
            self._add_disk(img, alpha, idx, captured, beta)
        return np.clip(img, 0.0, None)

    # ----------------------------------------------------------------- cielo
    def _sky(self, psi, alpha, idx, alive) -> np.ndarray:
        """Color del fondo segun hacia donde sale el rayo."""
        n, e1, e2 = self.cam.basis()
        a = alpha[alive]
        phi = self.phi_inf[idx[alive]]
        e = np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2
        d = np.cos(phi)[:, None] * n + np.sin(phi)[:, None] * e
        theta = np.arccos(np.clip(d[:, 2], -1.0, 1.0))
        lon = np.arctan2(d[:, 1], d[:, 0])
        mode = getattr(self, "sky_mode", "stars")
        if mode == "grid":
            return grid_texture(theta, lon)
        if mode == "image" and getattr(self, "sky_tex", None) is not None:
            return sample_equirect(self.sky_tex, theta, lon,
                                   getattr(self, "sky_brightness", 1.0))
        return star_texture(theta, lon)

    # ----------------------------------------------------------------- disco
    def _add_disk(self, img, alpha, idx, captured, beta) -> None:
        dk = self.disk
        n, e1, e2 = self.cam.basis()
        inc = np.deg2rad(self.cam.inclination_deg)
        phi0 = D.first_crossing_phi(alpha, inc)

        # DISCO OPTICAMENTE FINO: el gas es transparente, asi que la luz
        # atraviesa todas las capas y las emisiones se SUMAN. Antes cogia solo
        # el primer cruce y tapaba el resto, como un plato pintado. Sumar es lo
        # que hace que el anillo de fotones brille: ahi el rayo cruza el disco
        # muchas veces y va acumulando en cada pasada.
        accum = np.zeros_like(img)
        for k in range(dk.n_images):
            phi_k = phi0 + k * np.pi
            u = self._u_at(idx, phi_k)
            x = 1.0 / u
            # Sin excluir los capturados: un rayo que acaba cayendo tambien
            # recoge la luz del disco que atraveso por el camino.
            hit = D.disk_hit(x, dk.x_in, dk.x_out) & (phi_k <= self.phi_end[idx])
            if not hit.any():
                continue
            # b_z = L_z/E es CONSTANTE a lo largo del rayo (momento angular
            # conservado), no depende de donde emitio el disco. Sale de proyectar
            # el vector parametro de impacto sobre el eje perpendicular al de
            # rotacion: b_vec = beta (cos(a) e1 + sin(a) e2), luego su componente
            # util es beta*sin(a), y image_to_bz le aplica el sin(inclinacion).
            g = D.redshift_g(x, D.image_to_bz(beta * np.sin(alpha), inc))
            prof = D.temperature_profile(x, dk.x_in)
            # Color: cuerpo negro a la temperatura observada. El factor g es el
            # que hace que el lado que se acerca salga mas azul y el que se aleja
            # mas rojo. Escala de presentacion, ver DiskParams.display_temp_K.
            t_obs = np.clip(g * dk.display_temp_K * prof, 800.0, None)
            # Brillo: g^4 por la invariancia de Liouville, y el perfil radial.
            bright = np.clip(g, 0, None) ** 4 * prof ** 4
            if dk.turbulence > 0.0:
                # Azimut del punto de emision en el plano del disco, para poder
                # texturizarlo. La posicion es x * (cos(phi) n + sin(phi) e).
                px = np.cos(phi_k) * np.sin(inc) - np.sin(phi_k) * np.cos(alpha) * np.cos(inc)
                py = -np.sin(phi_k) * np.sin(alpha)
                th_disk = np.arctan2(py, px)
                bright = bright * disk_turbulence(x, th_disk, dk.turbulence)
            rgb = blackbody_rgb(t_obs) * bright[..., None]
            accum += np.where(hit[..., None], rgb, 0.0)
        img += accum


# --------------------------------------------------------------------------
def grid_texture(theta, lon, step_deg: float = 15.0, width_deg: float = 0.7):
    """Rejilla de coordenadas como cielo de fondo.

    Se usa ESTO y no un campo de estrellas para validar: si el mapeo de
    direcciones esta girado o espejado, una rejilla lo canta al instante y un
    campo de estrellas no.
    """
    s = np.deg2rad(step_deg)
    dlat = np.abs((theta % s) - s / 2) < np.deg2rad(width_deg)
    dlon = np.abs((lon % s) - s / 2) < np.deg2rad(width_deg)
    rgb = np.zeros(theta.shape + (3,), np.float32)
    rgb[..., 0] = 0.06 + 0.5 * dlat
    rgb[..., 1] = 0.07 + 0.5 * dlon
    rgb[..., 2] = 0.11 + 0.3 * (dlat | dlon)
    ecuador = np.abs(theta - np.pi / 2) < np.deg2rad(1.2)
    rgb[ecuador] = np.array([0.9, 0.4, 0.1], np.float32)
    return rgb


def load_sky_image(path):
    """Carga un panorama equirectangular y lo pasa a luz lineal.

    Debe ser una imagen de proyeccion equirectangular (relacion 2:1): el eje
    horizontal cubre 360 grados de longitud y el vertical 180 de latitud. Es el
    formato estandar de los panoramas de cielo.

    Se deshace la gamma (sRGB -> lineal) porque el renderer trabaja en luz
    lineal y vuelve a aplicar gamma al final. Sin esto el fondo sale lavado.
    """
    import matplotlib.image as mpimg
    tex = np.asarray(mpimg.imread(str(path)), dtype=np.float32)
    if tex.ndim == 2:
        tex = np.stack([tex] * 3, -1)
    tex = tex[..., :3]
    if tex.max() > 1.5:
        tex /= 255.0
    return tex ** 2.2


def sample_equirect(tex, theta, lon, brightness: float = 1.0):
    """Muestrea el panorama con interpolacion bilineal.

    Bilineal y no vecino mas cercano porque cerca de la sombra la lente comprime
    el cielo muchisimo, y ahi el muestreo tosco se ve como escalones.
    """
    h, w = tex.shape[:2]
    u = ((lon + np.pi) / (2.0 * np.pi)) % 1.0 * w
    v = np.clip(theta / np.pi, 0.0, 1.0) * (h - 1)
    j0 = u.astype(np.int32) % w
    j1 = (j0 + 1) % w
    i0 = np.clip(v.astype(np.int32), 0, h - 1)
    i1 = np.clip(i0 + 1, 0, h - 1)
    fu = (u - np.floor(u))[..., None]
    fv = (v - i0)[..., None]
    top = tex[i0, j0] * (1 - fu) + tex[i0, j1] * fu
    bot = tex[i1, j0] * (1 - fu) + tex[i1, j1] * fu
    return (top * (1 - fv) + bot * fv) * brightness


def star_texture(theta, lon, cells: int = 2600, fraction: float = 0.0016):
    """Campo de estrellas procedural, sin fichero de textura.

    La esfera se divide en celdas y un hash entero decide si cada una tiene
    estrella y con que brillo y color. Es determinista: la misma direccion da
    siempre la misma estrella, que es lo que hace falta para que el fondo no
    parpadee al mover la camara.
    """
    i = np.clip((theta / np.pi * cells).astype(np.int64), 0, cells)
    j = ((lon + np.pi) / (2 * np.pi) * (2 * cells)).astype(np.int64)
    h = (i * 73856093) ^ (j * 19349663)
    h = (h ^ (h >> 13)) * 1274126177
    h = (h ^ (h >> 16)) & 0x7FFFFFFF

    is_star = (h & 0xFFFF) / 65535.0 < fraction
    mag = ((h >> 16) & 0xFF) / 255.0
    bright = (mag ** 5) * 3.0                       # pocas muy brillantes
    warm = ((h >> 8) & 0xFF) / 255.0                # color estelar

    rgb = np.zeros(theta.shape + (3,), np.float32)
    rgb[..., 0] = 0.008 + is_star * bright * (0.75 + 0.35 * warm)
    rgb[..., 1] = 0.010 + is_star * bright * 0.85
    rgb[..., 2] = 0.018 + is_star * bright * (1.15 - 0.35 * warm)
    return rgb


def blackbody_rgb(t_kelvin):
    """Color aproximado de un cuerpo negro a temperatura T (aprox. de Helland)."""
    t = np.clip(np.asarray(t_kelvin, float), 1000.0, 40000.0) / 100.0
    # np.where evalua las dos ramas, asi que hay que proteger la base negativa
    tm = np.maximum(t - 60.0, 1e-9)
    r = np.where(t <= 66, 255.0, 329.7 * tm ** -0.1332)
    g = np.where(t <= 66, 99.47 * np.log(t) - 161.1, 288.1 * tm ** -0.0755)
    b = np.where(t >= 66, 255.0,
                 np.where(t <= 19, 0.0, 138.5 * np.log(np.maximum(t - 10, 1e-9)) - 305.0))
    return np.clip(np.stack([r, g, b], -1) / 255.0, 0.0, 1.0).astype(np.float32)


def _value_noise(a, b):
    """Ruido de valor en una malla entera, interpolado suave. Devuelve [0, 1).

    Se usa un hash entero en vez de numeros aleatorios guardados para que sea
    DETERMINISTA y sin memoria: la misma coordenada da siempre el mismo valor,
    que es lo que hace falta para que el disco no hierva entre fotogramas.
    """
    ia = np.floor(a).astype(np.int64)
    ib = np.floor(b).astype(np.int64)
    fa = (a - ia).astype(np.float32)
    fb = (b - ib).astype(np.float32)
    # smoothstep: derivada nula en los nodos, si no se ve la retícula del hash
    ua = fa * fa * (3.0 - 2.0 * fa)
    ub = fb * fb * (3.0 - 2.0 * fb)

    def h(x, y):
        n = (x * 73856093) ^ (y * 19349663)
        n = (n ^ (n >> 13)) * 1274126177
        return ((n ^ (n >> 16)) & 0x7FFFFFFF).astype(np.float32) / 2147483647.0

    n00, n10 = h(ia, ib), h(ia + 1, ib)
    n01, n11 = h(ia, ib + 1), h(ia + 1, ib + 1)
    return (n00 + (n10 - n00) * ua) * (1.0 - ub) + (n01 + (n11 - n01) * ua) * ub


def disk_turbulence(x, theta, amount: float = 0.85, octaves: int = 5):
    """Grumos y filamentos en la emision del disco.

    El perfil de Shakura-Sunyaev es una funcion suave, y un disco real no lo es:
    es plasma turbulento. Se modula el brillo con ruido de varias escalas a lo
    largo de una coordenada que sigue el cizallamiento kepleriano (el interior
    gira mas rapido que el exterior), lo que produce filamentos en espiral en
    vez de manchas redondas.

    El ruido es fBm (suma de octavas de ruido de valor). Antes esto se hacia
    multiplicando senos y cosenos, pero un producto de sinusoides es un patron
    de INTERFERENCIA regular: se veian bandas concentricas limpias en vez de
    turbulencia. El ruido con hash no tiene periodo, asi que no hay bandas.
    """
    lx = np.log(np.maximum(x, 1e-6)).astype(np.float32)
    s = (theta + 5.0 * lx).astype(np.float32)   # sigue el cizallamiento

    v = np.zeros(np.shape(x), np.float32)
    amp, freq, norm = 1.0, 1.0, 0.0
    for _ in range(octaves):
        # anisotropia 1:4 -> los grumos salen ESTIRADOS a lo largo del flujo,
        # como filamentos, en vez de manchas redondas
        v += amp * _value_noise(s * freq * 2.0, lx * freq * 8.0)
        norm += amp
        amp *= 0.5
        freq *= 2.0
    v /= norm                                   # queda en [0, 1)

    # ridged: |2v-1| concentra el contraste en crestas finas, que es lo que da
    # aspecto de filamento y no de nube difusa
    ridge = 1.0 - np.abs(2.0 * v - 1.0)
    return np.clip(1.0 + amount * (1.5 * ridge - 0.75), 0.08, None)


def bloom(img, threshold: float = 0.8, sigmas=(3.0, 9.0, 26.0),
          weights=(0.35, 0.22, 0.14)):
    """Resplandor alrededor de las zonas muy brillantes.

    No es fisica del agujero negro: es como responden las lentes y los sensores
    reales ante una fuente intensa. Sin esto el plasma parece pintura mate en vez
    de algo incandescente.
    """
    from scipy.ndimage import gaussian_filter
    hot = np.maximum(img - threshold, 0.0)
    out = img.copy()
    for s, w in zip(sigmas, weights):
        out += w * gaussian_filter(hot, (s, s, 0))
    return out


def tonemap(img, exposure: float = 1.0, gamma: float = 2.2, mode: str = "aces"):
    """HDR -> imagen visible.

    ACES conserva el color en las zonas quemadas; Reinhard las lava hacia el
    blanco. Es la diferencia entre una foto revelada y una sobreexpuesta.
    """
    x = np.asarray(img, float) * exposure
    if mode == "aces":
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        y = (x * (a * x + b)) / (x * (c * x + d) + e)
    else:
        y = x / (1.0 + x)
    return np.clip(y, 0.0, 1.0) ** (1.0 / gamma)
