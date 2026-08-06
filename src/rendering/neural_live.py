"""Motor de render neural en tiempo real. Todo en GPU, todo en float32.

Por que existe aparte de render_neural.py
-----------------------------------------
El renderer offline puede permitirse un KD-tree en CPU y calcular las entradas
en doble precision. Para tiempo real no: medido a 640x360, el KD-tree costaba
1017 ms y las features en float64 otros 358, mientras que la red sola tardaba
12 ms. O sea que el 99% del fotograma se iba en preparar datos, no en inferir.

Aqui las tres cosas viven en la GPU y en float32:

    distancia al borde de la sombra : punto-segmento vectorizado
    entradas de la red              : float32
    inferencia                      : float32

La distancia al contorno se calcula contra los SEGMENTOS del poligono, no
contra sus vertices. Con vertices haria falta un contorno finisimo para que la
distancia fuese suave, y aun asi quedaria escalonada; con segmentos, 256 bastan
y el resultado es exacto salvo por la propia poligonizacion.

Ojo con la precision: la GeForce va a 1/64 en float64 (medido: 0.11 frente a
4.39 TFLOPS), asi que cualquier cosa en doble precision aqui mata el
rendimiento. Es seguro porque estas cuentas son geometria suave, no una
integracion adaptativa; la que si necesitaba float64 era el trazado de rayos.
"""
from __future__ import annotations

import numpy as np
import torch

from models.kerr_net import MAX_CRUCES, KerrDiskNet, KerrSkyNet, marco_tangente
from physics.kerr import KMetric


def _cuerpo_negro_gpu(t_kelvin: torch.Tensor) -> torch.Tensor:
    """Color de cuerpo negro, en GPU. Misma aproximacion que blackbody_rgb."""
    t = t_kelvin.clamp(1000.0, 40000.0) / 100.0
    tm = (t - 60.0).clamp_min(1e-9)
    r = torch.where(t <= 66, torch.full_like(t, 255.0), 329.7 * tm ** -0.1332)
    g = torch.where(t <= 66, 99.47 * torch.log(t) - 161.1, 288.1 * tm ** -0.0755)
    b = torch.where(t >= 66, torch.full_like(t, 255.0),
                    torch.where(t <= 19, torch.zeros_like(t),
                                138.5 * torch.log((t - 10).clamp_min(1e-9)) - 305.0))
    return (torch.stack([r, g, b], -1) / 255.0).clamp(0.0, 1.0)


def _troceado(n_pts: int, n_seg: int, tope=32_000_000):
    """Cuantos puntos caben por trozo sin pasarse de memoria.

    Las cuentas punto-segmento son de tamaño N*M, y a 1280x720 con 256
    segmentos eso son 236 M numeros por intermedio, casi 2 GB. En una tarjeta
    de 4 GB se agota enseguida, asi que hay que trocear.
    """
    return max(1, min(n_pts, tope // max(n_seg, 1)))


def distancia_a_poligono(pts: torch.Tensor, poly: torch.Tensor) -> torch.Tensor:
    """Distancia de N puntos a un poligono cerrado de M SEGMENTOS, en GPU.

    Contra segmentos y no contra vertices: con vertices la distancia queda
    escalonada y haria falta un contorno finisimo para suavizarla.

    Sustituye al KD-tree en CPU, que a 640x360 costaba un segundo entero
    mientras que la red tardaba 12 ms.
    """
    a = poly
    b = torch.roll(poly, -1, dims=0)
    ab = b - a                                     # (M, 2)
    denom = (ab * ab).sum(-1).clamp_min(1e-30)     # (M,)
    out = torch.empty(pts.shape[0], device=pts.device, dtype=pts.dtype)
    paso = _troceado(pts.shape[0], poly.shape[0])
    for i in range(0, pts.shape[0], paso):
        p = pts[i:i + paso]
        ap = p[:, None, :] - a[None, :, :]
        t = ((ap * ab[None]).sum(-1) / denom[None]).clamp(0.0, 1.0)
        proy = a[None] + t[..., None] * ab[None]
        out[i:i + paso] = (p[:, None, :] - proy).norm(dim=-1).min(dim=1).values
    return out


def dentro_de_poligono(pts: torch.Tensor, poly: torch.Tensor) -> torch.Tensor:
    """Test de interior por paridad de cruces, troceado igual que la distancia."""
    a = poly
    b = torch.roll(poly, -1, dims=0)
    out = torch.empty(pts.shape[0], device=pts.device, dtype=torch.bool)
    paso = _troceado(pts.shape[0], poly.shape[0])
    for i in range(0, pts.shape[0], paso):
        p = pts[i:i + paso][:, None, :]
        cond = (a[None, :, 1] > p[..., 1]) != (b[None, :, 1] > p[..., 1])
        xint = ((b[None, :, 0] - a[None, :, 0]) * (p[..., 1] - a[None, :, 1])
                / (b[None, :, 1] - a[None, :, 1] + 1e-30) + a[None, :, 0])
        out[i:i + paso] = ((cond & (p[..., 0] < xint)).sum(1) % 2).bool()
    return out


class MotorNeural:
    """Mantiene en GPU todo lo que no cambia, para que el fotograma sea barato.

    La rejilla de pixeles y el cielo se suben una vez. Al mover la camara solo
    hay que rehacer la distancia al borde y las entradas, que en GPU son
    milisegundos.
    """

    def __init__(self, ruta_red, width=640, height=360, half_width=80.0,
                 r_obs=1000.0, device="cuda", sky=None, n_poly=256,
                 ruta_disco=None, r_out=18.0, temp_K=3800.0, norm_r=12.0):
        self.dev = torch.device(device)
        self.net = KerrSkyNet.cargar(ruta_red, device).to(self.dev).eval()
        self.disco = (KerrDiskNet.cargar(ruta_disco, device).to(self.dev).eval()
                      if ruta_disco else None)
        self.r_out, self.temp_K, self.norm_r = float(r_out), float(temp_K), float(norm_r)
        self.W, self.H = int(width), int(height)
        self.hw, self.r_obs = float(half_width), float(r_obs)
        self.n_poly = int(n_poly)

        ax = torch.linspace(-self.hw, self.hw, self.W, device=self.dev)
        ay = torch.linspace(-self.hw, self.hw, self.H, device=self.dev) * (self.H / self.W)
        B, A = torch.meshgrid(ay, ax, indexing="ij")
        self.al, self.be = A.reshape(-1), B.reshape(-1)
        self.pts = torch.stack([self.al, self.be], 1)

        self.sky = None if sky is None else torch.from_numpy(
            np.ascontiguousarray(sky)).float().to(self.dev)
        self._cache = None          # (spin, inc) de lo ultimo preparado

    # ------------------------------------------------------- tabla de camara
    def precalcular(self, spin: float, incs) -> None:
        """Precalcula la geometria para una lista de inclinaciones.

        Mover la camara cambia el contorno de la sombra, y recalcular la
        distancia a ese contorno cuesta unos 100 ms: suficiente para bajar de
        78 a 8 fps. Como solo depende de (espin, inclinacion) y no de nada que
        cambie dentro del fotograma, se puede tener hecha de antemano.

        Se guardan tambien las ENTRADAS de la red y el marco tangente, no solo
        la distancia. Con solo la distancia el visor se quedaba en 21 fps porque
        rehacer las entradas cuesta 35 de los 47 ms del fotograma. Ocupan unos
        15 MB por inclinacion, asi que 80 inclinaciones son 1.2 GB: cabe en una
        tarjeta de 4 GB, pero conviene vigilarlo al subir la resolucion.
        """
        self.tabla_incs = np.asarray(incs, dtype=float)
        self.tabla_spin = float(spin)
        self.tabla = []
        c = lambda v: torch.full_like(self.al, v)
        for i in self.tabla_incs:
            k = KMetric(M=1.0, a=float(spin))
            x, y = k.shadow_outline(float(i), n=self.n_poly)
            poly = torch.from_numpy(np.stack([x, y], 1)).float().to(self.dev)
            dist = distancia_a_poligono(self.pts, poly)
            dentro = dentro_de_poligono(self.pts, poly)
            X = self.net.norm.features(self.al, self.be, c(spin), c(float(i)),
                                       dist, xp=torch)
            marco = marco_tangente(self.al, self.be,
                                   c(float(np.deg2rad(i))), self.r_obs, xp=torch)
            Xd = None
            if self.disco is not None:
                # una entrada por cada orden de cruce; el indice k va DENTRO de
                # las entradas, asi que una sola red sirve para todos
                Xd = [torch.cat([self.disco.norm.features(
                    self.al, self.be, c(spin), c(float(i)), dist, xp=torch),
                    KerrDiskNet.codificar_k(
                        torch.full_like(self.al, float(kk)), xp=torch)], -1)
                    for kk in range(MAX_CRUCES)]
            self.tabla.append((dentro, X, marco, Xd,
                               float(np.deg2rad(i)),
                               KMetric(M=1.0, a=float(spin)).r_isco()))

    def _de_tabla(self, spin: float, inc_deg: float) -> bool:
        """Coge la entrada mas cercana de la tabla, si sirve para este espin."""
        if not hasattr(self, "tabla_incs") or abs(spin - self.tabla_spin) > 1e-9:
            return False
        j = int(np.argmin(np.abs(self.tabla_incs - inc_deg)))
        (self.dentro, self.X, self.marco, self.Xd,
         self.th_obs, self.r_in) = self.tabla[j]
        self._cache = (spin, inc_deg)
        return True

    # ------------------------------------------------------------------ disco
    def _redshift(self, r, xi, spin):
        """g = 1/[u^t (1 - Omega xi)] en el ecuador, en GPU.

        Devuelve 0 donde la combinacion (r, xi) es imposible: ese denominador es
        la energia del foton medida por el gas, y si sale <= 0 ese elemento no
        pudo emitir hacia nosotros. Como el brillo va con g^4, un solo pixel mal
        tratado se lleva toda la escala.
        """
        M, a = 1.0, float(spin)
        om = np.sqrt(M) / (r ** 1.5 + a * np.sqrt(M))
        gtt = -(1.0 - 2.0 * M / r)
        gtp = -2.0 * M * a / r
        gpp = r * r + a * a + 2.0 * M * a * a / r
        norm = -(gtt + 2.0 * gtp * om + gpp * om * om)
        ut = 1.0 / norm.clamp_min(1e-12).sqrt()
        den = ut * (1.0 - om * xi)
        return torch.where((norm > 0) & (den > 1e-6), 1.0 / den.clamp_min(1e-6),
                           torch.zeros_like(den))

    def _perfil(self, x, x_in):
        """Perfil radial de Shakura-Sunyaev, renormalizado en un radio FIJO.

        Fijo y no en su maximo: el maximo esta en (49/36) x_in, asi que al
        cambiar el espin el ISCO lo mueve y el disco exterior parece apagarse,
        cuando en realidad la eficiencia radiativa SUBE con el espin.
        """
        def f(u):
            dentro = (1.0 - torch.sqrt(torch.as_tensor(x_in) / u.clamp_min(1e-9))).clamp_min(0.0)
            return (dentro / u.clamp_min(1e-9) ** 3) ** 0.25
        ref = f(torch.tensor(self.norm_r, device=x.device, dtype=x.dtype))
        return torch.where(x >= x_in, f(x) / ref.clamp_min(1e-9),
                           torch.zeros_like(x))

    # ------------------------------------------------------------ geometria
    def preparar(self, spin: float, inc_deg: float):
        """Recalcula lo que depende de (espin, inclinacion). Milisegundos."""
        if self._cache == (spin, inc_deg):
            return
        if self._de_tabla(spin, inc_deg):
            return
        k = KMetric(M=1.0, a=float(spin))
        x, y = k.shadow_outline(float(inc_deg), n=self.n_poly)
        poly = torch.from_numpy(np.stack([x, y], 1)).float().to(self.dev)
        self.dist = distancia_a_poligono(self.pts, poly)
        self.dentro = dentro_de_poligono(self.pts, poly)

        th = float(np.deg2rad(inc_deg))
        c = lambda v: torch.full_like(self.al, v)
        self.X = self.net.norm.features(self.al, self.be, c(spin),
                                        c(inc_deg), self.dist, xp=torch)
        self.marco = marco_tangente(self.al, self.be, c(th), self.r_obs,
                                    xp=torch)
        self._cache = (spin, inc_deg)

    # -------------------------------------------------------------- fotograma
    @torch.no_grad()
    def fotograma(self, spin: float, inc_deg: float, roll: float = 0.0,
                  brillo: float = 0.5) -> torch.Tensor:
        """Devuelve la imagen (H, W, 3) en luz lineal, sin salir de la GPU."""
        self.preparar(spin, inc_deg)
        d = self.net(self.X, self.marco)
        if roll:
            cr, sr = float(np.cos(roll)), float(np.sin(roll))
            dx, dy = d[:, 0] * cr - d[:, 1] * sr, d[:, 0] * sr + d[:, 1] * cr
            d = torch.stack([dx, dy, d[:, 2]], 1)

        img = torch.zeros(self.al.numel(), 3, device=self.dev)
        if self.sky is not None:
            th = torch.arccos(d[:, 2].clamp(-1, 1))
            lo = torch.atan2(d[:, 1], d[:, 0])
            hgt, wid = self.sky.shape[0], self.sky.shape[1]
            u = (((lo + np.pi) / (2 * np.pi)) % 1.0 * wid).long().clamp(0, wid - 1)
            v = (th / np.pi * (hgt - 1)).long().clamp(0, hgt - 1)
            img = self.sky[v, u] * brillo
        img[self.dentro] = 0.0

        # ---------------------------------------------------------- el disco
        if self.disco is not None and self.Xd is not None:
            xi = -self.al * float(np.sin(self.th_obs))     # L/E del foton
            acum = torch.zeros_like(img)
            for Xk in self.Xd:
                r, ang, logit = self.disco(Xk)
                # Es la RED la que dice si hay cruce, no el radio. Deducirlo de
                # que el radio caiga en rango daba un borron sobre toda la
                # imagen, porque donde no hay disco la red extrapola y devuelve
                # radios plausibles igualmente.
                ok = (logit > 0) & (r >= self.r_in) & (r <= self.r_out)
                if not bool(ok.any()):
                    continue
                g = self._redshift(r, xi, self.tabla_spin)
                prof = self._perfil(r, self.r_in)
                t_obs = (g * self.temp_K * prof).clamp_min(800.0)
                bri = g.clamp_min(0.0) ** 4 * prof ** 4      # g^4 por Liouville
                acum += torch.where(ok[:, None],
                                    _cuerpo_negro_gpu(t_obs) * bri[:, None],
                                    torch.zeros_like(img))
            img = img + acum
            img[self.dentro] = 0.0
        return img.reshape(self.H, self.W, 3)
