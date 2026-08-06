"""Red para Kerr: (alpha, beta, espin, inclinacion) -> lo que usa el renderer.

Por que NO es la misma red que en Schwarzschild
-----------------------------------------------
En Schwarzschild la red aprende la trayectoria u(phi), y sale a cuenta: con
simetria esferica basta beta para determinarla, y el renderer la muestrea donde
quiera. Aqui eso no compensa por dos motivos.

1. El movimiento es 3D. El arrastre de marcos saca al foton de su plano inicial,
   asi que la trayectoria ya no es una curva u(phi) sino (r(l), theta(l), phi(l)),
   y ademas depende de cuatro parametros, no de uno.

2. El renderer no necesita la trayectoria. Solo usa tres cosas de cada rayo:

       a) si cae al agujero
       b) hacia donde sale (para el fondo de estrellas)
       c) donde cruza el plano ecuatorial (para el disco)

   Reconstruir la geodesica entera para luego tirar el 95% seria pagar de mas.

Asi que esta red predice (b) y (c) DIRECTAMENTE.

(a) no se aprende: el borde de la sombra se conoce en forma cerrada con
KMetric.shadow_outline(). Comprobado sobre 3200 rayos que la captura trazada
coincide con el interior de esa curva sin una sola discrepancia. Aprenderlo
solo meteria error justo en el borde, que es la parte mas visible de la imagen.

Salidas
-------
    direccion : 3 componentes del vector unitario asintotico
    cruces    : (r, phi) de cada cruce con el ecuador, hasta MAX_CRUCES

Los cruces se predicen con el indice k como ENTRADA en vez de sacar 2*MAX
salidas de golpe. Asi la red se entrena solo con los cruces que existen de
verdad (un rayo lejano no cruza el disco ninguna vez, uno pegado al borde lo
cruza seis) y no hay que inventar valores de relleno que la red aprenderia como
si fueran datos.

La direccion se predice como vector de 3 componentes normalizado al final, en
vez de dos angulos. Con angulos habria una discontinuidad artificial en el
meridiano de corte y en los polos, y la red gastaria capacidad en aprender un
salto que no existe en la fisica.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

MAX_CRUCES = 6


class NormalizadorKerr:
    """Escalas de las entradas. Se serializa con los pesos.

    dist es la distancia al borde de la sombra, y entra en LOGARITMO por el
    mismo motivo que en Schwarzschild: toda la estructura interesante (las
    imagenes de orden alto, el anillo de fotones) se acumula pegada a ese borde,
    y el dataset ya se muestrea de forma logaritmica en esa distancia.
    """

    def __init__(self, hw=90.0, d_min=-5.0, d_max=2.3):
        self.hw = float(hw)
        self.d_min, self.d_max = float(d_min), float(d_max)

    def features(self, alpha, beta, spin, inc_deg, dist, xp=np):
        r = xp.sqrt(alpha * alpha + beta * beta)
        ld = xp.log10(dist + 1e-9)
        ld_n = 2.0 * (ld - self.d_min) / (self.d_max - self.d_min) - 1.0
        i = inc_deg * (np.pi / 180.0)
        return xp.stack([
            alpha / self.hw,
            beta / self.hw,
            2.0 * spin - 1.0,                    # espin en [0,1) -> [-1,1)
            xp.cos(i), xp.sin(i),                # la inclinacion, sin salto
            r / self.hw,
            ld_n,
        ], -1)

    def estado(self) -> dict:
        return {"hw": self.hw, "d_min": self.d_min, "d_max": self.d_max}

    @staticmethod
    def desde(e: dict) -> "NormalizadorKerr":
        return NormalizadorKerr(**e)


def base_imagen(theta_obs, xp=np):
    """Base ortonormal del plano imagen: (n_cam, e_alpha, e_beta).

    n_cam apunta del agujero a la camara. e_alpha es perpendicular al eje de
    giro proyectado y e_beta paralelo, siguiendo el convenio de Bardeen.

    El signo de e_beta NO es el obvio. Aqui beta entra como p_theta, y theta
    crece hacia el SUR, asi que beta > 0 corresponde al sur: e_beta apunta en
    sentido contrario al eje de giro proyectado. Comprobado contra rayos
    trazados con espin cero, donde la desviacion respecto a la recta tiene que
    valer 4M/b: con este signo sale 1.16 grados frente a 1.15 teorico, y con el
    contrario sale 21.9.
    """
    n = xp.stack([xp.sin(theta_obs), xp.zeros_like(theta_obs),
                  xp.cos(theta_obs)], -1)
    z = xp.zeros_like(n)
    z[..., 2] = 1.0
    eb = z - (z * n).sum(-1, keepdims=True) * n
    eb = eb / xp.linalg.norm(eb, axis=-1, keepdims=True) if xp is np else \
        eb / eb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    ea = xp.cross(n, eb, axis=-1) if xp is np else torch.cross(n, eb, dim=-1)
    return n, ea, -eb                      # ojo al signo, ver docstring


def direccion_recta(alpha, beta, theta_obs, r_obs, xp=np):
    """Direccion asintotica SIN agujero negro: la recta de parametro b.

    Es la referencia sobre la que la red aprende solo la DESVIACION. Sin esto
    la red tiene que reproducir el mapa entero, que abarca los 180 grados del
    cielo, y gasta casi toda su capacidad en aprender la identidad. Es el mismo
    principio que en Schwarzschild predecir u/u_max en vez de u.

    El angulo con el eje sale de |P x d| = b con P = r_obs n_cam, o sea
    arcsin(b/r_obs); NO arctan, que difiere ya medio grado a b/r_obs ~ 0.3.
    """
    n, ea, eb = base_imagen(theta_obs, xp=xp)
    b = xp.sqrt(alpha * alpha + beta * beta)
    x = xp.arcsin(xp.clip(b / r_obs, -1.0, 1.0))
    bs = xp.clip(b, 1e-12, None) if xp is np else b.clamp_min(1e-12)
    u = (alpha[..., None] * ea + beta[..., None] * eb) / bs[..., None]
    return -xp.cos(x)[..., None] * n + xp.sin(x)[..., None] * u


def marco_tangente(alpha, beta, theta_obs, r_obs, xp=torch):
    """Devuelve (d_recta, t1, t2), una base ortonormal con d_recta como eje.

    t1 es la direccion RADIAL en el plano imagen (hacia o desde el agujero) y
    t2 la tangencial. La deflexion gravitatoria es casi toda radial, asi que
    separandolas la red tiene que aprender una componente grande y suave y otra
    pequeña, en vez de mezclarlas en tres numeros acoplados.
    """
    n, ea, eb = base_imagen(theta_obs, xp=xp)
    b = xp.sqrt(alpha * alpha + beta * beta)
    bs = b.clamp_min(1e-12) if xp is torch else xp.clip(b, 1e-12, None)
    x = xp.arcsin(xp.clip(b / r_obs, -1.0, 1.0))
    u = (alpha[..., None] * ea + beta[..., None] * eb) / bs[..., None]
    v = (-beta[..., None] * ea + alpha[..., None] * eb) / bs[..., None]
    cx, sx = xp.cos(x)[..., None], xp.sin(x)[..., None]
    return (-cx * n + sx * u), (sx * n + cx * u), v


class KerrSkyNet(nn.Module):
    """(alpha, beta, espin, inclinacion) -> direccion asintotica en el cielo.

    Es la mitad cara del fondo estrellado: sin ella hay que integrar una
    geodesica por pixel, que es lo que hace inviable Kerr en tiempo real.

    Predice la DESVIACION respecto a la recta, no la direccion absoluta. El
    mapa absoluto abarca los 180 grados del cielo, asi que una red que lo
    aprenda entero gasta casi toda su capacidad reproduciendo la propagacion
    libre, que ya se conoce en forma cerrada. La desviacion en cambio tiende a
    cero lejos del agujero y solo es grande en el anillo de fotones.

    Es el mismo principio que en Schwarzschild predecir u/u_max en vez de u,
    que ahi valio un factor 128 en el error.

    La salida son dos numeros: la deflexion radial y la tangencial, en el marco
    tangente a la recta. Se aplican y se renormaliza, asi |d| = 1 sale por
    construccion.
    """

    N_ENTRADAS = 7

    def __init__(self, ancho: int = 64, capas: int = 3,
                 norm: NormalizadorKerr | None = None, residual: bool = True):
        super().__init__()
        self.norm = norm or NormalizadorKerr()
        self.ancho, self.capas, self.residual = ancho, capas, bool(residual)
        seq, d = [], self.N_ENTRADAS
        for _ in range(capas):
            seq += [nn.Linear(d, ancho), nn.Tanh()]
            d = ancho
        ultima = nn.Linear(d, 2 if residual else 3)
        if residual:
            # Arrancar en CERO desviacion. Con la inicializacion normal la
            # ultima capa saca numeros de orden 1, mientras que la deflexion
            # real vale ~0.01 rad en casi todo el dominio: la red empieza
            # empujando cien veces de mas y gasta muchas epocas solo en volver.
            # Partiendo de cero, el estado inicial ES la propagacion libre, que
            # ya es la respuesta correcta lejos del agujero.
            nn.init.zeros_(ultima.weight)
            nn.init.zeros_(ultima.bias)
        seq += [ultima]
        self.red = nn.Sequential(*seq)

    def forward(self, x, marco=None):
        v = self.red(x)
        if not self.residual:
            return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        d0, t1, t2 = marco
        d = d0 + v[..., :1] * t1 + v[..., 1:2] * t2
        return d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    def guardar(self, ruta, extra=None):
        torch.save({"pesos": self.state_dict(), "ancho": self.ancho,
                    "capas": self.capas, "norm": self.norm.estado(),
                    "residual": self.residual, **(extra or {})}, ruta)

    @staticmethod
    def cargar(ruta, device="cpu"):
        ck = torch.load(ruta, map_location=device, weights_only=False)
        m = KerrSkyNet(ck["ancho"], ck["capas"],
                       NormalizadorKerr.desde(ck["norm"]),
                       residual=ck.get("residual", False))
        m.load_state_dict(ck["pesos"])
        return m.eval()


class KerrDiskNet(nn.Module):
    """(alpha, beta, espin, inclinacion, k) -> (r, phi) del cruce k-esimo.

    El indice del cruce entra como una entrada mas, con codificacion
    sinusoidal, de modo que una sola red sirve para todos los ordenes.

    Devuelve (r, [cos phi, sin phi], logit_existe). El angulo sale en seno y
    coseno y no como numero para que no haya un salto artificial al dar la
    vuelta.

    La tercera salida, si el cruce EXISTE, no es un adorno. La primera version
    solo predecia (r, phi) y se entrenaba unicamente con cruces reales; al
    renderizar, los pixeles que no cruzan el disco caian en la extrapolacion de
    la red y devolvian radios plausibles, asi que el disco salia como un borron
    que cubria toda la imagen. Hay que preguntarle a la red si hay cruce, no
    deducirlo de que el radio caiga en rango.
    """

    N_ENTRADAS = 7 + 4          # las de arriba + la codificacion de k

    def __init__(self, ancho: int = 64, capas: int = 4,
                 norm: NormalizadorKerr | None = None, r_max: float = 25.0):
        super().__init__()
        self.norm = norm or NormalizadorKerr()
        self.ancho, self.capas, self.r_max = ancho, capas, float(r_max)
        seq, d = [], self.N_ENTRADAS
        for _ in range(capas):
            seq += [nn.Linear(d, ancho), nn.Tanh()]
            d = ancho
        seq += [nn.Linear(d, 4)]        # r, cos, sin, logit(existe)
        self.red = nn.Sequential(*seq)

    @staticmethod
    def codificar_k(k, xp=torch):
        """k -> 4 numeros suaves. Ver el docstring de la clase."""
        k = k.float() if xp is torch else k.astype(np.float64)
        return xp.stack([k / MAX_CRUCES,
                         xp.cos(np.pi * k / MAX_CRUCES),
                         xp.sin(np.pi * k / MAX_CRUCES),
                         xp.cos(2 * np.pi * k / MAX_CRUCES)], -1)

    def forward(self, x):
        """Devuelve (r, [cos phi, sin phi], logit_existe).

        El logit se saca crudo, sin sigmoide, para poder usar la perdida
        estable con logits al entrenar. Al renderizar basta con logit > 0.
        """
        y = self.red(x)
        r = torch.sigmoid(y[..., 0]) * self.r_max
        ang = y[..., 1:3]
        ang = ang / ang.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return r, ang, y[..., 3]

    def guardar(self, ruta, extra=None):
        torch.save({"pesos": self.state_dict(), "ancho": self.ancho,
                    "capas": self.capas, "r_max": self.r_max,
                    "norm": self.norm.estado(), **(extra or {})}, ruta)

    @staticmethod
    def cargar(ruta, device="cpu"):
        ck = torch.load(ruta, map_location=device, weights_only=False)
        m = KerrDiskNet(ck["ancho"], ck["capas"],
                        NormalizadorKerr.desde(ck["norm"]), ck["r_max"])
        m.load_state_dict(ck["pesos"])
        return m.eval()
