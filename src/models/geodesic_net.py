"""Red que sustituye al integrador: (beta, phi) -> u_tilde.

Que aprende
-----------
La misma funcion que devuelve `physics.integrator`: dado el parametro de
impacto y el angulo recorrido, a que distancia esta el foton. Con `u = r_s/r`,
asi que u=0 es el infinito y u=1 el horizonte.

Por que no se le meten (beta, phi) crudos
-----------------------------------------
Porque la fisica no es suave en beta: la deflexion DIVERGE logaritmicamente al
acercarse a beta_crit = 3*sqrt(3)/2. Un rayo con beta = 2.59808 da mas de dos
vueltas y uno con beta = 3 apenas se desvia. Una red con beta lineal gastaria
casi toda su capacidad en la zona aburrida y resolveria mal justo el anillo de
fotones, que es lo que mas se ve.

La variable natural es la DISTANCIA LOGARITMICA al critico. No es un truco:
`generate_data.py` ya muestrea asi (s_min=-8, s_max=-1 en el config), de modo
que el dataset esta repartido de forma uniforme en esta coordenada y no en beta.

Entradas (7):
    s       = log10(|beta - beta_crit|), reescalado      <- resuelve el critico
    lado    = signo(beta - beta_crit)                    <- escapa o cae
    beta    = reescalado                                 <- ancla el campo lejano
    |phi|   = reescalado por el maximo global
    u_p     = periastro, en forma cerrada                <- la escala del rayo
    phi_rel = |phi| / beta                               <- ver abajo
    phi_log = log10(1 + |phi|)                           <- ver abajo

Se usa |phi| y no phi porque la trayectoria es SIMETRICA respecto al periastro
(comprobado: |u(phi) - u(-phi)| < 1.6e-4 sobre 303 rayos). Eso reduce el dominio
a la mitad sin perder nada.

Las dos ultimas entradas no estaban al principio y se añadieron tras medir
donde fallaba: cada rayo barre un rango de phi completamente distinto (uno de
captura profunda recorre +-0.039 y uno pegado al critico +-18), asi que con una
sola normalizacion GLOBAL los rayos casi radiales quedaban comprimidos en el
0.2% del rango de entrada y la red no podia separar sus puntos.

El lado (escapa/cae) se le pasa como dato en vez de hacerselo aprender, porque
beta_crit se conoce en forma CERRADA y es exacto. Aprenderlo solo introduciria
error en el borde de la sombra, que es la parte mas visible de la imagen.

Activacion tanh y no ReLU: mas adelante hace falta du/dphi y dPhi/dbeta por
diferenciacion automatica (starfield.py las necesita para la magnificacion, y
con RK45 solo salen por diferencias finitas ruidosas). ReLU daria derivadas
constantes a trozos y discontinuas.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

BETA_CRIT = 3.0 * np.sqrt(3.0) / 2.0        # 2.5980762...
S_FLOOR = 1e-9                              # evita log10(0) justo en el critico


def u_periastro(beta, xp=np):
    """u_tilde en el periastro, en forma CERRADA. Solo vale si beta > beta_crit.

    De la primera integral (du/dphi)^2 = 1/beta^2 - u^2 (1-u), el periastro es
    donde la derivada se anula, o sea la raiz de

        u^2 (1 - u) = 1 / beta^2

    Es una cubica con tres raices reales; la del periastro es la rama k=1 de la
    solucion trigonometrica. Comprobado contra np.roots: da 2/3 exacto en
    beta_crit (la esfera de fotones, r = 1.5 r_s) y tiende a 1/beta en campo
    lejano, que es la recta de parametro de impacto beta.
    """
    c = 1.0 / (beta * beta)
    arg = xp.clip(1.0 - 13.5 * c, -1.0, 1.0)
    return 1.0 / 3.0 + (2.0 / 3.0) * xp.cos(
        xp.arccos(arg) / 3.0 - 2.0 * np.pi / 3.0)


def escala_u(beta, xp=np):
    """El maximo que puede alcanzar u_tilde en ese rayo. Se conoce EXACTO.

    - Si escapa (beta > beta_crit): llega hasta el periastro y vuelve.
    - Si cae (beta < beta_crit): no hay periastro, se hunde hasta el horizonte,
      donde u_tilde = 1 por definicion.

    Sirve para que la red prediga u/escala en vez de u. Sin esto, un rayo
    lejano con beta=15 tiene todos sus valores por debajo de u=0.07, y con una
    perdida en u absoluta esa region pesa nada: la red la resolvia fatal (era
    donde estaba el peor error, no cerca del critico como parecia razonable).
    Normalizando, la salida es de orden 1 en todo el dominio.
    """
    escapa = beta > BETA_CRIT
    up = u_periastro(xp.clip(beta, BETA_CRIT + 1e-12, None), xp=xp)
    return xp.where(escapa, up, xp.ones_like(beta))


class Normalizador:
    """Guarda las escalas de las entradas junto al modelo.

    Va aparte y se serializa con los pesos porque si al inferir se normaliza
    distinto que al entrenar, la red da resultados plausibles pero equivocados,
    y no salta ningun error.
    """

    def __init__(self, s_min=-9.0, s_max=1.2, beta_max=15.0, phi_max=22.0):
        self.s_min, self.s_max = float(s_min), float(s_max)
        self.beta_max, self.phi_max = float(beta_max), float(phi_max)

    def features(self, beta, phi, xp=np):
        """(beta, phi) -> las 7 entradas de la red, en [-1, 1] aprox.

        La resta beta - beta_crit se hace en DOBLE PRECISION a proposito. En
        float32 el ULP cerca de 2.598 vale 2.4e-7, o sea del mismo orden que la
        distancia al critico de los rayos mas interesantes: medido sobre el
        dataset, f32 colapsa 576 valores distintos de |beta-beta_crit| en solo
        22, deja 10112 muestras en CERO exacto, y mete hasta 2.34 de error en s
        (que va de -7 a 1). Toda la resolucion del anillo de fotones se perdia
        ahi, y la red no tenia forma de recuperarla.
        """
        if xp is np:
            b64 = np.asarray(beta, dtype=np.float64)
            d = b64 - BETA_CRIT
        else:
            d = beta.double() - BETA_CRIT
        s = xp.log10(xp.abs(d) + S_FLOOR)
        s_n = 2.0 * (s - self.s_min) / (self.s_max - self.s_min) - 1.0
        lado = xp.sign(d)
        if xp is not np:                       # volver a la precision de trabajo
            s_n, lado = s_n.float(), lado.float()
        b_n = 2.0 * beta / self.beta_max - 1.0
        ap = xp.abs(phi)
        p_n = ap / self.phi_max
        # el periastro, que fija la ESCALA de la trayectoria y es exacto
        up = escala_u(beta, xp=xp)

        # phi en escala RELATIVA al rayo, no global. Cada rayo barre un rango de
        # phi distintisimo: uno de captura profunda (beta=0.039) recorre +-0.039
        # y uno pegado al critico +-18. Normalizando por un maximo global, el
        # primero queda comprimido en el 0.2% del rango de entrada y la red no
        # puede distinguir sus puntos. Para captura casi radial dphi ~ beta
        # (comprobado: beta=0.039 barre exactamente 0.039), asi que phi/beta es
        # la escala natural de ese extremo.
        p_rel = xp.clip(ap / (beta + 1e-3), 0.0, 8.0) / 4.0 - 1.0
        # y en logaritmo, para tener resolucion en los dos extremos a la vez
        p_log = xp.log10(1.0 + ap) / 1.4 - 1.0

        return xp.stack([s_n, lado, b_n, p_n, 2.0 * up - 1.0, p_rel, p_log], -1)

    def estado(self) -> dict:
        return {"s_min": self.s_min, "s_max": self.s_max,
                "beta_max": self.beta_max, "phi_max": self.phi_max}

    @staticmethod
    def desde(estado: dict) -> "Normalizador":
        return Normalizador(**estado)


class GeodesicNet(nn.Module):
    """MLP pequeño. La salida pasa por sigmoide porque u_tilde vive en (0, 1].

    Se prefiere pequeña a proposito: medido en este proyecto, un MLP 64x3 va 3
    veces mas rapido que uno 128x4 en CPU (3.3 vs 1.1 M puntos/s), y el cuello
    de botella del renderer neural sera justamente evaluarla millones de veces.
    """

    N_ENTRADAS = 7

    def __init__(self, ancho: int = 64, capas: int = 3,
                 norm: Normalizador | None = None):
        super().__init__()
        self.norm = norm or Normalizador()
        self.ancho, self.capas = ancho, capas
        seq, d = [], self.N_ENTRADAS
        for _ in range(capas):
            seq += [nn.Linear(d, ancho), nn.Tanh()]
            d = ancho
        seq += [nn.Linear(d, 1)]
        self.red = nn.Sequential(*seq)

    def forward(self, x):
        """Devuelve la RAZON u/escala, no u. Ver escala_u()."""
        return torch.sigmoid(self.red(x)).squeeze(-1)

    def predecir(self, beta, phi) -> torch.Tensor:
        """(beta, phi) crudos -> u_tilde. Reaplica la escala por dentro.

        beta entra en float64: la distancia al critico se pierde en float32
        (ver Normalizador.features).
        """
        beta = torch.as_tensor(beta, dtype=torch.float64)
        phi = torch.as_tensor(phi, dtype=torch.float32)
        razon = self(self.norm.features(beta, phi, xp=torch).float())
        return razon * escala_u(beta, xp=torch).float()

    def guardar(self, ruta, extra: dict | None = None) -> None:
        torch.save({"pesos": self.state_dict(),
                    "ancho": self.ancho, "capas": self.capas,
                    "norm": self.norm.estado(), **(extra or {})}, ruta)

    @staticmethod
    def cargar(ruta, device="cpu") -> "GeodesicNet":
        ck = torch.load(ruta, map_location=device, weights_only=False)
        m = GeodesicNet(ck["ancho"], ck["capas"], Normalizador.desde(ck["norm"]))
        m.load_state_dict(ck["pesos"])
        return m.eval()
