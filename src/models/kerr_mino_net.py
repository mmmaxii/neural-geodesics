"""Redes que sustituyen al integrador de Kerr: resuelven la EDO, no el pixel.

Dos redes, porque las EDOs estan desacopladas (ver src/physics/kerr_mino.py):

    KerrThetaNet(x, u_mas, m_tilde)  -> cos(theta)/sqrt(u_mas), ratio_G
    KerrRNet(features 9D)            -> u/escala, Phi_r residual comprimido

Ninguna de las dos ve la camara (r_obs, theta_obs) ni el encuadre: esas
cantidades solo entran en la ARITMETICA DE FASE (exacta, en kerr_mino.py) que
convierte una posicion de pixel en (x_cam, lambda_cam) antes de llamar a la
red, y que reconstruye (r, theta, phi) despues. Es el mismo patron que
GeodesicNet en Schwarzschild: la red aprende u(phi), no imagen->imagen.

KerrThetaNet
------------
Por el colapso demostrado en validate_kerr_mino.py (test 6): la forma
mu(lambda)/sqrt(u_mas) depende SOLO de (x, m) con x = lambda/Lam_theta. Con
u_mas y m_tilde = m/(m-1) in [0,1) como entradas extra, 3 numeros bastan.
Dominio reducido a x in [0, 1/2] por la paridad de mu; el resto se pliega
fuera de la red (ver kerr_mino.plegar_x).

KerrRNet
--------
Analogo directo del Normalizador de Schwarzschild: la feature critica es
s = log10(delta_gap), SIEMPRE en float64 antes de tomar el logaritmo (mismo
bug de f32 que costo 128x en MSE en Schwarzschild -- aqui delta_gap hace el
papel de |beta - beta_crit|). Salida = razon u/escala (sigmoide, vale 1 en el
anclaje) + el residuo secular de Phi_r comprimido con asinh.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

S_FLOOR = 1e-12


# =========================================================================== theta
class NormalizadorTheta:
    """No hace falta ajustar nada a un dataset: x, u_mas y m_tilde ya viven
    en rangos fijos y conocidos por construccion (ver kerr_mino.py)."""

    def features(self, x2, u_mas, m_tilde, xp=np):
        """x2 = 2x in [0,1] (x ya reducido a [0, 1/2] fuera de la red)."""
        return xp.stack([2.0 * x2 - 1.0, 2.0 * u_mas - 1.0, 2.0 * m_tilde - 1.0], -1)

    def estado(self) -> dict:
        return {}

    @staticmethod
    def desde(estado: dict) -> "NormalizadorTheta":
        return NormalizadorTheta()


class KerrThetaNet(nn.Module):
    """MLP 3 -> 64x3 tanh -> 2. Salidas: cos(theta)/sqrt(u_mas), ratio_G."""

    N_ENTRADAS = 3

    def __init__(self, ancho: int = 64, capas: int = 3,
                norm: NormalizadorTheta | None = None):
        super().__init__()
        self.norm = norm or NormalizadorTheta()
        self.ancho, self.capas = ancho, capas
        seq, d = [], self.N_ENTRADAS
        for _ in range(capas):
            seq += [nn.Linear(d, ancho), nn.Tanh()]
            d = ancho
        seq += [nn.Linear(d, 2)]
        self.red = nn.Sequential(*seq)

    def forward(self, x):
        """Devuelve (y_theta, ratio_g) crudos, sin aplicar activacion final.

        y_theta = tanh(...) porque cos(theta)/sqrt(u_mas) in [-1, 1].
        ratio_g = sigmoid(...) porque G_phi(x)/G_phi(1/2) es monotono en [0,1].
        """
        out = self.red(x)
        y_theta = torch.tanh(out[..., 0])
        ratio_g = torch.sigmoid(out[..., 1])
        return y_theta, ratio_g

    def predecir(self, x, u_mas, m, xp=torch):
        """x, u_mas, m crudos (ya reducidos a x in [0, 1/2]) -> (mu, ratio_g).

        Es responsabilidad del llamante haber plegado x y aplicado la
        paridad; esta funcion solo evalua la red en el dominio fundamental.
        """
        x = torch.as_tensor(x, dtype=torch.float32)
        u_mas = torch.as_tensor(u_mas, dtype=torch.float32)
        m_tilde = torch.as_tensor(m / (m - 1.0), dtype=torch.float32)
        feats = self.norm.features(2.0 * x, u_mas, m_tilde, xp=torch)
        y_theta, ratio_g = self(feats)
        return y_theta * torch.sqrt(torch.clamp(u_mas, min=0.0)), ratio_g

    def guardar(self, ruta, extra: dict | None = None) -> None:
        torch.save({"pesos": self.state_dict(), "ancho": self.ancho,
                    "capas": self.capas, "norm": self.norm.estado(),
                    **(extra or {})}, ruta)

    @staticmethod
    def cargar(ruta, device="cpu") -> "KerrThetaNet":
        ck = torch.load(ruta, map_location=device, weights_only=False)
        m = KerrThetaNet(ck["ancho"], ck["capas"], NormalizadorTheta.desde(ck["norm"]))
        m.load_state_dict(ck["pesos"])
        return m.eval()


# =============================================================================== r
class NormalizadorR:
    """Escalas de las 9 entradas de KerrRNet, ajustadas a los extremos del dataset.

    Va serializado con los pesos por la misma razon que en Schwarzschild: si
    se normaliza distinto al inferir que al entrenar, el error es silencioso.
    """

    def __init__(self, s_min=-9.0, s_max=1.0, xi_max=20.0, eta_max=900.0,
                lam_hat_max=1.0, log_lam_max=4.0, asinh_max=50.0):
        self.s_min, self.s_max = float(s_min), float(s_max)
        self.xi_max, self.eta_max = float(xi_max), float(eta_max)
        self.lam_hat_max = float(lam_hat_max)
        self.log_lam_max = float(log_lam_max)
        self.asinh_max = float(asinh_max)

    def features(self, delta_gap, lado, xi, eta, a, u_escala, u_pl, lam,
                lambda_inf, xp=np):
        """Todas las entradas en float64 hasta el resultado final.

        delta_gap hace aqui el papel de |beta - beta_crit|: la resta que lo
        forma (r4 - r3, dentro de kerr_mino.raices_radiales) ya se calcula en
        f64 en la capa fisica, asi que solo hace falta que el LOG lo siga
        siendo -- ese es el paso que en Schwarzschild se perdia en f32.
        """
        if xp is np:
            dg = np.asarray(delta_gap, dtype=np.float64)
        else:
            dg = delta_gap.double()
        s = xp.log10(dg + S_FLOOR)
        s_n = 2.0 * (xp.clip(s, self.s_min, self.s_max) - self.s_min) \
            / (self.s_max - self.s_min) - 1.0

        xi_n = xp.clip(xi, -self.xi_max, self.xi_max) / self.xi_max
        eta_n = 2.0 * xp.sqrt(xp.clip(eta, 0.0, self.eta_max)) \
            / (self.eta_max ** 0.5) - 1.0
        a_n = 2.0 * a - 1.0
        lam_hat = xp.clip(lam / xp.clip(lambda_inf, 1e-12, None), 0.0,
                          self.lam_hat_max)
        lam_hat_n = 2.0 * lam_hat - 1.0
        loglam_n = 2.0 * xp.clip(xp.log10(1.0 + lam), 0.0, self.log_lam_max) \
            / self.log_lam_max - 1.0

        if xp is not np:
            s_n, lado_t = s_n.float(), lado.float()
        else:
            lado_t = lado

        feats = xp.stack([
            s_n, lado_t, xi_n, eta_n, a_n,
            2.0 * u_escala - 1.0, 2.0 * u_pl - 1.0,
            lam_hat_n, loglam_n,
        ], -1)
        return feats.float() if xp is not np else feats

    def comprimir_phi(self, residuo, xp=np):
        return xp.arcsinh(residuo) / self.asinh_max

    def expandir_phi(self, y, xp=torch):
        return xp.sinh(y * self.asinh_max)

    def estado(self) -> dict:
        return {"s_min": self.s_min, "s_max": self.s_max, "xi_max": self.xi_max,
                "eta_max": self.eta_max, "lam_hat_max": self.lam_hat_max,
                "log_lam_max": self.log_lam_max, "asinh_max": self.asinh_max}

    @staticmethod
    def desde(estado: dict) -> "NormalizadorR":
        return NormalizadorR(**estado)


class KerrRNet(nn.Module):
    """MLP 9 -> 96x4 tanh -> 2. Salidas: razon u/escala, Phi_r comprimido."""

    N_ENTRADAS = 9

    def __init__(self, ancho: int = 96, capas: int = 4,
                norm: NormalizadorR | None = None):
        super().__init__()
        self.norm = norm or NormalizadorR()
        self.ancho, self.capas = ancho, capas
        seq, d = [], self.N_ENTRADAS
        for _ in range(capas):
            seq += [nn.Linear(d, ancho), nn.Tanh()]
            d = ancho
        seq += [nn.Linear(d, 2)]
        self.red = nn.Sequential(*seq)

    def forward(self, x):
        """Devuelve (ratio_u, y_phi) crudos.

        ratio_u = sigmoid(...): vale 1 en el anclaje (lambda=0), como el
        periastro de Schwarzschild, y decrece hacia 0 lejos de el.
        y_phi es LINEAL: es asinh(residuo)/A, ya no vive en un rango fijo.
        """
        out = self.red(x)
        ratio_u = torch.sigmoid(out[..., 0])
        y_phi = out[..., 1]
        return ratio_u, y_phi

    def guardar(self, ruta, extra: dict | None = None) -> None:
        torch.save({"pesos": self.state_dict(), "ancho": self.ancho,
                    "capas": self.capas, "norm": self.norm.estado(),
                    **(extra or {})}, ruta)

    @staticmethod
    def cargar(ruta, device="cpu") -> "KerrRNet":
        ck = torch.load(ruta, map_location=device, weights_only=False)
        m = KerrRNet(ck["ancho"], ck["capas"], NormalizadorR.desde(ck["norm"]))
        m.load_state_dict(ck["pesos"])
        return m.eval()
