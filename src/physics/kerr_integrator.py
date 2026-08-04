"""Integrador de geodesicas nulas en Kerr, en formulacion hamiltoniana.

Por que hamiltoniana y no las ecuaciones separadas
--------------------------------------------------
Las ecuaciones de primer orden con la constante de Carter son elegantes:

    Sigma dr/dl = +- sqrt(R(r)),   Sigma dtheta/dl = +- sqrt(Theta(theta))

pero esas raices cuadradas tienen un signo que hay que VOLTEAR A MANO cada vez
que el rayo pasa por un punto de retorno (periastro, o el maximo/minimo de
theta). Detectar esos puntos y no equivocarse de rama es la fuente clasica de
errores en trazadores de Kerr, y falla justo donde mas importa: cerca del
anillo de fotones, donde el rayo da varias vueltas y cruza muchos retornos.

La forma hamiltoniana no tiene raices:

    dx^mu/dl  = g^{mu nu} p_nu
    dp_mu/dl  = -1/2 (d_mu g^{alpha beta}) p_alpha p_beta

Los puntos de retorno salen solos, porque p_r simplemente pasa por cero y
cambia de signo de forma continua. A cambio hay que integrar 5 EDO en vez de
2 y evaluar derivadas de la metrica inversa.

Como la metrica es ESTACIONARIA (no depende de t) y AXISIMETRICA (no depende
de phi), p_t y p_phi son constantes de movimiento y no hay que integrarlas:

    p_t = -E   (energia),    p_phi = L_z  (momento angular sobre el eje)

asi que el estado es solo (r, theta, phi, p_r, p_theta).

Comprobacion interna
--------------------
Para geodesicas nulas el hamiltoniano H = 1/2 g^{ab} p_a p_b vale CERO en todo
momento. Como no se impone en ningun sitio, sirve de test independiente: si al
final del rayo H se ha alejado de cero, la integracion se degrado.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from physics.kerr import KMetric


class KerrGeodesicIntegrator:
    def __init__(self, metric: KMetric, h_deriv: float = 1e-6):
        self.k = metric
        self.h = h_deriv          # paso de la derivada numerica de g^{ab}

    # ------------------------------------------------------------------ RHS
    def _rhs(self, lam, y, E, L):
        """Lado derecho hamiltoniano, escrito para ser barato.

        Todo en floats de Python y con las derivadas analiticas de la metrica
        inversa: no se construye ninguna matriz y no se llama a numpy. La
        version anterior sacaba las derivadas por diferencias centradas, o sea
        4 evaluaciones extra de la metrica por paso, y pasaba por arrays 4x4;
        el perfil mostraba que ahi se iba practicamente todo el tiempo.

        El contraido g^ab p_a p_b se escribe a mano aprovechando que la metrica
        de Kerr solo tiene 5 componentes no nulas y que el bloque cruzado es
        (t, phi).
        """
        r, th, _ph, pr, pth = y
        comp, dr_, dt_ = self.k.inverse_components_and_derivs(r, th)
        gtt, gtp, grr, gthth, gpp = comp

        pt = -E                                 # p_t constante (Killing d_t)
        # dx^mu/dl = g^{mu nu} p_nu
        drdl = grr * pr
        dthdl = gthth * pth
        dphdl = gtp * pt + gpp * L

        # dp_mu/dl = -1/2 (d_mu g^{ab}) p_a p_b
        pt2, pl2, pr2, pth2, ptl = pt * pt, L * L, pr * pr, pth * pth, 2.0 * pt * L
        dpr = -0.5 * (dr_[0] * pt2 + dr_[1] * ptl + dr_[2] * pr2
                      + dr_[3] * pth2 + dr_[4] * pl2)
        dpth = -0.5 * (dt_[0] * pt2 + dt_[1] * ptl + dt_[2] * pr2
                       + dt_[3] * pth2 + dt_[4] * pl2)
        return [drdl, dthdl, dphdl, dpr, dpth]

    def hamiltonian(self, r, th, pr, pth, E, L) -> float:
        """H = 1/2 g^{ab} p_a p_b. Debe ser 0 para un rayo de luz."""
        p = np.array([-E, pr, pth, L])
        return float(0.5 * p @ self.k.inverse_metric_matrix(r, th) @ p)

    # ------------------------------------------------ condiciones iniciales
    def initial_from_celestial(self, alpha: float, beta: float,
                               r_obs: float, theta_obs: float):
        """Rayo que llega al pixel (alpha, beta) del observador lejano.

        (alpha, beta) son las coordenadas celestes de Bardeen, en unidades de
        longitud: alpha sobre el eje perpendicular al de rotacion proyectado y
        beta paralelo a el. Son las MISMAS que usa KMetric.shadow_outline(),
        asi que la sombra analitica y la trazada tienen que coincidir.

        Se normaliza E = 1 (solo fija la escala del parametro afin).
        """
        st, ct = np.sin(theta_obs), np.cos(theta_obs)
        E = 1.0
        L = -alpha * st                               # xi = L/E
        pth = beta                                    # p_theta^2 = Theta(theta_o)

        # p_r sale de imponer que el rayo sea NULO, no de una formula aparte:
        #   g^tt pt^2 + 2 g^tphi pt pphi + g^phiphi pphi^2 + g^rr pr^2
        #   + g^thth pth^2 = 0
        gi = self.k.inverse_metric_matrix(r_obs, theta_obs)
        rest = (gi[0, 0] * E**2 - 2.0 * gi[0, 3] * E * L
                + gi[3, 3] * L**2 + gi[2, 2] * pth**2)
        pr2 = -rest / gi[1, 1]
        # raiz negativa: el rayo viaja HACIA el agujero (trazado hacia atras)
        pr = -np.sqrt(max(pr2, 0.0))
        return np.array([r_obs, theta_obs, 0.0, pr, pth]), E, L

    # ------------------------------------------------------------ integracion
    def trace(self, alpha: float, beta: float, r_obs: float,
              theta_obs: float, r_escape: float | None = None,
              rtol: float = 1e-8, atol: float = 1e-10,
              lam_max: float = 1e4) -> dict:
        """Traza un rayo hacia atras desde el observador.

        Devuelve status 'captured' (cae al horizonte), 'escaped' (se va al
        infinito) o 'maxsteps'. Si escapa, incluye la direccion asintotica en
        el cielo, ya en cartesianas.
        """
        y0, E, L = self.initial_from_celestial(alpha, beta, r_obs, theta_obs)
        r_esc = r_escape if r_escape is not None else 1.05 * r_obs
        # margen sobre el horizonte: en r+ exacto la metrica BL es singular
        r_cap = self.k.r_horizon * 1.0001

        hit = lambda l, y, *a: y[0] - r_cap
        hit.terminal = True
        hit.direction = -1
        out = lambda l, y, *a: y[0] - r_esc
        out.terminal = True
        out.direction = 1

        sol = solve_ivp(self._rhs, (0.0, lam_max), y0, args=(E, L),
                        events=[hit, out], rtol=rtol, atol=atol,
                        method="DOP853", dense_output=False)

        r, th, ph, pr, pth = sol.y[:, -1]
        if len(sol.t_events[0]):
            status = "captured"
        elif len(sol.t_events[1]):
            status = "escaped"
        else:
            status = "maxsteps"

        res = {"status": status, "r": r, "theta": th, "phi": ph,
               "H": self.hamiltonian(r, th, pr, pth, E, L),
               "n_steps": sol.t.size, "E": E, "L": L}

        if status == "escaped":
            res["direction"] = self._sky_direction(r, th, ph, pr, pth, E, L)
        return res

    # ============================================================== por lotes
    # Coeficientes de Dormand-Prince (los mismos que usa RK45 de scipy).
    _C = np.array([0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0])
    _A = [np.array([]),
          np.array([1/5]),
          np.array([3/40, 9/40]),
          np.array([44/45, -56/15, 32/9]),
          np.array([19372/6561, -25360/2187, 64448/6561, -212/729]),
          np.array([9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]),
          np.array([35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84])]
    _B = np.array([35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0])
    _E = _B - np.array([5179/57600, 0.0, 7571/16695, 393/640,
                        -92097/339200, 187/2100, 1/40])

    def _rhs_vec(self, y, E, L):
        """Lado derecho para N rayos a la vez. y tiene forma (N, 5)."""
        r, th, pr, pth = y[:, 0], y[:, 1], y[:, 3], y[:, 4]
        comp, dr_, dt_ = self.k.inverse_components_and_derivs_vec(r, th)
        pt = -E
        pt2, pl2, pr2, pth2, ptl = pt * pt, L * L, pr * pr, pth * pth, 2.0 * pt * L
        out = np.empty_like(y)
        out[:, 0] = comp[2] * pr                       # dr
        out[:, 1] = comp[3] * pth                      # dtheta
        out[:, 2] = comp[1] * pt + comp[4] * L         # dphi
        out[:, 3] = -0.5 * (dr_[0]*pt2 + dr_[1]*ptl + dr_[2]*pr2
                            + dr_[3]*pth2 + dr_[4]*pl2)
        out[:, 4] = -0.5 * (dt_[0]*pt2 + dt_[1]*ptl + dt_[2]*pr2
                            + dt_[3]*pth2 + dt_[4]*pl2)
        return out

    def trace_batch(self, alphas, betas, r_obs: float, theta_obs: float,
                    rtol: float = 1e-6, atol: float = 1e-8,
                    max_steps: int = 20000, progress=None,
                    disk_in: float | None = None, disk_out: float | None = None,
                    max_crossings: int = 6) -> dict:
        """Traza N rayos SIMULTANEAMENTE. Devuelve direcciones y capturas.

        Cada rayo lleva su PROPIO paso adaptativo: el control de error es por
        rayo, igual que en solve_ivp, pero el trabajo se hace en operaciones
        de numpy sobre los N rayos vivos a la vez. Asi el coste por rayo del
        interprete de Python se reparte entre todos, que es de donde sale casi
        toda la ganancia frente a llamar solve_ivp N veces.

        Los rayos que terminan (caen al horizonte o escapan) se van sacando del
        lote, de forma que los pasos siguientes solo mueven a los que quedan.

        Con disk_in/disk_out se anotan ademas los cruces por el plano
        ecuatorial (theta = pi/2) cuyo radio caiga dentro del disco. Se guardan
        HASTA max_crossings por rayo: el disco es opticamente fino, asi que la
        luz lo atraviesa varias veces y las emisiones se suman. Ahi es donde
        nace el anillo de fotones, en los cruces de orden alto.

        El punto exacto del cruce se interpola linealmente dentro del paso. El
        paso adaptativo ya es pequeno donde la trayectoria se curva, asi que el
        error de esa interpolacion queda muy por debajo de un pixel.
        """
        alphas = np.atleast_1d(np.asarray(alphas, float))
        betas = np.atleast_1d(np.asarray(betas, float))
        n = alphas.size

        st = np.sin(theta_obs)
        Ec = np.ones(n)
        Lc = -alphas * st
        y = np.zeros((n, 5))
        y[:, 0] = r_obs
        y[:, 1] = theta_obs
        y[:, 4] = betas
        # p_r de la condicion nula, raiz negativa (el rayo entra)
        comp, _, _ = self.k.inverse_components_and_derivs_vec(
            np.full(n, r_obs), np.full(n, theta_obs))
        rest = (comp[0]*Ec**2 - 2.0*comp[1]*Ec*Lc + comp[4]*Lc**2
                + comp[3]*betas**2)
        y[:, 3] = -np.sqrt(np.maximum(-rest / comp[2], 0.0))

        r_esc = 1.05 * r_obs
        r_cap = self.k.r_horizon * 1.0001

        alive = np.arange(n)
        captured = np.zeros(n, bool)
        finished = np.zeros(n, bool)
        dirs = np.zeros((n, 3))
        h = np.full(n, 0.5)                      # paso inicial por rayo

        want_disk = disk_in is not None and disk_out is not None
        if want_disk:
            # float32: con millones de rayos estas tablas dominan la memoria y
            # no hace falta doble precision para sombrear
            cross_r = np.zeros((n, max_crossings), np.float32)
            cross_phi = np.zeros((n, max_crossings), np.float32)
            n_cross = np.zeros(n, np.int8)

        for step in range(max_steps):
            if alive.size == 0:
                break
            ya, ha = y[alive], h[alive]
            Ea, La = Ec[alive], Lc[alive]

            ks = np.empty((7, ya.shape[0], 5))
            ks[0] = self._rhs_vec(ya, Ea, La)
            for i in range(1, 7):
                acc = np.tensordot(self._A[i], ks[:i], axes=(0, 0))
                ks[i] = self._rhs_vec(ya + ha[:, None] * acc, Ea, La)

            y_new = ya + ha[:, None] * np.tensordot(self._B, ks, axes=(0, 0))
            err = ha[:, None] * np.tensordot(self._E, ks, axes=(0, 0))
            scale = atol + rtol * np.maximum(np.abs(ya), np.abs(y_new))
            enorm = np.sqrt(np.mean((err / scale) ** 2, axis=1))

            good = enorm <= 1.0
            # factor de correccion del paso, acotado para no oscilar
            fac = np.clip(0.9 * np.where(enorm > 0, enorm, 1e-10) ** -0.2, 0.2, 5.0)
            ha_new = ha * fac

            idx_good = alive[good]

            # ---- cruces por el plano ecuatorial, ANTES de pisar y[idx_good]
            if want_disk and idx_good.size:
                th_old = ya[good, 1]
                th_new = y_new[good, 1]
                d_old, d_new = th_old - np.pi / 2, th_new - np.pi / 2
                cr = (d_old * d_new) < 0.0            # cambio de signo = cruce
                if cr.any():
                    sel = idx_good[cr]
                    f = d_old[cr] / (d_old[cr] - d_new[cr])   # en [0,1]
                    rc = ya[good, 0][cr] + f * (y_new[good, 0][cr] - ya[good, 0][cr])
                    pc = ya[good, 2][cr] + f * (y_new[good, 2][cr] - ya[good, 2][cr])
                    dentro = (rc >= disk_in) & (rc <= disk_out)
                    hueco = n_cross[sel] < max_crossings
                    m = dentro & hueco
                    if m.any():
                        s2_ = sel[m]
                        k2 = n_cross[s2_]
                        cross_r[s2_, k2] = rc[m]
                        cross_phi[s2_, k2] = pc[m]
                        n_cross[s2_] = k2 + 1

            y[idx_good] = y_new[good]
            h[alive] = ha_new

            # ---- terminaciones, solo entre los que acaban de avanzar
            rg = y[idx_good, 0]
            cap = rg <= r_cap
            esc = rg >= r_esc
            done = cap | esc
            if done.any():
                di = idx_good[done]
                captured[di] = cap[done]
                finished[di] = True
                out = di[esc[done]]
                if out.size:
                    dirs[out] = self._sky_direction_vec(y[out], Ec[out], Lc[out])
            alive = alive[~finished[alive]]
            if progress and step % 200 == 0:
                progress(step, alive.size)

        # los que se quedaron sin pasos se tratan como capturados (negro):
        # es mucho mas seguro que inventarles una direccion de salida
        captured[~finished] = True
        out = {"direction": dirs, "captured": captured,
               "L": Lc, "unfinished": int((~finished).sum())}
        if want_disk:
            out.update(cross_r=cross_r, cross_phi=cross_phi, n_cross=n_cross)
        return out

    def _sky_direction_vec(self, y, E, L) -> np.ndarray:
        """Direccion asintotica para un lote de rayos."""
        r, th, ph, pr, pth = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4]
        comp, _, _ = self.k.inverse_components_and_derivs_vec(r, th)
        dr, dth, dph = comp[2]*pr, comp[3]*pth, comp[1]*(-E) + comp[4]*L
        st, ct = np.sin(th), np.cos(th)
        sp, cp = np.sin(ph), np.cos(ph)
        v = np.stack([
            dr*st*cp + r*dth*ct*cp - r*dph*st*sp,
            dr*st*sp + r*dth*ct*sp + r*dph*st*cp,
            dr*ct - r*dth*st], axis=1)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-30)

    def _sky_direction(self, r, th, ph, pr, pth, E, L) -> np.ndarray:
        """Direccion asintotica del rayo, en cartesianas unitarias.

        Lejos del agujero Boyer-Lindquist tiende a esfericas, asi que se pasa
        la velocidad (dr, dtheta, dphi) a cartesianas y se normaliza. Se usa la
        VELOCIDAD y no la posicion porque a r finito el rayo todavia no ha
        llegado a su direccion final, pero su velocidad ya casi si.
        """
        p = np.array([-E, pr, pth, L])
        dx = self.k.inverse_metric_matrix(r, th) @ p
        dr, dth, dph = dx[1], dx[2], dx[3]
        st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
        v = np.array([
            dr * st * cp + r * dth * ct * cp - r * dph * st * sp,
            dr * st * sp + r * dth * ct * sp + r * dph * st * cp,
            dr * ct - r * dth * st,
        ])
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
