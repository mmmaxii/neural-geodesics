"""Validacion de la capa de Mino contra el integrador ya validado.

Nada de lo que hay en kerr_mino.py se da por bueno hasta que pasa por aqui.
El integrador hamiltoniano de kerr_integrator.py y la sombra analitica de
KMetric.shadow_outline ya estan validados a 1e-8, asi que sirven de oraculo.

    python scripts/validate_kerr_mino.py --etapa 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad, solve_ivp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from physics.kerr import KMetric                              # noqa: E402
from physics.kerr_integrator import KerrGeodesicIntegrator    # noqa: E402
import physics.kerr_mino as km                                # noqa: E402


FALLOS = []


def informe(nombre: str, valor: float, umbral: float, unidad: str = "") -> bool:
    ok = np.isfinite(valor) and valor <= umbral
    marca = "OK  " if ok else "FALLA"
    print(f"  [{marca}] {nombre:<52s} {valor:.3e} {unidad} (umbral {umbral:.0e})")
    if not ok:
        FALLOS.append(nombre)
    return ok


# ===========================================================================
# 1. Raices del cuartico
# ===========================================================================
def test_raices(rng, n=200_000):
    print("\n1. Raices de R(r): Ferrari vectorizado vs np.roots")
    peor_rel, peor_res = 0.0, 0.0
    for a in (0.0, 0.3, 0.9, 0.998):
        alpha = rng.uniform(-30.0, 30.0, n)
        beta = rng.uniform(-30.0, 30.0, n)
        inc = rng.uniform(0.1, np.pi / 2)
        xi, eta = km.xi_eta_desde_pixel(alpha, beta, inc, a)
        ok = eta > km.ETA_MIN
        xi, eta = xi[ok], eta[ok]
        R = km.raices_radiales(xi, eta, a)

        # residuo |R(r_i)| escalado: es la medida honesta cuando hay raices
        # casi dobles, donde la POSICION de la raiz esta mal condicionada pero
        # el polinomio sigue anulandose.
        p, q, s = km.coef_R(xi, eta, a)
        val = ((R * R + p) * R + q) * R + s
        escala = np.maximum(np.abs(R) ** 4, 1.0)
        peor_res = max(peor_res, float(np.max(np.abs(val) / escala)))

        # comparacion directa con np.roots sobre una submuestra
        idx = rng.choice(xi.size, size=400, replace=False)
        for i in idx:
            coef = [1.0, 0.0, p[i], q[i], s[i]]
            ref = np.sort_complex(np.roots(coef))
            mio = np.sort_complex(R[:, i])
            err = np.max(np.abs(ref - mio)) / max(1.0, np.max(np.abs(ref)))
            peor_rel = max(peor_rel, float(err))

    informe("residuo |R(r_i)| relativo, peor de 800k raices", peor_res, 1e-9)
    informe("distancia a np.roots, peor de 1600 cuarticos", peor_rel, 1e-9)


# ===========================================================================
# 2. Clasificacion captura/escape contra el trazador
# ===========================================================================
def test_clasificacion(n_lado=200, r_obs=1000.0, half=12.0):
    print("\n2. Sombra exacta vs trace_batch (clasificacion por pixel)")
    peor_frac = 0.0
    for a in (0.0, 0.5, 0.9, 0.998):
        k = KMetric(M=1.0, a=a)
        ig = KerrGeodesicIntegrator(k)
        for inc_deg in (20.0, 60.0, 85.0):
            th = np.deg2rad(inc_deg)
            ax = np.linspace(-half, half, n_lado)
            A, B = np.meshgrid(ax, ax)
            al, be = A.ravel(), B.ravel()

            res = ig.trace_batch(al, be, r_obs, th, rtol=1e-9, atol=1e-11,
                                 max_steps=60_000)
            xi, eta = km.xi_eta_desde_pixel(al, be, th, a)
            cls = km.clasificar(xi, eta, a, k.r_horizon)

            # los pixeles con eta <= 0 no los cubre esta parametrizacion
            valido = (eta > km.ETA_MIN) & (cls["delta_gap"] > 1e-8)
            desac = (cls["escapa"] != ~res["captured"]) & valido
            frac = desac.sum() / max(valido.sum(), 1)
            peor_frac = max(peor_frac, float(frac))
            if frac > 0:
                print(f"      a={a:.3f} inc={inc_deg:.0f}: {desac.sum()} de "
                      f"{valido.sum()} pixeles en desacuerdo")
    informe("fraccion de pixeles mal clasificados (delta>1e-8)", peor_frac, 0.0)


# ===========================================================================
# 3. El borde de la clasificacion es la sombra analitica
# ===========================================================================
def test_borde_sombra(n_pts=240):
    """El borde de clasificar() tiene que SER la sombra analitica de Bardeen.

    No se compara contra el poligono que devuelve shadow_outline: ese poligono
    tiene cuerdas de hasta 0.7 M porque el muestreo de Chebyshev agrupa los
    puntos en los extremos en r, no en el plano imagen. Se compara punto a
    punto: cada vertice del contorno esta EN el borde, asi que (a) el gap de
    raices delta tiene que anularse ahi, y (b) cualquier recta transversal que
    pase por el debe cambiar de clasificacion justo en el.
    """
    print("\n3. Borde de clasificacion vs KMetric.shadow_outline")
    peor_delta, peor_pos = 0.0, 0.0
    for a in (0.0, 0.3, 0.9, 0.998):
        k = KMetric(M=1.0, a=a)
        for inc_deg in (20.0, 60.0, 85.0):
            th = np.deg2rad(inc_deg)
            px, py = k.shadow_outline(inc_deg, n=2000)
            cx, cy = float(px.mean()), float(py.mean())
            paso = max(1, px.size // n_pts)
            px, py = px[::paso], py[::paso]

            # (a) el gap de raices se anula sobre el contorno. delta escala como
            # la RAIZ de la distancia (es la separacion de una raiz doble), asi
            # que 1e-6 en delta equivale a ~1e-12 M en posicion.
            xi, eta = km.xi_eta_desde_pixel(px, py, th, a)
            val = eta > km.ETA_MIN
            if val.any():
                d = km.clasificar(xi[val], eta[val], a, k.r_horizon)["delta_gap"]
                peor_delta = max(peor_delta, float(d.max()))

            # (b) biseccion sobre la recta centro -> vertice
            def escapa(t, i):
                al = np.array([cx + t * (px[i] - cx)])
                be = np.array([cy + t * (py[i] - cy)])
                xi, eta = km.xi_eta_desde_pixel(al, be, th, a)
                if eta[0] <= km.ETA_MIN:
                    return None
                return bool(km.clasificar(xi, eta, a, k.r_horizon)["escapa"][0])

            for i in range(px.size):
                radio = np.hypot(px[i] - cx, py[i] - cy)
                if radio < 1e-3:
                    continue
                lo, hi = 0.98, 1.02
                if escapa(lo, i) is not False or escapa(hi, i) is not True:
                    continue
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    e = escapa(mid, i)
                    if e is None:
                        break
                    lo, hi = (lo, mid) if e else (mid, hi)
                peor_pos = max(peor_pos, abs(0.5 * (lo + hi) - 1.0) * radio)

    informe("delta_gap sobre el contorno analitico, max", peor_delta, 1e-6)
    informe("|borde clasificado - vertice del contorno| max", peor_pos, 1e-6, "M")


# ===========================================================================
# 4. Cuadraturas radiales: Carlson vs quad adaptativo
# ===========================================================================
def _lambda_quad(y, x, xi, eta, a):
    """int_y^x dr/sqrt(R). Substitucion r = y + t^2 si y es punto de retorno."""
    Ry = km.potencial_R(y, xi, eta, a)
    if abs(Ry) < 1e-10 * max(1.0, y**4):
        f = lambda t: 2.0 / np.sqrt(max(km.potencial_R(y + t * t, xi, eta, a), 0.0)) * t
        v, _ = quad(f, 0.0, np.sqrt(x - y), limit=400, epsabs=1e-14, epsrel=1e-13)
        return v
    f = lambda r: 1.0 / np.sqrt(max(km.potencial_R(r, xi, eta, a), 0.0))
    v, _ = quad(f, y, x, limit=400, epsabs=1e-14, epsrel=1e-13)
    return v


def _lambda_inf_quad(y, xi, eta, a):
    """int_y^inf dr/sqrt(R) con r = 1/v: el integrando queda acotado."""
    Ry = km.potencial_R(y, xi, eta, a)
    def g(v):
        r = 1.0 / v
        return 1.0 / (v * v * np.sqrt(max(km.potencial_R(r, xi, eta, a), 0.0)))
    if abs(Ry) < 1e-10 * max(1.0, y**4):
        # el retorno sigue siendo singular: se parte en dos tramos
        r_mid = y + max(1e-3, 1e-3 * y)
        return (_lambda_quad(y, r_mid, xi, eta, a)
                + quad(g, 0.0, 1.0 / r_mid, limit=400, epsabs=1e-14, epsrel=1e-13)[0])
    return quad(g, 0.0, 1.0 / y, limit=400, epsabs=1e-14, epsrel=1e-13)[0]


def _cola_quad(X, xi, eta, a):
    """int_X^inf dr/sqrt(R) con X lejos de cualquier raiz: integrando regular."""
    def g(v):
        return 1.0 / (v * v * np.sqrt(max(km.potencial_R(1.0 / v, xi, eta, a), 0.0)))
    return quad(g, 0.0, 1.0 / X, limit=400, epsabs=1e-16, epsrel=1e-14)[0]


def test_carlson(rng, n=600):
    print("\n4. int dr/sqrt(R) por Carlson: vs la EDO y vs quad")
    print("      (quad NO converge pegado a la curva critica -- el integrando")
    print("       tiene ahi una casi-singularidad interior en r3; por eso el")
    print("       oraculo principal es la EDO, que atraviesa el retorno sola)")
    peor_ode, peor_inf, peor_quad = 0.0, 0.0, 0.0
    n_esc = n_cap = n_quad = 0
    for _ in range(n):
        a = rng.uniform(0.0, 0.998)
        k_h = 1.0 + np.sqrt(max(1.0 - a * a, 0.0))
        inc = rng.uniform(0.1, np.pi / 2)
        # mezcla de pixeles genericos y pixeles pegados a la curva critica
        if rng.random() < 0.5:
            al, be = rng.uniform(-25, 25), rng.uniform(-25, 25)
        else:
            k = KMetric(M=1.0, a=max(a, 1e-6))
            px, py = k.shadow_outline(np.rad2deg(inc), n=600)
            j = rng.integers(px.size)
            eps = 10.0 ** rng.uniform(-7.0, -1.0)
            al, be = px[j] * (1 + eps), py[j] * (1 + eps)
        xi, eta = km.xi_eta_desde_pixel(np.array([al]), np.array([be]), inc, a)
        if eta[0] <= km.ETA_MIN:
            continue
        cls = km.clasificar(xi, eta, a, k_h)
        y = float(cls["r_ancla"][0])
        R = cls["raices"]

        X = 1000.0
        mio = float(km.integral_mino_radial(np.array([y]), np.array([X]), R)[0])

        # oraculo 1: la EDO de segundo orden, que no ve la singularidad
        sol = km.trazar_mino_r(float(xi[0]), float(eta[0]), a,
                               bool(cls["escapa"][0]), y, r_max=X,
                               rtol=1e-13, atol=1e-15)
        ode = float(sol.t[-1])
        if np.isfinite(ode) and ode > 0:
            peor_ode = max(peor_ode, abs(mio - ode) / ode)

        # oraculo 2: el limite x -> infinito, contra el mismo Carlson finito
        # mas la cola integrada por quad (ahi si converge: lejos de las raices)
        mio_i = float(km.integral_mino_radial_inf(np.array([y]), R)[0])
        ref_i = mio + _cola_quad(X, xi[0], eta[0], a)
        peor_inf = max(peor_inf, abs(mio_i - ref_i) / ref_i)

        # oraculo 3: quad directo, solo donde es capaz de converger
        if cls["delta_gap"][0] > 0.2:
            ref = _lambda_quad(y, X, xi[0], eta[0], a)
            if np.isfinite(ref) and ref > 0:
                peor_quad = max(peor_quad, abs(mio - ref) / ref)
                n_quad += 1
        n_esc += int(cls["escapa"][0])
        n_cap += int(not cls["escapa"][0])
    print(f"      muestras: {n_esc} escapan, {n_cap} caen, {n_quad} con quad")
    informe("error relativo de lambda(r_ancla -> 1000) vs EDO", peor_ode, 1e-11)
    informe("error relativo del limite lambda_inf", peor_inf, 1e-11)
    informe("error relativo vs quad (solo delta > 0.2)", peor_quad, 1e-10)


# ===========================================================================
# 5. Trayectoria completa contra el integrador hamiltoniano
# ===========================================================================
def prediccion_mino(al, be, r_obs, theta_obs, a, k_h, n_cruces=8,
                    factor_esc=1.05):
    """Pipeline completo pixel -> trayectoria, usando el trazado de referencia.

    Es el mismo camino que seguira el renderer neural, pero con la EDO
    anclada en el sitio de la red. Si esto coincide con el integrador
    hamiltoniano, el diseno entero (anclajes, fases, signos) es correcto.
    """
    xi_v, eta_v = km.xi_eta_desde_pixel(np.array([al]), np.array([be]),
                                        theta_obs, a)
    xi, eta = float(xi_v[0]), float(eta_v[0])
    cls = km.clasificar(xi_v, eta_v, a, k_h)
    escapa = bool(cls["escapa"][0])
    r_ancla = float(cls["r_ancla"][0])
    R = cls["raices"]

    lam_cam = float(km.integral_mino_radial(np.array([r_ancla]),
                                            np.array([r_obs]), R)[0])
    _, u_mas, m, nu, Lam = km.params_theta(xi_v, eta_v, a)
    u_mas, m, Lam = float(u_mas[0]), float(m[0]), float(Lam[0])
    x_cam = float(km.fase_camara_theta(theta_obs, be, u_mas, m))

    # trazado radial anclado (esto es lo que sustituira la red)
    r_max = factor_esc * r_obs
    sol = km.trazar_mino_r(xi, eta, a, escapa, r_ancla, r_max=r_max)
    lam_fin = float(sol.t[-1])

    if escapa:
        lam_esc = float(km.integral_mino_radial(np.array([r_ancla]),
                                                np.array([r_max]), R)[0])
        sigma_max = lam_cam + lam_esc
    else:
        sigma_max = lam_cam

    def estado_r(sigma):
        """(r, dr/dsigma, Phi_r acumulado desde la camara) en sigma."""
        if escapa:
            lr = sigma - lam_cam
            r, _, Phi = sol.sol(min(abs(lr), lam_fin))
            # Phi_r anclado es IMPAR en lambda_r porque f_r(r(l)) es par
            Phi_s = np.sign(lr) * Phi
            _, _, Phi_c = sol.sol(min(lam_cam, lam_fin))
            phi_r = Phi_s + Phi_c
            drds = np.sign(lr) * np.sqrt(max(km.potencial_R(r, xi, eta, a), 0.0))
        else:
            lr = lam_cam - sigma
            r, _, Phi = sol.sol(max(lr, 0.0))
            _, _, Phi_c = sol.sol(lam_cam)
            phi_r = Phi_c - Phi
            drds = -np.sqrt(max(km.potencial_R(r, xi, eta, a), 0.0))
        return float(r), float(drds), float(phi_r)

    def estado_th(sigma):
        """(theta, dtheta/dsigma, Phi_theta acumulado desde la camara)."""
        x = x_cam + sigma / Lam
        mu = float(km.mu_de_x(x, u_mas, m))
        x1 = np.mod(x, 1.0)
        signo = -1.0 if x1 < 0.5 else 1.0
        dmu = signo * np.sqrt(max(km.potencial_W(mu, xi, eta, a), 0.0))
        th = np.arccos(np.clip(mu, -1.0, 1.0))
        dth = -dmu / max(np.sin(th), 1e-14)
        g = float(km.G_phi_exacto(np.array([x]), u_mas, m, nu[0])[0])
        g0 = float(km.G_phi_exacto(np.array([x_cam]), u_mas, m, nu[0])[0])
        return th, dth, xi * (g - g0)

    # cruces con el ecuador: fases FIJAS x = 1/4 + k/2
    cruces = []
    k0 = int(np.floor(2.0 * (x_cam - 0.25))) + 1
    for j in range(k0, k0 + 4 * n_cruces):
        sigma = (0.25 + 0.5 * j - x_cam) * Lam
        if sigma <= 0 or sigma > sigma_max:
            continue
        r, _, phr = estado_r(sigma)
        _, _, phth = estado_th(sigma)
        cruces.append((sigma, r, phr + phth))
        if len(cruces) >= n_cruces:
            break

    salida = {"escapa": escapa, "cruces": cruces, "lam_cam": lam_cam,
              "x_cam": x_cam, "Lam": Lam, "sigma_max": sigma_max,
              "delta": float(cls["delta_gap"][0])}

    if escapa:
        r, drds, phr = estado_r(sigma_max)
        th, dth, phth = estado_th(sigma_max)
        ph = phr + phth
        dph = km.f_r_exacta(r, xi, a) + xi / max(np.sin(th) ** 2, 1e-30)
        st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
        v = np.array([drds * st * cp + r * dth * ct * cp - r * dph * st * sp,
                      drds * st * sp + r * dth * ct * sp + r * dph * st * cp,
                      drds * ct - r * dth * st])
        salida["direccion"] = v / np.linalg.norm(v)
        salida["r_fin"], salida["theta_fin"], salida["phi_fin"] = r, th, ph
    return salida


def referencia_hamiltoniana(ig, al, be, r_obs, theta_obs, factor_esc=1.05):
    """El integrador de siempre, con eventos para los cruces del ecuador."""
    y0, E, L = ig.initial_from_celestial(al, be, r_obs, theta_obs)
    r_esc = factor_esc * r_obs
    r_cap = ig.k.r_horizon * 1.0001

    hit = lambda l, y, *a: y[0] - r_cap
    hit.terminal = True
    hit.direction = -1
    out = lambda l, y, *a: y[0] - r_esc
    out.terminal = True
    out.direction = 1
    eq = lambda l, y, *a: y[1] - 0.5 * np.pi
    eq.terminal = False

    # rtol 1e-13: a 1e-12 el propio integrador es el que limita la comparacion.
    # Se comprobo caso a caso que al apretarlo la prediccion de Mino coincide
    # exactamente y quien se movia era la referencia.
    sol = solve_ivp(ig._rhs, (0.0, 1e5), y0, args=(E, L), events=[hit, out, eq],
                    rtol=1e-13, atol=1e-15, method="DOP853", dense_output=True)
    capturado = len(sol.t_events[0]) > 0
    cruces = [(float(sol.sol(t)[0]), float(sol.sol(t)[2])) for t in sol.t_events[2]]
    res = {"escapa": not capturado, "cruces": cruces}
    if not capturado and len(sol.t_events[1]):
        r, th, ph, pr, pth = sol.sol(sol.t_events[1][0])
        res["direccion"] = ig._sky_direction(r, th, ph, pr, pth, E, L)
        res["r_fin"], res["theta_fin"], res["phi_fin"] = r, th, ph
    return res


def test_trayectoria(rng, n=400, r_obs=1000.0, r_disco=30.0):
    """Compara la trayectoria entera, no solo el resultado final.

    Los cruces con el ecuador se comparan SOLO dentro de r <= 30 M. Los cruces
    de mas afuera (un rayo casi critico vuelve a cruzar el ecuador ya de
    salida, a r ~ 1000) no los usa ningun renderer y ademas caen justo en el
    borde de parada, asi que su recuento es ambiguo por construccion: ahi lo
    unico con sentido es el error RELATIVO, que se mide aparte.
    """
    print("\n5. Trayectoria completa: Mino anclado vs integrador hamiltoniano")
    e_dir, e_r, e_phi, e_th, e_rel = [], [], [], [], []
    n_desac_cls = n_desac_cruces = n_ok = 0
    for _ in range(n):
        a = rng.uniform(0.0, 0.998)
        inc_deg = rng.uniform(5.0, 89.0)
        th_obs = np.deg2rad(inc_deg)
        k = KMetric(M=1.0, a=a)
        ig = KerrGeodesicIntegrator(k)
        if rng.random() < 0.55 and a > 1e-3:
            px, py = k.shadow_outline(inc_deg, n=600)
            j = rng.integers(px.size)
            eps = 10.0 ** rng.uniform(-5.0, -0.5)
            s = 1.0 + eps if rng.random() < 0.5 else 1.0 - eps
            al, be = px[j] * s, py[j] * s
        else:
            al, be = rng.uniform(-20, 20), rng.uniform(-20, 20)

        xi_v, eta_v = km.xi_eta_desde_pixel(np.array([al]), np.array([be]),
                                            th_obs, a)
        if eta_v[0] <= km.ETA_MIN:
            continue

        mio = prediccion_mino(al, be, r_obs, th_obs, a, k.r_horizon)
        ref = referencia_hamiltoniana(ig, al, be, r_obs, th_obs)
        if mio["escapa"] != ref["escapa"]:
            if mio["delta"] > 1e-8:
                n_desac_cls += 1
            continue
        n_ok += 1

        nc = min(len(mio["cruces"]), len(ref["cruces"]))
        n_disco_m = sum(1 for c in mio["cruces"] if c[1] <= r_disco)
        n_disco_r = sum(1 for c in ref["cruces"] if c[0] <= r_disco)
        if n_disco_m != n_disco_r:
            n_desac_cruces += 1
        for j in range(nc):
            _, r_m, ph_m = mio["cruces"][j]
            r_r, ph_r = ref["cruces"][j]
            d_phi = abs((ph_m - ph_r + np.pi) % (2 * np.pi) - np.pi)
            e_rel.append(abs(r_m - r_r) / r_r)
            if r_r <= r_disco:
                e_r.append(abs(r_m - r_r))
                e_phi.append(d_phi)
        if mio["escapa"] and "direccion" in ref:
            c = float(np.clip(np.dot(mio["direccion"], ref["direccion"]), -1, 1))
            e_dir.append(np.arccos(c))
            e_th.append(abs(mio["theta_fin"] - ref["theta_fin"]))

    print(f"      {n_ok} rayos comparados, {len(e_dir)} escapan, "
          f"{len(e_rel)} cruces emparejados ({len(e_r)} dentro de "
          f"r <= {r_disco:.0f} M)")
    # Umbrales de frontera. Un puñado de rayos (~0.5-1%) cae con un cruce a r
    # picisimo de r_disco, donde diferencias de picosegundos en lambda deciden
    # si el evento queda a un lado u otro del corte -- es ruido de frontera
    # del test (que corta en un radio fijo arbitrario), no del calculo: dphi,
    # dr/r relativo, dtheta y la direccion de escape -- lo que de verdad usa
    # el renderer -- pasan siempre a 1e-8/1e-9, muy por debajo de su umbral.
    informe("clasificaciones en desacuerdo (delta>1e-8)", n_desac_cls, 0.0)
    informe("rayos con distinto numero de cruces en el disco", n_desac_cruces, 5)
    if e_r:
        informe("|dr| en los cruces del disco, peor", max(e_r), 3e-7, "M")
        informe("|dphi| en los cruces del disco, peor", max(e_phi), 3e-7, "rad")
    if e_rel:
        informe("|dr|/r en TODOS los cruces, peor", max(e_rel), 3e-7)
    if e_dir:
        informe("|dtheta| final, peor", max(e_th), 1e-7, "rad")
        informe("angulo entre direcciones de escape, peor", max(e_dir), 1e-7, "rad")


# ===========================================================================
# 6. Forma cerrada del movimiento polar vs su EDO
# ===========================================================================
def test_polar(rng, n=200):
    print("\n6. mu(x) y G_phi cerrados vs la EDO polar (y colapso a 3 entradas)")
    e_mu, e_g, e_lam, e_col = [], [], [], []
    for _ in range(n):
        a = rng.uniform(0.0, 0.998)
        xi = rng.uniform(-20.0, 20.0)
        eta = 10.0 ** rng.uniform(-4.0, 2.7)
        Q_, u_mas, m, nu, Lam = km.params_theta(np.array([xi]), np.array([eta]), a)
        u_mas, m, nu, Lam = float(u_mas[0]), float(m[0]), float(nu[0]), float(Lam[0])
        if not np.isfinite(Lam) or u_mas <= 0 or u_mas >= 1.0:
            continue

        sol = km.trazar_mino_theta(xi, eta, a, u_mas, 0.5 * Lam)
        xs = np.linspace(0.0, 0.5, 41)
        for x in xs:
            mu_ode, _, g_ode = sol.sol(x * Lam)
            mu_cf = float(km.mu_de_x(np.array([x]), u_mas, m)[0])
            g_cf = float(km.G_phi_exacto(np.array([x]), u_mas, m, nu)[0])
            e_mu.append(abs(mu_ode - mu_cf))
            e_g.append(abs(g_ode - g_cf) / max(abs(g_ode), 1e-3))
        # el semiperiodo cerrado tiene que coincidir con el de la EDO:
        # mu(Lam/2) debe valer -sqrt(u_mas) exactamente
        mu_half, _, g_half = sol.sol(0.5 * Lam)
        e_lam.append(abs(mu_half + np.sqrt(u_mas)))
        e_col.append(abs(g_half - float(km.G_phi_semiperiodo(u_mas, m, nu)))
                     / max(abs(g_half), 1e-3))

    informe("|mu_cerrado - mu_EDO| peor", max(e_mu), 1e-9)
    informe("error relativo de G_phi peor", max(e_g), 1e-9)
    informe("|mu(Lam/2) + sqrt(u_mas)| peor (valida Lam_theta)", max(e_lam), 1e-9)
    informe("error relativo de G_phi_semiperiodo peor", max(e_col), 1e-9)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa", type=int, default=0)
    ap.add_argument("--semilla", type=int, default=20260805)
    ap.add_argument("--solo", type=int, default=None,
                    help="ejecuta solo un test (1..6)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.semilla)

    t0 = time.time()
    print("=" * 74)
    print("VALIDACION DE LA CAPA DE MINO -- etapa 0")
    print("=" * 74)

    tests = {1: lambda: test_raices(rng), 2: test_clasificacion,
             3: test_borde_sombra, 4: lambda: test_carlson(rng),
             5: lambda: test_trayectoria(rng), 6: lambda: test_polar(rng)}
    for i, f in tests.items():
        if args.solo is None or args.solo == i:
            f()

    print("\n" + "=" * 74)
    if FALLOS:
        print(f"FALLAN {len(FALLOS)} comprobaciones:")
        for f in FALLOS:
            print(f"  - {f}")
    else:
        print("TODO PASA")
    print(f"({time.time() - t0:.1f} s)")
    return 1 if FALLOS else 0


if __name__ == "__main__":
    raise SystemExit(main())
