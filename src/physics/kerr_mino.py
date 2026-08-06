"""Kerr en tiempo de Mino: la parte que se resuelve EXACTA.

Por que existe este modulo
--------------------------
kerr_integrator.py resuelve las geodesicas en forma hamiltoniana, con 5 EDO
acopladas y sin raices cuadradas. Eso es lo correcto para un trazador de fuerza
bruta, pero para APRENDER la solucion con una red es la peor eleccion posible:
el estado (r, theta, phi, p_r, p_theta) depende de la camara, de la
inclinacion y del encuadre, asi que la red acabaria aprendiendo un mapa
imagen->imagen en vez de la solucion de una ecuacion diferencial.

En tiempo de Mino, definido por dlambda = dtau / Sigma, el factor Sigma que
mezclaba r con theta desaparece y las ecuaciones se DESACOPLAN exactamente:

    (dr/dlambda)^2     = R(r; xi, eta, a)
    (dtheta/dlambda)^2 = Theta(theta; xi, eta, a)
    dphi/dlambda       = f_r(r) + f_theta(theta)

Las dos primeras son EDO de una sola variable, y la tercera es una suma de dos
cuadraturas independientes. Toda la trayectoria queda fijada por TRES numeros
-- xi = L_z/E, eta = Q/E^2 y el espin a -- igual que en Schwarzschild quedaba
fijada por el parametro de impacto b. La camara no entra: (r_obs, theta_obs)
solo deciden en que FASE de esas curvas se engancha el rayo, y esa fase se
calcula en forma cerrada aqui, sin red.

Lo que se resuelve exacto en este modulo
----------------------------------------
- Constantes del rayo desde el pixel:              xi_eta_desde_pixel
- Raices del cuartico R(r) por Ferrari:            raices_radiales
- Sombra / captura sin trazar ningun rayo:         clasificar
- Periodo y parametros del movimiento en theta:    params_theta
- Tiempos de Mino por integrales elipticas:        integral_mino_radial
- Fase de la camara en el ciclo de theta:          fase_camara_theta
- Cruces con el ecuador (el disco):                fases fijas x = 1/4 + k/2

Lo unico que NO tiene forma cerrada elemental es r(lambda) y theta(lambda), o
sea la INVERSION de las cuadraturas. Eso es exactamente lo que aprende la red.

Convenios
---------
M = 1, G = c = 1, E = 1 (solo fija la escala de lambda). xi = L_z, eta = Q.
Las mismas coordenadas celestes de Bardeen (alpha, beta) que usan
KMetric.shadow_outline y KerrGeodesicIntegrator.initial_from_celestial.

El anclaje de lambda:
  - rayo que ESCAPA: lambda = 0 en el punto de retorno radial r4, donde
    dr/dlambda = 0. r(lambda) es PAR en lambda, asi que basta aprender
    lambda >= 0. Es el analogo exacto del origen en el periastro que se uso
    en el dataset de Schwarzschild.
  - rayo CAPTURADO: no hay retorno, asi que se ancla en r_stop = 1.02 r_+ y
    lambda crece hacia fuera. No se ancla en el horizonte porque f_r tiene un
    polo en Delta = 0 y Phi_r divergiria; no se ancla en la camara porque eso
    volveria a meter r_obs dentro de la red.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import ellipj, ellipk, ellipkinc, elliprf, elliprj

# Por debajo de este eta el rayo es casi vortical (no cruza el ecuador) y la
# parametrizacion en theta degenera. Esos pixeles van al trazador clasico.
ETA_MIN = 1e-5

# Margen sobre el horizonte para anclar los rayos capturados. El ISCO mas
# interno posible (a = 0.998) esta en r = 1.237, y 1.02 r_+ = 1.084, asi que
# ningun cruce con el disco se queda por dentro del anclaje.
FACTOR_R_STOP = 1.02


# ============================================================ pixel -> rayo
def xi_eta_desde_pixel(alpha, beta, theta_obs: float, a: float):
    """Constantes de movimiento del rayo que llega al pixel (alpha, beta).

    Sale de las mismas dos lineas que usa initial_from_celestial: xi = -alpha
    sin(theta_obs) por definicion de la coordenada de Bardeen, y eta de
    imponer Theta(theta_obs) = p_theta^2 = beta^2.
    """
    st, ct = np.sin(theta_obs), np.cos(theta_obs)
    al = np.asarray(alpha, dtype=np.float64)
    be = np.asarray(beta, dtype=np.float64)
    xi = -al * st
    eta = be * be + ct * ct * (al * al - a * a)
    return xi, eta


# ================================================================ potencial R
def coef_R(xi, eta, a):
    """Coeficientes de R(r) = r^4 + p r^2 + q r + s.

    R = (r^2 + a^2 - a xi)^2 - Delta [(xi - a)^2 + eta], y al expandir el
    termino en r^3 se CANCELA: el cuartico ya sale deprimido, que es
    justo lo que necesita Ferrari.
    """
    K = eta + (xi - a) ** 2          # constante de Carter "K" de Walker-Penrose
    p = a * a - xi * xi - eta
    q = 2.0 * K
    s = -a * a * eta
    return p, q, s


def potencial_R(r, xi, eta, a):
    p, q, s = coef_R(xi, eta, a)
    return ((r * r + p) * r + q) * r + s


def dR_dr(r, xi, eta, a):
    p, q, _ = coef_R(xi, eta, a)
    return (4.0 * r * r + 2.0 * p) * r + q


def f_r_exacta(r, xi, a):
    """Parte radial de dphi/dlambda.

    dphi/dlambda = a (P/Delta - 1) + xi/sin^2(theta) con P = r^2 + a^2 - a xi.
    Escrito como a (P - Delta)/Delta = a (2r - a xi)/Delta se evita la resta de
    dos numeros grandes cuando r es grande, donde P y Delta casi coinciden.
    """
    delta = r * r - 2.0 * r + a * a
    return a * (2.0 * r - a * xi) / delta


# ============================================================ potencial Theta
def potencial_Theta(theta, xi, eta, a):
    ct, st = np.cos(theta), np.sin(theta)
    return eta + a * a * ct * ct - xi * xi * (ct * ct) / (st * st)


def potencial_W(mu, xi, eta, a):
    """(dmu/dlambda)^2 con mu = cos(theta).

    W(mu) = (1 - mu^2) Theta. El cambio de variable quita el cot^2 y con el
    la singularidad de coordenadas en el eje: W es un polinomio limpio, par,
    de grado 4 en mu. Es la forma en la que conviene integrar.
    """
    return -a * a * mu**4 + (a * a - eta - xi * xi) * mu * mu + eta


def dW_dmu(mu, xi, eta, a):
    return -4.0 * a * a * mu**3 + 2.0 * (a * a - eta - xi * xi) * mu


def params_theta(xi, eta, a):
    """Parametros del movimiento polar: (Q_menos, u_mas, m, nu, Lam_theta).

    W(mu) = 0 es una cuadratica en u = mu^2 con una raiz positiva u_mas (el
    turning point, mu_max^2 = cos^2(theta_min)) y otra negativa u_menos. En
    esas variables la EDO es literalmente la de la funcion sn de Jacobi:

        mu(lambda) = sqrt(u_mas) sn(K(m) - nu lambda, m),  m = u_mas/u_menos

    con periodo Lam_theta = 4 K(m) / nu. O sea: el movimiento polar es
    EXACTAMENTE periodico y se conoce su periodo en forma cerrada. Los cruces
    con el ecuador (mu = 0) caen en fases fijas, y de ahi sale el disco sin
    detectar nada por cambio de signo.

    Se parametriza con Q_menos = -a^2 u_menos en vez de u_menos para no
    dividir por a^2: asi a = 0 sale bien (m = 0, Lam = 2 pi / sqrt(xi^2+eta),
    que es la vuelta completa del plano orbital de Schwarzschild).
    """
    B = xi * xi + eta - a * a
    Q_menos = 0.5 * (B + np.sqrt(B * B + 4.0 * a * a * eta))
    u_mas = eta / Q_menos
    m = -(a * a) * eta / (Q_menos * Q_menos)      # = u_mas/u_menos, <= 0
    nu = np.sqrt(Q_menos)
    Lam_theta = 4.0 * ellipk(m) / nu
    return Q_menos, u_mas, m, nu, Lam_theta


# ------------------------------------------- Jacobi con parametro negativo
def jacobi_m_negativo(u, m):
    """sn, cn, dn para m <= 0, que es el unico caso que aparece aqui.

    scipy.special.ellipj solo acepta m en [0,1], asi que se usa la
    transformacion de modulo imaginario. Con m1 = -m y m_t = m1/(1+m1):

        sn(u|m) = sd(u f | m_t)/f,  cn(u|m) = cd(u f | m_t),
        dn(u|m) = nd(u f | m_t),    f = sqrt(1+m1)

    (comprobado algebraicamente: se verifican sn^2+cn^2=1 y dn^2 = 1 - m sn^2)
    Ese m_t es el mismo m~ = m/(m-1) que se le pasa a la red como entrada, y
    esta en [0,1) siempre, que es justo lo que quiere un normalizador.
    """
    m1 = -np.asarray(m, dtype=np.float64)
    fac = np.sqrt(1.0 + m1)
    mt = m1 / (1.0 + m1)
    sn_, cn_, dn_, _ = ellipj(np.asarray(u, dtype=np.float64) * fac, mt)
    return sn_ / (dn_ * fac), cn_ / dn_, 1.0 / dn_


def mu_de_x(x, u_mas, m):
    """cos(theta) en la fase x = lambda / Lam_theta, anclado en theta_min.

    x = 0    -> theta_min  (mu maximo)
    x = 1/4  -> ecuador
    x = 1/2  -> theta_max
    y de ahi en adelante se repite. mu es PAR respecto de x = 0 y de x = 1/2,
    asi que el dominio fundamental es x en [0, 1/2] y todo lo demas es plegado.
    """
    kk = ellipk(m)
    sn, _, _ = jacobi_m_negativo(kk * (1.0 - 4.0 * np.asarray(x)), m)
    return np.sqrt(u_mas) * sn


def plegar_x(x):
    """Lleva cualquier fase x al dominio fundamental [0, 1/2].

    mu tiene periodo 1 en x y es par, asi que basta reflejar.
    """
    x1 = np.mod(np.asarray(x, dtype=np.float64), 1.0)
    return np.where(x1 > 0.5, 1.0 - x1, x1)


# ==================================================== raices del cuartico R
def _cubica_raiz_mayor(b, c, d):
    """Mayor raiz real de z^3 + b z^2 + c z + d = 0, vectorizado.

    Es la cubica resolvente de Ferrari. Se quiere la raiz MAYOR porque de ella
    sale w = sqrt(2z) y cuanto mas grande es w mejor condicionado queda el
    reparto en dos cuadraticas. Siempre existe una raiz real >= 0: en z = 0 la
    cubica vale d = -q^2/8 <= 0 y en +infinito diverge a +infinito.
    """
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    P = c - b * b / 3.0
    Q = (2.0 * b**3) / 27.0 - b * c / 3.0 + d
    disc = (0.5 * Q) ** 2 + (P / 3.0) ** 3

    with np.errstate(divide="ignore", invalid="ignore"):
        # Tres raices reales (solo puede pasar con P < 0): forma trigonometrica
        Pn = np.where(P < 0.0, P, -1.0)
        arg = np.clip((3.0 * Q) / (2.0 * Pn) * np.sqrt(-3.0 / Pn), -1.0, 1.0)
        t3 = 2.0 * np.sqrt(-Pn / 3.0) * np.cos(np.arccos(arg) / 3.0)
        # Una sola raiz real: Cardano
        sq = np.sqrt(np.maximum(disc, 0.0))
        t1 = np.cbrt(-0.5 * Q + sq) + np.cbrt(-0.5 * Q - sq)

    t = np.where(disc <= 0.0, t3, t1)
    z = t - b / 3.0

    # Dos pasos de Newton. Las formulas cerradas de la cubica pierden digitos
    # cuando los coeficientes son muy dispares, y aqui se pagaria despues en
    # las raices del cuartico; Newton lo recupera casi gratis.
    for _ in range(2):
        f = ((z + b) * z + c) * z + d
        fp = (3.0 * z + 2.0 * b) * z + c
        paso = np.where(np.abs(fp) > 1e-300, f / np.where(fp == 0.0, 1.0, fp), 0.0)
        z = z - paso
    return z


def raices_radiales(xi, eta, a: float) -> np.ndarray:
    """Las 4 raices de R(r), complejas, ordenadas por parte real ascendente.

    Devuelve un array (4, N). Ferrari sobre el cuartico deprimido:

        (r^2 + p/2 + z)^2 = 2z r^2 - q r + (z^2 + zp + p^2/4 - s)

    y el lado derecho es un cuadrado perfecto justo cuando z es raiz de la
    cubica resolvente. De ahi salen dos cuadraticas y las cuatro raices.

    La estructura fisica: con eta > 0 siempre hay una raiz real negativa y
    otra positiva pequena. Las dos GRANDES son las que importan -- son las que
    colisionan sobre la curva critica (la orbita esferica de fotones) y las
    que deciden si el rayo escapa.
    """
    xi = np.atleast_1d(np.asarray(xi, dtype=np.float64))
    eta = np.atleast_1d(np.asarray(eta, dtype=np.float64))
    p, q, s = coef_R(xi, eta, a)

    z = _cubica_raiz_mayor(p, (p * p - 4.0 * s) / 4.0, -q * q / 8.0)
    z = np.maximum(z, 0.0)
    w = np.sqrt(2.0 * z)

    # r^2 - w r + (p/2 + z + q/(2w)) = 0   y   r^2 + w r + (p/2 + z - q/(2w)) = 0
    seguro = w > 1e-150
    t = np.where(seguro, q / (2.0 * np.where(seguro, w, 1.0)), 0.0)
    c1 = 0.5 * p + z + t
    c2 = 0.5 * p + z - t

    d1 = np.sqrt((w * w - 4.0 * c1).astype(np.complex128))
    d2 = np.sqrt((w * w - 4.0 * c2).astype(np.complex128))
    R = np.stack([0.5 * (w + d1), 0.5 * (w - d1),
                  0.5 * (-w + d2), 0.5 * (-w - d2)], axis=0)

    orden = np.argsort(R.real, axis=0, kind="stable")
    return np.take_along_axis(R, orden, axis=0)


def clasificar(xi, eta, a: float, r_horizon: float) -> dict:
    """Sombra, criticidad y anclajes, sin trazar un solo rayo.

    Un rayo que viene del infinito da la vuelta si y solo si R(r) tiene ceros
    por encima del horizonte. Como R(r_+) = (r_+^2 + a^2 - a xi)^2 >= 0 y
    R -> +infinito, esos ceros vienen SIEMPRE de dos en dos: existen si y solo
    si la mayor raiz real esta por encima de r_+. Eso es la sombra, exacta y
    en forma cerrada: sustituye a la cabeza "existe el cruce?" que hubo que
    parchear en la red anterior.

    delta = |r4 - r3| es la distancia efectiva a la curva critica: se anula
    exactamente sobre ella (raiz doble) por los dos lados, es suave, y sale
    gratis de las raices. Es el analogo de |beta - beta_crit| de Schwarzschild
    y va a ser la feature critica de la red, en float64 SIEMPRE.
    """
    R = raices_radiales(xi, eta, a)
    r3, r4 = R[2], R[3]
    delta_gap = np.abs(r4 - r3)

    escala = np.maximum(1.0, np.abs(r4))
    r4_es_real = np.abs(r4.imag) < 1e-11 * escala
    escapa = r4_es_real & (r4.real > r_horizon)

    r_stop = FACTOR_R_STOP * r_horizon
    r_ancla = np.where(escapa, r4.real, r_stop)
    # Radio de la MESETA: donde el rayo se queda pegado cuando casi orbita.
    # Es el que fija el ritmo secular de phi (ver Phi_r en la red).
    r_pl_cap = np.maximum(np.where(r4_es_real, r4.real, r4.real), r_stop)
    r_plateau = np.where(escapa, r4.real, r_pl_cap)

    return {"raices": R, "escapa": escapa, "delta_gap": delta_gap,
            "r_ancla": r_ancla, "r_plateau": r_plateau, "r_stop": r_stop}


# ================================================ cuadraturas radiales (Mino)
def integral_mino_radial(y, x, raices: np.ndarray) -> np.ndarray:
    """int_y^x dr / sqrt(R(r)) por la reduccion simetrica de Carlson.

    Formula estandar (Carlson 1988): con X_i = sqrt(x - a_i), Y_i = sqrt(y - a_i)
    y U_ij = (X_i X_j Y_k Y_l + Y_i Y_j X_k X_l)/(x - y),

        int_y^x dt / sqrt(prod (t - a_i)) = 2 R_F(U12^2, U13^2, U14^2)

    Dos cosas la hacen especialmente comoda aqui:

    1. Vale IGUAL con raices complejas, sin cambiar de formula. Basta hacer la
       aritmetica en complejo: el conjunto {U12^2, U13^2, U14^2} sale con un
       elemento real y dos conjugados, y R_F -- que es simetrico en sus tres
       argumentos -- devuelve un real. Eso cubre de un golpe el caso capturado
       (dos raices reales y un par complejo) sin ramas aparte.
    2. Al ser R_F simetrico, el ORDEN de las raices da igual: una permutacion
       solo reordena las tres particiones en pares.

    Requisito: y y x por encima de todas las raices REALES, que es siempre
    nuestro caso (se ancla en r4 si el rayo escapa, y si cae en r_stop, que
    esta por encima de todas las raices porque no hay ninguna sobre r_+).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X = np.sqrt(x[None, :] - raices)
    Y = np.sqrt(y[None, :] - raices)
    dxy = x - y

    def U(i, j, k, l):
        return (X[i] * X[j] * Y[k] * Y[l] + Y[i] * Y[j] * X[k] * X[l]) / dxy

    u12, u13, u14 = U(0, 1, 2, 3), U(0, 2, 1, 3), U(0, 3, 1, 2)
    return (2.0 * elliprf(u12 * u12, u13 * u13, u14 * u14)).real


def integral_mino_radial_inf(y, raices: np.ndarray) -> np.ndarray:
    """int_y^infinito dr / sqrt(R(r)). El limite x -> infinito del anterior.

    Es FINITO siempre: R ~ r^4, asi que el integrando cae como 1/r^2. Ese
    numero, lambda_inf, es la escala natural con la que normalizar lambda para
    la red: el rayo entero cabe en [0, lambda_inf] por construccion, sin
    importar cuantas vueltas de al agujero.
    """
    y = np.asarray(y, dtype=np.float64)
    Y = np.sqrt(y[None, :] - raices)
    u12 = Y[0] * Y[1] + Y[2] * Y[3]
    u13 = Y[0] * Y[2] + Y[1] * Y[3]
    u14 = Y[0] * Y[3] + Y[1] * Y[2]
    return (2.0 * elliprf(u12 * u12, u13 * u13, u14 * u14)).real


def integral_phi_r(r_hi, raices: np.ndarray, r_ancla, escapa, xi, a,
                   n_nodos: int = 64):
    """Phi_r = int_{r_ancla}^{r_hi} f_r(r) dr/sqrt(R(r)), en forma casi cerrada.

    Es la parte RADIAL de phi. Se calcula exacta en vez de aprenderla porque
    medido en el renderer era la fuente dominante de error angular: la cabeza
    de red daba ~2.5e-3 rad, unos 25 pixeles.

    El truco es el cambio de variable. Escribiendo R = (r-r1)(r-r2) Q(r) con
    Q el par de raices GRANDES -- las que colisionan sobre la curva critica --
    hay una sustitucion que absorbe exactamente la casi-singularidad de Q:

      escapa (r3, r4 reales, ancla en r4):  r = r4 + delta sinh^2(w),  dr/sqrt(Q) = 2 dw
      cae, par complejo mu +- i nu:         r = mu + nu sinh(w),       dr/sqrt(Q) =   dw
      cae, r3, r4 reales:                   r = mu + kappa cosh(w),    dr/sqrt(Q) =   dw

    En los tres casos dr/sqrt(Q) es CONSTANTE por dw, asi que

        dlambda = dr/sqrt(R) = C dw / sqrt((r-r1)(r-r2))

    y lo que queda del integrando es suave y acotado por muy pegado que este el
    rayo a la curva critica. Por eso una cuadratura de Gauss-Legendre fija
    basta, mientras que quad adaptativo sobre la variable r no converge ahi
    (comprobado en validate_kerr_mino.py, test 4).

    Devuelve (Phi_r, lambda): lambda sale de la misma cuadratura y sirve de
    comprobacion cruzada contra integral_mino_radial, que la calcula por una
    via completamente distinta (Carlson).
    """
    r_hi = np.atleast_1d(np.asarray(r_hi, dtype=np.float64))
    r_ancla = np.atleast_1d(np.asarray(r_ancla, dtype=np.float64))
    escapa = np.atleast_1d(np.asarray(escapa, dtype=bool))
    xi = np.atleast_1d(np.asarray(xi, dtype=np.float64))
    n = r_hi.size

    r1, r2 = raices[0], raices[1]
    r3, r4 = raices[2], raices[3]
    # las dos "pequenas" siempre son reales en nuestro dominio (eta > 0)
    peq_a, peq_b = r1.real, r2.real

    complejo = np.abs(r4.imag) > 1e-13 * np.maximum(1.0, np.abs(r4.real))

    x_gl, w_gl = np.polynomial.legendre.leggauss(n_nodos)
    x_gl = x_gl[None, :]
    w_gl = w_gl[None, :]

    # --- limites en w y la parametrizacion r(w), segun el caso -------------
    w_lo = np.zeros(n)
    w_hi = np.zeros(n)
    C = np.ones(n)

    esc = escapa
    delta = np.where(esc, (r4.real - r3.real), 1.0)
    delta = np.maximum(delta, 1e-300)
    if esc.any():
        s = np.sqrt(np.maximum(r_hi - r4.real, 0.0))
        w_hi = np.where(esc, np.arcsinh(s / np.sqrt(delta)), w_hi)
        C = np.where(esc, 2.0, C)

    cae_c = (~esc) & complejo
    mu_c = r4.real
    nu_c = np.maximum(np.abs(r4.imag), 1e-300)
    if cae_c.any():
        w_lo = np.where(cae_c, np.arcsinh((r_ancla - mu_c) / nu_c), w_lo)
        w_hi = np.where(cae_c, np.arcsinh((r_hi - mu_c) / nu_c), w_hi)

    cae_r = (~esc) & (~complejo)
    mu_r = 0.5 * (r3.real + r4.real)
    kap_r = np.maximum(0.5 * (r4.real - r3.real), 1e-300)
    if cae_r.any():
        arg_lo = np.maximum((r_ancla - mu_r) / kap_r, 1.0)
        arg_hi = np.maximum((r_hi - mu_r) / kap_r, 1.0)
        w_lo = np.where(cae_r, np.arccosh(arg_lo), w_lo)
        w_hi = np.where(cae_r, np.arccosh(arg_hi), w_hi)

    # --- nodos de Gauss-Legendre mapeados a [w_lo, w_hi] ------------------
    med = 0.5 * (w_hi + w_lo)[:, None]
    semi = 0.5 * (w_hi - w_lo)[:, None]
    w = med + semi * x_gl
    peso = semi * w_gl

    sh = np.sinh(w)
    r_w = np.where(esc[:, None], r4.real[:, None] + delta[:, None] * sh * sh,
                   np.where(cae_c[:, None], mu_c[:, None] + nu_c[:, None] * sh,
                            mu_r[:, None] + kap_r[:, None] * np.cosh(w)))

    raiz_peq = np.sqrt(np.maximum((r_w - peq_a[:, None]) * (r_w - peq_b[:, None]),
                                  1e-300))
    dlam = C[:, None] * peso / raiz_peq
    lam = dlam.sum(1)
    phi = (km_f_r(r_w, xi[:, None], a) * dlam).sum(1)
    return phi, lam


def km_f_r(r, xi, a):
    """f_r vectorizado sobre la malla de nodos (mismo que f_r_exacta)."""
    delta = r * r - 2.0 * r + a * a
    return a * (2.0 * r - a * xi) / delta


def refinar_r(r0, lam_obj, raices: np.ndarray, r_ancla, escapa,
              n_iter: int = 3):
    """Pule la r que dio la red imponiendo el lambda EXACTO, por Newton.

    La red devuelve r(lambda) con un error pequeno pero no nulo. Como el
    lambda de cada cruce se conoce exacto (los cruces caen en fases fijas del
    ciclo polar) y la relacion inversa lambda(r) tambien es exacta (Carlson),
    se puede cerrar el lazo: basta corregir r hasta que su lambda coincida.

    Newton se hace en la variable w de integral_phi_r, no en r. En r la
    derivada dlambda/dr = 1/sqrt(R) DIVERGE en el punto de retorno, que es
    justo donde caen los rayos interesantes; en w vale C/sqrt((r-r1)(r-r2)),
    que es finita y suave en todo el dominio.

    Con la red como semilla bastan 2-3 iteraciones. Sin semilla haria falta
    una biseccion de ~50 pasos: la red es lo que hace barato el refinamiento.
    """
    r0 = np.atleast_1d(np.asarray(r0, dtype=np.float64))
    lam_obj = np.atleast_1d(np.asarray(lam_obj, dtype=np.float64))
    r_ancla = np.atleast_1d(np.asarray(r_ancla, dtype=np.float64))
    escapa = np.atleast_1d(np.asarray(escapa, dtype=bool))

    r3, r4 = raices[2], raices[3]
    peq_a, peq_b = raices[0].real, raices[1].real
    complejo = np.abs(r4.imag) > 1e-13 * np.maximum(1.0, np.abs(r4.real))

    esc = escapa
    delta = np.maximum(np.where(esc, r4.real - r3.real, 1.0), 1e-300)
    mu_c, nu_c = r4.real, np.maximum(np.abs(r4.imag), 1e-300)
    mu_r = 0.5 * (r3.real + r4.real)
    kap_r = np.maximum(0.5 * (r4.real - r3.real), 1e-300)
    cae_c = (~esc) & complejo
    C = np.where(esc, 2.0, 1.0)

    def r_de_w(w):
        sh = np.sinh(w)
        return np.where(esc, r4.real + delta * sh * sh,
                        np.where(cae_c, mu_c + nu_c * sh,
                                 mu_r + kap_r * np.cosh(w)))

    def w_de_r(r):
        return np.where(
            esc, np.arcsinh(np.sqrt(np.maximum(r - r4.real, 0.0) / delta)),
            np.where(cae_c, np.arcsinh((r - mu_c) / nu_c),
                     np.arccosh(np.maximum((r - mu_r) / kap_r, 1.0))))

    w = w_de_r(np.maximum(r0, r_ancla))
    for _ in range(n_iter):
        r = r_de_w(w)
        lam = integral_mino_radial(r_ancla, r, raices)
        dlam_dw = C / np.sqrt(np.maximum((r - peq_a) * (r - peq_b), 1e-300))
        w = w - (lam - lam_obj) / np.maximum(dlam_dw, 1e-300)
        w = np.maximum(w, w_de_r(r_ancla))
    return r_de_w(w)


# ===================================================== cuadraturas polares
def _Pi_incompleta(n, sn, cn, dn):
    """Pi(n; am(v), m) evaluada a partir de las funciones de Jacobi en v.

    Pi(n; phi, m) = sin(phi) R_F(cos^2, 1 - m sin^2, 1)
                    + (n/3) sin^3(phi) R_J(cos^2, 1 - m sin^2, 1, 1 - n sin^2)

    y con phi = am(v) se tiene sin(phi) = sn, cos^2(phi) = cn^2,
    1 - m sin^2(phi) = dn^2. Asi no hace falta calcular am() en ningun sitio.
    """
    c2, d2 = cn * cn, dn * dn
    return (sn * elliprf(c2, d2, 1.0)
            + (n / 3.0) * sn**3 * elliprj(c2, d2, 1.0, 1.0 - n * sn * sn))


def G_phi_semiperiodo(u_mas, m, nu):
    """El valor de G_phi sobre medio periodo de theta, x = 1/2.

    Es la pieza que hace que las vueltas multiples no le cuesten nada a la red:
    G_phi(x) = floor(2x) * G_medio + G_phi(x mod 1/2), porque el integrando
    1/(1-mu^2) tiene periodo 1/2 en x. El rayo puede dar 30 vueltas cerca del
    anillo de fotones y todas menos la ultima son una multiplicacion.
    """
    kk = ellipk(m)
    sn0, cn0, dn0 = jacobi_m_negativo(kk, m)
    return 2.0 * _Pi_incompleta(u_mas, sn0, cn0, dn0) / nu


def G_phi_exacto(x, u_mas, m, nu):
    """int_0^{x Lam_theta} dlambda / (1 - mu^2), la cuadratura polar de phi.

    Es la parte de phi que viene de theta: dphi/dlambda incluye xi/sin^2(theta)
    = xi/(1 - mu^2). Con mu = sqrt(u_mas) sn(v) sale una integral eliptica de
    TERCERA especie con caracteristica n = u_mas.

    OJO con el rango. La formula de Pi via las funciones de Jacobi devuelve la
    RAMA PRINCIPAL: para |v| > K el angulo am(v) se pliega dentro de
    [-pi/2, pi/2] y se pierden los cuasi-periodos acumulados. En un rayo que da
    varias vueltas cerca del anillo de fotones eso se nota como un phi
    desviado en multiplos de pi. Hay que reducir a mano con la periodicidad
    del integrando, que en la variable x es exactamente 1/2:

        G_phi(x) = floor(2x) G_medio + G_phi(x - floor(2x)/2)

    Sale en forma cerrada, asi que en principio no hace falta red para phi. Se
    conserva como referencia exacta y como camino de respaldo del renderer.
    """
    x = np.asarray(x, dtype=np.float64)
    n_medios = np.floor(2.0 * x)
    x_red = x - 0.5 * n_medios                       # en [0, 1/2)

    kk = ellipk(m)
    sn0, cn0, dn0 = jacobi_m_negativo(kk, m)                 # v = K  (x = 0)
    snx, cnx, dnx = jacobi_m_negativo(kk * (1.0 - 4.0 * x_red), m)
    J0 = _Pi_incompleta(u_mas, sn0, cn0, dn0)
    Jx = _Pi_incompleta(u_mas, snx, cnx, dnx)
    return n_medios * (2.0 * J0 / nu) + (J0 - Jx) / nu


def fase_camara_theta(theta_obs: float, beta, u_mas, m):
    """Fase x de la camara dentro del ciclo de theta, con signo.

    Invirtiendo mu = sqrt(u_mas) sn(K(1-4x), m) se obtiene x en forma cerrada
    con la integral eliptica incompleta de primera especie:

        v = F(arcsin(mu_obs/sqrt(u_mas)), m),   x_desc = (1 - v/K)/4

    Esa x_desc esta en [0, 1/2] y corresponde a la rama en la que mu BAJA. El
    signo lo decide beta: dtheta/dlambda = p_theta = beta en la camara, o sea
    dmu/dlambda = -sin(theta_obs) beta. Con beta > 0 el rayo va por la rama
    descendente (x_cam = +x_desc); con beta < 0 por la ascendente, que por
    paridad de mu es la misma curva reflejada (x_cam = -x_desc).

    Con eso la fase avanza SIEMPRE como x(sigma) = x_cam + sigma/Lam_theta,
    con sigma el tiempo de Mino recorrido desde la camara. Un solo signo, sin
    ramas: ese era justo el error clasico que evita la forma hamiltoniana, y
    aqui se elimina de otra manera.
    """
    mu_obs = np.cos(theta_obs)
    y = np.clip(mu_obs / np.sqrt(u_mas), -1.0, 1.0)
    v = ellipkinc(np.arcsin(y), m)
    x_desc = 0.25 * (1.0 - v / ellipk(m))
    signo = np.where(np.asarray(beta) < 0.0, -1.0, 1.0)
    return signo * x_desc


# ======================================================= verdad de referencia
def trazar_mino_r(xi: float, eta: float, a: float, escapa: bool,
                  r_ancla: float, r_max: float = 2000.0,
                  lam_max: float = 1e4, rtol: float = 1e-12,
                  atol: float = 1e-14):
    """Integra r(lambda) y Phi_r(lambda) desde el anclaje. Verdad de referencia.

    Se integra la forma de SEGUNDO orden, r'' = R'(r)/2, que se obtiene
    derivando r'^2 = R(r). Asi desaparecen la raiz cuadrada y su signo: el
    punto de retorno se atraviesa solo, igual que en la forma hamiltoniana,
    pero con UNA sola variable en vez de cinco y sin la camara dentro.

    |r'^2 - R(r)| es un invariante exacto de esa EDO y no se impone en ningun
    sitio, asi que sirve de test independiente de la calidad del trazado --
    el mismo papel que juega H en kerr_integrator.
    """
    def rhs(lam, s):
        r, rp, _ = s
        return [rp, 0.5 * dR_dr(r, xi, eta, a), f_r_exacta(r, xi, a)]

    if escapa:
        rp0 = 0.0                       # r_ancla = r4, punto de retorno
    else:
        rp0 = float(np.sqrt(max(potencial_R(r_ancla, xi, eta, a), 0.0)))

    fuera = lambda l, s: s[0] - r_max
    fuera.terminal = True
    fuera.direction = 1

    sol = solve_ivp(rhs, (0.0, lam_max), [float(r_ancla), rp0, 0.0],
                    events=[fuera], rtol=rtol, atol=atol,
                    method="DOP853", dense_output=True)
    return sol


def trazar_mino_theta(xi: float, eta: float, a: float, u_mas: float,
                      lam_half: float, rtol: float = 1e-12,
                      atol: float = 1e-14):
    """Integra mu(lambda) y G(lambda) sobre medio periodo. Verdad de referencia.

    Igual que en r: se deriva mu'^2 = W(mu) para obtener mu'' = W'(mu)/2 y se
    arranca en el punto de retorno mu = sqrt(u_mas), mu' = 0. G acumula la
    cuadratura de phi.
    """
    def rhs(lam, s):
        mu, mup, _ = s
        return [mup, 0.5 * dW_dmu(mu, xi, eta, a), 1.0 / (1.0 - mu * mu)]

    return solve_ivp(rhs, (0.0, float(lam_half)),
                     [float(np.sqrt(u_mas)), 0.0, 0.0],
                     rtol=rtol, atol=atol, method="DOP853", dense_output=True)
