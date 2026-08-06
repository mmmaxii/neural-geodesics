"""Metrica de Kerr en coordenadas de Boyer-Lindquist.

Que cambia respecto a Schwarzschild
-----------------------------------
Todo lo que hacia facil a Schwarzschild se rompe:

1. La simetria deja de ser esferica y pasa a ser AXIAL. El movimiento ya no
   es plano: el arrastre de marcos saca al foton de su plano inicial. Por eso
   la reduccion de Binet (2 EDO en u(phi)) no existe aqui, y por eso la tabla
   1D del renderer clasico deja de servir: la trayectoria ya no depende solo
   de beta.

2. Hacen falta TRES constantes de movimiento, no dos. Las dos obvias siguen
   siendo E y L_z (por los Killing d_t y d_phi), pero no bastan para fijar la
   orbita. La tercera es la constante de CARTER Q, que no viene de ninguna
   simetria evidente sino de que la ecuacion de Hamilton-Jacobi resulta
   separable en estas coordenadas. Sin ella el sistema no es integrable.

3. g_t_phi != 0: la metrica es ESTACIONARIA pero no ESTATICA. De ahi salen
   el arrastre de marcos y la ergosfera.

4. Hay dos horizontes (Delta = 0) en vez de uno, y la ergosfera queda FUERA
   del horizonte exterior: dentro de ella ningun observador puede quedarse
   quieto, pero todavia puede escapar.

5. La sombra deja de ser un circulo. Se aplana del lado que se acerca, porque
   las orbitas de fotones progradas se pegan mucho mas al agujero que las
   retrogradas.

Convenios
---------
Unidades geometrizadas G = c = 1, igual que en schwarzschild.py. El espin se
guarda como `a = J/M` (dimensiones de longitud). El parametro adimensional es
`a_star = a/M`, entre -1 y 1; |a| > M seria una singularidad desnuda y se
rechaza. a = 0 devuelve Schwarzschild exactamente.

Signatura (-,+,+,+), coordenadas ordenadas (t, r, theta, phi).
"""
from __future__ import annotations

import math

import numpy as np

import utils.constants as c


class KMetric:
    """Metrica de Kerr. Con a=0 se reduce identicamente a SMetric."""

    def __init__(self, M: float = c.DEFAULT_M, a: float = 0.0):
        if M <= 0:
            raise ValueError("la masa debe ser positiva")
        if abs(a) > M:
            raise ValueError(
                f"|a| = {abs(a)} > M = {M}: eso es una singularidad desnuda, "
                "no un agujero negro. El limite extremo es |a| = M.")
        self.M = float(M)
        self.a = float(a)
        self.a_star = self.a / self.M          # espin adimensional en [-1, 1]
        self.r_s = c.RS_FACTOR * self.M        # 2M, por continuidad con SMetric

        # Horizontes: raices de Delta = 0. El de fuera es el horizonte de
        # sucesos; el de dentro es el de Cauchy.
        disc = self.M**2 - self.a**2
        root = np.sqrt(max(disc, 0.0))
        self.r_horizon = self.M + root          # r_+
        self.r_horizon_inner = self.M - root    # r_-

    # ------------------------------------------------------------ auxiliares
    def sigma(self, r, theta):
        """Sigma = r^2 + a^2 cos^2(theta). Es el factor conforme de la metrica."""
        return np.asarray(r) ** 2 + self.a**2 * np.cos(np.asarray(theta)) ** 2

    def delta(self, r):
        """Delta = r^2 - 2Mr + a^2. Se anula en los horizontes.

        Es el analogo de f_aux = 1 - r_s/r de Schwarzschild: marca donde la
        metrica se vuelve singular en estas coordenadas (singularidad de
        COORDENADAS, no de curvatura).
        """
        r = np.asarray(r)
        return r**2 - 2.0 * self.M * r + self.a**2

    def a_func(self, r, theta):
        """A = (r^2+a^2)^2 - a^2 Delta sin^2(theta). Aparece en g_phiphi."""
        r = np.asarray(r)
        s2 = np.sin(np.asarray(theta)) ** 2
        return (r**2 + self.a**2) ** 2 - self.a**2 * self.delta(r) * s2

    # -------------------------------------------------------- metrica (cov.)
    def g_tt(self, r, theta):
        return -(1.0 - 2.0 * self.M * np.asarray(r) / self.sigma(r, theta))

    def g_tphi(self, r, theta):
        """Termino cruzado. Es CERO en Schwarzschild; aqui es el arrastre.

        Ojo con el convenio: en ds^2 el termino cruzado aparece como
        2 g_tphi dt dphi, asi que g_tphi = -2Mar sin^2(theta)/Sigma.
        """
        r = np.asarray(r)
        s2 = np.sin(np.asarray(theta)) ** 2
        return -2.0 * self.M * self.a * r * s2 / self.sigma(r, theta)

    def g_rr(self, r, theta):
        return self.sigma(r, theta) / self.delta(r)

    def g_thetatheta(self, r, theta):
        return self.sigma(r, theta)

    def g_phiphi(self, r, theta):
        s2 = np.sin(np.asarray(theta)) ** 2
        return self.a_func(r, theta) * s2 / self.sigma(r, theta)

    def metric_matrix(self, r, theta) -> np.ndarray:
        """Matriz 4x4 g_mu_nu en el orden (t, r, theta, phi)."""
        g = np.zeros((4, 4), float)
        g[0, 0] = self.g_tt(r, theta)
        g[1, 1] = self.g_rr(r, theta)
        g[2, 2] = self.g_thetatheta(r, theta)
        g[3, 3] = self.g_phiphi(r, theta)
        g[0, 3] = g[3, 0] = self.g_tphi(r, theta)
        return g

    # ---------------------------------------------------- metrica (contrav.)
    # Distancia minima al eje de giro. Ver la nota en inverse_metric_matrix.
    AXIS_EPS = 1e-8

    def inverse_metric_matrix(self, r, theta) -> np.ndarray:
        """Matriz 4x4 g^mu^nu, en forma cerrada.

        Se escribe a mano en vez de invertir numericamente porque el bloque
        (t, phi) tiene inversa analitica simple y porque esto se llama en el
        lado derecho del integrador, millones de veces.

        AVISO sobre el eje de giro
        --------------------------
        g^phiphi lleva un sin^2(theta) en el denominador, asi que DIVERGE en
        theta = 0 y theta = pi. Eso no es un fallo de esta clase: es una
        singularidad de COORDENADAS de Boyer-Lindquist, igual que la que tiene
        cualquier sistema esferico en los polos. La curvatura ahi es finita.

        Como paliativo se aparta theta del eje por AXIS_EPS. Es un PARCHE, no
        una solucion: un rayo que pase muy cerca del eje seguira perdiendo
        precision. La solucion de verdad es cambiar a coordenadas de
        Kerr-Schild, que son regulares en el eje y en el horizonte. Pendiente.
        """
        r = np.asarray(r, float)
        theta = np.clip(np.asarray(theta, float),
                        self.AXIS_EPS, np.pi - self.AXIS_EPS)
        S, D = self.sigma(r, theta), self.delta(r)
        A = self.a_func(r, theta)
        s2 = np.sin(theta) ** 2

        ginv = np.zeros((4, 4), float)
        ginv[0, 0] = -A / (S * D)
        ginv[1, 1] = D / S
        ginv[2, 2] = 1.0 / S
        ginv[3, 3] = (D - self.a**2 * s2) / (S * D * s2)
        ginv[0, 3] = ginv[3, 0] = -2.0 * self.M * self.a * r / (S * D)
        return ginv

    # -------------------------------------------------------- camino rapido
    def inverse_components(self, r: float, theta: float):
        """Las 5 componentes independientes de g^ab, como floats de Python.

        Existe aparte de inverse_metric_matrix() por rendimiento: aquella
        asigna un array 4x4 nuevo en cada llamada y pasa por numpy para operar
        con escalares, lo que en el lado derecho del integrador domina el
        tiempo total (np.asarray llegaba a aparecer ~800k veces por un puñado
        de rayos). Aqui se usa math sobre floats y no se asigna nada.

        Devuelve (gtt, gtp, grr, gthth, gpp).
        """
        M, a = self.M, self.a
        if theta < self.AXIS_EPS:
            theta = self.AXIS_EPS
        elif theta > math.pi - self.AXIS_EPS:
            theta = math.pi - self.AXIS_EPS
        s = math.sin(theta)
        c = math.cos(theta)
        s2 = s * s
        a2 = a * a
        r2 = r * r

        S = r2 + a2 * c * c
        D = r2 - 2.0 * M * r + a2
        A = (r2 + a2) ** 2 - a2 * D * s2
        P = S * D

        return (-A / P, -2.0 * M * a * r / P, D / S, 1.0 / S,
                (D - a2 * s2) / (P * s2))

    def inverse_components_and_derivs(self, r: float, theta: float):
        """Igual que inverse_components(), mas sus derivadas en r y en theta.

        Las derivadas son ANALITICAS. Antes el integrador las sacaba por
        diferencias centradas, lo que costaba 4 evaluaciones extra de la
        metrica por cada evaluacion del lado derecho: el 80% del trabajo era
        calcular derivadas. Aqui salen del mismo bloque de cuentas, casi
        gratis, y ademas sin el error de truncamiento del paso finito.

        Devuelve (comp, d_dr, d_dtheta), cada una la tupla de 5 componentes.
        """
        M, a = self.M, self.a
        if theta < self.AXIS_EPS:
            theta = self.AXIS_EPS
        elif theta > math.pi - self.AXIS_EPS:
            theta = math.pi - self.AXIS_EPS
        s = math.sin(theta)
        c = math.cos(theta)
        s2 = s * s
        sc = s * c
        a2 = a * a
        r2 = r * r

        S = r2 + a2 * c * c
        D = r2 - 2.0 * M * r + a2
        r2a2 = r2 + a2
        A = r2a2 * r2a2 - a2 * D * s2
        P = S * D
        P2 = P * P
        S2 = S * S
        N = D - a2 * s2                       # numerador de g^phiphi

        comp = (-A / P, -2.0 * M * a * r / P, D / S, 1.0 / S, N / (P * s2))

        # ---- derivadas en r  (theta fijo: Delta_r = 2(r-M), Sigma_r = 2r)
        S_r = 2.0 * r
        D_r = 2.0 * (r - M)
        A_r = 4.0 * r * r2a2 - a2 * s2 * D_r
        P_r = S_r * D + S * D_r
        d_dr = (
            -(A_r * P - A * P_r) / P2,
            -2.0 * M * a * (P - r * P_r) / P2,
            (D_r * S - D * S_r) / S2,
            -S_r / S2,
            (D_r * P - N * P_r) / (P2 * s2),
        )

        # ---- derivadas en theta  (r fijo: Delta no depende de theta)
        S_t = -2.0 * a2 * sc
        A_t = -2.0 * a2 * D * sc
        P_t = S_t * D
        Dden = P * s2                          # denominador de g^phiphi
        Dden_t = P_t * s2 + P * 2.0 * sc
        d_dt = (
            -(A_t * P - A * P_t) / P2,
            2.0 * M * a * r * P_t / P2,
            -D * S_t / S2,
            -S_t / S2,
            (-2.0 * a2 * sc * Dden - N * Dden_t) / (Dden * Dden),
        )
        return comp, d_dr, d_dt

    def inverse_components_and_derivs_vec(self, r, theta):
        """Version vectorizada de inverse_components_and_derivs().

        Mismas formulas, pero sobre arrays: permite avanzar MILES de rayos en
        una sola llamada. Es la diferencia entre pagar el overhead de Python
        una vez por rayo y pagarlo una vez por paso de integracion para todos.

        r y theta son arrays de la misma forma; devuelve tres tuplas de cinco
        arrays cada una: (componentes, d/dr, d/dtheta).
        """
        M, a = self.M, self.a
        theta = np.clip(theta, self.AXIS_EPS, np.pi - self.AXIS_EPS)
        s = np.sin(theta)
        c_ = np.cos(theta)
        s2 = s * s
        sc = s * c_
        a2 = a * a
        r2 = r * r

        S = r2 + a2 * c_ * c_
        D = r2 - 2.0 * M * r + a2
        r2a2 = r2 + a2
        A = r2a2 * r2a2 - a2 * D * s2
        P = S * D
        P2 = P * P
        S2 = S * S
        N = D - a2 * s2

        comp = (-A / P, -2.0 * M * a * r / P, D / S, 1.0 / S, N / (P * s2))

        S_r = 2.0 * r
        D_r = 2.0 * (r - M)
        A_r = 4.0 * r * r2a2 - a2 * s2 * D_r
        P_r = S_r * D + S * D_r
        d_dr = (-(A_r * P - A * P_r) / P2,
                -2.0 * M * a * (P - r * P_r) / P2,
                (D_r * S - D * S_r) / S2,
                -S_r / S2,
                (D_r * P - N * P_r) / (P2 * s2))

        S_t = -2.0 * a2 * sc
        A_t = -2.0 * a2 * D * sc
        P_t = S_t * D
        Dden = P * s2
        Dden_t = P_t * s2 + P * 2.0 * sc
        d_dt = (-(A_t * P - A * P_t) / P2,
                2.0 * M * a * r * P_t / P2,
                -D * S_t / S2,
                -S_t / S2,
                (-2.0 * a2 * sc * Dden - N * Dden_t) / (Dden * Dden))
        return comp, d_dr, d_dt

    def inverse_components_and_derivs_xp(self, r, theta, xp):
        """Igual que la version vectorizada, pero valida para numpy Y torch.

        Existe para poder correr el trazado entero en GPU. Las formulas son las
        mismas; lo unico que cambia es que aqui no se usa nada especifico de
        numpy, asi que el mismo codigo sirve con tensores de torch en CUDA.
        """
        M, a = self.M, self.a
        lo, hi = self.AXIS_EPS, np.pi - self.AXIS_EPS
        theta = theta.clamp(lo, hi) if hasattr(theta, "clamp") else np.clip(theta, lo, hi)
        s = xp.sin(theta)
        c_ = xp.cos(theta)
        s2 = s * s
        sc = s * c_
        a2 = a * a
        r2 = r * r

        S = r2 + a2 * c_ * c_
        D = r2 - 2.0 * M * r + a2
        r2a2 = r2 + a2
        A = r2a2 * r2a2 - a2 * D * s2
        P = S * D
        P2 = P * P
        S2 = S * S
        N = D - a2 * s2

        comp = (-A / P, -2.0 * M * a * r / P, D / S, 1.0 / S, N / (P * s2))

        S_r = 2.0 * r
        D_r = 2.0 * (r - M)
        A_r = 4.0 * r * r2a2 - a2 * s2 * D_r
        P_r = S_r * D + S * D_r
        d_dr = (-(A_r * P - A * P_r) / P2,
                -2.0 * M * a * (P - r * P_r) / P2,
                (D_r * S - D * S_r) / S2,
                -S_r / S2,
                (D_r * P - N * P_r) / (P2 * s2))

        S_t = -2.0 * a2 * sc
        A_t = -2.0 * a2 * D * sc
        P_t = S_t * D
        Dden = P * s2
        Dden_t = P_t * s2 + P * 2.0 * sc
        d_dt = (-(A_t * P - A * P_t) / P2,
                2.0 * M * a * r * P_t / P2,
                -D * S_t / S2,
                -S_t / S2,
                (-2.0 * a2 * sc * Dden - N * Dden_t) / (Dden * Dden))
        return comp, d_dr, d_dt

    # ------------------------------------------------------------ estructura
    def r_ergosphere(self, theta):
        """Borde exterior de la ergosfera: donde g_tt = 0.

        Toca el horizonte en los polos y se abulta al maximo en el ecuador,
        donde llega a 2M sea cual sea el espin. Dentro de ella ningun
        observador puede permanecer estatico: el arrastre lo obliga a girar.
        """
        ct2 = np.cos(np.asarray(theta)) ** 2
        return self.M + np.sqrt(np.maximum(self.M**2 - self.a**2 * ct2, 0.0))

    def r_photon_equatorial(self, prograde: bool = True) -> float:
        """Radio de la orbita circular de fotones en el ecuador.

        Para a=0 da 3M (= 1.5 r_s) en los dos sentidos. Al subir el espin la
        prograda se hunde hasta M y la retrograda se aleja hasta 4M: el
        arrastre ayuda al foton que gira con el agujero y estorba al que va
        en contra.

        `prograde` significa CO-ROTANTE con el agujero, sea cual sea el signo
        de a. Por eso se usa |a|: con a<0 el agujero gira al reves, pero el
        foton co-rotante sigue siendo el que se hunde hasta M. Sin el valor
        absoluto esta funcion devolvia la orbita contraria con a<0, mientras
        que r_isco (que si usa abs) devolvia la correcta: las dos se
        contradecian en silencio.
        """
        sign = -1.0 if prograde else 1.0
        x = np.clip(sign * abs(self.a) / self.M, -1.0, 1.0)
        return float(2.0 * self.M * (1.0 + np.cos(2.0 / 3.0 * np.arccos(x))))

    def r_isco(self, prograde: bool = True) -> float:
        """Ultima orbita circular estable para materia (formula de Bardeen).

        a=0 -> 6M (= 3 r_s, el X_ISCO de disk.py). a=M -> M prograda, 9M
        retrograda. Esto mueve el borde interior del disco y por tanto su
        temperatura maxima.
        """
        astar = abs(self.a_star) * (1.0 if prograde else -1.0)
        z1 = 1.0 + (1.0 - astar**2) ** (1.0 / 3.0) * (
            (1.0 + astar) ** (1.0 / 3.0) + (1.0 - astar) ** (1.0 / 3.0))
        z2 = np.sqrt(3.0 * astar**2 + z1**2)
        sign = -1.0 if prograde else 1.0
        return float(self.M * (3.0 + z2 + sign * np.sqrt(
            (3.0 - z1) * (3.0 + z1 + 2.0 * z2))))

    def omega_frame_drag(self, r, theta):
        """Velocidad angular del arrastre de marcos: -g_tphi/g_phiphi.

        Es la velocidad a la que un observador en caida libre desde el
        infinito y sin momento angular (ZAMO) gira igualmente. En
        Schwarzschild es identicamente cero.
        """
        return -self.g_tphi(r, theta) / self.g_phiphi(r, theta)

    # ----------------------------------------------------------- disco (Kerr)
    def omega_kepler(self, r, prograde: bool = True):
        """Velocidad angular de una orbita circular ecuatorial de materia.

            Omega = +- sqrt(M) / (r^{3/2} +- a sqrt(M))

        El termino `a sqrt(M)` del denominador es el arrastre: a igual radio,
        el gas prograde gira MAS DESPACIO que en Schwarzschild y el retrogrado
        mas rapido. Con a=0 se reduce a la kepleriana de siempre,
        Omega = sqrt(M)/r^{3/2}.
        """
        s = 1.0 if prograde else -1.0
        sm = np.sqrt(self.M)
        return s * sm / (np.asarray(r) ** 1.5 + s * self.a * sm)

    def redshift_g(self, r, xi, prograde: bool = True):
        """Factor de corrimiento nu_obs/nu_emit para el gas del disco.

            g = 1 / [ u^t (1 - Omega xi) ]

        con xi = L_z/E del FOTON (constante a lo largo del rayo) y u^t el
        factor de dilatacion temporal del emisor. Reune las tres cosas a la
        vez: Doppler por la rotacion, dilatacion gravitatoria, y el arrastre
        metido dentro de Omega y de u^t.

        Es el mismo papel que juega disk.redshift_g en Schwarzschild, pero
        aqui no se puede separar en factores porque g_tphi != 0 los mezcla.

        Devuelve 0 donde la combinacion (r, xi) es IMPOSIBLE. El denominador
        u^t (1 - Omega xi) es -p.u, o sea la energia del foton medida por el
        gas; si sale <= 0 ese elemento de disco no pudo emitir ese foton hacia
        nosotros. Sin este cuidado el denominador pasa por cero y g explota:
        como el brillo va con g^4, un solo pixel asi quema la imagen entera.
        """
        r = np.asarray(r, float)
        om = self.omega_kepler(r, prograde)
        # En el ecuador Sigma = r^2, asi que las componentes se simplifican
        gtt = -(1.0 - 2.0 * self.M / r)
        gtp = -2.0 * self.M * self.a / r
        gpp = (r * r + self.a**2 + 2.0 * self.M * self.a**2 / r)
        # normalizacion u.u = -1 para u^mu = u^t (1, 0, 0, Omega)
        norm = -(gtt + 2.0 * gtp * om + gpp * om * om)
        valido = norm > 0.0
        ut = 1.0 / np.sqrt(np.where(valido, norm, 1.0))
        denom = ut * (1.0 - om * np.asarray(xi, float))
        return np.where(valido & (denom > 1e-6), 1.0 / np.where(denom > 1e-6, denom, 1.0), 0.0)

    # ---------------------------------------------------------------- sombra
    # Por debajo de esta inclinacion la parametrizacion de Bardeen degenera.
    MIN_INCLINATION_DEG = 0.5

    def shadow_outline(self, inclination_deg: float = 90.0, n: int = 720):
        """Contorno de la sombra visto desde el infinito (Bardeen).

        Devuelve (alpha, beta) en unidades de M: las coordenadas celestes en
        el plano imagen del observador lejano, con el convenio de Bardeen:

            alpha : PERPENDICULAR al eje de rotacion proyectado. Sale de
                    xi = L_z/E, o sea del momento angular azimutal.
            beta  : PARALELO al eje de rotacion proyectado. Sale de eta, la
                    constante de Carter, que gobierna el movimiento en theta.

        Por eso el aplanamiento por el espin aparece en alpha (horizontal si
        se dibuja alpha en x) y la altura en beta no cambia nunca con a.

        Se parametriza por el radio r de las orbitas ESFERICAS de fotones (las
        que mantienen r constante pero recorren theta). Cada una de esas
        orbitas es el limite de un rayo que casi queda atrapado, asi que su
        par (xi, eta) dibuja el borde de la sombra.

        Con a=0 sale un circulo exacto de radio 3*sqrt(3) M = 5.196 M, que es
        el 2.598 r_s de BETA_CRIT. Al crecer el espin el circulo se DESPLAZA y
        se APLANA por el lado que se acerca.

        `n` es el numero de radios muestreados, NO el numero de puntos
        devueltos: el contorno se cierra reflejando en beta y ademas se
        descartan los radios sin solucion real, asi que la salida tiene como
        mucho 2n puntos y en general menos.
        """
        M, a = self.M, self.a
        if inclination_deg < self.MIN_INCLINATION_DEG:
            raise ValueError(
                f"inclinacion {inclination_deg} deg demasiado baja: por debajo "
                f"de {self.MIN_INCLINATION_DEG} deg la parametrizacion de "
                "Bardeen degenera (alpha = -xi/sin(theta_o) explota y el "
                "discriminante de beta pierde casi todos los radios validos). "
                "Vista de polo la sombra es un circulo; tratalo como caso "
                "aparte en vez de confiar en esta formula.")
        theta_o = np.deg2rad(inclination_deg)

        if abs(a) < 1e-12:
            # Limite de Schwarzschild: circulo exacto, la formula general es 0/0
            rad = 3.0 * np.sqrt(3.0) * M
            t = np.linspace(0.0, 2.0 * np.pi, n)
            return rad * np.cos(t), rad * np.sin(t)

        # Radios de las orbitas esfericas de fotones: entre la prograda y la
        # retrograda ecuatoriales.
        r1 = self.r_photon_equatorial(prograde=True)
        r2 = self.r_photon_equatorial(prograde=False)
        # Muestreo de Chebyshev en vez de uniforme: los extremos en alpha caen
        # justo en r1 y r2, y con linspace el error del ancho de la sombra baja
        # solo como 1/n. Agrupando puntos en los bordes se captura el extremo
        # mucho mejor con el mismo n.
        t = np.linspace(0.0, np.pi, n)
        lo, hi = min(r1, r2), max(r1, r2)
        r = 0.5 * (lo + hi) - 0.5 * (hi - lo) * np.cos(t)

        # Constantes de movimiento de la orbita esferica de radio r
        # xi = L_z/E,  eta = Q/E^2
        xi = ((r**2 * (3.0 * M - r) - a**2 * (r + M)) / (a * (r - M)))
        eta = (r**3 * (4.0 * M * a**2 - r * (r - 3.0 * M) ** 2)
               / (a**2 * (r - M) ** 2))

        st, ct = np.sin(theta_o), np.cos(theta_o)
        # Coordenadas celestes del observador lejano
        alpha = -xi / st
        disc = eta + a**2 * ct**2 - xi**2 * (ct / st) ** 2
        ok = np.isfinite(alpha) & np.isfinite(disc) & (disc >= 0.0)
        if not ok.any():
            raise ValueError(
                f"ningun radio da solucion real con a*={self.a_star:.4g} e "
                f"i={inclination_deg} deg; el contorno saldria vacio")

        # Quedarse SOLO con el tramo contiguo mas largo. Si `ok` fuese
        # True/False/True, concatenar sin mas uniria dos arcos separados y el
        # contorno saldria cruzado sobre si mismo.
        idx = np.flatnonzero(ok)
        cortes = np.flatnonzero(np.diff(idx) > 1)
        tramos = np.split(idx, cortes + 1)
        idx = max(tramos, key=len)

        beta = np.sqrt(np.maximum(disc[idx], 0.0))
        # el contorno es simetrico respecto al eje alpha: media arriba, media abajo
        al = np.concatenate([alpha[idx], alpha[idx][::-1]])
        be = np.concatenate([beta, -beta[::-1]])
        return al, be

    # ------------------------------------------------------------------ misc
    def __repr__(self) -> str:
        return (f"KMetric(M={self.M:g}, a={self.a:g}, a*={self.a_star:+.3f}, "
                f"r+={self.r_horizon:.4f}, r_isco={self.r_isco():.4f})")
