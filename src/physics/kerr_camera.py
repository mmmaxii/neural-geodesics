"""Camara fisica en Kerr: del pixel a las constantes del rayo, por la tetrada.

Por que existe este modulo
--------------------------
Hasta ahora el plano imagen se construia DIRECTAMENTE en coordenadas celestes
de Bardeen (alpha, beta). Esas coordenadas estan en unidades de M y son
parametros de impacto, NO angulos. La consecuencia se ve a simple vista:

    la sombra mide ~5 M en (alpha, beta) sea cual sea r_obs

asi que con --half-width fijo la sombra ocupaba siempre los mismos pixeles y lo
unico que cambiaba al mover la camara era la escala angular del cielo. El
render decia que el agujero estaba clavado y que lo que se alejaba era el
fondo, que es justo al reves de lo que hace la gravedad.

La cura es la misma que ya usaba el renderer de Schwarzschild
(rendering/classical_renderer.py, Camera.impact_parameter): fijar un CAMPO DE
VISION angular y derivar de ahi los parametros de impacto, con el factor
metrico que traduce angulo local a parametro de impacto. Alli era una linea:

    b = r sin(psi) / sqrt(1 - r_s/r)

Este modulo es esa linea generalizada a Kerr, hecha por el camino correcto: la
tetrada ortonormal del observador local. Con a = 0 se reduce identicamente a la
formula de arriba (lo comprueba scripts/validate_kerr_camera.py), asi que no es
un reescalado ad hoc sino la misma fisica que ya estaba validada.

Los tres actores, y quien mueve a quien
---------------------------------------
    camara      -- en (r_obs, theta_obs), mirando al agujero
    agujero     -- en el origen, quieto
    esfera celeste -- en el infinito, quieta y SIN deformar

El agujero no deforma nada: es un transformador de direcciones. Cada pixel da
una direccion de mirada en el marco LOCAL de la camara; la geodesica la
convierte en una direccion de llegada al cielo; ahi se lee el color. Este
modulo hace el primer paso, que es el que estaba mal.

Ese primer paso es tambien el que responde a la pregunta de la estrella que
esta por ENCIMA de la camara y cuya luz baja, se curva y entra por el objetivo.
El trazado va hacia atras, asi que esa estrella aparece sola: sale del pixel
cuya direccion local, propagada, termina apuntando a ella. Lo que hacia falta
para que eso saliera en el sitio correcto es que el mapa pixel -> direccion
local fuera fisicamente cierto a r_obs FINITO, no solo en campo lejano. Es
exactamente lo que arregla la tetrada.

Por que ZAMO y no un observador estatico
----------------------------------------
El observador de momento angular nulo (e_t proporcional a grad t) es el marco
natural en Kerr: existe hasta el horizonte, mientras que el estatico deja de
existir al entrar en la ergosfera. Fuera de la ergosfera los dos son marcos
legitimos y solo se diferencian por un boost azimutal, que es una aberracion
del cielo, no un error. Con a = 0 el ZAMO ES el estatico.

Convenios
---------
M = 1, G = c = 1, salvo que se pase otra KMetric. La salida son las MISMAS
coordenadas de Bardeen (alpha, beta) que consume todo lo de aguas abajo
(kerr_mino.xi_eta_desde_pixel, kerr_integrator.initial_from_celestial,
KMetric.shadow_outline), asi que este modulo se enchufa delante del motor sin
tocar nada mas. El paso por (alpha, beta) no pierde informacion: el mapa
(alpha, beta) <-> (xi, eta, signo de p_theta) es exactamente invertible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Por debajo de esta inclinacion alpha = -xi/sin(theta_obs) degenera, igual que
# en KMetric.shadow_outline. No es un problema de este modulo sino del sistema
# de coordenadas de Bardeen: vista de polo la sombra es un circulo y hay que
# tratarla aparte.
MIN_INCLINACION_DEG = 0.5


@dataclass(frozen=True)
class TetradaZAMO:
    """Las cinco cantidades de la tetrada del ZAMO que hacen falta aqui.

    No se guarda la base como matriz 4x4 porque solo se usa para bajar indices
    sobre un vector nulo, y eso se reduce a estos escalares.
    """

    lapso: float        # alpha_lapso = sqrt(Sigma Delta / A). Vale 1 en el infinito.
    omega: float        # arrastre de marcos, 2 M a r / A
    raiz_gpp: float     # sqrt(g_phiphi) = sin(theta) sqrt(A/Sigma)
    raiz_sigma: float   # sqrt(Sigma)
    raiz_sigma_delta: float  # sqrt(Sigma/Delta), el factor radial


def tetrada_zamo(k, r: float, theta: float) -> TetradaZAMO:
    """Tetrada ortonormal del observador de momento angular nulo en (r, theta).

    La base es

        e_t = (1/lapso) (d_t + omega d_phi)
        e_r = sqrt(Delta/Sigma) d_r
        e_theta = (1/sqrt(Sigma)) d_theta
        e_phi = (1/sqrt(g_phiphi)) d_phi

    y su comprobacion es g(e_t, e_t) = -1, que sale de la identidad
    g_tt - g_tphi^2/g_phiphi = -Delta Sigma / A. La verifica numericamente
    scripts/validate_kerr_camera.py; no se da por supuesta.
    """
    S = float(k.sigma(r, theta))
    D = float(k.delta(r))
    A = float(k.a_func(r, theta))
    if D <= 0.0:
        raise ValueError(
            f"r = {r} esta dentro del horizonte (Delta = {D} <= 0): ahi no hay "
            "observador estacionario ninguno y la tetrada no existe.")
    return TetradaZAMO(
        lapso=np.sqrt(S * D / A),
        omega=2.0 * k.M * k.a * r / A,
        raiz_gpp=np.sin(theta) * np.sqrt(A / S),
        raiz_sigma=np.sqrt(S),
        raiz_sigma_delta=np.sqrt(S / D),
    )


# ==================================================== direccion local -> cielo
def celestes_desde_direccion(k, r_obs: float, theta_obs: float,
                             n_r, n_theta, n_phi):
    """(alpha, beta) de Bardeen del rayo que la camara ve en la direccion n.

    n = (n_r, n_theta, n_phi) es un vector UNITARIO en el marco ortonormal del
    ZAMO: la direccion hacia la que mira ese pixel. Para un pixel que apunta al
    agujero n_r < 0.

    La cuenta
    ---------
    El fotón tiene p = E_loc (e_t + n^i e_i). Bajando indices con la co-tetrada

        w^t = lapso dt,  w^r = sqrt(Sigma/Delta) dr,
        w^theta = sqrt(Sigma) dtheta,  w^phi = sqrt(g_phiphi) (dphi - omega dt)

    y usando p_a = eta_ab p^b (o sea p_t = -p^t) sale, de una tacada,

        E   = -p_t   = E_loc (lapso + omega sqrt(g_phiphi) n_phi)
        L   =  p_phi = E_loc sqrt(g_phiphi) n_phi
        p_theta      = E_loc sqrt(Sigma) n_theta

    E_loc se cancela en todos los cocientes, que es lo que tiene que pasar: la
    trayectoria no sabe con cuanta energia la lanzaste. Queda

        xi   = L/E     = sqrt(g_phiphi) n_phi / Dn
        beta = p_theta/E = sqrt(Sigma) n_theta / Dn
        alpha = -xi/sin(theta_obs) = -sqrt(A/Sigma) n_phi / Dn

    con Dn = lapso + omega sqrt(g_phiphi) n_phi.

    El denominador Dn es E/E_loc: el corrimiento gravitatorio mas el Doppler
    del arrastre. Es POSITIVO en todo el exterior de la ergosfera (alli d_t es
    temporal, asi que ningun foton puede tener E <= 0) y es justo el factor que
    faltaba: con Dn = 1 se recupera la version ingenua "alpha = r_obs por el
    angulo", que es la que hacia que la sombra no cambiara de tamano.

    Comprobacion de cordura con a = 0
    ---------------------------------
    Sigma = r^2, A = r^4, omega = 0, lapso = sqrt(1 - 2M/r), asi que

        b = sqrt(alpha^2 + beta^2) = r sin(psi) / sqrt(1 - 2M/r)

    que es literalmente Camera.impact_parameter de classical_renderer.py. O
    sea: esto no es codigo nuevo sin validar, es la generalizacion de una
    formula que el renderer clasico ya usaba y que ya estaba comprobada contra
    el tamano exacto de la sombra.
    """
    st = np.sin(theta_obs)
    if abs(st) < np.sin(np.deg2rad(MIN_INCLINACION_DEG)):
        raise ValueError(
            f"inclinacion demasiado cerca del eje (sin(theta_obs) = {st:.2e}): "
            "alpha = -xi/sin(theta_obs) degenera. Es la misma limitacion que "
            "documenta KMetric.shadow_outline.")

    tt = tetrada_zamo(k, r_obs, theta_obs)
    n_phi = np.asarray(n_phi, np.float64)
    n_theta = np.asarray(n_theta, np.float64)

    # n_r no entra en (alpha, beta): esas dos coordenadas solo llevan (xi, eta)
    # y el signo de p_theta. El p_r lo rehace cada consumidor imponiendo que el
    # rayo sea nulo y quedandose con la raiz NEGATIVA -- o sea, dando por hecho
    # que el rayo entra. Aqui se comprueba que esa suposicion es cierta; si no
    # lo fuera, el trazador daria la vuelta al rayo sin decir nada.
    # (validate_kerr_camera.py, prueba 2, comprueba que el p_r reconstruido
    # coincide con el que predice la tetrada, signo incluido)
    if np.any(np.asarray(n_r, np.float64) >= 0.0):
        raise ValueError(
            "hay pixeles cuya mirada se aleja del agujero (n_r >= 0). Las "
            "coordenadas de Bardeen no distinguen ese caso: el trazador "
            "supondria que el rayo entra y devolveria otro rayo distinto.")

    Dn = tt.lapso + tt.omega * tt.raiz_gpp * n_phi
    if np.any(Dn <= 0.0):
        raise ValueError(
            "hay rayos con energia en el infinito no positiva (Dn <= 0). Eso "
            "solo pasa dentro de la ergosfera; la camara esta demasiado cerca.")

    xi = tt.raiz_gpp * n_phi / Dn
    beta = tt.raiz_sigma * n_theta / Dn
    alpha = -xi / st
    return alpha, beta


# ============================================================ pixel -> mirada
def direcciones_pixel(width: int, height: int, fov_deg: float):
    """Pixel -> direccion unitaria de mirada en el marco local, aplanado.

    Camara estenopeica: el plano imagen esta a distancia 1 sobre el eje optico,
    que apunta al agujero (-e_r). fov_deg es el campo de vision HORIZONTAL
    completo, asi que el pixel del borde queda a fov/2 del centro. El eje
    vertical hereda la escala del horizontal, de forma que los pixeles son
    cuadrados; es el mismo reparto que hacian malla_imagen() y
    Camera.pixel_angles().

    Orientacion, que es donde estan todas las trampas
    -------------------------------------------------
    En campo lejano alpha ~ -n_phi y beta ~ +n_theta (ver
    celestes_desde_direccion). El convenio de imagen que hay que reproducir es
    el de render_kerr.py:265-277:

        columna que crece a la derecha  ->  alpha crece   ->  n_phi decrece
        fila 0 (arriba)                ->  beta NEGATIVO ->  n_theta negativo

    de ahi el (-1, ty, -tx). Lo segundo no es una eleccion: beta entra como
    p_theta y theta crece hacia el sur, asi que beta > 0 pone la fuente al SUR.
    Invertirlo espeja el render entero de arriba abajo, y ya fue un bug real.

    Ojo: la convencion de fila es la CONTRARIA a la de Camera.pixel_angles() de
    Schwarzschild, que lleva un -ty. No es incoherencia: alli el signo se
    compensa aguas abajo al construir la direccion con la base de la camara.
    """
    half = np.tan(np.deg2rad(fov_deg) / 2.0)
    tx = np.linspace(-half, half, width)
    ty = np.linspace(-half, half, height) * (height / width)
    TX, TY = np.meshgrid(tx, ty)

    n = np.stack([-np.ones_like(TX), TY, -TX], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n[..., 0].ravel(), n[..., 1].ravel(), n[..., 2].ravel()


def malla_celeste(k, r_obs: float, theta_obs: float, fov_deg: float,
                  width: int, height: int):
    """Sustituto directo de malla_imagen(): devuelve (alpha, beta) aplanados.

    Toda la cadena de aguas abajo -- xi_eta_desde_pixel, precalcular, las dos
    redes, trace_batch -- sigue funcionando sin cambios, porque lo que cambia
    no es el lenguaje (siguen siendo coordenadas de Bardeen) sino QUE valores
    de (alpha, beta) le toca a cada pixel.
    """
    n_r, n_th, n_ph = direcciones_pixel(width, height, fov_deg)
    return celestes_desde_direccion(k, r_obs, theta_obs, n_r, n_th, n_ph)


# ================================================================== auxiliares
def rad_por_pixel(fov_deg: float, width: int) -> float:
    """Angulo que abarca un pixel en el CENTRO de la imagen, en radianes.

    En el centro dpsi/dtx = 1, y la malla reparte 2 tan(fov/2) entre width-1
    huecos (linspace incluye los dos extremos). Hacia los bordes el pixel
    abarca menos angulo, como en cualquier camara estenopeica; para medir
    errores en pixeles el valor central es la referencia util.
    """
    return 2.0 * np.tan(np.deg2rad(fov_deg) / 2.0) / max(width - 1, 1)


def fov_desde_half_width(r_obs: float, half_width: float, M: float = 1.0) -> float:
    """Convierte el viejo --half-width (en M) al FOV angular equivalente.

    Sirve para que los renders antiguos se puedan reproducir dando la misma
    cifra. Invierte b = r sin(psi)/sqrt(1 - 2M/r), o sea la relacion de
    Schwarzschild; con espin el encuadre queda a un pelo del anterior porque el
    arrastre corrige el borde en O(a M/r^2), no la escala. Es una comodidad de
    compatibilidad, no fisica: para trabajar, usar el FOV.
    """
    s = half_width * np.sqrt(max(1.0 - 2.0 * M / r_obs, 0.0)) / r_obs
    if s >= 1.0:
        raise ValueError(
            f"half_width = {half_width} M es imposible desde r_obs = {r_obs} M: "
            "pide un semiangulo de mas de 90 grados.")
    return float(2.0 * np.rad2deg(np.arcsin(s)))


def radio_escape(r_obs: float) -> float:
    """A que radio se puede leer ya la direccion asintotica del rayo.

    El motor lee la direccion de escape tomando la VELOCIDAD del rayo en un
    radio r_esc. Para una recta eso ya es la direccion final exacta, asi que el
    unico error es la curvatura que al rayo le queda por delante mas alla de
    r_esc, que va como 2 M b / r_esc^2.

    El valor de siempre era 1.05 r_obs, y esta atado a la distancia de la
    camara, que es justo lo que no debe ser: acercar la camara reducia r_esc y
    empeoraba el fondo. Medido contra el trazador a rtol 1e-12, con la camara a
    30 M y encuadre ancho:

        r_esc = 1.05 r_obs  ->  hasta 2.5 px de sesgo
        r_esc =   10 r_obs  ->  0.004 px
        r_esc =  100 r_obs  ->  por debajo de 0.001 px

    De ahi max(1e4, 20 r_obs): la parte constante cubre las camaras cercanas
    (donde 20 r_obs se quedaria corto en terminos absolutos) y el factor cubre
    las lejanas, cuyo encuadre es mas cerrado y por tanto mas exigente en
    pixeles. En el camino exacto subirlo no cuesta nada; el que se resiente es
    el trazador clasico de --compare, que si tiene que integrar hasta alli.
    """
    return max(1.0e4, 20.0 * float(r_obs))


def radio_angular_sombra_schwarzschild(r_obs: float, M: float = 1.0) -> float:
    """Radio angular EXACTO de la sombra con a = 0, visto desde r_obs.

    sin(psi) = b_crit sqrt(1 - 2M/r) / r,  con b_crit = 3 sqrt(3) M.

    Es la prueba de fuego del mapeo pixel -> rayo: no depende de ningun
    trazador, sale solo de la metrica. Si la camara esta bien planteada, el
    borde de captura medido sobre la malla tiene que caer aqui a precision de
    maquina, y tiene que MOVERSE al mover r_obs. Es el mismo criterio que usa
    Camera.shadow_angle() en el renderer de Schwarzschild.
    """
    b_crit = 3.0 * np.sqrt(3.0) * M
    if r_obs <= 3.0 * M:
        raise ValueError(
            f"r_obs = {r_obs} M esta dentro de la esfera de fotones: la sombra "
            "ocupa mas de un hemisferio y esta formula cambia de rama.")
    return float(np.arcsin(np.clip(b_crit * np.sqrt(1.0 - 2.0 * M / r_obs)
                                   / r_obs, -1.0, 1.0)))
