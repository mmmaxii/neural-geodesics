"""Valida la camara angular de Kerr contra la FISICA, no contra el trazador.

Por que esta suite es distinta de validate_kerr_mino.py
-------------------------------------------------------
Aquella compara el motor de Mino contra trace_batch, y las dos partes usan las
MISMAS coordenadas (alpha, beta). Un convenio de camara equivocado pero
coherente pasaria esa prueba con nota: los dos caminos harian lo mismo mal.

Por eso aqui la referencia no es ningun trazador sino cantidades cerradas que
salen solo de la metrica:

    0. la tetrada del ZAMO es ortonormal:  g(e_a, e_b) = eta_ab
    1. con a = 0 se reduce a Camera.impact_parameter de Schwarzschild
    2. el paso por (alpha, beta) no pierde informacion: se recuperan xi, eta
       Y el p_r que reconstruye el trazador
    3. PRUEBA DE FUEGO: el radio angular de la sombra con a = 0 coincide con
       sin(psi) = 3 sqrt(3) sqrt(1 - 2M/r) / r  a varias distancias
    4. el sintoma que motivo todo: tamano de la sombra en pixeles vs r_obs,
       con la camara vieja y con la nueva
    5. con espin, el borde de captura converge al contorno de Bardeen
    6. cuanto se sale del dominio entrenado de KerrRNet el FOV que se use

    python scripts/validate_kerr_camera.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import physics.kerr_camera as kc                                # noqa: E402
import physics.kerr_mino as km                                  # noqa: E402
from physics.kerr import KMetric                                # noqa: E402
from physics.kerr_integrator import KerrGeodesicIntegrator      # noqa: E402

FALLOS = []


def informe(nombre, valor, umbral, unidad=""):
    ok = valor <= umbral
    if not ok:
        FALLOS.append(nombre)
    print(f"  [{'ok ' if ok else 'FALLO'}] {nombre:<52s} "
          f"{valor:10.3e} {unidad:<6s} (umbral {umbral:.0e})")
    return ok


# ============================================================ 0. ortonormalidad
def test_tetrada():
    print("\n0. la tetrada del ZAMO es ortonormal")
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    peor = 0.0
    for a in (0.0, 0.5, 0.9, 0.998):
        k = KMetric(1.0, a)
        for r in (3.0, 10.0, 50.0, 1000.0):
            for th in np.deg2rad([5.0, 45.0, 89.0, 130.0]):
                tt = kc.tetrada_zamo(k, r, th)
                # columnas = vectores de la base, en el orden (t, r, theta, phi)
                E = np.zeros((4, 4))
                E[0, 0] = 1.0 / tt.lapso
                E[3, 0] = tt.omega / tt.lapso
                E[1, 1] = 1.0 / tt.raiz_sigma_delta
                E[2, 2] = 1.0 / tt.raiz_sigma
                E[3, 3] = 1.0 / tt.raiz_gpp
                g = k.metric_matrix(r, th)
                peor = max(peor, np.abs(E.T @ g @ E - eta).max())
    informe("|E^T g E - eta| maximo", peor, 1e-11)


# ================================================ 1. limite de Schwarzschild
def test_limite_schwarzschild():
    print("\n1. con a = 0 se reduce a Camera.impact_parameter (ya validado)")
    k = KMetric(1.0, 0.0)
    th_obs = np.deg2rad(80.0)
    psi = np.deg2rad(np.linspace(0.0, 40.0, 400))
    chi = 0.7                                   # azimut cualquiera en la imagen
    peor_rel = 0.0
    for r_obs in (5.0, 10.0, 50.0, 1000.0):
        n_r = -np.cos(psi)
        n_th = np.sin(psi) * np.cos(chi)
        n_ph = np.sin(psi) * np.sin(chi)
        al, be = kc.celestes_desde_direccion(k, r_obs, th_obs, n_r, n_th, n_ph)
        b_mio = np.hypot(al, be)
        # la formula del renderer clasico, escrita en M (alli va en r_s)
        b_ref = r_obs * np.sin(psi) / np.sqrt(1.0 - 2.0 / r_obs)
        rel = np.abs(b_mio - b_ref) / np.maximum(b_ref, 1e-12)
        peor_rel = max(peor_rel, np.nanmax(rel[1:]))
    informe("error relativo peor en b(psi)", peor_rel, 1e-13)


# ============================================ 2. el paso por (alpha, beta)
def test_ida_y_vuelta():
    """(alpha, beta) tiene que llevar TODA la informacion del rayo.

    Es la unica pieza que permite no tocar nada aguas abajo, asi que conviene
    comprobarla y no razonarla: se rehace la cuenta por la tetrada y se compara
    con lo que reconstruye cada consumidor.

    Ojo con p_r: (alpha, beta) NO lo lleva. initial_from_celestial lo vuelve a
    sacar imponiendo que el rayo sea nulo, y aqui se comprueba que lo que sale
    de ahi es exactamente el p_r que predice la tetrada -- signo incluido. Si
    no coincidiera, la camara estaria mandando rayos hacia otro sitio.
    """
    print("\n2. (alpha, beta) no pierde informacion: xi, eta y p_r")
    peor_xi = peor_eta = peor_pr = 0.0
    for a in (0.0, 0.5, 0.9, 0.998):
        k = KMetric(1.0, a)
        ig = KerrGeodesicIntegrator(k)
        for r_obs in (10.0, 100.0, 1000.0):
            for inc in (20.0, 60.0, 85.0):
                th_obs = np.deg2rad(inc)
                n_r, n_th, n_ph = kc.direcciones_pixel(31, 17, 25.0)
                al, be = kc.celestes_desde_direccion(k, r_obs, th_obs,
                                                     n_r, n_th, n_ph)
                # --- lo que reconstruye la capa de Mino
                xi_c, eta_c = km.xi_eta_desde_pixel(al, be, th_obs, a)

                # --- la misma cuenta, directa desde la tetrada
                tt = kc.tetrada_zamo(k, r_obs, th_obs)
                Dn = tt.lapso + tt.omega * tt.raiz_gpp * n_ph
                xi_t = tt.raiz_gpp * n_ph / Dn
                pth_t = tt.raiz_sigma * n_th / Dn
                ct, st = np.cos(th_obs), np.sin(th_obs)
                eta_t = pth_t**2 + ct**2 * (xi_t**2 / st**2 - a**2)

                esc = np.maximum(np.abs(xi_t), 1.0)
                peor_xi = max(peor_xi, np.max(np.abs(xi_c - xi_t) / esc))
                peor_eta = max(peor_eta, np.max(np.abs(eta_c - eta_t)
                                                / np.maximum(np.abs(eta_t), 1.0)))

                # --- p_r: la tetrada dice cuanto vale, el trazador lo rededuce
                pr_t = tt.raiz_sigma_delta * n_r / Dn
                for j in range(0, al.size, 37):
                    y0, E, _ = ig.initial_from_celestial(al[j], be[j],
                                                         r_obs, th_obs)
                    peor_pr = max(peor_pr, abs(y0[3] / E - pr_t[j])
                                  / max(abs(pr_t[j]), 1e-3))
    informe("error relativo en xi", peor_xi, 1e-13)
    informe("error relativo en eta", peor_eta, 1e-12)
    informe("error relativo en p_r/E (signo incluido)", peor_pr, 1e-11)


# ================================================== 3. la prueba de fuego
def _psi_borde_sombra(k, r_obs, th_obs, chi=0.0, lo=1e-6, hi=None, n_iter=90):
    """Angulo de mirada al que empieza la sombra, por biseccion.

    No traza ningun rayo: km.clasificar decide captura desde las raices del
    cuartico R(r), que es algebra exacta. Asi el resultado mide SOLO el mapeo
    pixel -> rayo, sin el error de ningun integrador de por medio.
    """
    a = k.a
    k_h = 1.0 + np.sqrt(max(1.0 - a * a, 0.0))

    def captura(psi):
        n_r = -np.cos(psi)
        n_th = np.sin(psi) * np.cos(chi)
        n_ph = np.sin(psi) * np.sin(chi)
        al, be = kc.celestes_desde_direccion(k, r_obs, th_obs, n_r, n_th, n_ph)
        xi, eta = km.xi_eta_desde_pixel(np.atleast_1d(al), np.atleast_1d(be),
                                        th_obs, a)
        return not bool(km.clasificar(xi, eta, a, k_h)["escapa"][0])

    if hi is None:
        hi = 0.5 * np.pi - 1e-6
    if not captura(lo):
        return np.nan                      # ni mirando al centro hay sombra
    if captura(hi):
        return np.nan                      # la sombra se sale del hemisferio
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if captura(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_sombra_angular():
    print("\n3. PRUEBA DE FUEGO: radio angular de la sombra con a = 0")
    print("     (la referencia es la metrica, no ningun trazador)")
    k = KMetric(1.0, 0.0)
    th_obs = np.deg2rad(80.0)
    peor = 0.0
    print(f"     {'r_obs (M)':>10s} {'medido (deg)':>14s} {'exacto (deg)':>14s} "
          f"{'error (rad)':>12s}")
    for r_obs in (5.0, 10.0, 20.0, 50.0, 100.0, 1000.0):
        med = _psi_borde_sombra(k, r_obs, th_obs)
        ref = kc.radio_angular_sombra_schwarzschild(r_obs)
        peor = max(peor, abs(med - ref))
        print(f"     {r_obs:10.1f} {np.rad2deg(med):14.6f} "
              f"{np.rad2deg(ref):14.6f} {abs(med-ref):12.2e}")
    informe("error peor en el radio angular de la sombra", peor, 1e-9, "rad")


# ================================================== 4. el sintoma original
def test_sintoma():
    """La queja que abrio todo esto, medida en numeros.

    Con la camara vieja (--half-width fijo, que es un parametro de impacto) la
    sombra ocupa SIEMPRE los mismos pixeles pase lo que pase con la distancia,
    y lo unico que se mueve es la escala del fondo: de ahi la sensacion de que
    el agujero esta clavado y lo que se aleja es el cielo.
    """
    print("\n4. el sintoma: tamano de la sombra en pixeles vs distancia")
    k = KMetric(1.0, 0.0)
    th_obs = np.deg2rad(80.0)
    W = 480
    hw, fov = 11.0, 3.0
    print(f"     camara vieja: --half-width {hw} M fijo   |   "
          f"camara nueva: FOV {fov} deg fijo")
    print(f"     {'r_obs (M)':>10s} {'vieja (px)':>12s} {'nueva (px)':>12s}"
          f" {'nueva/1000':>12s}")
    viejos, nuevos = [], []
    for r_obs in (250.0, 500.0, 1000.0, 2000.0, 4000.0):
        # vieja: alpha = hw * u, u en [-1, 1] repartido en W-1 huecos
        px_vieja = 3.0 * np.sqrt(3.0) / (2.0 * hw / (W - 1))
        # nueva: el borde cae donde la mirada forma psi_sombra con el eje
        psi = kc.radio_angular_sombra_schwarzschild(r_obs)
        px_nueva = np.tan(psi) / kc.rad_por_pixel(fov, W)
        viejos.append(px_vieja); nuevos.append(px_nueva)
        print(f"     {r_obs:10.0f} {px_vieja:12.2f} {px_nueva:12.2f}"
              f" {px_nueva*r_obs/1000.0:12.2f}")
    var_vieja = (max(viejos) - min(viejos)) / np.mean(viejos)
    # la nueva tiene que ir como 1/r_obs: el producto px*r_obs es casi constante
    prod = np.array(nuevos) * np.array([250., 500., 1000., 2000., 4000.])
    var_nueva = (prod.max() - prod.min()) / prod.mean()
    print(f"     variacion del tamano con la camara vieja: {100*var_vieja:.4f}% "
          f"(deberia ser 0: ese es el bug)")
    informe("desviacion de la ley 1/r_obs de la camara nueva", var_nueva, 2e-2)


# ============================================== 5. convergencia a Bardeen
def _dist_a_poligonal(px, py, cx, cy):
    """Distancia de un punto a la poligonal CERRADA (cx, cy), por segmentos.

    Hay que medir contra los segmentos y no contra los vertices. shadow_outline
    muestrea en el radio de las orbitas esfericas y cierra reflejando en beta,
    asi que el reparto de puntos es muy desigual: la mediana del hueco es
    ~0.007 M pero el mayor llega a 0.17 M. Midiendo a los vertices el error
    medido se quedaba clavado en exactamente la mitad de ese hueco (0.087 M) a
    r_obs = 100, 1000 y 10000 -- identico, o sea que no era un error que
    convergiera: era la resolucion de la REFERENCIA, no la camara.
    """
    ax, ay = cx, cy
    bx, by = np.roll(cx, -1), np.roll(cy, -1)
    ex, ey = bx - ax, by - ay
    L2 = ex * ex + ey * ey
    t = np.clip(((px - ax) * ex + (py - ay) * ey) / np.maximum(L2, 1e-300),
                0.0, 1.0)
    return float(np.min(np.hypot(px - (ax + t * ex), py - (ay + t * ey))))


def test_contorno_bardeen():
    """El borde de captura en (alpha, beta) es el contorno de Bardeen. Exacto.

    Y lo es a CUALQUIER r_obs, no solo en campo lejano: la captura la decide
    (xi, eta) y nada mas, asi que la FORMA de la sombra en coordenadas de
    Bardeen no depende de la distancia. Eso no es un accidente, es exactamente
    la razon de que la camara vieja pareciera rota: dibujando el plano imagen
    directamente en (alpha, beta) se estaba dibujando una figura que por
    construccion no cambia con r_obs. Lo que si cambia con la distancia es su
    tamano ANGULAR, y eso es lo que mide la prueba 3.

    El residuo que queda aqui (~7e-4 M, igual a las tres distancias) es la
    sagita de los segmentos de shadow_outline, o sea otra vez la referencia.
    """
    print("\n5. con espin, el borde de captura ES el contorno de Bardeen")
    inc = 85.0
    th_obs = np.deg2rad(inc)
    peor_lejos = 0.0
    for a in (0.5, 0.9, 0.998):
        k = KMetric(1.0, a)
        ao, bo = k.shadow_outline(inclination_deg=inc, n=2400)
        print(f"     a = {a}")
        for r_obs in (100.0, 1000.0, 10000.0):
            d = []
            for chi in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False):
                psi = _psi_borde_sombra(k, r_obs, th_obs, chi=chi)
                if not np.isfinite(psi):
                    continue
                n_r = -np.cos(psi)
                n_th = np.sin(psi) * np.cos(chi)
                n_ph = np.sin(psi) * np.sin(chi)
                al, be = kc.celestes_desde_direccion(k, r_obs, th_obs,
                                                     n_r, n_th, n_ph)
                d.append(_dist_a_poligonal(float(al), float(be), ao, bo))
            d = float(np.max(d))
            print(f"       r_obs = {r_obs:7.0f} M   distancia al contorno "
                  f"{d:.3e} M")
            if r_obs >= 10000.0:
                peor_lejos = max(peor_lejos, d)
    informe("distancia al contorno de Bardeen a r_obs = 1e4 M",
            peor_lejos, 5e-3, "M")


# ================================================ 6. dominio de la red
def test_dominio_red():
    """Que tanto por ciento de la imagen se sale del rango entrenado.

    NormalizadorR RECORTA xi a +-20 y eta a 900 sin avisar (kerr_mino_net.py:
    143-145), asi que fuera de ahi la red ve una entrada falsa. No es fatal
    -- en evaluar() la salida de KerrRNet solo es la SEMILLA de refinar_r, que
    la pule con Newton sobre Carlson hasta precision de maquina -- pero una
    semilla mala puede dejar a Newton sin converger en sus 6 iteraciones. Se
    mide para poder avisar, no para enrutar.
    """
    print("\n6. dominio de KerrRNet segun el encuadre (xi_max=20, eta_max=900)")
    k = KMetric(1.0, 0.9)
    th_obs = np.deg2rad(80.0)
    print(f"     {'r_obs (M)':>10s} {'FOV (deg)':>10s} {'|xi| max':>10s} "
          f"{'eta max':>10s} {'% recortado':>12s}")
    for r_obs, fov in ((1000.0, 1.5), (1000.0, 3.0), (1000.0, 10.0),
                       (100.0, 15.0), (30.0, 40.0), (30.0, 90.0)):
        al, be = kc.malla_celeste(k, r_obs, th_obs, fov, 120, 68)
        xi, eta = km.xi_eta_desde_pixel(al, be, th_obs, 0.9)
        fuera = (np.abs(xi) > 20.0) | (eta > 900.0)
        print(f"     {r_obs:10.0f} {fov:10.1f} {np.abs(xi).max():10.2f} "
              f"{eta.max():10.1f} {100*fuera.mean():11.2f}%")
    print("     (informativo: no hay umbral, solo sirve para elegir el encuadre)")


# ====================================== 7. la lente, contra la formula debil
def test_desvio_debil():
    """Cuanto se mueve una estrella de sitio, contra 4M/b.

    Es la prueba que hay que hacer cuando alguien dice "el fondo se ve
    demasiado deformado": poner una fuente en una direccion CONOCIDA y medir
    cuanto se desplaza. Para la camara casi en el infinito (r_obs >> b) el
    desvio total tiene desarrollo conocido

        delta = 4M/b + (15 pi/4) (M/b)^2 + ...

    o sea  delta / (4M/b) = 1 + (15 pi/16)(M/b) = 1 + 2.945/b.

    Se compara contra eso, no contra 4M/b a secas: a b = 44 M el segundo orden
    ya vale un 7%, y confundirlo con un error seria un falso positivo.

    Ojo con el otro extremo: la formula supone la camara en el infinito, asi
    que solo vale mientras b << r_obs. Por eso se piden b <= r_obs/40.
    """
    print("\n7. la lente: desvio medido vs 4M/b + segundo orden")
    k = KMetric(1.0, 0.0)
    ig = KerrGeodesicIntegrator(k)
    th_obs = np.deg2rad(80.0)
    r_obs = 1.0e5
    st, ct = np.sin(th_obs), np.cos(th_obs)
    rh = np.array([st, 0.0, ct])
    thh = np.array([ct, 0.0, -st])
    phh = np.array([0.0, 1.0, 0.0])

    peor = 0.0
    print(f"     {'b (M)':>10} {'medido':>13} {'4M/b(1+2.945/b)':>18} {'razon':>9}"
          f" {'|1-razon|':>11} {'tolerado':>10}")
    for b_obj in (50.0, 100.0, 300.0, 1000.0, 2500.0):
        psi = np.arcsin(np.clip(b_obj * np.sqrt(1.0 - 2.0 / r_obs) / r_obs,
                                -1.0, 1.0))
        n_r, n_th, n_ph = -np.cos(psi), np.sin(psi), 0.0
        al, be = kc.celestes_desde_direccion(k, r_obs, th_obs, n_r, n_th, n_ph)
        out = ig.trace(float(al), float(be), r_obs, th_obs,
                       r_escape=1.0e9, rtol=1e-12, atol=1e-14, lam_max=1e11)
        if out["status"] != "escaped":
            continue
        n_cart = n_r * rh + n_th * thh + n_ph * phh
        med = float(np.arccos(np.clip(np.dot(n_cart, out["direction"]), -1, 1)))
        b = float(np.hypot(al, be))
        pred = (4.0 / b) * (1.0 + 15.0 * np.pi / 16.0 / b)
        razon = med / pred
        # El umbral tiene que seguir a la serie, no ser una constante: lo que
        # queda sin modelar es el TERCER orden, que va como (M/b)^2, mas un
        # suelo por el ruido del trazador (a b grande el desvio es minusculo y
        # medirlo relativo se vuelve duro). Con un umbral fijo, b = 50 daba
        # falso positivo por 4.4e-3 -- que es exactamente (2.945/50)^2.
        tol = max(15.0 / (b * b), 3.0e-4)
        peor = max(peor, abs(razon - 1.0) / tol)
        print(f"     {b:10.2f} {med:13.6e} {pred:18.6e} {razon:9.5f}"
              f" {abs(razon-1.0):11.2e} {tol:10.2e}")
    informe("desvio debil / lo que admite el siguiente orden", peor, 1.0)


def test_encuadre_einstein():
    """Aviso de encuadre: que fraccion del cuadro esta en lensing FUERTE.

    theta_E = sqrt(4M/r_obs) es el radio angular de Einstein con la fuente en
    el infinito: dentro de el, el cielo esta fuertemente comprimido y
    enrollado. Si el campo de vision entero cae dentro de theta_E, el render
    sale con aspecto de TUNEL en toda la imagen -- no porque la lente este mal,
    sino porque no queda nada de cielo sin deformar con que comparar.

    Esto no es un fallo, es una eleccion de encuadre, pero conviene tenerla a
    la vista porque explica un aspecto que parece un error y no lo es.
    """
    print("\n8. encuadre: campo de vision frente al radio de Einstein")
    print("     (informativo; theta_E = sqrt(4M/r_obs), fuente en el infinito)")
    print(f"     {'r_obs':>8} {'FOV':>10} {'theta_E':>10} {'sombra':>10} "
          f"{'FOV/theta_E':>12}  veredicto")
    for r_o, fov in ((100.0, 60.0), (1000.0, 1.259), (4000.0, 0.315),
                     (20000.0, 0.25)):
        thE = np.rad2deg(np.sqrt(4.0 / r_o))
        sh = 2.0 * np.rad2deg(kc.radio_angular_sombra_schwarzschild(r_o))
        q = fov / thE
        v = ("cielo sin deformar en los bordes" if q > 2.0 else
             "TODO el cuadro en lensing fuerte" if q < 1.0 else "mixto")
        print(f"     {r_o:8.0f} {fov:9.3f}d {thE:9.3f}d {sh:9.4f}d {q:12.3f}"
              f"  {v}")


def main():
    print("=" * 78)
    print("VALIDACION DE LA CAMARA ANGULAR DE KERR")
    print("=" * 78)
    test_tetrada()
    test_limite_schwarzschild()
    test_ida_y_vuelta()
    test_sombra_angular()
    test_sintoma()
    test_contorno_bardeen()
    test_dominio_red()
    test_desvio_debil()
    test_encuadre_einstein()
    print("\n" + "=" * 78)
    if FALLOS:
        print(f"FALLAN {len(FALLOS)} pruebas: " + ", ".join(FALLOS))
        return 1
    print("todas las pruebas pasan")
    return 0


if __name__ == "__main__":
    sys.exit(main())
