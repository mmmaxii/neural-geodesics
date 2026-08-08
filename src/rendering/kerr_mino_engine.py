"""Motor de render de Kerr en tiempo de Mino: geometria por red, fases exactas.

Reparto de trabajo
------------------
Todo lo que tiene forma cerrada se calcula EXACTO en float64 en CPU
(src/physics/kerr_mino.py), y solo la INVERSION de las cuadraturas -- que no
tiene forma cerrada elemental -- sale de las dos redes:

    exacto  |  xi, eta desde el pixel
            |  clasificacion captura/escape  (la sombra, sin trazar un rayo)
            |  lambda_camara y lambda_escape por integrales de Carlson
            |  Lam_theta, fase de camara x_cam, G_phi de medio periodo
            |  fases de los cruces con el ecuador: x = 1/4 + k/2, EXACTAS
            |  termino secular de Phi_r
    ------- + --------------------------------------------------------------
    red     |  r(lambda)  y el residuo de Phi_r(lambda)      <- KerrRNet
            |  mu(x) = cos(theta)  y el ratio de G_phi(x)    <- KerrThetaNet

Eso es lo que arregla el problema del baseline anterior: los rayos que dan
muchas vueltas cerca del anillo de fotones no le cuestan nada a la red, porque
las vueltas son aritmetica de fase (exacta) y la red solo ve el dominio
fundamental. El caos se queda fuera de la red.

Separacion de responsabilidades: aqui se devuelve GEOMETRIA (donde cruza el
rayo el disco, hacia donde escapa). El sombreado -- redshift, perfil de
temperatura, cuerpo negro -- lo hace scripts/render_kerr_mino.py reutilizando
el mismo bloque que render_kerr.py, para que las dos imagenes sean comparables.
"""
from __future__ import annotations

import numpy as np
import torch

import physics.kerr_mino as km

# Pixeles que esta parametrizacion no cubre (eta <= 0 son rayos vorticales, que
# no cruzan el ecuador; |xi| ~ 0 pasa por el eje, donde phi da un salto de pi).
# Se marcan para que el renderer los mande al trazador clasico.
XI_MIN = 1e-9


def _f64(v, n, device):
    return torch.as_tensor(np.broadcast_to(np.asarray(v, np.float64), (n,)).copy(),
                           dtype=torch.float64, device=device)


def precalcular(a: float, theta_obs: float, r_obs: float, alpha: np.ndarray,
                beta: np.ndarray, r_esc_factor: float = 1.05,
                n_cruces: int = 6, r_esc: float | None = None) -> dict:
    """Todo el precalculo EXACTO en float64. No toca ninguna red.

    Devuelve, ademas de las constantes por pixel, la tabla (n, n_cruces) de
    tiempos de Mino en los que el rayo cruza el ecuador. Esa tabla es exacta:
    los cruces caen en fases FIJAS del ciclo polar (x = 1/4 + k/2), no hay que
    detectarlos por cambio de signo como en el trazador clasico.

    r_esc fija en que radio se lee la direccion de escape. Con r_esc = None se
    conserva el viejo r_esc_factor * r_obs, que solo vale con la camara lejos:
    medido, a r_obs = 30 M sesga la direccion hasta 2.5 px porque al rayo aun
    le queda curvatura por delante. Aqui subirlo es gratis -- lambda(r_esc) es
    una integral de Carlson, no un trazado -- y ademas no saca a las redes de
    su dominio, porque en el punto de escape r se conoce exacto y a la red de
    theta se le pasa la fase ya plegada.
    """
    alpha = np.asarray(alpha, np.float64)
    beta = np.asarray(beta, np.float64)
    n = alpha.size

    xi, eta = km.xi_eta_desde_pixel(alpha, beta, theta_obs, a)
    k_h = 1.0 + np.sqrt(max(1.0 - a * a, 0.0))
    cls = km.clasificar(xi, eta, a, k_h)
    escapa = cls["escapa"]
    r_ancla, r_plateau, R = cls["r_ancla"], cls["r_plateau"], cls["raices"]

    # pixeles fuera del dominio de esta parametrizacion -> trazador clasico
    respaldo = (eta <= km.ETA_MIN) | (np.abs(xi) < XI_MIN)

    lam_cam = km.integral_mino_radial(r_ancla, np.full(n, r_obs), R)
    if r_esc is None:
        r_esc = r_esc_factor * r_obs
    r_esc = max(float(r_esc), r_esc_factor * r_obs)
    lam_esc = km.integral_mino_radial(r_ancla, np.full(n, r_esc), R)
    lam_inf = km.integral_mino_radial_inf(r_ancla, R)

    _, u_mas, m, nu, Lam = km.params_theta(xi, eta, a)
    x_cam = km.fase_camara_theta(theta_obs, beta, u_mas, m)
    G_medio = km.G_phi_semiperiodo(u_mas, m, nu)

    # sigma = tiempo de Mino recorrido DESDE la camara.
    # Si el rayo ESCAPA hace el viaje completo: entra de r_obs hasta el punto de
    # retorno (eso cuesta lam_cam) y vuelve a salir hasta r_esc (otro lam_esc),
    # asi que se SUMAN. Si CAE, solo baja de r_obs a r_stop: lam_cam a secas.
    sigma_max = np.where(escapa, lam_cam + lam_esc, lam_cam)

    # ---- cruces con el ecuador, vectorizado -------------------------------
    # mu = 0 ocurre exactamente en x = 1/4 (mod 1/2). El primer cruce despues
    # de la camara es el k0-esimo:
    k0 = np.floor(2.0 * (x_cam - 0.25)) + 1.0
    j = np.arange(n_cruces, dtype=np.float64)[None, :]
    x_k = 0.25 + 0.5 * (k0[:, None] + j)
    sigma_k = (x_k - x_cam[:, None]) * Lam[:, None]
    valido_k = (sigma_k > 0.0) & (sigma_k <= sigma_max[:, None]) \
        & ~respaldo[:, None]

    return {"n": n, "a": a, "theta_obs": theta_obs, "r_obs": r_obs,
            "r_esc": r_esc, "alpha": alpha, "beta": beta,
            "xi": xi, "eta": eta, "escapa": escapa, "respaldo": respaldo,
            "delta_gap": cls["delta_gap"], "lado": np.where(escapa, 1.0, -1.0),
            "r_ancla": r_ancla, "r_plateau": r_plateau, "raices": R,
            "lam_cam": lam_cam, "lam_inf": lam_inf, "sigma_max": sigma_max,
            "u_mas": u_mas, "m": m, "nu": nu, "Lam": Lam,
            "x_cam": x_cam, "G_medio": G_medio,
            "sigma_k": sigma_k, "valido_k": valido_k, "x_k": x_k}


def semilla_cruda(pre, idx, ell):
    """Semilla analitica de r(lambda), sin tocar ninguna red.

    Con lambda anclado en el punto de retorno (o en r_stop si el rayo cae), la
    curva r(lambda) tiene los dos extremos conocidos en forma cerrada:

        lambda = 0        ->  r = r_ancla
        lambda = lam_inf  ->  r = infinito

    y ademas se conoce COMO diverge: en campo lejano R(r) ~ r^4, asi que
    dlambda = dr/r^2 y por tanto lam_inf - lambda = 1/r al orden dominante.
    La interpolacion mas simple que respeta las tres cosas es

        r = r_ancla + 1/(lam_inf - lambda) - 1/lam_inf

    Vale exactamente en los dos extremos y acierta la asintotica de la cola,
    que es justo donde la red es PEOR (su ratio_u = r_ancla/r cae a ~5e-4 en
    campo lejano y el error relativo se dispara).

    Existe para poder responder la pregunta de la ablacion: si Newton converge
    igual desde aqui que desde la red, la red no esta aportando el resultado.
    """
    lam_inf = pre["lam_inf"][idx]
    hueco = np.maximum(lam_inf - ell, 1e-12)
    return pre["r_ancla"][idx] + 1.0 / hueco - 1.0 / np.maximum(lam_inf, 1e-12)


def _consultar_r(net_r, pre, idx, ell, device, r_conocido=None, refinar=True,
                 semilla="cruda", n_iter=6):
    """r y Phi_r acumulado desde el anclaje, en los tiempos de Mino ell.

    La red da r(lambda) -- la inversion de la cuadratura, que es lo unico sin
    forma cerrada. A partir de ahi:

      1. Si el radio ya se conoce exacto (la camara esta en r_obs, el escape en
         r_esc), se usa ese y la red ni se consulta para r.
      2. Si no, se pule con Newton imponiendo el lambda exacto (km.refinar_r).
         La red pone la semilla; el lazo lo cierra Carlson.
      3. Phi_r sale de la cuadratura exacta en la variable w, NUNCA de la
         cabeza de red: medido en el renderer, esa cabeza daba ~2.5e-3 rad de
         error angular (unos 25 pixeles) y era la fuente dominante.

    `semilla` elige de donde sale el punto de partida de Newton: "red"
    (KerrRNet) o "cruda" (semilla_cruda, sin red). Con "cruda" no hace falta
    pasar net_r. Junto con refinar=False y n_iter, permite separar los tres
    caminos -- clasico puro, red pura, hibrido -- y medir cual paga.
    """
    n = ell.size
    if r_conocido is not None:
        r = np.broadcast_to(np.asarray(r_conocido, np.float64), (n,)).copy()
    else:
        if semilla == "cruda":
            r = semilla_cruda(pre, idx, ell)
        else:
            t = lambda v: torch.as_tensor(np.ascontiguousarray(v),
                                          dtype=torch.float64, device=device)
            with torch.no_grad():
                feats = net_r.norm.features(
                    t(pre["delta_gap"][idx]), t(pre["lado"][idx]),
                    t(pre["xi"][idx]), t(pre["eta"][idx]),
                    _f64(pre["a"], n, device),
                    t(1.0 / pre["r_ancla"][idx]), t(1.0 / pre["r_plateau"][idx]),
                    t(ell), t(pre["lam_inf"][idx]), xp=torch)
                ratio_u, _ = net_r(feats)
            r = pre["r_ancla"][idx] / np.maximum(ratio_u.cpu().numpy(), 1e-12)
        r = np.maximum(r, pre["r_ancla"][idx])
        if refinar and n_iter > 0:
            # Por que la semilla por defecto es la CRUDA y no la red
            # ----------------------------------------------------------------
            # Medido en la ablacion del 2026-08-08 (benchmark --ablacion), sobre
            # 170k cruces y tres configuraciones de espin/inclinacion:
            #
            #   semilla   iter   |dr| p99    |dr| PEOR   cruces nan
            #   cruda        4    3.0e-13     4.7e-08 M       0
            #   red          6    5.0e-13     3.1e+01 M      67
            #
            # O sea que la red no solo no ayuda: HACE DANO. Newton desde la
            # semilla analitica converge en 4 iteraciones a precision de
            # maquina, mientras que desde la red sigue con 31 M de error a las
            # 6 -- porque la red llega a errar 3.3e3 M en su peor rayo y desde
            # ahi Newton cae por debajo de una raiz real, saca la raiz de un
            # negativo y devuelve nan. Los cruces nan que el renderer
            # descartaba en silencio salian TODOS de aqui: con semilla cruda
            # son cero en las tres configuraciones.
            #
            # Se dejan 6 iteraciones y no 4 por margen; cada una es una llamada
            # a Carlson y son baratas.
            r = km.refinar_r(r, ell, pre["raices"][:, idx],
                             pre["r_ancla"][idx], pre["escapa"][idx],
                             n_iter=n_iter)

    phi, _ = km.integral_phi_r(r, pre["raices"][:, idx], pre["r_ancla"][idx],
                               pre["escapa"][idx], pre["xi"][idx], pre["a"])
    return r, phi


def _consultar_theta(net_theta, pre, idx, x, device, g_exacto=True,
                     mu_exacto=False):
    """mu = cos(theta) por la red, y G_phi por su forma CERRADA.

    Fuera de la red se aplican los dos plegados, que son exactos: mu tiene
    periodo 1 y es par, y el integrando de G_phi tiene periodo 1/2. Asi un rayo
    que da veinte vueltas se resuelve con la misma red que uno que no da
    ninguna -- las vueltas son una multiplicacion, no capacidad de la red.

    G_phi NO se le pide a la red aunque haya una cabeza entrenada para ello.
    G_phi tiene forma cerrada (integral eliptica de tercera especie, validada a
    1e-9 en validate_kerr_mino.py test 6), y entra en phi multiplicada por xi,
    que llega a ~7: un error de 1e-3 en el ratio se convertia en 2e-3 rad de
    error angular, casi 9 pixeles. Lo mismo que con r_esc -- no se le pregunta a
    la red lo que se sabe exacto. La cabeza se conserva como camino rapido para
    el visor en vivo, donde importa mas el fotograma que el ultimo digito.
    """
    x = np.asarray(x, np.float64)
    x_mu = km.plegar_x(x)
    n_medios = np.floor(2.0 * x)
    x_g = x - 0.5 * n_medios

    t32 = lambda v: torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float32,
                                    device=device)
    um, mm = pre["u_mas"][idx], pre["m"][idx]
    mt = mm / (mm - 1.0)

    if mu_exacto:
        # mu(x) = sqrt(u_mas) sn(K(m)(1-4x), m): forma cerrada exacta, validada
        # a 1e-12 contra la EDO polar. La red se queda en ~2.5e-3 y no baja de
        # ahi ni multiplicando por 6 sus parametros: con m -> -infinito la sn
        # se vuelve casi una onda cuadrada y ajustarla es intrinsecamente duro.
        mu = km.mu_de_x(x_mu, um, mm)
    else:
        with torch.no_grad():
            y_mu, _ = net_theta(net_theta.norm.features(
                t32(2.0 * x_mu), t32(um), t32(mt), xp=torch))
        mu = y_mu.cpu().numpy().astype(np.float64) * np.sqrt(np.maximum(um, 0.0))

    ratio_g = None
    if not g_exacto:
        with torch.no_grad():
            _, ratio_g = net_theta(net_theta.norm.features(
                t32(2.0 * x_g), t32(um), t32(mt), xp=torch))

    if g_exacto:
        G = km.G_phi_exacto(x, um, mm, pre["nu"][idx])
    else:
        G = (n_medios + ratio_g.cpu().numpy().astype(np.float64)) \
            * pre["G_medio"][idx]
    return mu, G


def evaluar(pre: dict, net_r, net_theta, device: str = "cpu",
            mu_exacto: bool = False, semilla: str = "cruda",
            refinar: bool = True, n_iter: int = 6) -> dict:
    """Geometria por pixel: cruces con el disco y direccion de escape.

    Devuelve
        cross_r, cross_phi : (n, n_cruces)  radio y angulo de cada cruce
        n_cross            : (n,)           cuantos cruces validos hay
        direccion          : (n, 3)         direccion asintotica, unitaria
        captured           : (n,)           dentro de la sombra
        respaldo           : (n,)           pixeles para el trazador clasico

    semilla="cruda" es el DEFECTO desde la ablacion del 2026-08-08: con ella el
    motor no consulta KerrRNet en ningun momento y net_r puede ser None. La
    razon esta medida y documentada en _consultar_r -- la semilla de red daba
    31 M de error en el peor cruce y era la fuente de todos los cruces nan.
    semilla="red" se conserva para poder reproducir la ablacion y para
    validate_kerr_mino_net.py, que la fija explicitamente.

    Nota de alcance: la red de r SOLO intervenia en los cruces con el disco. En
    la camara y en el punto de escape se pasa r_conocido (r_obs y r_esc son
    exactos por construccion), asi que un render --no-disk nunca la uso.
    Despues de este cambio, lo UNICO que hace una red en este motor es mu(x)
    cuando mu_exacto=False -- y eso cuesta 1.1 px frente a 0.001 px, sin ganar
    velocidad (0.98x, benchmark del 2026-08-07).
    """
    n, nk = pre["n"], pre["sigma_k"].shape[1]
    escapa, respaldo = pre["escapa"], pre["respaldo"]
    lam_cam, sigma_max = pre["lam_cam"], pre["sigma_max"]

    # ---- phi de la camara: hay que restarlo para que phi arranque en 0 ------
    todos = np.arange(n)
    # en la camara r vale r_obs EXACTAMENTE: no hay que preguntarselo a la red
    _, Phi_cam = _consultar_r(net_r, pre, todos, lam_cam, device,
                              r_conocido=pre["r_obs"])
    _, G_cam = _consultar_theta(net_theta, pre, todos, pre["x_cam"], device)

    cross_r = np.zeros((n, nk))
    cross_phi = np.zeros((n, nk))
    val = pre["valido_k"]

    if val.any():
        fi, fj = np.nonzero(val)
        sig = pre["sigma_k"][fi, fj]
        esc_f = escapa[fi]
        # lambda medido desde el ANCLAJE (donde vive la red), con su signo:
        # si el rayo escapa, lambda_r = sigma - lam_cam y r(lambda) es par;
        # si cae, lambda_r = lam_cam - sigma y es monotono.
        lr = np.where(esc_f, sig - lam_cam[fi], lam_cam[fi] - sig)
        ell = np.where(esc_f, np.abs(lr), np.maximum(lr, 0.0))
        signo = np.where(esc_f, np.where(lr < 0, -1.0, 1.0), 1.0)

        # UNICO punto del motor donde r sale de la red (o de la semilla cruda):
        # aqui el radio del cruce no se conoce de antemano.
        r_q, Phi_q = _consultar_r(net_r, pre, fi, ell, device,
                                  refinar=refinar, semilla=semilla,
                                  n_iter=n_iter)
        # Phi_r acumulado DESDE LA CAMARA. Si el rayo escapa, la camara esta en
        # lambda_r = -lam_cam y Phi anclado es IMPAR en lambda_r (porque
        # f_r(r(lambda)) es par), asi que Phi^anc(-lam_cam) = -Phi_cam y
        #     phi_r = signo*Phi(|lambda_r|) - (-Phi_cam) = signo*Phi_q + Phi_cam
        # Si cae, lambda_r = lam_cam - sigma decrece y phi_r = Phi_cam - Phi_q.
        phi_r = np.where(esc_f, signo * Phi_q + Phi_cam[fi], Phi_cam[fi] - Phi_q)

        x_q = pre["x_cam"][fi] + sig / pre["Lam"][fi]
        _, G_q = _consultar_theta(net_theta, pre, fi, x_q, device)
        phi_th = pre["xi"][fi] * (G_q - G_cam[fi])

        cross_r[fi, fj] = r_q
        cross_phi[fi, fj] = phi_r + phi_th

    n_cross = val.sum(1).astype(np.int32)

    # ---- direccion de escape ----------------------------------------------
    direccion = np.zeros((n, 3))
    sel = escapa & ~respaldo
    if sel.any():
        idx = np.flatnonzero(sel)
        sig = sigma_max[idx]
        ell = np.abs(sig - lam_cam[idx])
        # r en el punto de escape NO se le pregunta a la red: sigma_max se
        # definio como el tiempo de Mino que tarda el rayo en llegar a r_esc,
        # asi que ahi r vale r_esc EXACTAMENTE. Ademas es justo donde peor
        # resuelve la red: ratio_u = r_ancla/r cae a ~5e-4 en campo lejano y
        # el error relativo se dispara aunque el absoluto sea pequeno.
        r_f = np.full(idx.size, pre["r_esc"])
        _, Phi_q = _consultar_r(net_r, pre, idx, ell, device, r_conocido=r_f)
        phi_r = Phi_q + Phi_cam[idx]          # el rayo sale por lambda_r > 0

        x_f = pre["x_cam"][idx] + sig / pre["Lam"][idx]
        mu_f, G_q = _consultar_theta(net_theta, pre, idx, x_f, device,
                                     mu_exacto=mu_exacto)
        phi = phi_r + pre["xi"][idx] * (G_q - G_cam[idx])
        th = np.arccos(np.clip(mu_f, -1.0, 1.0))

        # las derivadas salen EXACTAS de los potenciales, no de la red: solo
        # hace falta el signo de la rama, que lo da la fase.
        xi_s, eta_s, a = pre["xi"][idx], pre["eta"][idx], pre["a"]
        signo_mu = np.where(np.mod(x_f, 1.0) < 0.5, -1.0, 1.0)
        dmu = signo_mu * np.sqrt(np.maximum(km.potencial_W(mu_f, xi_s, eta_s, a), 0.0))
        dth = -dmu / np.maximum(np.sin(th), 1e-14)
        drs = np.sqrt(np.maximum(km.potencial_R(r_f, xi_s, eta_s, a), 0.0))
        dph = km.f_r_exacta(r_f, xi_s, a) + xi_s / np.maximum(np.sin(th) ** 2, 1e-30)

        st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(phi), np.cos(phi)
        v = np.stack([drs * st * cp + r_f * dth * ct * cp - r_f * dph * st * sp,
                      drs * st * sp + r_f * dth * ct * sp + r_f * dph * st * cp,
                      drs * ct - r_f * dth * st], axis=1)
        direccion[idx] = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-30)

    return {"cross_r": cross_r, "cross_phi": cross_phi, "n_cross": n_cross,
            "direccion": direccion, "captured": ~escapa, "respaldo": respaldo,
            "xi": pre["xi"]}
