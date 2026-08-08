"""Benchmark de velocidad del motor de Mino frente al trazador clasico.

Mide por SEPARADO las piezas del coste, porque no escalan igual y saber cual
manda decide donde optimizar:

    precalculo : raices, clasificacion, integrales elipticas de Carlson (CPU f64)
    geometria  : la inversion de las cuadraturas, por RED o por FORMA CERRADA
    total      : lo que tarda un fotograma

Los dos caminos de geometria se cronometran POR SEPARADO, que es la pregunta
que de verdad importa en este proyecto:

    exacto : cos(theta) por Jacobi sn, r pulido por Newton sobre Carlson.
             Reproduce el trazador a 0.001 px.
    red    : cos(theta) sale de KerrThetaNet. Un forward en GPU en vez de
             decenas de evaluaciones de scipy.

Aviso: la version anterior de este script cronometraba `mu_exacto=True` y
etiquetaba esa columna como "redes". Media el camino EXACTO, no la red.

El clasico se mide a dos tolerancias. No es un detalle: trace_batch a rtol 1e-6
es un render "de vista", y a 1e-10 es el que se usa como referencia de
precision. El speedup honesto depende de contra cual se compare, asi que se
reportan los dos.

Sobre el radio de escape: se usa el MISMO para los dos lados (1.05 r_obs por
defecto, el valor clasico). El renderer usa hoy uno mucho mayor
(kerr_camera.radio_escape) por precision, y eso encarece SOLO al trazador
clasico -- para el camino de Mino r_esc es una integral de Carlson y da igual
lo lejos que este. Medirlo asi inflaria el speedup, por eso aqui se compara en
las condiciones que menos favorecen al surrogate. `--r-esc-render` lo mide
aparte para poder citar las dos cifras.

    python scripts/benchmark_kerr_mino.py
    python scripts/benchmark_kerr_mino.py --resoluciones 320x180 960x540
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import physics.kerr_camera as kc                                # noqa: E402
import rendering.kerr_mino_engine as eng                       # noqa: E402
from physics.kerr import KMetric                               # noqa: E402
from physics.kerr_integrator import KerrGeodesicIntegrator     # noqa: E402
from models.kerr_mino_net import KerrRNet, KerrThetaNet        # noqa: E402


def cronometrar(fn, n_rep=1):
    """Ejecuta fn n_rep veces y devuelve el MEJOR tiempo, no la media.

    El mejor es la estimacion mas limpia del coste real: el ruido de un sistema
    operativo solo puede hacer que una medida salga mas lenta, nunca mas
    rapida.
    """
    mejor = float("inf")
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mejor = min(mejor, time.perf_counter() - t0)
    return mejor


def ablacion(a, k, net_r, net_th, device):
    """?Aporta algo KerrRNet, o solo pone una semilla que Newton ya no necesita?

    Es el ultimo rincon donde la red interviene en el motor. La red de r SOLO
    se consulta en los cruces con el disco: en la camara y en el escape se pasa
    r_conocido, asi que un render --no-disk no la usa en absoluto.

    Tres caminos, con la misma escena y el mismo precalculo:

        A  semilla analitica cruda  + Newton      (sin red en ningun sitio)
        B  semilla de KerrRNet      + nada        (la red resolviendo de verdad)
        C  semilla de KerrRNet      + Newton      (lo que hace hoy el renderer)

    Se mide a IGUAL PRECISION, no error y tiempo por separado: para A y C se
    barre el numero de iteraciones y se busca cual es el camino mas barato que
    alcanza cada umbral. Comparar "la red es rapida pero imprecisa" contra
    "Carlson es lento pero exacto" sin ese eje comun no dice nada.

    theta va en modo exacto en todos los casos, para que lo unico que cambie
    entre filas sea la inversion radial.
    """
    th = np.deg2rad(a.inc)
    W, H = (int(v) for v in a.res_ablacion.lower().split("x"))
    al, be = kc.malla_celeste(k, a.r_obs, th, a.fov_deg, W, H)
    r_esc = kc.radio_escape(a.r_obs) if a.r_esc_render else 1.05 * a.r_obs
    pre = eng.precalcular(a.spin, th, a.r_obs, al, be, n_cruces=6, r_esc=r_esc)
    val = pre["valido_k"]
    n_cruces = int(val.sum())

    print(f"\n{'=' * 96}")
    print("ABLACION DE KerrRNet: la semilla de Newton, ?hace falta?")
    print(f"{'=' * 96}")
    print(f"escena {W}x{H} = {al.size:,} rayos, {n_cruces:,} cruces con el disco "
          f"(a={a.spin}, inc={a.inc}, r_obs={a.r_obs:.0f} M)")

    def correr(semilla, n_iter, red):
        return eng.evaluar(pre, red, net_th, device, mu_exacto=True,
                           semilla=semilla, refinar=(n_iter > 0), n_iter=n_iter)

    # --- referencia: semilla cruda muy refinada. Se comprueba que convergio
    # comparandola con una todavia mas refinada; si no coincidieran, la
    # referencia seria el limite y no los metodos (ya nos ha pasado tres veces).
    ref = correr("cruda", 40, None)["cross_r"][val]
    ref2 = correr("cruda", 60, None)["cross_r"][val]
    deriva = float(np.nanmax(np.abs(ref - ref2)))
    print(f"referencia: semilla cruda + 40 iteraciones; deriva contra 60 "
          f"iteraciones = {deriva:.2e} M")
    if deriva > 1e-11:
        print("  AVISO: la referencia no esta convergida; los numeros de abajo "
              "miden la referencia, no los metodos")

    def medir(etiqueta, semilla, n_iter, red):
        geo = correr(semilla, n_iter, red)
        d = np.abs(geo["cross_r"][val] - ref)
        d = d[np.isfinite(d)]
        t = cronometrar(lambda: correr(semilla, n_iter, red), a.repeticiones)
        return {"metodo": etiqueta, "semilla": semilla, "n_iter": n_iter,
                "p50": float(np.median(d)), "p99": float(np.percentile(d, 99)),
                "peor": float(d.max()), "t": t}

    filas = []
    print(f"\n{'metodo':<34} {'iter':>5} {'|dr| p50':>11} {'|dr| p99':>11} "
          f"{'|dr| peor':>11} {'tiempo':>9}")
    print("-" * 96)
    for n_iter in (0, 1, 2, 3, 4, 6, 8, 12):
        filas.append(medir("A  cruda + Newton", "cruda", n_iter, None))
        f = filas[-1]
        print(f"{f['metodo']:<34} {n_iter:>5} {f['p50']:11.2e} {f['p99']:11.2e} "
              f"{f['peor']:11.2e} {f['t']:8.3f}s")
    print()
    for n_iter in (0, 1, 2, 3, 4, 6, 8, 12):
        et = "B  KerrRNet sola" if n_iter == 0 else "C  KerrRNet + Newton"
        filas.append(medir(et, "red", n_iter, net_r))
        f = filas[-1]
        print(f"{f['metodo']:<34} {n_iter:>5} {f['p50']:11.2e} {f['p99']:11.2e} "
              f"{f['peor']:11.2e} {f['t']:8.3f}s")

    # --- frente de Pareto: lo mas barato que alcanza cada umbral.
    # El criterio es el PEOR caso, no la p99. Con p99 las dos ramas parecen
    # equivalentes (+-5%), porque el dano de la semilla de red esta concentrado
    # en un punado de rayos: la p99 lo esconde justo donde esta la historia.
    print(f"\n{'umbral PEOR |dr|':>16}  {'mas barato SIN red':<30} "
          f"{'mas barato CON red':<30}  veredicto")
    print("-" * 96)
    for umbral in (1e-3, 1e-6, 1e-9, 1e-12):
        def mejor(pred):
            c = [f for f in filas if pred(f) and f["peor"] <= umbral]
            return min(c, key=lambda f: f["t"]) if c else None
        sa = mejor(lambda f: f["semilla"] == "cruda")
        sr = mejor(lambda f: f["semilla"] == "red")
        d = lambda f: f"{f['n_iter']:2d} iter, {f['t']:.3f}s" if f else "inalcanzable"
        if sa and sr:
            v = (f"la red cuesta {100*(sr['t']/sa['t'] - 1):+.1f}%")
        elif sa:
            v = "solo lo alcanza SIN red"
        elif sr:
            v = "solo lo alcanza CON red"
        else:
            v = "ninguno lo alcanza"
        print(f"{umbral:>16.0e}  {d(sa):<30} {d(sr):<30}  {v}")

    salida = a.out.with_name("kerr_mino_ablacion.json")
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(
        {"escena": {"W": W, "H": H, "rayos": al.size, "cruces": n_cruces,
                    "spin": a.spin, "inc": a.inc, "r_obs": a.r_obs,
                    "fov_deg": a.fov_deg, "r_esc": r_esc},
         "deriva_referencia": deriva, "filas": filas},
        indent=2), encoding="utf-8")
    print(f"\n-> {salida}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resoluciones", nargs="+", default=["160x90", "480x270", "960x540"])
    ap.add_argument("--spin", type=float, default=0.9)
    ap.add_argument("--inc", type=float, default=85.0)
    ap.add_argument("--fov-deg", type=float, default=1.259)
    ap.add_argument("--r-obs", type=float, default=1000.0)
    ap.add_argument("--max-rayos-1e10", type=int, default=200_000,
                    help="por encima de esto se salta el clasico a rtol 1e-10, "
                         "que a 960x540 tarda del orden de 8 minutos el solo")
    ap.add_argument("--r-esc-render", action="store_true",
                    help="usa el radio de escape que usa hoy el renderer "
                         "(max(1e4, 20 r_obs)) en AMBOS caminos. Encarece solo "
                         "al clasico; sirve para citar la cifra realista")
    ap.add_argument("--repeticiones", type=int, default=3,
                    help="repeticiones del camino de Mino; se toma el mejor")
    ap.add_argument("--ablacion", action="store_true",
                    help="en vez de la tabla de rendimiento, corre la ablacion "
                         "de KerrRNet: semilla cruda vs red, con y sin Newton, "
                         "comparadas a igual precision")
    ap.add_argument("--res-ablacion", default="480x270",
                    help="resolucion de la escena para --ablacion")
    ap.add_argument("--modelos", type=Path, default=ROOT / "models/checkpoints_mino")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/benchmarks/kerr_mino_benchmark.json")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net_r = KerrRNet.cargar(a.modelos / "kerr_mino_r.pt", device).to(device)
    net_th = KerrThetaNet.cargar(a.modelos / "kerr_mino_theta.pt", device).to(device)
    k = KMetric(1.0, a.spin)
    ig = KerrGeodesicIntegrator(k)
    th = np.deg2rad(a.inc)
    r_in, r_out = float(k.r_isco()), 20.0
    r_esc = kc.radio_escape(a.r_obs) if a.r_esc_render else 1.05 * a.r_obs

    entorno = {"python": platform.python_version(), "torch": torch.__version__,
               "device": device,
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
               "cpu": platform.processor()}
    print(f"{entorno['gpu'] or entorno['cpu']}   a={a.spin}  inc={a.inc}  "
          f"FOV={a.fov_deg} deg  r_obs={a.r_obs:.0f} M  r_esc={r_esc:.0f} M")

    # Calentamiento: la primera llamada a CUDA arrastra la creacion del
    # contexto y los kernels, decimas de segundo que no son coste por rayo.
    # Sin esto la primera fila de la tabla sale sistematicamente inflada.
    al0, be0 = kc.malla_celeste(k, a.r_obs, th, a.fov_deg, 64, 36)
    pre0 = eng.precalcular(a.spin, th, a.r_obs, al0, be0, n_cruces=6, r_esc=r_esc)
    for me in (True, False):
        eng.evaluar(pre0, net_r, net_th, device, mu_exacto=me)
    eng.evaluar(pre0, None, net_th, device, mu_exacto=True, semilla="cruda")
    if device == "cuda":
        torch.cuda.synchronize()

    if a.ablacion:
        ablacion(a, k, net_r, net_th, device)
        return

    print(f"\n{'resolucion':>11} {'rayos':>9} | {'precalc':>8} {'geo red':>8} "
          f"{'geo exac':>9} | {'TOT red':>8} {'TOT exac':>9} {'ms/rayo':>8} "
          f"| {'clas 1e-6':>10} {'clas 1e-10':>11} | {'x1e-6':>6} {'x1e-10':>7}")
    print("-" * 124)

    filas = []
    for res in a.resoluciones:
        W, H = (int(v) for v in res.lower().split("x"))
        al, be = kc.malla_celeste(k, a.r_obs, th, a.fov_deg, W, H)
        n = al.size

        t_pre = cronometrar(
            lambda: eng.precalcular(a.spin, th, a.r_obs, al, be, n_cruces=6,
                                    r_esc=r_esc), a.repeticiones)
        pre = eng.precalcular(a.spin, th, a.r_obs, al, be, n_cruces=6,
                              r_esc=r_esc)

        t_red = cronometrar(
            lambda: eng.evaluar(pre, net_r, net_th, device, mu_exacto=False),
            a.repeticiones)
        t_exa = cronometrar(
            lambda: eng.evaluar(pre, net_r, net_th, device, mu_exacto=True),
            a.repeticiones)
        tot_red, tot_exa = t_pre + t_red, t_pre + t_exa

        clas = {}
        for tol, rt, at in (("1e-6", 1e-6, 1e-8), ("1e-10", 1e-10, 1e-12)):
            if tol == "1e-10" and n > a.max_rayos_1e10:
                clas[tol] = float("nan")
                continue
            t0 = time.perf_counter()
            ig.trace_batch(al, be, a.r_obs, th, rtol=rt, atol=at,
                           max_steps=40_000, disk_in=r_in, disk_out=r_out,
                           max_crossings=6, r_escape=r_esc)
            clas[tol] = time.perf_counter() - t0

        f = lambda v, w, d=2: (f"{v:{w}.{d}f}s" if np.isfinite(v)
                               else f"{'--':>{w}} ")
        g = lambda v, w: (f"{v:{w}.1f}x" if np.isfinite(v) else f"{'--':>{w}} ")
        print(f"{res:>11} {n:>9,} | {t_pre:7.2f}s {t_red:7.2f}s {t_exa:8.2f}s "
              f"| {tot_red:7.2f}s {tot_exa:8.2f}s {1e3*tot_red/n:7.4f} "
              f"| {f(clas['1e-6'], 9)} {f(clas['1e-10'], 10)} "
              f"| {g(clas['1e-6']/tot_red, 5)} {g(clas['1e-10']/tot_red, 6)}")
        filas.append({"resolucion": res, "rayos": n, "t_precalculo": t_pre,
                      "t_geometria_red": t_red, "t_geometria_exacto": t_exa,
                      "t_total_red": tot_red, "t_total_exacto": tot_exa,
                      "ms_por_rayo_red": 1e3 * tot_red / n,
                      "ms_por_rayo_exacto": 1e3 * tot_exa / n,
                      "t_clasico_rtol1e-6": clas["1e-6"],
                      "t_clasico_rtol1e-10": clas["1e-10"],
                      "speedup_red_vs_1e-6": clas["1e-6"] / tot_red,
                      "speedup_red_vs_1e-10": clas["1e-10"] / tot_red,
                      "speedup_exacto_vs_1e-6": clas["1e-6"] / tot_exa,
                      "ganancia_red_sobre_exacto": tot_exa / tot_red})

    print("\nQue aporta la RED frente a la forma cerrada:")
    for fi in filas:
        print(f"  {fi['resolucion']:>10}  geometria {fi['t_geometria_exacto']:.2f}s"
              f" -> {fi['t_geometria_red']:.2f}s   "
              f"fotograma completo {fi['ganancia_red_sobre_exacto']:.2f}x   "
              f"(el precalculo, {fi['t_precalculo']:.2f}s, es el mismo en los dos"
              f" y no lo toca ninguna red)")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"entorno": entorno, "config": vars(a) | {
        "modelos": str(a.modelos), "out": str(a.out)}, "filas": filas},
        indent=2, default=str), encoding="utf-8")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
