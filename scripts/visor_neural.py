"""Visor interactivo del agujero negro de Kerr, renderizado por la red.

Cada fotograma sale de la red neuronal, sin integrar ni una sola geodesica.
El trazado clasico de la misma imagen tarda unos 216 segundos; aqui va a
decenas de fotogramas por segundo.

    python scripts/visor_neural.py
    python scripts/visor_neural.py -W 960 -H 540 --spin 0.998

Controles:
    flechas arriba/abajo   inclinacion de la camara
    flechas izq/derecha    giro del cielo
    espacio                orbita automatica
    r                      reiniciar la vista
    q                      salir

La inclinacion se mueve por una tabla precalculada. Sin ella, cada cambio de
camara obliga a rehacer la distancia al borde de la sombra (~100 ms) y el visor
cae de 78 a 8 fps; con la tabla, mover la camara cuesta lo mismo que no moverla.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-W", "--width", type=int, default=640)
    p.add_argument("-H", "--height", type=int, default=360)
    p.add_argument("--spin", type=float, default=0.9)
    p.add_argument("--inc", type=float, default=80.0)
    # 30 M y no 80: a 80 la sombra ocupaba el 7% del ancho y solo se veian los
    # arcos del borde. Con 30 el disco es el sujeto y el agujero se ve como un
    # objeto, no como un punto.
    p.add_argument("--half-width", type=float, default=30.0)
    p.add_argument("--r-obs", type=float, default=1000.0)
    p.add_argument("--brillo", type=float, default=0.4)
    p.add_argument("--exposure", type=float, default=1.1)
    p.add_argument("--disco", type=Path,
                   default=ROOT / "models" / "checkpoints_v2" / "kerr_disk.pt")
    p.add_argument("--sin-disco", action="store_true")
    p.add_argument("--r-out", type=float, default=18.0)
    p.add_argument("--inc-min", type=float, default=8.0)
    p.add_argument("--inc-max", type=float, default=89.0)
    p.add_argument("--paso-inc", type=float, default=1.0,
                   help="resolucion de la tabla de inclinaciones, en grados")
    p.add_argument("--red", type=Path,
                   default=ROOT / "models" / "checkpoints_residual" / "kerr_sky.pt")
    p.add_argument("--sky-image", type=Path,
                   default=ROOT / "data" / "raw" / "gaia_panorama_mag12.png")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import torch
    from rendering.classical_renderer import load_sky_image, tonemap
    from rendering.neural_live import MotorNeural

    if not torch.cuda.is_available():
        print("aviso: sin CUDA esto va a ir lento")

    sky = load_sky_image(a.sky_image) if a.sky_image.exists() else None
    disco = None if a.sin_disco else (a.disco if a.disco.exists() else None)
    if disco is None and not a.sin_disco:
        print(f"aviso: no encuentro {a.disco}, va sin disco")
    motor = MotorNeural(a.red, a.width, a.height, a.half_width, a.r_obs,
                        "cuda" if torch.cuda.is_available() else "cpu", sky,
                        ruta_disco=disco, r_out=a.r_out)

    incs = np.arange(a.inc_min, a.inc_max + 1e-9, a.paso_inc)
    print(f"precalculando la geometria de {len(incs)} inclinaciones...")
    t0 = time.perf_counter()
    motor.precalcular(a.spin, incs)
    print(f"  listo en {time.perf_counter()-t0:.1f} s")

    estado = {"inc": float(np.clip(a.inc, a.inc_min, a.inc_max)),
              "roll": 0.0, "orbita": False, "t": time.perf_counter(),
              "fps": 0.0, "n": 0}

    fig, ax = plt.subplots(figsize=(a.width / 100, a.height / 100 + 0.5))
    fig.canvas.manager.set_window_title("Kerr neural")
    img = motor.fotograma(a.spin, estado["inc"], estado["roll"], a.brillo)
    disp = ax.imshow(tonemap(img.cpu().numpy(), exposure=a.exposure))
    ax.axis("off")
    titulo = ax.set_title("", fontsize=9)
    fig.tight_layout()

    def tecla(ev):
        if ev.key in ("up", "down"):
            d = a.paso_inc if ev.key == "up" else -a.paso_inc
            estado["inc"] = float(np.clip(estado["inc"] + d, a.inc_min, a.inc_max))
        elif ev.key in ("left", "right"):
            estado["roll"] += 0.05 * (1 if ev.key == "right" else -1)
        elif ev.key == " ":
            estado["orbita"] = not estado["orbita"]
        elif ev.key == "r":
            estado.update(inc=float(a.inc), roll=0.0, orbita=False)
        elif ev.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", tecla)

    def actualizar(_):
        if estado["orbita"]:
            estado["inc"] += a.paso_inc
            if estado["inc"] > a.inc_max or estado["inc"] < a.inc_min:
                estado["inc"] = float(np.clip(estado["inc"], a.inc_min, a.inc_max))
                estado["orbita"] = False
            estado["roll"] += 0.01
        im = motor.fotograma(a.spin, estado["inc"], estado["roll"], a.brillo)
        disp.set_data(tonemap(im.cpu().numpy(), exposure=a.exposure))
        estado["n"] += 1
        ahora = time.perf_counter()
        if ahora - estado["t"] > 0.5:
            estado["fps"] = estado["n"] / (ahora - estado["t"])
            estado["t"], estado["n"] = ahora, 0
        titulo.set_text(f"a* = {a.spin}   inclinacion = {estado['inc']:.0f} deg"
                        f"   {estado['fps']:.0f} fps   "
                        f"(el trazado clasico tarda ~216 s por fotograma)")
        return disp, titulo

    from matplotlib.animation import FuncAnimation
    _anim = FuncAnimation(fig, actualizar, interval=1, blit=False,
                         cache_frame_data=False)
    print("\ncontroles: flechas arriba/abajo inclinacion, izq/der giro del cielo,"
          "\n           espacio orbita automatica, r reiniciar, q salir")
    plt.show()


if __name__ == "__main__":
    main()
