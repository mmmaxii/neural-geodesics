"""Benchmark del trazador de Kerr: como escala el coste con la resolucion.

Mide el camino completo (trazado + sombreado del disco + fondo) a varias
resoluciones y guarda los tiempos en JSON, para poder comparar despues de
tocar el integrador y detectar regresiones.

Tambien anota el entorno (CPU, versiones), porque un tiempo sin la maquina en
la que se midio no sirve para comparar nada.

    python scripts/benchmark_kerr.py
    python scripts/benchmark_kerr.py --spin 0.998 --quick
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from physics.kerr import KMetric                              # noqa: E402
from rendering.classical_renderer import load_sky_image       # noqa: E402
from render_kerr import render                                # noqa: E402


def entorno() -> dict:
    return {
        "cpu": platform.processor() or "desconocido",
        "nucleos_logicos": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "so": f"{platform.system()} {platform.release()}",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spin", type=float, default=0.9)
    p.add_argument("--inc", type=float, default=80.0)
    p.add_argument("--half-width", type=float, default=24.0)
    p.add_argument("--r-out", type=float, default=16.0)
    p.add_argument("--r-obs", type=float, default=1000.0)
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="menos resoluciones")
    p.add_argument("--sky-image", type=Path,
                   default=ROOT / "data" / "raw" / "gaia_panorama.png")
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "benchmarks" / "kerr_benchmark.json")
    a = p.parse_args()

    if a.workers <= 0:
        a.workers = max(1, os.cpu_count() or 2)

    resoluciones = ([(320, 180, 1), (640, 360, 1)] if a.quick else
                    [(320, 180, 1), (480, 270, 1), (640, 360, 1),
                     (960, 540, 1), (1280, 720, 1), (640, 360, 2)])

    sky = load_sky_image(a.sky_image) if a.sky_image.exists() else None
    k = KMetric(M=1.0, a=a.spin)
    disk = {"r_in": k.r_isco(), "r_out": a.r_out, "n_images": 6,
            "turbulence": 0.45, "temp_K": 3800.0, "prograde": True}
    cfg = {"spin": a.spin, "r_obs": a.r_obs, "theta_obs": np.deg2rad(a.inc),
           "rtol": a.rtol, "atol": 1e-8, "sky_brightness": 0.6, "disk": disk}

    env = entorno()
    print(f"CPU: {env['cpu']}  ({env['nucleos_logicos']} hilos), "
          f"{a.workers} procesos")
    print(f"a* = {a.spin}, ISCO = {k.r_isco():.4f} M, inc = {a.inc} deg\n")
    print(f"{'resolucion':>14} {'ss':>3} {'rayos':>10} {'tiempo':>10} "
          f"{'ms/rayo':>9} {'rayos/s':>11}")

    runs = []
    for w, h, ss in resoluciones:
        _, st = render(cfg, w, h, a.half_width, sky, a.workers, ss)
        st.update(width=w, height=h, ss=ss)
        runs.append(st)
        print(f"{w:>6}x{h:<7} {ss:>3} {st['n_rays']:>10} "
              f"{st['t_trace']:>9.2f}s {st['ms_per_ray']:>9.4f} "
              f"{st['n_rays']/st['t_trace']:>11,.0f}")

    # Extrapolacion a resoluciones que no merece la pena medir aqui
    ms = np.median([r["ms_per_ray"] for r in runs])
    print(f"\nmediana: {ms:.4f} ms/rayo. Extrapolando:")
    proy = {}
    for nombre, npx in [("1920x1080", 1920*1080),
                        ("1920x1080 ss2", 1920*1080*4),
                        ("3840x2160 ss2", 3840*2160*4)]:
        s = npx * ms / 1000.0
        proy[nombre] = s
        print(f"  {nombre:>16}: {npx:>10,} rayos -> {s/60:>7.1f} min")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "entorno": env, "config": {k_: v for k_, v in vars(a).items()
                                   if not isinstance(v, Path)},
        "isco": k.r_isco(), "horizonte": k.r_horizon,
        "runs": runs, "mediana_ms_por_rayo": ms,
        "proyeccion_segundos": proy,
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")
    print(f"\nguardado en {a.out}")


if __name__ == "__main__":
    main()
