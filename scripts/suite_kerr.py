"""Tanda de simulaciones de Kerr, con registro de tiempos.

Lanza una bateria de renders variando un parametro cada vez, para poder ver
que efecto tiene cada cosa por separado, y deja todos los tiempos en un JSON.

    python scripts/suite_kerr.py                 # tanda completa
    python scripts/suite_kerr.py --only spin     # solo un bloque
    python scripts/suite_kerr.py --dry-run       # lista lo que haria
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures" / "kerr_suite"
CAT = ROOT / "data" / "raw" / "gaia_stars_mag12.npz"
PAN = ROOT / "data" / "raw" / "gaia_panorama_mag12.png"

# Encuadre comun a casi todo, para que las comparaciones sean justas.
# El revelado va aparte para poder cambiarlo sin duplicar argumentos: si un
# bloque añade su propio --exposure y BASE ya trae otro, argparse se queda con
# el ultimo y el ajuste pasa desapercibido.
GEOM = ["--half-width", "26", "--r-out", "18", "--n-images", "8",
        "--turbulence", "0.45"]
REVELADO = ["--exposure", "1.2", "--bloom", "0.6"]
BASE = GEOM + REVELADO
STARS = ["--stars", str(CAT), "--star-gain", "400", "--sky-brightness", "1.0"]
MED = ["-W", "1280", "-H", "720"]


def jobs(only: str | None) -> list[dict]:
    J: list[dict] = []

    # 1. La pregunta pendiente de ayer: panorama vs puntos en primer plano
    J.append(dict(grupo="fondo", nombre="fondo_panorama",
                  args=MED + BASE + ["--spin", "0.9", "--inc", "80",
                                     "--sky-image", str(PAN),
                                     "--sky-brightness", "0.55"]))
    J.append(dict(grupo="fondo", nombre="fondo_puntual",
                  args=MED + BASE + STARS + ["--spin", "0.9", "--inc", "80"]))

    # 2. Efecto del espin, todo lo demas igual
    for s in ["0.0", "0.5", "0.9", "0.998"]:
        J.append(dict(grupo="spin", nombre=f"spin_{s.replace('.','p')}",
                      args=MED + BASE + STARS + ["--spin", s, "--inc", "80"]))

    # 2b. El mismo barrido de espin pero con el disco normalizado en un radio
    # FIJO. Sin esto la comparacion engaña: temperature_profile normaliza a su
    # maximo, que esta en (49/36)*ISCO, asi que al subir el espin el pico se
    # mete hacia dentro y el disco exterior parece apagarse. En realidad la
    # eficiencia radiativa sube del 5.7% al 32%.
    for s in ["0.0", "0.5", "0.9", "0.998"]:
        J.append(dict(grupo="spin_norm", nombre=f"spinnorm_{s.replace('.','p')}",
                      args=MED + BASE + STARS + ["--spin", s, "--inc", "80",
                                                 "--norm-r", "12"]))

    # 3. Efecto de la inclinacion con espin alto
    for i in ["10", "40", "70", "88"]:
        J.append(dict(grupo="inc", nombre=f"inc_{i}",
                      args=MED + BASE + STARS + ["--spin", "0.9", "--inc", i]))

    # 4. Disco co-rotante vs contrarrotante: cambia el ISCO y el brillo
    J.append(dict(grupo="sentido", nombre="disco_prograde",
                  args=MED + BASE + STARS + ["--spin", "0.9", "--inc", "80"]))
    J.append(dict(grupo="sentido", nombre="disco_retrograde",
                  args=MED + BASE + STARS + ["--spin", "0.9", "--inc", "80",
                                             "--retrograde"]))

    # 5. Tamaño del disco
    for ro in ["8", "14", "25"]:
        J.append(dict(grupo="disco", nombre=f"rout_{ro}",
                      args=MED + ["--half-width", "26", "--r-out", ro,
                                  "--n-images", "8", "--turbulence", "0.45",
                                  "--exposure", "1.2", "--bloom", "0.6"]
                      + STARS + ["--spin", "0.9", "--inc", "80"]))

    # 5b. Cuanta ganancia hace falta para que el cielo puntual se pueble.
    # El flujo es 10^(-0.4 g): una estrella de g=12 vale 1.6e-5 y una de g=2
    # vale 0.16, o sea cuatro ordenes de magnitud de rango. Con ganancia baja
    # solo salen las brillantes y el campo parece vacio.
    for gain in ["400", "4000", "20000"]:
        J.append(dict(grupo="ganancia", nombre=f"gain_{gain}",
                      args=MED + BASE + ["--stars", str(CAT),
                                         "--sky-brightness", "1.0",
                                         "--star-gain", gain,
                                         "--spin", "0.9", "--inc", "80"]))

    # 6. Planos generales: alta resolucion, dos espines
    for s in ["0.998", "0.5"]:
        J.append(dict(grupo="hero", nombre=f"hero_spin_{s.replace('.','p')}",
                      args=["-W", "2560", "-H", "1440", "--ss", "2"]
                      + BASE + STARS + ["--spin", s, "--inc", "80"]))

    # 6b. Los mismos planos generales pero con el disco normalizado en radio
    # fijo. Los de arriba usan la normalizacion al pico, que con espin alto
    # apaga el disco entero (ver el bloque spin_norm). Estos son los buenos.
    for s, inc in [("0.998", "80"), ("0.9", "86"), ("0.5", "72")]:
        J.append(dict(grupo="hero2",
                      nombre=f"hero2_a{s.replace('.','p')}_i{inc}",
                      args=["-W", "2560", "-H", "1440", "--ss", "2",
                            "--norm-r", "12"]
                      + GEOM + ["--exposure", "0.75", "--bloom", "0.7"]
                      # ganancia 20000, no 400: el flujo va como 10^(-0.4 g) y
                      # abarca cuatro ordenes de magnitud entre g=2 y g=12. Con
                      # 400 solo salen un puñado de estrellas y el cielo parece
                      # vacio; ver el bloque `ganancia`.
                      + ["--stars", str(CAT), "--sky-brightness", "1.0",
                         "--star-gain", "20000"]
                      + ["--spin", s, "--inc", inc]))

    return [j for j in J if only is None or j["grupo"] == only]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "benchmarks" / "kerr_suite.json")
    a = p.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    lista = jobs(a.only)
    print(f"{len(lista)} renders\n")

    hechos = []
    t_all = time.perf_counter()
    for i, j in enumerate(lista, 1):
        salida = FIG / f"{j['nombre']}.png"
        cmd = [sys.executable, str(ROOT / "scripts" / "render_kerr.py"),
               *j["args"], "--out", str(salida)]
        print(f"[{i}/{len(lista)}] {j['grupo']}/{j['nombre']}")
        if a.dry_run:
            print("   ", " ".join(cmd[2:]))
            continue
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=str(ROOT), env={**os.environ})
        dt = time.perf_counter() - t0
        ok = r.returncode == 0
        # el renderer imprime los rayos y el ms/rayo; se rescata para el log
        linea = next((l.strip() for l in r.stdout.splitlines()
                      if "rayos en" in l), "")
        print(f"    {'ok ' if ok else 'FALLO'} {dt:6.1f} s   {linea}")
        if not ok:
            print("   ", (r.stderr or "").strip().splitlines()[-1:] or "")
        hechos.append(dict(**{k: v for k, v in j.items() if k != "args"},
                           segundos=dt, ok=ok, salida=str(salida),
                           detalle=linea))
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(
            {"total_s": time.perf_counter() - t_all, "runs": hechos},
            indent=2), encoding="utf-8")

    total = time.perf_counter() - t_all
    print(f"\ntotal {total/60:.1f} min   ->  {a.out}")


if __name__ == "__main__":
    main()
