"""Arma laminas comparativas a partir de los renders de suite_kerr.py.

Cada lamina agrupa los renders que varian UN solo parametro, para poder ver de
un vistazo que efecto tiene ese parametro. Se generan aparte de los renders
porque componer es barato y volver a trazar no.

    python scripts/suite_sheets.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures" / "kerr_suite"

LAMINAS = [
    ("lente", "EL ENCUADRE es lo que ocultaba el lensing. Sombra 5.2 M, "
              "anillo de Einstein 64.7 M",
     [("lente_hw26", "half-width 26 M\nel encuadre que veniamos usando:\n"
                     "tierra de nadie, no se ve nada"),
      ("lente_hw50", "half-width 50 M\nempiezan a asomar los arcos"),
      ("lente_hw80", "half-width 80 M\ncontiene el radio de Einstein"),
      ("lente_hw130", "half-width 130 M\nel anillo queda dentro con margen")], 2),
    ("centro", "Que parche del cielo queda detras (half-width 80 M)",
     [("centro_galactico", "Centro galactico detras\n245 estrellas/deg^2"),
      ("centro_sin_orientar", "Sin orientar\n74.9 estrellas/deg^2 (media)")], 2),
    ("distancia", "Distancia de la camara: b_E = sqrt(4 M r_obs)",
     [("dist_r200", "r_obs = 200 M\nb_E = 28 M"),
      ("dist_r1000", "r_obs = 1000 M\nb_E = 63 M"),
      ("dist_r5000", "r_obs = 5000 M\nb_E = 141 M")], 3),
    ("fondo", "Fondo: panorama vs estrellas puntuales (a*=0.9, i=80)",
     [("fondo_panorama", "Panorama 4096x2048\n(el cielo queda 75x mas grueso "
                         "que el render: manchas)"),
      ("fondo_puntual", "Estrellas puntuales\n(sin limite de resolucion: "
                        "nitidas)")], 2),
    ("spin", "Efecto del espin (i=80, disco co-rotante)",
     [("spin_0p0", "a* = 0  (Schwarzschild)\nISCO 6.00 M"),
      ("spin_0p5", "a* = 0.5\nISCO 4.23 M"),
      ("spin_0p9", "a* = 0.9\nISCO 2.32 M"),
      ("spin_0p998", "a* = 0.998\nISCO 1.24 M")], 2),
    ("spin_norm", "Efecto del espin con el disco normalizado en r=12 M "
                  "(comparacion JUSTA)",
     [("spinnorm_0p0", "a* = 0  (Schwarzschild)\nISCO 6.00 M"),
      ("spinnorm_0p5", "a* = 0.5\nISCO 4.23 M"),
      ("spinnorm_0p9", "a* = 0.9\nISCO 2.32 M"),
      ("spinnorm_0p998", "a* = 0.998\nISCO 1.24 M")], 2),
    ("inc", "Efecto de la inclinacion (a*=0.9)",
     [("inc_10", "i = 10 grados (casi de frente)"),
      ("inc_40", "i = 40 grados"),
      ("inc_70", "i = 70 grados"),
      ("inc_88", "i = 88 grados (casi de canto)")], 2),
    ("sentido", "Disco co-rotante vs contrarrotante (a*=0.9, i=80)",
     [("disco_prograde", "Prograde: ISCO 2.32 M\nel arrastre ayuda"),
      ("disco_retrograde", "Retrogrado: ISCO 8.72 M\nel arrastre estorba")], 2),
    ("ganancia", "Ganancia del cielo puntual: el flujo abarca 4 ordenes de "
                 "magnitud (g=2 vale 0.16, g=12 vale 1.6e-5)",
     [("gain_400", "ganancia 400\nsolo las mas brillantes"),
      ("gain_4000", "ganancia 4000"),
      ("gain_20000", "ganancia 20000\nel campo se puebla")], 3),
    ("disco", "Tamaño del disco (a*=0.9, i=80)",
     [("rout_8", "r_out = 8 M"),
      ("rout_14", "r_out = 14 M"),
      ("rout_25", "r_out = 25 M")], 3),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=FIG)
    a = p.parse_args()

    for nombre, titulo, items, ncol in LAMINAS:
        disponibles = [(f, c) for f, c in items if (a.dir / f"{f}.png").exists()]
        if not disponibles:
            print(f"  (salto {nombre}: no hay renders todavia)")
            continue
        nrow = -(-len(disponibles) // ncol)
        # alto generoso por fila: los pies llevan dos lineas y con tight_layout
        # solo se solapaban con la imagen de la fila de arriba
        fig, axs = plt.subplots(nrow, ncol, figsize=(7.4 * ncol, 5.1 * nrow),
                                squeeze=False)
        for ax in axs.ravel():
            ax.axis("off")
        for ax, (f, cap) in zip(axs.ravel(), disponibles):
            ax.imshow(mpimg.imread(a.dir / f"{f}.png"))
            ax.set_title(cap, fontsize=10.5, pad=8)
        fig.suptitle(titulo, fontsize=15, y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.6)
        out = a.dir / f"lamina_{nombre}.png"
        fig.savefig(out, dpi=100, facecolor="white")
        plt.close(fig)
        print(f"  {out.name}  ({len(disponibles)} renders)")


if __name__ == "__main__":
    main()
