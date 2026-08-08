"""Visor interactivo de Kerr en tiempo de Mino, con desglose de tiempos en vivo.

Controles en vivo: campo de vision, distancia de la camara, masa del agujero,
borde exterior del disco, espin e inclinacion. Los paneles laterales muestran
en que se va el tiempo, que es la pregunta interesante de este proyecto.

Por que hay DOS modos y merece la pena verlos comparados
--------------------------------------------------------
    exacto : cos(theta) por Jacobi sn, Phi_r por cuadratura, r pulido por
             Newton. Reproduce el trazador clasico a 0.001 px. Todo eso son
             funciones especiales en CPU, asi que es el modo CARO.
    red    : cos(theta) sale de KerrThetaNet y r de KerrRNet sin pulir. Un
             forward en GPU en vez de decenas de evaluaciones de scipy.

Ese es exactamente el papel que le queda a la red en Kerr: no la precision
--el camino exacto gana ahi por 3500x-- sino la velocidad de fotograma. El
visor deja verlo y medirlo en directo en vez de tener que creerselo.

Sobre la camara: que se mueve y que no
--------------------------------------
Hay tres actores y solo uno se mueve: la CAMARA. El agujero se queda en el
origen y la esfera celeste se queda en el infinito, quieta y sin deformar. El
agujero no dobla el fondo: transforma direcciones. Cada pixel da una direccion
de mirada, la geodesica la convierte en una direccion de llegada al cielo, y
ahi se lee el color.

El deslizador de FOV va en GRADOS y es el encuadre; el de distancia va aparte.
Antes el encuadre estaba en M (era un parametro de impacto disfrazado de campo
de vision), y como la sombra mide siempre ~5 M en esas coordenadas, mover la
distancia no cambiaba el tamano del agujero: solo reescalaba el fondo. Daba la
impresion de que el agujero estaba clavado y de que lo que se alejaba era el
cielo. Ahora acercarse agranda la sombra y el anillo de Einstein, y el fondo se
queda donde estaba. La conversion pixel -> rayo la hace
physics/kerr_camera.py por la tetrada del observador local.

Sobre la masa
-------------
Toda la fisica esta escrita en unidades de M (G = c = 1), asi que cambiar M
manteniendo TODO lo demas en unidades de M no cambiaria ni un pixel. Lo que se
ve de verdad es el tamano angular de la sombra, que va como M/D. Por eso el
deslizador de masa mantiene fija la distancia FISICA de la camara y recalcula
r_obs/M: subir la masa acerca el agujero en unidades de M y la sombra crece.
El FOV, al ser angular, ya no depende de M: no hay que reescalarlo.

    python scripts/visor_kerr_mino.py
    python scripts/visor_kerr_mino.py -W 200 -H 112 --modo red
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import RadioButtons, Slider

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import physics.disk as D                                       # noqa: E402
import physics.kerr_camera as kc                                # noqa: E402
import rendering.kerr_mino_engine as eng                       # noqa: E402
from physics.kerr import KMetric                               # noqa: E402
from models.kerr_mino_net import KerrRNet, KerrThetaNet        # noqa: E402
from rendering.classical_renderer import (                     # noqa: E402
    blackbody_rgb, disk_turbulence, grid_texture, load_sky_image,
    sample_equirect, star_texture, tonemap)


class Visor:
    def __init__(self, args):
        self.a = args
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net_r = KerrRNet.cargar(args.modelos / "kerr_mino_r.pt", self.dev).to(self.dev)
        self.net_th = KerrThetaNet.cargar(args.modelos / "kerr_mino_theta.pt", self.dev).to(self.dev)
        p = args.sky_image
        self.sky = load_sky_image(p) if p and p.exists() else None

        self.W, self.H = args.width, args.height
        self.estado = dict(spin=args.spin, inc=args.inc, fov=args.fov_deg,
                           dist=args.r_obs, masa=1.0, r_out=20.0,
                           modo=args.modo, grid=args.grid)
        # cache de geometria: solo se recalcula si cambia algo que la afecta
        self._clave = None
        self._geo = None
        self._pre = None
        self.t = dict(precalc=0.0, geom=0.0, sombra=0.0, total=0.0, fps=0.0)
        self._n, self._t0 = 0, time.perf_counter()

    # ------------------------------------------------------------------ fisica
    def _params(self):
        """Convierte los deslizadores a las unidades de M que usa el motor.

        El FOV ya no se toca: es un angulo, no una longitud, asi que no depende
        de la masa. La que si depende es la distancia, porque el deslizador de
        masa mantiene fija la distancia FISICA y lo que cambia es r_obs/M.
        """
        e = self.estado
        r_obs_M = max(e["dist"] / e["masa"], 50.0)
        return r_obs_M, e["fov"]

    def geometria(self):
        e = self.estado
        r_obs_M, fov = self._params()
        clave = (e["spin"], e["inc"], round(r_obs_M, 6), round(fov, 6), e["modo"])
        if clave == self._clave:
            self.t["precalc"] = self.t["geom"] = 0.0     # servido de cache
            return self._geo, self._pre

        th = np.deg2rad(e["inc"])
        k = KMetric(1.0, e["spin"])
        al, be = kc.malla_celeste(k, r_obs_M, th, fov, self.W, self.H)

        t0 = time.perf_counter()
        pre = eng.precalcular(e["spin"], th, r_obs_M, al, be, n_cruces=6,
                              r_esc=kc.radio_escape(r_obs_M))
        self.t["precalc"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        geo = eng.evaluar(pre, self.net_r, self.net_th, self.dev,
                          mu_exacto=(e["modo"] == "exacto"))
        if self.dev == "cuda":
            torch.cuda.synchronize()
        self.t["geom"] = time.perf_counter() - t0

        self._clave, self._geo, self._pre = clave, geo, pre
        return geo, pre

    def sombrear(self, geo):
        """Color por pixel. Barato, asi que los deslizadores del disco van fluidos."""
        e = self.estado
        k = KMetric(1.0, e["spin"])
        r_in = float(k.r_isco())
        n = geo["captured"].size
        col = np.zeros((n, 3), np.float32)

        esc = ~geo["captured"] & ~geo["respaldo"]
        if esc.any():
            d = geo["direccion"][esc]
            th_s = np.arccos(np.clip(d[:, 2], -1.0, 1.0))
            lo_s = np.arctan2(d[:, 1], d[:, 0])
            if e["grid"]:
                col[esc] = grid_texture(th_s, lo_s)
            else:
                col[esc] = (sample_equirect(self.sky, th_s, lo_s, 1.0)
                            if self.sky is not None else star_texture(th_s, lo_s))

        cr, cp, nc = geo["cross_r"], geo["cross_phi"], geo["n_cross"]
        xi = geo["xi"]
        for j in range(cr.shape[1]):
            m = (nc > j) & (cr[:, j] >= r_in) & (cr[:, j] <= e["r_out"])
            if not m.any():
                continue
            r_e, ph_e = cr[m, j], cp[m, j]
            g = k.redshift_g(r_e, xi[m], prograde=True)
            prof = D.temperature_profile(r_e, r_in)
            t_obs = np.clip(g * self.a.temp_K * prof, 800.0, None)
            bright = np.clip(g, 0.0, None) ** 4 * prof ** 4
            bright = bright * disk_turbulence(r_e, ph_e, self.a.turbulence)
            col[np.flatnonzero(m)] += (blackbody_rgb(t_obs)
                                       * bright[:, None]).astype(np.float32)
        return col.reshape(self.H, self.W, 3)

    def fotograma(self):
        t_ini = time.perf_counter()
        geo, _ = self.geometria()
        t0 = time.perf_counter()
        img = self.sombrear(geo)
        self.t["sombra"] = time.perf_counter() - t0
        self.t["total"] = time.perf_counter() - t_ini
        self._n += 1
        ahora = time.perf_counter()
        if ahora - self._t0 > 0.7:
            self.t["fps"] = self._n / (ahora - self._t0)
            self._t0, self._n = ahora, 0
        return np.clip(tonemap(img, exposure=self.a.exposure), 0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-W", "--width", type=int, default=180)
    p.add_argument("-H", "--height", type=int, default=101)
    p.add_argument("--spin", type=float, default=0.9)
    p.add_argument("--inc", type=float, default=80.0)
    p.add_argument("--fov-deg", type=float, default=1.4,
                   help="campo de vision horizontal en grados (encuadre inicial)")
    p.add_argument("--r-obs", type=float, default=1000.0)
    p.add_argument("--grid", action="store_true",
                   help="rejilla de coordenadas de fondo en vez del cielo: es "
                        "lo que hay que mirar para juzgar el lensing")
    p.add_argument("--modo", choices=["exacto", "red"], default="red")
    p.add_argument("--temp-K", type=float, default=3800.0)
    p.add_argument("--turbulence", type=float, default=0.45)
    p.add_argument("--exposure", type=float, default=1.0)
    p.add_argument("--sky-image", type=Path,
                   default=ROOT / "data/raw/gaia_panorama.png")
    p.add_argument("--modelos", type=Path, default=ROOT / "models/checkpoints_mino")
    a = p.parse_args()

    v = Visor(a)
    print(f"dispositivo {v.dev}   {a.width}x{a.height} = {a.width*a.height:,} rayos")

    fig = plt.figure(figsize=(15.5, 8.2))
    fig.canvas.manager.set_window_title("Kerr en tiempo de Mino - visor")
    ax_im = fig.add_axes([0.20, 0.30, 0.60, 0.66])
    ax_im.axis("off")
    disp = ax_im.imshow(v.fotograma(), interpolation="bilinear")
    titulo = ax_im.set_title("", fontsize=10)

    # ------- paneles laterales: IZQUIERDA que se calcula, DERECHA cuanto cuesta
    ax_izq = fig.add_axes([0.005, 0.30, 0.185, 0.66]); ax_izq.axis("off")
    ax_der = fig.add_axes([0.805, 0.30, 0.19, 0.66]); ax_der.axis("off")
    txt_izq = ax_izq.text(0.0, 1.0, "", va="top", ha="left", fontsize=8.5,
                          family="monospace", transform=ax_izq.transAxes)
    txt_der = ax_der.text(0.0, 1.0, "", va="top", ha="left", fontsize=8.5,
                          family="monospace", transform=ax_der.transAxes)

    # ------------------------------------------------------------ deslizadores
    def sl(y, label, lo, hi, ini, fmt="%.2f"):
        ax = fig.add_axes([0.28, y, 0.45, 0.022])
        return Slider(ax, label, lo, hi, valinit=ini, valfmt=fmt)

    def sl_log(y, label, lo, hi, ini, sufijo, fmt="{:.2f}"):
        """Deslizador logaritmico: guarda log10(valor) y muestra el valor.

        Hace falta para el FOV y la distancia porque los dos abarcan mas de dos
        ordenes de magnitud (0.2 a 60 grados, 50 a 20000 M) y en un
        deslizador lineal todo el rango util quedaria apelmazado en el extremo.
        """
        s = sl(y, label, np.log10(lo), np.log10(hi), np.log10(ini))
        s.valtext.set_text(fmt.format(ini) + sufijo)
        s._mostrar = lambda: s.valtext.set_text(fmt.format(10.0 ** s.val) + sufijo)
        return s

    # FOV y distancia son ya INDEPENDIENTES: mover uno no cambia lo que hace el
    # otro. Ese es justo el arreglo -- antes el "FOV" iba en M y era un
    # parametro de impacto, asi que la distancia solo reescalaba el fondo.
    s_fov = sl_log(0.235, "FOV (deg)", 0.2, 60.0, a.fov_deg, " deg")
    s_dist = sl_log(0.195, "distancia (M)", 50.0, 20000.0, a.r_obs, " M", "{:.0f}")
    s_masa = sl(0.155, "masa M", 0.3, 3.0, 1.0)
    s_rout = sl(0.115, "disco r_out (M)", 6.0, 40.0, 20.0)
    s_spin = sl(0.075, "espin a*", 0.0, 0.998, a.spin, "%.3f")
    s_inc = sl(0.035, "inclinacion", 5.0, 89.0, a.inc, "%.0f")

    ax_modo = fig.add_axes([0.03, 0.03, 0.14, 0.20])
    ax_modo.set_title("modo", fontsize=9)
    radio = RadioButtons(ax_modo, ("red (rapido)", "exacto (0.001 px)"),
                         active=0 if a.modo == "red" else 1)

    def cambio(_=None):
        s_fov._mostrar(); s_dist._mostrar()
        v.estado.update(fov=10.0 ** s_fov.val, dist=10.0 ** s_dist.val,
                        masa=s_masa.val, r_out=s_rout.val, spin=s_spin.val,
                        inc=s_inc.val)
    for s in (s_fov, s_dist, s_masa, s_rout, s_spin, s_inc):
        s.on_changed(cambio)
    radio.on_clicked(lambda lbl: v.estado.update(
        modo="red" if lbl.startswith("red") else "exacto"))

    def tecla(ev):
        if ev.key == "q":
            plt.close(fig)
        elif ev.key == "r":
            for s in (s_fov, s_dist, s_masa, s_rout, s_spin, s_inc):
                s.reset()

    fig.canvas.mpl_connect("key_press_event", tecla)

    def actualizar(_):
        img = v.fotograma()
        disp.set_data(img)
        t, e = v.t, v.estado
        r_obs_M, fov = v._params()
        n = v.W * v.H

        # tamano angular de la sombra: es la cifra que delataba el problema.
        # Con la camara vieja no se movia al cambiar la distancia; ahora va
        # como ~1/r_obs, que es lo que hace un objeto al alejarse. La formula
        # cerrada solo existe con a = 0; con espin la sombra no es un circulo,
        # asi que se marca como aproximada.
        psi_sh = kc.radio_angular_sombra_schwarzschild(r_obs_M)
        px_sh = np.tan(psi_sh) / kc.rad_por_pixel(fov, v.W)
        aprox = " " if e["spin"] == 0.0 else "~"

        cache = " (cache)" if t["geom"] == 0.0 else ""
        txt_izq.set_text(
            "QUE SE CALCULA\n"
            "==============\n\n"
            "EXACTO (forma cerrada)\n"
            "  xi, eta desde el pixel\n"
            "  sombra: raices del\n"
            "    cuartico R(r)\n"
            "  lambda: Carlson R_F\n"
            "  Lam_theta, fase camara\n"
            "  cruces: x = 1/4 + k/2\n"
            "  G_phi: Carlson R_J\n"
            "  Phi_r: cuadratura w\n\n"
            f"MODO: {e['modo'].upper()}\n"
            "  r: Newton sobre Carlson\n"
            "     (semilla analitica;\n"
            "      KerrRNet retirada:\n"
            "      31 M de error en el\n"
            "      peor cruce y +37%\n"
            "      de coste, medido)\n"
            + ("  cos(theta): Jacobi sn\n"
               if e["modo"] == "exacto" else
               "  cos(theta): KerrThetaNet\n")
            + "\n"
            "CAMARA\n"
            f"  r_obs = {r_obs_M:8.1f} M\n"
            f"  FOV   = {fov:8.3f} deg\n"
            f"  sombra{aprox}{2*np.rad2deg(psi_sh):7.3f} deg\n"
            f"       {aprox}{2*px_sh:7.1f} px de {v.W}\n"
            "  (al acercarse crece:\n"
            "   antes no se movia)\n\n"
            "GEOMETRIA\n"
            f"  r_isco= {KMetric(1.0, e['spin']).r_isco():8.3f} M\n"
            f"  r_+   = {KMetric(1.0, e['spin']).r_horizon:8.3f} M")

        tot = max(t["total"], 1e-9)
        barra = lambda x: "#" * int(round(28 * x / tot))
        txt_der.set_text(
            "CUANTO TARDA\n"
            "============\n\n"
            f"{n:,} rayos\n\n"
            f"precalculo exacto{cache}\n"
            f"  {t['precalc']*1e3:8.1f} ms  {100*t['precalc']/tot:4.1f}%\n"
            f"  {barra(t['precalc'])}\n\n"
            f"geometria ({e['modo']}){cache}\n"
            f"  {t['geom']*1e3:8.1f} ms  {100*t['geom']/tot:4.1f}%\n"
            f"  {barra(t['geom'])}\n\n"
            "sombreado (disco+cielo)\n"
            f"  {t['sombra']*1e3:8.1f} ms  {100*t['sombra']/tot:4.1f}%\n"
            f"  {barra(t['sombra'])}\n\n"
            "-----------------------\n"
            f"TOTAL {t['total']*1e3:9.1f} ms\n"
            f"      {t['fps']:9.1f} fps\n"
            f"      {1e3*t['total']/n:9.4f} ms/rayo\n\n"
            "el trazador clasico tarda\n"
            "~0.24 ms/rayo a rtol 1e-6,\n"
            "~2.1 ms/rayo a rtol 1e-10")

        titulo.set_text(f"a* = {e['spin']:.3f}   inc = {e['inc']:.0f} deg   "
                        f"M = {e['masa']:.2f}   disco hasta {e['r_out']:.0f} M   "
                        f"[{e['modo']}]  {t['fps']:.1f} fps")
        return disp, titulo, txt_izq, txt_der

    from matplotlib.animation import FuncAnimation
    _anim = FuncAnimation(fig, actualizar, interval=30, blit=False,
                          cache_frame_data=False)
    print("controles: deslizadores; 'r' reinicia, 'q' sale")
    print("nota: mover solo el disco NO recalcula la geometria (sale de cache)")
    plt.show()


if __name__ == "__main__":
    main()
