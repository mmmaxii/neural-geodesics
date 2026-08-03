"""Exploración visual del dataset y del modelo de disco.

Genera figuras en results/figures/. No modifica nada: es solo para mirar.

Uso:
    python scripts/explore_dataset.py
    python scripts/explore_dataset.py --path /tmp/pilot.npz --inc 75
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from physics.schwarzschild import SMetric          # noqa: E402
from physics.integrator import GeodesicIntegrator  # noqa: E402
from physics import disk as D                      # noqa: E402

BETA_CRIT = D.BETA_CRIT
BOZZA_C = np.log(216.0 * (7.0 - 4.0 * np.sqrt(3.0))) - np.pi

plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#0d0d16", "axes.facecolor": "#0d0d16",
    "savefig.facecolor": "#0d0d16", "axes.edgecolor": "#4a4a63",
    "grid.color": "#2a2a3d", "font.size": 9, "axes.titlesize": 10,
})


def _bh_background(ax, x_out=22.0, disk=True):
    """Horizonte, esfera de fotones, ISCO y anillo del disco."""
    th = np.linspace(0, 2 * np.pi, 400)
    if disk:
        for xr in (D.X_ISCO, x_out):
            ax.plot(xr * np.cos(th), xr * np.sin(th), color="#f39c12",
                    lw=0.8, ls="--", alpha=0.5, zorder=1)
        ax.fill_between(np.linspace(D.X_ISCO, x_out, 2), -0.02, 0.02,
                        color="#f39c12", alpha=0.0)
    ax.add_patch(plt.Circle((0, 0), D.X_PHOTON, fill=False, ec="#f1c40f",
                            ls=":", lw=1.2, zorder=3))
    ax.add_patch(plt.Circle((0, 0), 1.0, color="black", ec="#e74c3c",
                            lw=1.4, zorder=4))


# ---------------------------------------------------------------- figura 1
def fig_trajectories(out: Path, r0_view=14.0, r_cam=20.0):
    """Abanico de geodésicas saliendo de una cámara en r_cam.

    La parametrización de Binet arranca en phi=0, r=r0, así que TODOS los rayos
    comparten punto de partida: es exactamente el abanico que sale de una
    cámara situada a r_cam. Cada beta corresponde a un píxel distinto, vía
    b = r sin(psi) / sqrt(1 - r_s/r).
    """
    metric = SMetric()
    integ = GeodesicIntegrator(metric)
    rs = metric.r_s

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))

    # --- panel A: barrido amplio de beta
    # Sin máscara sobre los datos: se recorta con xlim/ylim. Filtrar puntos
    # por radio crea cuerdas falsas al unir puntos no adyacentes.
    ax = axes[0]
    betas = np.concatenate([
        np.linspace(0.8, 2.3, 5),                              # caen
        BETA_CRIT * (1 + np.array([-3e-3, 1e-4, 3e-3, 4e-2])),  # frontera
        np.linspace(3.2, 9.0, 7),                              # escapan
    ])
    cmap = plt.get_cmap("turbo")
    for b in betas:
        res = integ.integrate_ray(b=b * rs, r0=r_cam * rs)
        x, phi = res["r"] / rs, res["phi"]
        captured = res["status"] != "escaped"
        col = "#8899aa" if captured else cmap(np.clip((b - 2.4) / 6.8, 0, 1))
        near = abs(b / BETA_CRIT - 1) < 5e-3
        ax.plot(x * np.cos(phi), x * np.sin(phi), color=col,
                lw=1.9 if near else 1.1, alpha=0.55 if captured else 0.95,
                ls="--" if captured else "-", zorder=2)
    _bh_background(ax)
    ax.plot(r_cam, 0, "*", ms=15, color="#ffffff", zorder=6)
    ax.annotate("cámara", (r_cam, 0), textcoords="offset points",
                xytext=(-14, 10), color="white", fontsize=8.5)
    ax.plot([], [], color="#8899aa", ls="--", label=r"capturados ($\beta<\beta_c$)")
    ax.plot([], [], color=cmap(0.6), label=r"escapan ($\beta>\beta_c$)")
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.3)
    ax.set_xlim(-22, 23); ax.set_ylim(-22, 22); ax.set_aspect("equal")
    ax.set_title(f"Abanico de rayos desde una cámara a {r_cam:.0f} $r_s$\n"
                 "(cada β es un píxel; gruesa: β a 0.01% del crítico)")
    ax.set_xlabel("x / r_s"); ax.set_ylabel("y / r_s")

    # --- panel B: zoom en la zona crítica
    ax = axes[1]
    eps = np.array([-3e-2, -3e-3, -3e-5, 3e-5, 3e-3, 3e-2, 2e-1])
    cols = plt.get_cmap("coolwarm")(np.linspace(0, 1, eps.size))
    for e, col in zip(eps, cols):
        b = BETA_CRIT * (1 + e)
        res = integ.integrate_ray(b=b * rs, r0=r0_view * rs)
        x = res["r"] / rs
        phi = res["phi"]
        lab = f"β/β_c−1 = {e:+.0e}"
        ax.plot(x * np.cos(phi), x * np.sin(phi), color=col, lw=1.3, label=lab)
    _bh_background(ax, disk=False)
    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_aspect("equal")
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.3)
    ax.set_title("Zona crítica: a un lado caen, al otro escapan\n"
                 "y en medio dan vueltas alrededor de la esfera de fotones")
    ax.set_xlabel("x / r_s")

    fig.tight_layout(pad=1.6)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figura 2
def fig_deflection(d, out: Path):
    """delta_phi(beta) contra los dos límites analíticos conocidos."""
    rb, st, dphi = d["ray_beta"], d["ray_status"], d["ray_delta_phi"]
    esc = st == 0
    b, y = rb[esc], dphi[esc]
    o = np.argsort(b); b, y = b[o], y[o]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    ax.plot(b, y, ".", ms=1.6, color="#4fc3f7", alpha=0.6, label="dataset")
    bb = np.linspace(BETA_CRIT * 1.02, b.max(), 400)
    ax.plot(bb, 2.0 / bb + 15 * np.pi / (16 * bb**2), color="#2ecc71", lw=1.3,
            label=r"débil: $2/\beta + 15\pi/16\beta^2$")
    bs = BETA_CRIT * (1 + np.logspace(-8, -0.7, 300))
    ax.plot(bs, -np.log(bs / BETA_CRIT - 1) + BOZZA_C, color="#e74c3c", lw=1.3,
            label=r"fuerte: $-\ln(\beta/\beta_c-1) - 0.4002$")
    ax.axvline(BETA_CRIT, color="#f1c40f", ls=":", lw=1.2)
    ax.text(BETA_CRIT * 1.02, 16.5, r"$\beta_c = 3\sqrt{3}/2$", color="#f1c40f", fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel(r"$\beta = b/r_s$")
    ax.set_ylabel(r"$\Delta\varphi$ [rad]")
    ax.set_title("Deflexión: el dataset cubre los dos regímenes")
    ax.legend(fontsize=7.5, framealpha=0.25); ax.grid(alpha=0.25)

    ax = axes[1]
    e = b / BETA_CRIT - 1.0
    m = e > 0
    s = -np.log(e[m])
    ax.plot(s, y[m], ".", ms=1.8, color="#4fc3f7", alpha=0.6, label="dataset")
    ss = np.linspace(s.min(), s.max(), 10)
    ax.plot(ss, ss + BOZZA_C, color="#e74c3c", lw=1.2, label="pendiente 1, ord. −0.4002")
    ax.set_xlabel(r"$-\ln(\beta/\beta_c - 1)$"); ax.set_ylabel(r"$\Delta\varphi$ [rad]")
    ax.set_title("La divergencia es logarítmica: aquí tiene que salir una recta")
    ax.legend(fontsize=7.5, framealpha=0.25); ax.grid(alpha=0.25)

    fig.tight_layout(pad=1.6); fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figura 3
def fig_sampling(d, out: Path):
    """Cómo quedó repartido el muestreo. Aquí se ve si el dataset está sesgado."""
    rb, band, st = d["ray_beta"], d["ray_band"], d["ray_status"]
    dphi, rmin = d["ray_delta_phi"], d["ray_r_min_over_rs"]
    beta, phi, u = d["beta"], d["phi"], d["u_tilde"]
    esc = st == 0

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))

    ax = axes[0, 0]
    names = ["crítica ext.", "crítica int.", "campo lejano", "captura prof."]
    cols = ["#e74c3c", "#9b59b6", "#2ecc71", "#7f8c8d"]
    for i, (n, c) in enumerate(zip(names, cols)):
        v = rb[band == i]
        if v.size:
            ax.hist(v, bins=np.logspace(np.log10(0.02), np.log10(16), 90),
                    color=c, alpha=0.75, label=f"{n} ({v.size})")
    ax.axvline(BETA_CRIT, color="#f1c40f", ls=":", lw=1.2)
    ax.set_xscale("log"); ax.set_xlabel(r"$\beta$"); ax.set_ylabel("rayos")
    ax.set_title("Muestreo por banda"); ax.legend(fontsize=7, framealpha=0.25)

    ax = axes[0, 1]
    ax.hist(dphi[esc], bins=70, color="#4fc3f7", alpha=0.85)
    ax.set_xlabel(r"$\Delta\varphi$ [rad]"); ax.set_ylabel("rayos")
    ax.set_title("Target de regresión: repartido, no aplastado en cero\n"
                 "(esto es lo que compra el muestreo log en β/β_c−1)")

    ax = axes[1, 0]
    ax.plot(rb[esc], rmin[esc], ".", ms=1.6, color="#2ecc71", alpha=0.6)
    ax.axhline(D.X_PHOTON, color="#f1c40f", ls=":", lw=1.2)
    ax.text(6, 1.56, "esfera de fotones (1.5 r_s)", color="#f1c40f", fontsize=7.5)
    ax.axvline(BETA_CRIT, color="#f1c40f", ls=":", lw=1.0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$r_{min} / r_s$")
    ax.set_title("Máximo acercamiento: asíntota exacta en 1.5 r_s")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    idx = d["ray_index"]
    first = np.concatenate(([0], np.flatnonzero(np.diff(idx)) + 1, [idx.size]))
    pick = [i for i in range(min(400, len(first) - 1))
            if esc[i] and 2.0 < rb[i] < 4.0][:3]
    for j, i in enumerate(pick):
        a, bb = first[i], first[i + 1]
        ax.plot(phi[a:bb], u[a:bb], ".-", ms=3, lw=0.6, alpha=0.85,
                label=f"β = {rb[i]:.3f}")
    ax.set_xlabel(r"$\varphi$ [rad]"); ax.set_ylabel(r"$\tilde u = r_s/r$")
    ax.set_title("φ = 0 en el periastro: la curva es simétrica, y los puntos\n"
                 "se apiñan en los hombros, no en la meseta (u ≈ 2/3)")
    ax.legend(fontsize=7, framealpha=0.25); ax.grid(alpha=0.25)

    fig.tight_layout(pad=1.6); fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figura 4
def fig_disk(out: Path, inc_deg=78.0, x_out=20.0):
    """El disco SIN lente: perfil, corrimiento y brillo. El 'antes' del render."""
    inc = np.deg2rad(inc_deg)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))

    ax = axes[0, 0]
    x = np.linspace(D.X_ISCO, 40, 600)
    ax.plot(x, D.temperature_profile(x), color="#e67e22", lw=1.6)
    ax.axvline(49 / 36 * D.X_ISCO, color="#f1c40f", ls=":", lw=1.1)
    ax.text(49 / 36 * D.X_ISCO + 0.6, 0.55, r"pico en $\frac{49}{36}x_{in}$",
            color="#f1c40f", fontsize=8)
    ax.set_xlabel(r"$x = r/r_s$"); ax.set_ylabel("T normalizada")
    ax.set_title("Perfil Shakura-Sunyaev (idéntico para cualquier masa)")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    x = np.linspace(1.55, 30, 600)
    ax.plot(x, D.orbital_velocity(x), color="#3498db", lw=1.5, label="v/c orbital")
    ax.plot(x, D.time_dilation(x), color="#e74c3c", lw=1.5,
            label=r"$\sqrt{1-3M/r}$ (dilatación)")
    ax.axvline(D.X_ISCO, color="#2ecc71", ls="--", lw=1.1)
    ax.text(D.X_ISCO + 0.3, 0.9, "ISCO: v = c/2 exacto", color="#2ecc71", fontsize=8)
    ax.axvline(D.X_PHOTON, color="#f1c40f", ls=":", lw=1.1)
    ax.set_xlabel(r"$x = r/r_s$"); ax.set_ylim(0, 1.05)
    ax.set_title("Cinemática orbital"); ax.legend(fontsize=8, framealpha=0.25)
    ax.grid(alpha=0.25)

    # proyección plana del disco (sin lente): X perpendicular al eje proyectado
    xr = np.linspace(D.X_ISCO, x_out, 400)
    th = np.linspace(0, 2 * np.pi, 720)
    XR, TH = np.meshgrid(xr, th)
    X = XR * np.sin(TH)
    Y = XR * np.cos(TH) * np.cos(inc)
    g = D.redshift_g(XR, D.image_to_bz(X, inc))

    ax = axes[1, 0]
    # cmap "RdBu" (no _r): g>1 -> azul = corrimiento al AZUL = se acerca.
    m = ax.pcolormesh(X, Y, g, cmap="RdBu", shading="auto",
                      norm=TwoSlopeNorm(vcenter=1.0, vmin=np.nanmin(g),
                                        vmax=np.nanmax(g)))
    fig.colorbar(m, ax=ax, label=r"$g = \nu_{obs}/\nu_{em}$")
    ax.add_patch(plt.Circle((0, 0), 1.0, color="black", ec="#e74c3c", lw=1.2))
    ax.set_aspect("equal"); ax.set_title(f"Corrimiento g, inclinación {inc_deg:.0f}°\n"
                                         "azul: g>1, se acerca — rojo: g<1, se aleja")
    ax.set_xlabel("X / r_s"); ax.set_ylabel("Y / r_s")

    ax = axes[1, 1]
    bright = D.beaming_factor(g) * D.temperature_profile(XR) ** 4
    m = ax.pcolormesh(X, Y, bright / np.nanmax(bright), cmap="inferno",
                      shading="auto")
    fig.colorbar(m, ax=ax, label="brillo relativo")
    ax.add_patch(plt.Circle((0, 0), 1.0, color="black", ec="#e74c3c", lw=1.2))
    ax.set_aspect("equal")
    ax.set_title(r"Brillo $\propto g^4 T^4$: un lado domina por Doppler" "\n"
                 "(SIN lente todavía — esto es lo que el renderer va a curvar)")
    ax.set_xlabel("X / r_s")

    fig.tight_layout(pad=1.6); fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=Path,
                   default=ROOT / "data" / "processed" / "geodesics_dataset.npz")
    p.add_argument("--outdir", type=Path, default=ROOT / "results" / "figures")
    p.add_argument("--inc", type=float, default=78.0)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.path, allow_pickle=False)
    cfg = json.loads(str(d["config_json"]))
    print(f"{args.path.name}: {d['ray_beta'].size} rayos, seed {cfg['seed']}")

    jobs = [
        ("01_trayectorias.png", lambda o: fig_trajectories(o)),
        ("02_deflexion.png", lambda o: fig_deflection(d, o)),
        ("03_muestreo.png", lambda o: fig_sampling(d, o)),
        ("04_disco.png", lambda o: fig_disk(o, args.inc)),
    ]
    for name, fn in jobs:
        fn(args.outdir / name)
        print(f"  {args.outdir.name}/{name}")


if __name__ == "__main__":
    main()
