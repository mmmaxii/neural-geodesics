"""Entrena KerrThetaNet y KerrRNet: las dos EDOs desacopladas de Kerr.

    python scripts/train_kerr_mino.py --etapa theta --epocas 3000
    python scripts/train_kerr_mino.py --etapa r --epocas 2500
    python scripts/train_kerr_mino.py --etapa ambos

La particion es POR TRACK (ray_index), nunca por punto suelto: los puntos de
un mismo rayo son casi la misma curva, y mezclarlos entre train/val filtraria
informacion (mismo argumento que en Schwarzschild, train_model.py:6-11).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import physics.kerr_mino as km                                    # noqa: E402
from models.kerr_mino_net import (KerrRNet, KerrThetaNet,          # noqa: E402
                                  NormalizadorR, NormalizadorTheta)


def particion_por_track(n_tracks: int, semilla: int, frac_val=0.1, frac_test=0.1):
    rng = np.random.default_rng(semilla)
    orden = rng.permutation(n_tracks)
    n_v = int(frac_val * n_tracks)
    n_t = int(frac_test * n_tracks)
    val = np.zeros(n_tracks, bool); val[orden[:n_v]] = True
    test = np.zeros(n_tracks, bool); test[orden[n_v:n_v + n_t]] = True
    train = ~(val | test)
    print(f"  tracks: {train.sum():,} entrenamiento / {val.sum():,} validacion "
          f"/ {test.sum():,} prueba  (particion POR TRACK)")
    return train, val, test


def entrenar(modelo, Xtr, Ytr, Xva, Yva, perdida_fn, metrica_fn, epocas,
            lote, lr, dev, salida, nombre):
    """Bucle generico: copiado del patron de train_kerr_model.py:46-85,
    simplificado porque aqui ningun modelo necesita un marco tangente externo.
    """
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epocas)
    n = Xtr.shape[0]
    mejor, hist = float("inf"), []
    t0 = time.perf_counter()
    for ep in range(epocas):
        modelo.train()
        perm = torch.randperm(n, device=dev)
        acum = 0.0
        for i in range(0, n, lote):
            idx = perm[i:i + lote]
            opt.zero_grad()
            pred = modelo(Xtr[idx])
            l = perdida_fn(pred, [t[idx] for t in Ytr])
            l.backward()
            opt.step()
            acum += float(l) * idx.numel()
        sch.step()
        modelo.eval()
        with torch.no_grad():
            met = metrica_fn(modelo(Xva), Yva)
        hist.append({"epoca": ep, "train": acum / n, **met})
        if met["principal"] < mejor:
            mejor = met["principal"]
            modelo.guardar(salida, {"epoca": ep, **met})
        if ep % max(1, epocas // 12) == 0 or ep == epocas - 1:
            det = "  ".join(f"{k} {v:.6g}" for k, v in met.items() if k != "principal")
            print(f"  [{nombre}] epoca {ep:>5}  train {acum/n:.3e}  {det}")
    print(f"  [{nombre}] {time.perf_counter()-t0:.0f} s, mejor {mejor:.6g}")
    return hist


# =========================================================================== theta
def entrenar_theta(args, dev):
    print("\n=== KerrThetaNet: (x, u_mas, m_tilde) -> cos(theta)/sqrt(u_mas) ===")
    d = np.load(args.datos_theta)
    x, y_theta, ratio_g = d["x"], d["y_theta"], d["ratio_g"]
    ray_index = d["ray_index"]
    u_mas_t, m_t = d["track_u_mas"], d["track_m"]
    n_tracks = u_mas_t.size
    print(f"  {x.size:,} muestras de {n_tracks:,} tracks")

    tr, va, _ = particion_por_track(n_tracks, args.semilla)
    m_tilde = m_t / (m_t - 1.0)
    norm = NormalizadorTheta()

    def tensores(mask_tracks):
        # misma convencion que KerrThetaNet.predecir(): x2 = 2*x (x en [0,1/2]
        # crudo), norm.features() hace su propio 2*x2-1 por dentro. Si aqui se
        # construyera 2*x-1 a mano (sin doblar x primero) el canal de x solo
        # cubriria [-1, 0] en vez de [-1, 1], y quedaria DISTINTO de lo que usa
        # predecir() en inferencia -- red entrenada con una convencion, evaluada
        # con otra, error silencioso.
        sel = mask_tracks[ray_index]
        idx = ray_index[sel]
        X = norm.features(
            torch.from_numpy(2.0 * x[sel]).float(),
            torch.from_numpy(u_mas_t[idx]).float(),
            torch.from_numpy(m_tilde[idx]).float(),
            xp=torch,
        ).to(dev)
        Y = [torch.from_numpy(y_theta[sel]).float().to(dev),
            torch.from_numpy(ratio_g[sel]).float().to(dev)]
        return X, Y

    Xtr, Ytr = tensores(tr)
    Xva, Yva = tensores(va)

    def perdida(pred, y):
        yt, rg = pred
        return ((yt - y[0]) ** 2).mean() + ((rg - y[1]) ** 2).mean()

    def metrica(pred, y):
        yt, rg = pred
        mae_mu = (yt - y[0]).abs().mean()
        mae_g = (rg - y[1]).abs().mean()
        return {"principal": float(mae_mu), "mae_mu": float(mae_mu),
                "mae_ratio_g": float(mae_g)}

    modelo = KerrThetaNet(args.ancho, args.capas_theta, NormalizadorTheta()).to(dev)
    print(f"  parametros: {sum(p.numel() for p in modelo.parameters()):,}")
    hist = entrenar(modelo, Xtr, Ytr, Xva, Yva, perdida, metrica,
                    args.epocas_theta, args.lote, args.lr, dev,
                    args.salida / "kerr_mino_theta.pt", "theta")
    return hist


# =============================================================================== r
def entrenar_r(args, dev):
    print("\n=== KerrRNet: 9 features -> (u/escala, Phi_r comprimido) ===")
    d = np.load(args.datos_r)
    lam, u, Phi_r, ray_index = d["lam"], d["u"], d["Phi_r"], d["ray_index"]
    t_xi, t_eta, t_a = d["track_xi"], d["track_eta"], d["track_a"]
    t_lado, t_delta = d["track_lado"], d["track_delta_gap"]
    t_ancla, t_pl = d["track_r_ancla"], d["track_r_plateau"]
    t_laminf = d["track_lambda_inf"]
    n_tracks = t_xi.size
    print(f"  {lam.size:,} muestras de {n_tracks:,} tracks")

    # sustraccion secular EXACTA: Phi_r(lambda) = f_r(r_plateau) * lambda + residuo
    f_r_pl = km.f_r_exacta(t_pl, t_xi, t_a)
    residuo = Phi_r - f_r_pl[ray_index] * lam
    # OJO: la escala va sobre el residuo YA comprimido, o sea asinh(max|res|),
    # no max|res|. Con max|res| (~572) la salida y_phi quedaba encerrada en
    # +-0.012 en vez de +-1, la MSE no tenia practicamente gradiente ahi, y al
    # descomprimir cualquier error de red se multiplicaba por 572: un error de
    # 1e-4 en y_phi se convertia en 0.06 rad en phi.
    asinh_max = float(np.arcsinh(np.max(np.abs(residuo)))) + 1e-6
    u_escala = 1.0 / t_ancla
    u_pl = 1.0 / t_pl
    ratio_u = u * t_ancla[ray_index]

    tr, va, _ = particion_por_track(n_tracks, args.semilla)

    delta_log = np.log10(t_delta + 1e-300)
    norm = NormalizadorR(
        s_min=float(np.percentile(delta_log, 0.1)),
        s_max=float(np.percentile(delta_log, 99.9)),
        xi_max=float(np.max(np.abs(t_xi))) + 1e-6,
        eta_max=float(np.max(t_eta)) + 1e-6,
        lam_hat_max=1.0,
        log_lam_max=float(np.max(np.log10(1.0 + lam))) + 1e-6,
        asinh_max=asinh_max,
    )
    args.salida.mkdir(parents=True, exist_ok=True)

    def tensores(mask_tracks):
        sel = mask_tracks[ray_index]
        idx = ray_index[sel]
        f64 = lambda a: torch.from_numpy(np.ascontiguousarray(a)).double()
        X = norm.features(
            f64(t_delta[idx]), f64(t_lado[idx]), f64(t_xi[idx]), f64(t_eta[idx]),
            f64(t_a[idx]), f64(u_escala[idx]), f64(u_pl[idx]), f64(lam[sel]),
            f64(t_laminf[idx]), xp=torch,
        ).to(dev)
        y_phi = norm.comprimir_phi(residuo[sel], xp=np)
        Y = [torch.from_numpy(ratio_u[sel]).float().to(dev),
            torch.from_numpy(y_phi).float().to(dev)]
        return X, Y

    Xtr, Ytr = tensores(tr)
    Xva, Yva = tensores(va)

    def perdida(pred, y):
        ru, yphi = pred
        # peso alto en phi: medido en el renderer, el error angular de la
        # direccion de escape lo domina Phi_r, no r. r solo hace falta preciso
        # dentro del disco (r <= 20 M), donde ratio_u es de orden 1 y la red ya
        # acierta; en campo lejano r ni siquiera se le pregunta a la red (el
        # motor usa r_esc exacto).
        return ((ru - y[0]) ** 2).mean() + args.peso_phi * ((yphi - y[1]) ** 2).mean()

    def metrica(pred, y):
        ru, yphi = pred
        mae_ratio = (ru - y[0]).abs().mean()
        phi_pred = norm.expandir_phi(yphi, xp=torch)
        phi_ref = norm.expandir_phi(y[1], xp=torch)
        mae_phi = (phi_pred - phi_ref).abs().mean()
        # La metrica de seleccion mezcla las dos: guardar el mejor por
        # mae_ratio_u a secas elegia epocas con phi malo, y es phi quien manda
        # en el error angular final del render. El factor 0.05 las pone en el
        # mismo orden de magnitud (mae_ratio_u ~ 1e-4 frente a mae_phi ~ 4e-3).
        return {"principal": float(mae_ratio) + 0.05 * float(mae_phi),
                "mae_ratio_u": float(mae_ratio), "mae_phi_r": float(mae_phi)}

    modelo = KerrRNet(args.ancho_r, args.capas_r, norm).to(dev)
    print(f"  parametros: {sum(p.numel() for p in modelo.parameters()):,}")
    hist = entrenar(modelo, Xtr, Ytr, Xva, Yva, perdida, metrica,
                    args.epocas_r, args.lote, args.lr, dev,
                    args.salida / "kerr_mino_r.pt", "r")
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa", choices=["theta", "r", "ambos"], default="ambos")
    ap.add_argument("--datos-theta", type=Path,
                    default=ROOT / "data/processed/kerr_mino_theta.npz")
    ap.add_argument("--datos-r", type=Path,
                    default=ROOT / "data/processed/kerr_mino_r.npz")
    ap.add_argument("--ancho", type=int, default=64)
    ap.add_argument("--capas-theta", type=int, default=3)
    ap.add_argument("--ancho-r", type=int, default=96)
    ap.add_argument("--capas-r", type=int, default=4)
    ap.add_argument("--epocas-theta", type=int, default=3000)
    ap.add_argument("--epocas-r", type=int, default=2500)
    ap.add_argument("--lote", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--peso-phi", type=float, default=0.5,
                    help="peso de la perdida de Phi_r frente a la de r")
    ap.add_argument("--semilla", type=int, default=20260805)
    ap.add_argument("--salida", type=Path, default=ROOT / "models/checkpoints_mino")
    args = ap.parse_args()

    torch.manual_seed(args.semilla)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dispositivo: {dev}")
    args.salida.mkdir(parents=True, exist_ok=True)

    hist = {}
    if args.etapa in ("theta", "ambos"):
        hist["theta"] = entrenar_theta(args, dev)[-50:]
    if args.etapa in ("r", "ambos"):
        hist["r"] = entrenar_r(args, dev)[-50:]

    (args.salida / "historial.json").write_text(
        json.dumps({"config": {k: str(v) for k, v in vars(args).items()},
                   **hist}, indent=2), encoding="utf-8")
    print(f"\nmodelos en {args.salida}")


if __name__ == "__main__":
    main()
