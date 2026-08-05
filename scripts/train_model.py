"""Entrena la red que sustituye al integrador de geodesicas.

    python scripts/train_model.py
    python scripts/train_model.py --ancho 128 --capas 4 --epocas 400

La particion se hace POR RAYO, no por punto. Es lo mas importante de este
script: los 128 puntos de un mismo rayo son casi la misma curva, asi que
repartirlos entre entrenamiento y validacion filtraria informacion y daria un
error de validacion optimista y falso. Separando rayos enteros, la validacion
mide de verdad lo que interesa: si generaliza a parametros de impacto que
nunca vio.
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

from models.geodesic_net import (BETA_CRIT, GeodesicNet,  # noqa: E402
                                 Normalizador, escala_u)


def cargar(ruta: Path, semilla: int, frac_val: float, frac_test: float):
    d = np.load(ruta, allow_pickle=False)
    # beta en DOBLE precision a proposito: en float32 el ULP cerca de
    # beta_crit=2.598 vale 2.4e-7, del mismo orden que la distancia al critico
    # de los rayos que forman el anillo de fotones. Degradarlo aqui borraria esa
    # informacion antes de que la red llegue a verla.
    beta = d["beta"].astype(np.float64)
    phi = d["phi"].astype(np.float32)
    u = d["u_tilde"].astype(np.float32)
    ray = d["ray_index"].astype(np.int64)

    n_rayos = int(ray.max()) + 1
    rng = np.random.default_rng(semilla)
    orden = rng.permutation(n_rayos)
    n_val = int(frac_val * n_rayos)
    n_test = int(frac_test * n_rayos)
    val_r = set(orden[:n_val].tolist())
    test_r = set(orden[n_val:n_val + n_test].tolist())

    es_val = np.isin(ray, list(val_r))
    es_test = np.isin(ray, list(test_r))
    es_tr = ~(es_val | es_test)
    print(f"rayos: {n_rayos - n_val - n_test} entrenamiento / {n_val} validacion "
          f"/ {n_test} prueba   (particion POR RAYO)")
    print(f"puntos: {es_tr.sum():,} / {es_val.sum():,} / {es_test.sum():,}")
    return (beta, phi, u, ray), (es_tr, es_val, es_test)


def tensores(beta, phi, u, m, norm, dev):
    """Devuelve (entradas, objetivo_razon, escala, u_real).

    El objetivo es la RAZON u/escala, no u. La escala es exacta (periastro si
    el rayo escapa, 1 si cae), asi que reaplicarla al final no mete error, y a
    cambio la salida es de orden 1 en todo el dominio en vez de valer 0.07 en
    campo lejano y 1 junto al horizonte.
    """
    b = torch.from_numpy(beta[m])              # float64, ver cargar()
    p = torch.from_numpy(phi[m])
    uu = torch.from_numpy(u[m])
    esc = escala_u(b, xp=torch).float()
    X = norm.features(b, p, xp=torch).float().to(dev)
    return X, (uu / esc).clamp(0.0, 1.0).to(dev), esc.to(dev), uu.to(dev)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datos", type=Path,
                   default=ROOT / "data" / "processed" / "geodesics_dataset.npz")
    p.add_argument("--ancho", type=int, default=64)
    p.add_argument("--capas", type=int, default=3)
    p.add_argument("--epocas", type=int, default=300)
    p.add_argument("--lote", type=int, default=8192)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--semilla", type=int, default=20260805)
    p.add_argument("--frac-val", type=float, default=0.1)
    p.add_argument("--frac-test", type=float, default=0.1)
    p.add_argument("--salida", type=Path,
                   default=ROOT / "models" / "checkpoints" / "geodesic_net.pt")
    a = p.parse_args()

    torch.manual_seed(a.semilla)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dispositivo: {dev}   red {a.ancho}x{a.capas}")

    (beta, phi, u, ray), (m_tr, m_val, m_te) = cargar(
        a.datos, a.semilla, a.frac_val, a.frac_test)

    # las escalas se toman SOLO del entrenamiento: usar todo el dataset seria
    # filtrar informacion de validacion, aunque sea poca
    s_tr = np.log10(np.abs(beta[m_tr] - BETA_CRIT) + 1e-9)
    norm = Normalizador(s_min=float(s_tr.min()), s_max=float(s_tr.max()),
                        beta_max=float(beta[m_tr].max()),
                        phi_max=float(np.abs(phi[m_tr]).max()))
    print(f"normalizacion: s en [{norm.s_min:.2f}, {norm.s_max:.2f}], "
          f"beta_max={norm.beta_max:.2f}, phi_max={norm.phi_max:.2f}")

    Xtr, ytr, _, _ = tensores(beta, phi, u, m_tr, norm, dev)
    Xva, yva, esc_va, u_va = tensores(beta, phi, u, m_val, norm, dev)
    Xte, yte, esc_te, u_te = tensores(beta, phi, u, m_te, norm, dev)

    modelo = GeodesicNet(a.ancho, a.capas, norm).to(dev)
    n_par = sum(q.numel() for q in modelo.parameters())
    print(f"parametros: {n_par:,}")

    opt = torch.optim.Adam(modelo.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epocas)
    perdida = torch.nn.MSELoss()

    n = Xtr.shape[0]
    mejor, mejor_ep, hist = float("inf"), -1, []
    t0 = time.perf_counter()
    for ep in range(a.epocas):
        modelo.train()
        perm = torch.randperm(n, device=dev)
        acum = 0.0
        for i in range(0, n, a.lote):
            idx = perm[i:i + a.lote]
            opt.zero_grad()
            l = perdida(modelo(Xtr[idx]), ytr[idx])
            l.backward()
            opt.step()
            acum += float(l) * idx.numel()
        sched.step()

        modelo.eval()
        with torch.no_grad():
            pv = modelo(Xva)
            l_val = float(perdida(pv, yva))          # en la razon: lo que optimiza
            # el error se reporta en u FISICA, que es lo que consume el
            # renderer; la razon es solo el truco de parametrizacion
            eu = (pv * esc_va - u_va).abs()
            mae = float(eu.mean())
            peor_v = float(eu.max())
        hist.append({"epoca": ep, "train_mse": acum / n,
                     "val_mse_razon": l_val, "val_mae_u": mae,
                     "val_peor_u": peor_v})
        if l_val < mejor:
            mejor, mejor_ep = l_val, ep
            a.salida.parent.mkdir(parents=True, exist_ok=True)
            modelo.guardar(a.salida, {"epoca": ep, "val_mse": l_val,
                                      "semilla": a.semilla})
        if ep % 20 == 0 or ep == a.epocas - 1:
            print(f"  epoca {ep:>4}  train {acum/n:.3e}  val {l_val:.3e}  "
                  f"MAE(u) {mae:.5f}  peor(u) {peor_v:.4f}")

    dt = time.perf_counter() - t0
    print(f"\nentrenado en {dt:.1f} s. Mejor validacion en la epoca {mejor_ep} "
          f"(MSE {mejor:.3e})")

    # prueba final con el MEJOR modelo, no con el ultimo
    modelo = GeodesicNet.cargar(a.salida, dev).to(dev)
    with torch.no_grad():
        pt = modelo(Xte) * esc_te
        e = (pt - u_te).abs()
        mse, mae, peor = float((e**2).mean()), float(e.mean()), float(e.max())
    print(f"PRUEBA (rayos nunca vistos, error en u): MSE {mse:.3e}  "
          f"MAE {mae:.5f}  peor {peor:.5f}")

    hp = a.salida.with_suffix(".historial.json")
    hp.write_text(json.dumps({"config": {k: str(v) for k, v in vars(a).items()},
                              "parametros": n_par, "segundos": dt,
                              "test": {"mse": mse, "mae": mae, "peor": peor},
                              "historial": hist}, indent=2), encoding="utf-8")
    print(f"modelo -> {a.salida}\nhistorial -> {hp}")


if __name__ == "__main__":
    main()
