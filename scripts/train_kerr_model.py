"""Entrena las dos redes de Kerr: cielo y disco.

    python scripts/train_kerr_model.py --epocas 3000

La particion se hace por CONFIGURACION (espin, inclinacion), no por rayo. Es
mas exigente que separar rayos sueltos y es la pregunta que de verdad importa:
el renderer va a pedir espines e inclinaciones que la red nunca vio, asi que
hay que medir si interpola en esos parametros, no solo dentro de una
configuracion ya conocida.
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

from models.kerr_net import (MAX_CRUCES, KerrDiskNet,  # noqa: E402
                             KerrSkyNet, NormalizadorKerr, marco_tangente)


def particion(spin, inc, semilla, frac_val=0.12, frac_test=0.12):
    """Separa CONFIGURACIONES enteras, no rayos."""
    cfg = np.stack([spin, inc], 1)
    uniq, inv = np.unique(cfg, axis=0, return_inverse=True)
    rng = np.random.default_rng(semilla)
    orden = rng.permutation(uniq.shape[0])
    n_v = int(frac_val * uniq.shape[0])
    n_t = int(frac_test * uniq.shape[0])
    val_c, test_c = set(orden[:n_v].tolist()), set(orden[n_v:n_v + n_t].tolist())
    es_val = np.isin(inv, list(val_c))
    es_test = np.isin(inv, list(test_c))
    print(f"configuraciones: {uniq.shape[0] - n_v - n_t} entrenamiento / "
          f"{n_v} validacion / {n_t} prueba   (particion POR CONFIGURACION)")
    return ~(es_val | es_test), es_val, es_test


def entrenar(modelo, Xtr, Ytr, Xva, Yva, perdida_fn, metrica_fn, epocas,
             lote, lr, dev, salida, nombre, Mtr=None, Mva=None):
    """Mtr/Mva son el marco tangente de cada muestra, si el modelo es residual.

    Va aparte de las entradas porque no es una feature que la red mire: es la
    base sobre la que se aplica su salida.
    """
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epocas)
    n = Xtr.shape[0]
    mejor, hist = float("inf"), []
    corta = lambda M, i: None if M is None else tuple(t[i] for t in M)
    t0 = time.perf_counter()
    for ep in range(epocas):
        modelo.train()
        perm = torch.randperm(n, device=dev)
        acum = 0.0
        for i in range(0, n, lote):
            idx = perm[i:i + lote]
            opt.zero_grad()
            pred = (modelo(Xtr[idx], corta(Mtr, idx)) if Mtr is not None
                    else modelo(Xtr[idx]))
            l = perdida_fn(pred, [t[idx] for t in Ytr])
            l.backward()
            opt.step()
            acum += float(l) * idx.numel()
        sch.step()
        modelo.eval()
        with torch.no_grad():
            pv = modelo(Xva, Mva) if Mva is not None else modelo(Xva)
            met = metrica_fn(pv, Yva)
        hist.append({"epoca": ep, "train": acum / n, **met})
        if met["principal"] < mejor:
            mejor = met["principal"]
            modelo.guardar(salida, {"epoca": ep, **met})
        if ep % max(1, epocas // 12) == 0 or ep == epocas - 1:
            det = "  ".join(f"{k} {v:.5f}" for k, v in met.items() if k != "principal")
            print(f"  [{nombre}] epoca {ep:>5}  train {acum/n:.3e}  {det}")
    print(f"  [{nombre}] {time.perf_counter()-t0:.0f} s, mejor {mejor:.6f}")
    return hist


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datos", type=Path,
                   default=ROOT / "data" / "processed" / "kerr_dataset.npz")
    p.add_argument("--ancho", type=int, default=64)
    p.add_argument("--capas-cielo", type=int, default=3)
    p.add_argument("--capas-disco", type=int, default=4)
    p.add_argument("--epocas", type=int, default=3000)
    p.add_argument("--lote", type=int, default=65536)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--semilla", type=int, default=20260805)
    p.add_argument("--r-obs", type=float, default=1000.0,
                   help="tiene que coincidir con el del dataset")
    p.add_argument("--sin-residual", dest="residual", action="store_false",
                   help="predice la direccion absoluta en vez de la desviacion")
    p.add_argument("--salida", type=Path, default=ROOT / "models" / "checkpoints")
    a = p.parse_args()

    torch.manual_seed(a.semilla)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dispositivo: {dev}")

    d = np.load(a.datos, allow_pickle=False)
    al, be = d["alpha"], d["beta"]
    sp, inc, dist = d["spin"], d["inc_deg"], d["dist_borde"]
    cap, dirs = d["captured"], d["direction"]
    cr, cphi, ncr = d["cross_r"], d["cross_phi"], d["n_cross"]
    print(f"rayos: {al.size:,}   capturados {cap.mean()*100:.1f}%")

    m_tr, m_va, m_te = particion(sp, inc, a.semilla)
    norm = NormalizadorKerr(hw=float(np.hypot(al, be).max()),
                            d_min=float(np.log10(dist[dist > 0].min() + 1e-9)),
                            d_max=float(np.log10(dist.max() + 1e-9)))
    a.salida.mkdir(parents=True, exist_ok=True)

    def feats(m):
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x[m])).double()
        return norm.features(t(al), t(be), t(sp), t(inc), t(dist),
                             xp=torch).float().to(dev)

    # ---------------------------------------------------------------- CIELO
    print("\n=== red del CIELO (direccion asintotica) ===")
    esc = ~cap
    Xtr, Xva = feats(m_tr & esc), feats(m_va & esc)
    Ytr = [torch.from_numpy(dirs[m_tr & esc]).float().to(dev)]
    Yva = [torch.from_numpy(dirs[m_va & esc]).float().to(dev)]
    print(f"  {Xtr.shape[0]:,} entrenamiento / {Xva.shape[0]:,} validacion")

    def perd_cielo(pred, y):
        # 1 - coseno: optimiza el ANGULO directamente. Con MSE sobre las
        # componentes se optimizaria una cuerda, que penaliza distinto segun
        # donde caiga el vector.
        return (1.0 - (pred * y[0]).sum(-1)).mean()

    def met_cielo(pred, y):
        cos = (pred * y[0]).sum(-1).clamp(-1, 1)
        ang = torch.rad2deg(torch.arccos(cos))
        return {"principal": float(ang.mean()), "err_medio_deg": float(ang.mean()),
                "p99_deg": float(torch.quantile(ang, 0.99)),
                "peor_deg": float(ang.max())}

    def marco(m):
        """Base tangente a la recta, para el modo residual."""
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x[m])).double()
        d0, t1, t2 = marco_tangente(t(al), t(be),
                                    torch.from_numpy(
                                        np.ascontiguousarray(np.deg2rad(inc[m]))),
                                    a.r_obs, xp=torch)
        return tuple(x.float().to(dev) for x in (d0, t1, t2))

    Mtr = marco(m_tr & esc) if a.residual else None
    Mva = marco(m_va & esc) if a.residual else None
    cielo = KerrSkyNet(a.ancho, a.capas_cielo, norm, residual=a.residual).to(dev)
    print(f"  parametros: {sum(q.numel() for q in cielo.parameters()):,}"
          + ("   (predice la DESVIACION respecto a la recta)" if a.residual else ""))
    h1 = entrenar(cielo, Xtr, Ytr, Xva, Yva, perd_cielo, met_cielo,
                  a.epocas, a.lote, a.lr, dev,
                  a.salida / "kerr_sky.pt", "cielo", Mtr, Mva)

    # ---------------------------------------------------------------- DISCO
    print("\n=== red del DISCO (cruces con el ecuador) ===")
    # Una fila por cada par (rayo, orden de cruce), EXISTA O NO el cruce.
    # La primera version solo metia los cruces reales, y al renderizar los
    # pixeles sin cruce caian en la extrapolacion de la red: devolvia radios
    # plausibles y el disco salia como un borron sobre toda la imagen. Hay que
    # enseñarle tambien donde NO hay disco.
    n_ray = al.size
    idx_ray = np.repeat(np.arange(n_ray), MAX_CRUCES)
    idx_k = np.tile(np.arange(MAX_CRUCES), n_ray)
    existe = (ncr[idx_ray] > idx_k)
    # los rayos capturados tampoco aportan fondo, pero SI cruzan el disco antes
    # de caer, asi que se quedan
    print(f"  filas: {idx_ray.size:,}  ({existe.sum():,} con cruce = "
          f"{existe.mean()*100:.1f}%)")

    def feats_disco(sel_ray, sel_k):
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x[sel_ray])).double()
        base = norm.features(t(al), t(be), t(sp), t(inc), t(dist),
                             xp=torch).float()
        ck = KerrDiskNet.codificar_k(torch.from_numpy(sel_k), xp=torch)
        return torch.cat([base, ck], -1).to(dev)

    def objetivo(sel_ray, sel_k):
        r = torch.from_numpy(cr[sel_ray, sel_k].astype(np.float32)).to(dev)
        ph = torch.from_numpy(cphi[sel_ray, sel_k].astype(np.float32)).to(dev)
        ex = torch.from_numpy((ncr[sel_ray] > sel_k).astype(np.float32)).to(dev)
        return [r, torch.stack([torch.cos(ph), torch.sin(ph)], -1), ex]

    sel_tr = m_tr[idx_ray]
    sel_va = m_va[idx_ray]
    Xd_tr = feats_disco(idx_ray[sel_tr], idx_k[sel_tr])
    Xd_va = feats_disco(idx_ray[sel_va], idx_k[sel_va])
    Yd_tr = objetivo(idx_ray[sel_tr], idx_k[sel_tr])
    Yd_va = objetivo(idx_ray[sel_va], idx_k[sel_va])
    print(f"  {Xd_tr.shape[0]:,} entrenamiento / {Xd_va.shape[0]:,} validacion")

    def perd_disco(pred, y):
        r, ang, logit = pred
        ex = y[2]
        # la existencia se aprende sobre TODAS las filas; el radio y el angulo
        # solo donde hay cruce, porque donde no lo hay el objetivo no significa
        # nada y entrenarlo ahi seria enseñar ruido
        l_ex = torch.nn.functional.binary_cross_entropy_with_logits(logit, ex)
        m = ex > 0.5
        if not bool(m.any()):
            return l_ex
        l_r = ((r[m] - y[0][m]) ** 2).mean() / 25.0
        l_a = (1.0 - (ang[m] * y[1][m]).sum(-1)).mean()
        return l_ex + l_r + l_a

    def met_disco(pred, y):
        r, ang, logit = pred
        ex = y[2] > 0.5
        acierto = ((logit > 0) == ex).float().mean()
        if not bool(ex.any()):
            return {"principal": 1.0 - float(acierto), "acierto": float(acierto)}
        er = (r[ex] - y[0][ex]).abs()
        ea = torch.rad2deg(torch.arccos((ang[ex] * y[1][ex]).sum(-1).clamp(-1, 1)))
        # la metrica principal mezcla las dos cosas: de nada sirve acertar el
        # radio si no se sabe si hay disco
        return {"principal": float(er.mean()) + 10.0 * (1.0 - float(acierto)),
                "acierto_existe": float(acierto), "err_r_M": float(er.mean()),
                "err_phi_deg": float(ea.mean())}

    disco = KerrDiskNet(a.ancho, a.capas_disco, norm).to(dev)
    print(f"  parametros: {sum(q.numel() for q in disco.parameters()):,}")
    h2 = entrenar(disco, Xd_tr, Yd_tr, Xd_va, Yd_va, perd_disco, met_disco,
                  a.epocas, a.lote, a.lr, dev,
                  a.salida / "kerr_disk.pt", "disco")

    # ------------------------------------------------------------- PRUEBA
    print("\n=== PRUEBA sobre configuraciones NUNCA vistas ===")
    cielo = KerrSkyNet.cargar(a.salida / "kerr_sky.pt", dev).to(dev)
    Xte = feats(m_te & esc)
    Yte = [torch.from_numpy(dirs[m_te & esc]).float().to(dev)]
    Mte = marco(m_te & esc) if a.residual else None
    with torch.no_grad():
        mc = met_cielo(cielo(Xte, Mte) if Mte else cielo(Xte), Yte)
    print(f"  cielo: error medio {mc['err_medio_deg']:.4f} grados, "
          f"p99 {mc['p99_deg']:.3f}, peor {mc['peor_deg']:.2f}")

    disco = KerrDiskNet.cargar(a.salida / "kerr_disk.pt", dev).to(dev)
    sel_te = m_te[idx_ray]
    with torch.no_grad():
        md = met_disco(disco(feats_disco(idx_ray[sel_te], idx_k[sel_te])),
                       objetivo(idx_ray[sel_te], idx_k[sel_te]))
    print(f"  disco: acierta si hay cruce el {md['acierto_existe']*100:.2f}% "
          f"de las veces; donde lo hay, error en r {md['err_r_M']:.4f} M "
          f"y en phi {md['err_phi_deg']:.3f} grados")

    (a.salida / "kerr_historial.json").write_text(json.dumps(
        {"config": {k: str(v) for k, v in vars(a).items()},
         "test_cielo": mc, "test_disco": md,
         "cielo": h1[-50:], "disco": h2[-50:]}, indent=2), encoding="utf-8")
    print(f"\nmodelos en {a.salida}")


if __name__ == "__main__":
    main()
