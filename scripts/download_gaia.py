"""Descarga un catalogo de estrellas de Gaia DR3 y lo deja listo para el render.

Hay que ejecutarlo en una maquina con acceso a internet. Guarda
data/raw/gaia_stars.npz con las estrellas ya convertidas a lo que necesita el
renderer: vector unitario de direccion, flujo relativo y color.

    pip install astroquery
    python scripts/download_gaia.py --gmax 10

Cuantas estrellas salen, aproximadamente:

    G < 10   ~350 mil      minutos
    G < 12   ~2-3 millones
    G < 13.5 ~10 millones  bastante rato

Se empieza en G = 3 porque Gaia casi no tiene estrellas mas brillantes: las muy
luminosas SATURAN el detector y no entran en el catalogo. Pedir franjas por
debajo de eso solo gasta viajes al servidor devolviendo cero filas.

La consulta se parte en franjas de magnitud porque un solo trabajo de millones
de filas suele agotar el tiempo del servidor.

ALTERNATIVA SIN ASTROQUERY
--------------------------
Si prefieres el archivo web (https://gea.esac.esa.int/archive/, pestana ADQL):

    SELECT ra, dec, phot_g_mean_mag, bp_rp
    FROM gaiadr3.gaia_source
    WHERE phot_g_mean_mag < 10

lo descargas en CSV y lo conviertes con:

    python scripts/download_gaia.py --from-csv ruta/al/resultado.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def radec_to_unit(ra_deg, dec_deg):
    """(RA, Dec) en grados -> vector unitario en coordenadas ecuatoriales ICRS."""
    ra, dec = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    cd = np.cos(dec)
    return np.stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)], -1)


def mag_to_flux(g):
    """Magnitud G -> flujo relativo.

    La escala de magnitudes es logaritmica e invertida: 5 magnitudes de
    diferencia son un factor 100 en brillo, y numeros MAS PEQUENOS son estrellas
    MAS brillantes. Se normaliza a G = 0.
    """
    return 10.0 ** (-0.4 * np.asarray(g, float))


def save(out: Path, ra, dec, gmag, bprp) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        direction=radec_to_unit(ra, dec).astype(np.float32),  # (N, 3) ICRS
        flux=mag_to_flux(gmag).astype(np.float32),
        gmag=gmag.astype(np.float32),
        bp_rp=bprp.astype(np.float32),      # indice de color: <0 azul, >1.5 rojo
    )
    print(f"\n{ra.size} estrellas -> {out}   ({out.stat().st_size / 1e6:.0f} MB)")


def from_csv(path: Path, out: Path) -> None:
    """Convierte el CSV que descarga el archivo web de Gaia."""
    import csv
    cols = {}
    with open(path, newline="") as f:
        # el CSV del archivo lleva lineas de comentario que empiezan por '#'
        rdr = csv.reader(l for l in f if not l.startswith("#"))
        head = next(rdr)
        idx = {n: head.index(n) for n in
               ("ra", "dec", "phot_g_mean_mag", "bp_rp") if n in head}
        faltan = {"ra", "dec", "phot_g_mean_mag"} - set(idx)
        if faltan:
            raise SystemExit(f"al CSV le faltan columnas: {sorted(faltan)}")
        rows = [r for r in rdr if r and r[idx["phot_g_mean_mag"]].strip()]
    get = lambda n, d: np.array(  # noqa: E731
        [float(r[idx[n]]) if n in idx and r[idx[n]].strip() else d for r in rows])
    save(out, get("ra", 0.0), get("dec", 0.0),
         get("phot_g_mean_mag", 20.0), get("bp_rp", 0.8))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gmax", type=float, default=10.0, help="magnitud G limite")
    p.add_argument("--gmin", type=float, default=3.0,
                   help="mas brillante que esto Gaia satura y no hay nada")
    p.add_argument("--step", type=float, default=1.0, help="ancho de franja")
    p.add_argument("--from-csv", type=Path, default=None,
                   help="convierte un CSV del archivo web en vez de descargar")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "gaia_stars.npz")
    a = p.parse_args()

    if a.from_csv is not None:
        from_csv(a.from_csv, a.out)
        return

    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT = -1                      # sin el, corta a 2000 filas

    edges = np.arange(a.gmin, a.gmax + a.step, a.step)
    ra, dec, gmag, bprp = [], [], [], []

    for lo, hi in zip(edges[:-1], edges[1:]):
        q = f"""
            SELECT ra, dec, phot_g_mean_mag, bp_rp
            FROM gaiadr3.gaia_source_lite
            WHERE phot_g_mean_mag >= {lo} AND phot_g_mean_mag < {hi}
              AND ra IS NOT NULL AND dec IS NOT NULL
        """
        print(f"  G en [{lo:5.1f}, {hi:5.1f}) ...", end="", flush=True)
        # async: los trabajos sincronos tienen limite de filas y de tiempo
        t = Gaia.launch_job_async(q).get_results()
        total = sum(x.size for x in gmag) + len(t)
        print(f" {len(t):>9} estrellas   (acumulado {total})")
        if len(t) == 0:
            continue
        ra.append(np.asarray(t["ra"], float))
        dec.append(np.asarray(t["dec"], float))
        gmag.append(np.asarray(t["phot_g_mean_mag"], float))
        bprp.append(np.asarray(np.ma.filled(t["bp_rp"], 0.8), float))

    if not ra:
        raise SystemExit("no llego ninguna estrella; revisa la conexion o el rango de G")
    save(a.out, np.concatenate(ra), np.concatenate(dec),
         np.concatenate(gmag), np.concatenate(bprp))


if __name__ == "__main__":
    main()
