"""Materializa recortes nativos de CHELSA mediante rangos HTTP, sin remuestreo."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
RECURSOS = Path(__import__('os').environ.get('RECURSOS_GENERALES', Path.home()/'Documents'/'Recursos generales'))  # datos de entrada fuera del repositorio

import numpy as np
import requests
import rasterio
from rasterio.windows import Window, from_bounds

VARIABLES = (
    "bio01", "bio02", "bio04", "bio05", "bio06", "bio12", "bio15",
    "hursmean", "vpdmean", "rsdsmean", "petmean",
)
PERIOD = "1981-2010"
ROOT = (RECURSOS/r'CHELSA\V2.1\1981-2010')
SOURCE = ROOT / "cartografia"


def same_grid(left, right):
    return (left["width"] == right["width"] and left["height"] == right["height"]
            and left["crs"] == right["crs"]
            and np.allclose(left["transform"], right["transform"], rtol=0, atol=1e-10))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def geom_bounds(geometry):
    points = []
    def visit(value):
        if isinstance(value[0], (float, int)):
            points.append(value)
        else:
            for part in value:
                visit(part)
    visit(geometry["coordinates"])
    return (min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points))


def recortar(url, target, bbox, periodo, common_grid, record, metadata_only=False):
    """Recorte nativo de bbox por rangos HTTP (sin remuestreo); llena record y devuelve la cuadricula comun."""
    started = time.perf_counter()
    response = requests.head(url, timeout=60, allow_redirects=True)
    response.raise_for_status()
    record.update(bytes_fuente=int(response.headers.get("Content-Length", 0)),
                  etag=response.headers.get("ETag"),
                  last_modified=response.headers.get("Last-Modified"))
    with rasterio.open(url) as src:
        tags = src.tags()
        if src.count != 1 or src.crs.to_epsg() != 4326 or tags.get("datetime") != periodo:
            raise ValueError(f"Producto o periodo inesperado: {url}")
        requested = from_bounds(*bbox, transform=src.transform)
        col, row = math.floor(requested.col_off), math.floor(requested.row_off)
        width = math.ceil(requested.col_off + requested.width) - col
        height = math.ceil(requested.row_off + requested.height) - row
        window = Window(col, row, width, height)
        transform = src.window_transform(window)
        grid = {"width": width, "height": height, "transform": list(transform)[:6], "crs": "EPSG:4326"}
        if common_grid is not None and not same_grid(grid, common_grid):
            raise ValueError(f"Cuadricula diferente para {url}; requiere revision explicita")
        record.update(grid=grid, dtype=src.dtypes[0], nodata=src.nodata,
                      scale=src.scales[0], offset=src.offsets[0], tags=tags,
                      ventana_fuente=[col, row, width, height], bloques_fuente=src.block_shapes,
                      ruta=str(target))
        if metadata_only:
            record["estado"] = "metadatos_verificados"
        else:
            profile = src.profile.copy()
            profile.update(width=width, height=height, transform=transform,
                           tiled=True, blockxsize=512, blockysize=512,
                           compress="deflate", zlevel=6)
            partial = target.with_suffix(".tif.part")
            minimum, maximum, valid_count = float("inf"), float("-inf"), 0
            with rasterio.open(partial, "w", **profile) as dst:
                dst.scales = src.scales
                dst.offsets = src.offsets
                dst.update_tags(**tags, recorte="Peru mas margen 0.1 grados",
                                source_url=url, resampling="none")
                for _, tile in dst.block_windows(1):
                    data = src.read(1, window=Window(col + tile.col_off, row + tile.row_off, tile.width, tile.height))
                    dst.write(data, 1, window=tile)
                    good = data != src.nodata
                    if good.any():
                        values = data[good].astype(np.float64) * src.scales[0] + src.offsets[0]
                        minimum = min(minimum, float(values.min()))
                        maximum = max(maximum, float(values.max()))
                        valid_count += int(good.sum())
            with rasterio.open(partial) as check:
                assert (check.width, check.height) == (width, height)
                assert check.scales == src.scales and check.offsets == src.offsets
            if valid_count == 0:
                raise ValueError(f"Capa sin datos validos: {url}")
            partial.replace(target)
            record.update(estado="completo", sha256=sha256(target), bytes_local=target.stat().st_size,
                          valores_fisicos_bbox={"min": minimum, "max": maximum}, n_validos_bbox=valid_count)
    record["segundos"] = time.perf_counter() - started
    return grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--variables", nargs="+", choices=VARIABLES, default=VARIABLES)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--pr-mensual", action="store_true",
                        help="12 climatologias mensuales de pr 1981-2010 (sesgo_precipitacion.py), en la misma cuadricula de las bio")
    args = parser.parse_args()
    root = args.root.resolve()
    raw = root / ("pr_mensual" if args.pr_mensual else "peru_nativo")
    raw.mkdir(parents=True, exist_ok=True)
    path = root / ("manifest_pr_mensual.json" if args.pr_mensual else "manifest_descarga.json")
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    country = json.loads((SOURCE / "Peru_limite_departamental.geojson").read_text(encoding="utf-8"))
    bounds = geom_bounds(country["features"][0]["geometry"])
    bbox = (bounds[0] - 0.1, bounds[1] - 0.1, bounds[2] + 0.1, bounds[3] + 0.1)
    manifest.update(periodo=PERIOD, producto="CHELSA V2.1 BIOCLIM y resumenes anuales BIOCLIM+",
                    bbox_solicitado=bbox, metodo="recorte nativo mediante rangos HTTP; sin remuestreo",
                    fuente_ficha="https://www.chelsa-climate.org/datasets/chelsa_bioclim")
    manifest.setdefault("capas", {})
    base = "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global"
    if args.pr_mensual:
        manifest.update(producto="CHELSA V2.1 climatologias mensuales de precipitacion (pr, kg m-2 mes-1)")
        items = [(f"pr_{m:02d}", f"{base}/climatologies/pr/{PERIOD}/CHELSA_pr_{m:02d}_{PERIOD}_V.2.1.tif",
                  raw / f"CHELSA_pr_{m:02d}_{PERIOD}_V.2.1_PERU.tif") for m in range(1, 13)]
        bio = json.loads((root / "manifest_descarga.json").read_text(encoding="utf-8"))
        common_grid = bio["capas"]["bio12"]["grid"]  # los meses deben caer en la cuadricula de los predictores
    else:
        items = [(v, f"{base}/bioclim/{v}/{PERIOD}/CHELSA_{v}_{PERIOD}_V.2.1.tif",
                  raw / f"CHELSA_{v}_{PERIOD}_V.2.1_PERU.tif") for v in args.variables]
        common_grid = None
    carto = root / "cartografia"
    carto.mkdir(exist_ok=True)
    for name in ("ANP_nacionales.geojson", "ANP_regionales.geojson", "ANP_privadas.geojson",
                 "Peru_limite_departamental.geojson", "cartografia_ANP_Peru.gpkg"):
        target = carto / name
        if not target.exists() or sha256(target) != sha256(SOURCE / name):
            shutil.copyfile(SOURCE / name, target)
    manifest["cartografia"] = {"fecha_catastro": "2026-10-09", "archivos":
                              {p.name: sha256(p) for p in carto.iterdir() if p.is_file()}}
    if args.pr_mensual:
        del manifest["cartografia"]
    atomic_json(path, manifest)
    with rasterio.Env(GDAL_CACHEMAX=32, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_TIMEOUT="60",
                      GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2",
                      GDAL_NUM_THREADS="1", VSI_CACHE=False):
        for variable, url, target in items:
            old = manifest["capas"].get(variable, {})
            if target.exists() and old.get("estado") == "completo" and old.get("sha256") == sha256(target):
                grid = old["grid"]
                if common_grid is not None and not same_grid(grid, common_grid):
                    raise ValueError(f"Cuadricula incompatible en archivo guardado: {variable}")
                common_grid = grid
                print(f"{variable}: conservado y verificado ({target.stat().st_size / 2**20:.1f} MiB)", flush=True)
                continue
            print(f"{variable}: comprobando metadatos y recortando...", flush=True)
            record = {"url": url, "estado": "iniciado", "fecha_utc": datetime.now(timezone.utc).isoformat()}
            manifest["capas"][variable] = record
            atomic_json(path, manifest)
            try:
                common_grid = recortar(url, target, bbox, PERIOD, common_grid, record, args.metadata_only)
                record["uso"] = "pr_mensual_correccion_F2" if args.pr_mensual else "predictor_nativo_30arcsec"
                atomic_json(path, manifest)
                print(f"{variable}: {record['estado']} en {record['segundos']:.1f} s; escala={record['scale']}; offset={record['offset']}", flush=True)
            except Exception as error:
                record.update(estado="error", error=str(error))
                atomic_json(path, manifest)
                raise
    keys = [i[0] for i in items] if args.pr_mensual else VARIABLES
    manifest["estado"] = "completo" if all(manifest["capas"].get(k, {}).get("estado") == "completo" for k in keys) else "parcial"
    manifest["bytes_locales"] = sum(c.get("bytes_local", 0) for c in manifest["capas"].values())
    atomic_json(path, manifest)
    print(f"Estado: {manifest['estado']}; capas: {len(manifest['capas'])}; datos locales: {manifest['bytes_locales']/2**20:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
