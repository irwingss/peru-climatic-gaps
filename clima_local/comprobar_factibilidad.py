"""Comprobacion con datos reales; no sustituye la calibracion del atlas."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.spatial import cKDTree
from shapely.geometry import shape, mapping

from descargar_chelsa import ROOT, VARIABLES, same_grid

FEATURES = VARIABLES
OUTPUT = Path(__file__).resolve().parent / "factibilidad_real.json"


def physical(values, variable, scale, offset):
    result = values.astype(np.float32) * scale + offset
    if variable == "bio04":
        result /= 100  # CHELSA guarda 100 veces la desviacion termica.
    if variable == "vpdmean":
        result /= 1000  # Pa a kPa.
    if variable == "petmean":
        result *= 12  # Media de los doce totales mensuales a total anual medio.
    return result


def transform_features(array):
    result = array.copy()
    for v in ("bio12", "vpdmean", "petmean"):
        k = FEATURES.index(v)
        result[:, k] = np.log1p(result[:, k])
    return result


def main():
    started = time.perf_counter()
    manifest = json.loads((ROOT / "manifest_descarga.json").read_text(encoding="utf-8"))
    if manifest.get("estado") != "completo":
        raise ValueError("Descarga incompleta")
    carto = ROOT / "cartografia"
    country = json.loads((carto / "Peru_limite_departamental.geojson").read_text(encoding="utf-8"))["features"][0]["geometry"]
    reference = manifest["capas"]["bio01"]["grid"]
    with rasterio.open(manifest["capas"]["bio01"]["ruta"]) as ds:
        transform, dimensions = ds.transform, (ds.height, ds.width)
    mask = rasterize([(country, 1)], out_shape=dimensions, transform=transform, all_touched=True, dtype="uint8").astype(bool)
    for variable in FEATURES:
        layer = manifest["capas"][variable]
        assert same_grid(reference, layer["grid"])
        with rasterio.open(layer["ruta"]) as ds:
            data = ds.read(1)
            mask &= data != ds.nodata
    rows, cols = np.nonzero(mask)
    array = np.empty((len(rows), len(FEATURES)), dtype=np.float32)
    ranges = {}
    for k, variable in enumerate(FEATURES):
        layer = manifest["capas"][variable]
        with rasterio.open(layer["ruta"]) as ds:
            array[:, k] = physical(ds.read(1)[rows, cols], variable, ds.scales[0], ds.offsets[0])
        ranges[variable] = [float(array[:, k].min()), float(array[:, k].max())]
    rng = np.random.default_rng(20261011)
    sample_index = rng.choice(len(rows), min(50000, len(rows)), replace=False)
    # La muestra de este benchmark es aleatoria; el atlas requiere muestra espacial equilibrada.
    transformed = transform_features(array)
    sample = transformed[sample_index].astype(np.float64)
    mu = sample.mean(axis=0)
    sd = sample.std(axis=0)
    standardized = (sample - mu) / sd
    correlation = standardized.T @ standardized / len(sample)
    eig, vec = np.linalg.eigh(correlation)
    condition_cap = 100
    required = max(0, (eig[-1] - condition_cap * eig[0]) / (condition_cap - 1 + eig[-1] - condition_cap * eig[0]))
    shrinkage = max(0.02, required)
    regularized_eig = (1 - shrinkage) * eig + shrinkage
    whitening = vec / np.sqrt(regularized_eig * len(FEATURES))
    query_index = rng.choice(len(rows), min(100000, len(rows)), replace=False)
    query = ((transformed[query_index] - mu) / sd) @ whitening
    units = []
    for file in ("ANP_nacionales", "ANP_regionales", "ANP_privadas"):
        units.extend(json.loads((carto / (file + ".geojson")).read_text(encoding="utf-8"))["features"])
    domain = shape(country)
    eligible = []
    for feature in units:
        geometry = shape(feature["geometry"]).intersection(domain)
        if not geometry.is_empty:
            eligible.append((feature, geometry))
    selected = []
    for modality in ("NACIONAL", "REGIONAL", "PRIVADA"):
        subset = [(f, g) for f, g in eligible if f["properties"]["modalidad"] == modality]
        selected.append(max(subset, key=lambda fg: fg[0]["properties"]["area_m2"]))
    selected.append(min(eligible, key=lambda fg: fg[0]["properties"]["area_m2"]))
    measurements = []
    for feature, geometry in selected:
        protected = rasterize([(mapping(geometry), 1)], out_shape=dimensions, transform=transform, all_touched=True, dtype="uint8")
        ref_index = np.flatnonzero(protected[rows, cols])
        references = ((transformed[ref_index] - mu) / sd) @ whitening
        built = time.perf_counter()
        tree = cKDTree(references, leafsize=32)
        build_seconds = time.perf_counter() - built
        before = time.perf_counter()
        distance, nearest = tree.query(query, k=1, eps=0, workers=1)
        seconds = time.perf_counter() - before
        # Comparacion directa independiente: 16 consultas y todas las referencias.
        brute = np.empty(16)
        for i in range(16):
            brute[i] = np.sqrt(np.min(np.sum((references - query[i]) ** 2, axis=1)))
        error = float(np.max(np.abs(brute - distance[:16])))
        if error > 1e-10:
            raise ValueError(f"Busqueda inexacta: {error}")
        measurement = {"anp_id": feature["properties"]["anp_id"], "nombre": feature["properties"]["nombre"],
                       "modalidad": feature["properties"]["modalidad"], "n_referencias": len(ref_index),
                       "n_consultas": len(query), "segundos_arbol": build_seconds, "segundos_consulta": seconds,
                       "segundos_nacional_extrapolado": seconds / len(query) * len(rows),
                       "error_max_busqueda_directa": error,
                       "distancias_p05_p50_p95": np.quantile(distance, [.05, .5, .95]).tolist()}
        measurements.append(measurement)
        print(json.dumps(measurement, ensure_ascii=False), flush=True)
    result = {"fecha": "2026-10-11", "tipo": "BENCHMARK CON CAPAS REALES; NO ATLAS NI VALIDACION ESTADISTICA",
              "n_celdas_rasterizadas_con_vector_completo": len(rows), "n_unidades_inventario": len(units),
              "n_unidades_con_interseccion_continental": len(eligible), "dimensiones_bbox": dimensions,
              "variables_distancia": FEATURES, "rangos_decodificados_peru": ranges,
              "advertencia_rsdsmean": "Escala absoluta de radiacion pendiente de contraste; la metrica estandarizada es invariante ante cambios lineales positivos de unidades.",
              "memoria_matriz_float32_mib": array.nbytes / 2**20, "shrinkage_benchmark": shrinkage,
              "condicion_covarianza_regularizada": float(regularized_eig[-1] / regularized_eig[0]),
              "mediciones": measurements, "segundos_total": time.perf_counter() - started,
              "limitaciones": ["Rasterizacion all_touched para medir rendimiento; no superficies legales exactas.",
                               "Muestra aleatoria en este benchmark; no calibracion espacial del atlas.",
                               "Tiempos nacionales extrapolan consultas; no incluyen atlas, graficos o intersecciones.",
                               "No se han calibrado umbrales binarios ni calculado tasas de falsos analogos."]}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Guardado: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
