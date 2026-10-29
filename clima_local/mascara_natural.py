"""Mascara de entornos naturales en la cuadricula de 30" de CHELSA: GHS-SMOD 2020 y MapBiomas Peru (coleccion 3, 2020).

--fase descargar  fraccion natural de MapBiomas agregada en Earth Engine a 30" (bloques computePixels; nunca 30 m)
--fase mascara    urbano GHSL + fraccion natural -> natural_30s.tif, calibracion con la red real y resumen
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
RECURSOS = Path(__import__('os').environ.get('RECURSOS_GENERALES', Path.home()/'Documents'/'Recursos generales'))  # datos de entrada fuera del repositorio

import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt

import ejecutar_investigacion as E
from descargar_chelsa import same_grid, sha256

PROYECTO_EE = 'ee-irwingssaldu'
ASSET = 'projects/mapbiomas-public/assets/peru/collection3/mapbiomas_peru_collection3_integration_v1'
ANO = 2020  # el mismo ano de GHS-SMOD
# Leyenda oficial de la coleccion 3 (Leyenda_MapBiomasPeru_3-Leyenda-CortaENES.pdf): columna Natural/Antropico.
NATURAL = (3, 4, 5, 6, 11, 12, 13, 23, 29, 32, 33, 34, 61, 66, 68, 70)
ANTROPICO = (9, 14, 15, 18, 21, 24, 25, 30, 31, 35, 40, 72)  # 27 "no observado" queda fuera de ambos
RAIZ = (RECURSOS/r'MapBiomas_Peru')
FRACCION = RAIZ/f'mapbiomas_peru_c3_{ANO}_fraccion_natural_30s.tif'
GHSL = (RECURSOS/r'GHSL_SMOD_E2020_R2023A\GHS_SMOD_E2020_GLOBE_R2023A_4326_30ss_V2_0.tif')
URBANO = 21  # clases >= 21: suburbano, semidenso, denso y centro urbano (misma regla que zonas_prioritarias.urbano)
BASE = Path(__file__).resolve().parents[1]/'resultados_v7'/'02_sesgo_precipitacion'/'original'  # CHELSA original
OUT = Path(__file__).resolve().parents[1]/'resultados_v7'/'03_mascara_natural'
UMBRAL_PIXEL = .5   # un pixel es natural si su fraccion natural es >= 0.5 y no es urbano
BLOQUE = 256


def descargar():
    """Fraccion natural (sobre los pixeles de 30 m observados) y fraccion observada, promedio exacto a 30"."""
    import ee
    ee.Initialize(project=PROYECTO_EE)
    grid = json.loads((BASE/'preparados'/'preparacion.json').read_text(encoding='utf-8'))['grid']
    a, _, x0, _, e, y0 = grid['transform']
    h, w = grid['height'], grid['width']
    img = ee.Image(ASSET).select(f'classification_{ANO}')
    clases = list(NATURAL+ANTROPICO)
    natural = img.remap(clases, [1]*len(NATURAL)+[0]*len(ANTROPICO)).toFloat()  # sin mapear (27, fuera) -> enmascarado
    observado = img.remap(clases, [1]*len(clases)).unmask(0).toFloat()
    # reduceResolution con promedio; ~31 x 31 pixeles de 30 m por celda, bien por debajo de maxPixels
    expr = ee.Image.cat(natural.rename('natural'), observado.rename('observado')).reduceResolution(ee.Reducer.mean(), maxPixels=4096)
    out = np.full((2, h, w), np.nan, np.float32)

    def bloque(rc):
        r0, c0 = rc
        hh, ww = min(BLOQUE, h-r0), min(BLOQUE, w-c0)
        for intento in range(4):
            try:
                raw = ee.data.computePixels({'expression': expr, 'fileFormat': 'NUMPY_NDARRAY', 'grid': {
                    'dimensions': {'width': ww, 'height': hh}, 'crsCode': 'EPSG:4326',
                    'affineTransform': {'scaleX': a, 'shearX': 0, 'translateX': x0+c0*a, 'shearY': 0, 'scaleY': e, 'translateY': y0+r0*e}}})
                break
            except Exception as error:  # cuota o tiempo de Earth Engine: se reintenta
                E.log(f'Bloque {r0},{c0}: {str(error)[:80]}; reintento {intento+1}')
                time.sleep(20)
        else:
            raise RuntimeError(f'Bloque {r0},{c0} sin respuesta')
        out[0, r0:r0+hh, c0:c0+ww] = raw['natural']
        out[1, r0:r0+hh, c0:c0+ww] = raw['observado']
        E.log(f'Bloque {r0},{c0} listo')

    started = time.perf_counter()
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(bloque, [(r, c) for r in range(0, h, BLOQUE) for c in range(0, w, BLOQUE)]))
    RAIZ.mkdir(parents=True, exist_ok=True)
    profile = dict(driver='GTiff', width=w, height=h, count=2, dtype='float32', nodata=np.nan, crs=grid['crs'],
                   transform=Affine(*grid['transform']), tiled=True, blockxsize=256, blockysize=256, compress='deflate', predictor=3)
    with rasterio.open(FRACCION, 'w', **profile) as ds:
        ds.write(out)
        ds.set_band_description(1, 'fraccion natural de los pixeles 30 m observados')
        ds.set_band_description(2, 'fraccion de pixeles 30 m con clase observada')
    E.save_json(RAIZ/'manifest_mapbiomas.json', {
        'asset': ASSET, 'banda': f'classification_{ANO}', 'proyecto_ee': PROYECTO_EE, 'natural': NATURAL, 'antropico': ANTROPICO,
        'no_observado': [27], 'agregacion': 'reduceResolution(mean) en Earth Engine a la cuadricula CHELSA 30"; computePixels por bloques de 256',
        'grid': grid, 'archivo': FRACCION.name, 'sha256': sha256(FRACCION), 'bytes': FRACCION.stat().st_size,
        'segundos': time.perf_counter()-started, 'descargado': datetime.now().isoformat(timespec='seconds'),
        'leyenda': 'https://peru.mapbiomas.org/wp-content/uploads/sites/6/2026/08/Leyenda_MapBiomasPeru_3-Leyenda-CortaENES.pdf',
        'cita': 'Proyecto MapBiomas Peru, Coleccion 3 de la serie anual de mapas de cobertura y uso del suelo del Peru'})
    E.log(f'MapBiomas a 30": {time.perf_counter()-started:.0f} s')


def urbano(grid, aff):
    """Celdas urbanas de GHS-SMOD (vecino mas cercano a nuestra cuadricula); copia de zonas_prioritarias.urbano()."""
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import from_bounds
    h, w = grid['height'], grid['width']
    with rasterio.open(GHSL) as ds:
        x0, y1 = aff.c, aff.f
        x1, y0 = x0+w*aff.a, y1+h*aff.e
        win = from_bounds(x0-.1, y0-.1, x1+.1, y1+.1, ds.transform).round_offsets().round_lengths()
        dst = np.full((h, w), -200, np.int16)
        reproject(ds.read(1, window=win), dst, src_transform=ds.window_transform(win), src_crs=ds.crs,
                  dst_transform=aff, dst_crs=ds.crs, resampling=Resampling.nearest)
    return dst >= URBANO


def mascara():
    started = time.perf_counter()
    prep = BASE/'preparados'
    grid = json.loads((prep/'preparacion.json').read_text(encoding='utf-8'))['grid']
    h, w = grid['height'], grid['width']
    aff = Affine(*grid['transform'])
    dom = np.load(prep/'dominio.npz')
    flat, area = dom['flat'], dom['area']
    manifest = json.loads((RAIZ/'manifest_mapbiomas.json').read_text(encoding='utf-8'))
    if sha256(FRACCION) != manifest['sha256'] or not same_grid(grid, manifest['grid']):
        raise ValueError('Fraccion MapBiomas alterada o de otra cuadricula')
    with rasterio.open(FRACCION) as ds:
        frac, obs = ds.read(1), ds.read(2)
    urb = urbano(grid, aff)
    # Celdas del dominio sin ningun pixel observado (costa, bordes): fraccion de la celda observada mas cercana.
    sin = ~(obs > 0) | ~np.isfinite(frac)
    nearest = distance_transform_edt(sin, return_distances=False, return_indices=True)
    relleno = frac[nearest[0], nearest[1]]
    fdom = relleno.reshape(-1)[flat]
    rellenas = sin.reshape(-1)[flat]
    udom = urb.reshape(-1)[flat]
    natural = (~udom) & (fdom >= UMBRAL_PIXEL)
    OUT.mkdir(parents=True, exist_ok=True)
    profile = dict(driver='GTiff', width=w, height=h, count=1, crs=grid['crs'], transform=aff, tiled=True,
                   blockxsize=256, blockysize=256, compress='deflate')
    band = np.full(h*w, 255, np.uint8); band[flat] = natural
    with rasterio.open(OUT/'natural_30s.tif', 'w', **profile, dtype='uint8', nodata=255) as ds:
        ds.write(band.reshape(h, w), 1)
        ds.update_tags(valores='1 natural; 0 urbano GHSL o fraccion natural < 0.5; 255 fuera del dominio')
    band = np.full(h*w, np.nan, np.float32); band[flat] = fdom
    with rasterio.open(OUT/'fraccion_natural_30s.tif', 'w', **profile, dtype='float32', nodata=np.nan, predictor=3) as ds:
        ds.write(band.reshape(h, w), 1)
    np.savez_compressed(OUT/'natural_dominio.npz', natural=natural, fraccion=fdom.astype(np.float32), urbano=udom, rellenada=rellenas)
    # Calibracion: fraccion de la superficie de cada area (pesos a_jc) en pixeles naturales.
    lookup_nat = natural.astype(np.float64)
    inventory = json.loads((prep/'inventario.json').read_text(encoding='utf-8'))
    filas = []
    for u in inventory:
        if u['estado'] != 'evaluable':
            continue
        z = np.load(prep/'asociaciones'/f"{u['anp_id']}.npz")
        ix, wa = z['index'], z['area']
        filas.append({'anp_id': u['anp_id'], 'nombre': u['nombre'], 'modalidad': u['modalidad'],
                      'superficie_km2': float(wa.sum()/1e6), 'n_celdas': len(ix),
                      'fraccion_pixeles_naturales': float(lookup_nat[ix].mean()),
                      'fraccion_superficie_natural': float(wa@lookup_nat[ix]/wa.sum()),
                      'fraccion_natural_media': float(wa@fdom[ix]/wa.sum()),
                      'fraccion_urbana': float(wa@udom[ix]/wa.sum())})
    E.write_csv(OUT/'calibracion_por_area.csv', filas)
    # La misma medida que E.colocar() aplica a las huellas de las redes nulas: pixeles naturales ponderados por a_jc.
    pix = np.array([f['fraccion_superficie_natural'] for f in filas])
    sup = np.array([f['superficie_km2'] for f in filas])
    union = np.load(prep/'superficie_union.npy')
    cumplen = {str(t): {'areas_pct': float(100*(pix >= t).mean()), 'superficie_pct': float(100*sup[pix >= t].sum()/sup.sum())}
               for t in (.95, .9, .85, .8, .75, .7, .6, .5)}
    umbral = max((t for t in (.95, .9, .85, .8, .75, .7, .6, .5) if (pix >= t).mean() >= .95), default=None)
    # Ningun umbral unico cumple la regla (areas diminutas en paisajes agricolas: 1 a 5 pixeles de 30"); ajuste: umbral
    # por area = min(0.9, su propia fraccion natural, truncada a 2 decimales). Toda area real lo cumple por definicion.
    por_area = {f['anp_id']: (umbral if umbral is not None else float(min(.9, np.floor(f['fraccion_superficie_natural']*100)/100)))
                for f in filas}
    for f in filas:
        f['umbral_F4'] = por_area[f['anp_id']]
    E.write_csv(OUT/'calibracion_por_area.csv', filas)
    eco, nombres = E.estratos_muestreo(prep)
    por_eco = [{'ecorregion': n, 'superficie_km2': float(area[eco == k].sum()/1e6),
                'natural_km2': float(area[(eco == k) & natural].sum()/1e6),
                'natural_pct': float(100*area[(eco == k) & natural].sum()/area[eco == k].sum()),
                'urbano_km2': float(area[(eco == k) & udom].sum()/1e6),
                'natural_fuera_red_km2': float((area-union)[(eco == k) & natural].sum()/1e6)} for k, n in enumerate(nombres)]
    E.write_csv(OUT/'natural_por_ecorregion.csv', por_eco)
    resumen = {'fuente_natural': manifest['asset'], 'ano': ANO, 'natural': NATURAL, 'antropico': ANTROPICO,
               'urbano': f'GHS-SMOD E2020 R2023A, clases >= {URBANO}', 'umbral_pixel': UMBRAL_PIXEL,
               'dominio_km2': float(area.sum()/1e6), 'natural_km2': float(area[natural].sum()/1e6),
               'natural_pct': float(100*area[natural].sum()/area.sum()), 'urbano_km2': float(area[udom].sum()/1e6),
               'antropico_no_urbano_km2': float(area[~udom & ~natural].sum()/1e6),
               'celdas_rellenadas': int(rellenas.sum()), 'superficie_rellenada_km2': float(area[rellenas].sum()/1e6),
               'red_real': {'areas': len(filas), 'fraccion_pixeles_naturales_total': float(lookup_nat[union > 0].mean()),
                            'superficie_natural_pct': float(100*(union*natural).sum()/union.sum()),
                            'cuantiles_por_area': dict(zip(('p05', 'p10', 'p25', 'p50'), map(float, np.quantile(pix, [.05, .1, .25, .5])))),
                            'cumplen_umbral': cumplen},
               'umbral_unico_que_cumple_95pct_areas': umbral,
               'umbral_aceptacion_F4': umbral if umbral is not None else por_area,
               'areas_con_umbral_menor_0_9': int(sum(v < .9 for v in por_area.values())),
               'superficie_km2_areas_con_umbral_menor_0_9': float(sup[pix < .9].sum()),
               'regla': 'Umbral unico = el mayor de 0.95, 0.9, ... que cumple el 95 % de las areas reales (pixeles naturales '
                        'ponderados por a_jc). Si ninguno cumple: umbral por area = min(0.9, fraccion natural de esa area).',
               'segundos': time.perf_counter()-started}
    E.save_json(OUT/'resumen.json', resumen)
    E.log(json.dumps({k: resumen[k] for k in ('natural_pct', 'urbano_km2', 'areas_con_umbral_menor_0_9')}))
    return resumen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fase', choices=['descargar', 'mascara'], required=True)
    args = parser.parse_args()
    descargar() if args.fase == 'descargar' else mascara()
