"""Analisis marino: Bio-ORACLE v3 (superficie y fondo) con el mismo flujo del atlas terrestre.

Este modulo solo descarga y prepara el dominio marino; las fases de ecorregiones, distancias, mapas y cobertura son las
de ejecutar_investigacion.py, con sus constantes de dominio reemplazadas (variables y transformacion).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
RECURSOS = Path(__import__('os').environ.get('RECURSOS_GENERALES', Path.home()/'Documents'/'Recursos generales'))  # datos de entrada fuera del repositorio

import numpy as np
import requests
import shapely
from scipy.io import netcdf_file
from scipy.spatial import cKDTree
from shapely.geometry import shape, mapping
from shapely.ops import transform as geom_transform

import ejecutar_investigacion as E
from descargar_chelsa import ROOT as CHELSA, sha256

RAIZ = (RECURSOS/r'Bio-ORACLE\v3')
OUT = Path(__file__).resolve().parents[1]/'resultados_marinos'/'20261011'
ERDDAP = 'https://erddap.bio-oracle.org/erddap/griddap'
ZEE_URL = ('https://geo.vliz.be/geoserver/MarineRegions/wfs?service=WFS&version=1.0.0&request=GetFeature'
           '&typeName=MarineRegions:eez&cql_filter=mrgid=8432&outputFormat=application/json')
ZEE = RAIZ/'zee_peru_mrgid8432.geojson'
CAJA = {'lat': (-20.3, -3.3), 'lon': (-84.8, -70.3)}  # ZEE del Peru (MRGID 8432: -20,21 a -3,39; -84,67 a -70,38) + margen
DECADAS = ('2000-01-01T00:00:00Z', '2010-01-01T00:00:00Z')
PERIODO = {'thetao': '2000_2019', 'so': '2000_2019', 'sws': '2000_2019', 'o2': '2000_2018', 'no3': '2000_2018', 'chl': '2000_2018'}
CAPAS = {'superficie': ('depthsurf', ('thetao_mean', 'thetao_min', 'thetao_max', 'so_mean', 'o2_mean', 'no3_mean', 'chl_mean', 'sws_mean')),
         'fondo': ('depthmean', ('thetao_mean', 'thetao_min', 'thetao_max', 'so_mean', 'o2_mean', 'no3_mean', 'sws_mean'))}
# Estratos de profundidad: el fondo se separa por la profundidad media del lecho (bathymetry_mean), que no entra al vector;
# la superficie tiene uno solo y se estratifica por ecorregion MINAM (fase ecorregiones). Limite superior de cada estrato en m.
ESTRATOS = {'superficie': [('MAR', 'Mar peruano', np.inf)],
            'fondo': [('PLATAFORMA', 'Plataforma (0–200 m)', 200), ('TALUD', 'Talud (200–1.000 m)', 1000),
                      ('PROFUNDO', 'Fondo profundo (> 1.000 m)', np.inf)]}
# La superficie se estratifica por las dos ecorregiones marinas del MINAM (2017) para el muestreo
# de las redes nulas y el reporte; el fondo sigue por profundidad. Codigo, nombre y valor del campo 'ecorregion' del DBF.
ECO_MINAM_MAR = [('MFH', 'Mar frío de la Corriente Peruana', 'Mar Frio de la Corriente Peruana (Humboldt)'),
                 ('MTR', 'Mar tropical', 'Mar Tropical')]
UNIDADES_FISICAS = {'thetao_mean': '°C', 'thetao_min': '°C', 'thetao_max': '°C', 'so_mean': 'PSU', 'o2_mean': 'mmol m-3',
                    'no3_mean': 'mmol m-3', 'chl_mean': 'mg m-3', 'sws_mean': 'm s-1'}


def dataset(base, sufijo):
    return f'{base}_baseline_{PERIODO[base]}_{sufijo}'


def descargar():
    """Ventanas de Bio-ORACLE v3 (dos decadas) y poligono de la ZEE; se conservan si coinciden con el manifiesto."""
    (RAIZ/'peru').mkdir(parents=True, exist_ok=True)
    manifest_path = RAIZ/'manifest_descarga.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {'archivos': {}}
    pedidos = {}
    for sufijo, variables in (v for v in CAPAS.values()):
        for base in dict.fromkeys(v.split('_')[0] for v in variables):
            names = [v for v in variables if v.startswith(base+'_')]
            (lat0, lat1), (lon0, lon1) = CAJA['lat'], CAJA['lon']
            query = ','.join(f'{n}[({DECADAS[0]}):({DECADAS[1]})][({lat0}):({lat1})][({lon0}):({lon1})]' for n in names)
            pedidos[f'peru/{dataset(base, sufijo)}.nc'] = f'{ERDDAP}/{dataset(base, sufijo)}.nc?{query}'
    (lat0, lat1), (lon0, lon1) = CAJA['lat'], CAJA['lon']
    pedidos['peru/terrain_characteristics.nc'] = (f'{ERDDAP}/terrain_characteristics.nc?bathymetry_mean'
                                                  f'[(1970-01-01T00:00:00Z)][({lat0}):({lat1})][({lon0}):({lon1})]')
    pedidos[ZEE.name] = ZEE_URL
    for rel, url in pedidos.items():
        path = RAIZ/rel
        known = manifest['archivos'].get(rel)
        if path.exists() and known and sha256(path) == known['sha256'] and known['url'] == url:
            continue
        started = time.perf_counter()
        response = requests.get(url, timeout=600)
        response.raise_for_status()
        temp = path.with_suffix(path.suffix+'.tmp')
        temp.write_bytes(response.content)
        temp.replace(path)
        manifest['archivos'][rel] = {'url': url, 'sha256': sha256(path), 'bytes': path.stat().st_size,
                                     'descargado': datetime.now().isoformat(timespec='seconds'),
                                     'segundos': time.perf_counter()-started}
        E.log(f'Descargado {rel}: {path.stat().st_size/2**20:.2f} MiB')
    manifest.update(fuente='Bio-ORACLE v3.0 (Assis et al. 2024) via ERDDAP; ZEE Marine Regions (Flanders Marine Institute) MRGID 8432',
                    licencia='Uso y redistribucion libres, no para fines legales (texto de ERDDAP y VLIZ); ver bio-oracle.org y marineregions.org',
                    caja=CAJA, decadas=DECADAS, estado='completo')
    E.save_json(manifest_path, manifest)
    return manifest


def transformar(array):
    """Unica transformacion: logaritmo de la clorofila (asimetrica, multiplicativa), como log(1 + P) en tierra."""
    result = np.array(array, dtype=np.float64)
    for k, v in enumerate(E.VARIABLES):
        if v == 'chl_mean':
            result[:, k] = np.log(result[:, k])
    return result


def configurar(capa):
    """Reemplaza las constantes de dominio del flujo terrestre; devuelve la carpeta de salida de la capa."""
    variables = CAPAS[capa][1]
    E.VARIABLES, E.P = variables, len(variables)
    E.transform_features = transformar
    E.TRANSFORMACIONES = 'log: chl_mean' if 'chl_mean' in variables else 'ninguna'
    E.UNITS = UNIDADES_FISICAS
    E.TXT = {'red': 'ANP marinas', 'superficie': 'superficie marina', 'territorio': 'Mar peruano'}
    anclas = OUT/capa/'modelo'/'anclas_d.json'
    if anclas.exists():
        a = json.loads(anclas.read_text(encoding='utf-8'))
        E.NOTA_D = (f"Referencias de d: celdas vecinas (~5,5 km) {E.es(a['vecinas_mediana'],2)}; d = 0,2 ≈ celdas a ~{E.es(a['km_d_0_2'])} km; "
                    f"dos lugares al azar del mar peruano {E.es(a['azar_mediana'],2)} (medianas).\n"
                    f"Bio-ORACLE v3 · {capa} · 2000–2019 · 0,05° · {len(variables)} predictores · Mahalanobis regularizada (cond ≤ 100). "
                    'Gris: sin datos marinos.')
    return OUT/capa


def leer_capas(capa):
    """Promedio de las dos decadas de cada variable en cuadricula norte-arriba; NaN si falta alguna decada."""
    sufijo, variables = CAPAS[capa]
    layers, lat_ref, lon_ref = {}, None, None
    for base in dict.fromkeys(v.split('_')[0] for v in variables):
        with netcdf_file(RAIZ/'peru'/f'{dataset(base, sufijo)}.nc', mmap=False) as f:
            lat, lon = f.variables['latitude'][:].astype(np.float64), f.variables['longitude'][:].astype(np.float64)
            if len(f.variables['time'][:]) != 2:
                raise ValueError(f'{base}: se esperaban dos decadas')
            if lat_ref is None:
                lat_ref, lon_ref = lat, lon
            if not (np.array_equal(lat, lat_ref) and np.array_equal(lon, lon_ref)):
                raise ValueError(f'{base}: cuadricula distinta')
            for name in (v for v in variables if v.startswith(base+'_')):
                var = f.variables[name]
                a = var[:].astype(np.float64)
                a[a == var._attributes['_FillValue']] = np.nan
                layers[name] = a.mean(axis=0)[::-1].reshape(-1)
    step = np.diff(lon_ref).mean()
    if not (np.allclose(np.diff(lon_ref), .05, atol=1e-4) and np.allclose(np.diff(lat_ref), .05, atol=1e-4)):
        raise ValueError(f'Paso de cuadricula inesperado: {step}')
    grid = {'width': len(lon_ref), 'height': len(lat_ref), 'crs': 'EPSG:4326',
            'transform': [.05, 0., round(lon_ref[0]-.025, 6), 0., -.05, round(lat_ref[-1]+.025, 6)]}
    return layers, grid


def preparar(out, capa):
    """Mismo contrato de archivos que prepare() terrestre: dominio ZEE sin tierra, asociaciones exactas por unidad."""
    manifest = json.loads((RAIZ/'manifest_descarga.json').read_text(encoding='utf-8'))
    sufijo, variables = CAPAS[capa]
    usados = [f'peru/{dataset(b, sufijo)}.nc' for b in dict.fromkeys(v.split('_')[0] for v in variables)]
    usados += ['peru/terrain_characteristics.nc', ZEE.name]
    hashes = {rel: sha256(RAIZ/rel) for rel in usados}
    if any(hashes[rel] != manifest['archivos'][rel]['sha256'] for rel in usados):
        raise ValueError('Descarga alterada; repetir --fase descargar')
    carto = CHELSA/'cartografia'
    payload = {'hashes': hashes, 'capa': capa, 'variables': list(variables), 'semilla': E.SEED, 'version_geometria': 3,
               'estratos': [(c, l) for c, _, l in ESTRATOS[capa][:-1]],
               'limite_terrestre': sha256(carto/'Peru_limite_departamental.geojson')}
    source_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    prep = out/'preparados'
    checkpoint = prep/'preparacion.json'
    if checkpoint.exists():
        record = json.loads(checkpoint.read_text(encoding='utf-8'))
        if record['fuentes_id'] != source_id:
            raise ValueError('El checkpoint tiene otras fuentes; elegir otra carpeta de salida')
        return record
    started = time.perf_counter()
    (prep/'asociaciones').mkdir(parents=True, exist_ok=True)
    repairs = []
    ea = lambda g, name: E.clean_geometry(geom_transform(E.FORWARD.transform, g), name, repairs)
    zee = ea(shape(json.loads(ZEE.read_text(encoding='utf-8'))['features'][0]['geometry']), 'ZEE')
    land = ea(shape(json.loads((carto/'Peru_limite_departamental.geojson').read_text(encoding='utf-8'))['features'][0]['geometry']), 'PERU')
    units = [(dict(f['properties']), ea(shape(f['geometry']), f['properties']['anp_id']))
             for name in ('ANP_nacionales', 'ANP_regionales', 'ANP_privadas')
             for f in json.loads((carto/(name+'.geojson')).read_text(encoding='utf-8'))['features']]
    # Agua protegida de las unidades con parte en la ZEE: incluye las franjas entre la costa nacional y la de la ZEE.
    marine_units = [(prop, g) for prop, g in units if g.intersection(zee).difference(land).area > 0]
    domain = E.clean_geometry(shapely.union_all([zee, *(g for _, g in marine_units)]).difference(land), 'MAR_PERUANO', repairs)
    layers, grid = leer_capas(capa)
    E.log(f'{capa}: fracciones marinas exactas')
    domain_flat, domain_area, closure = E.intersections(domain, grid)
    h, w = grid['height'], grid['width']
    aff, xx, yy, qrow = E.grid_geometry(grid)
    # Celdas costeras que Bio-ORACLE trata como tierra: valores de la celda de mar valida mas cercana (como la ecorregion
    # de las celdas sin poligono en tierra). Distancia local equirectangular en km.
    valid_all = np.all([np.isfinite(layers[v]) for v in variables], axis=0)
    need = domain_flat[~valid_all[domain_flat]]
    src = np.flatnonzero(valid_all)
    lon = aff.c+aff.a*(np.arange(w)+.5); lat = aff.f+aff.e*(np.arange(h)+.5)
    km = lambda f: np.column_stack((lon[f % w]*111.32*np.cos(np.deg2rad(lat[f//w])), lat[f//w]*110.57))
    fill_km, j = cKDTree(km(src)).query(km(need)) if len(need) else (np.empty(0), np.empty(0, int))
    with netcdf_file(RAIZ/'peru'/'terrain_characteristics.nc', mmap=False) as f:
        var = f.variables['bathymetry_mean']
        bathy = var[:].astype(np.float64)[0][::-1].reshape(-1)
        bathy[bathy == var._attributes['_FillValue']] = np.nan
    for v in variables:
        layers[v][need] = layers[v][src[j]]
    depth = -bathy
    depth[need] = depth[src[j]]  # misma celda de origen que el relleno de las variables
    filled = np.isin(domain_flat, need)
    flat, areas = domain_flat, domain_area
    np.savez_compressed(prep/'relleno_costero.npz', flat=need, origen=src[j], km=fill_km)
    array = np.column_stack([layers[v][flat] for v in variables]).astype(np.float32)
    ranges = {v: {'min': float(array[:, k].min()), 'max': float(array[:, k].max()), 'unidad': UNIDADES_FISICAS[v]}
              for k, v in enumerate(variables)}
    if 'chl_mean' in variables and array[:, variables.index('chl_mean')].min() <= 0:
        raise ValueError('Clorofila no positiva antes del logaritmo')
    if np.any(array[:, [variables.index(v) for v in ('so_mean', 'o2_mean', 'no3_mean', 'sws_mean')]] < 0):
        raise ValueError('Concentraciones o velocidades negativas')
    np.save(prep/'clima_fisico.npy', array)
    np.savez_compressed(prep/'dominio.npz', flat=flat, area=areas, todo_flat=domain_flat, todo_area=domain_area)
    rows, cols = flat//w, flat % w
    xy = np.column_stack(((xx[cols]+xx[cols+1])/2, (yy[rows]+yy[rows+1])/2))
    np.save(prep/'coordenadas_6933.npy', xy)
    lookup = np.full(h*w, -1, np.int32)
    lookup[flat] = np.arange(len(flat), dtype=np.int32)
    inventory, geometries, marine_ea = [], [], []
    for prop, geometry in marine_units:  # toda unidad con agua en el dominio se evalua
        uid = prop['anp_id']
        marine = E.clean_geometry(geometry.intersection(domain), uid+'_marina', repairs)
        water = geometry.difference(land).area
        ids, weights, stats = E.intersections(marine, grid)
        ix = lookup[ids]
        ok = ix >= 0
        ix, weights_valid = ix[ok], weights[ok]
        covered = float(weights_valid.sum())
        refill = float(weights[np.isin(ids, need)].sum())
        inventory.append({**prop, 'superficie_fuera_de_tierra_m2': water, 'superficie_marina_m2': float(weights.sum()),
                          'superficie_clima_m2': covered, 'superficie_rellenada_m2': refill, 'superficie_no_evaluable_m2': max(0., water-covered),
                          'cobertura_climatica': covered/water, 'n_celdas': len(ix),
                          'n_peso': covered**2/float(np.sum(weights_valid**2)) if covered else 0,
                          'estado': 'evaluable' if covered else 'sin_datos_marinos', 'error_cierre_m2': stats['error_m2']})
        np.savez_compressed(prep/'asociaciones'/(uid+'.npz'), index=ix, area=weights_valid, flat_todo=ids, area_todo=weights)
        marine_ea.append(marine)
        geometries.append({'type': 'Feature', 'properties': {'anp_id': uid},
                           'geometry': mapping(geom_transform(E.BACKWARD.transform, marine))})
        E.log(f'{uid}: {len(ix)} celdas marinas; {covered/1e6:.1f} de {water/1e6:.1f} km2 fuera de tierra')
    union = E.clean_geometry(shapely.union_all(marine_ea), 'UNION', repairs)
    union_flat, union_area, union_closure = E.intersections(union, grid)
    union_weights = np.zeros(len(flat))
    union_idx = lookup[union_flat]
    union_weights[union_idx[union_idx >= 0]] = union_area[union_idx >= 0]
    np.save(prep/'superficie_union.npy', np.minimum(union_weights, areas))
    E.save_json(prep/'inventario.json', inventory)
    E.write_csv(out/'inventario_unidades.csv', inventory)
    # nombres de archivo del contrato terrestre ("continental") para que las fases se reutilicen sin cambios
    E.save_json(prep/'unidades_continentales.geojson', {'type': 'FeatureCollection', 'features': geometries})
    E.save_json(prep/'peru_continental.geojson', {'type': 'Feature', 'properties': {},
                                                  'geometry': mapping(geom_transform(E.BACKWARD.transform, domain))})
    z = depth[flat]
    if np.isnan(z).any():
        raise ValueError('Celdas del dominio sin profundidad')
    estrato = np.searchsorted([l for _, _, l in ESTRATOS[capa][:-1]], z, side='left').astype(np.int8)  # limite incluido abajo
    np.savez_compressed(prep/'zonas.npz', ecorregion=estrato, departamento=np.zeros(len(flat), np.int8), profundidad_m=z.astype(np.float32))
    E.save_json(prep/'zonas.json', {'ecorregiones': [{'codigo': c, 'nombre': n, 'grupo': c.lower(), 'superficie_km2': float(areas[estrato == k].sum()/1e6)}
                                                     for k, (c, n, _) in enumerate(ESTRATOS[capa])],
                                    'departamentos': {'0': 'Mar peruano'},
                                    'criterio': 'Dominio: ZEE del Peru (MRGID 8432) y agua de las unidades con parte en ella, sin el limite terrestre. '
                                                'Estratos por profundidad media del lecho (Bio-ORACLE bathymetry_mean); celdas rellenadas: la de su origen.'})
    result = {'fuentes_id': source_id, 'fuentes': payload, 'grid': grid, 'n_celdas': len(flat), 'n_unidades': len(inventory),
              'n_evaluables': sum(u['estado'] == 'evaluable' for u in inventory), 'relleno_costero': {'celdas': len(need), 'km2': float(domain_area[filled].sum()/1e6),
                                  'km_mediana': float(np.median(fill_km)) if len(need) else 0., 'km_max': float(fill_km.max()) if len(need) else 0.},
              'superficie_dominio_m2': closure['area_m2'], 'superficie_clima_m2': float(areas.sum()),
              'cierre_dominio': closure, 'cierre_union': union_closure, 'reparaciones': repairs, 'rangos': ranges,
              'segundos': time.perf_counter()-started}
    E.save_json(checkpoint, result)
    E.log(f"{capa}: {len(flat)} celdas con datos; {areas.sum()/1e6:,.0f} de {closure['area_m2']/1e6:,.0f} km2 del dominio")
    return result


def anclas_d(out, cap=100):
    """Lectura empirica de d en la metrica marina: celdas vecinas, separacion con d mediana 0,2 y pares al azar."""
    target = out/'modelo'/'anclas_d.json'
    if target.exists():
        return json.loads(target.read_text(encoding='utf-8'))
    prep = out/'preparados'
    grid = json.loads((prep/'preparacion.json').read_text(encoding='utf-8'))['grid']
    w = grid['width']
    flat = np.load(prep/'dominio.npz')['flat']
    z = np.load(out/'modelo'/f'clima_transformado_cond{cap}.npy')
    xy = np.load(prep/'coordenadas_6933.npy')
    lookup = np.full(grid['height']*w, -1, np.int64)
    lookup[flat] = np.arange(len(flat))
    curve = []
    for k in range(1, 121):
        i = np.flatnonzero(flat % w+k < w)
        j = lookup[flat[i]+k]
        i, j = i[j >= 0], j[j >= 0]
        if not len(i):
            break
        d, km = np.linalg.norm(z[i]-z[j], axis=1), np.linalg.norm(xy[i]-xy[j], axis=1)/1000
        curve.append({'celdas': k, 'km_mediana': float(np.median(km)), 'd_mediana': float(np.median(d)), 'n_pares': len(i)})
    rng = np.random.default_rng(E.SEED)
    a, b = rng.integers(len(z), size=(2, 20000))
    cross = next((c for c in curve if c['d_mediana'] >= .2), curve[-1])
    result = {'vecinas_mediana': curve[0]['d_mediana'], 'km_d_0_2': cross['km_mediana'],
              'azar_mediana': float(np.median(np.linalg.norm(z[a]-z[b], axis=1))), 'curva_este_oeste': curve,
              'nota': 'Pares de celdas validas separadas k celdas en la misma fila; 20000 pares al azar del dominio.'}
    E.save_json(target, result)
    return result


FASES = ['preparar', 'ecorregiones', 'fichas', 'distancias', 'mapas', 'cobertura']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fase', choices=['descargar', *FASES, 'todo', 'prueba', 'factibilidad'], required=True)
    parser.add_argument('--capa', choices=[*CAPAS, 'ambas'], default='ambas')
    parser.add_argument('--cond', type=int, default=100)
    parser.add_argument('--redes', type=int, default=999)
    parser.add_argument('--out', type=Path, help='Fases prueba, cobertura y factibilidad: destino (<out>/<capa>/...); se leen las distancias de la fase todo')
    parser.add_argument('--diseno', choices=['N0', 'N2c'], default='N0', help='Diseno de las redes nulas; N1 no aplica en el mar')
    args = parser.parse_args()
    if args.fase == 'descargar':
        descargar()
        return
    if args.fase == 'prueba':
        for capa in (CAPAS if args.capa == 'ambas' else [args.capa]):
            E.recalcular_prueba(configurar(capa), args.cond, args.out/capa/'cobertura'/f'cond{args.cond}')
        return
    fases = FASES if args.fase == 'todo' else FASES[:1]+[args.fase]  # factibilidad: solo preparar (verifica)
    for capa in (CAPAS if args.capa == 'ambas' else [args.capa]):
        out = configurar(capa)
        preparar(out, capa)
        E.national_model(out, args.cond)
        anclas_d(out, args.cond)
        out = configurar(capa)  # NOTA_D con las anclas ya calculadas
        if 'ecorregiones' in fases and capa == 'superficie':
            E.ecorregiones(out, ECO_MINAM_MAR)
        if 'fichas' in fases:
            E.fichas(out, args.cond)
        if 'distancias' in fases:
            E.distances(out, args.cond)
        if 'mapas' in fases:
            E.mapas_distancia(out, args.cond)
        if args.fase == 'factibilidad':  # sin mascara natural: en el mar todo es "natural"
            E.factibilidad(out, args.out/capa)
        if 'cobertura' in fases:
            E.cobertura(out, args.cond, args.redes, diseno=args.diseno,
                        dst=args.out/capa/args.diseno if args.out else None)


if __name__ == '__main__':
    main()
