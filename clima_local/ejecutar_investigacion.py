"""Investigacion local reproducible: geometria exacta, metrica, distancias exactas y redes nulas."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path
RECURSOS = Path(__import__('os').environ.get('RECURSOS_GENERALES', Path.home()/'Documents'/'Recursos generales'))  # datos de entrada fuera del repositorio

os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('GDAL_NUM_THREADS', '1')
os.environ.setdefault('PYTHONUTF8', '1')

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.windows import Window, transform as window_transform
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import shape, mapping
from shapely.ops import transform as geom_transform
from pyproj import Transformer

from descargar_chelsa import ROOT, VARIABLES, sha256, same_grid
from comprobar_factibilidad import physical, transform_features

OUT = Path(__file__).resolve().parents[1] / 'resultados_climaticos' / '20261011'
SEED = 20261011
AREA_RTOL = 1e-6
AREA_ATOL = 0.05  # m2; tolerancia absoluta para predios diminutos.
P = len(VARIABLES)
FORWARD = Transformer.from_crs(4326, 6933, always_xy=True)
BACKWARD = Transformer.from_crs(6933, 4326, always_xy=True)
# constantes de dominio; ejecutar_marino.py las reemplaza para reutilizar las mismas fases en el mar.
TRANSFORMACIONES = 'log1p: bio12, petmean, vpdmean'
# unidades con < 90 % de su superficie legal en tierra; sus figuras avisan que muestran solo la tierra.
_MAR = 'Su parte marina tiene análisis propio (atlas marino).'
AGUA = {'NAC_RN03': _MAR, 'NAC_RN13': _MAR, 'NAC_RN15': _MAR, 'NAC_RN18': _MAR,
        'NAC_RN02': 'El lago queda fuera de ambos análisis.', 'NAC_RN05': 'El lago queda fuera de ambos análisis.',
        'REG_ACR03': 'El agua de la albúfera queda fuera de ambos análisis.'}
TXT = {'red': 'ANP + ACR + ACP', 'superficie': 'superficie continental', 'territorio': 'Perú', 'agua': AGUA}


def log(message):
    print(f'[{datetime.now().isoformat(timespec="seconds")}] {message}', flush=True)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def write_csv(path, records):
    records = list(records)
    if not records:
        return
    keys = list(dict.fromkeys(k for record in records for k in record))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def weighted_quantile(values, weights, probs):
    order = np.argsort(values, kind='stable')
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    keep = weights > 0
    values, weights = values[keep], weights[keep]
    cumulative = np.cumsum(weights)
    return values[np.minimum(np.searchsorted(cumulative, np.asarray(probs) * cumulative[-1]), len(values)-1)]


def clean_geometry(geometry, name, repairs):
    if not geometry.is_valid:
        old = geometry.area
        geometry = shapely.make_valid(geometry)
        repairs.append({'id': name, 'accion': 'make_valid', 'area_antes': old, 'area_despues': geometry.area})
    if geometry.geom_type == 'GeometryCollection':
        geometry = shapely.union_all([g for g in geometry.geoms if g.geom_type in ('Polygon', 'MultiPolygon')])
    return geometry


def grid_geometry(grid):
    aff = Affine(*grid['transform'])
    h, w = grid['height'], grid['width']
    lon_edges = aff.c + np.arange(w + 1) * aff.a
    lat_edges = aff.f + np.arange(h + 1) * aff.e
    xx, _ = FORWARD.transform(lon_edges, np.zeros(w + 1))
    _, yy = FORWARD.transform(np.zeros(h + 1), lat_edges)
    qrow = (xx[1]-xx[0]) * (yy[:-1]-yy[1:])
    return aff, np.asarray(xx), np.asarray(yy), qrow


def intersections(geometry_ea, grid):
    """Interiores por fila; interseccion exacta de la banda de frontera en EPSG:6933."""
    if geometry_ea.is_empty or geometry_ea.area <= 0:
        return np.empty(0, np.int32), np.empty(0, np.float64), {'area_m2': 0.0, 'error_m2': 0.0}
    aff, xx, yy, qrow = grid_geometry(grid)
    geo = geom_transform(BACKWARD.transform, geometry_ea)
    west, south, east, north = geo.bounds
    w, h = grid['width'], grid['height']
    c0 = max(0, int(np.floor((west-aff.c)/aff.a))-2)
    c1 = min(w, int(np.ceil((east-aff.c)/aff.a))+2)
    r0 = max(0, int(np.floor((north-aff.f)/aff.e))-2)
    r1 = min(h, int(np.ceil((south-aff.f)/aff.e))+2)
    win = Window(c0, r0, c1-c0, r1-r0)
    local_aff = window_transform(win, aff)
    dimensions = (r1-r0, c1-c0)
    touched = rasterize([(mapping(geo), 1)], out_shape=dimensions, transform=local_aff, all_touched=True, dtype='uint8').astype(bool)
    boundary = rasterize([(mapping(geo.boundary), 1)], out_shape=dimensions, transform=local_aff, all_touched=True, dtype='uint8').astype(bool)
    # Una banda adicional cubre diferencias subpixel entre segmentos proyectados y geograficos.
    boundary = binary_dilation(boundary, iterations=2)
    rr, cc = np.nonzero(touched | boundary)
    is_boundary = boundary[rr, cc]
    rr, cc = rr+r0, cc+c0
    areas = qrow[rr].copy()
    k = np.flatnonzero(is_boundary)
    # Recorte exacto por teselas de 64x64 celdas: cada celda solo intersecta la pieza local del poligono.
    tile = (rr[k]//64)*(w//64+1) + cc[k]//64
    order_tiles = np.argsort(tile, kind='stable')
    k, tile = k[order_tiles], tile[order_tiles]
    starts = np.flatnonzero(np.r_[True, np.diff(tile) != 0])
    for start, end in zip(starts, np.r_[starts[1:], len(k)]):
        index = k[start:end]
        r, c = rr[index], cc[index]
        piece = shapely.intersection(geometry_ea, shapely.box(xx[c.min()], yy[r.max()+1], xx[c.max()+1], yy[r.min()]))
        cells = shapely.box(xx[c], yy[r+1], xx[c+1], yy[r])
        areas[index] = shapely.area(shapely.intersection(cells, piece))
    keep = areas > 0
    flat = (rr[keep]*w+cc[keep]).astype(np.int32)
    areas = areas[keep]
    error = float(areas.sum()-geometry_ea.area)
    tolerance = max(AREA_ATOL, AREA_RTOL * geometry_ea.area)
    if abs(error) > tolerance:
        raise ValueError(f'No cierra superficie: error {error} m2, tolerancia {tolerance} m2')
    order = np.argsort(flat)
    return flat[order], areas[order], {'area_m2': float(geometry_ea.area), 'error_m2': error, 'tolerancia_m2': tolerance, 'celdas': len(flat)}


def load_sources(root):
    manifest = json.loads((root/'manifest_descarga.json').read_text(encoding='utf-8'))
    if manifest.get('estado') != 'completo' or manifest['periodo'] != '1981-2010':
        raise ValueError('Contrato temporal/descarga incompleto')
    hashes = {}
    first = manifest['capas']['bio01']['grid']
    for variable in VARIABLES:
        layer = manifest['capas'][variable]
        path = root/'peru_nativo'/Path(layer['ruta']).name
        actual = sha256(path)
        if actual != layer['sha256'] or not same_grid(first, layer['grid']):
            raise ValueError(f'Hash o cuadricula incompatible: {variable}')
        hashes[variable] = actual
    for filename, expected in manifest['cartografia']['archivos'].items():
        actual = sha256(root/'cartografia'/filename)
        if actual != expected:
            raise ValueError(f'Cartografia alterada: {filename}')
        hashes[filename] = actual
    payload = {'hashes': hashes, 'variables': list(VARIABLES), 'semilla': SEED, 'version_geometria': 1}
    source_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return manifest, first, payload, source_id


def prepare(root, out):
    started = time.perf_counter()
    manifest, grid, payload, source_id = load_sources(root)
    checkpoint = out/'preparados'/'preparacion.json'
    if checkpoint.exists():
        record = json.loads(checkpoint.read_text(encoding='utf-8'))
        if record['fuentes_id'] != source_id:
            raise ValueError('El checkpoint tiene otras fuentes; elegir otra carpeta de salida')
        log('Preparacion verificada; se reutiliza')
        return record
    prep = out/'preparados'
    (prep/'asociaciones').mkdir(parents=True, exist_ok=True)
    repairs = []
    country = json.loads((root/'cartografia'/'Peru_limite_departamental.geojson').read_text(encoding='utf-8'))
    domain = clean_geometry(geom_transform(FORWARD.transform, shape(country['features'][0]['geometry'])), 'PERU', repairs)
    log('Calculando fracciones continentales exactas')
    domain_flat, domain_area, closure = intersections(domain, grid)
    h, w = grid['height'], grid['width']
    all_valid = np.ones(h*w, bool)
    layers = {}
    ranges = {}
    for variable in VARIABLES:
        layer = manifest['capas'][variable]
        path = root/'peru_nativo'/Path(layer['ruta']).name
        with rasterio.open(path) as ds:
            actual_grid = {'width': ds.width, 'height': ds.height, 'crs': str(ds.crs), 'transform': list(ds.transform)[:6]}
            if not same_grid(grid, actual_grid) or ds.tags().get('datetime') != '1981-2010':
                raise ValueError(f'Contrato real incompatible: {variable}')
            raw = ds.read(1).reshape(-1)
            valid = (ds.read_masks(1).reshape(-1)>0) & (raw != ds.nodata)
            values = physical(raw, variable, ds.scales[0], ds.offsets[0])
            valid &= np.isfinite(values)
            all_valid &= valid
            layers[variable] = values
            ranges[variable] = {'scale': ds.scales[0], 'offset': ds.offsets[0], 'tags': ds.tags()}
    valid_domain = all_valid[domain_flat]
    flat = domain_flat[valid_domain]
    areas = domain_area[valid_domain]
    array = np.column_stack([layers[v][flat] for v in VARIABLES]).astype(np.float32)
    for k, variable in enumerate(VARIABLES):
        ranges[variable].update(min=float(array[:,k].min()), max=float(array[:,k].max()))
    if np.any(array[:,[VARIABLES.index(v) for v in ('bio12','petmean','vpdmean')]] < 0):
        raise ValueError('Flujos negativos antes de logaritmos')
    if np.any((array[:,7]<0)|(array[:,7]>100)):
        raise ValueError('Humedad fuera de 0-100')
    np.save(prep/'clima_fisico.npy', array)
    np.savez_compressed(prep/'dominio.npz', flat=flat, area=areas, todo_flat=domain_flat, todo_area=domain_area)
    aff, xx, yy, qrow = grid_geometry(grid)
    rows, cols = flat//w, flat%w
    xy = np.column_stack(((xx[cols]+xx[cols+1])/2, (yy[rows]+yy[rows+1])/2))
    np.save(prep/'coordenadas_6933.npy', xy)
    lookup = np.full(h*w, -1, np.int32)
    lookup[flat] = np.arange(len(flat), dtype=np.int32)
    inventory, geometries = [], []
    for filename in ('ANP_nacionales','ANP_regionales','ANP_privadas'):
        collection = json.loads((root/'cartografia'/(filename+'.geojson')).read_text(encoding='utf-8'))
        for feature in collection['features']:
            prop = dict(feature['properties'])
            uid = prop['anp_id']
            geometry = clean_geometry(geom_transform(FORWARD.transform, shape(feature['geometry'])), uid, repairs)
            continental = clean_geometry(geometry.intersection(domain), uid+'_continental', repairs)
            ids, weights, stats = intersections(continental, grid)
            ix = lookup[ids]
            valid = ix >= 0
            ix, weights_valid = ix[valid], weights[valid]
            total = float(weights.sum())
            covered = float(weights_valid.sum())
            n_eff = covered**2/float(np.sum(weights_valid**2)) if covered else 0
            record = {**prop, 'superficie_continental_m2': total, 'superficie_clima_m2': covered,
                      'superficie_no_evaluable_m2': max(0.,total-covered), 'cobertura_climatica': covered/total if total else None,
                      'n_celdas': len(ix), 'n_peso': n_eff,
                      'estado': 'evaluable' if covered else ('sin_datos_climaticos' if total else 'sin_superficie_continental'),
                      'error_cierre_m2': stats['error_m2']}
            inventory.append(record)
            np.savez_compressed(prep/'asociaciones'/(uid+'.npz'), index=ix, area=weights_valid, flat_todo=ids, area_todo=weights)
            geometries.append({'type':'Feature','properties':{'anp_id':uid},'geometry': mapping(geom_transform(BACKWARD.transform,continental))})
            log(f'{uid}: {len(ix)} celdas; n_peso {n_eff:.1f}')
    union = shapely.union_all([shape(g['geometry']) for g in geometries if not shape(g['geometry']).is_empty])
    union = clean_geometry(geom_transform(FORWARD.transform, union), 'UNION_CONSERVACION', repairs)
    union_flat, union_area, union_closure = intersections(union, grid)
    union_weights = np.zeros(len(flat), np.float64)
    union_idx = lookup[union_flat]
    good = union_idx>=0
    union_weights[union_idx[good]] = union_area[good]
    if np.any(union_weights > areas + np.maximum(AREA_ATOL,areas*AREA_RTOL)):
        raise ValueError('Union de conservacion supera dominio')
    np.save(prep/'superficie_union.npy', np.minimum(union_weights,areas))
    save_json(prep/'inventario.json', inventory)
    write_csv(out/'inventario_unidades.csv', inventory)
    save_json(prep/'unidades_continentales.geojson', {'type':'FeatureCollection','features':geometries})
    save_json(prep/'peru_continental.geojson', {'type':'Feature','properties':{},'geometry':mapping(geom_transform(BACKWARD.transform,domain))})
    result = {'fuentes_id':source_id,'fuentes':payload,'grid':grid,'n_celdas':len(flat), 'n_unidades':len(inventory),
              'n_evaluables':sum(u['estado']=='evaluable' for u in inventory),
              'superficie_dominio_m2':closure['area_m2'],'superficie_clima_m2':float(areas.sum()),
              'cierre_dominio':closure,'cierre_union':union_closure,'reparaciones':repairs,'rangos':ranges,
              'radiacion':'Indicador relativo; no publicar valores absolutos como W/m2; discrepancia de escala de origen.',
              'segundos':time.perf_counter()-started,'versiones':{'python':platform.python_version(),'numpy':np.__version__,
              'rasterio':rasterio.__version__,'shapely':shapely.__version__}}
    save_json(checkpoint,result)
    log(f'Preparacion completa: {len(flat)} celdas, {result["n_evaluables"]} unidades evaluables')
    return result


def national_model(out, cap=100):
    prep = out/'preparados'
    target = out/'modelo'/f'modelo_cond{cap}.json'
    if target.exists():
        return json.loads(target.read_text(encoding='utf-8'))
    X = np.load(prep/'clima_fisico.npy', mmap_mode='r')
    xy = np.load(prep/'coordenadas_6933.npy', mmap_mode='r')
    areas = np.load(prep/'dominio.npz')['area']
    tiles = np.floor(xy/5000).astype(np.int32)
    keys, inverse = np.unique(tiles, axis=0, return_inverse=True)
    weights = np.bincount(inverse, weights=areas)
    rng = np.random.default_rng(SEED)
    # Una celda por tesela, elegida con probabilidad proporcional a su superficie valida.
    race = -np.log(np.maximum(rng.random(len(X)),1e-300))/areas
    order = np.lexsort((race,inverse))
    first = np.r_[True, np.diff(inverse[order])!=0]
    sample = order[first]
    chosen_tiles = np.arange(len(keys))
    if len(sample)>50000:
        chosen_tiles = np.sort(rng.choice(len(sample),50000,replace=False))
        sample = sample[chosen_tiles]
    ww = weights[chosen_tiles]
    ww = ww/ww.sum()
    data = transform_features(np.asarray(X[sample],dtype=np.float64))
    mean = np.sum(data*ww[:,None],axis=0)
    centered = data-mean
    sd = np.sqrt(np.sum(centered**2*ww[:,None],axis=0))
    z = centered/sd
    corr = (z*ww[:,None]).T@z
    eig, vec = np.linalg.eigh(corr)
    need = max(0., (eig[-1]-cap*eig[0])/(cap-1+eig[-1]-cap*eig[0]))
    shrinkage = max(.02, float(need))
    regularized = (1-shrinkage)*eig+shrinkage
    whitening = vec/np.sqrt(regularized*P)
    model = {'variables':list(VARIABLES),'transformaciones':TRANSFORMACIONES,
             'mu':mean.tolist(),'sd':sd.tolist(),'R':corr.tolist(),'whitening':whitening.tolist(),
             'lambda':shrinkage,'autovalores':eig.tolist(),'autovalores_regularizados':regularized.tolist(),
             'condicion_inicial':float(eig[-1]/eig[0]),'condicion_final':float(regularized[-1]/regularized[0]),
             'condicion_maxima':cap,'n_muestra':len(sample),'n_teselas_validas':len(keys),'semilla':SEED,
             'muestreo':'Una celda PPS por tesela 5km, pesos por superficie continental valida de tesela; seleccion uniforme de teselas si >50000.'}
    target.parent.mkdir(exist_ok=True)
    np.savez_compressed(target.with_suffix('.npz'), index=sample, pesos=ww, teselas=keys[chosen_tiles])
    save_json(target,model)
    dest = np.lib.format.open_memmap(out/'modelo'/f'clima_transformado_cond{cap}.npy',mode='w+',dtype=np.float64,shape=X.shape)
    for start in range(0,len(X),65536):
        dest[start:start+65536] = apply_model(X[start:start+65536],model)
    dest.flush()
    log(f'Metrica nacional cond {cap}: {len(sample)} teselas, lambda={shrinkage:.6f}')
    return model


def apply_model(X,model):
    return ((transform_features(np.array(X,dtype=np.float64))-model['mu'])/model['sd'])@np.asarray(model['whitening'])


def pps_sample(weights,n,rng):
    """Sistematico PPS de tamano fijo, con inclusiones conocidas y unidades de certeza."""
    n = min(n,len(weights))
    if n==len(weights):
        return np.arange(n),np.ones(n)
    pi = np.zeros(len(weights),np.float64)
    active = np.ones(len(weights),bool)
    remaining = n
    while remaining:
        proposed = remaining*weights[active]/weights[active].sum()
        certainty = proposed>=1
        ix = np.flatnonzero(active)
        if not np.any(certainty):
            pi[ix] = proposed
            break
        pi[ix[certainty]] = 1
        active[ix[certainty]] = False
        remaining -= int(certainty.sum())
    pi[-1] += n-pi.sum()  # ajuste de redondeo; no modifica inclusiones en precision util.
    positions = rng.random()+np.arange(n)
    selected = np.searchsorted(np.cumsum(pi),positions)
    if len(np.unique(selected))!=n or np.any(pi>1+1e-12):
        raise ValueError('Muestra PPS invalida')
    return selected,pi[selected]


def peak_mib():
    """Memoria maxima del proceso (working set pico en Windows)."""
    if os.name!='nt':
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in (
            'PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
            'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    counters=Counters(); counters.cb=ctypes.sizeof(counters)
    ctypes.windll.kernel32.GetCurrentProcess.restype=wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(counters),counters.cb)
    return counters.PeakWorkingSetSize/2**20


def distances(out,cap=100,only=None):
    """d_j(c) exacta y vecino r* para todas las celdas; no depende del umbral."""
    model=national_model(out,cap)
    prep=out/'preparados'
    climate=np.load(out/'modelo'/f'clima_transformado_cond{cap}.npy')
    inventory=json.loads((prep/'inventario.json').read_text(encoding='utf-8'))
    folder=out/'distancias'/f'cond{cap}'
    folder.mkdir(parents=True,exist_ok=True)
    source=json.loads((prep/'preparacion.json').read_text(encoding='utf-8'))['fuentes_id']
    config=hashlib.sha256(json.dumps({'fuentes':source,'whitening':model['whitening'],'mu':model['mu'],'sd':model['sd']}).encode()).hexdigest()
    probe=np.random.default_rng(SEED).choice(len(climate),16,replace=False)
    for unit in inventory:
        uid=unit['anp_id']
        if only and uid not in only:
            continue
        meta=folder/f'{uid}.json'
        if meta.exists():
            if json.loads(meta.read_text(encoding='utf-8'))['config']!=config:
                raise ValueError(f'{uid}: distancia de otra configuracion en {folder}')
            continue
        index=np.load(prep/'asociaciones'/f'{uid}.npz')['index']
        if not len(index):
            continue
        started=time.perf_counter()
        refs=climate[index]
        tree=cKDTree(refs,leafsize=32)
        d=np.empty(len(climate)); nn=np.empty(len(climate),np.int32)
        for s in range(0,len(climate),65536):
            dd,jj=tree.query(climate[s:s+65536],k=1,eps=0,workers=1)
            d[s:s+65536]=dd; nn[s:s+65536]=index[jj]
        brute=np.array([np.sqrt(np.min(np.sum((refs-climate[q])**2,axis=1))) for q in probe])
        error=float(np.max(np.abs(brute-d[probe])))
        if error>1e-10 or np.any(d[index]!=0):
            raise ValueError(f'{uid}: busqueda no exacta ({error})')
        np.savez_compressed(folder/f'{uid}.npz',distancia=d.astype(np.float32),vecino=nn)
        save_json(meta,{'anp_id':uid,'config':config,'n_referencias':len(index),'segundos':time.perf_counter()-started,
                        'error_max_directo':error,'memoria_pico_mib':peak_mib(),'bytes':(folder/f'{uid}.npz').stat().st_size})
        log(f'Distancias cond{cap} {uid}: {len(index)} refs, {time.perf_counter()-started:.1f} s')


UNITS={'bio01':'°C','bio02':'°C','bio04':'°C','bio05':'°C','bio06':'°C','bio12':'mm/año','bio15':'%',
       'hursmean':'%','vpdmean':'kPa','rsdsmean':'z nacional (sin valor absoluto)','petmean':'mm/año'}


def fichas(out,cap=100):
    """Distribucion ponderada por superficie a_jc de cada predictor y relaciones T-P-HR."""
    prep=out/'preparados'
    model=national_model(out,cap)
    X=np.load(prep/'clima_fisico.npy').astype(np.float64)
    if 'rsdsmean' in VARIABLES:
        k=VARIABLES.index('rsdsmean')
        X[:,k]=(X[:,k]-model['mu'][k])/model['sd'][k]  # radiacion solo como indicador relativo
    inventory=json.loads((prep/'inventario.json').read_text(encoding='utf-8'))
    rows,relations=[],[]
    pair=[VARIABLES.index(v) for v in ('bio01','bio12','hursmean') if v in VARIABLES]
    for unit in inventory:
        assoc=np.load(prep/'asociaciones'/f"{unit['anp_id']}.npz")
        ix,w=assoc['index'],assoc['area']
        if not len(ix):
            continue
        data=X[ix]; w=w/w.sum()
        base={'anp_id':unit['anp_id'],'nombre':unit['nombre'],'modalidad':unit['modalidad'],'categoria':unit.get('categoria')}
        for j,variable in enumerate(VARIABLES):
            x=data[:,j]; mean=float(w@x)
            q=weighted_quantile(x,w,[.05,.25,.5,.75,.95])
            rows.append({**base,'variable':variable,'unidad':UNITS[variable],'media':mean,'sd':float(np.sqrt(w@(x-mean)**2)),
                         'min':float(x.min()),'p05':float(q[0]),'p25':float(q[1]),'mediana':float(q[2]),'p75':float(q[3]),
                         'p95':float(q[4]),'max':float(x.max())})
        if len(ix)>=3 and len(pair)==3:
            c=np.cov(data[:,pair].T,aweights=w)
            s=np.sqrt(np.diag(c))
            r=c/np.outer(s,s) if np.all(s>0) else np.full((3,3),np.nan)
            relations.append({**base,'n_celdas':len(ix),'r_T_P':r[0,1],'r_T_HR':r[0,2],'r_P_HR':r[1,2]})
    write_csv(out/'fichas_climaticas.csv',rows)
    write_csv(out/'fichas_relaciones_T_P_HR.csv',relations)
    log(f'Fichas: {len(rows)//P} unidades')


def controles(out,cap=100):
    """Coherencia espacial del indicador de radiacion frente a variables vinculadas (muestra nacional)."""
    from scipy.stats import spearmanr
    X=np.load(out/'preparados'/'clima_fisico.npy',mmap_mode='r')
    sample=np.load(out/'modelo'/f'modelo_cond{cap}.npz')['index']
    data=np.asarray(X[np.sort(sample)],np.float64)
    k=VARIABLES.index('rsdsmean')
    result={v:float(spearmanr(data[:,k],data[:,j])[0]) for j,v in enumerate(VARIABLES) if j!=k}
    raw=data[:,k]
    save_json(out/'control_radiacion.json',{'spearman_rsds_vs':result,'rango_decodificado_archivo':[float(raw.min()),float(raw.max())],
        'nota':'Solo coherencia de orden espacial; la escala absoluta sigue sin contrastar y no se publica.'})
    log(f'Control radiacion: {result}')


CAPAS=ROOT.parents[2]/'capas_sig'
DEPARTAMENTOS=CAPAS/'Peru'/'Departamentos_Peru'
# Ecorregiones del Peru (MINAM 2017): estrato de muestreo de las redes nulas y de reporte.
ECORREGIONES_MINAM=Path(__import__('os').environ.get('SIG_DIR',r'C:\Archivos\SIG'))/'N53_ECORREGIONES'/'TEM_BIOLO_ECORREGIONES_FINAL'  # MINAM (2017)
# Las 9 continentales, de la Amazonia a la costa: codigo, nombre y valor del campo 'ecorregion' del DBF. Las dos marinas
# (mar frio y mar tropical) estratifican la superficie del mar (ejecutar_marino.py); el fondo sigue por profundidad.
ECO_MINAM=[('BTA','Bosque tropical amazónico','Bosque Tropical Amazonico'),('SPA','Sabana de palmeras','Sabana de Palmeras'),
           ('SAL','Selva alta (yungas)','Selva Alta (Yungas)'),('PAR','Páramo','Paramo'),('PUN','Puna','Puna'),
           ('SES','Serranía esteparia','Serrania Esteparia'),('BSE','Bosque seco ecuatorial','Bosque Seco Ecuatorial'),
           ('BTP','Bosque tropical del Pacífico','Bosque Tropical del Pacifico Peruano'),
           ('DCP','Desierto costero del Pacífico','Desierto Costanero del Pacifico Peruano')]


def zonas(out):
    """Departamento INEI de cada celda valida (celdas sin poligono: vecino mas cercano)."""
    from scipy.ndimage import distance_transform_edt
    from shp_min import read_dbf, read_shp
    prep=out/'preparados'; target=prep/'zonas.json'
    if target.exists():
        return json.loads(target.read_text(encoding='utf-8'))
    grid=json.loads((prep/'preparacion.json').read_text(encoding='utf-8'))['grid']
    h,w=grid['height'],grid['width']; aff=Affine(*grid['transform'])
    flat=np.load(prep/'dominio.npz')['flat']
    rows=read_dbf(str(DEPARTAMENTOS)+'.dbf'); geoms=read_shp(str(DEPARTAMENTOS)+'.shp')
    deps={int(r['CCDD']):r['NOMBDEP'] for r in rows}
    raster=rasterize([(mapping(g),int(r['CCDD'])) for r,g in zip(rows,geoms)],out_shape=(h,w),transform=aff,fill=0,dtype='int16')
    nearest=distance_transform_edt(raster==0,return_distances=False,return_indices=True)
    departamento=raster[nearest[0],nearest[1]].reshape(-1)[flat]
    fill=int(np.count_nonzero(raster.reshape(-1)[flat]==0))
    np.savez_compressed(prep/'zonas.npz',departamento=departamento.astype(np.int8))
    result={'departamentos':deps,'celdas_rellenadas_departamento':fill,
            'fuentes':{DEPARTAMENTOS.name:sha256(Path(str(DEPARTAMENTOS)+'.shp'))},
            'criterio':'Departamento INEI por centro de celda; las celdas validas sin poligono toman el del vecino mas cercano.'}
    save_json(target,result)
    log(f'Zonas: {fill} celdas rellenadas (departamento)')
    return result


def ecorregiones(out,lista=None):
    """Ecorregion MINAM (2017) de cada celda valida por centro de celda; las celdas sin poligono de la lista (bordes de la
    capa) toman la del vecino mas cercano."""
    lista=lista or ECO_MINAM
    from scipy.ndimage import distance_transform_edt
    from shp_min import read_dbf, read_shp
    prep=out/'preparados'; target=prep/'ecorregiones_minam.json'
    if target.exists():
        return json.loads(target.read_text(encoding='utf-8'))
    grid=json.loads((prep/'preparacion.json').read_text(encoding='utf-8'))['grid']
    h,w=grid['height'],grid['width']; aff=Affine(*grid['transform'])
    dom=np.load(prep/'dominio.npz'); flat,area=dom['flat'],dom['area']
    utm=Transformer.from_crs(32718,4326,always_xy=True)  # la capa viene en WGS 84 / UTM 18S
    valor={v:k+1 for k,(_,_,v) in enumerate(lista)}
    shapes={}
    for r,g in zip(read_dbf(str(ECORREGIONES_MINAM)+'.dbf','latin-1'),read_shp(str(ECORREGIONES_MINAM)+'.shp')):
        for parte in getattr(g,'geoms',[g]):
            parte=shapely.make_valid(geom_transform(utm.transform,parte)); v=r['ecorregion']
            # error de atributo de la capa; cuatro partes de "Sabana de Palmeras" junto al lago Titicaca (> 3 800 m,
            # 137,5 km2) son puna. La sabana de palmeras del Peru es solo la de las Pampas del Heath (norte de 14 S).
            if v=='Sabana de Palmeras' and parte.representative_point().y<-14:
                v='Puna'
            if v in valor:
                shapes.setdefault(valor[v],[]).append(parte)
    if len(shapes)!=len(lista):
        raise ValueError(f'Faltan ecorregiones MINAM en la capa: {len(shapes)} de {len(lista)}')
    raster=rasterize([(mapping(shapely.union_all(v)),k) for k,v in shapes.items()],out_shape=(h,w),transform=aff,fill=0,dtype='int16')
    nearest=distance_transform_edt(raster==0,return_distances=False,return_indices=True)
    eco=raster[nearest[0],nearest[1]].reshape(-1)[flat]-1
    fill=int(np.count_nonzero(raster.reshape(-1)[flat]==0))
    np.savez_compressed(prep/'ecorregiones_minam.npz',ecorregion=eco.astype(np.int8))
    result={'ecorregiones':[{'codigo':c,'nombre':n,'superficie_km2':float(area[eco==i].sum()/1e6),'celdas':int((eco==i).sum())}
                            for i,(c,n,_) in enumerate(lista)],
            'celdas_rellenadas':fill,'superficie_rellenada_km2':float(area[raster.reshape(-1)[flat]==0].sum()/1e6),
            'fuente':{ECORREGIONES_MINAM.name:sha256(Path(str(ECORREGIONES_MINAM)+'.shp'))},
            'criterio':'Ecorregion MINAM (2017) por centro de celda; las celdas validas sin poligono toman la del vecino mas cercano.'}
    save_json(target,result)
    log(f'Ecorregiones MINAM: {fill} celdas rellenadas')
    return result


def estratos_muestreo(prep):
    """Estrato de muestreo y reporte de cada celda: ecorregion MINAM si la capa la tiene (tierra y superficie del mar);
    si no, el estrato de zonas.npz (fondo marino: profundidad)."""
    if (prep/'ecorregiones_minam.npz').exists():
        nombres=[e['nombre'] for e in json.loads((prep/'ecorregiones_minam.json').read_text(encoding='utf-8'))['ecorregiones']]
        return np.load(prep/'ecorregiones_minam.npz')['ecorregion'],nombres
    nombres=[e['nombre'] for e in json.loads((prep/'zonas.json').read_text(encoding='utf-8'))['ecorregiones']]
    return np.load(prep/'zonas.npz')['ecorregion'],nombres


CORTES_D=[0,.05,.1,.2,.3,.5,1.0]  # anclas: 0,05 celdas vecinas; 0,2 celdas a ~9 km; 1,0 dos lugares al azar del Peru
# Escala de clases de d definida por el usuario (color, alfa, lectura); d > 1 transparente (no analogo).
CLASES_D=[('#5a0010',1.0,'casi idéntico'),('#a50f15',1.0,'muy similar'),('#de2d26',1.0,'similar'),
          ('#fb6a4a',.55,'moderado'),('#fcae91',.35,'débil'),('#fde0d2',.25,'poco similar')]
NOTA_D=('Referencias de d: 0,05 ≈ dos celdas vecinas (~1 km); 0,2 ≈ celdas a ~9 km; 1,0 ≈ dos lugares al azar del Perú.\n'
        'CHELSA V2.1 1981–2010 · 30″ · 11 predictores · Mahalanobis regularizada (cond ≤ 100). Gris: sin datos climáticos.')


def es(x,nd=0):
    """Numero en formato espanol: punto de miles y coma decimal."""
    return f'{x:,.{nd}f}'.replace(',','\0').replace('.',',').replace('\0','.')


def _corto(x):
    return es(x,2).rstrip('0').rstrip(',') if x else '0'


def _lienzo(out):
    """Contexto comun de las figuras."""
    from types import SimpleNamespace
    prep=out/'preparados'
    grid=json.loads((prep/'preparacion.json').read_text(encoding='utf-8'))['grid']
    h,w=grid['height'],grid['width']; aff=Affine(*grid['transform'])
    dom=np.load(prep/'dominio.npz')
    domain=np.full(h*w,np.nan,np.float32); domain[dom['todo_flat']]=0; domain[dom['flat']]=1  # 0 sin clima, 1 con clima
    return SimpleNamespace(grid=grid,h=h,w=w,aff=aff,flat=dom['flat'],area=dom['area'],domain=domain.reshape(h,w),
        extent=(aff.c,aff.c+w*aff.a,aff.f+h*aff.e,aff.f),
        inventory=json.loads((prep/'inventario.json').read_text(encoding='utf-8')),
        shapes={f['properties']['anp_id']:shape(f['geometry']) for f in
                json.loads((prep/'unidades_continentales.geojson').read_text(encoding='utf-8'))['features']},
        peru=shape(json.loads((prep/'peru_continental.geojson').read_text(encoding='utf-8'))['geometry']).simplify(.005))


def _contorno(ax,geometry,**style):
    for poly in getattr(geometry,'geoms',[geometry]):
        for ring in [poly.exterior,*poly.interiors]:
            ax.plot(*ring.xy,**style)


def _panel(ax,base,values,cmap,norm,geometry=None):
    from matplotlib.colors import ListedColormap
    ground=ListedColormap(['#d6d6d2','#ffffff']); ground.set_bad((0,0,0,0))
    image=np.full(base.h*base.w,np.nan,np.float32); image[base.flat]=values
    ax.imshow(base.domain,cmap=ground,vmin=0,vmax=1,extent=base.extent,interpolation='nearest')
    im=ax.imshow(image.reshape(base.h,base.w),cmap=cmap,norm=norm,extent=base.extent,interpolation='nearest')
    _contorno(ax,base.peru,color='#6b6b66',lw=.4)
    if geometry is not None and not geometry.is_empty:
        _contorno(ax,geometry,color='#0096c7',lw=.9)
        if geometry.area<.01:  # < ~120 km2: el contorno no se ve a escala nacional
            ax.plot(geometry.centroid.x,geometry.centroid.y,marker='o',ms=10,mfc='none',mec='#0096c7',mew=1.3)
    ax.set_aspect(1/np.cos(np.deg2rad(-9.2))); ax.set_axis_off()
    return im


def _subtitulo(unit):
    km2=unit['superficie_clima_m2']/1e6
    return (f"{unit['modalidad'].capitalize()} · {unit.get('categoria') or ''} · {es(km2,0 if km2>=10 else 2)} km² con clima · "
            f"{es(unit['n_celdas'])} celdas")


def _aviso_agua(ax,unit):
    """Recuadro sobre el océano del mapa en las unidades que el dominio terrestre no representa completas."""
    texto=TXT.get('agua',{}).get(unit['anp_id'])
    if texto:
        pct=100*unit['superficie_continental_m2']/unit['area_m2']
        cifra=es(pct,0) if pct>=1 else es(pct,1) if pct>=.1 else '< 0,1'
        ax.text(.02,.03,f"Solo la parte terrestre: {cifra} % de la superficie legal.\n{texto}",
                transform=ax.transAxes,fontsize=7.5,color='#1f1f1d',va='bottom',ha='left',
                bbox=dict(boxstyle='round,pad=.5',fc='white',ec='#0096c7',lw=.8))


def _escala_d():
    from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba
    cmap=ListedColormap([to_rgba(c,a) for c,a,_ in CLASES_D]); cmap.set_over((0,0,0,0)); cmap.set_bad((0,0,0,0))
    return cmap,BoundaryNorm(CORTES_D,len(CLASES_D))


def _leyenda_d(ax,extra=()):
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch
    handles=[Patch(facecolor=to_rgba(c,a),edgecolor='#bdbdb5',lw=.4,label=f'{_corto(lo)}–{_corto(hi)} · {txt}')
             for (c,a,txt),lo,hi in zip(CLASES_D,CORTES_D[:-1],CORTES_D[1:])]
    handles.append(Patch(facecolor='white',edgecolor='#bdbdb5',lw=.4,label='> 1 · no análogo'))
    handles.extend(extra)
    ax.legend(handles=handles,title='Distancia climática d',title_fontsize=7.5,fontsize=7,loc='upper left',
              bbox_to_anchor=(1.0,.62),frameon=False,alignment='left')


def _figura_d(base,values,title,subtitle,geometry,path,unit=None):
    import matplotlib.pyplot as plt
    cmap,norm=_escala_d()
    fig,ax=plt.subplots(figsize=(8.2,9.4),dpi=150)
    _panel(ax,base,values,cmap,norm,geometry)
    if unit:
        _aviso_agua(ax,unit)
    ax.set_title(title,loc='left',fontsize=11,color='#1f1f1d',pad=24)
    ax.text(0,1.005,subtitle,transform=ax.transAxes,fontsize=7,color='#55554f',va='bottom',ha='left')
    _leyenda_d(ax)
    ax.text(0,-.005,NOTA_D,transform=ax.transAxes,fontsize=6,color='#55554f',va='top',ha='left')
    fig.savefig(path,bbox_inches='tight'); plt.close(fig)


def mapas_distancia(out,cap=100,only=None):
    """GeoTIFF y figura de d_j(c) (sin umbral) con escala de clases interpretables; mapa del sistema."""
    import matplotlib
    matplotlib.use('Agg')
    base=_lienzo(out)
    src=out/'distancias'/f'cond{cap}'; dst=out/'mapas_distancia'/f'cond{cap}'
    (dst/'figuras').mkdir(parents=True,exist_ok=True)
    profile=dict(driver='GTiff',width=base.w,height=base.h,count=1,dtype='float32',nodata=-9999.0,crs=base.grid['crs'],
                 transform=base.aff,tiled=True,blockxsize=256,blockysize=256,compress='deflate',predictor=3)
    best=np.full(len(base.flat),np.inf,np.float32); owner=np.full(len(base.flat),-1,np.int16); done=[]
    evaluable=[u for u in base.inventory if u['estado']=='evaluable' and (not only or u['anp_id'] in only)]
    for unit in evaluable:
        uid=unit['anp_id']
        if not (src/f'{uid}.npz').exists():
            continue
        d=np.load(src/f'{uid}.npz')['distancia']
        closer=d<best; best[closer]=d[closer]; owner[closer]=len(done); done.append(uid)
        tif=dst/f'distancia_{uid}.tif'
        if not tif.exists():
            band=np.full(base.h*base.w,-9999,np.float32); band[base.flat]=d
            with rasterio.open(tif.with_suffix('.tmp.tif'),'w',**profile) as ds:
                ds.write(band.reshape(base.h,base.w),1)
            tif.with_suffix('.tmp.tif').replace(tif)
        png=dst/'figuras'/f'distancia_{uid}.png'
        if not png.exists():
            _figura_d(base,d,f"{unit['nombre']} ({uid})",_subtitulo(unit)+f" · n_peso {es(unit['n_peso'])}\n"
                      'Distancia al clima más semejante dentro del área; sin umbral',base.shapes.get(uid),png,unit)
        log(f'Mapa distancia {uid}')
    if len(done)==sum(u['estado']=='evaluable' for u in base.inventory):
        band=np.full(base.h*base.w,-9999,np.float32); band[base.flat]=best
        with rasterio.open(dst/'sistema_distancia_minima.tif','w',**profile) as ds:
            ds.write(band.reshape(base.h,base.w),1)
        band=np.full(base.h*base.w,-1,np.int16); band[base.flat]=owner
        with rasterio.open(dst/'sistema_unidad_mas_cercana.tif','w',**{**profile,'dtype':'int16','nodata':-1,'predictor':2}) as ds:
            ds.write(band.reshape(base.h,base.w),1)
        write_csv(dst/'sistema_unidad_mas_cercana_codigos.csv',[{'codigo':i,'anp_id':u} for i,u in enumerate(done)])
        _figura_d(base,best,f"Distancia al clima protegido más semejante ({TXT['red']})",
                  f"Mínimo de d sobre las {len(done)} unidades con {TXT['superficie']} · 0 dentro de la unión de conservación",
                  shapely.union_all(list(base.shapes.values())),dst/'figuras'/'sistema_distancia_minima.png')
        q=weighted_quantile(best,base.area,[.5,.75,.9,.95,.99])
        save_json(dst/'sistema_resumen.json',{'n_unidades':len(done),'cuantiles_superficie_d_min':dict(zip(['p50','p75','p90','p95','p99'],map(float,q))),
                  'nota':f"Cuantiles ponderados por {TXT['superficie']} valida."})
        log(f'Sistema: {len(done)} unidades; cuantiles d_min {q}')


ALFA=.05  # nivel de la prueba frente a las redes nulas, despues de Holm


def prueba(ad,rd):
    """Prueba bilateral de D̄ real frente a las D̄ de las redes nulas; ses > 0: peor (mas lejos) que el azar."""
    n=len(ad); le=int((ad<=rd).sum()); ge=int((ad>=rd).sum()); sd=float(np.std(ad,ddof=1))
    return {'redes_azar_con_D_menor_o_igual':le,'redes_mayor_o_igual':ge,'n_redes':n,
            'p_bilateral':min(1.,2*min(1+le,1+ge)/(n+1)),'ses':(rd-float(np.mean(ad)))/sd if sd>0 else float('nan')}


def holm(summary,no_evaluables=()):
    """p_holm y direccion: Holm sobre los estratos de un mismo dominio o capa; el primero (nacional) va aparte.
    Los estratos no evaluables (casi todo protegido: no hay donde mover las areas) quedan fuera de la familia."""
    summary[0]['p_holm']=summary[0]['p_bilateral']
    rest=sorted([r for r in summary[1:] if r['estrato'] not in no_evaluables] if no_evaluables else summary[1:],
                key=lambda r:r['p_bilateral']); acumulado=0.
    for k,r in enumerate(rest):
        acumulado=max(acumulado,min(1.,(len(rest)-k)*r['p_bilateral'])); r['p_holm']=acumulado
    for r in summary:
        if no_evaluables and r is not summary[0] and r['estrato'] in no_evaluables:
            r['p_holm'],r['direccion']=None,'no evaluable'
            continue
        r['direccion']=('no distinguible' if r['p_holm']>=ALFA or not r['ses']==r['ses'] else 'peor' if r['ses']>0 else 'mejor')
    return summary


def _estratos(grupo,nombres,sample):
    return [(TXT['territorio'],np.ones(len(sample),bool))]+([(n,grupo[sample]==i) for i,n in enumerate(nombres)] if len(nombres)>1 else [])


def p_mediana(real_dmin,azar_dmin,weight,strata,no_evaluables=()):
    """D̄ (p-mediana: d_min medio ponderado por superficie) real y de las redes nulas, con la prueba de cada estrato."""
    summary=[]
    for name,m in strata:
        rd=float(np.average(real_dmin[m],weights=weight[m])); ad=np.average(azar_dmin[:,m],axis=1,weights=weight[m])
        summary.append({'estrato':name,'D_real':rd,'D_azar_mediana':float(np.median(ad)),'D_azar_p2_5':float(np.quantile(ad,.025)),
                        'D_azar_p97_5':float(np.quantile(ad,.975)),**prueba(ad,rd)})
    return holm(summary,no_evaluables)


PROTEGIDO_NO_EVALUABLE=.5  # estrato con >= 50 % de su superficie protegida: la prueba es degenerada ("no evaluable")
# Las 8 orientaciones discretas de una huella (grupo diedrico D4): rotaciones de 0, 90, 180 y 270 grados, con o sin reflejo.
# en celdas de 30" la rotacion de 90 grados distorsiona la forma en cos(lat) (< 3 % en el Peru)
ORIENTACIONES=np.array([[[1,0],[0,1]],[[0,1],[-1,0]],[[-1,0],[0,-1]],[[0,-1],[1,0]],
                        [[1,0],[0,-1]],[[0,1],[1,0]],[[-1,0],[0,1]],[[0,-1],[-1,0]]])
SEMILLA_DISENO={'N0':7,'N1':41,'N2c':42}  # desplazamiento de SEED por diseno
TVD_MAX=.2
DOMINIO_MIN=.9


def huellas_red(prep,grupo,n_grupos,w,flat):
    """Huella de cada unidad evaluable: desplazamientos (fila, columna) respecto de su celda de referencia, pesos a_jc y
    composicion por estrato. Referencia: la celda de la ecorregion dominante mas cercana al centroide (N1, N2c); la
    primera celda de la huella (N0)."""
    inventory=json.loads((prep/'inventario.json').read_text(encoding='utf-8'))
    rows,cols=flat//w,flat%w; result=[]
    for u in inventory:
        if u['estado']!='evaluable':
            continue
        z=np.load(prep/'asociaciones'/f"{u['anp_id']}.npz"); ix,a=z['index'],z['area']
        comp=np.bincount(grupo[ix],weights=a,minlength=n_grupos)/a.sum(); dom=int(np.argmax(comp))
        r,c=rows[ix],cols[ix]; rc,cc=a@r/a.sum(),a@c/a.sum()
        cand=np.flatnonzero(grupo[ix]==dom); ref=cand[np.argmin((r[cand]-rc)**2+(c[cand]-cc)**2)]
        result.append({'anp_id':u['anp_id'],'index':ix,'area':a,'comp':comp,'dominante':dom,
                       'dr':r-r[ref],'dc':c-c[ref],'dr0':r-r[0],'dc0':c-c[0]})
    return result


def colocar(hu,anchor,o,rows,cols,lookup,h,w,grupo,natural,n_grupos,origen0=False):
    """Celdas de la huella trasladada a anchor con la orientacion o; fraccion en el dominio (celdas), fraccion natural y
    TVD de su composicion por estrato (ambas ponderadas por a_jc)."""
    dr,dc=(hu['dr0'],hu['dc0']) if origen0 else (hu['dr'],hu['dc'])
    m=ORIENTACIONES[o]
    rr=rows[anchor]+m[0,0]*dr+m[0,1]*dc; cc=cols[anchor]+m[1,0]*dr+m[1,1]*dc
    ok=(rr>=0)&(rr<h)&(cc>=0)&(cc<w)
    cells=np.full(len(dr),-1,np.int64); cells[ok]=lookup[rr[ok]*w+cc[ok]]
    inside=cells>=0; c=cells[inside]; a=hu['area'][inside]
    if not len(c):
        return c,0.,0.,1.
    nat=float(a@natural[c]/a.sum()) if natural is not None else 1.
    tvd=.5*float(np.abs(np.bincount(grupo[c],weights=a,minlength=n_grupos)/a.sum()-hu['comp']).sum())
    return c,inside.sum()/len(dr),nat,tvd


def factibilidad(out,dst,natural=None,umbral=None,n_anclas=5000):
    """Prueba de factibilidad de N2c: para cada area, n_anclas al azar entre los pixeles naturales de su ecorregion
    dominante, cada una con las 8 orientaciones; cuenta las posiciones validas distintas (>= 90 % en el dominio, fraccion
    natural >= umbral, TVD <= 0.2)."""
    prep=out/'preparados'
    grid=json.loads((prep/'preparacion.json').read_text(encoding='utf-8'))['grid']; h,w=grid['height'],grid['width']
    flat=np.load(prep/'dominio.npz')['flat']; rows,cols=flat//w,flat%w
    grupo,nombres=estratos_muestreo(prep); K=len(nombres)
    lookup=np.full(h*w,-1,np.int64); lookup[flat]=np.arange(len(flat))
    nat=np.ones(len(flat),bool) if natural is None else natural
    rng=np.random.default_rng(SEED+SEMILLA_DISENO['N2c']+1000)
    inventory={u['anp_id']:u for u in json.loads((prep/'inventario.json').read_text(encoding='utf-8'))}
    filas=[]; started=time.perf_counter()
    for hu in huellas_red(prep,grupo,K,w,flat):
        pool=np.flatnonzero((grupo==hu['dominante'])&nat)
        anclas=rng.choice(pool,n_anclas) if len(pool) else np.empty(0,int)
        validas=set(); dominio=natural_ok=tvd_ok=0
        umbral_j=umbral.get(hu['anp_id']) if isinstance(umbral,dict) else umbral
        for anchor in anclas:
            for o in range(8):
                _,fd,fn,tvd=colocar(hu,anchor,o,rows,cols,lookup,h,w,grupo,natural,K)
                dominio+=fd>=DOMINIO_MIN; natural_ok+=(fd>=DOMINIO_MIN)&(umbral_j is None or fn>=umbral_j); tvd_ok+=tvd<=TVD_MAX
                if fd>=DOMINIO_MIN and (umbral_j is None or fn>=umbral_j) and tvd<=TVD_MAX:
                    validas.add((int(anchor),o))
        u=inventory[hu['anp_id']]; km10=12  # celdas de 30" por ~10 km
        filas.append({'anp_id':hu['anp_id'],'nombre':u['nombre'],'modalidad':u['modalidad'],
                      'superficie_km2':float(hu['area'].sum()/1e6),'n_celdas':len(hu['index']),
                      'ecorregion_dominante':nombres[hu['dominante']],'fraccion_dominante':float(hu['comp'].max()),
                      'multi_ecorregion':int((hu['comp']>=.1).sum()>=2),'pixeles_ancla':len(pool),
                      'posiciones_validas':len(validas),
                      'anclas_validas_distintas_10km':len({(rows[a_]//km10,cols[a_]//km10) for a_,_ in validas}),
                      'pct_dominio_ok':100*dominio/(8*max(1,len(anclas))),'pct_natural_ok':100*natural_ok/(8*max(1,len(anclas))),
                      'pct_tvd_ok':100*tvd_ok/(8*max(1,len(anclas)))})
        log(f"Factibilidad {hu['anp_id']}: {len(validas)} posiciones validas")
    dst.mkdir(parents=True,exist_ok=True); write_csv(dst/'factibilidad_N2c.csv',filas)
    n=np.array([f['posiciones_validas'] for f in filas]); s=np.array([f['superficie_km2'] for f in filas])
    grandes=s>1000
    regla={'areas_pct_30':float(100*(n>=30).mean()),'superficie_pct_30':float(100*s[n>=30].sum()/s.sum()),
           'grandes':int(grandes.sum()),'grandes_con_10':int((n[grandes]>=10).sum())}
    regla['viable']=bool(regla['areas_pct_30']>=95 and regla['superficie_pct_30']>=95 and regla['grandes_con_10']==regla['grandes'])
    regla['impiden']=[f['anp_id'] for f in filas if f['posiciones_validas']<30 or (f['superficie_km2']>1000 and f['posiciones_validas']<10)]
    regla['segundos']=time.perf_counter()-started; regla['n_anclas']=n_anclas
    regla['umbral_natural']='por area' if isinstance(umbral,dict) else umbral
    save_json(dst/'factibilidad_N2c.json',regla)
    log(f"Factibilidad N2c: {regla}")
    return regla


def recalcular_prueba(base,cap,dst):
    """Prueba con las redes nulas ya simuladas en base (redes_aleatorias.npz), sin rehacer simulaciones."""
    z=np.load(base/'cobertura'/f'cond{cap}'/'redes_aleatorias.npz')
    grupo,nombres=estratos_muestreo(base/'preparados')
    summary=p_mediana(z['real_dmin'],z['azar_dmin'],z['peso'],_estratos(grupo,nombres,z['muestra']))
    dst.mkdir(parents=True,exist_ok=True); write_csv(dst/'p_mediana.csv',summary)
    log(f'Prueba recalculada desde {base}: {len(summary)} estratos, {len(z["azar_dmin"])} redes')
    return summary


def cobertura(out,cap=100,n_redes=999,por_grupo=8000,diseno='N0',dst=None,natural=None,umbral=None,limitar=False,min_km2=0,
              por_red=None):
    """Curvas de cobertura climatica de la red real frente a redes al azar de igual numero, forma y tamano de unidades.

    Disenos de las redes nulas (cada red traslada la huella de celdas de cada unidad):
    - N0: a cualquier celda del dominio, solo traslacion; se acepta si >= 90 % de la huella cae en el dominio.
    - N1: ancla en un pixel natural, 8 orientaciones; ademas, fraccion natural de la huella >= umbral (mascara_natural.py).
    - N2c: ancla en un pixel natural de la ecorregion dominante, 8 orientaciones; ademas, TVD de la composicion por
      ecorregion <= 0.2. En N1 y N2c, hasta 500 intentos; si ninguno cumple, se guarda el mejor y se registra.
    limitar: las huellas se colocan de la mayor a la menor y se rechaza una posicion si la fraccion de sus celdas ya
      ocupadas por huellas anteriores supera la que esa area comparte con otras en la red real (500 intentos, tambien N0).
    por_red(i, refs, hus, evaluate): si se da, solo reconstruye las redes (misma semilla, mismas redes) y entrega las
      celdas de cada una sin evaluarlas ni escribir resultados (agregacion_nulas.py).
    Muestra PPS estratificada por ecorregion MINAM (en el mar, por estrato de profundidad), pesos HT.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    dst=dst or out/'cobertura'/f'cond{cap}'; dst.mkdir(parents=True,exist_ok=True)
    prep=out/'preparados'
    grid=json.loads((prep/'preparacion.json').read_text(encoding='utf-8'))['grid']; h,w=grid['height'],grid['width']
    dom=np.load(prep/'dominio.npz'); flat,area=dom['flat'],dom['area']
    grupo,nombres=estratos_muestreo(prep); K=len(nombres)
    climate=np.load(out/'modelo'/f'clima_transformado_cond{cap}.npy')
    lookup=np.full(h*w,-1,np.int64); lookup[flat]=np.arange(len(flat))
    hus=huellas_red(prep,grupo,K,w,flat)
    if min_km2:  # robustez sin las areas diminutas (superficie oficial < min_km2), fuera de la red real y de las nulas
        chicas={u['anp_id'] for u in json.loads((prep/'inventario.json').read_text(encoding='utf-8')) if u['area_m2']<min_km2*1e6}
        hus=[hu for hu in hus if hu['anp_id'] not in chicas]
    huellas=[hu['index'] for hu in hus]
    union=np.load(prep/'superficie_union.npy')
    protegido=np.bincount(grupo,weights=union,minlength=K)/np.bincount(grupo,weights=area,minlength=K)
    no_evaluables={n for n,p in zip(nombres,protegido) if p>=PROTEGIDO_NO_EVALUABLE} if K>1 else set()
    rng=np.random.default_rng(SEED+7)
    sample,weight=[],[]
    for g in range(len(nombres)):
        ix=np.flatnonzero(grupo==g); n=min(por_grupo,len(ix))
        pick,pi=pps_sample(area[ix],n,rng)
        sample.append(ix[pick]); weight.append(area[ix[pick]]/pi)
    sample=np.concatenate(sample); weight=np.concatenate(weight); query=climate[sample]
    def evaluate(refs):
        refs=np.unique(refs)
        dmin,_=cKDTree(climate[refs],leafsize=32).query(query,k=1,eps=0,workers=8)
        return dmin.astype(np.float32),area[refs].sum()
    rows,cols=flat//w,flat%w
    # N0 usa el mismo generador de la muestra; N1 y N2c usan su propia semilla.
    rng_red=rng if diseno=='N0' else np.random.default_rng(SEED+SEMILLA_DISENO[diseno])
    nat=np.ones(len(flat),bool) if natural is None else natural
    anclas=[np.flatnonzero(nat) if diseno=='N1' else np.flatnonzero(nat&(grupo==hu['dominante'])) for hu in hus]
    # Superposicion permitida por area: fraccion de sus celdas que comparte con otras areas en la red real.
    veces=np.bincount(np.concatenate(huellas),minlength=len(flat))
    compartida=[float((veces[fp]>1).mean()) for fp in huellas]
    orden=np.argsort([-hu['area'].sum() for hu in hus],kind='stable') if limitar else np.arange(len(hus))
    def random_network():
        """Celdas de cada huella trasladada y si cumplio los criterios del diseno."""
        refs,cumple=[None]*len(hus),np.zeros(len(hus),bool)
        ocupado=np.zeros(len(flat),bool)
        for j in orden:
            hu,pool=hus[j],anclas[j]
            best=None; umbral_j=(umbral.get(hu['anp_id']) if isinstance(umbral,dict) else umbral) or 0
            if limitar:
                for _ in range(500):
                    if diseno=='N0':
                        cells,fd,fn,tvd=colocar(hu,rng_red.integers(len(flat)),0,rows,cols,lookup,h,w,grupo,None,K,origen0=True)
                    else:
                        cells,fd,fn,tvd=colocar(hu,pool[rng_red.integers(len(pool))],rng_red.integers(8),rows,cols,lookup,h,w,grupo,natural,K)
                    sup=float(ocupado[cells].mean()) if len(cells) else 1.
                    falta=(max(0,DOMINIO_MIN-fd)+max(0,sup-compartida[j])+(max(0,umbral_j-fn) if diseno!='N0' else 0)
                           +(max(0,tvd-TVD_MAX) if diseno=='N2c' else 0))
                    if best is None or falta<best[2]:
                        best=(cells,falta==0,falta)
                    if falta==0:
                        break
            elif diseno=='N0':
                for _ in range(50):
                    cells,fd,_,_=colocar(hu,rng_red.integers(len(flat)),0,rows,cols,lookup,h,w,grupo,None,K,origen0=True)
                    if best is None or len(cells)>len(best[0]):
                        best=(cells,fd>=DOMINIO_MIN)
                    if fd>=DOMINIO_MIN:
                        break
            else:
                for _ in range(500):
                    cells,fd,fn,tvd=colocar(hu,pool[rng_red.integers(len(pool))],rng_red.integers(8),rows,cols,lookup,h,w,grupo,natural,K)
                    falta=max(0,DOMINIO_MIN-fd)+max(0,umbral_j-fn)+(max(0,tvd-TVD_MAX) if diseno=='N2c' else 0)
                    if best is None or falta<best[2]:
                        best=(cells,falta==0,falta)
                    if falta==0:
                        break
            refs[j]=best[0]; cumple[j]=best[1]; ocupado[best[0]]=True
        return refs,cumple
    def overlap(refs):
        return float(sum(area[r].sum() for r in refs)-area[np.unique(np.concatenate(refs))].sum())
    started=time.perf_counter()
    real=evaluate(np.concatenate(huellas))
    sims,diag,fallos=[],[],[]
    for i in range(n_redes):
        refs,cumple=random_network()
        if por_red:
            por_red(i,refs,hus,evaluate); continue
        sims.append(evaluate(np.concatenate(refs))); fallos.append(~cumple)
        diag.append({'red':i+1,'superficie_km2':sims[-1][1]/1e6,'superposicion_km2':overlap(refs)/1e6,'huellas_fallidas':int((~cumple).sum())})
        log(f'Red al azar {i+1}/{n_redes} ({diseno})')
    if por_red:
        return
    fallos=np.array(fallos)
    np.savez_compressed(dst/'redes_aleatorias.npz',muestra=sample,peso=weight,real_dmin=real[0],
                        azar_dmin=np.array([s[0] for s in sims]),fallos=fallos,
                        azar_superficie_m2=np.array([s[1] for s in sims]),real_superficie_m2=real[1])
    write_csv(dst/'diagnostico_huellas.csv',[{'red':0,'superficie_km2':real[1]/1e6,'superposicion_km2':overlap(huellas)/1e6,
                                              'huellas_fallidas':0},*diag])
    dominante=np.array([hu['dominante'] for hu in hus])
    fallos_eco={n:float(100*fallos[:,dominante==k].mean()) for k,n in enumerate(nombres) if (dominante==k).any()}
    xs=np.round(np.arange(0,1.2001,.01),2); curves=[]
    strata=_estratos(grupo,nombres,sample)
    def frac(d,m):
        ww=weight[m]; o=np.argsort(d[m]); c=np.cumsum(ww[o])/ww.sum()
        return np.concatenate([[0],c])[np.searchsorted(d[m][o],xs,side='right')]
    for name,m in strata:
        fr=frac(real[0],m); fa=np.array([frac(s[0],m) for s in sims])
        for x,a,lo,md,hi in zip(xs,fr,*np.quantile(fa,[.025,.5,.975],axis=0)):
            curves.append({'estrato':name,'d':float(x),'real':float(a),'azar_p2_5':float(lo),'azar_mediana':float(md),'azar_p97_5':float(hi)})
    summary=p_mediana(real[0],np.array([s[0] for s in sims]),weight,strata,no_evaluables)
    write_csv(dst/'curvas_cobertura.csv',curves); write_csv(dst/'p_mediana.csv',summary)
    save_json(dst/'resumen.json',{'n_redes':n_redes,'n_muestra':len(sample),'superficie_red_real_km2':float(real[1]/1e6),
              'superficie_redes_azar_km2_mediana':float(np.median([s[1] for s in sims])/1e6),'segundos':time.perf_counter()-started,
              'diseno':diseno,'semilla':SEED+SEMILLA_DISENO[diseno],'superposicion_limitada':limitar,
              'n_areas':len(hus),'sin_areas_menores_km2':min_km2,
              'umbral_natural':'por area: min(0.9, fraccion natural real)' if isinstance(umbral,dict) else umbral,
              'superposicion_real_km2':overlap(huellas)/1e6,
              'superposicion_azar_km2_mediana':float(np.median([d['superposicion_km2'] for d in diag])),
              'huellas_fallidas_pct':float(100*fallos.mean()),'huellas_fallidas_pct_por_ecorregion_dominante':fallos_eco,
              'estratos_no_evaluables':sorted(no_evaluables),'memoria_pico_mib':peak_mib(),
              'criterios':{'N0':'Traslacion; >= 90 % en el dominio; 50 intentos.',
                           'N1':'Ancla natural, 8 orientaciones; >= 90 % en el dominio y fraccion natural >= umbral; 500 intentos.',
                           'N2c':'Ancla natural de la ecorregion dominante, 8 orientaciones; ademas TVD <= 0.2; 500 intentos.'}[diseno]})
    nr,nc=(len(strata)+2)//3,min(3,len(strata))
    fig,axs=plt.subplots(nr,nc,figsize=(13*nc/3,4*nr),dpi=140,sharex=True,sharey=True,squeeze=False)
    for ax,(name,_) in zip(axs.flat,strata):
        cc=[r for r in curves if r['estrato']==name]; s=next(r for r in summary if r['estrato']==name)
        x=[r['d'] for r in cc]
        ax.fill_between(x,[r['azar_p2_5'] for r in cc],[r['azar_p97_5'] for r in cc],color='#bdbdb5',alpha=.6,lw=0,label='Redes al azar (95 %)')
        ax.plot(x,[r['azar_mediana'] for r in cc],color='#7a7a72',lw=1,label='Mediana al azar')
        ax.plot(x,[r['real'] for r in cc],color=CLASES_D[1][0],lw=1.8,label='Red real')
        ax.set_title(name,fontsize=10,loc='left')
        ax.text(.97,.05,f"d_min medio: real {es(s['D_real'],3)}\n"
                f"azar {es(s['D_azar_mediana'],3)} [{es(s['D_azar_p2_5'],3)}–{es(s['D_azar_p97_5'],3)}]",
                transform=ax.transAxes,ha='right',va='bottom',fontsize=7)
        ax.grid(alpha=.3,lw=.5)
    for ax in axs.flat[len(strata):]:
        ax.set_visible(False)
    for ax in axs.flat[max(0,len(strata)-nc):len(strata)]:  # ultimo panel visible de cada columna
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_xlabel('Distancia climática d al clima protegido más cercano',fontsize=8)
    for ax in axs[:,0]:
        ax.set_ylabel('Proporción del territorio con d ≤ x',fontsize=8)
    for ax in axs.flat:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:_corto(v))); ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:_corto(v)))
        ax.tick_params(labelsize=7)
    axs[0,0].legend(fontsize=7,frameon=False,loc='lower right',bbox_to_anchor=(1,.22))
    fig.suptitle(f'Cobertura climática de la red de conservación frente a {n_redes} redes al azar de igual forma y tamaño',x=.01,ha='left',fontsize=11)
    fig.savefig(dst/'curvas_cobertura.png',bbox_inches='tight'); plt.close(fig)
    log(f'Cobertura: {summary}')


def distancia_sistema(out,cap=100):
    """d_min(c): distancia exacta de cada celda al clima protegido mas semejante de la red (union de las referencias).

    Es el minimo de d_j(c) sobre las unidades; sirve para metricas sin mapas por unidad (p. ej., cond 50)."""
    dst=out/'mapas_distancia'/f'cond{cap}'; target=dst/'sistema_distancia_minima.tif'
    if target.exists():
        return
    national_model(out,cap)
    base=_lienzo(out); prep=out/'preparados'
    climate=np.load(out/'modelo'/f'clima_transformado_cond{cap}.npy')
    ref=np.zeros(len(climate),bool)
    for unit in base.inventory:
        if unit['estado']=='evaluable':
            ref[np.load(prep/'asociaciones'/f"{unit['anp_id']}.npz")['index']]=True
    tree=cKDTree(climate[ref],leafsize=32); d=np.empty(len(climate),np.float32)
    for s in range(0,len(climate),65536):
        d[s:s+65536]=tree.query(climate[s:s+65536],k=1,eps=0,workers=8)[0]
    dst.mkdir(parents=True,exist_ok=True)
    profile=dict(driver='GTiff',width=base.w,height=base.h,count=1,dtype='float32',nodata=-9999.0,crs=base.grid['crs'],
                 transform=base.aff,tiled=True,blockxsize=256,blockysize=256,compress='deflate',predictor=3)
    band=np.full(base.h*base.w,-9999,np.float32); band[base.flat]=d
    with rasterio.open(target,'w',**profile) as ds:
        ds.write(band.reshape(base.h,base.w),1)
    log(f'Distancia del sistema cond{cap}: {int(ref.sum())} referencias; mediana {float(np.median(d)):.3f}')


NORMALES_SENAMHI=(RECURSOS/r'SENAMHI_normales_OMM_1981-2010\WMO_Normals_Excel_Template-PERU.xls')
DEM=(RECURSOS/r'elevation\wc2.1_30s_elev.tif')
VARS_ESTACION=('bio01','bio02','bio04','bio05','bio06','bio12','bio15')  # las 7 de las 11 que salen de P, Tx y Tn mensuales


def _normales_senamhi(path=NORMALES_SENAMHI):
    """Normales 1981-2010 del SENAMHI (plantilla OMM, una hoja por estacion): P, Tx y Tn mensuales y anos con dato."""
    import xlrd
    def grados(s):
        g,m,sec,h=s.split('|')
        return (float(g)+float(m)/60+float(sec)/3600)*(-1 if h in 'SW' else 1)
    book=xlrd.open_workbook(str(path)); estaciones=[]
    for sh in book.sheets():
        if sh.name in ('Template','Example','Parameters','Calculation Methods'):
            continue
        cell=lambda r,c:sh.cell_value(r,c)
        e={'nombre':cell(6,1),'omm':int(cell(9,0)),'lat':grados(cell(9,1)),'lon':grados(cell(9,2)),'altitud_m':float(cell(9,3))}
        for r in range(sh.nrows-1):
            code=cell(r,1)
            if code in (1.0,3.0,4.0) and cell(r,3)!=98.0 and cell(r+1,3)==98.0:
                key={1.0:'P',3.0:'Tx',4.0:'Tn'}[code]
                e[key]=np.array([float(v) if v!='' else np.nan for v in sh.row_values(r,4,16)])
                e['anos_'+key]=np.array([float(v) if v!='' else np.nan for v in sh.row_values(r+1,4,16)])
        estaciones.append(e)
    return estaciones


def _bioclim(e):
    """Variables CHELSA a partir de normales mensuales. bio01 = media de (Tx+Tn)/2 (no hay Tmedia diaria);
    bio04 y bio15 con la convencion de biovars (sd muestral; bio15 = CV de P+1)."""
    tm=(e['Tx']+e['Tn'])/2
    return {'bio01':tm.mean(),'bio02':(e['Tx']-e['Tn']).mean(),'bio04':tm.std(ddof=1),'bio05':e['Tx'].max(),
            'bio06':e['Tn'].min(),'bio12':e['P'].sum(),'bio15':100*(e['P']+1).std(ddof=1)/(e['P']+1).mean()}


def estaciones(out,cap=100):
    """Contraste con las normales 1981-2010 del SENAMHI: CHELSA en la celda de cada estacion y analogia entre pares.

    Distancias con las 7 variables que dan las estaciones: misma estandarizacion nacional y submatriz de R_lambda
    (Mahalanobis marginal, por dimension). Los pares se resumen por clase de lectura de la d de 11 variables entre sus celdas.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    dst=out/'estaciones'; dst.mkdir(exist_ok=True)
    base=_lienzo(out); prep=out/'preparados'
    model=json.loads((out/'modelo'/f'modelo_cond{cap}.json').read_text(encoding='utf-8'))
    X=np.load(prep/'clima_fisico.npy',mmap_mode='r'); Z=np.load(out/'modelo'/f'clima_transformado_cond{cap}.npy',mmap_mode='r')
    S=[list(VARIABLES).index(v) for v in VARS_ESTACION]
    R=(1-model['lambda'])*np.asarray(model['R'])+model['lambda']*np.eye(P)
    W=np.linalg.cholesky(np.linalg.inv(R[np.ix_(S,S)]))  # d7 = |dz @ W| / sqrt(7)
    def z7(v):
        v=np.array(v,dtype=np.float64); v[VARS_ESTACION.index('bio12')]=np.log1p(v[VARS_ESTACION.index('bio12')])
        return (v-np.asarray(model['mu'])[S])/np.asarray(model['sd'])[S]@W/np.sqrt(len(S))
    rows_valid=np.full(base.h*base.w,-1,np.int64); rows_valid[base.flat]=np.arange(len(base.flat))
    with rasterio.open(DEM) as dem:
        elev=lambda lon,lat:float(next(dem.sample([(lon,lat)]))[0])
        filas=[]
        for e in _normales_senamhi():
            r,c=int((e['lat']-base.aff.f)/base.aff.e),int((e['lon']-base.aff.c)/base.aff.a)
            k=rows_valid[r*base.w+c] if 0<=r<base.h and 0<=c<base.w else -1
            if k<0:  # celda sin clima (costa): la celda valida mas cercana a <= 2 celdas
                cand=[(abs(dr)+abs(dc),rows_valid[(r+dr)*base.w+c+dc]) for dr in range(-2,3) for dc in range(-2,3)
                      if 0<=r+dr<base.h and 0<=c+dc<base.w and rows_valid[(r+dr)*base.w+c+dc]>=0]
                k=min(cand)[1] if cand else -1
            completa=all(k_ in e and np.isfinite(e[k_]).all() for k_ in ('P','Tx','Tn'))
            unidad=[u for u,g in base.shapes.items() if g.contains(shapely.Point(e['lon'],e['lat']))]
            fila={'estacion':e['nombre'],'omm':e['omm'],'lat':e['lat'],'lon':e['lon'],'altitud_m':e['altitud_m'],
                  'altitud_celda_m':elev(*base.aff*(int(base.flat[k])%base.w+.5,int(base.flat[k])//base.w+.5)) if k>=0 else None,'indice_celda':int(k),
                  'celda_desplazada':bool(k>=0 and rows_valid[r*base.w+c]<0),'completa':completa,
                  'anos_min':float(min(np.nanmin(e['anos_'+x]) for x in ('P','Tx','Tn') if 'anos_'+x in e)),
                  'unidad':'; '.join(unidad),
                  'excluida':'' if completa and k>=0 else 'coordenadas fuera del dominio (a > 2 celdas)' if k<0 else 'sin temperatura'}
            if completa and k>=0:
                obs=_bioclim(e); ch=dict(zip(VARIABLES,np.asarray(X[k],dtype=np.float64)))
                for v in VARS_ESTACION:
                    fila[f'{v}_obs']=float(obs[v]); fila[f'{v}_chelsa']=float(ch[v])
                fila['z_obs']=z7([obs[v] for v in VARS_ESTACION]); fila['z_chelsa']=z7([ch[v] for v in VARS_ESTACION])
            filas.append(fila)
    ok=[f for f in filas if 'z_obs' in f]
    errores={}
    for v in VARS_ESTACION:
        o=np.array([f[f'{v}_obs'] for f in ok]); m=np.array([f[f'{v}_chelsa'] for f in ok])
        err=np.log(m+1)-np.log(o+1) if v=='bio12' else m-o
        errores[v]={'sesgo':float(err.mean()),'error_absoluto_mediano':float(np.median(np.abs(err))),
                    'spearman':float(spearmanr(o,m)[0]),'unidad':'log(P+1)' if v=='bio12' else UNITS[v]}
    dz=np.array([f['altitud_m']-f['altitud_celda_m'] for f in ok]); e01=np.array([f['bio01_chelsa']-f['bio01_obs'] for f in ok])
    errores['bio01']['error_absoluto_mediano_corregido_altitud']=float(np.median(np.abs(e01-0.0065*dz)))  # gradiente 6,5 C/km
    pares=[]
    for a in range(len(ok)):
        for b in range(a+1,len(ok)):
            fa,fb=ok[a],ok[b]
            d11=float(np.linalg.norm(np.asarray(Z[fa['indice_celda']])-np.asarray(Z[fb['indice_celda']])))
            pares.append({'a':fa['estacion'],'b':fb['estacion'],'d11_chelsa':d11,
                          'd7_chelsa':float(np.linalg.norm(fa['z_chelsa']-fb['z_chelsa'])),'d7_obs':float(np.linalg.norm(fa['z_obs']-fb['z_obs'])),
                          'dif_bio01_obs':abs(fa['bio01_obs']-fb['bio01_obs']),'dif_logP_obs':abs(np.log1p(fa['bio12_obs'])-np.log1p(fb['bio12_obs'])),
                          'dif_bio01_chelsa':abs(fa['bio01_chelsa']-fb['bio01_chelsa']),
                          'dif_logP_chelsa':abs(np.log1p(fa['bio12_chelsa'])-np.log1p(fb['bio12_chelsa']))})
    d11,d7c,d7o=(np.array([p[k] for p in pares]) for k in ('d11_chelsa','d7_chelsa','d7_obs'))
    def acuerdo(x,y,corte):
        ax,ay=x<=corte,y<=corte
        return {'pares_analogos_x':int(ax.sum()),'pares_analogos_y':int(ay.sum()),'ambos':int((ax&ay).sum()),
                'coinciden_pct':float(100*(ax==ay).mean())}
    clases=[('d<=0.5',-1,.5),('0.5-1',.5,1),('>1',1,np.inf)]  # clases de lectura de d (11 variables) entre las celdas del par
    por_clase={}
    for nombre,lo,hi in clases:
        m=(d11>lo)&(d11<=hi)
        por_clase[nombre]={'n':int(m.sum()),
                           'mediana_dif_bio01_obs':float(np.median([p['dif_bio01_obs'] for p,x in zip(pares,m) if x])) if m.any() else None,
                           'mediana_dif_logP_obs':float(np.median([p['dif_logP_obs'] for p,x in zip(pares,m) if x])) if m.any() else None}
    resumen={'n_estaciones':len(filas),'n_completas':len(ok),'n_pares':len(pares),'errores_chelsa_menos_estacion':errores,
             'altitud_estacion_menos_celda_m':{'mediana':float(np.median(dz)),'mediana_abs':float(np.median(np.abs(dz)))},
             'excluidas':{f['estacion']:f['excluida'] for f in filas if f['excluida']},
             'spearman_d7_obs_vs_d7_chelsa':float(spearmanr(d7o,d7c)[0]),'mediana_d7_obs_sobre_d7_chelsa':float(np.median(d7o/d7c)),
             'spearman_dif_bio01_pares':float(spearmanr([p['dif_bio01_obs'] for p in pares],[p['dif_bio01_chelsa'] for p in pares])[0]),
             'spearman_dif_logP_pares':float(spearmanr([p['dif_logP_obs'] for p in pares],[p['dif_logP_chelsa'] for p in pares])[0]),'spearman_d7_obs_vs_d11_chelsa':float(spearmanr(d7o,d11)[0]),
             'acuerdo_0_2_d7_chelsa_vs_d7_obs':acuerdo(d7c,d7o,.2),'pares_por_clase_d11':por_clase,
             'fuente':'Normales climatologicas 1981-2010 del SENAMHI enviadas a la OMM (NOAA NCEI, normals-old/WMO/1981-2010/RA-III/Peru); '
                      'sha256 '+sha256(NORMALES_SENAMHI),
             'nota':'bio01 = media de (Tx+Tn)/2; d7 = Mahalanobis regularizada marginal en 7 variables, por dimension; '
                    'pares por clase de lectura de la d de 11 variables entre sus celdas.'}
    write_csv(dst/'estaciones.csv',[{k:v for k,v in f.items() if not k.startswith('z_')} for f in filas])
    write_csv(dst/'pares_estaciones.csv',pares); save_json(dst/'estaciones.json',resumen)
    fig,axs=plt.subplots(1,3,figsize=(13,4.4),dpi=150)
    for ax,v,lab in ((axs[0],'bio01','Temperatura media anual (°C)'),(axs[1],'bio12','Precipitación anual (mm)')):
        o=[f[f'{v}_obs'] for f in ok]; m=[f[f'{v}_chelsa'] for f in ok]
        ax.scatter(o,m,s=14,color='#1f1f1d'); lim=[min(o+m),max(o+m)]; ax.plot(lim,lim,color='#0096c7',lw=.8)
        if v=='bio12':
            ax.set_xscale('symlog',linthresh=10); ax.set_yscale('symlog',linthresh=10)
        ax.set_xlabel(f'Estación SENAMHI · {lab}',fontsize=8); ax.set_ylabel('CHELSA, celda de la estación',fontsize=8)
    for (nombre,lo,hi),color in zip(clases,('#a50f15','#fb6a4a','#bdbdb5')):
        m=(d11>lo)&(d11<=hi)
        axs[2].scatter(d7o[m],d7c[m],s=6,color=color,label=f'{nombre} (n = {int(m.sum())})')
    axs[2].legend(fontsize=7,frameon=False,title='d de 11 variables entre sus celdas',title_fontsize=7)
    axs[2].plot([0,d7o.max()],[0,d7o.max()],color='#0096c7',lw=.8)
    axs[2].set_xlabel('d entre estaciones, valores medidos (7 variables)',fontsize=8); axs[2].set_ylabel('d entre sus celdas CHELSA (7 variables)',fontsize=8)
    for ax in axs:
        ax.tick_params(labelsize=7)
    fig.suptitle(f"Contraste con {len(ok)} estaciones del SENAMHI (normales 1981–2010)",x=.01,ha='left',fontsize=10)
    fig.tight_layout(); fig.savefig(dst/'estaciones.png',bbox_inches='tight'); plt.close(fig)
    log(f"Estaciones: {len(ok)} completas de {len(filas)}; {len(pares)} pares; spearman d7 obs-chelsa {resumen['spearman_d7_obs_vs_d7_chelsa']:.2f}")
    return resumen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--fase',choices=['preparar','distancias','fichas','zonas','ecorregiones','mapas','sistema','cobertura','factibilidad','estaciones','prueba'],required=True)
    parser.add_argument('--unidades',nargs='*',help='Subconjunto (piloto)')
    parser.add_argument('--cond',type=int,default=100)
    parser.add_argument('--redes',type=int,default=999,help='Redes al azar para la cobertura')
    parser.add_argument('--diseno',choices=list(SEMILLA_DISENO),default='N0',help='Diseno de las redes nulas')
    parser.add_argument('--dst',type=Path,help='Fases cobertura y factibilidad: carpeta de salida (por defecto, dentro de --out)')
    parser.add_argument('--limitar-superposicion',action='store_true',help='Redes nulas: colocacion secuencial sin mas superposicion que la real')
    parser.add_argument('--sin-menores',type=float,default=0,help='Fase cobertura: excluye las areas de superficie oficial menor (km2) (robustez)')
    parser.add_argument('--mascara',type=Path,help='Carpeta de mascara_natural.py con natural_dominio.npz y resumen.json (N1, N2c)')
    parser.add_argument('--desde',type=Path,default=OUT,help='Fase prueba: carpeta con preparados/ y cobertura/ ya simulados')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.fase=='prueba':
        recalcular_prueba(args.desde,args.cond,args.out/'cobertura'/f'cond{args.cond}')
        return
    if shutil.disk_usage(args.out).free<20*2**30:
        raise ValueError('Se requieren al menos 20 GiB libres')
    with rasterio.Env(GDAL_CACHEMAX=64*1024*1024,GDAL_NUM_THREADS='1'):
        prepare(args.root,args.out)
        if args.fase=='fichas':
            fichas(args.out,args.cond); controles(args.out,args.cond)
        if args.fase=='distancias':
            distances(args.out,args.cond,args.unidades)
        if args.fase=='zonas':
            zonas(args.out)
        if args.fase=='ecorregiones':
            ecorregiones(args.out)
        if args.fase=='mapas':
            mapas_distancia(args.out,args.cond,args.unidades)
        if args.fase=='sistema':
            distancia_sistema(args.out,args.cond)
        if args.fase in ('cobertura','factibilidad'):
            natural=umbral=None
            if args.mascara and (args.fase=='factibilidad' or args.diseno!='N0'):  # pixeles naturales del dominio y umbral calibrado con la red real
                natural=np.load(args.mascara/'natural_dominio.npz')['natural']
                umbral=json.loads((args.mascara/'resumen.json').read_text(encoding='utf-8'))['umbral_aceptacion_F4']
            if args.fase=='factibilidad':
                factibilidad(args.out,args.dst,natural,umbral)
            else:
                cobertura(args.out,args.cond,args.redes,diseno=args.diseno,dst=args.dst,natural=natural,umbral=umbral,
                          limitar=args.limitar_superposicion,min_km2=args.sin_menores)
        if args.fase=='estaciones':
            estaciones(args.out,args.cond)


if __name__=='__main__':
    main()
