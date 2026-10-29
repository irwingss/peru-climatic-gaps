"""Sesgo de la precipitacion de CHELSA V2.1 frente a PISCOp v3.0 y correccion por factor de cambio mensual.

Fases:
  corregir  diagnostico a 0,1° (cociente CHELSA/PISCO anual y mensual), factores mensuales, bio12 y bio15 corregidos y
            una raiz CHELSA corregida (mismo contrato que la original) para correr ejecutar_investigacion.py con --root.
  comparar  original frente a corregido (dmin, F(0,1), F(0,2), vacios, Jaccard, prueba frente a las redes nulas) y criterio de equivalencia.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
RECURSOS = Path(__import__('os').environ.get('RECURSOS_GENERALES', Path.home()/'Documents'/'Recursos generales'))  # datos de entrada fuera del repositorio

import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt, map_coordinates

import ejecutar_investigacion as E
from descargar_chelsa import ROOT, sha256

DST = Path(__file__).resolve().parents[1]/'resultados_v7'/'02_sesgo_precipitacion'
PISCO = (RECURSOS/r'PISCO_SENAMIH\PISCOp_m.nc')
PISCO_MD5 = 'fbf4f19b5d537183a75d08615c404010'  # figshare, version 1
PISCO_AFF = Affine(.1, 0, -81.4, 0, -.1, 2.0)
PERIODO = 360  # bandas 1-360: enero de 1981 a diciembre de 2010, el periodo de CHELSA
HIPERARIDO = 1.  # mm/mes a 0,1°: por debajo, el factor queda acotado (registro aparte)
ACOTADO = (.1, 10.)
CUBIERTA_MIN = .25  # celda de 0,1° con < 25 % de pixeles CHELSA validos (astillas de costa) toma el factor vecino
CLAVE = ('Perú', 'Desierto costero del Pacífico', 'Serranía esteparia', 'Puna')  # estratos del criterio de equivalencia
EN = {'Perú': 'Peru', 'Bosque tropical amazónico': 'Tropical Amazon forest', 'Sabana de palmeras': 'Palm savanna',
      'Selva alta (yungas)': 'Yungas', 'Páramo': 'Páramo', 'Puna': 'Puna', 'Serranía esteparia': 'Steppe sierra',
      'Bosque seco ecuatorial': 'Equatorial dry forest', 'Bosque tropical del Pacífico': 'Pacific tropical forest',
      'Desierto costero del Pacífico': 'Pacific coastal desert'}
MESES = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')


def leer(path, band=1):
    with rasterio.open(path) as ds:
        a = ds.read(band).astype(np.float64)
        bad = a == ds.nodata
        a = a*ds.scales[band-1]+ds.offsets[band-1]
        a[bad] = np.nan
    return a


def bioclim_pr(pr):
    """bio12 y bio15 de CHELSA V2.1 a partir de 12 meses (verificado: suma; 100·de poblacional/media, sin +1)."""
    return pr.sum(0), 100*pr.std(0)/pr.mean(0)


def pisco_climatologia():
    import hashlib
    md5 = hashlib.md5(PISCO.read_bytes()).hexdigest()
    if md5 != PISCO_MD5:
        raise ValueError(f'PISCOp_m.nc no coincide con la version 1 de figshare: {md5}')
    with rasterio.open(PISCO) as ds:
        if ds.count != 540 or not np.allclose(list(ds.transform)[:6], list(PISCO_AFF)[:6]):
            raise ValueError('Cuadricula o bandas de PISCOp inesperadas')
        a = ds.read(list(range(1, PERIODO+1))).astype(np.float64)
        a[a < -1e30] = np.nan
    return a.reshape(30, 12, *a.shape[1:]).mean(0), md5  # NaN si falta algun mes


def corregir(dst=DST, ref=E.OUT):
    started = time.perf_counter()
    bio = json.loads((ROOT/'manifest_descarga.json').read_text(encoding='utf-8'))
    prm = json.loads((ROOT/'manifest_pr_mensual.json').read_text(encoding='utf-8'))
    if prm.get('estado') != 'completo':
        raise ValueError('Faltan meses de pr: python descargar_chelsa.py --pr-mensual')
    grid = bio['capas']['bio12']['grid']; aff = Affine(*grid['transform']); h, w = grid['height'], grid['width']
    pr = np.stack([leer(ROOT/'pr_mensual'/Path(prm['capas'][f'pr_{m:02d}']['ruta']).name) for m in range(1, 13)])
    b12, b15 = leer(ROOT/'peru_nativo'/Path(bio['capas']['bio12']['ruta']).name), leer(ROOT/'peru_nativo'/Path(bio['capas']['bio15']['ruta']).name)
    valid = np.isfinite(pr).all(0) & np.isfinite(b12) & np.isfinite(b15)
    r12, r15 = bioclim_pr(pr[:, valid])
    control = {'bio12_suma_menos_capa': [float(x) for x in np.percentile(r12-b12[valid], [0, 50, 100])],
               'bio15_max_abs': float(np.max(np.abs(r15-b15[valid]))), 'n_pixeles': int(valid.sum()),
               'formula': 'bio12 = suma de pr mensual (la capa la trunca a mm enteros); bio15 = 100·de(ddof=0)/media, sin +1'}
    if control['bio15_max_abs'] > .051 or not -1e-6 <= control['bio12_suma_menos_capa'][0] <= control['bio12_suma_menos_capa'][2] <= 1+1e-6:
        raise ValueError(f'No se reproduce bio12/bio15 de CHELSA: {control}')
    E.log(f'Control bio12/bio15: {control}')

    # Agregacion a 0,1° por promedio ponderado por superficie: cada pixel va a la celda PISCO de su centro.
    # el borde de la cuadricula CHELSA esta corrido 0,017 pixeles (~15 m) respecto de PISCO; se ignora.
    P, pisco_md5 = pisco_climatologia(); ph, pw = P.shape[1:]
    lat = aff.f+(np.arange(h)+.5)*aff.e; lon = aff.c+(np.arange(w)+.5)*aff.a
    prow = np.floor((PISCO_AFF.f-lat)/.1).astype(int); pcol = np.floor((lon-PISCO_AFF.c)/.1).astype(int)
    inside = (prow[:, None] >= 0) & (prow[:, None] < ph) & (pcol[None] >= 0) & (pcol[None] < pw)
    cell = np.where(inside, np.clip(prow, 0, ph-1)[:, None]*pw+np.clip(pcol, 0, pw-1)[None], -1)
    peso = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], (h, w))
    k = valid & (cell >= 0); idx = cell[k]; wk = peso[k]
    sw = np.bincount(idx, wk, ph*pw); n = np.bincount(idx, minlength=ph*pw)
    C = np.stack([np.bincount(idx, wk*pr[m][k], ph*pw) for m in range(12)])/np.where(sw > 0, sw, np.nan)
    C = C.reshape(12, ph, pw)
    cubierta = (n/144).reshape(ph, pw)  # 12 x 12 pixeles de 30" por celda de 0,1°
    celda_ok = np.isfinite(P).all(0) & (cubierta >= CUBIERTA_MIN)
    Ca, Pa = C.sum(0), P.sum(0)

    # Factor de cambio mensual, acotado; los meses hiperaridos se registran aparte.
    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.clip(P/C, *ACOTADO)
        acotado = celda_ok & ((P/C < ACOTADO[0]) | (P/C > ACOTADO[1]))
    hiper = celda_ok & ((C < HIPERARIDO) | (P < HIPERARIDO))
    logf = np.log(np.where(celda_ok, f, np.nan))
    near = distance_transform_edt(~celda_ok, return_distances=False, return_indices=True)
    logf = logf[:, near[0], near[1]]  # celdas sin dato: factor de la celda valida mas proxima
    # Interpolacion bilineal de log(factor) a 30"; solo el factor, nunca el predictor.
    rr = ((PISCO_AFF.f-lat)/.1-.5)[:, None]*np.ones((1, w)); cc = np.ones((h, 1))*((lon-PISCO_AFF.c)/.1-.5)[None]
    fac = np.stack([np.exp(map_coordinates(logf[m], [rr, cc], order=1, mode='nearest')) for m in range(12)])
    prc = pr*fac
    c12, c15 = np.full((h, w), np.nan), np.full((h, w), np.nan)
    c12[valid], c15[valid] = bioclim_pr(prc[:, valid])

    # Pixeles del dominio continental en celdas PISCO sin dato.
    dom = np.load(ref/'preparados'/'dominio.npz'); flat, area = dom['flat'], dom['area']
    cd = cell.reshape(-1)[flat]; sin = (cd < 0) | ~celda_ok.reshape(-1)[np.maximum(cd, 0)]
    eco = np.load(ref/'preparados'/'ecorregiones_minam.npz')['ecorregion']
    nombres = [e['nombre'] for e in json.loads((ref/'preparados'/'ecorregiones_minam.json').read_text(encoding='utf-8'))['ecorregiones']]
    # Cierre: CHELSA corregido re-agregado a 0,1° frente a PISCO (anual).
    Cc = (np.bincount(idx, wk*np.nansum(prc, 0)[k], ph*pw)/np.where(sw > 0, sw, np.nan)).reshape(ph, pw)
    cierre = np.log(Cc[celda_ok]/Pa[celda_ok])

    folder = dst/'factores'; folder.mkdir(parents=True, exist_ok=True)
    perfil = dict(driver='GTiff', width=pw, height=ph, crs='EPSG:4326', transform=PISCO_AFF, dtype='float32',
                  nodata=np.nan, compress='deflate')
    with rasterio.open(folder/'factor_mensual_0p1.tif', 'w', count=12, **perfil) as ds:
        ds.write(np.exp(logf).astype(np.float32))
        ds.update_tags(descripcion='PISCOp/CHELSA por mes (bandas 1-12), acotado a [0.1, 10]; celdas sin dato con el factor vecino')
    with rasterio.open(folder/'cociente_chelsa_pisco_0p1.tif', 'w', count=13, **perfil) as ds:
        q = np.where(celda_ok, np.concatenate([(Ca/Pa)[None], C/P]), np.nan)
        ds.write(q.astype(np.float32)); ds.update_tags(descripcion='CHELSA/PISCO 1981-2010; banda 1 anual, 2-13 enero a diciembre')
    with rasterio.open(folder/'celda_valida_0p1.tif', 'w', count=1, **{**perfil, 'dtype': 'uint8', 'nodata': None}) as ds:
        ds.write(celda_ok.astype(np.uint8)[None])

    # Raiz CHELSA corregida: mismas 11 capas y cartografia; bio12 y bio15 reemplazadas (float32, sin redondeo).
    raiz = dst/'chelsa_corregido'; (raiz/'peru_nativo').mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT/'cartografia', raiz/'cartografia', dirs_exist_ok=True)
    man = json.loads(json.dumps(bio))
    for v, layer in bio['capas'].items():
        src = ROOT/'peru_nativo'/Path(layer['ruta']).name
        if v not in ('bio12', 'bio15'):
            shutil.copyfile(src, raiz/'peru_nativo'/src.name)
            continue
        target = raiz/'peru_nativo'/src.name.replace('.tif', '_PISCOcorr.tif')
        with rasterio.open(src) as s:
            tags = s.tags()
        with rasterio.open(target, 'w', driver='GTiff', width=w, height=h, count=1, dtype='float32', nodata=-9999.,
                           crs='EPSG:4326', transform=aff, tiled=True, blockxsize=512, blockysize=512, compress='deflate', predictor=3) as ds:
            ds.write(np.where(valid, c12 if v == 'bio12' else c15, -9999.).astype(np.float32), 1)
            ds.update_tags(**tags, correccion='Factor de cambio mensual PISCOp v3.0 / CHELSA 1981-2010; sin redondeo')
        man['capas'][v] = {**layer, 'ruta': str(target), 'sha256': sha256(target), 'dtype': 'float32', 'nodata': -9999.,
                           'scale': 1., 'offset': 0., 'correccion': 'PISCOp v3.0'}
    man['producto'] += '; bio12 y bio15 corregidos con PISCOp v3.0'
    (raiz/'manifest_descarga.json').write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding='utf-8')
    E.save_json(dst/'manifiesto_recortes.json', {'pr_mensual': prm, 'pisco': {'ruta': str(PISCO), 'md5': pisco_md5, 'bandas': '1-360 (1981-2010)',
        'cita': 'Gutierrez, L., & Lavado-Casimiro, W. (2026). High-resolution grids of rainfall for Peru - PISCOp v3.0 dataset. '
                'figshare. https://doi.org/10.6084/m9.figshare.32411886.v1'},
        'raiz_corregida': {v: man['capas'][v] for v in ('bio12', 'bio15')}})

    # Diagnostico por ecorregion (celdas de 0,1° con su ecorregion mayoritaria por superficie del dominio).
    ok = valid.reshape(-1)[flat]  # siempre cierto en el dominio
    cdo = np.where(cd >= 0, cd, 0)
    tabla = np.zeros((ph*pw, len(nombres))); np.add.at(tabla, (cdo[ok & (cd >= 0)], eco[ok & (cd >= 0)]), area[ok & (cd >= 0)])
    eco_celda = np.where(tabla.sum(1) > 0, tabla.argmax(1), -1).reshape(ph, pw)
    lr_a = np.log(Ca/Pa); lr_m = np.log(C/P)
    usable = celda_ok & (Pa > 0) & (eco_celda >= 0)
    filas = []
    for nombre, m in [('Perú', usable)]+[(nm, usable & (eco_celda == i)) for i, nm in enumerate(nombres)]:
        x = lr_a[m]
        if not len(x):
            continue
        q = np.percentile(x, [10, 25, 50, 75, 90])
        fila = {'stratum': EN[nombre], 'cells_0p1': int(m.sum()), 'median_ratio': float(np.exp(q[2])), 'median_log_ratio': float(q[2]),
                'sd_log_ratio': float(x.std(ddof=1)) if len(x) > 1 else None, 'iqr_log_ratio': float(q[3]-q[1]),
                'p10_ratio': float(np.exp(q[0])), 'p90_ratio': float(np.exp(q[4])),
                'hyperarid_cell_months_pct': float(100*hiper[:, m].mean()), 'bounded_cell_months_pct': float(100*acotado[:, m].mean())}
        for j, mes in enumerate(MESES):
            y = lr_m[j][m]; y = y[np.isfinite(y)]
            fila[f'median_ratio_{mes}'] = float(np.exp(np.median(y))) if len(y) else None
        filas.append(fila)
    E.write_csv(dst/'cociente_por_ecorregion.csv', filas)
    estaciones = contraste_estaciones(aff, h, w, pr, prc, P)
    E.write_csv(dst/'estaciones_precipitacion.csv', estaciones)
    resumen = {'control_bio12_bio15': control, 'celdas_0p1_validas': int(celda_ok.sum()),
               'celdas_pisco_con_dato': int(np.isfinite(P).all(0).sum()),
               'meses_celda_hiperaridos': int(hiper.sum()), 'meses_celda_acotados': int(acotado.sum()),
               'meses_celda_validos': int(12*celda_ok.sum()),
               'pixeles_dominio_en_celda_sin_dato': int(sin.sum()), 'km2_dominio_en_celda_sin_dato': float(area[sin].sum()/1e6),
               'pixeles_dominio': int(len(flat)),
               'cierre_log_corr_sobre_pisco_anual': {'mediana': float(np.median(cierre)), 'mediana_abs': float(np.median(np.abs(cierre))),
                                                     'p95_abs': float(np.percentile(np.abs(cierre), 95))},
               'factor_30s_rango': [float(np.nanmin(fac[:, valid])), float(np.nanmax(fac[:, valid]))],
               'cambio_dominio': cambio(b12, b15, c12, c15, flat, eco, nombres, area),
               'estaciones': resumen_estaciones(estaciones),
               'segundos': time.perf_counter()-started}
    E.save_json(dst/'diagnostico.json', resumen)
    figuras_cociente(dst, Ca/Pa, C/P, celda_ok)
    E.log(f"Correccion lista en {resumen['segundos']:.0f} s; {resumen['pixeles_dominio_en_celda_sin_dato']} pixeles en celdas sin dato")
    return resumen


def cambio(b12, b15, c12, c15, flat, eco, nombres, area):
    """Cambio de bio12 (cociente) y bio15 (diferencia) en el dominio, ponderado por superficie, por ecorregion."""
    o12, o15, n12, n15 = (a.reshape(-1)[flat] for a in (b12, b15, c12, c15))
    filas = {}
    for nombre, m in [('Perú', np.ones(len(flat), bool))]+[(nm, eco == i) for i, nm in enumerate(nombres)]:
        q = lambda x: [float(v) for v in E.weighted_quantile(x[m], area[m], [.1, .5, .9])]
        filas[EN[nombre]] = {'bio12_corr_sobre_orig_p10_p50_p90': q(n12/o12), 'bio15_corr_menos_orig_p10_p50_p90': q(n15-o15)}
    return filas


def contraste_estaciones(aff, h, w, pr, prc, P):
    """Normales SENAMHI 1981-2010 (P mensual): CHELSA original, corregido y PISCO en la celda de cada estacion."""
    filas = []
    for e in E._normales_senamhi():
        if 'P' not in e or not np.isfinite(e['P']).all():
            continue
        r, c = int((e['lat']-aff.f)/aff.e), int((e['lon']-aff.c)/aff.a)
        pr_, pc_ = int((PISCO_AFF.f-e['lat'])/.1), int((e['lon']-PISCO_AFF.c)/.1)
        if not (0 <= r < h and 0 <= c < w) or not np.isfinite(pr[:, r, c]).all():
            continue
        fila = {'station': e['nombre'], 'lat': e['lat'], 'lon': e['lon'], 'altitude_m': e['altitud_m']}
        for clave, serie in (('obs', e['P']), ('chelsa', pr[:, r, c]), ('chelsa_corr', prc[:, r, c]), ('pisco', P[:, pr_, pc_])):
            b12, b15 = bioclim_pr(np.asarray(serie, np.float64)[:, None])
            fila[f'bio12_{clave}'], fila[f'bio15_{clave}'] = float(b12[0]), float(b15[0])
        filas.append(fila)
    return filas


def resumen_estaciones(filas):
    out = {'n': len(filas), 'nota': 'PISCO incorpora estas estaciones; no es un contraste independiente.'}
    for clave in ('chelsa', 'chelsa_corr', 'pisco'):
        e12 = [np.log1p(f[f'bio12_{clave}'])-np.log1p(f['bio12_obs']) for f in filas if np.isfinite(f[f'bio12_{clave}'])]
        e15 = [f[f'bio15_{clave}']-f['bio15_obs'] for f in filas if np.isfinite(f[f'bio15_{clave}'])]
        out[clave] = {'bio12_sesgo_log1p': float(np.mean(e12)), 'bio12_error_abs_mediano_log1p': float(np.median(np.abs(e12))),
                      'bio12_factor_medio': float(np.exp(np.mean(e12))),
                      'bio15_sesgo': float(np.mean(e15)), 'bio15_error_abs_mediano': float(np.median(np.abs(e15)))}
    return out


def figuras_cociente(dst, anual, mensual, ok):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    ext = (PISCO_AFF.c, PISCO_AFF.c+mensual.shape[2]*.1, PISCO_AFF.f-mensual.shape[1]*.1, PISCO_AFF.f)
    norm = LogNorm(.2, 5); cmap = plt.get_cmap('RdBu_r').copy(); cmap.set_bad('#e6e6e2')
    fig, axs = plt.subplots(3, 5, figsize=(12, 12), dpi=150)
    paneles = [('Annual', anual)]+list(zip(MESES, mensual))
    for ax, (t, a) in zip(axs.flat, paneles):
        im = ax.imshow(np.where(ok, a, np.nan), cmap=cmap, norm=norm, extent=ext, interpolation='nearest')
        ax.set_title(t, fontsize=9, loc='left'); ax.set_aspect(1/np.cos(np.deg2rad(-9.2))); ax.set_axis_off()
        ax.set_xlim(-81.5, -68.5); ax.set_ylim(-18.5, .2)
    for ax in axs.flat[len(paneles):]:
        ax.set_visible(False)
    cb = fig.colorbar(im, cax=fig.add_axes([.62, .14, .012, .18]))
    cb.minorticks_off(); cb.set_ticks([.2, .5, 1, 2, 5]); cb.set_ticklabels(['0.2', '0.5', '1', '2', '5'])
    cb.set_label('CHELSA / PISCOp v3.0 (1981–2010)', fontsize=8)
    fig.suptitle('Ratio of CHELSA V2.1 to PISCOp v3.0 precipitation at 0.1°', x=.01, ha='left', fontsize=11)
    fig.savefig(dst/'cociente_chelsa_pisco.png', bbox_inches='tight'); plt.close(fig)


def comparar(dst=DST, cap=100):
    """Original frente a corregido con el mismo pipeline, la misma muestra de teselas y las mismas redes N0 (semilla SEED+7)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    a, b = dst/'original', dst/'corregido'
    prep = a/'preparados'
    area = np.load(prep/'dominio.npz')['area']; free = np.maximum(area-np.load(prep/'superficie_union.npy'), 0)
    eco, nombres = E.estratos_muestreo(prep)
    if not np.array_equal(np.load(prep/'dominio.npz')['flat'], np.load(b/'preparados'/'dominio.npz')['flat']):
        raise ValueError('Los dominios original y corregido no coinciden')
    flat = np.load(prep/'dominio.npz')['flat']
    def dmin(o):
        with rasterio.open(o/'mapas_distancia'/f'cond{cap}'/'sistema_distancia_minima.tif') as ds:
            return ds.read(1).reshape(-1)[flat].astype(np.float64)
    d = {'orig': dmin(a), 'corr': dmin(b)}
    p = {k: {r['estrato']: r for r in csv.DictReader(open(o/'cobertura'/f'cond{cap}'/'p_mediana.csv', encoding='utf-8-sig'))}
         for k, o in (('orig', a), ('corr', b))}
    filas = []
    for nombre, m in [('Perú', np.ones(len(free), bool))]+[(nm, eco == i) for i, nm in enumerate(nombres)]:
        f = free[m]; fila = {'stratum': EN[nombre]}
        gaps = {}
        for k in ('orig', 'corr'):
            x = d[k][m]; r = p[k][nombre]
            gaps[k] = x > .2
            fila.update({f'F0.1_{k}': 100*f[x <= .1].sum()/f.sum(), f'F0.2_{k}': 100*f[x <= .2].sum()/f.sum(),
                         f'gaps_km2_{k}': f[gaps[k]].sum()/1e6, f'median_dmin_{k}': float(E.weighted_quantile(x, f, [.5])[0]),
                         f'D_real_{k}': float(r['D_real']), f'D_null_median_{k}': float(r['D_azar_mediana']),
                         f'p_two_sided_{k}': float(r['p_bilateral']), f'p_holm_{k}': float(r['p_holm']),
                         f'SES_{k}': float(r['ses']), f'direction_{k}': r['direccion']})
        fila['delta_F0.2_pp'] = fila['F0.2_corr']-fila['F0.2_orig']
        union = f[gaps['orig'] | gaps['corr']].sum()
        fila['jaccard_gaps'] = float(f[gaps['orig'] & gaps['corr']].sum()/union) if union > 0 else None
        fila['_clave'] = nombre
        filas.append(fila)
    reglas = {}
    for fila in filas:
        if fila['_clave'] in CLAVE:
            reglas[fila['stratum']] = {'misma_direccion_y_significancia': fila['direction_orig'] == fila['direction_corr'],
                                       'jaccard': fila['jaccard_gaps']}
    nac = filas[0]
    cumple = (all(r['misma_direccion_y_significancia'] for r in reglas.values())
              and all(r['jaccard'] is not None and r['jaccard'] >= .7 for k, r in reglas.items() if k != 'Peru')
              and abs(nac['delta_F0.2_pp']) < 5)
    decision = {'estratos': reglas, 'delta_F0.2_nacional_pp': nac['delta_F0.2_pp'], 'jaccard_nacional': nac['jaccard_gaps'],
                'conclusiones_no_cambian': bool(cumple),
                'decision': 'CHELSA original (correccion como robustez)' if cumple else 'CHELSA corregido con PISCO'}
    for fila in filas:
        del fila['_clave']
    E.write_csv(dst/'comparacion_original_corregido.csv', filas)
    E.save_json(dst/'decision_D2.json', decision)
    # Mapas de diferencias de dmin (corregido - original) y cambio de vacios.
    lien = E._lienzo(a)
    delta = (d['corr']-d['orig']).astype(np.float32)
    perfil = dict(driver='GTiff', width=lien.w, height=lien.h, count=1, dtype='float32', nodata=-9999., crs=lien.grid['crs'],
                  transform=lien.aff, tiled=True, blockxsize=256, blockysize=256, compress='deflate', predictor=3)
    band = np.full(lien.h*lien.w, -9999, np.float32); band[lien.flat] = delta
    with rasterio.open(dst/'dmin_corregido_menos_original.tif', 'w', **perfil) as ds:
        ds.write(band.reshape(lien.h, lien.w), 1)
    clase = np.where(d['orig'] > .2, np.where(d['corr'] > .2, 3, 2), np.where(d['corr'] > .2, 1, 0)).astype(np.float32)
    band = np.full(lien.h*lien.w, -1, np.int16); band[lien.flat] = clase
    with rasterio.open(dst/'vacios_cambio.tif', 'w', **{**perfil, 'dtype': 'int16', 'nodata': -1, 'predictor': 2}) as ds:
        ds.write(band.reshape(lien.h, lien.w), 1)
        ds.update_tags(clases='0 representado en ambos; 1 vacio solo corregido; 2 vacio solo original; 3 vacio en ambos')
    fig, axs = plt.subplots(1, 2, figsize=(13, 8.6), dpi=150)
    cmap = plt.get_cmap('PuOr_r', 10); norm = BoundaryNorm([-.5, -.2, -.1, -.05, -.02, 0, .02, .05, .1, .2, .5], 10)
    cmap.set_bad((0, 0, 0, 0))
    im = E._panel(axs[0], lien, delta, cmap, norm)
    fig.colorbar(im, ax=axs[0], shrink=.6).set_label('dmin corrected − original', fontsize=8)
    axs[0].set_title('Change in dmin after PISCO correction', fontsize=10, loc='left')
    cm2 = ListedColormap(['#f2f2ee', '#d7301f', '#2b8cbe', '#636363']); cm2.set_bad((0, 0, 0, 0))
    E._panel(axs[1], lien, clase, cm2, BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4))
    from matplotlib.patches import Patch
    axs[1].legend(handles=[Patch(color=c, label=t) for c, t in zip(cm2.colors, ('Represented in both', 'Gap only when corrected',
                  'Gap only in original', 'Gap in both'))], fontsize=7, frameon=False, loc='lower left')
    axs[1].set_title('Climatic gaps (dmin > 0.2)', fontsize=10, loc='left')
    fig.savefig(dst/'dmin_diferencias.png', bbox_inches='tight'); plt.close(fig)
    E.log(f'Decision: {decision}')
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fase', choices=['corregir', 'comparar'], required=True)
    parser.add_argument('--out', type=Path, default=DST)
    parser.add_argument('--cond', type=int, default=100)
    args = parser.parse_args()
    corregir(args.out) if args.fase == 'corregir' else comparar(args.out, args.cond)


if __name__ == '__main__':
    main()
