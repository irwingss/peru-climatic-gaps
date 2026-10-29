"""Numbers for the text and tables of the manuscript (continental and maritime domains), read from stored results.

Writes insumos_figuras/cifras.json. Run: python cifras_resultados.py
"""
import csv, json, sys
from pathlib import Path
RECURSOS = Path(__import__('os').environ.get('RECURSOS_GENERALES', Path.home()/'Documents'/'Recursos generales'))  # datos de entrada fuera del repositorio
import numpy as np
import rasterio

RAIZ = next((p for p in Path(__file__).resolve().parents if (p / 'resultados_v7').is_dir()), Path(__file__).resolve().parents[1])  # primera carpeta con los resultados (raíz del repositorio o del proyecto)
TIERRA = RAIZ / 'resultados_climaticos/20261011'
MAR = RAIZ / 'resultados_marinos/20261011'
# CHELSA is used as distributed (the PISCOp correction is a robustness test), so the inputs above are byte-identical to
# resultados_v7/02_sesgo_precipitacion/original; null networks (N2c), robustness and expansion come from resultados_v7.
RES = RAIZ / 'resultados_v7'
NULAS_T = RES / '04_redes_nulas/continental_superposicion_limitada/N2c'
NULAS_M = {c: RES / f'04_redes_nulas/{c}/N2c' for c in ('superficie', 'fondo')}
ROB = RES / '07_final'
OUT = Path(__file__).parent / 'insumos_figuras/cifras.json'
sys.path.insert(0, str(RAIZ / 'clima_local'))
from ejecutar_investigacion import ECO_MINAM, weighted_quantile  # noqa: E402

ECO = [c for c, _, _ in ECO_MINAM]  # MINAM (2017) continental ecoregions, pipeline order (value in ecorregiones_minam.npz)
CLASES = (.1, .2)          # cumulative reading classes reported in the tables (very similar or closer; similar or closer)
MENOS = .95                # least-represented climates: the 5 % of the unprotected surface farthest from any protected climate
DEP = {'AMAZONAS': 'Amazonas', 'ANCASH': 'Áncash', 'APURIMAC': 'Apurímac', 'AREQUIPA': 'Arequipa', 'AYACUCHO': 'Ayacucho',
       'CAJAMARCA': 'Cajamarca', 'CALLAO': 'Callao', 'CUSCO': 'Cusco', 'HUANCAVELICA': 'Huancavelica', 'HUANUCO': 'Huánuco',
       'ICA': 'Ica', 'JUNIN': 'Junín', 'LA LIBERTAD': 'La Libertad', 'LAMBAYEQUE': 'Lambayeque', 'LIMA': 'Lima',
       'LORETO': 'Loreto', 'MADRE DE DIOS': 'Madre de Dios', 'MOQUEGUA': 'Moquegua', 'PASCO': 'Pasco', 'PIURA': 'Piura',
       'PUNO': 'Puno', 'SAN MARTIN': 'San Martín', 'TACNA': 'Tacna', 'TUMBES': 'Tumbes', 'UCAYALI': 'Ucayali'}


def wq(x, w, q):
    return float(weighted_quantile(np.asarray(x, np.float64), np.asarray(w, np.float64), [q])[0])


def leer(base, path):
    flat = np.load(base / 'preparados/dominio.npz')['flat']
    with rasterio.open(path) as ds:
        return ds.read(1).reshape(-1)[flat].astype(np.float64)


class Dominio:
    def __init__(self, base, estrato_de):
        prep = base / 'preparados'
        self.base = base
        self.area = np.load(prep / 'dominio.npz')['area']
        self.union = np.load(prep / 'superficie_union.npy')
        self.free = np.maximum(self.area - self.union, 0)
        self.zonas = np.load(prep / 'zonas.npz')
        self.estrato = estrato_de(self.zonas.get('ecorregion'))  # continental zonas.npz has no ecoregion (MINAM file)
        self.inv = json.loads((prep / 'inventario.json').read_text(encoding='utf-8'))
        self.dmin = leer(base, base / 'mapas_distancia/cond100/sistema_distancia_minima.tif')
        self.Z = np.load(base / 'modelo/clima_transformado_cond100.npy', mmap_mode='r')
        g = json.loads((prep / 'preparacion.json').read_text(encoding='utf-8'))['grid']
        self.h, self.w, self.aff = g['height'], g['width'], g['transform']
        self.flat = np.load(prep / 'dominio.npz')['flat']

    def representacion(self, m):
        """Unprotected surface of mask m: share within each cumulative class and median dmin."""
        f = self.free[m]; d = self.dmin[m]
        r = {'km2_total': float(self.area[m].sum() / 1e6), 'km2_protegido': float(self.union[m].sum() / 1e6),
             'km2_no_protegido': float(f.sum() / 1e6), 'pct_protegido': float(100 * self.union[m].sum() / self.area[m].sum()),
             'mediana_dmin': wq(d, f, .5) if f.sum() > 0 else None}
        for c in CLASES:
            r[f'pct_d{c}'] = float(100 * f[d <= c].sum() / f.sum()) if f.sum() > 0 else None
        return r

    def superficies(self, uid):
        a = np.load(self.base / f'preparados/asociaciones/{uid}.npz')
        d = np.load(self.base / f'distancias/cond100/{uid}.npz')['distancia']
        own = np.zeros(len(self.area)); own[a['index']] = a['area']
        fuera = np.maximum(self.area - own, 0)
        return {f'km2_d{c}': float(fuera[d <= c].sum() / 1e6) for c in CLASES}, a

    def escala(self, lags):
        """Median d between pixels k columns apart along the rows (km from the cell width at each latitude)."""
        G = np.full(self.h * self.w, -1, np.int64); G[self.flat] = np.arange(len(self.flat)); G = G.reshape(self.h, self.w)
        lat = self.aff[5] + self.aff[4] * (np.arange(self.h) + .5)
        ancho = abs(self.aff[0]) * 111.32 * np.cos(np.radians(lat))
        Z = np.asarray(self.Z)
        filas = []
        for k in lags:
            a, b = G[:, :-k], G[:, k:]; m = (a >= 0) & (b >= 0)
            d = np.linalg.norm(Z[a[m]] - Z[b[m]], axis=1)
            km = (ancho[:, None] * k * np.ones_like(a))[m]
            filas.append({'columnas': int(k), 'km': float(np.median(km)), 'd': float(np.median(d)), 'pares': int(m.sum())})
        rng = np.random.default_rng(20261011)  # same uniform draw of 20 000 pairs as the stored reading anchors
        i, j = rng.integers(len(Z), size=(2, 20000))
        km, dd = np.log([f['km'] for f in filas]), [f['d'] for f in filas]
        k = next(n for n, v in enumerate(dd) if v >= .2)
        cruce = float(np.exp(km[k - 1] + (km[k] - km[k - 1]) * (.2 - dd[k - 1]) / (dd[k] - dd[k - 1])))
        return {'curva': filas, 'km_d0.2': cruce, 'azar': float(np.median(np.linalg.norm(Z[i] - Z[j], axis=1)))}

    def nulas(self, carpeta, estratos):
        """p-median (area-weighted mean dmin) of the real network vs the null networks, by stratum."""
        z = np.load(carpeta / 'redes_aleatorias.npz')
        s, w = z['muestra'], z['peso']
        res = {}
        for nombre, m in estratos(self.estrato[s]):
            ww = w[m]
            real = float(np.average(z['real_dmin'][m], weights=ww))
            azar = np.array([np.average(x[m], weights=ww) for x in z['azar_dmin']])
            cob = {}
            for c in CLASES:
                cr = float(100 * ww[z['real_dmin'][m] <= c].sum() / ww.sum())
                ca = np.array([100 * ww[x[m] <= c].sum() / ww.sum() for x in z['azar_dmin']])
                cob[f'd{c}'] = {'real': cr, 'azar_mediana': float(np.median(ca)), 'azar_p2_5': float(np.quantile(ca, .025)),
                                'azar_p97_5': float(np.quantile(ca, .975))}
            res[nombre] = {'D_real': real, 'D_azar_mediana': float(np.median(azar)), 'D_azar_p2_5': float(np.quantile(azar, .025)),
                           'D_azar_p97_5': float(np.quantile(azar, .975)), 'n_azar_menor_igual': int((azar <= real).sum()),
                           'n_redes': int(len(azar)), 'cobertura': cob}
        # test against the null networks (two-sided p, SES, Holm, direction) from p_mediana.csv; same strata in the same order
        prueba = list(csv.DictReader(open(carpeta / 'p_mediana.csv', encoding='utf-8-sig')))
        for (nombre, r), fila in zip(res.items(), prueba):
            assert abs(float(fila['D_real']) - r['D_real']) < 1e-9, (nombre, fila['estrato'])
            r.update({k: (float(fila[k]) if fila[k] not in ('', 'None') else None) for k in ('p_bilateral', 'ses', 'p_holm')},
                     direccion=fila['direccion'], redes_mayor_o_igual=int(fila['redes_mayor_o_igual']))
        return res


CLASES_LECTURA = [('similar', -1, .2), ('moderado', .2, .3), ('debil', .3, .5), ('poco_similar', .5, 1), ('no_analogo', 1, 99),
                  ('mas_alla_similar', .2, 99)]
DEM = (RECURSOS/r'elevation\wc2.1_30s_elev.tif')  # WorldClim 2.1, 30 arc s


def elevacion(D):
    """Elevation (m) of each continental pixel centre, nearest WorldClim 2.1 cell."""
    a = D.aff
    lon = a[2] + a[0] * (D.flat % D.w + .5); lat = a[5] + a[4] * (D.flat // D.w + .5)
    with rasterio.open(DEM) as ds:
        t = ds.transform
        r = np.floor((lat - t.f) / t.e).astype(int); c = np.floor((lon - t.c) / t.a).astype(int)
        win = rasterio.windows.Window(c.min(), r.min(), c.max() - c.min() + 1, r.max() - r.min() + 1)
        z = ds.read(1, window=win).astype(np.float64)
        z[z == ds.nodata] = np.nan
    return z[r - r.min(), c - c.min()]


def por_clase(dmin, f, variables, grupos=None, nombres_grupo=None, extra=None):
    """Unprotected surface of each reading class of dmin: area, share, medians of the variables, groups and extra counts."""
    res = {}
    for nombre, lo, hi in CLASES_LECTURA:
        m = (dmin > lo) & (dmin <= hi) & (f > 0)
        r = {'km2': float(f[m].sum() / 1e6), 'pct': float(100 * f[m].sum() / f.sum())}
        for v, x in variables.items():
            ok = m & np.isfinite(x)
            r[v] = [wq(x[ok], f[ok], q) for q in (.1, .25, .5, .75, .9)]  # p10, p25, median, p75, p90
        if grupos is not None:
            r['grupos'] = {nombres_grupo[i]: float(100 * f[m & (grupos == i)].sum() / f[m].sum()) for i in range(len(nombres_grupo))}
        if extra:
            for k, (codigo, nombres) in extra.items():
                tot = np.bincount(codigo[m], weights=f[m], minlength=len(nombres))
                orden = np.argsort(-tot)
                r[k] = [[nombres[i], float(tot[i] / 1e6), float(100 * tot[i] / tot.sum())] for i in orden[:6] if tot[i] > 0]
        res[nombre] = r
    return res


def tramo_bajo_banda(curvas, estrato):
    """Smallest and largest x at which the real curve lies below the 2.5 % null quantile."""
    rows = [r for r in curvas if r['estrato'] == estrato]
    x = [float(r['d']) for r in rows if float(r['real']) < float(r['azar_p2_5']) and float(r['d']) > 0]
    return [min(x), max(x)] if x else None


def minam(base):
    """MINAM (2017) ecoregion of each pixel (stratum of the null-network sample and of the report)."""
    return np.load(base / 'preparados/ecorregiones_minam.npz')['ecorregion'].astype(np.int64)


def continental():
    D = Dominio(TIERRA, lambda _: minam(TIERRA))
    X = np.load(TIERRA / 'preparados/clima_fisico.npy', mmap_mode='r')
    out = {'peru': D.representacion(np.ones(len(D.area), bool))}
    out['ecorregiones'] = {e: D.representacion(D.estrato == i) for i, e in enumerate(ECO)}
    # T and P medians of the protected and unprotected surfaces (an ecoregion without protected surface has no protected median)
    for k, m in [('peru', np.ones(len(D.area), bool))] + [(e, D.estrato == i) for i, e in enumerate(ECO)]:
        dst = out['peru'] if k == 'peru' else out['ecorregiones'][k]
        for v, j in (('bio01', 0), ('bio12', 5)):
            dst[f'{v}_prot'] = wq(X[m, j], D.union[m], .5) if D.union[m].sum() > 0 else None
            dst[f'{v}_no_prot'] = wq(X[m, j], D.free[m], .5)
    # classes of the unprotected surface beyond 0.2
    f, d = D.free, D.dmin
    out['clases_no_protegido'] = {'moderado': float(100 * f[(d > .2) & (d <= .3)].sum() / f.sum()),
                                  'debil': float(100 * f[(d > .3) & (d <= .5)].sum() / f.sum()),
                                  'poco_similar': float(100 * f[(d > .5) & (d <= 1)].sum() / f.sum()),
                                  'km2_poco_similar': float(f[(d > .5) & (d <= 1)].sum() / 1e6),
                                  'km2_no_analogo': float(f[d > 1].sum() / 1e6)}
    des = D.estrato == ECO.index('DCP')
    ps = (d > .5) & (d <= 1)
    out['clases_no_protegido']['pct_poco_similar_en_desierto'] = float(100 * f[ps & des].sum() / f[ps].sum())
    out['clases_no_protegido']['pct_desierto_poco_similar'] = float(100 * f[ps & des].sum() / f[des].sum())
    # dmin over the whole territory (protected = 0)
    out['dmin_territorio'] = {f'p{int(q * 100)}': wq(d, D.area, q) for q in (.5, .75, .9, .95, .99)}
    # least-represented climates
    umbral = wq(d, f, MENOS); m = d >= umbral
    out['menos_representados'] = {
        'cuantil': MENOS, 'dmin_desde': umbral, 'km2': float(f[m].sum() / 1e6),
        'pct_por_ecorregion': {e: float(100 * f[m & (D.estrato == i)].sum() / f[m].sum()) for i, e in enumerate(ECO)},
        'km2_por_ecorregion': {e: float(f[m & (D.estrato == i)].sum() / 1e6) for i, e in enumerate(ECO)},
        'clima': {v: {'menos': wq(X[m, j], f[m], .5), 'no_prot': wq(X[:, j], f, .5)}
                  for v, j in (('bio12', 5), ('bio15', 6), ('hursmean', 7), ('bio01', 0), ('bio02', 1))}}
    # departments
    nombres = json.loads((TIERRA / 'preparados/zonas.json').read_text(encoding='utf-8'))['departamentos']
    dep = D.zonas['departamento']; deps = []
    for k, n in nombres.items():
        mk = dep == int(k)
        if not mk.any():
            continue
        r = D.representacion(mk); r['departamento'] = DEP[n]
        r['km2_menos_representado'] = float(f[mk & m].sum() / 1e6)
        deps.append(r)
    deps.sort(key=lambda r: r['pct_d0.2'])
    out['departamentos'] = deps
    # unprotected climates by reading class of dmin: where they are and what they are like
    nom_dep = [''] * (max(int(k) for k in nombres) + 1)
    for k, v in nombres.items():
        nom_dep[int(k)] = DEP[v]
    variables = {v: np.asarray(X[:, j], np.float64) for v, j in (('bio01', 0), ('bio02', 1), ('bio04', 2), ('bio12', 5), ('bio15', 6),
                                                                 ('hursmean', 7), ('vpdmean', 8), ('petmean', 10))}
    variables['elevacion'] = elevacion(D)
    out['clases'] = por_clase(d, f, variables, D.estrato, ECO, {'departamentos': (dep, nom_dep)})
    # protected areas: dominant ecoregion and the share of each one
    areas = []
    for u in D.inv:
        if u['estado'] != 'evaluable':
            continue
        s, a = D.superficies(u['anp_id'])
        w = np.bincount(D.estrato[a['index']], weights=a['area'], minlength=len(ECO))
        areas.append({'anp_id': u['anp_id'], 'nombre': u['nombre'], 'categoria': u['categoria'], 'modalidad': u['modalidad'],
                      'km2': float(a['area'].sum() / 1e6), 'ecorregion': ECO[int(w.argmax())],
                      'pct_ecorregiones': {ECO[i]: float(100 * w[i] / w.sum()) for i in np.flatnonzero(w)}, **s})
    assert all(a['km2_d0.1'] <= a['km2_d0.2'] for a in areas), 'cumulative classes must be nested'
    out['areas'] = areas

    def resumen(sel):
        r = {'n': len(sel), 'mediana_km2': float(np.median([a['km2'] for a in sel]))}
        for c in CLASES:
            v = [a[f'km2_d{c}'] for a in sel]
            r[f'd{c}'] = [float(np.quantile(v, q)) for q in (.25, .5, .75)]
        return r
    out['areas_resumen'] = {'modalidad': {k: resumen([a for a in areas if a['modalidad'] == k]) for k in ('NACIONAL', 'REGIONAL', 'PRIVADA')},
                            'ecorregion': {e: resumen([a for a in areas if a['ecorregion'] == e]) for e in ECO
                                           if any(a['ecorregion'] == e for a in areas)},
                            'todas': resumen(areas)}
    for c in CLASES:
        v = [a[f'km2_d{c}'] for a in areas]
        out['areas_resumen']['todas'][f'rango_d{c}'] = [min(v), max(v)]
    out['areas_top'] = sorted(areas, key=lambda a: -a['km2_d0.2'])[:8]
    # null networks
    def estratos(e):
        return [('Perú', np.ones(len(e), bool))] + [(c, e == i) for i, c in enumerate(ECO)]
    out['nulas'] = D.nulas(NULAS_T, estratos)
    curvas = list(csv.DictReader(open(NULAS_T / 'curvas_cobertura.csv', encoding='utf-8-sig')))
    nombre_curva = {'Perú': 'Perú', **{c: n for c, n, _ in ECO_MINAM}}  # strata names in curvas_cobertura.csv
    out['tramo_bajo_banda'] = {k: tramo_bajo_banda(curvas, v) for k, v in nombre_curva.items()}
    out['nulas_sin_menores_1km2'] = D.nulas(ROB / 'sin_menores_1km2/N2c', estratos)  # robustness: without the areas < 1 km2
    c50 = ROB / 'cond50/N2c/redes_aleatorias.npz'
    if c50.exists():
        out['nulas_cond50'] = D.nulas(c50.parent, estratos)
        d50 = leer(TIERRA, RES / '02_sesgo_precipitacion/original/mapas_distancia/cond50/sistema_distancia_minima.tif')  # d_min with kappa <= 50
        out['cond50'] = {'mediana_dmin': wq(d50, f, .5), **{f'pct_d{c}': float(100 * f[d50 <= c].sum() / f.sum()) for c in CLASES},
                         'por_ecorregion': {e: {f'pct_d{c}': float(100 * f[(d50 <= c) & (D.estrato == i)].sum() / f[D.estrato == i].sum())
                                                for c in CLASES} for i, e in enumerate(ECO)}}
        out['cond100_por_ecorregion'] = {e: {f'pct_d{c}': out['ecorregiones'][e][f'pct_d{c}'] for c in CLASES} for e in ECO}
    out['escala'] = D.escala([1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500])
    # stations: measured differences by reading class of the 11-variable distance between their pixels
    pares = list(csv.DictReader(open(TIERRA / 'estaciones/pares_estaciones.csv', encoding='utf-8-sig')))
    d11 = np.array([float(p['d11_chelsa']) for p in pares])
    dT = np.array([float(p['dif_bio01_obs']) for p in pares]); dP = np.exp(np.array([float(p['dif_logP_obs']) for p in pares]))
    out['estaciones'] = {}
    for nombre, lo, hi in (('d<=0.5', -1, .5), ('0.5-1', .5, 1), ('>1', 1, 99)):
        mk = (d11 > lo) & (d11 <= hi)
        out['estaciones'][nombre] = {'n': int(mk.sum()), 'dT': float(np.median(dT[mk])) if mk.any() else None,
                                     'factorP': float(np.median(dP[mk])) if mk.any() else None}
    return out


def maritimo():
    out = {}
    for capa in ('superficie', 'fondo'):
        # sea surface: the two MINAM marine ecoregions; seafloor: depth strata (zonas.npz)
        fuente = 'ecorregiones_minam' if (MAR / capa / 'preparados/ecorregiones_minam.json').exists() else 'zonas'
        D = Dominio(MAR / capa, (lambda _, b=MAR / capa: minam(b)) if fuente == 'ecorregiones_minam' else (lambda e: np.asarray(e, np.int64)))
        nombres = [z['codigo'] for z in json.loads((MAR / capa / f'preparados/{fuente}.json').read_text(encoding='utf-8'))['ecorregiones']]
        r = {'todo': D.representacion(np.ones(len(D.area), bool))}
        r['estratos'] = {n: D.representacion(D.estrato == i) for i, n in enumerate(nombres)} if len(nombres) > 1 else {}
        f, d = D.free, D.dmin
        r['km2_0.2_1'] = float(f[(d > .2) & (d <= 1)].sum() / 1e6); r['km2_mayor_1'] = float(f[d > 1].sum() / 1e6)
        Xm = np.load(MAR / capa / 'preparados/clima_fisico.npy')
        vnames = json.loads((MAR / capa / 'modelo/modelo_cond100.json').read_text(encoding='utf-8'))['variables']
        variables = {v: Xm[:, j].astype(np.float64) for j, v in enumerate(vnames) if v in
                     ('thetao_mean', 'so_mean', 'o2_mean', 'no3_mean', 'chl_mean', 'sws_mean')}
        variables['latitud'] = D.aff[5] + D.aff[4] * (D.flat // D.w + .5)
        variables['profundidad'] = D.zonas['profundidad_m'].astype(np.float64)
        r['clases'] = por_clase(d, f, variables, D.estrato if len(nombres) > 1 else None, nombres if len(nombres) > 1 else None)
        def estratos(e, nombres=nombres):
            return [('todo', np.ones(len(e), bool))] + ([(n, e == i) for i, n in enumerate(nombres)] if len(nombres) > 1 else [])
        r['nulas'] = D.nulas(NULAS_M[capa], estratos)
        r['areas'] = {}
        for u in D.inv:
            if u['estado'] == 'evaluable':
                s, a = D.superficies(u['anp_id']); r['areas'][u['anp_id']] = {**s, 'km2_agua': float(a['area'].sum() / 1e6)}
        r['escala'] = D.escala(list(range(1, 21)) + [25, 30, 40, 50, 75, 100])
        out[capa] = r
    return out


if __name__ == '__main__':
    EXP = RES / '06_implicaciones'
    res = {'continental': continental(), 'maritimo': maritimo(), 'version': 'null networks N2c, two-sided test',
           'implicaciones': {k: json.loads((EXP / f'{k}_resumen.json').read_text(encoding='utf-8')) for k in ('P1', 'P1_curva', 'P1_sitios', 'P2')}
           | {'P3': list(csv.DictReader(open(EXP / 'P3_dejando_una_fuera.csv', encoding='utf-8-sig')))}
           | {'P1_umbral': json.loads((EXP / 'P1_umbral_esfuerzo.json').read_text(encoding='utf-8'))}}  # exact protected area and cut-off sensitivity
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    c = res['continental']
    print('Peru', {k: round(v, 3) for k, v in c['peru'].items()})
    print('nulas', {k: (round(v['D_real'], 3), round(v['D_azar_mediana'], 3), v['n_azar_menor_igual']) for k, v in c['nulas'].items()})
    print('escala', [(round(x['km'], 1), round(x['d'], 3)) for x in c['escala']['curva']], c['escala']['azar'])
    print('menos', round(c['menos_representados']['dmin_desde'], 3), round(c['menos_representados']['km2']))
    for capa, r in res['maritimo'].items():
        print(capa, {k: (round(v['D_real'], 3), round(v['D_azar_mediana'], 3), v['n_azar_menor_igual']) for k, v in r['nulas'].items()},
              [(round(x['km'], 1), round(x['d'], 3)) for x in r['escala']['curva']], round(r['escala']['azar'], 3))
