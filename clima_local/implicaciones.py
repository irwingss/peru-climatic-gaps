"""Esfuerzo de expansion y papel de cada nivel: P1 (vacios expandibles, curva de esfuerzo y sitios potenciales), P2 (papel
de las ACR y las ACP) y P3 (dominio maritimo, dejando una area fuera). Lee los resultados de las redes nulas y la mascara."""
from __future__ import annotations

import argparse
import heapq
import json
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy.spatial import cKDTree

import ejecutar_investigacion as E

BASE = Path(__file__).resolve().parents[1]
CONT = BASE/'resultados_v7'/'02_sesgo_precipitacion'/'original'  # CHELSA original
ZONAS = BASE/'resultados_climaticos'/'20261011'/'preparados'/'zonas.npz'  # departamento por celda (misma cuadricula)
MASCARA = BASE/'resultados_v7'/'03_mascara_natural'
NULAS = BASE/'resultados_v7'/'04_redes_nulas'/'continental_superposicion_limitada'/'N2c'  # diseno principal
MAR = BASE/'resultados_marinos'/'20261011'
DST = BASE/'resultados_v7'/'06_implicaciones'
UMBRAL = .2  # vacio climatico: d_min > 0.2
MODALIDADES = {'NACIONAL':'ANP','REGIONAL':'ACR','PRIVADA':'ACP'}
CLASES_VACIO = [(.2,.3,'0.2-0.3'),(.3,.5,'0.3-0.5'),(.5,1.,'0.5-1.0'),(1.,np.inf,'>1 (no analogo)')]
BLOQUE = 12  # celdas de 30" por lado: ~10 km (unidad de la primera version, en 06_implicaciones/bloques_10km)
# Unidades candidatas de P1: microcuencas HydroBASINS v1c nivel 12 (Lehner & Grill 2013); mediana de 137 km2 en el Peru.
HYBAS = Path(__import__('os').environ.get('SIG_DIR', r'C:\Archivos\SIG'))/'Basin SouthAmerica'/'hybas_sa_lev01-12_v1c.zip'  # HydroBASINS v1c


def _microcuenca(dom,grid):
    """HYBAS_ID de nivel 12 de cada celda del dominio continental (centro de celda; las celdas de borde sin poligono
    toman la microcuenca mas cercana)."""
    destino = DST/'microcuencas_lev12.npz'
    if destino.exists():
        return np.load(destino)['hybas_id']
    import pyogrio, shapely
    from rasterio import features
    from scipy.ndimage import distance_transform_edt
    h,w = grid['height'],grid['width']; aff = rasterio.Affine(*grid['transform'])
    oeste,norte = aff.c,aff.f; este,sur = oeste+w*aff.a,norte+h*aff.e
    _,_,wkb,campos = pyogrio.raw.read(f'/vsizip/{HYBAS.as_posix()}/hybas_sa_lev12_v1c.shp',columns=['HYBAS_ID'],
                                      bbox=(oeste,sur,este,norte))
    ids = np.asarray(campos[0],np.int64)
    r = features.rasterize([(g,k+1) for k,g in enumerate(shapely.from_wkb(wkb))],out_shape=(h,w),transform=aff,fill=0,dtype='int32')
    cerca = distance_transform_edt(r==0,return_distances=False,return_indices=True)
    k = r[cerca[0],cerca[1]].reshape(-1)[dom['flat']]
    rellenas = int((r.reshape(-1)[dom['flat']]==0).sum())
    hybas = ids[k-1]
    np.savez_compressed(destino,hybas_id=hybas)
    E.save_json(destino.with_suffix('.json'),{'fuente':str(HYBAS),'capa':'hybas_sa_lev12_v1c','cita':'Lehner & Grill (2013), HydroBASINS v1c',
                'n_microcuencas':int(len(np.unique(hybas))),'celdas_rellenadas':rellenas,
                'criterio':'Centro de celda; celdas de borde sin poligono: la microcuenca mas cercana.'})
    E.log(f'Microcuencas: {len(np.unique(hybas))} en el dominio, {rellenas} celdas rellenadas')
    return hybas


def _unidades(prep):
    """Celdas de referencia (indices) y modalidad de cada unidad evaluable."""
    inv = json.loads((prep/'inventario.json').read_text(encoding='utf-8'))
    return [(u['anp_id'],u['nombre'],u['modalidad'],np.load(prep/'asociaciones'/f"{u['anp_id']}.npz")['index'])
            for u in inv if u['estado']=='evaluable']


def _dmin(climate,refs,query=None):
    q = climate if query is None else query
    tree = cKDTree(climate[np.unique(refs)],leafsize=32); d = np.empty(len(q),np.float32)
    for s in range(0,len(q),65536):
        d[s:s+65536] = tree.query(q[s:s+65536],k=1,eps=0,workers=8)[0]
    return d


def _tabla(claves,nombres,area,columnas):
    """Suma de superficie (km2) por clave para cada mascara de columnas."""
    filas = []
    for k,n in enumerate(nombres):
        m = claves==k
        if not m.any():
            continue
        filas.append({'nombre':n,'superficie_km2':area[m].sum()/1e6,**{c:area[m&v].sum()/1e6 for c,v in columnas.items()}})
    return filas


def _contexto():
    prep = CONT/'preparados'
    dom = np.load(prep/'dominio.npz'); area = dom['area']
    grupo,ecos = E.estratos_muestreo(prep)
    dep = np.load(ZONAS)['departamento'].astype(int)
    deps = json.loads((ZONAS.with_suffix('.json')).read_text(encoding='utf-8'))['departamentos']
    nombres_dep = ['']+[deps[str(k)] for k in range(1,len(deps)+1)]
    return prep,dom,area,grupo,ecos,dep,nombres_dep


def p2():
    """dmin por modalidad (tres arboles k-d sobre la union de referencias de ANP, ACR y ACP), modalidad proveedora y
    superficie que solo cubren las ACR y las ACP (dmin_ANP > 0.2 y dmin_red <= 0.2)."""
    started = time.perf_counter()
    prep,dom,area,grupo,ecos,dep,nombres_dep = _contexto()
    climate = np.load(CONT/'modelo'/'clima_transformado_cond100.npy')
    unidades = _unidades(prep)
    d = {}
    for mod,sig in MODALIDADES.items():
        d[sig] = _dmin(climate,np.concatenate([ix for _,_,m,ix in unidades if m==mod]))
        E.log(f'P2 dmin {sig}: mediana {np.median(d[sig]):.3f}')
    D = np.stack([d['ANP'],d['ACR'],d['ACP']]); red = D.min(0)
    with rasterio.open(CONT/'mapas_distancia'/'cond100'/'sistema_distancia_minima.tif') as ds:
        control = float(np.abs(ds.read(1).reshape(-1)[dom['flat']]-red).max())
    proveedora = D.argmin(0)  # empate (celda de referencia de dos modalidades) -> la primera, ANP
    DST.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(DST/'dmin_por_modalidad.npz',ANP=d['ANP'],ACR=d['ACR'],ACP=d['ACP'],red=red,proveedora=proveedora.astype(np.int8))
    cub = red<=UMBRAL
    col = {'cubierta_red_km2':cub,'cubierta_solo_ANP_km2':d['ANP']<=UMBRAL,
           'solo_ACR_ACP_km2':(d['ANP']>UMBRAL)&cub,
           'solo_ACR_ACP_proveedora_ACR_km2':(d['ANP']>UMBRAL)&cub&(proveedora==1),
           'solo_ACR_ACP_proveedora_ACP_km2':(d['ANP']>UMBRAL)&cub&(proveedora==2),
           'proveedora_ANP_km2':proveedora==0,'proveedora_ACR_km2':proveedora==1,'proveedora_ACP_km2':proveedora==2}
    for nombre,claves,etiquetas in (('ecorregion',grupo,ecos),('departamento',dep,nombres_dep)):
        filas = _tabla(claves,etiquetas,area,col)
        tot = {'nombre':'Peru','superficie_km2':area.sum()/1e6,**{c:area[v].sum()/1e6 for c,v in col.items()}}
        E.write_csv(DST/f'P2_modalidades_por_{nombre}.csv',[tot,*filas])
    # Departamentos con ACR o ACP activas: los que contienen celdas de alguna ACR o ACP.
    con = {MODALIDADES[m]:sorted({nombres_dep[k] for _,_,mm,ix in unidades if mm==m for k in np.unique(dep[ix])})
           for m in ('REGIONAL','PRIVADA')}
    resumen = {'control_max_abs_vs_sistema':control,'superficie_km2':area.sum()/1e6,
               **{f'F02_{k}':float(area[v<=UMBRAL].sum()/area.sum()) for k,v in d.items()},'F02_red':float(area[cub].sum()/area.sum()),
               'solo_ACR_ACP_km2':float(area[col['solo_ACR_ACP_km2']].sum()/1e6),
               'departamentos_con':con,'n_unidades':{s:sum(m==k for _,_,m,_ in unidades) for k,s in MODALIDADES.items()},
               'segundos':time.perf_counter()-started}
    E.save_json(DST/'P2_resumen.json',resumen); E.log(f'P2: {resumen}')


def _distancia_costa(xy_mar):
    """Distancia (km) de cada celda marina al centro de celda continental mas cercano (EPSG:6933)."""
    # costa aproximada por los centros de celda continentales de 30"; error < 1 km
    tierra = np.load(CONT/'preparados'/'coordenadas_6933.npy')
    return cKDTree(tierra).query(xy_mar,k=1,workers=8)[0]/1e3


def p3():
    """Por capa marina: dmin, F(0.2) y D̄ (exactos sobre todo el dominio) de la red completa y dejando fuera cada area;
    vacios por latitud y distancia a la costa."""
    started = time.perf_counter(); filas,vacios = [],[]
    for capa in ('superficie','fondo'):
        out = MAR/capa; prep = out/'preparados'
        area = np.load(prep/'dominio.npz')['area']; grupo,estratos = E.estratos_muestreo(prep)
        climate = np.load(out/'modelo'/'clima_transformado_cond100.npy'); unidades = _unidades(prep)
        def medir(fuera,nombre):
            dm = _dmin(climate,np.concatenate([ix for uid,_,_,ix in unidades if uid!=fuera]))
            fila = {'capa':capa,'fuera':fuera or '(ninguna)','nombre':nombre,
                    'superficie_area_km2':0. if not fuera else float(area[next(ix for u,_,_,ix in unidades if u==fuera)].sum()/1e6),
                    'D_todo':float(np.average(dm,weights=area)),'F02_todo':float(area[dm<=UMBRAL].sum()/area.sum()),
                    'vacios_km2':float(area[dm>UMBRAL].sum()/1e6),'no_analogo_km2':float(area[dm>1].sum()/1e6)}
            for k,n in enumerate(estratos):
                m = grupo==k; fila[f'D_{n}'] = float(np.average(dm[m],weights=area[m])); fila[f'F02_{n}'] = float(area[m&(dm<=UMBRAL)].sum()/area[m].sum())
            return dm,fila
        base,fila = medir(None,'red completa'); filas.append(fila)
        with rasterio.open(out/'mapas_distancia'/'cond100'/'sistema_distancia_minima.tif') as ds:
            flat = np.load(prep/'dominio.npz')['flat']; control = float(np.abs(ds.read(1).reshape(-1)[flat]-base).max())
        fila['control_max_abs_vs_sistema'] = control
        for uid,nombre,_,_ in unidades:
            filas.append(medir(uid,nombre)[1]); E.log(f'P3 {capa} sin {uid}')
        xy = np.load(prep/'coordenadas_6933.npy'); lat = E.BACKWARD.transform(xy[:,0],xy[:,1])[1]
        costa = _distancia_costa(xy)
        banda_lat = np.floor(lat).astype(int)  # banda [k, k+1) grados
        banda_costa = np.digitize(costa,[50,100,200,370])
        nb = ['0-50','50-100','100-200','200-370','>370']
        for k,n in [(-1,'todos'),*enumerate(estratos)]:
            m = np.ones(len(area),bool) if k<0 else grupo==k
            for tipo,claves,etiquetas in (('latitud',banda_lat,None),('distancia_costa_km',banda_costa,nb)):
                for b in np.unique(claves[m]):
                    s = m&(claves==b)
                    vacios.append({'capa':capa,'estrato':n,'tipo':tipo,'banda':f'{b} a {b+1}' if etiquetas is None else etiquetas[b],
                                   'superficie_km2':area[s].sum()/1e6,'vacios_km2':area[s&(base>UMBRAL)].sum()/1e6,
                                   'no_analogo_km2':area[s&(base>1)].sum()/1e6})
            g = m&(base>1)
            if g.any():
                E.log(f'P3 {capa} {n}: no analogo, latitud mediana ponderada {E.weighted_quantile(lat[g],area[g],[.5])}')
        np.savez_compressed(DST/f'P3_{capa}_base.npz',dmin=base,latitud=lat,distancia_costa_km=costa)
    DST.mkdir(parents=True,exist_ok=True)
    E.write_csv(DST/'P3_dejando_una_fuera.csv',filas); E.write_csv(DST/'P3_vacios_latitud_costa.csv',vacios)
    E.log(f'P3: {time.perf_counter()-started:.0f} s')


def _elevacion(dom,grid):
    """Elevacion (m) de WorldClim 2.1 a 30" en el centro de cada celda del dominio continental."""
    h,w = grid['height'],grid['width']; a = grid['transform']
    r,c = dom['flat']//w,dom['flat']%w
    lon = a[2]+(c+.5)*a[0]; lat = a[5]+(r+.5)*a[4]
    with rasterio.open(E.DEM) as ds:
        win = rasterio.windows.from_bounds(lon.min()-.01,lat.min()-.01,lon.max()+.01,lat.max()+.01,ds.transform).round_offsets().round_lengths()
        z = ds.read(1,window=win); t = ds.window_transform(win)
        rr = ((lat-t.f)/t.e).astype(int); cc = ((lon-t.c)/t.a).astype(int)
        v = z[rr,cc].astype(np.float32); v[v==ds.nodata] = np.nan
    return v


def p1_tabla():
    """Vacios expandibles (vacio climatico en entorno natural) por ecorregion, departamento y franja de 500 m."""
    prep,dom,area,grupo,ecos,dep,nombres_dep = _contexto()
    grid = json.loads((prep/'preparacion.json').read_text(encoding='utf-8'))['grid']
    red = np.load(DST/'dmin_por_modalidad.npz')['red']; nat = np.load(MASCARA/'natural_dominio.npz')['natural']
    z = _elevacion(dom,grid); franja = np.floor(np.nan_to_num(z,nan=-1)/500).astype(int)
    vacio = red>UMBRAL; clase = np.digitize(red,[c[1] for c in CLASES_VACIO[:-1]])  # 0..3 dentro de los vacios
    claves = np.stack([grupo,dep,franja],1)[vacio]; a = area[vacio]; n = nat[vacio]; cl = clase[vacio]
    u,inv = np.unique(claves,axis=0,return_inverse=True); inv = inv.ravel(); filas = []
    for k,(g,dd,f) in enumerate(u):
        m = inv==k; por_clase = np.bincount(cl[m],weights=a[m],minlength=4)
        filas.append({'ecorregion':ecos[g],'departamento':nombres_dep[dd],'franja_m':f'{500*f}-{500*f+500}' if f>=0 else 'sin dato',
                      'vacios_km2':a[m].sum()/1e6,'vacios_naturales_km2':a[m&n].sum()/1e6,
                      'clase_d_predominante':CLASES_VACIO[int(np.argmax(por_clase))][2]})
    filas.sort(key=lambda r:-r['vacios_naturales_km2'])
    E.write_csv(DST/'P1_vacios_expandibles.csv',filas)
    res = {'vacios_km2':float(a.sum()/1e6),'vacios_naturales_km2':float(a[n].sum()/1e6),
           'vacios_por_clase_km2':{c[2]:float(a[cl==i].sum()/1e6) for i,c in enumerate(CLASES_VACIO)},
           'vacios_naturales_por_clase_km2':{c[2]:float(a[n&(cl==i)].sum()/1e6) for i,c in enumerate(CLASES_VACIO)},
           'celdas_sin_elevacion':int(np.isnan(z).sum())}
    for nombre,claves_,etiquetas in (('ecorregion',grupo,ecos),('departamento',dep,nombres_dep)):
        E.write_csv(DST/f'P1_vacios_por_{nombre}.csv',_tabla(claves_,etiquetas,area,{'vacios_km2':vacio,'vacios_naturales_km2':vacio&nat}))
    # Parte de los vacios expandibles en departamentos con ACR o ACP (instrumento disponible).
    con = json.loads((DST/'P2_resumen.json').read_text(encoding='utf-8'))['departamentos_con']
    for sig,lista in con.items():
        m = np.isin(dep,[nombres_dep.index(x) for x in lista])
        res[f'vacios_naturales_en_departamentos_con_{sig}_pct'] = float(100*area[vacio&nat&m].sum()/area[vacio&nat].sum())
    m = np.isin(dep,[nombres_dep.index(x) for x in set(con['ACR'])|set(con['ACP'])])
    res['vacios_naturales_en_departamentos_con_ACR_o_ACP_pct'] = float(100*area[vacio&nat&m].sum()/area[vacio&nat].sum())
    E.save_json(DST/'P1_resumen.json',res); E.log(f'P1 tabla: {res}')


def p1_curva(max_km2=400000):
    """Curva de esfuerzo: microcuencas HydroBASINS de nivel 12 elegidas con un algoritmo voraz perezoso por la reduccion
    de D̄ (p-mediana, Faith & Walker 1996). Cada microcuenca agrega como referencias sus celdas naturales no protegidas;
    el esfuerzo es su superficie. D̄ y F(0.2) se estiman con la muestra PPS del diseno principal (pesos HT)."""
    started = time.perf_counter()
    prep,dom,area,grupo,ecos,dep,nombres_dep = _contexto()
    grid = json.loads((prep/'preparacion.json').read_text(encoding='utf-8'))['grid']; w = grid['width']
    climate = np.load(CONT/'modelo'/'clima_transformado_cond100.npy')
    z = np.load(NULAS/'redes_aleatorias.npz'); muestra,peso,dcur = z['muestra'],z['peso'],z['real_dmin'].astype(np.float64)
    meta_nula = float(np.median(np.average(z['azar_dmin'],axis=1,weights=peso)))
    q = climate[muestra]; W = peso.sum()
    nat = np.load(MASCARA/'natural_dominio.npz')['natural']; protegido = np.load(prep/'superficie_union.npy')>0
    cand = np.flatnonzero(nat&~protegido)
    bloque = _microcuenca(dom,grid)[cand]
    ub,inv = np.unique(bloque,return_inverse=True); inv = inv.ravel(); orden = np.argsort(inv,kind='stable')
    cortes = np.r_[0,np.cumsum(np.bincount(inv))]; celdas = [cand[orden[cortes[k]:cortes[k+1]]] for k in range(len(ub))]
    E.log(f'P1 curva: {len(ub)} microcuencas candidatas, {len(q)} puntos de muestra, meta nula {meta_nula:.4f}')
    activos = np.flatnonzero(dcur>0)  # los puntos ya protegidos (d = 0) no pueden mejorar
    qa = q[activos]; qa2 = (qa**2).sum(1)
    def ganancia(k):
        # distancia exacta por algebra matricial; con pocas celdas por unidad el k-d no ayuda en 11 dimensiones
        c = climate[celdas[k]]; d2 = np.full(len(qa),np.inf)
        for i in range(0,len(c),150):  # por tramos: una microcuenca grande no dispara la memoria
            ci = c[i:i+150]; d2 = np.minimum(d2,(qa2[:,None]+(ci**2).sum(1)[None,:]-2*qa@ci.T).min(1))
        dk = np.zeros(len(q)); dk[activos] = np.sqrt(np.maximum(d2,0))
        return float(peso@np.maximum(0,dcur-dk)),dk
    control = cKDTree(climate[celdas[0]]).query(qa,k=1)[0]
    assert np.allclose(ganancia(0)[1][activos],control,atol=1e-6), 'distancia matricial distinta de la exacta'
    # la ganancia de la p-mediana es submodular, asi que el voraz perezoso es exacto respecto del voraz simple
    heap = []
    for k in range(len(ub)):
        heap.append((-ganancia(k)[0],k))
        if k%2000==0:
            E.log(f'P1 curva: ganancia inicial {k}/{len(ub)}')
    heapq.heapify(heap)
    usados = np.zeros(len(ub),bool); km2 = 0.; filas = [{'paso':0,'unidad':'','agregado_km2':0.,'D':float(peso@dcur/W),
             'F02':float(peso[dcur<=UMBRAL].sum()/W),'ecorregion':'','departamento':''}]
    paso = 0
    while heap and km2<max_km2:
        g,k = heapq.heappop(heap)
        nueva,dk = ganancia(k)
        if heap and nueva<-heap[0][0]-1e-12:
            heapq.heappush(heap,(-nueva,k)); continue
        if nueva<=0:
            break
        dcur = np.minimum(dcur,dk); usados[k] = True; paso += 1; c = celdas[k]; km2 += area[c].sum()/1e6
        filas.append({'paso':paso,'unidad':int(ub[k]),'agregado_km2':km2,'D':float(peso@dcur/W),'F02':float(peso[dcur<=UMBRAL].sum()/W),
                      'ecorregion':ecos[np.bincount(grupo[c],weights=area[c]).argmax()],
                      'departamento':nombres_dep[np.bincount(dep[c],weights=area[c]).argmax()]})
        if paso%50==0:
            E.log(f"P1 curva paso {paso}: +{km2:.0f} km2, D {filas[-1]['D']:.4f}, F02 {filas[-1]['F02']:.3f}")
        if filas[-1]['F02']>=.9 and filas[-1]['D']<=meta_nula:
            break
    E.write_csv(DST/'P1_curva_esfuerzo.csv',filas)
    total = area.sum()/1e6; prot = area[protegido].sum()/1e6  # huella: celda tocada = protegida (la superficie exacta la usa umbral_esfuerzo.py)
    def alcanza(cond):
        f = next((r for r in filas if cond(r)),None)
        return None if f is None else {'pasos':f['paso'],'agregado_km2':f['agregado_km2'],
                                       'protegido_total_pct':100*(prot+f['agregado_km2'])/total}
    res = {'meta_mediana_nula_D':meta_nula,'D_inicial':filas[0]['D'],'F02_inicial':filas[0]['F02'],
           'protegido_km2_celdas':prot,'continental_km2':total,'meta3_30pct_km2':.3*total,
           'alcanza_mediana_nula':alcanza(lambda r:r['D']<=meta_nula),'alcanza_F02_80':alcanza(lambda r:r['F02']>=.8),
           'alcanza_F02_90':alcanza(lambda r:r['F02']>=.9),'unidad':'microcuenca HydroBASINS nivel 12','n_unidades_candidatas':len(ub),'segundos':time.perf_counter()-started}
    # Esfuerzo por ecorregion y departamento hasta cada meta.
    for meta,clave in (('mediana_nula','alcanza_mediana_nula'),('F02_80','alcanza_F02_80'),('F02_90','alcanza_F02_90')):
        if res[clave]:
            pasos = [r for r in filas[1:] if r['paso']<=res[clave]['pasos']]
            ks = [int(np.flatnonzero(ub==r['unidad'])[0]) for r in pasos]
            c = np.concatenate([celdas[k] for k in ks])
            for nombre,claves_,etiquetas in (('ecorregion',grupo,ecos),('departamento',dep,nombres_dep)):
                E.write_csv(DST/f'P1_esfuerzo_{meta}_por_{nombre}.csv',
                            [{'nombre':etiquetas[i],'agregado_km2':float(area[c][claves_[c]==i].sum()/1e6)}
                             for i in np.unique(claves_[c])])
    E.save_json(DST/'P1_curva_resumen.json',res); E.log(f'P1 curva: {res}')


def p1_validar():
    """Control de la curva: D̄ y F(0.2) exactos sobre todo el dominio (no la muestra con que se eligieron las microcuencas)
    con la red real mas las microcuencas elegidas hasta cada meta."""
    prep,dom,area,grupo,ecos,dep,nombres_dep = _contexto()
    grid = json.loads((prep/'preparacion.json').read_text(encoding='utf-8'))['grid']
    climate = np.load(CONT/'modelo'/'clima_transformado_cond100.npy')
    nat = np.load(MASCARA/'natural_dominio.npz')['natural']; protegido = np.load(prep/'superficie_union.npy')>0
    refs = np.concatenate([ix for *_,ix in _unidades(prep)])
    bloque = _microcuenca(dom,grid); candidato = nat&~protegido
    res = json.loads((DST/'P1_curva_resumen.json').read_text(encoding='utf-8'))
    filas = list(__import__('csv').DictReader(open(DST/'P1_curva_esfuerzo.csv',encoding='utf-8-sig')))
    out = {}
    for clave in ('inicial','alcanza_mediana_nula','alcanza_F02_80','alcanza_F02_90'):
        n = 0 if clave=='inicial' else res[clave]['pasos']
        elegidos = [int(r['unidad']) for r in filas[1:n+1]]
        extra = np.flatnonzero(candidato&np.isin(bloque,elegidos))
        d = _dmin(climate,np.concatenate([refs,extra]))
        out[clave] = {'pasos':n,'D_todo':float(np.average(d,weights=area)),'F02_todo':float(area[d<=UMBRAL].sum()/area.sum()),
                      'agregado_km2':float(area[extra].sum()/1e6)}
        E.log(f'P1 validacion {clave}: {out[clave]}')
    res['validacion_dominio_completo'] = out; E.save_json(DST/'P1_curva_resumen.json',res)


METAS = [('alcanza_mediana_nula','N2c null median','#3f007d'),('alcanza_F02_80','80 % at d ≤ 0.2','#807dba'),
         ('alcanza_F02_90','90 % at d ≤ 0.2','#cbc9e2')]
COLOR_VACIO = ['#fb6a4a','#ef3b2c','#a50f15','#3d0008']  # vacios naturales por clase de d (CLASES_VACIO)


def p1_sitios():
    """Sitios potenciales: las microcuencas de la curva de esfuerzo hasta la meta del 90 %, unidas si se tocan (vecindad
    de 8 celdas). El sitio es la parte natural no protegida de sus microcuencas. Tabla, GeoPackage y GeoTIFF de sitios y
    de vacios expandibles, y los dos mapas (en ingles)."""
    import pyogrio, shapely
    from rasterio import features
    from scipy import ndimage
    prep,dom,area,grupo,ecos,dep,nombres_dep = _contexto()
    base = E._lienzo(CONT); h,w,aff = base.h,base.w,base.aff; flat = dom['flat']
    red = np.load(DST/'dmin_por_modalidad.npz')['red']; nat = np.load(MASCARA/'natural_dominio.npz')['natural']
    union = np.load(prep/'superficie_union.npy'); protegido = union>0; z = _elevacion(dom,base.grid)
    res = json.loads((DST/'P1_curva_resumen.json').read_text(encoding='utf-8'))
    curva = list(__import__('csv').DictReader(open(DST/'P1_curva_esfuerzo.csv',encoding='utf-8-sig')))
    pasos = {int(r['unidad']):(int(r['paso']),float(curva[k-1]['D'])-float(r['D']),float(r['F02'])-float(curva[k-1]['F02']))
             for k,r in enumerate(curva) if k and int(r['paso'])<=res['alcanza_F02_90']['pasos']}
    meta_de = lambda p: next(i for i,(m,_,_) in enumerate(METAS) if p<=res[m]['pasos'])
    bloque = _microcuenca(dom,base.grid); elegida = np.isin(bloque,list(pasos))
    celda = nat&~protegido&elegida  # celdas que se agregarian
    # Sitios: componentes conexas (8 vecinos) de las microcuencas elegidas completas; cada microcuenca va a su componente
    # mayoritaria.
    malla = np.zeros(h*w,bool); malla[flat[elegida]] = True
    etiqueta,n = ndimage.label(malla.reshape(h,w),structure=np.ones((3,3))); etiqueta = etiqueta.reshape(-1)[flat]
    sitio_bloque = {b:int(np.bincount(etiqueta[bloque==b]).argmax()) for b in pasos}
    usadas = sorted(set(sitio_bloque.values())); renum = {v:k+1 for k,v in enumerate(usadas)}
    sitio_bloque = {b:renum[v] for b,v in sitio_bloque.items()}; n = len(usadas)
    sitio = np.zeros(len(flat),np.int32); sitio[celda] = [sitio_bloque[b] for b in bloque[celda]]
    lat = aff.f+aff.e*(flat//w+.5); lon = aff.c+aff.a*(flat%w+.5)
    # Area protegida mas cercana (centro de celda a centro de celda, EPSG:6933).
    xy = np.load(prep/'coordenadas_6933.npy'); unidades = _unidades(prep)
    ref_ix = np.concatenate([ix for *_,ix in unidades]); ref_u = np.concatenate([[k]*len(ix) for k,(*_,ix) in enumerate(unidades)])
    arbol = cKDTree(xy[ref_ix])
    filas = []
    for s in range(1,n+1):
        m = sitio==s; a = area[m]; bs = [b for b,v in sitio_bloque.items() if v==s]
        eco = np.bincount(grupo[m],weights=a,minlength=len(ecos)); deps = np.bincount(dep[m],weights=a)
        dd,jj = arbol.query(xy[m],k=1); cerca = int(np.argmin(dd))
        filas.append({'_s':s,'lat':float(np.average(lat[m],weights=a)),'lon':float(np.average(lon[m],weights=a)),
                      'area_km2':a.sum()/1e6,'area_microcuencas_km2':area[np.isin(bloque,bs)].sum()/1e6,'n_microcuencas':len(bs),'hybas_id':'; '.join(map(str,sorted(bs))),'primer_paso':min(pasos[b][0] for b in bs),
                      'meta':METAS[meta_de(min(pasos[b][0] for b in bs))][1],
                      'reduccion_D':sum(pasos[b][1] for b in bs),'aumento_F02_pp':100*sum(pasos[b][2] for b in bs),
                      'ecorregion':ecos[int(eco.argmax())],'pct_ecorregion':100*eco.max()/a.sum(),
                      'departamentos':'; '.join(nombres_dep[k] for k in np.argsort(-deps) if deps[k]>0 and k>0),
                      'elevacion_m_mediana':float(np.nanmedian(z[m])),'elevacion_m_min':float(np.nanmin(z[m])),
                      'elevacion_m_max':float(np.nanmax(z[m])),
                      'dmin_mediana':float(E.weighted_quantile(red[m],a,[.5])[0]),'pct_vacio':100*a[red[m]>UMBRAL].sum()/a.sum(),
                      'ap_cercana':unidades[ref_u[jj[cerca]]][1],'ap_cercana_modalidad':MODALIDADES[unidades[ref_u[jj[cerca]]][2]],
                      'dist_ap_km':float(dd[cerca]/1e3)})
    filas.sort(key=lambda r:-r['lat'])  # SP01 al norte
    nuevo = {r['_s']:k+1 for k,r in enumerate(filas)}
    ancho = len(str(len(filas)))
    for r in filas:
        r['id'] = f"SP{nuevo[r.pop('_s')]:0{max(2,ancho)}d}"
    sitio[sitio>0] = [nuevo[s] for s in sitio[sitio>0]]
    filas = [{'id':r['id'],**{k:v for k,v in r.items() if k!='id'}} for r in filas]
    E.write_csv(DST/'P1_sitios_potenciales.csv',filas)
    # GeoTIFF de sitios y de vacios expandibles (1-4 vacio natural por clase de d, 5 vacio no natural).
    perfil = dict(driver='GTiff',width=w,height=h,count=1,crs=base.grid['crs'],transform=aff,tiled=True,blockxsize=256,
                  blockysize=256,compress='deflate')
    def raster(v,dtype,nodata,path):
        band = np.full(h*w,nodata,dtype); band[flat] = v
        with rasterio.open(path,'w',dtype=dtype,nodata=nodata,**perfil) as ds:
            ds.write(band.reshape(h,w),1)
        return band.reshape(h,w)
    rs = raster(sitio.astype(np.uint16),'uint16',0,DST/'P1_sitios_potenciales_id.tif')
    clase = np.digitize(red,[c[1] for c in CLASES_VACIO[:-1]])+1
    vac = np.where(red>UMBRAL,np.where(nat,clase,5),0).astype(np.uint8)
    raster(vac,'uint8',255,DST/'P1_vacios_expandibles.tif')
    # Poligonos: celdas de cada sitio vectorizadas (EPSG:4326).
    geoms = {}
    for g,v in features.shapes(rs,mask=rs>0,transform=aff,connectivity=8):
        geoms.setdefault(int(v),[]).append(shapely.geometry.shape(g))
    wkb = shapely.to_wkb(np.array([shapely.union_all(geoms[k+1]) for k in range(len(filas))],dtype=object))
    columnas = list(filas[0])
    pyogrio.raw.write(DST/'P1_sitios_potenciales.gpkg',wkb,field_data=[np.array([r[c] for r in filas]) for c in columnas],
                      fields=columnas,crs='EPSG:4326',geometry_type='MultiPolygon',driver='GPKG',layer='sitios_potenciales',
                      promote_to_multi=True)
    _mapas(base,vac,sitio,protegido,filas,res)
    resumen = {'n_sitios':len(filas),'area_km2':float(area[sitio>0].sum()/1e6),'n_microcuencas':len(pasos),
               'por_meta':{et:{'n_sitios':sum(r['meta']==et for r in filas),'area_km2':sum(r['area_km2'] for r in filas if r['meta']==et)}
                           for _,et,_ in METAS},
               'por_ecorregion_km2':{e:float(area[(sitio>0)&(grupo==k)].sum()/1e6) for k,e in enumerate(ecos) if ((sitio>0)&(grupo==k)).any()},
               'pct_area_en_vacio':float(100*area[(sitio>0)&(red>UMBRAL)].sum()/area[sitio>0].sum()),
               'criterio':'Microcuencas HydroBASINS nivel 12 de la curva de esfuerzo hasta la meta del 90 %; celdas naturales sin parte '
                          'protegida; microcuencas que se tocan (8 vecinos) forman un sitio. Sin especies.'}
    E.save_json(DST/'P1_sitios_resumen.json',resumen); E.log(f'P1 sitios: {resumen}')


def _mapas(base,vac,sitio,protegido,filas,res):
    """Fig. A: vacios climaticos expandibles por clase de d. Fig. B: sitios potenciales por meta de la curva."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    h,w,flat = base.h,base.w,base.flat
    def lienzo(titulo,subtitulo):
        fig,ax = plt.subplots(figsize=(8.2,9.4),dpi=300)
        suelo = ListedColormap(['#d6d6d2','#f4f4f1']); suelo.set_bad((0,0,0,0))
        ax.imshow(base.domain,cmap=suelo,vmin=0,vmax=1,extent=base.extent,interpolation='nearest')
        img = np.full(h*w,np.nan,np.float32); img[flat[protegido]] = 1
        ax.imshow(img.reshape(h,w),cmap=ListedColormap(['#6baed6']),extent=base.extent,interpolation='nearest')
        E._contorno(ax,base.peru,color='#6b6b66',lw=.4)
        ax.set_aspect(1/np.cos(np.deg2rad(-9.2))); ax.set_axis_off()
        ax.set_title(titulo,loc='left',fontsize=11,color='#1f1f1d',pad=24)
        ax.text(0,1.005,subtitulo,transform=ax.transAxes,fontsize=7,color='#55554f',va='bottom',ha='left')
        return fig,ax
    def capa(ax,valores,colores):
        img = np.full(h*w,np.nan,np.float32); img[flat] = np.where(valores>0,valores,np.nan)
        ax.imshow(img.reshape(h,w),cmap=ListedColormap(colores),vmin=.5,vmax=len(colores)+.5,extent=base.extent,interpolation='nearest')
    nota = 'CHELSA V2.1 1981–2010 · 30″ · 11 predictors · regularized Mahalanobis (κ ≤ 100). Natural settings: MapBiomas Peru 2020 + GHS-SMOD 2020.'
    # Fig. A
    fig,ax = lienzo('Expandable climatic gaps','Climatic gaps (dmin > 0.2) in natural settings, by distance class to the nearest protected climate')
    capa(ax,vac.astype(float),COLOR_VACIO+['#9a9a94'])
    hs = [Patch(color=c,label=f"d {e.replace('no analogo','no analogue')}") for c,(_,_,e) in zip(COLOR_VACIO,CLASES_VACIO)]
    hs += [Patch(color='#9a9a94',label='Gap in transformed land or city'),Patch(color='#6baed6',label='Protected areas (ANP + ACR + ACP)')]
    ax.legend(handles=hs,fontsize=7,loc='lower left',frameon=False,title='Climatic gaps',title_fontsize=7.5)
    ax.text(0,-.005,nota,transform=ax.transAxes,fontsize=6,color='#55554f',va='top',ha='left')
    fig.savefig(DST/'P1_mapa_vacios_expandibles.png',bbox_inches='tight'); plt.close(fig)
    # Fig. B
    fig,ax = lienzo('Potential sites to close climatic gaps',
                    f"{len(filas)} sites made of HydroBASINS level-12 sub-basins chosen greedily by the reduction of D̄ (p-median); colour = first target each site serves")
    gris = np.where(vac>0,1,0).astype(float); capa(ax,gris,['#f2c4b5'])
    meta = np.zeros(len(flat)); idx = {r['id']:k for k,r in enumerate(filas)}
    etq = {k+1:[et for _,et,_ in METAS].index(r['meta'])+1 for k,r in enumerate(filas)}
    meta[sitio>0] = [etq[s] for s in sitio[sitio>0]]
    capa(ax,meta,[c for *_,c in METAS])
    for r in sorted(filas,key=lambda r:-r['reduccion_D'])[:25]:  # etiqueta solo los 25 que mas reducen D̄
        ax.annotate(r['id'],(r['lon'],r['lat']),fontsize=5.5,color='#1f1f1d',xytext=(3,2),textcoords='offset points')
    hs = [Patch(color=c,label=f"{et} (+{res[m]['agregado_km2']:,.0f} km² cumulative)") for m,et,c in METAS]
    hs += [Patch(color='#f2c4b5',label='Climatic gaps (dmin > 0.2)'),Patch(color='#6baed6',label='Protected areas (ANP + ACR + ACP)')]
    ax.legend(handles=hs,fontsize=7,loc='lower left',frameon=False,title='Target reached',title_fontsize=7.5)
    ax.text(0,-.005,nota+'\nLabels: the 25 sites with the largest reduction of D̄. Sites are not species-based.',
            transform=ax.transAxes,fontsize=6,color='#55554f',va='top',ha='left')
    fig.savefig(DST/'P1_mapa_sitios_potenciales.png',bbox_inches='tight'); plt.close(fig)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parte',choices=['p1','p1_curva','p1_validar','p1_sitios','p2','p3'])
    args = parser.parse_args()
    {'p1':p1_tabla,'p1_curva':p1_curva,'p1_validar':p1_validar,'p1_sitios':p1_sitios,'p2':p2,'p3':p3}[args.parte]()
