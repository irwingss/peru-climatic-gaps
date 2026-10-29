"""Priorizacion estandar con prioritizr. Cuatro escenarios, resueltos por conjunto minimo (min set) y por deficit minimo
con presupuesto (min shortfall):
A especies, solo en vacios climaticos; B especies en todo entorno natural no protegido; C especies + clima; D solo clima.

Unidades: microcuencas HydroBASINS nivel 12 (las de implicaciones.py), con su parte natural no protegida (en A, solo su parte en vacios).
La red real es una unidad mas, fija (locked in) y de costo 0. Costo = superficie (km2).
Especies: amenazadas (CR, EN, VU) con > 90 % del rango mundial en el Peru, aves incluidas; meta = fraccion del rango
segun su tamano (Rodrigues et al. 2004: 100 % bajo 1 000 km2, 10 % sobre 250 000 km2, lineal en log10).
Clima: tipos de clima (k-medias sobre la metrica de 11 variables); meta = META_CLIMA de la superficie de cada tipo.
Metas imposibles se recortan a lo alcanzable y se registran.

python prioritizacion.py preparar   -> resultados_v7/09_prioritizacion/entrada/
Rscript prioritizacion.R            -> resultados_v7/09_prioritizacion/solucion_<escenario>.csv
python prioritizacion.py evaluar    -> evaluacion.csv: D̄ y F(0.2), frente a la curva voraz y a microcuencas al azar
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

import ejecutar_investigacion as E
import implicaciones as I

DST = I.BASE/'resultados_v7'/'09_prioritizacion'
ENT = DST/'entrada'
K_CLIMA = 3000          # tipos de clima (k-medias): distancia mediana de una celda a su centroide 0.12, asi dos celdas
                        # de un mismo tipo suelen estar a d <= 0.2 (300 tipos: 0.20; 1 000: 0.16)
META_CLIMA = [.1, .17, .3]   # D se corre con las tres; C con la central (Aichi 11)
GLOBAL = 90             # % del rango mundial en el Peru
SEED = E.SEED+90


def meta_rodrigues(km2):
    t = np.interp(np.log10(km2),[3,np.log10(250000)],[1.,.1])
    return np.clip(t,.1,1.)


def _especies(n_celdas):
    """Especies elegidas y los indices de celda de su rango, aves incluidas (rasterizado como en cruce_uicn)."""
    import shapely
    sys.path.insert(0,str(I.BASE/'figuras_tablas'))
    import especies as E   # tablas de especies y rangos (figuras_tablas/especies.py)
    t,_ = E.cargar()
    S = t[t.amenazada&(t.pct_global_en_peru>GLOBAL)].reset_index(drop=True)
    dom = E.dominio_tierra(); assert dom.flat.size==n_celdas
    for g in S.grupo.unique():
        E.precargar(g,S[S.grupo==g].id_no)
    idx = []
    for e in S.itertuples():   # todas las celdas tocadas si el rango es menor que una celda
        g = [x for x in shapely.clip_by_rect(E.rango(e.grupo,e.id_no),*dom.bounds) if not x.is_empty]
        r = dom.mascara(g)
        idx.append(np.flatnonzero(r if r.any() else dom.mascara(g,True)))
    return S,idx


def preparar():
    t0 = time.perf_counter(); ENT.mkdir(parents=True,exist_ok=True)
    prep,dom,area,grupo,ecos,dep,nombres_dep = I._contexto()
    grid = json.loads((prep/'preparacion.json').read_text(encoding='utf-8'))['grid']
    km2 = area/1e6; prot_km2 = np.minimum(np.load(prep/'superficie_union.npy')/1e6,km2)
    protegido = prot_km2>0
    nat = np.load(I.MASCARA/'natural_dominio.npz')['natural']
    with __import__('rasterio').open(I.CONT/'mapas_distancia'/'cond100'/'sistema_distancia_minima.tif') as ds:
        dmin = ds.read(1).reshape(-1)[dom['flat']]
    hybas = I._microcuenca(dom,grid)
    candidato = nat&~protegido
    # Tipos de clima: k-medias sobre una muestra proporcional a la superficie; cada celda va a su centroide mas cercano.
    climate = np.load(I.CONT/'modelo'/'clima_transformado_cond100.npy')
    rng = np.random.default_rng(SEED); s = rng.choice(len(climate),100000,replace=False,p=area/area.sum())
    centro,_ = kmeans2(climate[s],K_CLIMA,seed=SEED,minit='points',iter=20)  # '++' de scipy es lentisimo con k=1000
    dc,tipo = cKDTree(centro).query(climate,workers=8)
    E.log(f'Tipos de clima: k={K_CLIMA}, distancia mediana al centroide {np.median(dc):.3f}')
    S,idx = _especies(len(km2))
    E.log(f'Especies: {len(S)} ({S.grupo.value_counts().to_dict()})')

    ub,pu = np.unique(hybas,return_inverse=True); pu = pu.ravel()+1   # 0 = red real
    for via,celda in (('vacios',candidato&(dmin>I.UMBRAL)),('todo',candidato)):
        # unidades: costo = superficie candidata de la microcuenca
        u = np.where(celda,pu,-1)
        costo = np.bincount(u[celda],km2[celda],len(ub)+1)
        U = pd.DataFrame({'id':np.arange(len(ub)+1),'hybas_id':np.r_[0,ub],'cost':costo,'locked_in':False})
        U.loc[0,['cost','locked_in']] = [0.,True]
        U = U[(U.id==0)|(U.cost>0)]
        filas = []
        for j,ix in enumerate(idx):   # especie j: superficie protegida (unidad 0) y candidata por unidad
            c = ix[celda[ix]]
            a = pd.Series(km2[c]).groupby(u[c]).sum()
            filas.append(pd.DataFrame({'pu':[0,*a.index],'species':j+1,'amount':[prot_km2[ix].sum(),*a.values]}))
        n = len(S)
        c = np.flatnonzero(celda)
        a = pd.DataFrame({'pu':u[c],'t':tipo[c],'km2':km2[c]}).groupby(['pu','t']).km2.sum().reset_index()
        pt = pd.DataFrame({'t':tipo,'km2':prot_km2}).groupby('t').km2.sum()
        filas.append(pd.DataFrame({'pu':a.pu,'species':n+1+a.t,'amount':a.km2}))
        filas.append(pd.DataFrame({'pu':0,'species':n+1+pt.index,'amount':pt.values}))
        R = pd.concat(filas); R = R[R.amount>0]
        disponible = R.groupby('species').amount.sum()
        tot_tipo = np.bincount(tipo,km2,K_CLIMA)
        F = pd.DataFrame({'id':np.arange(1,n+K_CLIMA+1),
                          'name':[*S.sci_name,*[f'clima_{k}' for k in range(K_CLIMA)]],
                          'tipo':['especie']*n+['clima']*K_CLIMA,
                          'total_km2':np.r_[S.tierra_km2.to_numpy(),tot_tipo]})
        F['meta_especie'] = np.where(F.tipo=='especie',meta_rodrigues(F.total_km2),np.nan)
        F['disponible_km2'] = F.id.map(disponible).fillna(0)
        F.to_csv(ENT/f'features_{via}.csv',index=False); U.to_csv(ENT/f'pu_{via}.csv',index=False)
        R.to_csv(ENT/f'rij_{via}.csv',index=False)
        E.log(f'{via}: {len(U)} unidades, {len(R)} filas rij')
    # para evaluar: unidad de cada celda candidata
    np.savez_compressed(ENT/'celdas.npz',pu=pu,candidato=candidato,vacio=dmin>I.UMBRAL,tipo=tipo,dc=dc)
    S[['grupo','id_no','sci_name','categoria','tierra_km2','tierra_pct_protegido','tierra_pct_mas_alla_similar',
       'pct_global_en_peru']].to_csv(ENT/'especies.csv',index=False)
    E.save_json(ENT/'resumen.json',{'k_clima':K_CLIMA,'d_mediana_centroide':float(np.median(dc)),
                'd_p90_centroide':float(np.percentile(dc,90)),'n_especies':len(S),
                'por_grupo':S.grupo.value_counts().to_dict(),'metas_clima':META_CLIMA,'segundos':time.perf_counter()-t0})


def evaluar():
    """D̄ y F(0.2) (muestra PPS del diseno N2c, como la curva de esfuerzo) de cada solucion; curva voraz a igual superficie;
    microcuencas al azar a igual superficie (mismo conjunto candidato); metas de especies cumplidas."""
    prep,dom,area,*_ = I._contexto()
    km2 = area/1e6
    climate = np.load(I.CONT/'modelo'/'clima_transformado_cond100.npy')
    z = np.load(I.NULAS/'redes_aleatorias.npz'); muestra,peso,d0 = z['muestra'],z['peso'],z['real_dmin'].astype(np.float64)
    q = climate[muestra]; W = peso.sum()
    C = np.load(ENT/'celdas.npz'); pu = C['pu']
    curva = pd.read_csv(I.DST/'P1_curva_esfuerzo.csv')
    def medir(celdas):
        d = np.minimum(d0,cKDTree(climate[celdas]).query(q,workers=8)[0]) if len(celdas) else d0
        return float(peso@d/W),float(peso[d<=I.UMBRAL].sum()/W)
    rng = np.random.default_rng(SEED+1); filas = []
    def azar(celda,sup,n=30):
        # microcuencas al azar del mismo conjunto candidato hasta igual superficie
        pool = np.unique(pu[celda]); por = pd.Series(km2[celda]).groupby(pu[celda]).sum(); az = []
        for _ in range(n):
            orden = rng.permutation(pool); k = np.searchsorted(por.loc[orden].cumsum().to_numpy(),sup)+1
            az.append(medir(np.flatnonzero(celda&np.isin(pu,orden[:k]))))
        return np.array(az)
    for f in sorted(DST.glob('solucion_*.csv')):
        esc = f.stem.removeprefix('solucion_'); via = 'vacios' if esc.startswith('A') else 'todo'
        sol = pd.read_csv(f); elegidas = sol.loc[(sol.solution_1==1)&(sol.id>0),'id'].to_numpy()
        celda = C['candidato']&(C['vacio'] if via=='vacios' else True)
        sel = np.flatnonzero(celda&np.isin(pu,elegidas)); sup = float(km2[sel].sum())
        D,F = medir(sel)
        # curva voraz a igual superficie
        dentro = sup<=curva.agregado_km2.max()  # la curva llega hasta la meta del 90 %; mas alla no hay comparacion
        Dc = float(np.interp(sup,curva.agregado_km2,curva.D)) if dentro else np.nan
        Fc = float(np.interp(sup,curva.agregado_km2,curva.F02)) if dentro else np.nan
        az = azar(celda,sup)
        # metas de especies
        Fe = pd.read_csv(ENT/f'features_{via}.csv'); R = pd.read_csv(ENT/f'rij_{via}.csv')
        logro = R[R.pu.isin([0,*elegidas])].groupby('species').amount.sum()
        e = Fe[Fe.tipo=='especie'].copy(); e['logro'] = e.id.map(logro).fillna(0)
        meta = np.minimum(e.meta_especie*e.total_km2,e.disponible_km2)
        filas.append({'escenario':esc,'unidades':len(elegidas),'superficie_km2':sup,
                      'pct_en_vacios':100*float(km2[sel][C['vacio'][sel]].sum())/max(sup,1e-9),
                      'D':D,'F02':F,'D_curva':Dc,'F02_curva':Fc,'D_azar_mediana':float(np.median(az[:,0])),
                      'F02_azar_mediana':float(np.median(az[:,1])),
                      'especies_meta_cumplida':int((e.logro>=meta*.999).sum()),'especies':len(e),
                      'especies_meta_rodrigues_cumplida':int((e.logro>=e.meta_especie*e.total_km2*.999).sum())})
        E.log(f'{esc}: {filas[-1]}')
    D0,F0 = medir(np.array([],int))
    E.write_csv(DST/'evaluacion.csv',[{'escenario':'red real','unidades':0,'superficie_km2':0.,'D':D0,'F02':F0},*filas])


if __name__=='__main__':
    {'preparar':preparar,'evaluar':evaluar}[sys.argv[1]]()
