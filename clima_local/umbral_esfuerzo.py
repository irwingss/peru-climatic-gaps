"""Sensitivity of the expansion figures to the 0.2 cut-off (Table S24).

The greedy curve of implicaciones.p1_curva minimizes D-bar and does not use the cut-off, so its sequence of microbasins is
the same for any cut-off; only where the targets are read changes. This replays that curve (same sample, candidates and
lazy greedy; the first 228 units must equal P1_curva_esfuerzo.csv), runs it on until F(0.15) >= 0.9, records F(x) for
x in UMBRALES and, for each x, reads the 80% and 90% targets, their microbasins and potential sites (8-neighbour
components, as in p1_sitios) and the share of the land still needed for 30% (exact protected area, not the footprint).
Writes resultados_v7/06_implicaciones/P1_umbral_esfuerzo.csv and .json. Run: python umbral_esfuerzo.py (~30 min)
"""
import csv
import heapq
import json

import numpy as np
from scipy import ndimage

import ejecutar_investigacion as E
import implicaciones as I

UMBRALES = (.15, .2, .25, .3)


def curva(max_km2=400000):
    prep, dom, area, grupo, ecos, dep, nd = I._contexto()
    grid = json.loads((prep/'preparacion.json').read_text(encoding='utf-8'))['grid']
    climate = np.load(I.CONT/'modelo'/'clima_transformado_cond100.npy')
    z = np.load(I.NULAS/'redes_aleatorias.npz'); muestra, peso, dcur = z['muestra'], z['peso'], z['real_dmin'].astype(np.float64)
    q = climate[muestra]; W = peso.sum()
    nat = np.load(I.MASCARA/'natural_dominio.npz')['natural']; protegido = np.load(prep/'superficie_union.npy') > 0
    cand = np.flatnonzero(nat & ~protegido); bloque = I._microcuenca(dom, grid)[cand]
    ub, inv = np.unique(bloque, return_inverse=True); inv = inv.ravel(); orden = np.argsort(inv, kind='stable')
    cortes = np.r_[0, np.cumsum(np.bincount(inv))]; celdas = [cand[orden[cortes[k]:cortes[k+1]]] for k in range(len(ub))]
    activos = np.flatnonzero(dcur > 0); qa = q[activos]; qa2 = (qa**2).sum(1)

    def ganancia(k):  # same as implicaciones.p1_curva
        c = climate[celdas[k]]; d2 = np.full(len(qa), np.inf)
        for i in range(0, len(c), 150):
            ci = c[i:i+150]; d2 = np.minimum(d2, (qa2[:, None]+(ci**2).sum(1)[None, :]-2*qa@ci.T).min(1))
        dk = np.zeros(len(q)); dk[activos] = np.sqrt(np.maximum(d2, 0))
        return float(peso@np.maximum(0, dcur-dk)), dk

    F = lambda d: {f'F{x}': float(peso[d <= x].sum()/W) for x in UMBRALES}
    heap = [(-ganancia(k)[0], k) for k in range(len(ub))]; heapq.heapify(heap)
    filas = [{'paso': 0, 'unidad': '', 'agregado_km2': 0., 'D': float(peso@dcur/W), **F(dcur)}]; km2 = 0.
    while heap and km2 < max_km2:
        g, k = heapq.heappop(heap); nueva, dk = ganancia(k)
        if heap and nueva < -heap[0][0]-1e-12:
            heapq.heappush(heap, (-nueva, k)); continue
        if nueva <= 0:
            break
        dcur = np.minimum(dcur, dk); km2 += area[celdas[k]].sum()/1e6
        filas.append({'paso': len(filas), 'unidad': int(ub[k]), 'agregado_km2': km2, 'D': float(peso@dcur/W), **F(dcur)})
        if len(filas) % 50 == 0:
            E.log(f"umbral: paso {len(filas)-1}, +{km2:.0f} km2, {filas[-1]}")
        if filas[-1][f'F{min(UMBRALES)}'] >= .9:
            break
    return filas


def sitios(unidades, dom, grid, bloque):
    """Potential sites: 8-neighbour components of the chosen microbasins, each microbasin in its majority component."""
    h, w = grid['height'], grid['width']; flat = dom['flat']
    malla = np.zeros(h*w, bool); malla[flat[np.isin(bloque, unidades)]] = True
    et = ndimage.label(malla.reshape(h, w), structure=np.ones((3, 3)))[0].reshape(-1)[flat]
    return len({int(np.bincount(et[bloque == b]).argmax()) for b in unidades})


if __name__ == '__main__':
    filas = curva()
    ref = list(csv.DictReader(open(I.DST/'P1_curva_esfuerzo.csv', encoding='utf-8-sig')))
    assert [r['unidad'] for r in ref[1:]] == [str(r['unidad']) for r in filas[1:len(ref)]], 'greedy sequence differs from P1'
    E.write_csv(I.DST/'P1_umbral_curva.csv', filas)
    prep, dom, area, *_ = I._contexto()
    grid = json.loads((prep/'preparacion.json').read_text(encoding='utf-8'))['grid']; bloque = I._microcuenca(dom, grid)
    union = np.load(prep/'superficie_union.npy'); red = np.load(I.DST/'dmin_por_modalidad.npz')['red']
    total = area.sum()/1e6; exacto = np.minimum(union, area).sum()/1e6; falta = .3*total-exacto; libre = np.maximum(area-union, 0)
    out = []
    for x in UMBRALES:
        fila = {'umbral': x, 'F_territorio_inicial_pct': 100*filas[0][f'F{x}'],
                'vacios_km2': float(libre[red > x].sum()/1e6), 'vacios_pct_no_protegido': float(100*libre[red > x].sum()/libre.sum())}
        for meta in (.8, .9):
            r = next((r for r in filas if r[f'F{x}'] >= meta), None)
            m = int(100*meta)
            fila.update({f'{k}_{m}': None for k in ('pasos', 'km2', 'pct_falta30', 'protegido_pct', 'sitios')})
            if r is None or r['paso'] == 0:  # None: not reached; paso 0: already met by the real network
                fila.update({f'pasos_{m}': r and 0, f'km2_{m}': r and 0.}); continue
            uni = [f['unidad'] for f in filas[1:r['paso']+1]]
            fila.update({f'pasos_{m}': r['paso'], f'km2_{m}': r['agregado_km2'], f'pct_falta30_{m}': 100*r['agregado_km2']/falta,
                         f'protegido_pct_{m}': 100*(exacto+r['agregado_km2'])/total, f'sitios_{m}': sitios(uni, dom, grid, bloque)})
        out.append(fila); E.log(f'umbral: {fila}')
    E.write_csv(I.DST/'P1_umbral_esfuerzo.csv', out)
    E.save_json(I.DST/'P1_umbral_esfuerzo.json', {'protegido_exacto_km2': exacto, 'continental_km2': total,
                'falta_30_km2': falta, 'filas': out, 'criterio': __doc__.split('\n\n')[1]})
