"""Random sets of microbasins, the benchmark of the prioritization (Fig. 4, Tables S11 and S13).

Each of N_AZAR sets, with a fixed seed, draws whole natural microbasins of the candidate pool in random order until it reaches the area of the species-only solution
(B) of each budget, and both outcomes are measured on the same set:
- climate: D̄ and F(0.2) on the N2c sample, and the share of the greedy reduction of D̄ at the same area (greedy benchmark);
- species: mean share of the full Rodrigues target reached by the 271 species (each capped at 1) and full targets met.

python azar_prioritizacion.py          -> resultados_v7/09_prioritizacion/azar_999.csv and azar_resumen.json
python azar_prioritizacion.py minimo <esc>, then juntar -> azar_minimo_999.csv and azar_minimo_resumen.json (Table S13)
"""
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import implicaciones as I
import prioritizacion as P

N_AZAR = 999
SEED = P.SEED + 2          # P.SEED + 1 was used by the earlier 30 sets


def logro(F, R, elegidas):
    e = F[F.tipo == 'especie']
    lo = R[R.pu.isin([0, *elegidas])].groupby('species').amount.sum()
    meta = e.meta_especie * e.total_km2; x = e.id.map(lo).fillna(0)
    return float(np.minimum(x / meta, 1).mean()), int((x >= meta * .999).sum())


def main():
    prep, dom, area, *_ = I._contexto()
    km2 = area / 1e6
    climate = np.load(I.CONT / 'modelo' / 'clima_transformado_cond100.npy')
    z = np.load(I.NULAS / 'redes_aleatorias.npz'); muestra, peso, d0 = z['muestra'], z['peso'], z['real_dmin'].astype(np.float64)
    q = climate[muestra]; W = peso.sum()
    C = np.load(P.ENT / 'celdas.npz'); pu, celda = C['pu'], C['candidato']
    curva = pd.read_csv(I.DST / 'P1_curva_esfuerzo.csv')
    ev = pd.read_csv(P.DST / 'evaluacion.csv').set_index('escenario')
    F = pd.read_csv(P.ENT / 'features_todo.csv'); R = pd.read_csv(P.ENT / 'rij_todo.csv')
    D0 = float(peso @ d0 / W)
    assert abs(D0 - ev.loc['red real', 'D']) < 1e-9

    por = pd.Series(km2[celda]).groupby(pu[celda]).sum(); por = por[por.index > 0]
    ids, sup_id = por.index.to_numpy(), por.to_numpy()
    rng = np.random.default_rng(SEED); filas = []
    for k in ('19k', '41k'):
        sup = float(ev.loc[f'B_{k}', 'superficie_km2'])
        Dc = float(np.interp(sup, curva.agregado_km2, curva.D))
        for i in range(N_AZAR):
            o = rng.permutation(len(ids)); n = np.searchsorted(sup_id[o].cumsum(), sup) + 1; el = ids[o[:n]]
            sel = np.flatnonzero(celda & np.isin(pu, el))
            d = np.minimum(d0, cKDTree(climate[sel]).query(q, workers=8)[0])
            D = float(peso @ d / W)
            lg, mg = logro(F, R, el)
            filas.append({'presupuesto': k, 'conjunto': i, 'superficie_km2': float(km2[sel].sum()), 'D': D,
                          'F02': float(peso[d <= I.UMBRAL].sum() / W), 'clima_optimo_pct': 100 * (D0 - D) / (D0 - Dc),
                          'especies_meta_media_pct': 100 * lg, 'especies_meta_completa': mg})
        print(k, 'listo')
    t = pd.DataFrame(filas); t.to_csv(P.DST / f'azar_{N_AZAR}.csv', index=False)
    res = {}
    for k, g in t.groupby('presupuesto'):
        res[k] = {c: {'mediana': float(g[c].median()), 'p2_5': float(g[c].quantile(.025)), 'p97_5': float(g[c].quantile(.975)),
                      'min': float(g[c].min()), 'max': float(g[c].max())}
                  for c in ('clima_optimo_pct', 'especies_meta_media_pct', 'D', 'F02', 'superficie_km2')}
    res['n'] = N_AZAR; res['semilla'] = SEED
    (P.DST / 'azar_resumen.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1))


MINIMOS = ('A', 'B', 'C17', 'D10', 'D17', 'D30')


def minimo(esc):
    """Table S13: N_AZAR random sets up to the area of each minimum-set solution, D̄ and F(0.2) only.
    Scenario A draws from the candidate land in climatic gaps, the others from all candidate land. One process per scenario,
    so the six can run in parallel: python azar_prioritizacion.py minimo <esc>; then python azar_prioritizacion.py juntar"""
    prep, dom, area, *_ = I._contexto()
    km2 = area / 1e6
    climate = np.load(I.CONT / 'modelo' / 'clima_transformado_cond100.npy')
    z = np.load(I.NULAS / 'redes_aleatorias.npz'); muestra, peso, d0 = z['muestra'], z['peso'], z['real_dmin'].astype(np.float64)
    q = climate[muestra]; W = peso.sum()
    C = np.load(P.ENT / 'celdas.npz'); pu = C['pu']
    celda = C['candidato'] & (C['vacio'] if esc == 'A' else True)
    sup = float(pd.read_csv(P.DST / 'evaluacion.csv').set_index('escenario').loc[esc, 'superficie_km2'])
    por = pd.Series(km2[celda]).groupby(pu[celda]).sum(); por = por[por.index > 0]
    ids, sup_id = por.index.to_numpy(), por.to_numpy()
    rng = np.random.default_rng(SEED + 1 + MINIMOS.index(esc)); filas = []
    for i in range(N_AZAR):
        o = rng.permutation(len(ids)); n = np.searchsorted(sup_id[o].cumsum(), sup) + 1
        sel = np.flatnonzero(celda & np.isin(pu, ids[o[:n]]))
        d = np.minimum(d0, cKDTree(climate[sel]).query(q, workers=2)[0])
        filas.append({'escenario': esc, 'conjunto': i, 'superficie_km2': float(km2[sel].sum()), 'D': float(peso @ d / W),
                      'F02': float(peso[d <= I.UMBRAL].sum() / W)})
    pd.DataFrame(filas).to_csv(P.DST / f'azar_minimo_{esc}_{N_AZAR}.csv', index=False)
    print(esc, 'listo')


def juntar():
    t = pd.concat([pd.read_csv(P.DST / f'azar_minimo_{e}_{N_AZAR}.csv') for e in MINIMOS])
    t.to_csv(P.DST / f'azar_minimo_{N_AZAR}.csv', index=False)
    for e in MINIMOS:
        (P.DST / f'azar_minimo_{e}_{N_AZAR}.csv').unlink()
    res = {e: {c: {'mediana': float(g[c].median()), 'p2_5': float(g[c].quantile(.025)), 'p97_5': float(g[c].quantile(.975))}
               for c in ('D', 'F02', 'superficie_km2')} for e, g in t.groupby('escenario')}
    res['n'] = N_AZAR
    (P.DST / 'azar_minimo_resumen.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1))




if __name__ == '__main__':
    import sys
    {'minimo': lambda: minimo(sys.argv[2]), 'juntar': juntar}.get(sys.argv[1] if len(sys.argv) > 1 else '', main)()
