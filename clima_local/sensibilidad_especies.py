"""Sensitivity of the species-led prioritization to the target rule and to the climatic measure (Table S12).

(a) Species-only solutions with a flat target of 17% or 30% of each range (B17f, B30f; prioritizacion.R) instead of the
    range-size-scaled targets of Rodrigues et al. (2004).
(b) Climatic gain measured as F(0.2), a coverage measure, besides D̄: gain in percentage points over the current network
    and share of the gain of the greedy curve at the same area.
Random sets: the same 999 draws as azar_prioritizacion.py (same seed and order, checked by their area), so climate comes
from azar_999.csv and only the species scores under the flat targets are recomputed.

python sensibilidad_especies.py  -> resultados_v7/09_prioritizacion/sensibilidad_metas.csv and sensibilidad_metas.json
"""
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import azar_prioritizacion as AZ
import implicaciones as I
import prioritizacion as P

ESC = ('B', 'B17f', 'B30f', 'C17', 'D17')
METAS = {'rodrigues': None, 'fija17': .17, 'fija30': .30}


def p_bilateral(x, r):
    return min(1., 2 * (1 + min((r <= x).sum(), (r >= x).sum())) / (len(r) + 1))


def main():
    prep, dom, area, *_ = I._contexto()
    km2 = area / 1e6
    climate = np.load(I.CONT / 'modelo' / 'clima_transformado_cond100.npy')
    z = np.load(I.NULAS / 'redes_aleatorias.npz'); muestra, peso, d0 = z['muestra'], z['peso'], z['real_dmin'].astype(np.float64)
    q = climate[muestra]; W = peso.sum()
    C = np.load(P.ENT / 'celdas.npz'); pu, celda = C['pu'], C['candidato']
    cu = pd.read_csv(I.DST / 'P1_curva_esfuerzo.csv')
    ev = pd.read_csv(P.DST / 'evaluacion.csv').set_index('escenario')
    F = pd.read_csv(P.ENT / 'features_todo.csv'); R = pd.read_csv(P.ENT / 'rij_todo.csv')
    e = F[F.tipo == 'especie']
    D0, F0 = float(peso @ d0 / W), float(peso[d0 <= I.UMBRAL].sum() / W)

    def especies(elegidas):
        x = e.id.map(R[R.pu.isin([0, *elegidas])].groupby('species').amount.sum()).fillna(0)
        return {m: 100 * float(np.minimum(x / ((e.meta_especie if f is None else f) * e.total_km2), 1).mean())
                for m, f in METAS.items()}

    # random sets, redrawn exactly as in azar_prioritizacion.main()
    az = pd.read_csv(P.DST / f'azar_{AZ.N_AZAR}.csv')
    por = pd.Series(km2[celda]).groupby(pu[celda]).sum(); por = por[por.index > 0]
    ids, sup_id = por.index.to_numpy(), por.to_numpy()
    rng = np.random.default_rng(AZ.SEED); azsp = {}
    for k in ('19k', '41k'):
        sup = float(ev.loc[f'B_{k}', 'superficie_km2']); g = az[az.presupuesto == k].reset_index(drop=True); filas = []
        for i in range(AZ.N_AZAR):
            o = rng.permutation(len(ids)); n = np.searchsorted(sup_id[o].cumsum(), sup) + 1; el = ids[o[:n]]
            if i in (0, AZ.N_AZAR - 1):   # same draw as the stored one
                assert abs(km2[celda & np.isin(pu, el)].sum() - g.superficie_km2[i]) < 1e-6
            filas.append(especies(el))
        azsp[k] = pd.DataFrame(filas)
        assert np.allclose(azsp[k].rodrigues, g.especies_meta_media_pct)
        print(k, 'azar listo')

    res, filas = {}, []
    for k in ('19k', '41k'):
        g = az[az.presupuesto == k]
        Fz = 100 * (g.F02.to_numpy() - F0)
        for esc in ESC:
            s = pd.read_csv(P.DST / f'solucion_{esc}_{k}.csv'); el = s.loc[(s.solution_1 == 1) & (s.id > 0), 'id'].to_numpy()
            sel = np.flatnonzero(celda & np.isin(pu, el)); sup = float(km2[sel].sum())
            d = np.minimum(d0, cKDTree(climate[sel]).query(q, workers=8)[0])
            D, F2 = float(peso @ d / W), float(peso[d <= I.UMBRAL].sum() / W)
            Dc = float(np.interp(sup, [0, *cu.agregado_km2], [D0, *cu.D]))
            Fc = float(np.interp(sup, [0, *cu.agregado_km2], [F0, *cu.F02]))
            sp = especies(el)
            fila = {'presupuesto': k, 'escenario': esc, 'superficie_km2': sup, 'pct_en_vacios': 100 * float(km2[sel][C['vacio'][sel]].sum()) / sup,
                    'D': D, 'F02': F2, 'clima_D_pct_voraz': 100 * (D0 - D) / (D0 - Dc),
                    'F02_ganancia_pp': 100 * (F2 - F0), 'clima_F02_pct_voraz': 100 * (F2 - F0) / (Fc - F0),
                    'p_clima_D': p_bilateral(100 * (D0 - D) / (D0 - Dc), g.clima_optimo_pct.to_numpy()),
                    'p_F02': p_bilateral(100 * (F2 - F0), Fz),
                    **{f'especies_{m}_pct': v for m, v in sp.items()},
                    **{f'p_especies_{m}': p_bilateral(v, azsp[k][m].to_numpy()) for m, v in sp.items()}}
            if esc in ('B', 'C17', 'D17'):   # same solution as evaluacion.csv
                assert abs(D - ev.loc[f'{esc}_{k}', 'D']) < 1e-9
            filas.append(fila)
        Fcb = float(np.interp(float(ev.loc[f'B_{k}', 'superficie_km2']), [0, *cu.agregado_km2], [F0, *cu.F02]))
        q3 = lambda x: {'mediana': float(np.median(x)), 'p2_5': float(np.quantile(x, .025)), 'p97_5': float(np.quantile(x, .975))}  # noqa: E731
        res[k] = {'azar': {'clima_D_pct_voraz': q3(g.clima_optimo_pct), 'F02_ganancia_pp': q3(Fz),
                           'clima_F02_pct_voraz': q3(Fz / (100 * (Fcb - F0)) * 100),
                           **{f'especies_{m}_pct': q3(azsp[k][m]) for m in METAS}},
                  'voraz_F02_ganancia_pp': 100 * (Fcb - F0)}
    t = pd.DataFrame(filas); t.to_csv(P.DST / 'sensibilidad_metas.csv', index=False, float_format='%.6f')
    res['red_real'] = {'D': D0, 'F02': F0, **{f'especies_{m}_pct': v for m, v in especies([]).items()}}
    res['n_azar'] = AZ.N_AZAR
    (P.DST / 'sensibilidad_metas.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(t.round(3).to_string()); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
