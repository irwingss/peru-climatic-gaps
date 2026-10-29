"""Practical magnitude of the difference between the real network and the N2c null networks (Table S4).

Reads only saved results (cifras.json of the figures: D-bar and F(0.2) of the real and null networks by stratum, and the
d-vs-separation curve of Fig. S4 for each layer). For each stratum of Table S3 it gives
- the relative difference of D-bar (real / null median - 1);
- D-bar read as a separation: the distance between pixels of a row at which the median d equals D-bar, interpolated
  log-log on the curve of Fig. S4 (None below its first lag, i.e. closer than neighbouring pixels);
- the extra territory beyond 0.2: (F_null - F_real) x surface of the stratum, F over the whole territory, protected
  surface included, as in Section 2.3 (median and 95% interval of the nulls).
Writes resultados_v7/04_redes_nulas/magnitud_N2c.json. Run: python magnitud_nulas.py
"""
import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
C = json.loads((RAIZ / 'figuras_tablas/insumos_figuras/cifras.json').read_text(encoding='utf-8'))  # cifras_resultados.py
SALIDA = RAIZ / 'resultados_v7/04_redes_nulas/magnitud_N2c.json'


def km_de(d, escala):
    """Separation (km) at which the median d of Fig. S4 equals d; None below the first lag."""
    km = np.array([f['km'] for f in escala['curva']]); dd = np.array([f['d'] for f in escala['curva']])
    if d < dd[0]:
        return None
    return float(np.exp(np.interp(np.log(d), np.log(dd), np.log(km))))


def filas(capa, nulas, superficies, escala):
    out = {}
    for k, n in nulas.items():
        if n['direccion'] == 'no evaluable':
            continue
        f = n['cobertura']['d0.2']; s = superficies[k]
        out[k] = {'D_real': n['D_real'], 'D_nulo': n['D_azar_mediana'], 'ses': n['ses'], 'p_holm': n['p_holm'],
                  'direccion': n['direccion'], 'dif_relativa_pct': 100 * (n['D_real'] / n['D_azar_mediana'] - 1),
                  'km_real': km_de(n['D_real'], escala), 'km_nulo': km_de(n['D_azar_mediana'], escala),
                  'F_real_pct': f['real'], 'F_nulo_pct': f['azar_mediana'], 'F_nulo_p2_5': f['azar_p2_5'], 'F_nulo_p97_5': f['azar_p97_5'],
                  'km2_estrato': s, 'km2_extra_mas_alla_0.2': (f['azar_mediana'] - f['real']) / 100 * s,
                  'km2_extra_ic95': [(f['azar_p2_5'] - f['real']) / 100 * s, (f['azar_p97_5'] - f['real']) / 100 * s]}
    return out


def superficies(bloque, nacional):
    s = {k: v['km2_total'] for k, v in bloque['ecorregiones' if 'ecorregiones' in bloque else 'estratos'].items()}
    s[nacional] = bloque['peru' if 'peru' in bloque else 'todo']['km2_total']
    return s


if __name__ == '__main__':
    R = {}
    for capa, bloque in (('continental', C['continental']), ('superficie', C['maritimo']['superficie']),
                         ('fondo', C['maritimo']['fondo'])):
        nulas = bloque['nulas']; nac = next(k for k in nulas if k in ('Perú', 'Mar peruano', 'todo', 'Peru'))
        R[capa] = filas(capa, nulas, superficies(bloque, nac), bloque['escala'])
    # check against the text: national continental D-bar 0.151 vs 0.133 and F(0.2) 69.9% (Section 3.3)
    p = R['continental']['Perú']
    assert round(p['D_real'], 3) == .151 and round(p['D_nulo'], 3) == .133 and round(p['F_real_pct'], 1) == 69.9
    SALIDA.write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding='utf-8')
    for capa, r in R.items():
        for k, v in r.items():
            km = lambda x: '<1 lag' if x is None else f'{x:.1f}'
            print(f"{capa:11s} {k[:28]:28s} D {v['D_real']:.3f}/{v['D_nulo']:.3f} ({v['dif_relativa_pct']:+.0f}%) "
                  f"km {km(v['km_real'])}/{km(v['km_nulo'])}  F {v['F_real_pct']:.1f}/{v['F_nulo_pct']:.1f} "
                  f"extra {v['km2_extra_mas_alla_0.2']:,.0f} [{v['km2_extra_ic95'][0]:,.0f}, {v['km2_extra_ic95'][1]:,.0f}] "
                  f"SES {v['ses']:+.1f} {v['direccion']}")
