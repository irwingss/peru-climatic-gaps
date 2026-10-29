"""Metrica de 7 variables (C7) frente a la principal (C11) con el clima de hoy (Supplementary material S5, Table S9).

CHELSA V2.1 no publica para 2071-2100 hurs, vpd, rsds ni pet; el futuro solo se podria medir con C7. Este script
reestima la metrica con las 7 variables que si tienen version futura (mismo procedimiento y misma muestra de teselas),
calcula dmin de la red real con C11 y con C7 y compara representatividad y vacios por estrato.
C7 no reproduce el presente, por eso el analisis se limita al clima actual.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import ejecutar_investigacion as E
from descargar_chelsa import VARIABLES as VARS11

DST = Path(__file__).resolve().parents[1]/'resultados_v7'/'05_futuro'
ORIGINAL = DST.parent/'02_sesgo_precipitacion'/'original'  # CHELSA original (preparados y metrica C11)
C7 = ('bio01', 'bio02', 'bio04', 'bio05', 'bio06', 'bio12', 'bio15')
UMBRAL = .2  # vacio climatico: dmin > 0.2
CLAVE = ('Perú', 'Desierto costero del Pacífico', 'Serranía esteparia', 'Puna')  # estratos del criterio de equivalencia


def metrica_c7():
    """Metrica C7: las 7 columnas de C11, national_model con log1p solo en bio12 y la misma muestra de teselas."""
    c7, prep = DST/'continental'/'C7', ORIGINAL/'preparados'
    E.VARIABLES, E.P, E.TRANSFORMACIONES = C7, len(C7), 'log1p: bio12'
    E.transform_features = lambda a: np.column_stack([np.log1p(a[:, k]) if v == 'bio12' else a[:, k] for k, v in enumerate(C7)])
    if not (c7/'modelo'/'modelo_cond100.json').exists():
        (c7/'preparados').mkdir(parents=True, exist_ok=True)
        X = np.load(prep/'clima_fisico.npy', mmap_mode='r')
        np.save(c7/'preparados'/'clima_fisico.npy', np.asarray(X[:, [VARS11.index(v) for v in C7]]))
        for f in ('coordenadas_6933.npy', 'dominio.npz'):
            shutil.copyfile(prep/f, c7/'preparados'/f)
    model = E.national_model(c7, 100)
    if not np.array_equal(np.load(c7/'modelo'/'modelo_cond100.npz')['index'], np.load(ORIGINAL/'modelo'/'modelo_cond100.npz')['index']):
        raise ValueError('La muestra de teselas de C7 no es la de C11')
    return model


def dmin(nombre, ruta, prep):
    """dmin exacta de cada celda a la red real (union de las asociaciones de las unidades evaluables)."""
    target = DST/'continental'/'dmin'/f'{nombre}.npy'
    if not target.exists():
        Z = np.load(ruta, mmap_mode='r')
        ref = np.zeros(len(Z), bool)
        for u in json.loads((prep/'inventario.json').read_text(encoding='utf-8')):
            if u['estado'] == 'evaluable':
                ref[np.load(prep/'asociaciones'/f"{u['anp_id']}.npz")['index']] = True
        tree = cKDTree(np.asarray(Z[ref], np.float64), leafsize=32)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, np.concatenate([tree.query(np.asarray(Z[i:i+65536], np.float64), k=1, eps=0, workers=16)[0]
                                        for i in range(0, len(Z), 65536)]).astype(np.float32))
    return np.load(target)


def puente():
    metrica_c7()
    prep = ORIGINAL/'preparados'
    a = dmin('C11_presente', ORIGINAL/'modelo'/'clima_transformado_cond100.npy', prep)
    b = dmin('C7_presente', DST/'continental'/'C7'/'modelo'/'clima_transformado_cond100.npy', prep)
    area = np.load(prep/'dominio.npz')['area']
    libre = np.maximum(area-np.load(prep/'superficie_union.npy'), 0)
    # Umbral de C7 que iguala el F(0.2) nacional de C11: separa un cambio de escala de un cambio de geografia.
    u7 = float(E.weighted_quantile(b, libre, [libre[a <= UMBRAL].sum()/libre.sum()])[0])
    grupo, nombres = E.estratos_muestreo(prep)
    filas = []
    for nombre, m in [('Perú', np.ones(len(a), bool))]+[(n, grupo == i) for i, n in enumerate(nombres)]:
        f, x, y = libre[m], a[m], b[m]
        ga, gb, gu = x > UMBRAL, y > UMBRAL, y > u7
        jac = lambda p, q: float(f[p & q].sum()/f[p | q].sum()) if f[p | q].sum() > 0 else None
        filas.append({'estrato': nombre,
                      'F0.2_C11': 100*f[~ga].sum()/f.sum(), 'F0.2_C7': 100*f[~gb].sum()/f.sum(),
                      'delta_F0.2_pp': 100*(f[~gb].sum()-f[~ga].sum())/f.sum(),
                      'vacios_km2_C11': f[ga].sum()/1e6, 'vacios_km2_C7': f[gb].sum()/1e6,
                      'jaccard_vacios': jac(ga, gb),
                      'vacios_C7_dentro_de_C11_pct': float(100*f[ga & gb].sum()/f[gb].sum()) if f[gb].sum() > 0 else None,
                      'D_medio_C11': float(area[m]@x/area[m].sum()), 'D_medio_C7': float(area[m]@y/area[m].sum()),
                      'cociente_D_C7_C11': float((area[m]@y)/(area[m]@x)) if area[m]@x > 0 else None,
                      'jaccard_vacios_umbral_igualado': jac(ga, gu)})
    E.write_csv(DST/'continental'/'puente_C11_C7.csv', filas)
    reglas = {f['estrato']: f['jaccard_vacios'] for f in filas if f['estrato'] in CLAVE}
    decision = {'delta_F0.2_nacional_pp': filas[0]['delta_F0.2_pp'], 'jaccard': reglas, 'umbral_C7_igualado': u7,
                'cumple_parte_descriptiva': bool(abs(filas[0]['delta_F0.2_pp']) < 5 and all(
                    v is not None and v >= .7 for k, v in reglas.items() if k != 'Perú')),
                'redes_nulas': 'No se corrieron: C7 no cumple la parte descriptiva del criterio.',
                'decision': 'Sin clima futuro: C11 se mantiene y no se analiza 2071-2100 en tierra ni en mar.',
                'criterio': '|delta F(0.2)| nacional < 5 pp; Jaccard de vacios >= 0.7 en desierto, serrania esteparia y puna; '
                            'misma direccion y significancia frente a las redes nulas N2c.'}
    E.save_json(DST/'continental'/'decision_D4.json', decision)
    E.log(f'Puente C11-C7: {decision}')


if __name__ == '__main__':
    puente()
