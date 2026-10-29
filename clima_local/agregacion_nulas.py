"""Spatial aggregation and climatic redundancy of the real network and of the N2c null networks (Table S5).

The 999 continental N2c null networks are rebuilt with the same seed and inputs as
resultados_v7/04_redes_nulas/continental_superposicion_limitada/N2c (same networks; checked against the saved surface of
every network and the saved d_min of the first and last). D-bar is not recomputed. For the real network and each null:
- aggregation: for each area, the edge-to-edge distance (km) to the nearest other area of the network, between the
  centres of their boundary cells (0 = shared cells); contiguous = nearest area at <= 1.5 km (adjacent cells, diagonal
  included).
- climatic redundancy: for each area, the mean d (same transformed climate as D-bar) from its cells to the nearest
  climate in the rest of the network (leave-one-out); low = its climate is already in other areas. Up to 200 cells per
  area, drawn with a fixed generator.
Network values: means over areas, and by dominant ecoregion (kept by N2c). Test: prueba() of ejecutar_investigacion
(two-sided; SES < 0 = the real network is more aggregated or more redundant than chance).
Writes resultados_v7/04_redes_nulas/agregacion_N2c/. Run: python agregacion_nulas.py [--redes 999]
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import ejecutar_investigacion as E

RAIZ = Path(__file__).resolve().parents[1]
O = RAIZ / 'resultados_v7/02_sesgo_precipitacion/original'
M = RAIZ / 'resultados_v7/03_mascara_natural'
NULAS = RAIZ / 'resultados_v7/04_redes_nulas/continental_superposicion_limitada/N2c'
SALIDA = RAIZ / 'resultados_v7/04_redes_nulas/agregacion_N2c'
CELDAS_MAX = 200
CONTIGUA_KM = 1.5
KM_GRADO = 111.32


class Medidor:
    def __init__(self, w, flat, climate, transform):
        self.w, self.flat, self.climate = w, flat, climate
        r, c = flat // w, flat % w
        lat = transform[5] + (r + .5) * transform[4]; lon = transform[2] + (c + .5) * transform[0]
        self.xy = np.c_[lon * KM_GRADO * np.cos(np.radians(lat)), lat * KM_GRADO]
        self.rng_seed = E.SEED + 2002

    def borde(self, cells):
        """Boundary cells of a footprint (a 4-neighbour outside it)."""
        f = np.sort(self.flat[cells])
        dentro = [np.isin(f + k, f, assume_unique=True) for k in (1, -1, self.w, -self.w)]
        return cells[np.argsort(self.flat[cells])][~np.logical_and.reduce(dentro)]

    def vecino(self, refs):
        """Edge-to-edge distance (km) from each area to the nearest other area; bounding boxes prune exact searches."""
        pts = [self.xy[self.borde(c)] for c in refs]
        lo = np.array([p.min(0) for p in pts]); hi = np.array([p.max(0) for p in pts])
        trees = [cKDTree(p) for p in pts]; out = np.empty(len(refs))
        for j, p in enumerate(pts):
            gap = np.hypot(*np.maximum(0, np.maximum(lo - hi[j], lo[j] - hi)).T); gap[j] = np.inf
            best = np.inf
            for i in np.argsort(gap):
                if gap[i] >= best:
                    break
                best = min(best, float(trees[i].query(p, distance_upper_bound=best)[0].min()))
            out[j] = best
        return out

    def redundancia(self, refs):
        """Mean leave-one-out d of each area: its cells against the climates of all other areas (divide and conquer:
        each half of the list is queried against a tree of the other half)."""
        rng = np.random.default_rng(self.rng_seed)
        q = [c if len(c) <= CELDAS_MAX else rng.choice(c, CELDAS_MAX, replace=False) for c in refs]
        best = [np.full(len(x), np.inf) for x in q]
        pila = [(0, len(refs))]
        while pila:
            lo, hi = pila.pop()
            if hi - lo < 2:
                continue
            mid = (lo + hi) // 2
            for a, b, c, d in ((lo, mid, mid, hi), (mid, hi, lo, mid)):
                tree = cKDTree(self.climate[np.unique(np.concatenate(refs[c:d]))], leafsize=32)
                dd, _ = tree.query(self.climate[np.concatenate(q[a:b])], k=1, workers=8)
                for j, part in zip(range(a, b), np.split(dd, np.cumsum([len(x) for x in q[a:b]])[:-1])):
                    best[j] = np.minimum(best[j], part)
            pila += [(lo, mid), (mid, hi)]
        return np.array([x.mean() for x in best])


def resumen_red(vec, red, peso, dominante, nombres):
    fila = {'vecino_km_media': vec.mean(), 'vecino_km_mediana': float(np.median(vec)),
            'contiguas_pct': 100 * (vec <= CONTIGUA_KM).mean(), 'redundancia_media': red.mean(),
            'redundancia_ponderada': float(np.average(red, weights=peso))}
    for k, n in enumerate(nombres):
        m = dominante == k
        if m.sum() >= 2:
            fila[f'vecino_km_media|{n}'] = vec[m].mean(); fila[f'redundancia_media|{n}'] = red[m].mean()
    return fila


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--redes', type=int, default=999)
    args = ap.parse_args()
    SALIDA.mkdir(parents=True, exist_ok=True)
    prep = O / 'preparados'
    grid = json.loads((prep / 'preparacion.json').read_text(encoding='utf-8'))['grid']
    flat = np.load(prep / 'dominio.npz')['flat']
    climate = np.load(O / 'modelo' / 'clima_transformado_cond100.npy')
    grupo, nombres = E.estratos_muestreo(prep)
    natural = np.load(M / 'natural_dominio.npz')['natural']
    umbral = json.loads((M / 'resumen.json').read_text(encoding='utf-8'))['umbral_aceptacion_F4']
    guardado = np.load(NULAS / 'redes_aleatorias.npz')
    with open(NULAS / 'diagnostico_huellas.csv', encoding='utf-8-sig') as f:
        superficie = {int(r['red']): float(r['superficie_km2']) for r in csv.DictReader(f)}
    area = np.load(prep / 'dominio.npz')['area']
    med = Medidor(grid['width'], flat, climate, grid['transform'])
    vec_nulo, red_nulo, filas = [], [], []
    inicio = time.perf_counter()

    def por_red(i, refs, hus, evaluate):
        s = area[np.unique(np.concatenate(refs))].sum() / 1e6
        assert abs(s - superficie[i + 1]) < 1e-6 * s, f'red {i + 1}: {s} km2 frente a {superficie[i + 1]} guardados'
        if i == 0 or i == len(guardado['azar_dmin']) - 1:
            assert np.allclose(evaluate(np.concatenate(refs))[0], guardado['azar_dmin'][i]), f'red {i + 1}: d_min distinto'
        if not filas:  # red real, una vez
            real = [hu['index'] for hu in hus]
            por_red.dominante = np.array([hu['dominante'] for hu in hus]); por_red.peso = np.array([hu['area'].sum() for hu in hus])
            por_red.real = (med.vecino(real), med.redundancia(real))
            filas.append({'red': 0, **resumen_red(*por_red.real, por_red.peso, por_red.dominante, nombres)})
        v, r = med.vecino(refs), med.redundancia(refs)
        vec_nulo.append(v); red_nulo.append(r)
        filas.append({'red': i + 1, **resumen_red(v, r, por_red.peso, por_red.dominante, nombres)})
        E.log(f'Agregacion red {i + 1}/{args.redes}: vecino {v.mean():.2f} km, redundancia {r.mean():.3f} '
              f'({time.perf_counter() - inicio:.0f} s)')

    E.cobertura(O, 100, args.redes, diseno='N2c', dst=SALIDA / 'tmp', natural=natural, umbral=umbral, limitar=True,
                por_red=por_red)
    E.write_csv(SALIDA / 'por_red.csv', filas)
    np.savez_compressed(SALIDA / 'por_area.npz', vecino_real=por_red.real[0], redundancia_real=por_red.real[1],
                        vecino_nulo=np.array(vec_nulo), redundancia_nulo=np.array(red_nulo), dominante=por_red.dominante,
                        peso_m2=por_red.peso)
    pruebas = {}
    for k in filas[0]:
        if k == 'red':
            continue
        nulo = np.array([f[k] for f in filas[1:]]); real = filas[0][k]
        pruebas[k] = {'real': real, 'nulo_mediana': float(np.median(nulo)), 'nulo_p2_5': float(np.quantile(nulo, .025)),
                      'nulo_p97_5': float(np.quantile(nulo, .975)), **E.prueba(nulo, real)}
    E.save_json(SALIDA / 'resumen.json', {'n_redes': args.redes, 'n_areas': len(por_red.peso), 'celdas_max_por_area': CELDAS_MAX,
                                          'contigua_km': CONTIGUA_KM, 'segundos': time.perf_counter() - inicio,
                                          'nota': 'SES < 0: la red real esta mas agregada (vecino) o es mas redundante que el azar.',
                                          'pruebas': pruebas})
    (SALIDA / 'tmp').rmdir()
    E.log('Agregacion: listo')


if __name__ == '__main__':
    main()
