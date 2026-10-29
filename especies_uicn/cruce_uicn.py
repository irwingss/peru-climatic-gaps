"""Overlap of the IUCN and BirdLife range maps with the reading classes of dmin (Supplementary material S6, File S1).

For every species whose standard range (presence 1-2, origin 1-2, seasonal 1-3) reaches continental Peru or its maritime
domain: range area inside the domain, protected share, share of the range in each reading class of the unprotected
surface, and share of the global range inside Peru (endemism proxy, land groups only).
Writes resultados/especies_uicn_<GRUPO>.csv. Run: python -P -E cruce_uicn.py [GRUPO ...]
"""
import json, sys, time
from pathlib import Path
RECURSOS = Path(__import__('os').environ.get('RECURSOS_GENERALES', Path.home()/'Documents'/'Recursos generales'))  # datos de entrada fuera del repositorio
import numpy as np, pandas as pd, pyogrio, rasterio, shapely
from pyproj import Transformer
from rasterio import features

UICN = (RECURSOS/r'IUCN')
RAIZ = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / 'resultados'
GRUPOS = sys.argv[1:] or ['AMPHIBIANS', 'MAMMALS', 'REPTILES', 'PLANTS', 'MARINEFISH']
COLS = ['id_no', 'sci_name', 'presence', 'origin', 'seasonal', 'category', 'class', 'order_', 'family', 'genus',
        'marine', 'terrestria', 'freshwater', 'subspecies', 'subpop']
CLASES = [('similar', -1, .2), ('moderado', .2, .3), ('debil', .3, .5), ('poco_similar', .5, 1), ('no_analogo', 1, 99)]
BBOX = (-84.85, -20.3, -68.55, 0.07)        # land domain + maritime domain
# Birds: BirdLife/HBW range maps 2025 (one GeoPackage) and the HBW-BirdLife checklist v10 for the Red List category
BIRDLIFE = UICN / 'BIRDLIFE_DATAZONE'
BOTW = BIRDLIFE / 'BOTW_2025.gpkg'
CHECKLIST = BIRDLIFE / 'Handbook of the Birds of the World and BirdLife International Digital Checklist of the Birds of the World_Version_10.xlsx'
A6933 = Transformer.from_crs(4326, 6933, always_xy=True)
# Seabirds: these families and the phalaropes, when their range reaches the maritime domain. BirdLife maps carry no IUCN
# systems, so the seabirds that also use inland waters are listed here; they keep their land range (both domains), and the
# rest, like cetaceans, pinnipeds and sea turtles, are reported only for the maritime domain (figuras_tablas/especies.py, cargar)
FAMILIAS_MARINAS = {'Spheniscidae', 'Diomedeidae', 'Procellariidae', 'Hydrobatidae', 'Oceanitidae', 'Phaethontidae', 'Fregatidae',
                    'Sulidae', 'Phalacrocoracidae', 'Pelecanidae', 'Stercorariidae', 'Laridae'}
AVES_DULCEACUICOLAS = {'Nannopterum brasilianum', 'Larus serranus', 'Larus cirrocephalus', 'Larus maculipennis', 'Larus pipixcan',
                       'Larus dominicanus', 'Larus atricilla', 'Xema sabini', 'Rynchops niger', 'Phaetusa simplex',
                       'Sternula superciliaris', 'Gelochelidon nilotica', 'Chlidonias niger', 'Sterna hirundo', 'Sterna paradisaea',
                       'Stercorarius parasiticus', 'Stercorarius pomarinus', 'Phalaropus fulicarius', 'Phalaropus lobatus',
                       'Phalaropus tricolor'}


class Dominio:
    def __init__(self, base, capas):
        prep = base / 'preparados'
        z = np.load(prep / 'dominio.npz'); g = json.loads((prep / 'preparacion.json').read_text(encoding='utf-8'))['grid']
        self.flat, self.area = z['flat'], z['area']
        self.union = np.load(prep / 'superficie_union.npy'); self.free = np.maximum(self.area - self.union, 0)
        self.shape, self.aff = (g['height'], g['width']), rasterio.Affine(*g['transform'][:6])
        a = self.aff
        self.bounds = (a.c, a.f + a.e * self.shape[0], a.c + a.a * self.shape[1], a.f)
        self.dmin = {}
        for capa, tif in capas.items():
            with rasterio.open(tif) as ds:
                self.dmin[capa] = np.ravel(ds.read(1))[self.flat]

    def mascara(self, geoms, todo=False):
        r = features.rasterize(((x, 1) for x in geoms), out_shape=self.shape, transform=self.aff, dtype='uint8', all_touched=todo)
        return r.reshape(-1)[self.flat] > 0

    def cruzar(self, geoms, pref):
        g = [x for x in shapely.clip_by_rect(geoms, *self.bounds) if not x.is_empty]
        if not g:
            return {}
        m = self.mascara(g)
        if not m.any():  # ranges narrower than a cell: count every cell they touch
            m = self.mascara(g, True)
        if not m.any():
            return {}
        tot = self.area[m].sum()
        r = {f'{pref}km2': tot / 1e6, f'{pref}pct_protegido': 100 * self.union[m].sum() / tot, f'{pref}celdas_tocadas': int(m.sum())}
        for capa, d in self.dmin.items():
            p = f'{pref}{capa}_' if len(self.dmin) > 1 else pref
            for nombre, lo, hi in CLASES:
                r[f'{p}pct_{nombre}'] = 100 * self.free[m & (d > lo) & (d <= hi)].sum() / tot
            r[f'{p}pct_mas_alla_similar'] = 100 * self.free[m & (d > .2)].sum() / tot
        return r


def km2(g):
    """Equal-area (EASE-Grid 2.0) area in km2."""
    g = shapely.clip_by_rect(g, -180, -85, 180, 85)
    return shapely.area(shapely.transform(g, lambda xy: np.column_stack(A6933.transform(xy[:, 0], xy[:, 1])))) / 1e6


def leer_aves():
    """BirdLife polygons reaching the bbox, with the IUCN fields that the other groups carry (id_no = SISRecID).
    Seabirds (FAMILIAS_MARINAS and the phalaropes) are also crossed with the maritime domain."""
    meta, _, wkb, data = pyogrio.raw.read(BOTW, bbox=BBOX, columns=['sisid', 'sci_name', 'presence', 'origin', 'seasonal'])
    df = pd.DataFrame(dict(zip(meta['fields'], data))); df['geom'] = shapely.from_wkb(wkb)
    df = df.rename(columns={'sisid': 'id_no'})
    ck = pd.read_excel(CHECKLIST, header=3).dropna(subset=['SISRecID', 'Seq.'])
    ck = ck.set_index(ck.SISRecID.astype(int))
    df['category'] = df.id_no.map(ck['2025 IUCN Red List category'].str.replace(' (PE)', '', regex=False))  # CR (possibly extinct) = CR
    df['order_'] = df.id_no.map(ck['Order']); df['family'] = df.id_no.map(ck['Family name'])
    df['class'] = 'AVES'; df['subpop'] = np.nan
    marina = df.family.isin(FAMILIAS_MARINAS) | df.sci_name.str.startswith('Phalaropus ')
    df['marine'] = np.where(marina, 'true', 'false'); df['terrestria'] = 'true'
    df['freshwater'] = np.where(df.sci_name.isin(AVES_DULCEACUICOLAS), 'true', 'false')
    return df


def main():
    tierra = Dominio(RAIZ / 'resultados_climaticos/20261011',
                     {'t': RAIZ / 'resultados_climaticos/20261011/mapas_distancia/cond100/sistema_distancia_minima.tif'})
    mb = RAIZ / 'resultados_marinos/20261011'
    mar = Dominio(mb / 'superficie', {c: mb / c / 'mapas_distancia/cond100/sistema_distancia_minima.tif' for c in ('superficie', 'fondo')})
    _, _, wkb, _ = pyogrio.raw.read(RAIZ / 'resultados_climaticos/20261011/preparados/peru_continental.geojson')
    peru = shapely.from_wkb(wkb)[0]; shapely.prepare(peru)
    OUT.mkdir(exist_ok=True)
    for grupo in GRUPOS:
        t0, filas, partes = time.time(), [], []
        for f in sorted((UICN / grupo).glob('*_PART*.shp')):
            meta, _, wkb, data = pyogrio.raw.read(f, bbox=BBOX, columns=COLS)
            df = pd.DataFrame(dict(zip(meta['fields'], data))); df['geom'] = shapely.from_wkb(wkb)
            partes.append(df)
        df = leer_aves() if grupo == 'BIRDS' else pd.concat(partes, ignore_index=True)
        estandar = df.presence.isin([1, 2]) & df.origin.isin([1, 2]) & df.seasonal.isin([1, 2, 3])
        for id_no, s in df.groupby('id_no'):
            e = s[estandar.loc[s.index]]
            base = s[s.subpop.isna()] if s.subpop.isna().any() else s
            r = {'grupo': grupo, 'id_no': int(id_no), 'sci_name': s.sci_name.iat[0], 'clase': s['class'].iat[0],
                 'orden': s.order_.iat[0], 'familia': s.family.iat[0], 'categoria': base.category.iat[0],
                 'marina': str(s.marine.iat[0]).lower() == 'true', 'terrestre': str(s.terrestria.iat[0]).lower() == 'true',
                 'dulceacuicola': str(s.freshwater.iat[0]).lower() == 'true',
                 'n_poligonos': len(s), 'n_poligonos_estandar': len(e),
                 'codigos_excluidos': ';'.join(sorted({f'{p}/{o}/{z}' for p, o, z in s.loc[~estandar.loc[s.index], ['presence', 'origin', 'seasonal']].itertuples(index=False)}))}
            if len(e):
                g = e.geom.values
                if grupo == 'BIRDS':
                    # global area as the sum of the polygons, without the global union (hours for widespread
                    # species); BirdLife draws one polygon per presence/origin/season, so the standard ones barely overlap
                    km2_global = float(km2(g).sum())
                    g = shapely.clip_by_rect(g, *BBOX); g = g[~shapely.is_empty(g)]
                bad = ~shapely.is_valid(g)
                if bad.any():
                    g = g.copy(); g[bad] = shapely.make_valid(g[bad])
                u = shapely.union_all(g) if len(g) > 1 else g[0]
                if grupo != 'MARINEFISH':  # land presence decided on the vector outline, so border slivers of the raster do not count
                    r['km2_en_peru'] = float(km2(shapely.intersection(u, peru)))
                    if r['km2_en_peru'] > 0:
                        r['km2_global'] = km2_global if grupo == 'BIRDS' else float(km2(u))
                        r['pct_global_en_peru'] = 100 * r['km2_en_peru'] / r['km2_global']
                        r.update(tierra.cruzar(g, 'tierra_'))
                if grupo == 'MARINEFISH' or r['marina']:
                    r.update(mar.cruzar(g, 'mar_'))
                    if r.get('mar_km2'):  # share of the global range inside the maritime domain (raster area / polygon area)
                        r.setdefault('km2_global', km2_global if grupo == 'BIRDS' else float(km2(u)))
                        r['pct_global_en_mar'] = 100 * r['mar_km2'] / r['km2_global']
                    elif grupo == 'BIRDS':  # inland species of seabird families (river terns) are not marine
                        r['marina'] = False
                if grupo == 'BIRDS' and (r.get('km2_en_peru', 0) > 0 or r.get('mar_km2')):
                    np.save(OUT / 'rangos' / f'{int(id_no)}.npy', shapely.to_wkb(g), allow_pickle=True)  # range cache read by figuras_tablas/especies.py, rango
            filas.append(r)
        res = pd.DataFrame(filas)
        res.to_csv(OUT / f'especies_uicn_{grupo}.csv', index=False, encoding='utf-8-sig')
        print(grupo, len(res), 'especies candidatas,', int(res.get('tierra_km2', pd.Series()).notna().sum()), 'en tierra,',
              int(res.get('mar_km2', pd.Series()).notna().sum()), f'en mar, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
