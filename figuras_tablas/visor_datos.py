"""Layers of the web atlas viewer (docs/, GitHub Pages): the analogue map of every protected area (File S2) and the map
layers of Figs. 1, 3 and 4. Rasters become indexed PNG (one byte per pixel, d classes of the figures) reprojected to Web
Mercator by rows and cropped to their extent, so Leaflet shows them as image overlays; potential sites and the microbasins
chosen by prioritizr become GeoJSON. Writes docs/capas/ and docs/capas/indice.json.
Run: python visor_datos.py
"""
import math
from PIL import Image
from rasterio import features
import pandas as pd
import pyogrio
from figuras_comun import *  # noqa: F401,F403  (shared constants, canvases and map helpers)

DST = REPO / 'docs' / 'capas'
RES = RAIZ / 'resultados_v7'
PRI = REPO / 'data' / 'prioritization'
VERDE = '#23745B'   # protected, as in the graphical abstract and the species maps

# palette index: 0 transparent, 1 no climate data, 2-7 d classes, 8 d > 1, 9 protected (d = 0)
PAL_D = [('#000000', 0), (SIN_DATO, .8)] + [(c, a) for c, a, _ in CLASES_D] + [('#ffffff', .55), (VERDE, .9)]
# Fig. 3a: gap classes 1-4 on natural land, 5 on transformed land (colours of figuras_articulo.fig3)
PAL_V = [('#000000', 0), ('#fb6a4a', 1), ('#fcae91', 1), ('#fde0d2', 1), ('#fff5f0', 1), ('#c9c8c2', .9)]


def png(idx, aff, pal, path):
    """Crop to the non-transparent pixels, resample rows to Web Mercator and save as indexed PNG; returns Leaflet bounds."""
    r, c = np.nonzero(idx)
    r0, r1, c0, c1 = r.min(), r.max() + 1, c.min(), c.max() + 1
    idx = idx[r0:r1, c0:c1]
    top, bot = aff.f + r0 * aff.e, aff.f + r1 * aff.e
    y = lambda lat: math.asinh(math.tan(math.radians(lat)))  # noqa: E731  (Mercator northing on the unit sphere)
    n = math.ceil(idx.shape[0] * (y(top) - y(bot)) / math.radians(top - bot))   # >= source rows, so no row is lost
    lat = np.degrees(np.arctan(np.sinh(y(top) - (np.arange(n) + .5) * (y(top) - y(bot)) / n)))
    rows = np.clip(((lat - top) / aff.e).astype(int), 0, idx.shape[0] - 1)
    im = Image.fromarray(idx[rows].astype(np.uint8), 'P')
    im.putpalette([v for h, _ in pal for v in (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))])
    im.save(path, optimize=True, transparency=bytes(round(255 * a) for _, a in pal))
    return [[round(bot, 6), round(aff.c + c0 * aff.a, 6)], [round(top, 6), round(aff.c + c1 * aff.a, 6)]]


def clases_d(base, d, sistema=False):
    """Palette index of d over the grid; the analogue maps of one area leave d > 1 and missing data transparent."""
    k = np.digitize(d, CORTES_D[1:-1]) + 2
    k[d > 1] = 8 if sistema else 0
    k[d == 0] = 9
    k[~np.isfinite(d)] = 0
    img = base.raster(k.astype(np.float32), fill=0)
    if sistema:
        img[base.domain == 0] = 1
    return np.nan_to_num(img).astype(np.uint8)


GRUPO = {'NACIONAL': 'ANP', 'REGIONAL': 'ACR', 'PRIVADA': 'ACP'}


def sitio(u, g):
    """Group, a point inside the area and its bounding box (for the zoom), never its outline."""
    p, (x0, y0, x1, y1) = g.representative_point(), g.bounds
    return {'grupo': GRUPO[u['modalidad']], 'centro': [round(p.y, 5), round(p.x, 5)],
            'caja': [[round(y0, 5), round(x0, 5)], [round(y1, 5), round(x1, 5)]]}


def atlas():
    (DST / 'atlas').mkdir(parents=True, exist_ok=True)
    out = []
    base = lienzo(TIERRA)
    areas = {a['anp_id']: a for a in CIFRAS['continental']['areas']}
    for u in base.inventory:
        uid = u['anp_id']
        if u['estado'] != 'evaluable':
            continue
        d = np.load(TIERRA / f'distancias/cond100/{uid}.npz')['distancia']
        a = areas[uid]
        out.append({'id': uid, 'dominio': 'land', 'nombre': nombre_en(u), 'modalidad': MOD_EN[u['modalidad']],
                    'km2': round(u['superficie_clima_m2'] / 1e6, 2), 'eco': ECO_EN[a['ecorregion']],
                    'km2_01': round(a['km2_d0.1']), 'km2_02': round(a['km2_d0.2']), **sitio(u, base.shapes[uid]),
                    'png': f'atlas/{uid}.png', 'bounds': png(clases_d(base, d), base.aff, PAL_D, DST / f'atlas/{uid}.png')})
        print('atlas', uid, flush=True)
    for capa, nombre in (('superficie', 'sea surface'), ('fondo', 'seafloor')):
        base = lienzo(MAR / capa)
        for u in base.inventory:
            uid = u['anp_id']
            d = np.load(MAR / capa / f'distancias/cond100/{uid}.npz')['distancia']
            a = CIFRAS['maritimo'][capa]['areas'][uid]
            out.append({'id': f'{uid}_{capa}', 'dominio': nombre, 'nombre': f'{nombre_en(u)} · {nombre}',
                        'modalidad': MOD_EN[u['modalidad']], 'km2_01': round(a['km2_d0.1']), 'km2_02': round(a['km2_d0.2']),
                        **sitio(u, base.shapes[uid]), 'png': f'atlas/{uid}_{capa}.png',
                        'bounds': png(clases_d(base, d), base.aff, PAL_D, DST / f'atlas/{uid}_{capa}.png')})
    return out


def figuras():
    capas = {}
    for k, b in (('fig1_land', TIERRA), ('fig1_sea_surface', MAR / 'superficie'), ('fig1_seafloor', MAR / 'fondo')):
        base = lienzo(b)
        d = base.leer(b / 'mapas_distancia/cond100/sistema_distancia_minima.tif')
        capas[k] = {'png': f'{k}.png', 'bounds': png(clases_d(base, d, sistema=True), base.aff, PAL_D, DST / f'{k}.png')}
    with rasterio.open(REPO / 'data/rasters/climatic_gaps_expandable_30s.tif') as ds:
        v = ds.read(1); v[v == 255] = 0
        capas['fig3_gaps'] = {'png': 'fig3_gaps.png', 'bounds': png(v, ds.transform, PAL_V, DST / 'fig3_gaps.png')}
    # Fig. 3a potential sites
    meta, _, wkb, attrs = pyogrio.raw.read(REPO / 'data/potential_sites/potential_sites.gpkg')
    geoms = shapely.set_precision(shapely.simplify(shapely.from_wkb(wkb), .002), 1e-5)
    limpio = lambda x: round(float(x), 3) if isinstance(x, (float, np.floating)) else int(x) if isinstance(x, np.integer) else x  # noqa: E731
    feats = [{'type': 'Feature', 'properties': {k: limpio(a[i]) for k, a in zip(meta['fields'], attrs)},
              'geometry': shapely.geometry.mapping(g)} for i, g in enumerate(geoms)]
    (DST / 'fig3_sites.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': feats}, ensure_ascii=False,
                                                       separators=(',', ':')), encoding='utf-8')
    # Fig. 4c microbasins chosen with 41 347 km² by the three plans, vectorized from the planning-unit grid
    base = lienzo(TIERRA)
    cel = np.load(PRI / 'entrada/celdas.npz'); pu = np.where(cel['candidato'], cel['pu'], 0)
    sel = {e: set(pd.read_csv(PRI / f'solucion_{e}_41k.csv').query('solution_1 == 1 and id > 0').id) for e in ('B', 'C17', 'D17')}
    todos = np.array(sorted(set().union(*sel.values())))
    grid = base.raster(np.where(np.isin(pu, todos), pu, 0).astype(np.float32), fill=0).astype(np.int32)
    partes = {}
    for g, v in features.shapes(grid, mask=grid > 0, transform=base.aff):
        partes.setdefault(int(v), []).append(shape(g))
    km2 = pd.Series(base.area / 1e6).groupby(pu).sum()
    feats = [{'type': 'Feature', 'properties': {'pu': i, 'km2': round(km2[i], 1), **{e: int(i in s) for e, s in sel.items()}},
              'geometry': shapely.geometry.mapping(shapely.set_precision(shapely.simplify(shapely.union_all(p), .002), 1e-5))}
             for i, p in partes.items()]
    (DST / 'fig4_microbasins.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': feats}, separators=(',', ':')))
    return capas


def teselas():
    """Protected-area outlines as XYZ raster tiles, one set per administration (ap/NACIONAL, ap/REGIONAL, ap/PRIVADA):
    SERNANP authorizes derived maps citing it, not the original vectors."""
    por_mod = {}
    for base in (lienzo(TIERRA), lienzo(MAR / 'superficie')):   # the seafloor areas are the same six as the sea surface
        for u in base.inventory:
            g = base.shapes.get(u['anp_id'])
            if g is not None and not g.is_empty:
                por_mod.setdefault(u['modalidad'], []).append(g)
    for mod, geoms in por_mod.items():
        teselas_grupo(geoms, DST / 'ap' / mod)


def teselas_grupo(geoms, dst, z0=4, z1=11):
    from PIL import ImageDraw
    arbol = shapely.STRtree(geoms)
    fill, line = (*bytes.fromhex(VERDE[1:]), 50), (*bytes.fromhex(VERDE[1:]), 255)
    for z in range(z0, z1 + 1):
        n = 2 ** z
        tx = lambda lon: (lon + 180) / 360 * n  # noqa: E731  (tile coordinates, Web Mercator)
        ty = lambda lat: (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n  # noqa: E731
        lon_t = lambda x: x / n * 360 - 180  # noqa: E731
        lat_t = lambda y: np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n))))  # noqa: E731
        tol = 360 / (256 * n) / 2   # half a screen pixel
        simples = shapely.simplify(np.array(geoms), tol)
        tiles = {(x, y) for g in geoms for x in range(int(tx(g.bounds[0])), int(tx(g.bounds[2])) + 1)
                 for y in range(int(ty(g.bounds[3])), int(ty(g.bounds[1])) + 1)}
        for x, y in tiles:
            caja = (lon_t(x), lat_t(y + 1), lon_t(x + 1), lat_t(y))
            idx = arbol.query(shapely.box(*caja))
            if not len(idx):
                continue
            im = Image.new('RGBA', (256, 256)); dr = ImageDraw.Draw(im)
            px = lambda c: [((tx(a) - x) * 256, (ty(b) - y) * 256) for a, b in c]  # noqa: E731
            ancho = 1 if z < 8 else 2
            for g in shapely.clip_by_rect(simples[idx], *caja):
                for p in getattr(g, 'geoms', [g]):
                    if p.geom_type == 'Polygon' and not p.is_empty:
                        dr.polygon(px(p.exterior.coords), fill=fill)
                        for h in p.interiors:
                            dr.polygon(px(h.coords), fill=(0, 0, 0, 0))
            for g in shapely.clip_by_rect(shapely.boundary(simples[idx]), *caja):
                for l in getattr(g, 'geoms', [g]):
                    if not l.is_empty:
                        dr.line(px(l.coords), fill=line, width=ancho)
            if im.getbbox():
                (dst / f'{z}/{x}').mkdir(parents=True, exist_ok=True)
                im.save(dst / f'{z}/{x}/{y}.png', optimize=True)
        print('tiles', dst.name, 'z', z, len(tiles), flush=True)


if __name__ == '__main__':
    if 'teselas' in sys.argv:
        sys.exit(teselas())
    DST.mkdir(parents=True, exist_ok=True)
    indice = {'clases_d': [{'color': c, 'alfa': a, 'label': f'{corto(lo)}–{corto(hi)} · {t}'}
                           for (c, a, t), lo, hi in zip(CLASES_D, CORTES_D[:-1], CORTES_D[1:])],
              'verde': VERDE, 'capas': figuras(), 'areas': atlas()}
    (DST / 'indice.json').write_text(json.dumps(indice, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print('areas', len(indice['areas']))
