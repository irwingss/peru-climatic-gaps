"""Shared constants, canvases and map helpers of every figure (figuras_articulo.py, figuras_suplementarias.py, atlas.py,
mapas_especies.py and flujogramas_datos.py).

Numbers use a point for decimals and a space for thousands. Reads only stored results and insumos_figuras/cifras.json
(written by cifras_resultados.py).
"""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba, to_rgb
from matplotlib.patches import Patch, Polygon as MplPolygon
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import rasterio
from rasterio.transform import Affine
import shapely
from shapely.geometry import shape

RAIZ = next((p for p in Path(__file__).resolve().parents if (p / 'resultados_v7').is_dir()), Path(__file__).resolve().parents[1])  # primera carpeta con los resultados (raiz del repositorio o del proyecto)
TIERRA = RAIZ / 'resultados_climaticos/20261011'
MAR = RAIZ / 'resultados_marinos/20261011'
MS = Path(__file__).parent
FIG = MS / 'figuras'
REPO = MS.parent
PRINCIPAL, SUPL = REPO / 'figures' / 'main', REPO / 'figures' / 'supplementary'   # figures of the article
ATLAS, ESPECIES = FIG / 'atlas', FIG / 'especies'   # File S2 and File S3 (outside git: large, or derived from IUCN/BirdLife)
INSUMOS = MS / 'insumos_figuras'

plt.rcParams.update({'font.family': 'Arial', 'font.size': 8.5, 'axes.linewidth': .6,
                     'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.labelcolor': '#0b0b0b', 'savefig.dpi': 300, 'axes.formatter.use_locale': False})

# MINAM (2017) ecoregions: the 9 continental ones in pipeline order (value stored in ecorregiones_minam.npz) and the 2 marine
# ones that stratify the sea surface. Colours of the official MINAM map N-53, sampled from its legend.
ECO = ['BTA', 'SPA', 'SAL', 'PAR', 'PUN', 'SES', 'BSE', 'BTP', 'DCP']
ECO_MAR = ['MFH', 'MTR']
ECO_EN = {'BTA': 'Amazonian tropical forest', 'SPA': 'Palm savanna', 'SAL': 'High jungle (Yungas)', 'PAR': 'Páramo',
          'PUN': 'Puna', 'SES': 'Steppe mountains', 'BSE': 'Equatorial dry forest', 'BTP': 'Pacific tropical forest',
          'DCP': 'Pacific coastal desert', 'MFH': 'Cold sea of the Humboldt Current', 'MTR': 'Tropical sea'}
COLOR = {'BTA': '#74c14d', 'SPA': '#ff9de8', 'SAL': '#dfff9c', 'PAR': '#ff4d4d', 'PUN': '#e4c8bc', 'SES': '#fec1a5',
         'BSE': '#f8a2a1', 'BTP': '#c2744d', 'DCP': '#ffff9d', 'MFH': '#d3fff0', 'MTR': '#9ec9fe'}
SOMBRA, ALFA_PROF = '#0b2a4a', [0.0, .22, .44]  # seafloor depth shading over the sea colour: shelf, slope, deep
CAT_EN = {'Bosque de Protección': 'Protection Forest', 'Coto de Caza': 'Hunting Reserve', 'Parque Nacional': 'National Park',
          'Refugio de Vida Silvestre': 'Wildlife Refuge', 'Reserva Comunal': 'Communal Reserve',
          'Reserva Nacional': 'National Reserve', 'Reserva Paisajistica': 'Landscape Reserve',
          'Santuario Histórico': 'Historic Sanctuary', 'Santuario Nacional': 'National Sanctuary',
          'Área de Conservación Regional': 'Regional Conservation Area', 'Área de Conservación Privada': 'Private Conservation Area'}
CAT_ES = {'Reserva Paisajistica': 'Reserva Paisajística'}
MOD_EN = {'NACIONAL': 'National administration (ANP)', 'REGIONAL': 'Regional administration (ACR)',
          'PRIVADA': 'Private (ACP)'}
# One meaning per colour in every figure, the atlas and the viewer (the authors' species-map palette): green = protected,
# blue = protected analogue (d <= 0.2, darker = more similar), magenta = beyond 0.2 (darker = farther)
VERDE, VERDE_CLARO, VERDE_OSCURO = '#23745B', '#a9d4bd', '#123d30'
ANALOGO = ['#2f5f93', '#6b93c4', '#b3c9e4']                # d 0–0.05, 0.05–0.1, 0.1–0.2
VACIO = ['#ecc9e0', '#d394be', '#A64E82', '#6a1d52']       # d 0.2–0.3, 0.3–0.5, 0.5–1, > 1
# d classes defined by the authors; d > 1 is VACIO[3] ('not analogous')
CORTES_D = [0, .05, .1, .2, .3, .5, 1.0]
CLASES_D = [(ANALOGO[0], 1.0, 'nearly identical'), (ANALOGO[1], 1.0, 'very similar'), (ANALOGO[2], 1.0, 'similar'),
            (VACIO[0], 1.0, 'moderate'), (VACIO[1], 1.0, 'weak'), (VACIO[2], 1.0, 'low similarity')]
CIAN, TIERRA_GRIS, SIN_DATO = '#0096c7', '#e4e3df', '#d6d6d2'
REAL, BANDA, MEDIANA = '#a50f15', '#d6d6d0', '#6f6e69'
halo = [pe.withStroke(linewidth=2.2, foreground='white')]
CIFRAS = json.loads((INSUMOS / 'cifras.json').read_text(encoding='utf-8')) if (INSUMOS / 'cifras.json').exists() else None


def en(x, nd=0):
    """English number: point for decimals, space for thousands."""
    return f'{x:,.{nd}f}'.replace(',', ' ')


def corto(x):
    return (f'{x:.2f}'.rstrip('0').rstrip('.') if x else '0')


def nombre_en(u):
    n = u['nombre'].strip()
    for p in ('de la ', 'del ', 'de '):
        if n.startswith(p):
            n = n[len(p):]
            break
    n = n[0].upper() + n[1:]
    return f"{n} {CAT_EN[u['categoria']]}"


def nombre_es(u):
    n = u['nombre'].strip()
    if n.split()[0] in ('los', 'las', 'la', 'el'):
        n = n[0].upper() + n[1:]
    return f"{CAT_ES.get(u['categoria'], u['categoria'])} {n}"


def escala_d():
    cmap = ListedColormap([to_rgba(c, a) for c, a, _ in CLASES_D]); cmap.set_over(VACIO[3]); cmap.set_bad((0, 0, 0, 0))
    return cmap, BoundaryNorm(CORTES_D, len(CLASES_D))


def leyenda_d(extra=(), titulo='Mahalanobis distance d'):
    h = [Patch(facecolor=to_rgba(c, a), edgecolor='#bdbdb5', lw=.4, label=f'{corto(lo)}–{corto(hi)} · {t}')
         for (c, a, t), lo, hi in zip(CLASES_D, CORTES_D[:-1], CORTES_D[1:])]
    h.append(Patch(facecolor=VACIO[3], edgecolor='#bdbdb5', lw=.4, label='> 1 · not analogous'))
    return list(extra) + h


def contorno(ax, geometry, **style):
    for poly in getattr(geometry, 'geoms', [geometry]):
        if poly.geom_type != 'Polygon':
            continue
        for ring in [poly.exterior, *poly.interiors]:
            ax.plot(*ring.xy, **style)


def relleno(ax, geometry, **style):
    for poly in getattr(geometry, 'geoms', [geometry]):
        if poly.geom_type == 'Polygon':
            ax.add_patch(MplPolygon(np.asarray(poly.exterior.coords), closed=True, **style))


def geojson(path):
    return [(f['properties'], shape(f['geometry'])) for f in json.loads(Path(path).read_text(encoding='utf-8'))['features']]


RECURSOS = Path(__import__('os').environ.get('RECURSOS_GENERALES', Path.home() / 'Documents' / 'Recursos generales'))  # inputs outside the repository
DEPARTAMENTOS_INEI = RECURSOS / 'capas_sig' / 'Peru' / 'Departamentos_Peru.shp'   # INEI boundaries, not redistributed (data/insumos_sha256.json)


def departamentos():
    """Department boundaries of Peru for the maps: INEI polygons simplified to 0.002 degrees (about 200 m)."""
    import pyogrio
    _, _, wkb, (nombre,) = pyogrio.raw.read(DEPARTAMENTOS_INEI, columns=['NOMBDEP'])
    return [({'NOMBDEP': n}, g) for n, g in zip(nombre, shapely.simplify(shapely.from_wkb(wkb), .002, preserve_topology=False))]


class Lienzo:
    """Grid, domain and outlines of one analysis (continental or one maritime layer)."""

    def __init__(self, base):
        prep = base / 'preparados'
        g = json.loads((prep / 'preparacion.json').read_text(encoding='utf-8'))['grid']
        self.h, self.w = g['height'], g['width']; self.aff = Affine(*g['transform'])
        dom = np.load(prep / 'dominio.npz')
        self.flat, self.area = dom['flat'], dom['area']
        d = np.full(self.h * self.w, np.nan, np.float32)
        if 'todo_flat' in dom.files:
            d[dom['todo_flat']] = 0
        d[self.flat] = 1
        self.domain = d.reshape(self.h, self.w)
        self.extent = (self.aff.c, self.aff.c + self.w * self.aff.a, self.aff.f + self.h * self.aff.e, self.aff.f)
        self.xs = self.aff.c + self.aff.a * (np.arange(self.w) + .5); self.ys = self.aff.f + self.aff.e * (np.arange(self.h) + .5)
        self.inventory = json.loads((prep / 'inventario.json').read_text(encoding='utf-8'))
        self.shapes = {p['anp_id']: s for p, s in geojson(prep / 'unidades_continentales.geojson')}
        self.outline = shape(json.loads((prep / 'peru_continental.geojson').read_text(encoding='utf-8'))['geometry']).simplify(.005)
        self.prep = prep

    def raster(self, values, fill=np.nan):
        img = np.full(self.h * self.w, fill, np.float32); img[self.flat] = values
        return img.reshape(self.h, self.w)

    def leer(self, path):
        with rasterio.open(path) as ds:
            return ds.read(1).reshape(-1)[self.flat]


_CACHE = {}


def lienzo(base):
    if base not in _CACHE:
        _CACHE[base] = Lienzo(base)
    return _CACHE[base]


def fondo_paises(ax, gris=TIERRA_GRIS, borde='#b9b8b2'):
    peru = shape(json.loads((TIERRA / 'preparados/peru_continental.geojson').read_text(encoding='utf-8'))['geometry'])
    for p, s in geojson(INSUMOS / 'paises.geojson'):
        relleno(ax, s if p['admin'] != 'Peru' else peru, facecolor=gris, edgecolor='none', zorder=0)
        contorno(ax, s if p['admin'] != 'Peru' else peru, color=borde, lw=.4, zorder=1)


def panel_d(ax, base, d, geometry=None, gris_tierra=False, circulo=True):
    """Distance classes; the protected pixels (d = 0) in green, with a dark green outline and, for tiny areas, a circle."""
    cmap, norm = escala_d()
    if gris_tierra:
        fondo_paises(ax)
    ground = ListedColormap([SIN_DATO, '#ffffff']); ground.set_bad((0, 0, 0, 0))
    ax.imshow(base.domain, cmap=ground, vmin=0, vmax=1, extent=base.extent, interpolation='nearest', zorder=2)
    ax.imshow(base.raster(d), cmap=cmap, norm=norm, extent=base.extent, interpolation='nearest', zorder=3)
    ax.imshow(base.raster(np.where(d == 0, 0., np.nan)), cmap=ListedColormap([VERDE]), vmin=-.5, vmax=.5, extent=base.extent,
              interpolation='nearest', zorder=3.5)
    contorno(ax, base.outline, color='#6b6b66', lw=.4, zorder=4)
    if geometry is not None and not geometry.is_empty:
        contorno(ax, geometry, color=VERDE_OSCURO, lw=.6, zorder=6)
        if circulo and geometry.area < .01:
            ax.plot(geometry.centroid.x, geometry.centroid.y, marker='o', ms=10, mfc='none', mec=VERDE_OSCURO, mew=1.3, zorder=6)
    ax.set_aspect(1 / np.cos(np.deg2rad(-9.2)))
    ax.set_axis_off()


def encuadre_mar(ax):
    ax.set_xlim(-85.2, -69.8); ax.set_ylim(-20.6, -2.9)
    ax.set_axis_on(); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#bdbcb6'); s.set_linewidth(.5)
