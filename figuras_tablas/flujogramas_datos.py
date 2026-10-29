"""Data and thumbnails for the two workflow figures, Fig. S2 (fig02_flujo.html) and Fig. S3 (fig03_busqueda.html), read from
stored results and checked against them. Needs the outputs of atlas.py and mapas_especies.py."""
import base64, csv, io, json
import numpy as np
from figuras_comun import *  # noqa: F401,F403
from figuras_comun import _CACHE  # noqa: F401

DST = Path(__import__('os').environ.get('FLUJOGRAMAS_DST', MS / 'flujogramas'))  # next to the HTML; git-ignored (IUCN-derived thumbnails)
DST.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261011)


def png64(fig, dpi=200):
    buf = io.BytesIO(); fig.savefig(buf, dpi=dpi, transparent=True, bbox_inches='tight', pad_inches=0); plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def mapa_variable(base, valores, cmap, extent=None, gris=False):
    fig, ax = plt.subplots(figsize=(2.2, 2.6))
    if gris:
        fondo_paises(ax, gris='#ecebe6', borde='#cfcec8')
    img = base.raster(valores)
    ax.imshow(img, cmap=cmap, extent=base.extent, interpolation='nearest',
              vmin=np.nanpercentile(valores, 2), vmax=np.nanpercentile(valores, 98), zorder=2)
    contorno(ax, base.outline, color='#52514e', lw=.5, zorder=3)
    if extent:
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1 / np.cos(np.deg2rad(-9.2))); ax.set_axis_off()
    return png64(fig)


out = {}
bt = lienzo(TIERRA)
X = np.load(TIERRA / 'preparados/clima_fisico.npy', mmap_mode='r')
VARS = ['bio01', 'bio02', 'bio04', 'bio05', 'bio06', 'bio12', 'bio15', 'hursmean', 'vpdmean', 'rsdsmean', 'petmean']
col = {v: np.asarray(X[:, i], np.float32) for i, v in enumerate(VARS) if v in ('bio01', 'bio12', 'hursmean')}
out['capas_tierra'] = [mapa_variable(bt, col['bio01'], 'RdYlBu_r'), mapa_variable(bt, np.log1p(col['bio12']), 'YlGnBu'),
                       mapa_variable(bt, col['hursmean'], 'PuBuGn')]
EXT_MAR = (-85.2, -69.8, -20.6, -2.9)
for capa, k, cmap, key in (('superficie', 0, 'RdYlBu_r', 'mar1'), ('superficie', 6, 'YlGn', 'mar2'), ('fondo', 4, 'magma', 'mar3')):
    bm = lienzo(MAR / capa)
    Xm = np.load(MAR / capa / 'preparados/clima_fisico.npy')
    v = Xm[:, k] if k != 6 else np.log(Xm[:, k])
    out[key] = mapa_variable(bm, v, cmap, EXT_MAR, gris=True)

# protected areas and reporting strata thumbnails
fig, ax = plt.subplots(figsize=(2.2, 2.6))
contorno(ax, bt.outline, color='#52514e', lw=.5)
colmod = {'NACIONAL': '#1b7837', 'REGIONAL': '#2a78d6', 'PRIVADA': '#d65a8a'}
inv = {u['anp_id']: u for u in bt.inventory}
for uid, g in bt.shapes.items():
    relleno(ax, g, facecolor=colmod[inv[uid]['modalidad']], edgecolor='none', alpha=.85)
bm = lienzo(MAR / 'fondo')
for uid, g in bm.shapes.items():
    relleno(ax, g, facecolor='#1b7837', edgecolor='none', alpha=.85)
ax.set_xlim(-82.2, -68.4); ax.set_ylim(-18.6, .2); ax.set_aspect(1 / np.cos(np.deg2rad(-9.2))); ax.set_axis_off()
out['areas'] = png64(fig)
zt = np.load(TIERRA / 'preparados/ecorregiones_minam.npz')['ecorregion']
rgb = np.array([to_rgb(COLOR[e]) for e in ECO])
img = np.zeros((bt.h * bt.w, 4)); img[bt.flat, :3] = rgb[zt]; img[bt.flat, 3] = 1
bs = lienzo(MAR / 'superficie')  # sea surface by marine ecoregion; seafloor strata as darker shading with depth
rgbm = np.array([to_rgb(COLOR[e]) for e in ECO_MAR])
imm = np.zeros((bs.h * bs.w, 4)); imm[bs.flat, :3] = rgbm[np.load(MAR / 'superficie/preparados/ecorregiones_minam.npz')['ecorregion']]
imm[bs.flat, 3] = 1
zm = np.load(MAR / 'fondo/preparados/zonas.npz')['ecorregion']
som = np.zeros((bm.h * bm.w, 4)); som[bm.flat, :3] = to_rgb(SOMBRA); som[bm.flat, 3] = np.array(ALFA_PROF)[zm]
fig, ax = plt.subplots(figsize=(2.2, 2.6))
ax.imshow(imm.reshape(bs.h, bs.w, 4), extent=bs.extent, interpolation='nearest')
ax.imshow(som.reshape(bm.h, bm.w, 4), extent=bm.extent, interpolation='nearest')
ax.imshow(img.reshape(bt.h, bt.w, 4), extent=bt.extent, interpolation='nearest')
ax.set_xlim(-84.8, -68.4); ax.set_ylim(-20.4, .2); ax.set_aspect(1 / np.cos(np.deg2rad(-9.2))); ax.set_axis_off()
out['estratos'] = png64(fig)

# climate space: national sample, Machupicchu references, two query pixels
Z = np.load(TIERRA / 'modelo/clima_transformado_cond100.npy', mmap_mode='r')
muestra = np.sort(rng.choice(len(bt.flat), 3000, replace=False))
Zs = np.asarray(Z[muestra], np.float64)
mu = Zs.mean(0); U, S, Vt = np.linalg.svd(Zs - mu, full_matrices=False); W = Vt[:2].T
proj = lambda a: (np.asarray(a, np.float64) - mu) @ W  # noqa: E731
uid = 'NAC_SH03'
idx = np.load(TIERRA / f'preparados/asociaciones/{uid}.npz')['index']
refs = np.unique(idx)
dd = np.load(TIERRA / f'distancias/cond100/{uid}.npz')
d, vecino = dd['distancia'], dd['vecino']
lon = bt.aff.c + bt.aff.a * (bt.flat % bt.w + .5); lat = bt.aff.f + bt.aff.e * (bt.flat // bt.w + .5)
fuera = np.ones(len(d), bool); fuera[idx] = False
cand = np.flatnonzero(fuera & (d > .12) & (d < .18) & (lat > -11))  # a distant analogue, > 250 km north of the area
c1 = int(cand[np.argmin(np.abs(d[cand] - .15))])
cand2 = np.flatnonzero(fuera & (d > .55) & (d < .70) & (lat < -14.5))
c2 = int(cand2[0])
r1 = int(vecino[c1])  # global pixel index of r*
assert abs(np.linalg.norm(np.asarray(Z[c1]) - np.asarray(Z[r1])) - d[c1]) < 1e-5, 'vecino is not r*'
r2 = int(vecino[c2])
assert abs(np.linalg.norm(np.asarray(Z[c2]) - np.asarray(Z[r2])) - d[c2]) < 1e-5
out['espacio'] = {
    'nube': np.round(proj(Zs), 3).tolist(),
    'refs': [np.round(proj(Z[r]), 3).tolist() for r in refs],
    'c1': np.round(proj(Z[c1]), 3).tolist(), 'r1': np.round(proj(Z[r1]), 3).tolist(), 'd1': float(d[c1]),
    'c2': np.round(proj(Z[c2]), 3).tolist(), 'r2': np.round(proj(Z[r2]), 3).tolist(), 'd2': float(d[c2]),
    'c1_lonlat': [float(lon[c1]), float(lat[c1])], 'c2_lonlat': [float(lon[c2]), float(lat[c2])],
    'km_c1': float(round(6371 * np.arccos(np.clip(np.sin(np.radians(lat[c1])) * np.sin(np.radians(bt.shapes[uid].centroid.y)) +
                     np.cos(np.radians(lat[c1])) * np.cos(np.radians(bt.shapes[uid].centroid.y)) *
                     np.cos(np.radians(lon[c1] - bt.shapes[uid].centroid.x)), -1, 1)), -1)),
    'var': (S[:2] ** 2 / (S ** 2).sum()).round(3).tolist(), 'n_refs': int(len(refs))}

# zoom of the area: its pixels
g = bt.shapes[uid]
x0, y0, x1, y1 = g.bounds; pad = .06
fig, ax = plt.subplots(figsize=(2.2, 2.2))
r_lon = bt.aff.c + bt.aff.a * (bt.flat[refs] % bt.w); r_lat = bt.aff.f + bt.aff.e * (bt.flat[refs] // bt.w)
for lo, la in zip(r_lon, r_lat):
    ax.add_patch(matplotlib.patches.Rectangle((lo, la + bt.aff.e), bt.aff.a, -bt.aff.e, fc='#2a78d6', ec='white', lw=.25, alpha=.9))
contorno(ax, g, color=CIAN, lw=1.4)
ax.set_xlim(x0 - pad, x1 + pad); ax.set_ylim(y0 - pad, y1 + pad); ax.set_aspect(1 / np.cos(np.deg2rad(-13))); ax.set_axis_off()
out['zoom_area'] = png64(fig, 250)

# Peru locator with the area and the two query pixels
fig, ax = plt.subplots(figsize=(2.0, 2.4))
relleno(ax, bt.outline, facecolor='#f1f0ec', edgecolor='#8a8986', lw=.5)
relleno(ax, g.buffer(.12), facecolor=CIAN, edgecolor='none')
ax.set_xlim(-81.5, -68.5); ax.set_ylim(-18.5, .2); ax.set_aspect(1 / np.cos(np.deg2rad(-9.2))); ax.set_axis_off()
out['peru_loc'] = png64(fig)
out['peru_loc_ext'] = [-81.5, -68.5, -18.5, .2]

# final map of the area and system map (no legend)
fig, ax = plt.subplots(figsize=(2.6, 3.1)); panel_d(ax, bt, d, g)
out['mapa_area'] = png64(fig, 260)
ds = bt.leer(TIERRA / 'mapas_distancia/cond100/sistema_distancia_minima.tif')
fig, ax = plt.subplots(figsize=(2.6, 3.1)); panel_d(ax, bt, ds, shapely.union_all(list(bt.shapes.values())), circulo=False)
out['mapa_sistema'] = png64(fig, 220)
a = next(x for x in CIFRAS['continental']['areas'] if x['anp_id'] == uid)
out['sup_area'] = [a['km2_d0.1'], a['km2_d0.2']]
out['n_pixeles'] = int(len(bt.flat))
out['d_classes'] = [[c, al, t] for c, al, t in CLASES_D]
out['cortes'] = CORTES_D

# p-median of the real network and of the 999 N2c null networks (Peru; main design)
N2C = RAIZ / 'resultados_v7/04_redes_nulas/continental_superposicion_limitada/N2c'
z = np.load(N2C / 'redes_aleatorias.npz')
w = z['peso']
out['pmediana'] = {'real': float(np.average(z['real_dmin'], weights=w)),
                   'azar': [round(float(np.average(x, weights=w)), 5) for x in z['azar_dmin']]}
# national network coverage curve (Peru)
rows = [r for r in csv.DictReader(open(N2C / 'curvas_cobertura.csv', encoding='utf-8-sig')) if r['estrato'] == 'Perú']
out['curva'] = {k: [round(float(r[k]), 4) for r in rows] for k in ('d', 'real', 'azar_p2_5', 'azar_mediana', 'azar_p97_5')}

# ======================================================================== Fig. S2: the whole study, inputs to deliverables
import pandas as pd, pyogrio
from PIL import Image
RES, REPO = RAIZ / 'resultados_v7', MS.parent
IMP, PRI = RES / '06_implicaciones', RES / '09_prioritizacion'
NATURAL, TRANSF = '#9cc27a', '#e6dccb'


def pagina(path, ancho=360):
    """Downscaled JPEG of a page or figure; None when the file is not in this copy (e.g. IUCN plates in the repository)."""
    if not Path(path).exists():
        return None
    im = Image.open(path).convert('RGB'); im.thumbnail((ancho, ancho * 4))
    buf = io.BytesIO(); im.save(buf, 'JPEG', quality=84)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def mapa_peru(pinta, xlim=(-82.2, -68.4), ylim=(-18.6, .2), size=(2.2, 2.6)):
    fig, ax = plt.subplots(figsize=size); pinta(ax)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(ylim)))); ax.set_axis_off()
    return png64(fig)


C, I = CIFRAS['continental'], CIFRAS['implicaciones']
# natural land (MapBiomas + GHS-SMOD) and the candidate microbasins
nat = np.load(RES / '03_mascara_natural/natural_dominio.npz')['natural']
assert round(100 * bt.area[nat].sum() / bt.area.sum(), 1) == 91.1
img = np.zeros((bt.h * bt.w, 4)); img[bt.flat, :3] = np.where(nat[:, None], to_rgb(NATURAL), to_rgb(TRANSF)); img[bt.flat, 3] = 1
out['natural'] = mapa_peru(lambda ax: (ax.imshow(img.reshape(bt.h, bt.w, 4), extent=bt.extent, interpolation='nearest'),
                                       contorno(ax, bt.outline, color='#52514e', lw=.5)))
cel = np.load(PRI / 'entrada/celdas.npz')
pu, cand = cel['pu'], cel['candidato']
assert len(np.unique(pu[cand])) == I['P1_curva']['n_unidades_candidatas'] == 8963
paleta = np.array([to_rgb(c) for c in ('#e3cf97', '#d4b977', '#ecdcb0', '#c9ab68', '#dfc488', '#efe2c0')])
tono = rng.permutation(pu.max() + 1) % len(paleta)
img = np.zeros((bt.h * bt.w, 4)); img[bt.flat, :3] = np.where(cand[:, None], paleta[tono[pu]], to_rgb('#efeeea'))
img[bt.flat[pu == 0], :3] = to_rgb('#bfe6f2')  # unit 0 is the real network (locked in)
img[bt.flat, 3] = 1
r2 = bt.raster(np.where(cand, pu, -1).astype(np.float32))
borde = np.zeros_like(r2, bool); borde[:, 1:] |= r2[:, 1:] != r2[:, :-1]; borde[1:] |= r2[1:] != r2[:-1]
img = img.reshape(bt.h, bt.w, 4); img[borde & ~np.isnan(r2), :3] = to_rgb('#8a7444')
out['microcuencas'] = mapa_peru(lambda ax: ax.imshow(img, extent=bt.extent, interpolation='nearest'),
                                xlim=(-76.3, -74.9), ylim=(-12.9, -11.5), size=(2.2, 2.2))
# species ranges: the 410 continental species mapped in File S3 (IUCN and BirdLife; mapas_especies.py sig, outside git)
shp = sorted((ESPECIES / 'sig/vectores').glob('FigS3-[1-7]_*.shp'))
if shp:
    n = [pyogrio.read_info(p)['features'] for p in shp]
    assert sum(n[:5]) == 410 and sum(n[5:]) == 90, n

    def rangos(ax):
        relleno(ax, bt.outline, facecolor='#f4f3ef', edgecolor='none')
        for p in shp[:5]:
            for g in shapely.from_wkb(pyogrio.raw.read(p)[2]):
                relleno(ax, g, facecolor='#7b3294', edgecolor='none', alpha=.07)
        contorno(ax, bt.outline, color='#52514e', lw=.5)
    out['rangos'] = mapa_peru(rangos)
# SENAMHI stations (PISCOp is gridded and covers the whole country)
est = pd.read_csv(TIERRA / 'estaciones/estaciones.csv', encoding='utf-8-sig').dropna(subset=['bio01_obs'])  # as Fig. S11
assert len(est) == 22
out['estaciones'] = mapa_peru(lambda ax: (relleno(ax, bt.outline, facecolor='#f1f0ec', edgecolor='none'),
                                          contorno(ax, bt.outline, color='#52514e', lw=.5),
                                          ax.scatter(est.lon, est.lat, s=42, marker='^', c='#1f1f1d', ec='white', lw=.6, zorder=3)))
# numbers of every branch, checked against the text of the manuscript and supplement
nul = {k: pd.read_csv(RES / f'04_redes_nulas/{p}/N2c/p_mediana.csv', encoding='utf-8-sig').iloc[0]
       for k, p in (('tierra', 'continental_superposicion_limitada'), ('sup', 'superficie'), ('fondo', 'fondo'))}
P1, P2 = I['P1_curva'], I['P2']
dej = pd.read_csv(IMP / 'P3_dejando_una_fuera.csv')
f02 = lambda capa, q: float(dej[(dej.capa == capa) & (dej.fuera == q)].F02_todo.iloc[0] * 100)  # noqa: E731
pri = pd.read_csv(REPO / 'tables/main/Fig4_prioritization_table.csv').set_index(['budget_km2', 'set'])
s1 = pd.read_excel(REPO / 'supplementary/File_S1_species_IUCN_BirdLife.xlsx', sheet_name=None)
tc, tm = s1['Continental domain'], s1['Maritime domain']
gap = 'Climatic gaps, unprotected d_min > 0.2 (%)'
amen = tc['Red List category'].isin(['CR', 'EN', 'VU'])
iucn_gap = amen & (tc.Group != 'Birds') & (tc[gap] > 50)
N = {'pix_tierra': len(bt.flat), 'km2_tierra': C['peru']['km2_total'], 'pix_mar': len(bm.flat),
     'km2_mar': CIFRAS['maritimo']['superficie']['todo']['km2_total'],
     'areas': len(bt.inventory), 'areas_tierra': sum(u['n_celdas'] > 0 for u in bt.inventory), 'areas_mar': len(bm.shapes),
     'modalidades_n': [sum(u['modalidad'] == k for u in bt.inventory) for k in ('NACIONAL', 'REGIONAL', 'PRIVADA')],
     'vacios_km2': I['P1']['vacios_km2'], 'vacios_pct': 100 - C['peru']['pct_d0.2'],
     'sup_f02': CIFRAS['maritimo']['superficie']['todo']['pct_d0.2'], 'fondo_f02': CIFRAS['maritimo']['fondo']['todo']['pct_d0.2'],
     'area_d02': C['areas_resumen']['todas']['rango_d0.2'], 'area_d02_mediana': C['areas_resumen']['todas']['d0.2'][1],
     'D_real': nul['tierra'].D_real, 'D_nula': nul['tierra'].D_azar_mediana, 'p_tierra': nul['tierra'].p_bilateral,
     'ses_tierra': nul['tierra'].ses, 'p_sup': nul['sup'].p_bilateral, 'p_fondo': nul['fondo'].p_bilateral,
     'F02_ANP': P2['F02_ANP'] * 100, 'F02_red': P2['F02_red'] * 100, 'solo_ACR_ACP': P2['solo_ACR_ACP_km2'],
     'nazca': [f02('superficie', '(ninguna)'), f02('superficie', 'NAC_RN16'), f02('fondo', '(ninguna)'), f02('fondo', 'NAC_RN16')],
     'metas': [[P1[k]['agregado_km2'], P1[k]['pasos']] for k in ('alcanza_mediana_nula', 'alcanza_F02_80', 'alcanza_F02_90')],
     'sitios': I['P1_sitios']['n_sitios'], 'falta_30': I['P1_umbral']['falta_30_km2'],
     'clima_41k': {k: pri.loc[(41347, s)].share_closed for k, s in (('B', 'Species only (B)'), ('azar', 'Random natural microbasins (median of 999)'),
                                                                     ('D', 'Climate only (D)'), ('C', 'Species + climate (C)'))},
     'sp_41k': {k: pri.loc[(41347, s)].species_mean_target * 100 for k, s in (('B', 'Species only (B)'), ('C', 'Species + climate (C)'),
                                                                               ('D', 'Climate only (D)'))},
     'sp_tierra': len(tc), 'aves': int((tc.Group == 'Birds').sum()), 'sp_mar': len(tm), 'aves_mar': int((tm.Group == 'Seabirds').sum()),
     'amen_vacios': int(iucn_gap.sum()), 'amen_vacios_end': int((iucn_gap & tc.iloc[:, 7].notna()).sum()),
     'azar_tierra': I['P1']['vacios_km2'] / C['peru']['km2_total'] * 100,
     'mapas_s3': [410, 90], 'atlas': sum(u['n_celdas'] > 0 for u in bt.inventory) + 2 * len(bm.shapes)}
assert (N['pix_tierra'], round(N['km2_tierra']), N['pix_mar'], round(N['km2_mar'])) == (1535585, 1286457, 29171, 854669)
assert (N['areas'], N['areas_tierra'], N['areas_mar'], N['modalidades_n']) == (259, 258, 6, [71, 39, 149])
assert (round(N['vacios_km2']), round(N['vacios_pct'], 1)) == (387232, 36.7)
assert (round(N['sup_f02'], 1), round(N['fondo_f02'], 1)) == (7.8, 74.9)
assert [round(N['area_d02'][0]), round(N['area_d02'][1]), round(N['area_d02_mediana'])] == [38, 265955, 2446]
assert (round(N['D_real'], 3), round(N['D_nula'], 3), N['p_tierra'], round(N['ses_tierra'], 1)) == (.151, .133, .002, 6.4)
assert (N['p_sup'], N['p_fondo']) == (.008, .298)
assert (round(N['F02_ANP'], 1), round(N['F02_red'], 1), round(N['solo_ACR_ACP'])) == (63.7, 69.9, 79553)
assert [round(v, 1) for v in N['nazca']] == [15.1, 3.0, 76.9, 4.5]
assert [[round(a), p] for a, p in N['metas']] == [[2419, 13], [12044, 64], [41347, 228]] and round(N['falta_30']) == 154280
assert [round(N['clima_41k'][k]) for k in 'B azar D C'.split()] == [35, 71, 77, 78]
assert [round(N['sp_41k'][k], 1) for k in 'BCD'] == [59.8, 34.4, 21.8]
assert (N['sp_tierra'], N['aves'], N['sp_mar'], N['aves_mar']) == (3900, 1799, 988, 75)
assert (N['amen_vacios'], N['amen_vacios_end'], round(N['azar_tierra'], 1), N['atlas']) == (143, 100, 30.1, 270)
t6s = pd.read_csv(REPO / 'tables/supplementary/TableS6.csv', encoding='utf-8-sig').iloc[0]
t7 = pd.read_csv(REPO / 'tables/supplementary/TableS7.csv', encoding='utf-8-sig').iloc[0]
t9 = pd.read_csv(REPO / 'tables/supplementary/TableS9.csv', encoding='utf-8-sig').iloc[0]
N.update({'candidatas': I['P1_curva']['n_unidades_candidatas'], 'natural_pct': 100 * bt.area[nat].sum() / bt.area.sum(),
          'microcuencas': json.loads((IMP / 'microcuencas_lev12.json').read_text(encoding='utf-8'))['n_microcuencas'],
          'menos_rep': C['menos_representados']['dmin_desde'],
          'muestra': [len(np.load(RES / f'04_redes_nulas/{p}/N2c/redes_aleatorias.npz')['peso'])
                      for p in ('continental_superposicion_limitada', 'superficie', 'fondo')],
          'features_sp': len(pd.read_csv(REPO / 'data/prioritization/species_features.csv')),
          'n_pequenas': int(t6s.index[3].split()[2]), 'ses_rob': [float(v.split('(')[1][:-1]) for v in t6s.iloc[2:4]],
          'pisco': [float(v) for v in t7.iloc[1].split(' → ')] + [float(t7.iloc[3])], 'c7': [float(t9.iloc[1]), float(t9.iloc[2])]})
assert (round(N['natural_pct'], 1), N['microcuencas'], round(N['menos_rep'], 3)) == (91.1, 10187, .484), N
assert (N['muestra'], N['features_sp'], N['n_pequenas'], N['ses_rob']) == ([53509, 11521, 12934], 271, 63, [6.4, 6.2]), N
assert (N['pisco'], N['c7']) == ([63.3, 63.0, .773], [63.3, 80.1]), N
out['N'] = N
# mini charts
GR = ['Amphibians', 'Mammals', 'Reptiles', 'Plants', 'Birds']
out['grupos'] = [[g, float(tc[amen & (tc.Group == g)][gap].median()), float(tc[tc.Group == g][gap].median())] for g in GR]
assert all(abs(v[1] - e) < .06 for v, e in zip(out['grupos'], [23.4, 8.5, 66.1, 53.7, 29.3])), out['grupos']  # File S1 rounds to 0.1
cu = pd.read_csv(IMP / 'P1_curva_esfuerzo.csv')
out['esfuerzo'] = [[round(a / 1e3, 3), round(f * 100, 2)] for a, f in zip(cu.agregado_km2, cu.F02)]
m = pd.read_csv(IMP / 'P2_modalidades_por_ecorregion.csv').set_index('nombre')
ECO6 = ['BTA', 'SAL', 'PUN', 'SES', 'BSE', 'DCP']
ECO_ES = {'BTA': 'Bosque tropical amazónico', 'SAL': 'Selva alta (yungas)', 'PUN': 'Puna', 'SES': 'Serranía esteparia',
          'BSE': 'Bosque seco ecuatorial', 'DCP': 'Desierto costero del Pacífico'}  # names in P2_modalidades_por_ecorregion.csv
out['modalidades'] = [[e, float(100 - m.loc[ECO_ES[e]].cubierta_solo_ANP_km2 / m.loc[ECO_ES[e]].cubierta_red_km2 * 100)] for e in ECO6]
assert [round(v) for _, v in out['modalidades']] == [1, 15, 31, 7, 53, 27]
t6 = pd.read_csv(REPO / 'tables/supplementary/TableS6.csv', encoding='utf-8-sig')
t6 = t6[t6.iloc[:, 1] != '–']
ABR = {'Amazonian tropical forest': 'Amazonian', 'Yungas (high jungle)': 'Yungas', 'Páramo': 'Páramo', 'Puna': 'Puna',
       'Steppe mountains': 'Steppe', 'Equatorial dry forest': 'Dry forest', 'Pacific coastal desert': 'Desert'}
out['robustez'] = {'estratos': ['Peru'] + [ABR[s] for s in t6.Stratum[1:]],
                   'filas': [[c, [('worse' in v) * 1 for v in t6[c]]] for c in t6.columns[1:]]}
assert len(out['robustez']['estratos']) == 8 and all(f[1] == out['robustez']['filas'][0][1] for f in out['robustez']['filas'])
# three pages of the atlas (File S2)
out['paginas'] = {'atlas': [pagina(ATLAS / c / f'atlas_{u}.png', 300) for c, u in
                            (('continental', 'NAC_RN08'), ('continental', 'NAC_SH03'), ('maritimo', 'NAC_RN16_fondo'))]}
(DST / 'datos.js').write_text('const DATA = ' + json.dumps(out) + ';', encoding='utf-8')
print('ok', {k: (len(v) if isinstance(v, (list, str)) else '') for k, v in out.items()})
print('c1', out['espacio']['c1_lonlat'], out['espacio']['d1'], 'c2', out['espacio']['c2_lonlat'], out['espacio']['d2'])
print('pmediana', out['pmediana']['real'], np.median(out['pmediana']['azar']))
