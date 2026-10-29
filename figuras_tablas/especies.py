"""Species of the IUCN Red List and of BirdLife in climatic gaps (Supplementary material S6, Tables S10, Figs. S12-S13,
File S3): loads the per-species results of especies_uicn/cruce_uicn.py, caches the range polygons, draws the per-species
figures used by figuras_suplementarias.py and mapas_especies.py, and writes the numbers quoted in S6 to
insumos_figuras/especies_resumen.json. Run: python especies.py
"""
import functools, sys
import pandas as pd, pyogrio
from scipy import ndimage
from figuras_comun import *  # noqa: F401,F403  (shared constants, canvases and map helpers)

sys.path.insert(0, str(RAIZ / 'especies_uicn'))
import cruce_uicn as cu  # noqa: E402

RES = RAIZ / 'especies_uicn/resultados'
OUT_JSON = INSUMOS / 'especies_resumen.json'
AMENAZA = ['CR', 'EN', 'VU']
ENDEMICA = 95                          # % of the global range inside Peru
ATLAS_GLOBAL = 90                      # species maps (File S3): threatened or above this share, see seleccion_mapas
TIERRA_G = ['AMPHIBIANS', 'MAMMALS', 'REPTILES', 'PLANTS', 'BIRDS']   # birds: BirdLife and HBW (2025)
MAR_G = ['MARINEFISH', 'MAMMALS', 'REPTILES', 'BIRDS']
G_EN = {'AMPHIBIANS': 'Amphibians', 'MAMMALS': 'Mammals', 'REPTILES': 'Reptiles', 'PLANTS': 'Plants', 'MARINEFISH': 'Marine fishes',
        'BIRDS': 'Birds'}
CLS = ['similar', 'moderado', 'debil', 'poco_similar', 'no_analogo']
CLS_EN = ['≤ 0.2', '0.2–0.3', '0.3–0.5', '0.5–1', '> 1']
CAPAS = ['superficie', 'fondo']
G_MAR_EN = {'MARINEFISH': 'Marine fishes', 'MAMMALS': 'Marine mammals', 'REPTILES': 'Marine reptiles', 'BIRDS': 'Seabirds'}


def cargar():
    d = pd.concat([pd.read_csv(f) for f in sorted(RES.glob('especies_uicn_*.csv'))], ignore_index=True)
    d['amenazada'] = d.categoria.isin(AMENAZA)
    d['endemica'] = d.pct_global_en_peru >= ENDEMICA
    # marine mammals, reptiles and birds without freshwater habitat (cetaceans, pinnipeds, sea turtles, most seabirds) only
    # touch the coast on land
    solo_mar = d.marina & d.grupo.isin(['MAMMALS', 'REPTILES', 'BIRDS']) & ~d.dulceacuicola
    t = d[d.tierra_km2.notna() & ~solo_mar & d.grupo.isin(TIERRA_G)].copy()
    t['km2_fuera'] = t.tierra_km2 * t.tierra_pct_mas_alla_similar / 100
    m = d[d.mar_km2.notna() & (d.grupo.eq('MARINEFISH') | (d.grupo.isin(['MAMMALS', 'REPTILES', 'BIRDS']) & d.marina))].copy()
    # seabirds live and feed at the sea surface: the seafloor layer does not describe their habitat
    m.loc[m.grupo.eq('BIRDS'), [c for c in m.columns if c.startswith('mar_fondo_')]] = np.nan
    for c in CAPAS:
        m[f'km2_fuera_{c}'] = m.mar_km2 * m[f'mar_{c}_pct_mas_alla_similar'] / 100
    return t, m


def referencias():
    """Share of each domain or layer that is unprotected and beyond the similar class (expected share for a random range)."""
    c, mar = CIFRAS['continental'], CIFRAS['maritimo']
    tot_t = lienzo(TIERRA).area.sum() / 1e6; tot_m = lienzo(MAR / 'superficie').area.sum() / 1e6
    return {'tierra': 100 * c['clases']['mas_alla_similar']['km2'] / tot_t,
            **{c_: 100 * mar[c_]['clases']['mas_alla_similar']['km2'] / tot_m for c_ in CAPAS}}


def resumen(t, m):
    filas = []
    for g in TIERRA_G + ['ALL']:
        s = t if g == 'ALL' else t[t.grupo == g]; a, e = s[s.amenazada], s[s.endemica]; x = s.tierra_pct_mas_alla_similar
        filas.append({'dominio': 'tierra', 'grupo': g, 'n': len(s), 'amenazadas': len(a), 'endemicas': len(e),
                      'amenazadas_endemicas': int((s.amenazada & s.endemica).sum()),
                      'med_todas': x.median(), 'med_amenazadas': a.tierra_pct_mas_alla_similar.median(),
                      'med_endemicas': e.tierra_pct_mas_alla_similar.median(),
                      'amenazadas_mitad': int((a.tierra_pct_mas_alla_similar > 50).sum()),
                      'endemicas_mitad': int((e.tierra_pct_mas_alla_similar > 50).sum()),
                      'amen_endem_mitad': int((s.amenazada & s.endemica & (x > 50)).sum())})
    for c in CAPAS:
        for g in MAR_G:
            s = m[m.grupo == g]; a = s[s.amenazada]; x = f'mar_{c}_pct_mas_alla_similar'
            filas.append({'dominio': c, 'grupo': g, 'n': len(s), 'amenazadas': len(a), 'med_todas': s[x].median(),
                          'med_amenazadas': a[x].median(), 'amenazadas_mitad': int((a[x] > 50).sum())})
    return filas


def seleccion(t, m, n=2):
    """Most salient species: threatened (and, on land, endemic) with more than half of their range beyond the similar
    class, ranked by the unprotected area of their range beyond it."""
    sel = {}
    for g in TIERRA_G:
        s = t[(t.grupo == g) & t.amenazada & t.endemica & (t.tierra_pct_mas_alla_similar > 50)]
        sel[g] = s.sort_values('km2_fuera', ascending=False).head(n).to_dict('records')
    s = m[(m.grupo == 'MARINEFISH') & m.amenazada & (m.mar_superficie_pct_mas_alla_similar > 50) &
          (m.mar_fondo_pct_mas_alla_similar > 50)]
    sel['MARINEFISH'] = s.sort_values('km2_fuera_fondo', ascending=False).head(n).to_dict('records')
    return sel


# ---------------------------------------------------------------- species maps
DMIN = r'$d_{\mathrm{min}}$'
# palette chosen by the author: dark blue protected, crimson marks the climatic gaps
DENTRO = [('#003f91', 'Range inside protected areas'), ('#7dceff', f'Range with a protected analogue ({DMIN} ≤ 0.2)'),
          ('#99174d', f'Range in climatic gaps ({DMIN} > 0.2)')]
FUERA = ('#e4b7ce', 'Outside the range, climatic gaps')


def precargar(grupo, ids):
    """Standard range polygons of several species, read in one pass per shapefile part and cached as WKB in
    especies_uicn/resultados/rangos (local only: the IUCN terms forbid redistributing the polygons)."""
    dst = RES / 'rangos'; dst.mkdir(exist_ok=True)
    faltan = [int(i) for i in ids if not (dst / f'{int(i)}.npy').exists()]
    if not faltan:
        return
    por = {i: [] for i in faltan}
    for f in sorted((cu.UICN / grupo).glob('*_PART*.shp')):
        # the filtered field must be among the columns read, or GDAL returns nothing
        _, _, wkb, (idn, p, o, s) = pyogrio.raw.read(f, where=f"id_no IN ({','.join(map(str, faltan))})",
                                                     columns=['id_no', 'presence', 'origin', 'seasonal'])
        ok = np.isin(p, [1, 2]) & np.isin(o, [1, 2]) & np.isin(s, [1, 2, 3])
        for i, w in zip(idn[ok], wkb[ok]):
            por[int(i)].append(w)
    for i, ws in por.items():
        g = shapely.make_valid(shapely.from_wkb(np.array(ws, dtype=object)))
        np.save(dst / f'{i}.npy', shapely.to_wkb(g), allow_pickle=True)


def rango(grupo, id_no):
    precargar(grupo, [id_no])
    return shapely.from_wkb(np.load(RES / 'rangos' / f'{int(id_no)}.npy', allow_pickle=True))


def clases_mapa(dom, geoms, dmin):
    """0 outside, 1 outside without a similar protected climate, 2 protected, 3 similar, 4 without a similar protected climate."""
    g = [x for x in shapely.clip_by_rect(geoms, *dom.bounds) if not x.is_empty]
    r = dom.mascara(g)
    if not r.any():
        r = dom.mascara(g, True)
    prot = dom.union >= .5 * dom.area
    k = np.where(prot, 2, np.where(dmin <= .2, 3, 4))
    return np.where(r, k, np.where(~prot & (dmin > .2), 1, 0)).astype(np.float32), r


def ventana(base, r, razon, minimo=3.0, pad=.12):
    """Window around the range with the panel's height/width ratio (degrees of longitude scaled by latitude)."""
    i = base.flat[r]
    x, y = base.xs[i % base.w], base.ys[i // base.w]
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    k = np.cos(np.deg2rad(cy))
    alto = max((y.max() - y.min()) * (1 + 2 * pad), (x.max() - x.min()) * k * (1 + 2 * pad) * razon, minimo)
    ancho = alto / razon / k
    return (cx - ancho / 2, cx + ancho / 2), (cy - alto / 2, cy + alto / 2), k


def recorte(base, img, xl, yl):
    """Part of the domain raster inside the window (drawing the whole grid in every panel is slow)."""
    a = base.aff
    c0, c1 = max(int((xl[0] - a.c) / a.a) - 1, 0), min(int((xl[1] - a.c) / a.a) + 2, base.w)
    r0, r1 = max(int((yl[1] - a.f) / a.e) - 1, 0), min(int((yl[0] - a.f) / a.e) + 2, base.h)
    return base.raster(img)[r0:r1, c0:c1], (a.c + c0 * a.a, a.c + c1 * a.a, a.f + r1 * a.e, a.f + r0 * a.e)


@functools.lru_cache(maxsize=None)
def formas():
    """Countries (Peru replaced by the continental outline) and departments, read once."""
    peru = lienzo(TIERRA).outline
    paises = [peru if p['admin'] == 'Peru' else s for p, s in geojson(INSUMOS / 'paises.geojson')]
    return paises, [s for _, s in departamentos()]


def panel_especie(ax, base, img, r, titulo, linea, razon, mar=False, esc=1.0, minimo=3.0):
    colores = ['#ffffff', FUERA[0]] + [c for c, _ in DENTRO]
    cmap = ListedColormap(colores); cmap.set_bad((0, 0, 0, 0))
    xl, yl, k = ventana(base, r, razon, minimo)
    corta = lambda s: shapely.clip_by_rect(s, xl[0], yl[0], xl[1], yl[1])  # noqa: E731
    paises, deps = formas()
    ax.set_facecolor('#f4f3ef')
    for s in map(corta, paises):
        if not s.is_empty:
            relleno(ax, s, facecolor='#e4e3df' if mar else '#ecebe6', edgecolor='none', zorder=0)
            contorno(ax, s, color='#b9b8b2', lw=.4 * esc, zorder=1)
    sub, ext = recorte(base, img, xl, yl)
    ax.imshow(sub, cmap=cmap, vmin=-.5, vmax=4.5, extent=ext, interpolation='nearest', zorder=2)
    if not mar:
        for s in map(corta, deps):
            contorno(ax, s, color='#9a9994', lw=.25 * esc, alpha=.6, zorder=3)
    contorno(ax, corta(base.outline), color='#52514e', lw=.45 * esc, zorder=4)
    if r.sum() < 60:  # ranges of a few pixels would be lost at this scale: circle each separate patch
        m2 = np.zeros(base.h * base.w, bool); m2[base.flat[r]] = True
        lab, n = ndimage.label(m2.reshape(base.h, base.w), structure=np.ones((3, 3)))
        for fy, fx in ndimage.center_of_mass(lab > 0, lab, range(1, n + 1)):
            ax.plot(base.xs[0] + base.aff.a * fx, base.ys[0] + base.aff.e * fy, marker='o', ms=13 * esc, mfc='none',
                    mec='#1f1f1d', mew=.9 * esc, zorder=5)
    ax.set_xlim(xl); ax.set_ylim(yl); ax.set_aspect(1 / k, adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#bdbcb6'); s.set_linewidth(.5)
    ax.set_title(titulo, loc='left', fontsize=7.4 * esc, style='italic', weight='bold', pad=21 * esc)
    ax.text(0, 1.012, linea, transform=ax.transAxes, fontsize=6 * esc, color='#52514e', va='bottom', ha='left')
    if shapely.clip_by_rect(base.outline, xl[0], yl[0], xl[1], yl[1]).area >= .5 * base.outline.area:
        return  # the window holds at least half of the domain (Peru or the 200 nautical miles), so it locates itself
    # locator: the domain with the window, on a white card in a corner that is usually outside the domain
    # (lower left, open ocean, on land; upper right, land, at sea)
    ins = ax.inset_axes([.675, .66, .30, .32] if mar else [.025, .02, .30, .32]); ins.set_xticks([]); ins.set_yticks([])
    for s in ins.spines.values():
        s.set_color('#bdbcb6'); s.set_linewidth(.4)
    lineas = [lienzo(TIERRA).outline, base.outline] if mar else [base.outline]
    for s in lineas:
        contorno(ins, s, color='#6b6b66', lw=.35 * esc)
    ins.add_patch(plt.Rectangle((xl[0], yl[0]), xl[1] - xl[0], yl[1] - yl[0], fc='none', ec='#d9480f', lw=.8 * esc))
    x0, y0, x1, y1 = shapely.union_all(lineas).bounds
    ins.set_xlim(x0 - .4, x1 + .4); ins.set_ylim(y0 - .4, y1 + .4)
    ins.set_aspect(1 / np.cos(np.deg2rad((y0 + y1) / 2)), adjustable='datalim'); ins.set_facecolor('white')


def leyenda(fig, y, tam=6.4):
    h = [Patch(fc=c, ec='#bdbdb5', lw=.4, label=l) for c, l in DENTRO + [FUERA]]
    h.append(Patch(fc='#ffffff', ec='#bdbdb5', lw=.4, label='Outside the range, with a protected analogue or protected'))
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(.5, y), ncol=2, fontsize=tam, frameon=False,
               handlelength=1.4, columnspacing=1.6, labelspacing=.35)


def cuantizar(path):
    """Flat-colour maps: 256 colours keep them sharp and make the Word file several times lighter."""
    from PIL import Image
    Image.open(path).convert('RGB').quantize(colors=256).save(path, optimize=True)


def dominio_tierra():
    return cu.Dominio(TIERRA, {'t': TIERRA / 'mapas_distancia/cond100/sistema_distancia_minima.tif'})


ATLAS_FIG = {'AMPHIBIANS': 'S3-1', 'MAMMALS': 'S3-2', 'REPTILES': 'S3-3', 'PLANTS': 'S3-4', 'BIRDS': 'S3-5', 'MARINEFISH': 'S3-6',
             'SEABIRDS': 'S3-7'}   # figures of File S3


def pct_global(x):
    """Share of the global range in map subtitles; ranges spread over the oceans round to 0, so they read '<1'."""
    return '<1' if x < 1 else f'{x:.0f}'


def seleccion_mapas(t, m):
    """Species mapped in File S3, three conditions: threatened (CR, EN or VU) or > 90 % of the global range in Peru (at sea,
    in the 200 nautical miles), and > 50 % of the range in the domain in climatic gaps (at sea, in at least one layer;
    seabirds have the sea surface only)."""
    q = t[(t.amenazada | (t.pct_global_en_peru > ATLAS_GLOBAL)) & (t.tierra_pct_mas_alla_similar > 50)].sort_values('sci_name')
    qm = m[(m.amenazada | (m.pct_global_en_mar > ATLAS_GLOBAL)) &
           ((m.mar_superficie_pct_mas_alla_similar > 50) | (m.mar_fondo_pct_mas_alla_similar > 50))]
    qm = qm.sort_values(['grupo', 'sci_name'], key=lambda s: s.map(MAR_G.index) if s.name == 'grupo' else s)
    return q, qm


def fig_atlas(atlas, filas=4, cols=4):
    """Land species of seleccion_mapas, alphabetically, 16 per page."""
    base, dom = lienzo(TIERRA), dominio_tierra()
    paginas, n = {}, filas * cols
    for g, especies in atlas.items():
        precargar(g, [e['id_no'] for e in especies])
        paginas[g] = []
        for k in range(0, len(especies), n):
            fig, axs = plt.subplots(filas, cols, figsize=(7.4, 9.0))
            for ax, e in zip(axs.flat, especies[k:k + n]):
                img, r = clases_mapa(dom, rango(g, e['id_no']), dom.dmin['t'])
                linea = (f"{e['categoria']} · {en(e['tierra_km2'], 0 if e['tierra_km2'] >= 10 else 1)} km² in Peru "
                         f"({pct_global(e['pct_global_en_peru'])}% of global)\n"
                         f"{e['tierra_pct_mas_alla_similar']:.1f}% in climatic gaps")
                panel_especie(ax, base, img, r, e['sci_name'], linea, 1.0, esc=.78, minimo=1.0)
            for ax in axs.flat[len(especies[k:k + n]):]:
                ax.set_axis_off()
            fig.subplots_adjust(left=.01, right=.99, top=.97, bottom=.08, wspace=.10, hspace=.30)
            leyenda(fig, .0, tam=6)
            path = SUPL / f"Fig{ATLAS_FIG[g]}_{G_EN[g].lower()}_{k // n + 1:02d}.png"
            fig.savefig(path, dpi=220, bbox_inches='tight'); plt.close(fig)
            cuantizar(path)
            paginas[g].append(path.name)
            print('atlas', path.name, flush=True)
    return paginas


def fig_atlas_mar(especies, por_pagina=8, archivo=f"Fig{ATLAS_FIG['MARINEFISH']}_marine_species"):
    """Marine fishes, mammals and reptiles of seleccion_mapas: sea surface and seafloor side by side, 8 species per page."""
    mb = MAR / 'superficie'
    dom = cu.Dominio(mb, {c: MAR / c / 'mapas_distancia/cond100/sistema_distancia_minima.tif' for c in CAPAS})
    for g in {e['grupo'] for e in especies}:
        precargar(g, [e['id_no'] for e in especies if e['grupo'] == g])
    paginas = []
    for k in range(0, len(especies), por_pagina):
        fig, axs = plt.subplots(4, 4, figsize=(7.4, 9.0))
        for n, e in enumerate(especies[k:k + por_pagina]):
            geoms = rango(e['grupo'], e['id_no'])
            for j, (c, lab) in enumerate((('superficie', 'sea surface'), ('fondo', 'seafloor'))):
                img, r = clases_mapa(dom, geoms, dom.dmin[c])
                linea = (f"{e['categoria']} · {lab} · {en(e['mar_km2'])} km² ({pct_global(e['pct_global_en_mar'])}% of global)\n"
                         f"{e[f'mar_{c}_pct_mas_alla_similar']:.1f}% in climatic gaps")
                panel_especie(axs.flat[2 * n + j], lienzo(MAR / c), img, r, e['sci_name'], linea, 1.0, mar=True, esc=.78)
        for ax in axs.flat[2 * len(especies[k:k + por_pagina]):]:
            ax.set_axis_off()
        fig.subplots_adjust(left=.01, right=.99, top=.97, bottom=.08, wspace=.10, hspace=.30)
        leyenda(fig, .0, tam=6)
        path = SUPL / f'{archivo}_{k // por_pagina + 1:02d}.png'
        fig.savefig(path, dpi=220, bbox_inches='tight'); plt.close(fig)
        cuantizar(path)
        paginas.append(path.name)
    return paginas


def fig_atlas_aves_mar(especies, archivo=f"Fig{ATLAS_FIG['SEABIRDS']}_seabirds", filas=4, cols=4):
    """Seabirds at the sea surface, the only layer that describes their habitat: one map per species, 16 per page."""
    dom = cu.Dominio(MAR / 'superficie', {'superficie': MAR / 'superficie/mapas_distancia/cond100/sistema_distancia_minima.tif'})
    base, n, paginas = lienzo(MAR / 'superficie'), filas * cols, []
    for k in range(0, len(especies), n):
        fig, axs = plt.subplots(filas, cols, figsize=(7.4, 9.0))
        for ax, e in zip(axs.flat, especies[k:k + n]):
            img, r = clases_mapa(dom, rango('BIRDS', e['id_no']), dom.dmin['superficie'])
            linea = (f"{e['categoria']} · sea surface · {en(e['mar_km2'])} km² ({pct_global(e['pct_global_en_mar'])}% of global)\n"
                     f"{e['mar_superficie_pct_mas_alla_similar']:.1f}% in climatic gaps")
            panel_especie(ax, base, img, r, e['sci_name'], linea, 1.0, mar=True, esc=.78)
        for ax in axs.flat[len(especies[k:k + n]):]:
            ax.set_axis_off()
        fig.subplots_adjust(left=.01, right=.99, top=.97, bottom=.08, wspace=.10, hspace=.30)
        leyenda(fig, .0, tam=6)
        path = SUPL / f'{archivo}_{k // n + 1:02d}.png'
        fig.savefig(path, dpi=220, bbox_inches='tight'); plt.close(fig)
        cuantizar(path)
        paginas.append(path.name)
    return paginas


ESTRATOS = [('pct_protegido', 'Protected', VERDE), ('pct_similar', 'Unprotected, ≤ 0.2', ANALOGO[1]),
            ('pct_moderado', 'Unprotected, 0.2–0.3', VACIO[0]), ('pct_debil', 'Unprotected, 0.3–0.5', VACIO[1]),
            ('pct_poco_similar', 'Unprotected, 0.5–1', VACIO[2]), ('pct_no_analogo', 'Unprotected, > 1', VACIO[3]),
            ('pct_mas_alla_similar', 'Climatic gaps, > 0.2', VACIO[2])]


def campo(pre, clave):
    """Column of a stratum: protection is the same in both maritime layers, the classes are not."""
    return 'mar_pct_protegido' if clave == 'pct_protegido' and pre.startswith('mar_') else f'{pre}{clave}'


def fig_cuotas(columnas, archivo, panel=1.037, izq=.814, der=.074, dst=None):
    """Histograms, across the species of each group, of the share of their range in each stratum (protected and
    unprotected classes of d_min). Bars, all species; line, threatened species. Panel width and margins in inches,
    fixed, so that more columns widen the figure instead of narrowing the panels."""
    bins = np.linspace(0, 100, 11)
    n = len(columnas); ancho = izq + der + panel * (n + .32 * (n - 1))
    fig, axs = plt.subplots(len(ESTRATOS), n, figsize=(ancho, 8.8), sharex=True)
    for j, (tit, s, pre) in enumerate(columnas):
        a = s[s.amenazada]
        axs[0, j].annotate(f'{tit}\n(n = {en(len(s))}; threatened {len(a)})', (.5, 1.2), xycoords='axes fraction',
                           ha='center', va='bottom', fontsize=7, weight='bold')
        for i, (clave, lab, col) in enumerate(ESTRATOS):
            ax = axs[i, j]; x = s[campo(pre, clave)].to_numpy()
            ax.bar(bins[:-1], 100 * np.histogram(x, bins)[0] / len(x), width=10, align='edge', color=col,
                   ec='#8a8986', lw=.4, zorder=2)
            if len(a):
                ax.stairs(100 * np.histogram(a[campo(pre, clave)], bins)[0] / len(a), bins, color='#1f1f1d', lw=.9, zorder=3)
            med = f'median {np.median(x):.1f}' + (f' · threatened {np.median(a[campo(pre, clave)]):.1f}' if len(a) else '')
            ax.set_title(med, loc='right', fontsize=5.4, color='#1f1f1d', pad=2)
            ax.set_xlim(0, 100); ax.set_ylim(0, None); ax.set_xticks([0, 25, 50, 75, 100])
            ax.spines[['top', 'right']].set_visible(False); ax.tick_params(labelsize=5.8, length=2)
            if j == 0:
                ax.set_ylabel(f'{lab}\n% of species', fontsize=6.2)
            if i == len(ESTRATOS) - 1:
                ax.set_xlabel('% of the range in the domain', fontsize=6.2)
    fig.legend(handles=[Patch(fc='#d6d6d0', ec='#8a8986', lw=.4, label='All species'),
                        Line2D([], [], color='#1f1f1d', lw=.9, label='Threatened species (CR, EN, VU)')],
               loc='lower center', bbox_to_anchor=(.5, 0), ncol=2, fontsize=6.6, frameon=False)
    fig.subplots_adjust(left=izq / ancho, right=1 - der / ancho, top=.92, bottom=.08, wspace=.32, hspace=.42)
    fig.savefig((dst or SUPL) / archivo, bbox_inches='tight'); plt.close(fig)


def fig_cuotas_tierra(t, dst=None):
    fig_cuotas([(G_EN[g], t[t.grupo == g], 'tierra_') for g in TIERRA_G], 'FigS12_species_shares_continental.png', dst=dst)


def fig_cuotas_mar(m, dst=None):
    for k, (c, lab) in enumerate((('superficie', 'sea surface'), ('fondo', 'seafloor')), 1):
        fig_cuotas([(f'{G_MAR_EN[g]},\n{lab}', m[m.grupo == g], f'mar_{c}_') for g in MAR_G
                    if m.loc[m.grupo == g, f'mar_{c}_pct_mas_alla_similar'].notna().any()],   # seabirds, surface only
                   f'FigS13_species_shares_maritime_{k:02d}.png', dst=dst)


def distribuciones(t, m):
    """Numbers quoted with Figs. S12 and S13."""
    out = {}
    for g in TIERRA_G:
        s = t[t.grupo == g]
        out[g] = {'med_prot': s.tierra_pct_protegido.median(), 'prot_menos10': 100 * (s.tierra_pct_protegido < 10).mean(),
                  'med_prot_amen': s[s.amenazada].tierra_pct_protegido.median(),
                  'fuera_menos10': 100 * (s.tierra_pct_mas_alla_similar < 10).mean(),
                  'prot_cero': 100 * (s.tierra_pct_protegido == 0).mean(), 'med_similar': s.tierra_pct_similar.median(),
                  'fuera_mitad': 100 * (s.tierra_pct_mas_alla_similar > 50).mean(),
                  'fuera_cero': 100 * (s.tierra_pct_mas_alla_similar == 0).mean(),
                  'no_analogo_cero': 100 * (s.tierra_pct_no_analogo == 0).mean(),
                  'poco_similar_cero': 100 * (s.tierra_pct_poco_similar == 0).mean()}
    p = m[m.grupo == 'MARINEFISH']
    out['MARINEFISH'] = {'med_prot': p.mar_pct_protegido.median(), 'prot_menos10': 100 * (p.mar_pct_protegido < 10).mean(),
                         **{f'fuera_mitad_{c}': 100 * (p[f'mar_{c}_pct_mas_alla_similar'] > 50).mean() for c in CAPAS}}
    return out


def main():
    t, m = cargar()
    ref = referencias()
    res = resumen(t, m)
    sel = seleccion(t, m)
    q, qm = seleccion_mapas(t, m)
    atlas = {g: q[q.grupo == g].to_dict('records') for g in TIERRA_G}
    cols_t = ['grupo', 'sci_name', 'familia', 'categoria', 'endemica', 'km2_en_peru', 'pct_global_en_peru', 'tierra_km2',
              'tierra_pct_protegido'] + [f'tierra_pct_{c}' for c in CLS] + ['tierra_pct_mas_alla_similar']
    cols_m = ['grupo', 'clase', 'sci_name', 'familia', 'categoria', 'mar_km2', 'mar_pct_protegido'] + \
             [f'mar_{c}_pct_{x}' for c in CAPAS for x in ('similar', 'mas_alla_similar')]
    amen_t = t[t.amenazada].sort_values(['grupo', 'tierra_pct_mas_alla_similar', 'sci_name'], ascending=[True, False, True],
                                        key=lambda s: s.map(TIERRA_G.index) if s.name == 'grupo' else s)
    amen_m = m[m.amenazada].sort_values(['grupo', 'mar_fondo_pct_mas_alla_similar', 'sci_name'], ascending=[True, False, True],
                                        key=lambda s: s.map(MAR_G.index) if s.name == 'grupo' else s)
    out = {'referencias': ref, 'resumen': res, 'n_tierra': len(t), 'n_mar': len(m),
           'seleccion': {g: [{k: e[k] for k in e if not isinstance(e[k], float) or np.isfinite(e[k])} for e in v]
                         for g, v in sel.items()},
           'amenazadas_tierra': amen_t[cols_t].to_dict('records'), 'amenazadas_mar': amen_m[cols_m].to_dict('records'),
           'totales_tierra': {'n': len(t), 'amenazadas': int(t.amenazada.sum()), 'endemicas': int(t.endemica.sum()),
                              'amenazadas_mitad': int((t.amenazada & (t.tierra_pct_mas_alla_similar > 50)).sum()),
                              'amen_endem_mitad': int((t.amenazada & t.endemica & (t.tierra_pct_mas_alla_similar > 50)).sum()),
                              'menor_pixel': int((t.km2_en_peru < 1).sum()),
                              'menor_pixel_amenazadas': int((t.amenazada & (t.km2_en_peru < 1)).sum())},
           'familias_plantas': t[t.grupo == 'PLANTS'].familia.value_counts().head(3).to_dict(),
           'distribuciones': distribuciones(t, m),
           'atlas': {g: [e['sci_name'] for e in v] for g, v in atlas.items()},
           'atlas_resumen': {'n': len(q), 'amenazadas': int(q.amenazada.sum()), 'DD': int((q.categoria == 'DD').sum())},
           'atlas_mar': qm.sci_name.tolist(),
           'atlas_mar_resumen': {'amenazadas': int(qm.amenazada.sum()), 'DD': int((qm.categoria == 'DD').sum())},
           'max_global_mar': {g: float(m[m.grupo == g].pct_global_en_mar.max()) for g in MAR_G}}
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)),
                        encoding='utf-8')
    print(pd.DataFrame(res).round(1).to_string(index=False))
    for g, v in sel.items():
        print(g, [e['sci_name'] for e in v])


if __name__ == '__main__':
    main()
