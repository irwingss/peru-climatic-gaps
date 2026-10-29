"""Supplementary figures drawn from stored results: Figs. S1 (study area), S4 (reading scale of d), S5 (climatic gaps),
S6 (example analogue maps), S7-S8 (coverage curves against the null networks), S10 (precipitation correction), S11
(SENAMHI stations) and S12-S13 (species shares). Figs. S2 and S3 are drawn in flujogramas/ (HTML) and Fig. S9 by
clima_local/sesgo_precipitacion.py.
Run: python figuras_suplementarias.py [figS1 figS4 ...] [--dst carpeta]   (default: all of them into figures/supplementary)
"""
import sys
from figuras_comun import *  # noqa: F401,F403  (shared constants, canvases and map helpers)
from figuras_comun import _CACHE  # noqa: F401

DST = Path(sys.argv[sys.argv.index('--dst') + 1]) if '--dst' in sys.argv else SUPL
RES = RAIZ / 'resultados_v7'
NUL = {'continental': RES / '04_redes_nulas/continental_superposicion_limitada/N2c',
       'superficie': RES / '04_redes_nulas/superficie/N2c', 'fondo': RES / '04_redes_nulas/fondo/N2c'}
W2 = 7.2                                                    # double-column width (in)
C = CIFRAS['continental']


M = CIFRAS['maritimo']


ECO_ES = {e['codigo']: e['nombre'] for b in (TIERRA, MAR / 'superficie')  # stratum names used in curvas_cobertura.csv
          for e in json.loads((b / 'preparados/ecorregiones_minam.json').read_text(encoding='utf-8'))['ecorregiones']}


def minam(base):
    return np.load(base / 'preparados/ecorregiones_minam.npz')['ecorregion']


def figS1_area_estudio():
    zt = minam(TIERRA)
    base = lienzo(TIERRA)
    rgb = np.array([to_rgb(COLOR[e]) for e in ECO])
    img = np.zeros((base.h * base.w, 4)); img[base.flat, :3] = rgb[zt]; img[base.flat, 3] = 1
    # sea surface: the two MINAM marine ecoregions; seafloor depth strata as the 200 m and 1 000 m boundaries
    bs = lienzo(MAR / 'superficie')
    rgbm = np.array([to_rgb(COLOR[e]) for e in ECO_MAR])
    imm = np.zeros((bs.h * bs.w, 4)); imm[bs.flat, :3] = rgbm[minam(MAR / 'superficie')]; imm[bs.flat, 3] = 1
    fig, ax = plt.subplots(figsize=(7.4, 6.9))
    ax.set_facecolor('#f7f7f5')
    paises = geojson(INSUMOS / 'paises.geojson')
    for p, s in paises:
        if p['admin'] != 'Peru':
            relleno(ax, s, facecolor='#ecebe6', edgecolor='none', zorder=0)
    ax.imshow(imm.reshape(bs.h, bs.w, 4), extent=bs.extent, interpolation='nearest', zorder=1)
    bm = lienzo(MAR / 'fondo')
    zm = np.load(MAR / 'fondo/preparados/zonas.npz')['ecorregion']
    som = np.zeros((bm.h * bm.w, 4)); som[bm.flat, :3] = to_rgb(SOMBRA); som[bm.flat, 3] = np.array(ALFA_PROF)[zm]
    ax.imshow(som.reshape(bm.h, bm.w, 4), extent=bm.extent, interpolation='nearest', zorder=1.5)
    ax.imshow(img.reshape(base.h, base.w, 4), extent=base.extent, interpolation='nearest', zorder=2.5)
    for _, s in departamentos():
        contorno(ax, s, color='#7d7c77', lw=.28, alpha=.55, zorder=3)
    for p, s in paises:
        if p['admin'] != 'Peru':
            contorno(ax, s, color='#8a8986', lw=.55, zorder=4)
    contorno(ax, base.outline, color='#52514e', lw=.6, zorder=4)
    for txt, (x, y) in {'Ecuador': (-79.1, -1.2), 'Colombia': (-72.6, -.3), 'Brazil': (-68.4, -8.6),
                        'Bolivia': (-67.6, -16.0), 'Chile': (-69.1, -19.9)}.items():
        ax.text(x, y, txt, ha='center', va='center', fontsize=7.6, color='#52514e', path_effects=halo, zorder=6)
    ax.text(-82.6, -12.5, 'Pacific\nOcean', ha='center', va='center', fontsize=8, style='italic', color='#52514e',
            path_effects=halo, zorder=6)
    ax.set_xlim(-85, -66.6); ax.set_ylim(-20.6, .6); ax.set_aspect('equal')
    ax.set_xticks(range(-84, -67, 4)); ax.set_yticks(range(-20, 1, 4))
    ax.set_xticklabels([f'{-x}°W' for x in range(-84, -67, 4)])
    ax.set_yticklabels([f'{-y}°S' if y else '0°' for y in range(-20, 1, 4)])
    ax.tick_params(labelsize=7, length=2)
    ax.grid(color='#bdbcb6', lw=.3, ls=':', zorder=0)
    # legend drawn by hand so that headings sit flush with the left edge of the colour boxes
    fig.canvas.draw()
    pos = ax.get_position()
    lax = fig.add_axes([pos.x1 + .02, pos.y0, .4, pos.height]); lax.set_axis_off()
    lax.set_xlim(0, .4 * fig.get_figwidth() * 72); lax.set_ylim(pos.height * fig.get_figheight() * 72, 0)  # points, from the top
    y = [4.0]

    def cabecera(t, size, gap):
        y[0] += gap; lax.text(0, y[0], t, fontsize=size, weight='bold', va='center', ha='left', color='#0b0b0b'); y[0] += 11.5

    def fila(t, color=None, linea=None):
        if linea:
            lax.plot([0, 11], [y[0], y[0]], **linea)
        else:  # a pair of colours fills the box in two halves
            colores = color if isinstance(color, list) else [color]
            for k, c in enumerate(colores):
                ancho = 11 / len(colores)
                lax.add_patch(plt.Rectangle((k * ancho, y[0] - 3.6), ancho, 7.2, fc=c, ec='none'))
        lax.text(15, y[0], t, fontsize=6.6, va='center', ha='left', color='#0b0b0b'); y[0] += 11.2

    cabecera('Ecoregions – Continental Domain of Peru', 7, 0)
    for e in ECO:
        fila(ECO_EN[e], COLOR[e])
    cabecera('Ecoregions – Sea Surface, Maritime Domain of Peru', 7, 20)
    for e in ECO_MAR:
        fila(ECO_EN[e], COLOR[e])
    cabecera('Seafloor Depth – Maritime Domain of Peru (shading)', 7, 20)
    for al, l in zip(ALFA_PROF, ['Shelf (0–200 m)', 'Slope (200–1 000 m)', 'Deep floor (> 1 000 m)']):
        fila(l, [tuple((1 - al) * np.array(to_rgb(COLOR[e])) + al * np.array(to_rgb(SOMBRA))) for e in ECO_MAR])
    y[0] += 6
    fila('Country boundary', linea=dict(color='#8a8986', lw=.6))
    fila('Department boundary (Peru)', linea=dict(color='#7d7c77', lw=.4, alpha=.7))
    fig.savefig(DST / 'FigS1_study_area.png', bbox_inches='tight'); plt.close(fig)


def figS4_escala():
    """Median Mahalanobis distance between pixels as a function of their separation, with the reading classes."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.5), sharey=True)
    alto = {'weak': .45, 'low similarity': .6}   # clear of the seafloor dashed line (0.40) and of the curves of (b)
    for ax in (a1, a2):
        for (c, al, t), lo, hi in zip(CLASES_D, CORTES_D[:-1], CORTES_D[1:]):
            ax.axhspan(lo, hi, color=to_rgba(c, al * .55), lw=0, zorder=0)
            ax.text(1.0, alto.get(t, (lo + hi) / 2 if hi - lo > .07 else hi - .012), t, transform=ax.get_yaxis_transform(),
                    ha='right', va='center', fontsize=5.8, color='#52514e', zorder=1)
        ax.set_xscale('log'); ax.set_ylim(0, 1.6); ax.set_xlim(.7, 700)
        ax.set_xlabel('Separation between pixels along a row (km)')
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: en(v, 0 if v >= 1 else 1)))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.1f}'))
        ax.spines[['top', 'right']].set_visible(False); ax.tick_params(labelsize=7, length=2)

    def nota(ax, xy, lab, dx, dy, col='#1f1f1d', **kw):
        ax.annotate(lab, xy, xytext=(dx, dy), textcoords='offset points', fontsize=6.2, color=col,
                    arrowprops=dict(arrowstyle='-', color='#52514e', lw=.5), zorder=4, **kw)

    def cruce(ax, e, col, dx, dy):   # d = 0.2 interpolated on the curve (open marker: not a measured lag)
        ax.plot(e['km_d0.2'], .2, 'o', ms=3.4, mfc='white', mec=col, mew=.9, zorder=4)
        nota(ax, (e['km_d0.2'], .2), f"≈ {e['km_d0.2']:.0f} km apart (0.2)", dx, dy, col,
             bbox=dict(fc='white', ec='none', alpha=.75, pad=.6))   # legible over the dark classes

    e = C['escala']
    km = [f['km'] for f in e['curva']]; dd = [f['d'] for f in e['curva']]
    a1.plot(km, dd, '-o', color='#1f1f1d', ms=3, lw=1.2, zorder=3, label='Median d between pixels')
    a1.axhline(e['azar'], color='#1f1f1d', lw=.8, ls='--', zorder=2)
    a1.text(.75, e['azar'] + .03, f"random pairs of pixels ({e['azar']:.2f})", fontsize=6.4, color='#1f1f1d', zorder=4)
    nota(a1, (km[0], dd[0]), f'neighbouring pixels\n({dd[0]:.2f}, ≈ {km[0]:.1f} km)', -4, 30)
    cruce(a1, e, '#1f1f1d', 8, -14)
    a1.set_ylabel('Mahalanobis distance d (median)')
    a1.set_title('(a) Continental domain', loc='left', fontsize=8, weight='bold')
    vecinos = []
    for capa, col, mk, lab, txt, dx, dy in (('superficie', '#2a78d6', 'o', 'Sea surface', 'sea surface', 14, -19),
                                            ('fondo', '#0b3c6b', 's', 'Seafloor', 'seafloor', 10, -11)):
        e = M[capa]['escala']
        a2.plot([f['km'] for f in e['curva']], [f['d'] for f in e['curva']], '-' + mk, color=col, ms=2.6, lw=1.2, zorder=3, label=lab)
        a2.axhline(e['azar'], color=col, lw=.8, ls='--', zorder=2)
        a2.text(4.3, e['azar'] + .03, f"random pairs of pixels, {txt} ({e['azar']:.2f})", fontsize=6.4, color=col, zorder=4)
        cruce(a2, e, col, dx, dy)
        vecinos.append(e['curva'][0])
    d0 = ' and '.join(dict.fromkeys(f"{v['d']:.2f}" for v in vecinos))
    nota(a2, (vecinos[0]['km'], vecinos[0]['d']), f'neighbouring pixels\n({d0}, ≈ 5.5 km)', -2, 22)   # 0.05° pixel, as in the text
    a2.set_xlim(4, 700)
    a2.legend(fontsize=6.4, frameon=False, loc='upper left')
    a2.set_title('(b) Maritime domain (metric of each layer)', loc='left', fontsize=8, weight='bold')
    fig.tight_layout(w_pad=1.5)
    fig.savefig(DST / 'FigS4_distance_scale.png', bbox_inches='tight'); plt.close(fig)


def _leyenda_mapa(ax, extra, loc=(1.0, .62), titulo='Mahalanobis distance d'):
    ax.legend(handles=leyenda_d(extra), title=titulo, title_fontsize=7.5, fontsize=7, loc='upper left',
              bbox_to_anchor=loc, frameon=False, alignment='left')


CLASE_FIG = [('similar', 'Similar or closer (≤ 0.2)', ANALOGO[1]), ('moderado', 'Moderate (0.2–0.3)', VACIO[0]),
             ('debil', 'Weak (0.3–0.5)', VACIO[1]), ('poco_similar', 'Low similarity (0.5–1)', VACIO[2]),
             ('no_analogo', 'Not analogous (> 1)', VACIO[3])]


def figS5_vacios():
    """Unprotected surface by reading class of d_min: where (ecoregions) and what climate (weighted quantiles)."""
    cl = C['clases']
    fig = plt.figure(figsize=(7.4, 7.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.3, 1.15], hspace=.42, wspace=.55)
    a = fig.add_subplot(gs[0, :])
    filas = ECO[::-1]
    izq = np.zeros(len(filas))
    for clave, lab, col in CLASE_FIG:
        v = np.array([cl[clave]['km2'] * cl[clave]['grupos'][g] / 100 / 1e3 for g in filas])
        a.barh(range(len(filas)), v, left=izq, color=col, ec='#8a8986', lw=.4, height=.62, label=lab)
        izq += v
    a.set_yticks(range(len(filas))); a.set_yticklabels([ECO_EN[g] for g in filas], fontsize=7)
    a.set_xlabel('Unprotected surface (thousand km²)', fontsize=7.5)
    a.spines[['top', 'right']].set_visible(False); a.tick_params(labelsize=7, length=2)
    a.legend(fontsize=6.4, frameon=False, ncol=2, loc='center right', bbox_to_anchor=(1.0, .42), title='Class of $d_{\\mathrm{min}}$',
             title_fontsize=6.8)
    a.set_title('(a) Unprotected surface by ecoregion and class of distance to the most similar protected climate',
                loc='left', fontsize=7.8, weight='bold')
    paneles = [('elevacion', 'Elevation\n(m)', False, '(b)'), ('bio01', 'Mean annual\ntemperature (°C)', False, '(c)'),
               ('bio12', 'Annual\nprecipitation (mm)', True, '(d)'), ('bio15', 'Precipitation\nseasonality (CV, %)', False, '(e)')]
    for k, (v, lab, logy, letra) in enumerate(paneles):
        ax = fig.add_subplot(gs[1, k])
        for j, (clave, _, col) in enumerate(CLASE_FIG):
            p10, p25, p50, p75, p90 = cl[clave][v]
            ax.plot([j, j], [p10, p90], color='#52514e', lw=.8, zorder=1)
            ax.add_patch(plt.Rectangle((j - .32, p25), .64, p75 - p25, fc=col, ec='#52514e', lw=.6, zorder=2))
            ax.plot([j - .32, j + .32], [p50, p50], color='#0b0b0b', lw=1.2, zorder=3)
        ax.set_xticks(range(len(CLASE_FIG))); ax.set_xticklabels(['≤ 0.2', '0.2–0.3', '0.3–0.5', '0.5–1', '> 1'], rotation=55, fontsize=6.2)
        if logy:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: en(x)))
        else:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: en(x)))
        ax.set_xlim(-.6, len(CLASE_FIG) - .4)
        ax.set_title(f'{letra} {lab}', loc='left', fontsize=7, weight='bold')
        ax.spines[['top', 'right']].set_visible(False); ax.tick_params(labelsize=6.4, length=2)
        ax.set_xlabel('Class of $d_{\\mathrm{min}}$', fontsize=6.4)
    fig.savefig(DST / 'FigS5_climatic_gap_conditions.png', bbox_inches='tight'); plt.close(fig)


def figS6_ejemplos():
    base = lienzo(TIERRA)
    areas = {a['anp_id']: a for a in C['areas']}
    casos = [('NAC_RN08', '(a) Pacaya-Samiria National Reserve'), ('NAC_SH03', '(b) Machupicchu Historic Sanctuary'),
             ('NAC_RN04', '(c) Lachay National Reserve')]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.7))
    for ax, (uid, tit) in zip(axs, casos):
        pct = areas[uid]['pct_ecorregiones']  # ecoregions with at least 10 % of the area, largest first
        sub = ', '.join(ECO_EN[e] for e in sorted(pct, key=lambda e: -pct[e]) if pct[e] >= 10)
        d = np.load(TIERRA / f'distancias/cond100/{uid}.npz')['distancia']
        panel_d(ax, base, d, base.shapes[uid])
        ax.set_title(f"{tit}\n{sub}\nd ≤ 0.2 outside the area: {en(areas[uid]['km2_d0.2'])} km²", loc='left',
                     fontsize=6.6, color='#1f1f1d')
    fig.subplots_adjust(wspace=.02, right=.80)
    _leyenda_mapa(axs[-1], [Patch(facecolor=VERDE, edgecolor=VERDE_OSCURO, lw=.6, label='Protected area')], loc=(1.0, .78))
    fig.savefig(DST / 'FigS6_example_analogue_maps.png', bbox_inches='tight'); plt.close(fig)


ESTRATO_CURVA = {'Perú': ('Perú', 'Peru (continental domain)'), **{ECO_ES[e]: (e, ECO_EN[e]) for e in ECO}}


def _curvas(ax, rows, nul, tit, xmax):
    d = np.array([float(r['d']) for r in rows])
    g = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    ax.fill_between(d, g('azar_p2_5'), g('azar_p97_5'), color=BANDA, lw=0, label='Null networks (central 95%)')
    ax.plot(d, g('azar_mediana'), color=MEDIANA, lw=1, label='Null median')
    ax.plot(d, g('real'), color=REAL, lw=1.8, label='Real network')
    ax.axvline(.2, color='#52514e', lw=.8, ls=(0, (1, 1.6)), zorder=1.5,   # where each curve gives F(0.2)
               label='$x$ = 0.2, cut-off for a protected analogue')
    ax.text(.97, .05, f"$\\bar{{D}}$: real network {nul['D_real']:.3f}\n"
            f"null median {nul['D_azar_mediana']:.3f} [{nul['D_azar_p2_5']:.3f}–{nul['D_azar_p97_5']:.3f}]",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=6.2, color='#0b0b0b')
    ax.set_title(tit, loc='left', fontsize=7.8, weight='bold')
    ax.set_xlim(0, xmax); ax.set_ylim(0, 1.02)
    ax.spines[['top', 'right']].set_visible(False); ax.grid(color='#ecebe7', lw=.4)
    ax.set_xticks(np.arange(0, xmax + .01, .2)); ax.set_yticks(np.arange(0, 1.01, .2))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.1f}'))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.1f}'))
    ax.tick_params(labelsize=7, length=2)


def figS7_curvas_continentales():
    """Coverage curves F(x) of the real network against the 999 N2c null networks, continental domain."""
    rows = list(csv.DictReader(open(NUL['continental'] / 'curvas_cobertura.csv', encoding='utf-8-sig')))
    fig, axs = plt.subplots(4, 3, figsize=(7.2, 9.0), sharey=True); axs = axs.ravel()
    for k, (ax, (est, (clave, tit))) in enumerate(zip(axs, ESTRATO_CURVA.items())):
        _curvas(ax, [r for r in rows if r['estrato'] == est], C['nulas'][clave], f'({"abcdefghij"[k]}) {tit}', 1.2)
    for ax in axs[[0, 3, 6, 9]]:
        ax.set_ylabel(r'$F(x)$, share with $d_{\mathrm{min}} \leq x$')
    for ax in axs[7:10]:
        ax.set_xlabel('Mahalanobis distance x')
    axs[10].axis('off'); axs[11].axis('off')
    axs[10].legend(*axs[0].get_legend_handles_labels(), loc='center', fontsize=7.2, frameon=False)
    fig.tight_layout(h_pad=1.4)
    fig.savefig(DST / 'FigS7_null_curves_continental.png', bbox_inches='tight'); plt.close(fig)


def figS8_curvas_maritimas():
    """As Fig. S7, for the sea surface and its ecoregions and the seafloor and its depth strata."""
    paneles = [('superficie', 'Mar peruano', 'todo', '(a) Sea surface, all ecoregions'),
               ('superficie', ECO_ES['MFH'], 'MFH', f"(b) {ECO_EN['MFH']}"), ('superficie', ECO_ES['MTR'], 'MTR', f"(c) {ECO_EN['MTR']}"),
               ('fondo', 'Mar peruano', 'todo', '(d) Seafloor, all strata'), ('fondo', 'Plataforma (0–200 m)', 'PLATAFORMA', '(e) Seafloor: shelf (0–200 m)'),
               ('fondo', 'Talud (200–1.000 m)', 'TALUD', '(f) Seafloor: slope (200–1 000 m)'),
               ('fondo', 'Fondo profundo (> 1.000 m)', 'PROFUNDO', '(g) Seafloor: deep floor (> 1 000 m)')]
    fig, axs = plt.subplots(3, 3, figsize=(7.2, 7.0), sharey=True); axs = axs.ravel(); axs[7].axis('off'); axs[8].axis('off')
    for ax, (capa, est, clave, tit) in zip(axs, paneles):
        rows = [r for r in csv.DictReader(open(NUL[capa] / 'curvas_cobertura.csv', encoding='utf-8-sig')) if r['estrato'] == est]
        _curvas(ax, rows, CIFRAS['maritimo'][capa]['nulas'][clave], tit, 1.2)
    for ax in axs[[0, 3, 6]]:
        ax.set_ylabel(r'$F(x)$, share with $d_{\mathrm{min}} \leq x$')
    for ax in axs[4:7]:
        ax.set_xlabel('Mahalanobis distance x')
    axs[7].legend(*axs[0].get_legend_handles_labels(), loc='center', fontsize=7.2, frameon=False)
    fig.tight_layout(h_pad=1.4)
    fig.savefig(DST / 'FigS8_null_curves_maritime.png', bbox_inches='tight'); plt.close(fig)


def _titulo(ax, t):
    ax.set_title(t, loc='left', fontsize=7.6, weight='bold')


# ======================================================================== supplementary: PISCO-corrected precipitation (Fig. S10)
def figS10_pisco():
    """Redrawn from the rasters stored by clima_local/sesgo_precipitacion.py, with the notation of the manuscript."""
    src, base = RES / '02_sesgo_precipitacion', lienzo(TIERRA)
    with rasterio.open(src / 'dmin_corregido_menos_original.tif') as ds:
        delta = ds.read(1, masked=True).astype(np.float32).filled(np.nan)
    with rasterio.open(src / 'vacios_cambio.tif') as ds:
        clase = ds.read(1, masked=True).astype(np.float32).filled(np.nan)
    ground = ListedColormap([SIN_DATO, '#ffffff']); ground.set_bad((0, 0, 0, 0))
    fig, axs = plt.subplots(1, 2, figsize=(W2, 5.2))
    cortes = [-.5, -.2, -.1, -.05, -.02, 0, .02, .05, .1, .2, .5]
    cm1 = plt.get_cmap('PuOr_r', 10); cm1.set_bad((0, 0, 0, 0))
    cm2 = ListedColormap(['#f2f2ee', '#d7301f', '#2b8cbe', '#636363']); cm2.set_bad((0, 0, 0, 0))
    for ax, v, cm, norm in ((axs[0], delta, cm1, BoundaryNorm(cortes, 10)), (axs[1], clase, cm2, BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4))):
        ax.imshow(base.domain, cmap=ground, vmin=0, vmax=1, extent=base.extent, interpolation='nearest')
        im = ax.imshow(v, cmap=cm, norm=norm, extent=base.extent, interpolation='nearest')
        contorno(ax, base.outline, color='#6b6b66', lw=.4)
        ax.set_aspect(1 / np.cos(np.deg2rad(-9.2))); ax.set_axis_off()
        if ax is axs[0]:
            cb = fig.colorbar(im, ax=ax, shrink=.55, ticks=cortes, format=matplotlib.ticker.FuncFormatter(lambda x, _: corto(x).replace('-', '−')))
            cb.set_label('$d_{\\mathrm{min}}$ corrected − original', fontsize=6.6); cb.ax.tick_params(labelsize=6, length=2)
    _titulo(axs[0], 'Change in $d_{\\mathrm{min}}$ after PISCO correction')
    _titulo(axs[1], 'Climatic gaps ($d_{\\mathrm{min}}$ > 0.2)')
    axs[1].legend(handles=[Patch(fc=c, ec='#bdbdb5', lw=.4, label=t) for c, t in zip(cm2.colors, (
        'Protected analogue in both', 'Gap only when corrected', 'Gap only in original', 'Gap in both'))],
        fontsize=6.2, frameon=False, loc='upper center', bbox_to_anchor=(.5, 0), ncol=2, handlelength=1.3)   # below the map
    fig.savefig(DST / 'FigS10_dmin_corrected_minus_original.png', bbox_inches='tight', dpi=300); plt.close(fig)


# ================================================================ supplementary figure
def figS11_estaciones():
    ok = [f for f in csv.DictReader(open(TIERRA / 'estaciones/estaciones.csv', encoding='utf-8-sig')) if f.get('bio01_obs')]
    pares = list(csv.DictReader(open(TIERRA / 'estaciones/pares_estaciones.csv', encoding='utf-8-sig')))
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7))
    for ax, v, lab, tit in ((axs[0], 'bio01', 'mean annual temperature (°C)', '(a)'),
                            (axs[1], 'bio12', 'annual precipitation (mm)', '(b)')):
        o = [float(f[f'{v}_obs']) for f in ok]; m = [float(f[f'{v}_chelsa']) for f in ok]
        ax.scatter(o, m, s=12, color='#1f1f1d'); lim = [min(o + m), max(o + m)]; ax.plot(lim, lim, color=CIAN, lw=.8)
        if v == 'bio12':
            ax.set_xscale('symlog', linthresh=10); ax.set_yscale('symlog', linthresh=10)
            fmt = matplotlib.ticker.FuncFormatter(lambda x, _: en(x))
            ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
        ax.set_xlabel(f'SENAMHI station, {lab}', fontsize=7); ax.set_ylabel('CHELSA, station pixel', fontsize=7)
        ax.set_title(tit, loc='left', fontsize=8, weight='bold')
    d7o = np.array([float(p['d7_obs']) for p in pares]); d7c = np.array([float(p['d7_chelsa']) for p in pares])
    d11 = np.array([float(p['d11_chelsa']) for p in pares])
    for lo, hi, col, s, lab in ((1, 99, VACIO[3], 5, '> 1 · not analogous'), (.5, 1, VACIO[2], 8, '0.5–1 · low similarity'),
                                (-1, .5, VACIO[1], 16, '≤ 0.5 · weak or closer')):
        m = (d11 > lo) & (d11 <= hi)
        axs[2].scatter(d7o[m], d7c[m], s=s, color=col, label=f'{lab} (n = {m.sum()})', zorder=3)
    axs[2].plot([0, d7o.max()], [0, d7o.max()], color=CIAN, lw=.8)
    axs[2].set_xlabel('d between stations, measured values (7 variables)', fontsize=7)
    axs[2].set_ylabel('d between their CHELSA pixels (7 variables)', fontsize=7)
    axs[2].set_title('(c)', loc='left', fontsize=8, weight='bold')
    axs[2].legend(fontsize=5.6, frameon=False, loc='upper left', title='11-variable d between their pixels', title_fontsize=5.8)
    for ax in axs:
        ax.tick_params(labelsize=6.5, length=2); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(DST / 'FigS11_stations.png', bbox_inches='tight'); plt.close(fig)


def figS12_S13_especies():
    """Shares of the range of each species in each stratum, continental (Fig. S12) and maritime (Fig. S13)."""
    import especies as E
    t, m = E.cargar()
    E.fig_cuotas_tierra(t, DST); E.fig_cuotas_mar(m, DST)


TODAS = [figS1_area_estudio, figS4_escala, figS5_vacios, figS6_ejemplos, figS7_curvas_continentales, figS8_curvas_maritimas,
         figS10_pisco, figS11_estaciones, figS12_S13_especies]

if __name__ == '__main__':
    DST.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:sys.argv.index('--dst')] + sys.argv[sys.argv.index('--dst') + 2:] if '--dst' in sys.argv else sys.argv[1:]
    for f in TODAS:
        if not args or f.__name__.split('_')[0] in args:
            f(); print('ok', f.__name__, flush=True)
