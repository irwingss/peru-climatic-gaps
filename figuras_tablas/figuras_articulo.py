"""Figures of the article and the table behind Fig. 4. Reads only stored results; nothing is re-run.

Fig. 1  Climatic gaps on land and at sea (dmin maps and coverage by ecoregion).
Fig. 2  The network against 999 N2c null networks (null D̄ histograms and SES by stratum).
Fig. 3  How much land, and who sustains current representation: effort curve, role of ACR/ACP and of the Nazca Ridge.
Fig. 4  Where to expand: climate and species outcomes at equal area, and one map per set of the scatter plots
        (greedy curve, prioritizr A, B, C17, D17, flat 17% and 30% species targets, median random set).
Graphical abstract (1328 × 531 px).

python figuras_articulo.py [fig1 fig2 fig3 fig4 ga] [--dst carpeta]   (default: all of them into figures/main)
"""
import sys
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.gridspec import GridSpec
from figuras_comun import *  # noqa: F401,F403  (shared constants, canvases and map helpers)

RES = RAIZ / 'resultados_v7'
NUL = {'continental': RES / '04_redes_nulas/continental_superposicion_limitada/N2c',
       'superficie': RES / '04_redes_nulas/superficie/N2c', 'fondo': RES / '04_redes_nulas/fondo/N2c'}
IMP, PRI = RES / '06_implicaciones', RES / '09_prioritizacion'
DST = Path(sys.argv[sys.argv.index('--dst') + 1]) if '--dst' in sys.argv else PRINCIPAL
C = CIFRAS['continental']
ECO_ES = {'BTA': 'Bosque tropical amazónico', 'SPA': 'Sabana de palmeras', 'SAL': 'Selva alta (yungas)', 'PAR': 'Páramo',
          'PUN': 'Puna', 'SES': 'Serranía esteparia', 'BSE': 'Bosque seco ecuatorial', 'BTP': 'Bosque tropical del Pacífico',
          'DCP': 'Desierto costero del Pacífico'}
ECO7 = ['BTA', 'SAL', 'PAR', 'PUN', 'SES', 'BSE', 'DCP']   # evaluable ecoregions (< 50 % protected)
PEOR, MEJOR, ND = '#a50f15', '#2c7fb8', '#9a9993'
W2 = 7.2                                                    # double-column width (in)
# One meaning per colour in the four figures (the authors' species-map palette): green = protected, blue = protected
# analogue (d_min <= 0.2, darker = more similar), magenta = climatic gap (d_min > 0.2, darker = farther)
# (VERDE, VERDE_CLARO, ANALOGO and VACIO are in figuras_comun.py)
CLASES_ART = list(zip(ANALOGO + VACIO, ('nearly identical', 'very similar', 'similar', 'moderate', 'weak',
                                        'low similarity', 'not analogous')))


def _ejes(ax):
    ax.spines[['top', 'right']].set_visible(False); ax.tick_params(labelsize=6.6, length=2)


def _titulo(ax, t):
    ax.set_title(t, loc='left', fontsize=7.6, weight='bold')


def _miles(ax, eje='x'):
    f = matplotlib.ticker.FuncFormatter(lambda v, _: en(v))
    (ax.xaxis if eje == 'x' else ax.yaxis).set_major_formatter(f)


def _prueba(dominio):
    t = pd.read_csv(NUL[dominio] / 'p_mediana.csv')
    return t.set_index('estrato')


def _eco_es(e):  # names used in p_mediana.csv
    return ECO_ES[e]


# ======================================================================== Fig. 1
def _mapa_d(ax, base, d, gris_tierra=False):
    """d_min in the article palette; protected pixels (footprint, d_min = 0) in green."""
    if gris_tierra:
        fondo_paises(ax)
    ground = ListedColormap([SIN_DATO, '#ffffff']); ground.set_bad((0, 0, 0, 0))
    ax.imshow(base.domain, cmap=ground, vmin=0, vmax=1, extent=base.extent, interpolation='nearest', zorder=2)
    k = np.digitize(d, CORTES_D[1:], right=True).astype(np.float32); k[d == 0] = 7   # 0–6 classes, 7 protected
    cmap = ListedColormap([*ANALOGO, *VACIO, VERDE]); cmap.set_bad((0, 0, 0, 0))
    ax.imshow(base.raster(k), cmap=cmap, vmin=-.5, vmax=7.5, extent=base.extent, interpolation='nearest', zorder=3)
    contorno(ax, base.outline, color='#6b6b66', lw=.4, zorder=4)
    ax.set_aspect(1 / np.cos(np.deg2rad(-9.2))); ax.set_axis_off()


def fig1():
    base = lienzo(TIERRA)
    d = base.leer(TIERRA / 'mapas_distancia/cond100/sistema_distancia_minima.tif')
    fig = plt.figure(figsize=(W2, 6.3))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.45, .66, .66], height_ratios=[1, .78], wspace=.06, hspace=.16)
    a = fig.add_subplot(gs[:, 0])
    _mapa_d(a, base, d)
    h = [Patch(facecolor=VERDE, label='Protected area')]
    h += [Patch(facecolor=c, edgecolor='#bdbdb5', lw=.4, label=f'{corto(lo)}–{corto(hi)} · {t}') for (c, t), lo, hi
          in zip(CLASES_ART, CORTES_D[:-1], CORTES_D[1:])]
    h.append(Patch(facecolor=VACIO[-1], edgecolor='#bdbdb5', lw=.4, label='> 1 · not analogous'))
    a.legend(handles=h, title='Distance to the most similar\nprotected climate, $d_{\\mathrm{min}}$', title_fontsize=6.2, fontsize=5.9,
             loc='lower left', bbox_to_anchor=(-.04, -.01), frameon=False, alignment='left', handlelength=1.5, labelspacing=.3)
    _titulo(a, '(a) Continental domain')
    a.text(.98, .985, f"Climatic gaps ($d_{{\\mathrm{{min}}}}$ > 0.2):\n{en(CIFRAS['implicaciones']['P1']['vacios_km2'])} km², "
           f"{100 - C['peru']['pct_d0.2']:.1f}% of the\nunprotected territory", transform=a.transAxes, ha='right', va='top',
           fontsize=6.4, color='#1f1f1d')
    for k, (capa, tit) in enumerate((('superficie', '(b) Sea surface'), ('fondo', '(c) Seafloor'))):
        ax = fig.add_subplot(gs[0, 1 + k])
        bm = lienzo(MAR / capa)
        dm = bm.leer(MAR / capa / 'mapas_distancia/cond100/sistema_distancia_minima.tif')
        _mapa_d(ax, bm, dm, gris_tierra=True)
        encuadre_mar(ax); ax.set_xlim(-85.2, -74.5)
        m = CIFRAS['maritimo'][capa]['todo']
        ax.text(.04, .02, f"$d_{{\\mathrm{{min}}}}$ ≤ 0.2:\n{m['pct_d0.2']:.1f}% of the\nunprotected sea", transform=ax.transAxes,
                fontsize=5.9, va='bottom', color='#1f1f1d')
        _titulo(ax, tit)
    # (d) unprotected surface of each ecoregion by class of dmin (100 % bars), gap area at the right
    ax = fig.add_subplot(gs[1, 1:])
    cl = C['clases']
    clases = [('similar', '≤ 0.2', ANALOGO[1])] + list(zip(('moderado', 'debil', 'poco_similar', 'no_analogo'),
                                                          ('0.2–0.3', '0.3–0.5', '0.5–1', '> 1'), VACIO))
    filas = ['peru'] + ECO7
    for i, g in enumerate(filas):
        tot = C['peru']['km2_no_protegido'] if g == 'peru' else C['ecorregiones'][g]['km2_no_protegido']
        izq = 0
        for clave, lab, col in clases:
            v = cl[clave]['km2'] * (1 if g == 'peru' else cl[clave]['grupos'][g] / 100) / tot * 100
            ax.barh(i, v, left=izq, color=col, ec='#8a8986', lw=.35, height=.66, label=lab if i == 0 else None)
            izq += v
        gap = tot * (100 - (C['peru'] if g == 'peru' else C['ecorregiones'][g])['pct_d0.2']) / 100
        prot = (C['peru'] if g == 'peru' else C['ecorregiones'][g])['pct_protegido']
        ax.text(101.5, i, f'{en(gap / 1e3, 0 if gap > 9e3 else 1)}k km² · {prot:.0f}%', va='center', fontsize=5.8, color='#1f1f1d')
    ax.set_yticks(range(len(filas)))
    ax.set_yticklabels(['Peru (continental)'] + [ECO_EN[g].replace('High jungle (Yungas)', 'Yungas') for g in ECO7], fontsize=6.2)
    ax.invert_yaxis(); ax.set_xlim(0, 100); _ejes(ax)
    ax.set_xlabel('Share of the unprotected surface (%)', fontsize=6.6)
    ax.text(101.5, -1.0, 'Gaps · protected', fontsize=5.8, color='#55554f', va='center')
    ax.legend(title='$d_{\\mathrm{min}}$ class', title_fontsize=6, fontsize=5.8, ncol=5, frameon=False, loc='upper left',
              bbox_to_anchor=(0, -.2), handlelength=1.2, columnspacing=.8)
    ax.set_title('(d) Climatic representation by ecoregion', loc='left', fontsize=7.6, weight='bold', pad=17)  # above 'Gaps · protected'
    p = ax.get_position(); ax.set_position([p.x0 + .16, p.y0, p.width - .19, p.height])   # names clear of the map
    fig.savefig(DST / 'Fig1_climatic_gaps.png', bbox_inches='tight', dpi=600); plt.close(fig)


# ======================================================================== Fig. 2
def _hist(ax, dominio, tit):
    z = np.load(NUL[dominio] / 'redes_aleatorias.npz')
    w = z['peso']; nul = z['azar_dmin'].astype(np.float64) @ w / w.sum(); real = float(z['real_dmin'].astype(np.float64) @ w / w.sum())
    t = _prueba(dominio).iloc[0]
    assert abs(t.D_real - real) < 1e-6
    col = PEOR if t.direccion == 'peor' else ND
    ax.hist(nul, bins=40, color='#cfcec8', ec='#a8a7a1', lw=.3)
    ax.axvline(real, color=col, lw=1.8)
    ax.axvline(np.median(nul), color=MEDIANA, lw=.9, ls='--')
    lab = {'peor': 'worse than chance', 'no distinguible': 'not distinguishable'}[t.direccion]
    ax.text(.97, .96, f"Real network: {real:.3f}\nNull median: {t.D_azar_mediana:.3f}\np = {t.p_bilateral:.3f} · SES = {t.ses:+.1f}\n{lab}",
            transform=ax.transAxes, ha='right', va='top', fontsize=5.9, color=col if col == PEOR else '#1f1f1d',
            bbox=dict(fc='white', ec='none', alpha=.85, pad=1.5), zorder=5)
    ax.set_xlabel('Mean distance to the most similar\nprotected climate, $\\bar{D}$', fontsize=6.4)
    ax.set_yticks([]); ax.spines[['top', 'right', 'left']].set_visible(False); ax.tick_params(labelsize=6.2, length=2)
    lo, hi = min(nul.min(), real), max(nul.max(), real); pad = (hi - lo) * .15
    ax.set_xlim(lo - pad, hi + pad * 2.2)
    _titulo(ax, tit)


def fig2():
    fig = plt.figure(figsize=(W2, 5.6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[.8, 1.25], hspace=.62, wspace=.18)
    for k, (dom, tit) in enumerate((('continental', '(a) Continental domain'), ('superficie', '(b) Sea surface'),
                                    ('fondo', '(c) Seafloor'))):
        _hist(fig.add_subplot(gs[0, k]), dom, tit)
    ax = fig.add_subplot(gs[1, :])
    p = ax.get_position(); ax.set_position([p.x0, p.y0, p.width * .76, p.height])   # room for the two columns
    tc, ts, tf = _prueba('continental'), _prueba('superficie'), _prueba('fondo')
    filas = [('Peru, continental', tc.loc['Perú'], C['peru']['pct_protegido'], True)]
    filas += [(ECO_EN[e].replace('High jungle (Yungas)', 'Yungas'), tc.loc[_eco_es(e)], C['ecorregiones'][e]['pct_protegido'], False) for e in ECO7]
    filas += [('Sea surface', ts.iloc[0], None, True), (ECO_EN['MFH'], ts.iloc[1], None, False), (ECO_EN['MTR'], ts.iloc[2], None, False),
              ('Seafloor', tf.iloc[0], None, True), ('Shelf (0–200 m)', tf.iloc[1], None, False),
              ('Slope (200–1 000 m)', tf.iloc[2], None, False), ('Deep floor (> 1 000 m)', tf.iloc[3], None, False)]
    y = np.arange(len(filas))[::-1].astype(float)
    y[8:] -= .6; y[11:] -= .6   # gaps between domains
    # magnitude (clima_local/magnitud_nulas.py): surface left without a protected analogue beyond the median null network,
    # (F(0.2) of the median null - F(0.2) of the real network) x surface of the stratum, two significant figures
    mg = json.loads((RES / '04_redes_nulas/magnitud_N2c.json').read_text(encoding='utf-8'))
    claves = [('continental', 'Perú'), *[('continental', e) for e in ECO7], ('superficie', 'todo'), ('superficie', 'MFH'),
              ('superficie', 'MTR'), ('fondo', 'todo'), ('fondo', 'PLATAFORMA'), ('fondo', 'TALUD'), ('fondo', 'PROFUNDO')]
    extra = [mg[dm][k]['km2_extra_mas_alla_0.2'] for dm, k in claves]
    x1, x2 = 1.03, 1.25   # columns right of the axes (axes fraction)
    col_t = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.axvline(0, color='#52514e', lw=.6); ax.axvspan(-2, 2, color='#f1f0ec', lw=0, zorder=0)
    for yi, (lab, r, prot, nac), ex in zip(y, filas, extra):
        ax.text(x2, yi, en(float(f'{ex:.2g}')), transform=col_t, ha='right', va='center', fontsize=6,
                color=PEOR if r.direccion == 'peor' else '#9a9993', weight='bold' if nac else 'normal')
        col = PEOR if r.direccion == 'peor' else MEJOR if r.direccion == 'mejor' else ND
        ax.plot([0, r.ses], [yi, yi], color=col, lw=.8, zorder=1)
        ax.scatter(r.ses, yi, s=46 if nac else 26, marker='D' if nac else 'o', zorder=3,
                   fc=col if r.direccion != 'no distinguible' else 'white', ec=col, lw=1.1)
        ax.text(-3.3, yi, lab, ha='right', va='center', fontsize=6.4, weight='bold' if nac else 'normal')
        if prot is not None:
            ax.text(x1, yi, f'{prot:.1f}%', transform=col_t, ha='right', va='center', fontsize=6, color='#55554f')
        ax.text(r.ses + (.35 if r.ses >= 0 else -.35), yi + .02, ("p" if nac else "p$_{\\mathrm{Holm}}$") + f" = {r.p_holm:.3f}", fontsize=5.3,
                ha='left' if r.ses >= 0 else 'right', va='center', color='#55554f')
    ax.text(x1, y[0] + 1.0, 'Protected', transform=col_t, ha='right', fontsize=6, color='#55554f')
    ax.text(x2, y[0] + 1.0, 'Extra km² without a\nprotected analogue', transform=col_t, ha='right', fontsize=6,
            color='#55554f', linespacing=1.05)
    ax.set_xlim(-3.2, 10.8); ax.set_ylim(y.min() - .8, y.max() + 1.4); ax.set_yticks([])
    ax.spines[['top', 'right', 'left']].set_visible(False); ax.tick_params(labelsize=6.4, length=2)
    ax.set_xlabel('Standardized effect size (SES) of $\\bar{D}$ against 999 null networks  ·  positive = worse than chance', fontsize=6.6)
    ax.legend(handles=[Line2D([], [], marker='o', ls='', mfc=PEOR, mec=PEOR, label='Worse than chance (p < 0.05; Holm for strata)'),
                       Line2D([], [], marker='o', ls='', mfc='white', mec=ND, label='Not distinguishable')],
              fontsize=6, frameon=False, loc='lower right', bbox_to_anchor=(1, 0))
    _titulo(ax, '(d) Departure from the null networks, by stratum')
    fig.savefig(DST / 'Fig2_null_networks.png', bbox_inches='tight', dpi=600); plt.close(fig)


# ======================================================================== Fig. 3
def fig3():
    fig = plt.figure(figsize=(W2, 4.3))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.15], wspace=.62, hspace=.95)
    # (a) effort curve
    b = fig.add_subplot(gs[:, 0])
    cu = pd.read_csv(IMP / 'P1_curva_esfuerzo.csv')
    P = CIFRAS['implicaciones']['P1_curva']
    # the same greedy sequence read with other cut-offs x (P1_umbral_curva.csv, Table S24): F(x) and its 90% point
    uc = pd.read_csv(IMP / 'P1_umbral_curva.csv')
    um = {r['umbral']: r['km2_90'] for r in json.loads((IMP / 'P1_umbral_esfuerzo.json').read_text(encoding='utf-8'))['filas']}
    # blue (share with a protected analogue), one lightness and dash per cut-off; 0.2, the cut-off of the paper, in black
    for x_, col, lw, ls in ((.15, '#1d3f66', 1.1, (0, (5, 1.6))), (.2, '#1f1f1d', 1.5, '-'), (.25, '#3f7fc4', 1.1, (0, (5, 1.4, 1, 1.4))),
                            (.3, '#86b0e0', 1.3, (0, (1.2, 1.2)))):
        f = uc[f'F{x_}'] * 100
        k90 = um[x_]
        b.plot(uc.agregado_km2 / 1e3, f, color=col, lw=lw, ls=ls, label=f'x = {x_:g} · 90% at +{en(k90)} km²', zorder=2 + (x_ == .2))
        b.plot(k90 / 1e3, 90, 'o', ms=3.6, mfc='white', mec=col, mew=1, zorder=4)
    assert np.allclose(uc['F0.2'][:len(cu)], cu.F02)   # the same greedy sequence as P1_curva_esfuerzo.csv, continued
    b.axhline(90, color='#bdbcb6', lw=.5, ls=':', zorder=1)
    for k, lab in (('alcanza_mediana_nula', 'Null median $\\bar{D}$'), ('alcanza_F02_80', '80% of Peru with a\nprotected analogue')):
        x = P[k]['agregado_km2']; yv = np.interp(x, cu.agregado_km2, cu.F02) * 100
        b.plot(x / 1e3, yv, 'o', ms=3.6, color='#1f1f1d', zorder=4)
        # labels in the empty lower right, joined to their points, so no text crosses the curves of the other cut-offs
        b.annotate(f'{lab}\n+{en(x)} km²', (x / 1e3, yv), xytext=(58, 77 if k == 'alcanza_F02_80' else 72), textcoords='data',
                   fontsize=5.6, ha='left', va='top', arrowprops=dict(arrowstyle='-', color='#8a8986', lw=.4, shrinkA=0, shrinkB=3))
    b.legend(title='Cut-off x of a protected analogue', title_fontsize=5.6, fontsize=5.5, frameon=False, loc='lower right',
             handlelength=2.6, labelspacing=.35, alignment='left')
    b.set_xlim(0, 112); b.set_ylim(58, 93); _ejes(b)
    b.set_xlabel('Natural surface added (thousand km²)', fontsize=6.4)
    b.set_ylabel('Territory at $d_{\\mathrm{min}}$ ≤ x, with a protected analogue (%)', fontsize=6.4)
    tot, prot = P['continental_km2'], CIFRAS['implicaciones']['P1_umbral']['protegido_exacto_km2']  # exact area of the PAs, not their footprint
    top = b.secondary_xaxis('top', functions=(lambda x: (prot + x * 1e3) / tot * 100, lambda p: (p * tot / 100 - prot) / 1e3))
    top.set_xlabel('Protected share of the territory (%)', fontsize=6); top.tick_params(labelsize=6, length=2)
    falta = P['meta3_30pct_km2'] - prot
    # 30% lies far right of the axis; a note under the top axis names it instead of squeezing the curve
    b.annotate(f'30% protected (Target 3)\nat +{en(falta)} km² →', xy=(1, 1), xycoords='axes fraction', xytext=(-2, -4),
               textcoords='offset points', ha='right', va='top', fontsize=5.7, color='#55554f')
    _titulo(b, '(a) Effort curve')
    # (b) role of ACR and ACP: surface at d <= 0.2 that only regional and private areas provide
    c = fig.add_subplot(gs[0, 1])
    m = pd.read_csv(IMP / 'P2_modalidades_por_ecorregion.csv').set_index('nombre')
    eco = ECO7
    for i, e in enumerate(eco):
        r = m.loc[ECO_ES[e]]
        anp = r.cubierta_solo_ANP_km2 / r.cubierta_red_km2 * 100
        c.barh(i, anp, color=VERDE, ec='#8a8986', lw=.3, height=.62, label='National (ANP)' if i == 0 else None)
        c.barh(i, 100 - anp, left=anp, color=VERDE_CLARO, ec='#8a8986', lw=.3, height=.62,
               label='Only regional or private (ACR, ACP)' if i == 0 else None)
        c.text(101, i, f'{100 - anp:.0f}%', va='center', fontsize=5.8)
    c.set_yticks(range(len(eco))); c.set_yticklabels([ECO_EN[e].replace('High jungle (Yungas)', 'Yungas') for e in eco], fontsize=6)
    c.invert_yaxis(); c.set_xlim(0, 100); _ejes(c)
    c.set_xlabel('Share of the covered surface, $d_{\\mathrm{min}}$ ≤ 0.2 (%)', fontsize=6.4)
    c.legend(fontsize=5.6, frameon=False, ncol=2, loc='upper left', bbox_to_anchor=(-.02, -.32), handlelength=1.2)
    _titulo(c, '(b) Who provides the protected analogue')
    # (c) leave one out at sea
    dd = fig.add_subplot(gs[1, 1])
    t = pd.read_csv(IMP / 'P3_dejando_una_fuera.csv')
    quien = [('(ninguna)', 'Full network'), ('NAC_RN16', 'without Nazca Ridge'), ('NAC_RN13', 'without Guano Islands'),
             ('NAC_RN03', 'without Paracas'), ('NAC_RN18', 'without Mar Tropical de Grau')]
    for j, (capa, col) in enumerate((('superficie', ANALOGO[1]), ('fondo', ANALOGO[0]))):   # blue: share with an analogue
        vals = [t[(t.capa == capa) & (t.fuera == q)].F02_todo.iloc[0] * 100 for q, _ in quien]
        dd.barh(np.arange(len(quien)) + (j - .5) * .36, vals, height=.34, color=col, label={'superficie': 'Sea surface', 'fondo': 'Seafloor'}[capa])
        for i, vv in enumerate(vals):
            dd.text(vv + 1, i + (j - .5) * .36, f'{vv:.1f}', va='center', fontsize=5.4)
    dd.set_yticks(range(len(quien))); dd.set_yticklabels([l for _, l in quien], fontsize=6); dd.invert_yaxis()
    dd.set_xlim(0, 92); dd.set_xticks(range(0, 81, 20)); _ejes(dd); dd.set_xlabel('Whole sea at $d_{\\mathrm{min}}$ ≤ 0.2, protected surface included (%)', fontsize=6.4)
    dd.legend(fontsize=5.6, frameon=False, ncol=2, loc='upper left', bbox_to_anchor=(-.02, -.32), handlelength=1.2)   # as in (b)
    _titulo(dd, '(c) Maritime representation, leaving one area out')
    fig.savefig(DST / 'Fig3_how_much_and_who.png', bbox_inches='tight', dpi=600); plt.close(fig)


# ======================================================================== Fig. 4 and its table
N_AZAR = 999
RND = f'Random natural microbasins (median of {N_AZAR})'
# plans: no green (protected), blue-grey or magenta (analogue, gaps) of the maps
ESC = {'B': ('Species only', '#8c510a'), 'C17': ('Species + climate', '#e66101'), 'D17': ('Climate only', '#2c7fb8')}
GAP_NAT, GAP_TRANS, PROT = '#efcfe1', '#d3d1ca', '#cfe6d8'
ENGROSAR = 2   # chosen microbasins drawn with a margin of 2 pixels (≈2 km) so that small units stay visible in print
VARIANTES = [('Species, gaps only (A)', 'D', True), ('Species only, flat 17% targets', 's', True),
             ('Species only, flat 30% targets', '^', False)]
AZAR_SEED = 20261103   # seed of clima_local/azar_prioritizacion.py (prioritizacion.SEED + 2)
SENS = PRI /('sensibilidad_metas.csv' if (PRI / 'sensibilidad_metas.csv').exists() else 'sensibilidad_M4.csv')  # sensibilidad_especies.py


def _logro(via, elegidas):
    """Mean share of the full Rodrigues target reached by the 271 species (each capped at 1) and full targets met."""
    F = pd.read_csv(PRI / f'entrada/features_{via}.csv'); R = pd.read_csv(PRI / f'entrada/rij_{via}.csv')
    e = F[F.tipo == 'especie']
    lo = R[R.pu.isin([0, *elegidas])].groupby('species').amount.sum()
    meta = e.meta_especie * e.total_km2; x = e.id.map(lo).fillna(0)
    return float(np.minimum(x / meta, 1).mean()), int((x >= meta * .999).sum())


def _azar_mediana(cel, km2):
    """Replays the N_AZAR random draws of azar_prioritizacion.py (19k then 41k, same seed and pool) and returns the cells of the
    41k set at the median climatic outcome; the replay is checked against the stored area of that set."""
    ev = pd.read_csv(PRI / 'evaluacion.csv').set_index('escenario')
    az = pd.read_csv(PRI / f'azar_{N_AZAR}.csv'); g = az[az.presupuesto == '41k'].set_index('conjunto')
    pu, cand = cel['pu'], cel['candidato']
    por = pd.Series(km2[cand]).groupby(pu[cand]).sum(); por = por[por.index > 0]
    ids, sup_id = por.index.to_numpy(), por.to_numpy()
    rng = np.random.default_rng(AZAR_SEED); sets = {}
    for k in ('19k', '41k'):
        sup = float(ev.loc[f'B_{k}', 'superficie_km2'])
        for i in range(N_AZAR):
            o = rng.permutation(len(ids)); sets[k, i] = ids[o[:np.searchsorted(sup_id[o].cumsum(), sup) + 1]]
    i = int((g.clima_optimo_pct - g.clima_optimo_pct.median()).abs().idxmin())
    m = cand & np.isin(pu, sets['41k', i])
    assert abs(km2[m].sum() - g.loc[i, 'superficie_km2']) < 1e-6
    return m


def tabla_prioritizacion():
    """Table S of Fig. 4: climate and species outcomes of each set at equal area. Random sets: the N_AZAR draws of whole
    natural microbasins up to the area of the species-only solution (B) stored by clima_local/azar_prioritizacion.py."""
    ev = pd.read_csv(PRI / 'evaluacion.csv').set_index('escenario')
    pu = pd.read_csv(PRI / 'entrada/pu_todo.csv'); pu = pu[pu.id > 0]
    cu = pd.read_csv(IMP / 'P1_curva_esfuerzo.csv').dropna(subset=['unidad'])
    D0 = ev.loc['red real', 'D']
    az = pd.read_csv(PRI / f'azar_{N_AZAR}.csv')   # clima_local/azar_prioritizacion.py
    filas = []
    for b, k in ((19000, '19k'), (41347, '41k')):
        Dc = float(np.interp(ev.loc[f'B_{k}', 'superficie_km2'], [0, *cu.agregado_km2], [D0, *cu.D]))
        cierre = lambda D: 100 * (D0 - D) / (D0 - Dc)  # noqa: E731
        hy = cu[cu.agregado_km2 <= b * 1.0001].unidad.astype('int64')
        gr = pu[pu.hybas_id.isin(hy)].id
        lg, mg = _logro('todo', gr)
        filas.append({'budget_km2': b, 'set': 'Greedy climate curve (p-median)', 'D': Dc,
                      'F02': float(np.interp(b, cu.agregado_km2, cu.F02)), 'share_closed': 100., 'species_mean_target': lg,
                      'species_full_targets': mg})
        g = az[az.presupuesto == k]   # climate and species measured on the same random sets
        filas.append({'budget_km2': b, 'set': RND, 'D': g.D.median(), 'F02': g.F02.median(),
                      'share_closed': g.clima_optimo_pct.median(), 'species_mean_target': g.especies_meta_media_pct.median() / 100,
                      'species_full_targets': int(g.especies_meta_completa.median()),
                      'share_closed_p2_5': g.clima_optimo_pct.quantile(.025), 'share_closed_p97_5': g.clima_optimo_pct.quantile(.975),
                      'species_p2_5': g.especies_meta_media_pct.quantile(.025) / 100,
                      'species_p97_5': g.especies_meta_media_pct.quantile(.975) / 100})
        for esc in ('A', 'B', 'C17', 'D17'):
            via = 'vacios' if esc == 'A' else 'todo'
            s = pd.read_csv(PRI / f'solucion_{esc}_{k}.csv'); el = s.loc[(s.solution_1 == 1) & (s.id > 0), 'id']
            lg, mg = _logro(via, el)
            r = ev.loc[f'{esc}_{k}']
            filas.append({'budget_km2': b, 'set': {'A': 'Species, gaps only (A)', 'B': 'Species only (B)',
                                                   'C17': 'Species + climate (C)', 'D17': 'Climate only (D)'}[esc],
                          'D': r.D, 'F02': r.F02, 'share_closed': cierre(r.D), 'species_mean_target': lg, 'species_full_targets': mg})
        # species only with flat targets (Table S12), scored on the range-size-scaled targets like every other set
        s4 = pd.read_csv(SENS); s4 = s4[s4.presupuesto == k].set_index('escenario')
        for esc, lab in (('B17f', 'Species only, flat 17% targets'), ('B30f', 'Species only, flat 30% targets')):
            r = s4.loc[esc]
            assert abs(cierre(r.D) - r.clima_D_pct_voraz) < .5, esc
            filas.append({'budget_km2': b, 'set': lab, 'D': r.D, 'F02': r.F02, 'share_closed': cierre(r.D),
                          'species_mean_target': r.especies_rodrigues_pct / 100})
    t = pd.DataFrame(filas)
    t.attrs['inicio'] = _logro('todo', [])[0]
    return t


def fig4():
    t = tabla_prioritizacion(); t.to_csv(DST / 'Fig4_prioritization_table.csv', index=False, float_format='%.6f')
    ini = t.attrs['inicio']
    fig = plt.figure(figsize=(W2 * 5.12 / 3.12, 7.4))   # four map columns at the size panels had with two, widened to the right
    gs = GridSpec(2, 5, figure=fig, width_ratios=[1.12, 1, 1, 1, 1], wspace=.06, hspace=.3)
    est = {'Greedy climate curve (p-median)': ('Greedy benchmark (p-median)', 'black', '*', 70),
           'Climate only (D)': ('Climate only (D)', ESC['D17'][1], 'o', 46),
           'Species + climate (C)': ('Species + climate (C)', ESC['C17'][1], 'o', 46),
           'Species only (B)': ('Species only (B)', ESC['B'][1], 'o', 46)}
    GRIS = '#9a9993'
    for k, (b, tit) in enumerate(((19000, '(a) 19 000 km² added'), (41347, '(b) 41 347 km² added'))):
        ax = fig.add_subplot(gs[k, 0])
        sub = t[t.budget_km2 == b].set_index('set')
        # random sets: a box spanning the 95% interval on both axes (they vary in climate and in species), median as a tick
        r = sub.loc[RND]; x0, x1 = 100 * r.species_p2_5, 100 * r.species_p97_5
        ax.add_patch(matplotlib.patches.Rectangle((x0, r.share_closed_p2_5), x1 - x0, r.share_closed_p97_5 - r.share_closed_p2_5,
                                                  facecolor=to_rgba(GRIS, .45), edgecolor=GRIS, lw=.6, zorder=2))
        ax.plot([x0, x1], [r.share_closed] * 2, color='#5f5e59', lw=.8, solid_capstyle='butt', zorder=3)
        ax.annotate(f'{r.share_closed:.0f}%', (x1, r.share_closed_p2_5), xytext=(2, -1), textcoords='offset points',
                    fontsize=5.6, color='#77766f', ha='left', va='top')
        for nombre, (lab, col, mk, sz) in est.items():
            r = sub.loc[nombre]
            ax.scatter(r.species_mean_target * 100, r.share_closed, s=sz, marker=mk, color=col, zorder=3, label=lab)
            ax.annotate(f'{r.share_closed:.0f}%', (r.species_mean_target * 100, r.share_closed), xytext=(5, 2),
                        textcoords='offset points', fontsize=5.6, color=col)
        for nombre, mk, izq in VARIANTES:   # the other species-only rules, open symbols; labels left or right of the symbol
            r = sub.loc[nombre]
            ax.scatter(r.species_mean_target * 100, r.share_closed, s=24, marker=mk, facecolor='white', edgecolor=ESC['B'][1],
                       lw=.9, zorder=3)
            ax.annotate(f'{r.share_closed:.0f}%', (r.species_mean_target * 100, r.share_closed), xytext=(-5 if izq else 5, 0),
                        textcoords='offset points', fontsize=5.4, color=ESC['B'][1], ha='right' if izq else 'left', va='center')
        ax.axvline(ini * 100, color='#bdbcb6', lw=.7, ls=':')
        ax.text(ini * 100 + .6, 4, 'current\nnetwork', fontsize=5.4, color='#77766f')
        ax.set_xlim(14, 64); ax.set_ylim(0, 108); _ejes(ax)
        ax.set_xlabel('Species: mean share of the representation target reached (%)', fontsize=6.2)
        ax.set_ylabel('Climate: share of the greedy\nbenchmark reduction of $\\bar{D}$ achieved (%)', fontsize=6.2)
        _titulo(ax, tit)
        if k == 1:   # legend under (b), two columns: the plans in the order of the maps, then the random sets
            pt = lambda lab, col, mk, sz: Line2D([], [], ls='', marker=mk, color=col, markersize=sz ** .5 * .9, label=lab)  # noqa: E731
            h = [pt(*v) for v in est.values()]
            h += [Line2D([], [], ls='', marker=mk, mfc='white', mec=ESC['B'][1], mew=.9, markersize=4.4,
                         label={'D': 'Species, gaps only (A)', 's': 'Species only, flat 17% targets',
                                '^': 'Species only, flat 30% targets'}[mk]) for _, mk, _ in VARIANTES]
            h.append(Patch(facecolor=to_rgba(GRIS, .45), edgecolor=GRIS, lw=.6, label='Random microbasins (95% interval)'))
            ax.legend(handles=h, ncol=2, fontsize=5.6, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.17),
                      handletextpad=.4, columnspacing=1.6, labelspacing=.45, handlelength=1.1, borderaxespad=0)
    # (c-j) where each set of the scatter plots puts the same 41 347 km², one map each
    base = lienzo(TIERRA)
    cel = np.load(PRI / 'entrada/celdas.npz'); pu = cel['pu']; cand = cel['candidato']
    with rasterio.open(IMP / 'P1_vacios_expandibles.tif') as ds:   # gap classes 1-4 on natural land, 5 on transformed land
        v = ds.read(1)
    fondo = np.where((v >= 1) & (v <= 4), 0., np.where(v == 5, 1., np.nan))
    tabla = pd.read_csv(PRI / 'entrada/pu_todo.csv'); tabla = tabla[tabla.id > 0]
    cu = pd.read_csv(IMP / 'P1_curva_esfuerzo.csv').dropna(subset=['unidad'])
    from scipy.ndimage import binary_dilation
    P = CIFRAS['implicaciones']['P1_curva']
    hy = cu[cu.agregado_km2 <= P['alcanza_F02_90']['agregado_km2'] * 1.0001].unidad.astype('int64')
    sel = {'greedy': cand & np.isin(pu, tabla[tabla.hybas_id.isin(hy)].id)}   # the 223 potential sites
    for esc in ('A', 'B', 'C17', 'D17', 'B17f', 'B30f'):
        s = pd.read_csv(PRI / f'solucion_{esc}_41k.csv'); sel[esc] = cand & np.isin(pu, s.loc[(s.solution_1 == 1) & (s.id > 0), 'id'])
    sel['A'] &= cel['vacio']   # scenario A counts only the gap land of each microbasin
    km2 = base.area / 1e6
    sel['random'] = _azar_mediana(cel, km2)
    union = shapely.union_all(list(base.shapes.values()))
    ground = ListedColormap([SIN_DATO, '#ffffff']); ground.set_bad((0, 0, 0, 0))
    gapcm = ListedColormap([GAP_NAT, GAP_TRANS]); gapcm.set_bad((0, 0, 0, 0))
    paneles = [('(c) Greedy benchmark (p-median)', 'greedy', '#1f1f1d'), ('(d) Species, gaps only (A)', 'A', ESC['B'][1]),
               *[(f'({l}) {ESC[e][0]} ({n})', e, ESC[e][1]) for l, e, n in (('e', 'B', 'B'), ('f', 'C17', 'C'), ('g', 'D17', 'D'))],
               ('(h) Species only, flat 17% targets', 'B17f', ESC['B'][1]), ('(i) Species only, flat 30% targets', 'B30f', ESC['B'][1]),
               ('(j) Random microbasins (median set)', 'random', '#5f5e59')]
    for i, (tit, e, col) in enumerate(paneles):
        ax = fig.add_subplot(gs[i // 4, 1 + i % 4])
        ax.imshow(base.domain, cmap=ground, vmin=0, vmax=1, extent=base.extent, interpolation='nearest', zorder=1)
        ax.imshow(fondo, cmap=gapcm, vmin=-.5, vmax=1.5, extent=base.extent, interpolation='nearest', zorder=2)
        relleno(ax, union, facecolor=PROT, edgecolor='none', zorder=3)
        img = binary_dilation(base.raster(sel[e].astype(np.float32), fill=0) > 0, iterations=ENGROSAR)
        ax.imshow(np.where(img, 0., np.nan), cmap=ListedColormap([col]), vmin=-.5, vmax=.5, extent=base.extent,
                  interpolation='nearest', zorder=4)
        contorno(ax, base.outline, color='#6b6b66', lw=.4, zorder=5)
        ax.set_aspect(1 / np.cos(np.deg2rad(-9.2))); ax.set_axis_off()
        enVacio = sel[e] & cel['vacio']   # share measured on the units themselves, not on their drawn margin
        ax.text(.02, .03, f'{km2[enVacio].sum() / km2[sel[e]].sum() * 100:.0f}% of its area\nin climatic gaps',
                transform=ax.transAxes, ha='left', va='bottom', fontsize=5.6, color='#55554f')
        _titulo(ax, tit)
    h = [Patch(fc='#1f1f1d', label='Microbasins chosen by each plan'),
         Patch(fc=GAP_NAT, ec='#bdbdb5', lw=.4, label='Climatic gap on natural land'),
         Patch(fc=GAP_TRANS, ec='#bdbdb5', lw=.4, label='Climatic gap on transformed land'),
         Patch(fc=PROT, ec='#bdbdb5', lw=.4, label='Protected-area network')]
    fig.legend(handles=h, ncol=2, fontsize=5.6, frameon=False, loc='upper center', bbox_to_anchor=(.6, .06),
               handlelength=1.2, columnspacing=1.6, labelspacing=.35)
    fig.savefig(DST / 'Fig4_where_to_expand.png', bbox_inches='tight', dpi=600); plt.close(fig)


# ======================================================================== graphical abstract (1328 × 531 px)
# Map colours are the authors' palette of the species maps: green protected, blue protected analogue, magenta climatic gaps
GA = {'prot': '#23745B', 'analogo': '#507FB5', 'vacio': ['#ecc9e0', '#d394be', '#A64E82', '#6a1d52'],
      'tinta': '#17171a', 'gris': '#62615d', 'suave': '#9d9c96', 'pista': '#efeee9', 'fondo': '#ffffff',
      'puntos': '#b4b3ac', 'borde': '#3d3d3a'}
SERIF, SANS = 'Source Serif 4', 'Source Sans Pro'
DEM = RECURSOS / 'elevation/wc2.1_30s_elev.tif'   # WorldClim 2.1, 30 arc s


def _ga_mapa(ax):
    """Continental dmin in five classes, protected pixels in green, over a hillshade of WorldClim 2.1 elevation."""
    from matplotlib.colors import LightSource
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    base = lienzo(TIERRA)
    with rasterio.open(DEM) as ds:   # WorldClim 30" read onto the CHELSA grid, the same 1/120° lattice
        x0, x1, y0, y1 = base.extent
        z = ds.read(1, window=from_bounds(x0, y0, x1, y1, ds.transform), out_shape=(base.h, base.w),
                    resampling=Resampling.bilinear).astype(np.float32)
    z[z < -1000] = np.nan
    d = base.leer(TIERRA / 'mapas_distancia/cond100/sistema_distancia_minima.tif')
    k = np.digitize(d, [.2, .3, .5, 1.0], right=True); k[np.load(base.prep / 'superficie_union.npy') > 0] = 5
    cols = np.array([to_rgb(c) for c in (GA['analogo'], *GA['vacio'], GA['prot'])])
    rgb = np.ones((base.h * base.w, 3)); rgb[base.flat] = cols[k]; rgb = rgb.reshape(base.h, base.w, 3)
    hs = LightSource(azdeg=315, altdeg=40).hillshade(np.nan_to_num(z), vert_exag=1.2, dx=915, dy=927)
    i = np.clip(1 + 3.0 * (hs - np.nanmedian(hs[z < 300])), .5, 1.35)[..., None]   # 1 on flat lowland
    rgb = np.where(i < 1, rgb * i, rgb + (1 - rgb) * (i - 1) * .8)
    a = np.zeros(base.h * base.w); a[base.flat] = 1; a = a.reshape(base.h, base.w)
    from scipy.ndimage import gaussian_filter, shift
    sombra = gaussian_filter(shift(a, (14, 9), order=0), 9) * .38          # soft drop shadow, offset down and right
    ax.imshow(np.dstack([np.zeros((base.h, base.w, 3)), sombra]), extent=base.extent, interpolation='antialiased', zorder=1)
    ax.imshow(np.dstack([rgb, a]), extent=base.extent, interpolation='antialiased', zorder=2)
    contorno(ax, base.outline, color=GA['borde'], lw=.6, zorder=3)
    ax.set_aspect(1 / np.cos(np.deg2rad(-9.2))); ax.set_axis_off()
    return base.outline


def _ga_puntos(ax, nul, w_in, h_in):
    """Dot plot, one dot per null network stacked in bins; the bin width is the largest whose tallest stack fits."""
    x0, x1 = ax.get_xlim()
    for bw in np.linspace(.0012, .0002, 41):
        n = np.floor((nul - x0) / bw).astype(int)
        if np.bincount(n).max() * bw / (x1 - x0) * w_in <= h_in:
            break
    diam = bw / (x1 - x0) * w_in * 72                     # dot diameter in points; one y unit = one dot
    ax.set_ylim(0, h_in * 72 / diam)
    fila = np.zeros(n.max() + 1, int); xs, ys = [], []
    for j in n:
        xs.append(x0 + (j + .5) * bw); ys.append(fila[j] + .5); fila[j] += 1
    ax.scatter(xs, ys, s=(diam * .92) ** 2, color=GA['puntos'], lw=0, zorder=2)
    return fila.max()


def ga():
    plt.rcParams.update({'mathtext.fontset': 'custom', 'mathtext.rm': SANS, 'mathtext.it': f'{SANS}:italic'})
    W, H = 13.28, 5.31                                      # 1328 x 531 px at 100 dpi; saved at 300 dpi (3984 x 1593 px)
    fig = plt.figure(figsize=(W, H), facecolor=GA['fondo'])
    tx = lambda x, y, s, **k: fig.text(x, y, s, **{'family': SANS, 'color': GA['tinta'], **k})  # noqa: E731
    caja = lambda x, y, w, h, c: fig.patches.append(plt.Rectangle((x, y), w, h, color=c, lw=0, transform=fig.transFigure))  # noqa: E731
    P = CIFRAS['implicaciones']['P1_curva']; C0 = CIFRAS['continental']['peru']
    t, ts = _prueba('continental').loc['Perú'], _prueba('superficie').iloc[0]
    s = pd.read_csv(DST / 'Fig4_prioritization_table.csv'); s = s[s.budget_km2 == 41347].set_index('set')

    def titulo(x, n, h, sub=None):
        tx(x, .925, n, size=31, weight='bold', family=SERIF, color=GA['vacio'][2], va='baseline')
        tx(x + .025, .925, h, size=22, weight='bold', family=SERIF, va='baseline')
        if sub:
            tx(x + .025, .875, sub, size=14, color=GA['gris'], va='top', linespacing=1.12)

    # ---- 1  where: the map, with the number and legend in the empty Pacific corner (no text may touch the map)
    a = fig.add_axes([0, 0, .335, 1]); peru = _ga_mapa(a); a.set_xlim(-86.69, -68.35); a.set_ylim(-19, 2.6)
    titulo(.012, '1', 'Climatic gaps', 'No similar climate\nin any protected area')
    tx(.012, .262, f"{en(CIFRAS['implicaciones']['P1']['vacios_km2'])} km²", size=25, weight='bold', color=GA['vacio'][3])
    tx(.012, .242, f'{100 - C0["pct_d0.2"]:.0f}% of unprotected land,\nmostly arid and Andean', size=13.5, va='top', linespacing=1.12)
    for i, (c, lab) in enumerate(((GA['prot'], 'Protected area'), (GA['analogo'], 'Protected analogue'),
                                  (None, 'Climatic gap, near to far'))):
        y = .118 - i * .043
        if c:
            caja(.012, y - .013, .016, .027, c)
        else:
            for j, cj in enumerate(GA['vacio']):
                caja(.012 + j * .004, y - .013, .004, .027, cj)
        tx(.034, y, lab, size=13, va='center')

    # ---- 2  how good: the real network against 999 context-preserving null networks
    X2, W2_ = .37, .265
    titulo(X2, '2', 'Worse than chance', "Real network against 999 null networks\nthat keep each area's size, shape,\nnatural setting and ecoregion mix")
    z = np.load(NUL['continental'] / 'redes_aleatorias.npz'); w = z['peso']
    nul = z['azar_dmin'].astype(np.float64) @ w / w.sum(); real = float(z['real_dmin'].astype(np.float64) @ w / w.sum())
    b = fig.add_axes([X2, .25, W2_, .37]); b.set_xlim(.1235, .1565); b.set_facecolor('none')
    alto = _ga_puntos(b, nul, W2_ * W, .37 * H * .8)
    tope = alto * 1.06
    b.plot([real, real], [0, tope], color=GA['vacio'][2], lw=2.8, solid_capstyle='butt', zorder=3)
    b.scatter([real], [tope], s=170, color=GA['vacio'][2], zorder=4, lw=0)
    b.text(real, tope + 3, 'Real\nnetwork', ha='center', va='bottom', family=SANS, size=14.5, weight='bold', color=GA['vacio'][3],
           linespacing=1.0)
    b.text(np.median(nul) - .0003, alto + 2.5, '999 null\nnetworks', ha='center', va='bottom', family=SANS, size=13.5,
           color=GA['gris'], linespacing=1.0)
    b.spines[['top', 'right', 'left']].set_visible(False); b.spines['bottom'].set_color(GA['suave']); b.set_yticks([])
    b.set_xticks([.13, .14, .15]); b.tick_params(labelsize=12, length=3, colors=GA['gris'])
    for lab in b.get_xticklabels():
        lab.set_family(SANS)
    tx(X2, .165, 'Mean distance to the most similar protected climate', size=12.5, color=GA['gris'])
    tx(X2, .115, '← more representative', size=12.5, color=GA['suave'])
    tx(X2 + W2_, .115, 'less representative →', size=12.5, color=GA['suave'], ha='right')
    tx(X2, .035, f'Land p = {t.p_bilateral:.3f}; sea surface also worse (p = {ts.p_bilateral:.3f})', size=13, color=GA['tinta'])

    # ---- 3  what next: species-led and climate-led expansion at equal area
    X3, W3 = .68, .305
    titulo(X3, '3', 'Species alone fall short', f"Share of the greedy climate benchmark\nrecovered by adding "
           f"{en(P['alcanza_F02_90']['agregado_km2'])} km² of natural land")
    c = fig.add_axes([X3, .33, W3, .36]); c.set_xlim(0, 100); c.set_ylim(-.75, 1.75); c.set_axis_off()
    r = s.loc[RND, 'share_closed']
    for i, (lab, v, col) in enumerate((('Species targets only', s.loc['Species only (B)', 'share_closed'], ESC['B'][1]),
                                       ('Species + climatic targets', s.loc['Species + climate (C)', 'share_closed'], ESC['C17'][1]))):
        y = 1 - i * 1.15
        c.barh(y, 100, height=.46, color=GA['pista'], lw=0, zorder=1)
        c.barh(y, v, height=.46, color=col, lw=0, zorder=2)
        c.text(0, y + .33, lab, family=SANS, size=15, weight='bold', color=col, va='bottom')
        c.text(v + 1.8, y, f'{v:.0f}%', family=SANS, size=19, weight='bold', color=col, va='center', ha='left', zorder=3)
        c.plot([r, r], [y - .3, y + .3], color=GA['tinta'], lw=1.4, ls=(0, (2.2, 1.6)), zorder=4)
    c.text(r, -.15 - .38, f'Random sites {r:.0f}%', family=SANS, size=13, color=GA['tinta'], ha='center', va='top')
    caja(X3, .035, W3, .17, GA['prot'])
    tx(X3 + W3 / 2, .12, 'Threatened-species plans toward 30×30\nneed climatic targets', size=16, weight='bold', color='white',
       ha='center', va='center', linespacing=1.12)
    for x in (.338, .648):   # arrows between the three steps
        fig.patches.append(matplotlib.patches.FancyArrow(x, .47, .014, 0, width=.0028, head_width=.022, head_length=.007,
                                                          length_includes_head=True, color=GA['suave'], transform=fig.transFigure))
    fig.canvas.draw()   # the authors' rule: no text touches the map
    inv = a.transData.inverted()
    textos = [t_ for t_ in fig.texts + b.texts + c.texts if t_.get_text().strip()]
    cajas = [t_.get_window_extent() for t_ in textos]
    for t_, bb in zip(textos, cajas):
        (x0, y0), (x1, y1) = inv.transform(bb.get_points())
        assert not shapely.box(x0, y0, x1, y1).intersects(peru.buffer(.12)), t_.get_text()
    for i in range(len(cajas)):   # nor another text
        for j in range(i):
            assert not cajas[i].overlaps(cajas[j]), (textos[i].get_text(), textos[j].get_text())
    fig.savefig(DST / 'Graphical_abstract.png', dpi=300, facecolor=GA['fondo']); plt.close(fig)


if __name__ == '__main__':
    DST.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:sys.argv.index('--dst')] + sys.argv[sys.argv.index('--dst') + 2:] if '--dst' in sys.argv else sys.argv[1:]
    que = args or ['fig1', 'fig2', 'fig3', 'fig4', 'ga']
    for q in que:
        globals()[q](); print('ok', q, flush=True)
