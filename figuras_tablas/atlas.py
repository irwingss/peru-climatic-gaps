"""File S2, Atlas of Climate Analogues: the analogue map of every protected area (258 continental, and the six maritime
areas for the sea surface and the seafloor) and the Word file that gathers them with Table S2.1 (insumos_figuras/tabla_s1.json,
written by tabla_s1_datos.py). Maps go to figuras/atlas/ (outside git; about 1 GB with their GeoTIFFs).
Run: python atlas.py [png] [docx]   (both without arguments)
"""
import json, sys
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.shared import Cm, Pt
from figuras_comun import *  # noqa: F401,F403  (shared constants, canvases and map helpers)

DOCX = FIG / 'File_S2_Atlas_climate_analogues.docx'


def nota_tierra():
    a = CIFRAS['continental']['escala']
    return (f"Reading references for d (medians): neighbouring pixels (≈ {a['curva'][0]['km']:.1f} km) {a['curva'][0]['d']:.2f}; "
            f"pixels ≈ {a['km_d0.2']:.0f} km apart 0.2; random pairs of pixels {a['azar']:.2f}.\n"
            'CHELSA V2.1 · 1981–2010 · 30″ · 11 predictors · regularized Mahalanobis distance (κ ≤ 100). Grey: no climate data.')

AVISO = {'NAC_RN03': 'its marine part is mapped in the maritime atlas', 'NAC_RN13': 'its marine part is mapped in the maritime atlas',
         'NAC_RN15': 'its marine part is mapped in the maritime atlas', 'NAC_RN18': 'its marine part is mapped in the maritime atlas',
         'NAC_RN02': 'the lake is outside both analyses', 'NAC_RN05': 'the lake is outside both analyses',
         'REG_ACR03': 'the lagoon water is outside both analyses'}


def mapa_atlas(ax, base, u, d, geometry, km_01, km_02, linea2, nota, aviso=None, tierra=True):
    panel_d(ax, base, d, geometry, gris_tierra=not tierra)
    km2 = u['superficie_clima_m2'] / 1e6
    ax.set_title(f"{nombre_en(u)} ({u['anp_id']})", loc='left', fontsize=11, color='#1f1f1d', pad=34)
    ax.text(0, 1.005, f"{MOD_EN[u['modalidad']]} · {en(km2, 0 if km2 >= 10 else 2)} km² with data · {en(u['n_celdas'])} pixels · "
            f"area outside the protected area: {en(km_01)} km² with d ≤ 0.1, {en(km_02)} km² with d ≤ 0.2\n{linea2}",
            transform=ax.transAxes, fontsize=7, color='#55554f', va='bottom', ha='left')
    if aviso:
        ax.text(.02, .03, aviso, transform=ax.transAxes, fontsize=7.5, color='#1f1f1d', va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=.5', fc='white', ec=VERDE, lw=.8), zorder=8)
    ax.legend(handles=leyenda_d([Patch(facecolor=VERDE, edgecolor=VERDE_OSCURO, lw=.6, label='Protected area')]),
              title='Mahalanobis distance d', title_fontsize=7.5, fontsize=7, loc='upper left',
              bbox_to_anchor=(1.0, .62), frameon=False, alignment='left')
    ax.text(0, -.005, nota, transform=ax.transAxes, fontsize=6, color='#55554f', va='top', ha='left')


def atlas_continental(only=None, dpi=150):
    base = lienzo(TIERRA)
    areas = {a['anp_id']: a for a in CIFRAS['continental']['areas']}
    dst = ATLAS / 'continental'; dst.mkdir(parents=True, exist_ok=True)
    for u in base.inventory:
        uid = u['anp_id']
        if u['estado'] != 'evaluable' or (only and uid not in only):
            continue
        d = np.load(TIERRA / f'distancias/cond100/{uid}.npz')['distancia']
        aviso = None
        if uid in AVISO:
            pct = 100 * u['superficie_continental_m2'] / u['area_m2']
            cifra = f'{pct:.0f}' if pct >= 1 else f'{pct:.1f}' if pct >= .1 else '< 0.1'
            aviso = f'Land part only: {cifra}% of the official area;\n{AVISO[uid]}.'
        a = areas[uid]
        fig, ax = plt.subplots(figsize=(8.2, 9.4))
        mapa_atlas(ax, base, u, d, base.shapes.get(uid), a['km2_d0.1'], a['km2_d0.2'],
                   f"Dominant ecoregion: {ECO_EN[a['ecorregion']]}", nota_tierra(), aviso)
        fig.savefig(dst / f'atlas_{uid}.png', dpi=dpi, bbox_inches='tight'); plt.close(fig)
        print('atlas', uid, flush=True)


CAPA_EN = {'superficie': 'sea surface', 'fondo': 'seafloor'}
ESTRATO_EN = {'PLATAFORMA': 'shelf (0–200 m)', 'TALUD': 'slope (200–1 000 m)',
              'PROFUNDO': 'deep floor (> 1 000 m)'}


def nota_mar(capa):
    a = CIFRAS['maritimo'][capa]['escala']
    n = 8 if capa == 'superficie' else 7
    return (f"Reading references for d ({CAPA_EN[capa]} metric; medians): neighbouring pixels (≈ 5.5 km) {a['curva'][0]['d']:.2f}; "
            f"pixels ≈ {a['km_d0.2']:.0f} km apart 0.2; random pairs of pixels {a['azar']:.2f}.\n"
            f"Bio-ORACLE v3.0 · {CAPA_EN[capa]} · 2000–2019 · 0.05° · {n} predictors · regularized Mahalanobis distance (κ ≤ 100). "
            'Grey: land.')


def atlas_maritimo(dpi=150):
    dst = ATLAS / 'maritimo'; dst.mkdir(parents=True, exist_ok=True)
    for capa in ('superficie', 'fondo'):
        base = lienzo(MAR / capa)
        eco = np.load(MAR / capa / ('preparados/ecorregiones_minam.npz' if capa == 'superficie' else 'preparados/zonas.npz'))['ecorregion']
        codes = ['MFH', 'MTR'] if capa == 'superficie' else ['PLATAFORMA', 'TALUD', 'PROFUNDO']  # pipeline order
        areas = CIFRAS['maritimo'][capa]['areas']
        for u in base.inventory:
            uid = u['anp_id']
            idx = np.load(MAR / capa / f'preparados/asociaciones/{uid}.npz')['index']
            linea2 = ('Sea surface ecoregions of the area: ' + '; '.join(ECO_EN[codes[e]] for e in np.unique(eco[idx])) if capa == 'superficie'
                      else 'Seafloor depth strata of the area: ' + '; '.join(ESTRATO_EN[codes[e]] for e in np.unique(eco[idx])))
            d = np.load(MAR / capa / f'distancias/cond100/{uid}.npz')['distancia']
            fig, ax = plt.subplots(figsize=(8.2, 9.4))
            mapa_atlas(ax, base, u, d, base.shapes.get(uid), areas[uid]['km2_d0.1'], areas[uid]['km2_d0.2'], linea2,
                       nota_mar(capa), tierra=False)
            ax.set_title(f"{nombre_en(u)} ({uid}) · {CAPA_EN[capa]}", loc='left', fontsize=11, color='#1f1f1d', pad=34)
            encuadre_mar(ax)
            fig.savefig(dst / f'atlas_{uid}_{capa}.png', dpi=dpi, bbox_inches='tight'); plt.close(fig)
            print('atlas', uid, capa, flush=True)


def docx():
    inv = {u['anp_id']: u for u in json.loads((TIERRA / 'preparados/inventario.json').read_text(encoding='utf-8'))}
    orden = json.loads((INSUMOS / 'tabla_s1.json').read_text(encoding='utf-8'))
    mar = [u['anp_id'] for u in json.loads((MAR / 'fondo/preparados/inventario.json').read_text(encoding='utf-8'))]

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for m in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'):
        setattr(sec, m, Cm(2.0))
    st = doc.styles['Normal']; st.font.name = 'Times New Roman'; st.font.size = Pt(11)


    def parrafo(texto, size=11, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=6):
        p = doc.add_paragraph(); p.alignment = align; p.paragraph_format.space_after = Pt(after)
        r = p.add_run(texto); r.font.size = Pt(size); r.bold = bold; r.italic = italic
        return p


    def salto():
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


    parrafo('File S2. Atlas of Climate Analogues of the Protected Areas of Peru', 15, bold=True,
            align=WD_ALIGN_PARAGRAPH.LEFT, after=4)
    parrafo('Supplementary material for: Protected-area networks can represent climate worse than chance, and threatened-species plans toward 30×30 need climatic targets to close the gaps: evidence from Peru. '
            'Irwing S. Saldaña and Armando Ugaz-Cherre.', 10, italic=True, align=WD_ALIGN_PARAGRAPH.LEFT, after=14)
    parrafo('This file gathers the climate analogy maps of the 258 protected areas with continental surface (maps S2.1 to '
            'S2.258), ordered by management modality (ANP, ACR and ACP), category and name, and the maps of the six protected '
            'areas with a maritime component for the sea surface and for the seafloor (maps S2.259 to S2.270).')
    parrafo('Each map shows, for each pixel of the domain, the Mahalanobis distance d to the most similar climate within the '
            'protected area, with the seven fixed classes used in the manuscript (0–0.05, nearly identical; 0.05–0.1, very '
            'similar; 0.1–0.2, similar; 0.2–0.3, moderate; 0.3–0.5, weak; 0.5–1, low similarity; > 1, not analogous), in blue up '
            'to 0.2 and in magenta beyond it. The protected area is shown in green with a dark green outline (a dark green '
            'circle marks the areas too small to be seen at the national scale). The subtitle '
            'gives the modality, the area with data, the number of pixels, the surface outside the area at d ≤ 0.1 and at '
            'd ≤ 0.2 and the dominant ecoregion (Ministerio del Ambiente [MINAM], 2017) or, in the maritime domain, the layer and '
            'the sea-surface ecoregions or the seafloor depth strata of the protected area.')
    parrafo('In the continental domain, d is computed with 11 predictors from CHELSA V2.1 (1981–2010, 30″); as reading '
            'references, d ≈ 0.05 between neighbouring pixels, ≈ 0.2 between pixels ≈ 8 km apart and ≈ 1.0 between two random '
            'places in the country. In the maritime domain, d is computed separately for the sea surface (8 predictors) and for '
            'the seafloor (7 predictors) from Bio-ORACLE v3.0 (2000–2019, 0.05°), and its values are not comparable with the '
            'continental ones. Protected areas represented only in part in the continental domain carry an inset with the '
            'continental proportion of their official area.')
    n = 0
    MOD = {'NACIONAL': 'Natural protected areas under national administration (ANP)',
           'REGIONAL': 'Regional conservation areas (ACR)', 'PRIVADA': 'Private conservation areas (ACP)'}
    for mod in ('NACIONAL', 'REGIONAL', 'PRIVADA'):
        for k, fila in enumerate(r for r in orden[mod] if r['d02'] is not None):
            salto()
            if k == 0:
                parrafo(f'Continental domain. {MOD[mod]}', 12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
            n += 1
            uid = fila['code']
            doc.add_picture(str(ATLAS / 'continental' / f'atlas_{uid}.png'), width=Cm(16.5))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            p = parrafo('', 10, align=WD_ALIGN_PARAGRAPH.LEFT)
            r = p.add_run(f'Map S2.{n}. '); r.bold = True; r.font.size = Pt(10)
            r = p.add_run(f'{nombre_en(inv[uid])} ({uid}). Climate analogy in the continental domain.'); r.font.size = Pt(10)
    for uid in mar:
        for capa, txt in (('superficie', 'sea surface'), ('fondo', 'seafloor')):
            salto()
            if uid == mar[0] and capa == 'superficie':
                parrafo('Maritime domain. Protected areas with a maritime component', 12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
            n += 1
            doc.add_picture(str(ATLAS / 'maritimo' / f'atlas_{uid}_{capa}.png'), width=Cm(16.5))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            p = parrafo('', 10, align=WD_ALIGN_PARAGRAPH.LEFT)
            r = p.add_run(f'Map S2.{n}. '); r.bold = True; r.font.size = Pt(10)
            r = p.add_run(f'{nombre_en(inv[uid])} ({uid}). Climate analogy in the maritime domain, {txt}.'); r.font.size = Pt(10)
    doc.save(DOCX)
    print('ok', n, 'maps', DOCX.stat().st_size // 2**20, 'MiB')


if __name__ == '__main__':
    pasos = sys.argv[1:] or ['png', 'docx']
    if 'png' in pasos:
        atlas_continental(); atlas_maritimo()
    if 'docx' in pasos:
        docx()
