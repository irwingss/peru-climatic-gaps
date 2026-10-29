"""Rows of Table S2.1 of File S2: official data and results for the 259 protected areas (English labels), read by atlas.py."""
import json
from figuras_comun import TIERRA, MAR, MS, ECO_EN, CAT_EN, CIFRAS

inv = json.loads((TIERRA / 'preparados/inventario.json').read_text(encoding='utf-8'))
mar = {u['anp_id']: u for u in json.loads((MAR / 'fondo/preparados/inventario.json').read_text(encoding='utf-8'))}
areas = {a['anp_id']: a for a in CIFRAS['continental']['areas']}


def nombre_oficial(u):
    n = u['nombre'].strip()
    return n[0].upper() + n[1:] if n.split()[0] in ('los', 'las', 'la', 'el') else n


filas = {'NACIONAL': [], 'REGIONAL': [], 'PRIVADA': []}
for u in inv:
    uid = u['anp_id']
    row = {'code': uid, 'category': CAT_EN[u['categoria']], 'name': nombre_oficial(u), 'created': u['f_creacion'] or '',
           'area': u['area_m2'] / 1e6, 'cont': u['superficie_continental_m2'] / 1e6,
           'mar': mar[uid]['superficie_clima_m2'] / 1e6 if uid in mar else 0.0}
    if u['estado'] == 'evaluable':
        a = areas[uid]
        row.update(ecoregion=ECO_EN[a['ecorregion']], d01=a['km2_d0.1'], d02=a['km2_d0.2'])
    else:
        row.update(ecoregion='– (maritime only)', d01=None, d02=None)
    filas[u['modalidad']].append(row)
for k in filas:
    filas[k].sort(key=lambda r: (r['category'], r['name']))
(MS / 'insumos_figuras/tabla_s1.json').write_text(json.dumps(filas, ensure_ascii=False, indent=0), encoding='utf-8')
print({k: len(v) for k, v in filas.items()})
