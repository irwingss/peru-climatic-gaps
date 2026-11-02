"""P2 sin las 63 areas < 1 km2 (61 ACP y 2 ANP; area_m2 del inventario, como en la prueba de robustez de 3.6) y superficie
exacta de cada modalidad. Escribe 06_implicaciones/P2_modalidades_por_ecorregion_sin_pequenas.csv y P2_sin_pequenas.json.
Responde a M11 de manuscrito/envio_PECON/revision_discusion_v13.md (1/11/2026)."""
import json

import numpy as np

import ejecutar_investigacion as E
import implicaciones as I

prep, dom, area, grupo, ecos, dep, nombres_dep = I._contexto()
inv = json.loads((prep / 'inventario.json').read_text(encoding='utf-8'))
ev = {u['anp_id']: u for u in inv if u['estado'] == 'evaluable'}
pequenas = {k for k, u in ev.items() if u['area_m2'] < 1e6}
assert len(pequenas) == 63, len(pequenas)
climate = np.load(I.CONT / 'modelo' / 'clima_transformado_cond100.npy')
unidades = [u for u in I._unidades(prep) if u[0] not in pequenas]
d = {s: I._dmin(climate, np.concatenate([ix for _, _, m, ix in unidades if m == mod])) for mod, s in I.MODALIDADES.items()}
D = np.stack([d['ANP'], d['ACR'], d['ACP']]); red = D.min(0); prov = D.argmin(0)
cub = red <= I.UMBRAL
col = {'cubierta_red_km2': cub, 'cubierta_solo_ANP_km2': d['ANP'] <= I.UMBRAL,
       'solo_ACR_ACP_km2': (d['ANP'] > I.UMBRAL) & cub,
       'solo_ACR_ACP_proveedora_ACR_km2': (d['ANP'] > I.UMBRAL) & cub & (prov == 1),
       'solo_ACR_ACP_proveedora_ACP_km2': (d['ANP'] > I.UMBRAL) & cub & (prov == 2)}
tot = {'nombre': 'Peru', 'superficie_km2': area.sum() / 1e6, **{c: area[v].sum() / 1e6 for c, v in col.items()}}
E.write_csv(I.DST / 'P2_modalidades_por_ecorregion_sin_pequenas.csv', [tot, *I._tabla(grupo, ecos, area, col)])
sup = {s: sum(u['superficie_continental_m2'] for u in ev.values() if u['modalidad'] == m) / 1e6 for m, s in I.MODALIDADES.items()}
res = {'n_excluidas': len(pequenas), 'excluidas_km2': sum(ev[k]['area_m2'] for k in pequenas) / 1e6,
       'excluidas_por_modalidad': {s: sum(ev[k]['modalidad'] == m for k in pequenas) for m, s in I.MODALIDADES.items()},
       'superficie_exacta_km2': sup, 'F02_red': float(area[cub].sum() / area.sum()),
       **{k: tot[k] for k in col if k.startswith('solo')}}
E.save_json(I.DST / 'P2_sin_pequenas.json', res)
print(json.dumps(res, indent=1))
