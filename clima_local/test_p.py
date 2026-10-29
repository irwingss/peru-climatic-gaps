"""Prueba minima de la p bilateral, el SES y Holm. Ejecutar: python test_p.py"""
import numpy as np

from ejecutar_investigacion import holm, prueba

r = prueba(np.arange(1, 100, dtype=float), 100.)
assert r['p_bilateral'] == .02 and r['redes_mayor_o_igual'] == 0 and r['ses'] > 0
assert holm([{**r}])[0]['direccion'] == 'peor'
assert prueba(np.arange(1, 100, dtype=float), 0.)['ses'] < 0  # real mejor que todas
# Holm: nacional aparte; p ordenadas .01, .02, .04 con m = 3 -> .03, .04, .04
s = holm([{'p_bilateral': .5, 'ses': 1.}] + [{'p_bilateral': p, 'ses': -1.} for p in (.04, .01, .02)])
assert [round(x['p_holm'], 10) for x in s] == [.5, .04, .03, .04]
assert [x['direccion'] for x in s] == ['no distinguible', 'mejor', 'mejor', 'mejor']
print('ok')
