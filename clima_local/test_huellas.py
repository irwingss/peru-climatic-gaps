"""Prueba minima de colocar(): 8 orientaciones distintas de igual tamano, TVD y fraccion natural. python test_huellas.py"""
import numpy as np

from ejecutar_investigacion import colocar

h = w = 20
flat = np.arange(h*w); rows, cols = flat//w, flat % w
lookup = flat.copy()
grupo = (cols >= 10).astype(int)            # estrato 0 a la izquierda, 1 a la derecha
natural = rows < 15                          # las 5 filas de abajo no son naturales
# Huella en L de 4 celdas, toda en el estrato 0; referencia en la esquina (fila 0, columna 0 de la L)
hu = {'dr': np.array([0, 1, 2, 2]), 'dc': np.array([0, 0, 0, 1]), 'area': np.ones(4), 'comp': np.array([1., 0.])}
formas = set()
for o in range(8):
    c, fd, fn, tvd = colocar(hu, 5*w+5, o, rows, cols, lookup, h, w, grupo, natural, 2)
    assert fd == 1 and fn == 1 and tvd == 0 and len(c) == 4
    formas.add(tuple(sorted(c)))
assert len(formas) == 8                      # la L no tiene simetrias: 8 posiciones distintas
c, fd, fn, tvd = colocar(hu, 5*w+9, 0, rows, cols, lookup, h, w, grupo, natural, 2)
assert tvd == .25                            # una de 4 celdas cae en el estrato 1
c, fd, fn, tvd = colocar(hu, 13*w+5, 0, rows, cols, lookup, h, w, grupo, natural, 2)
assert fn == .5                              # filas 13, 14, 15 y 15: solo 13 y 14 son naturales
c, fd, fn, tvd = colocar(hu, 18*w+5, 0, rows, cols, lookup, h, w, grupo, natural, 2)
assert fd == .5                              # la mitad cae fuera de la cuadricula
print('ok')
