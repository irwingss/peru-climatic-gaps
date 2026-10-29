# Robustez del test (kappa <= 50 y sin las areas < 1 km2): corridas pesadas, de una en una por la RAM.
set -e
O=resultados_v7/02_sesgo_precipitacion/original; M=resultados_v7/03_mascara_natural; D=resultados_v7/07_final
E="python clima_local/ejecutar_investigacion.py --out $O"
$E --fase sistema --cond 50
$E --fase cobertura --cond 50 --diseno N2c --limitar-superposicion --mascara $M --dst $D/cond50/N2c
$E --fase cobertura --cond 100 --diseno N2c --limitar-superposicion --mascara $M --sin-menores 1 --dst $D/sin_menores_1km2/N2c
