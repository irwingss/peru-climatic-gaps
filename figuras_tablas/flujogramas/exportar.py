"""Exporta el HTML del flujograma a PNG (≈600 dpi) y PDF vectorial con Chrome o Edge sin interfaz.

    python exportar.py figura_flujo.html ../../resultados/figuras/Figura_flujo [--ancho-mm 180 --dpi 600]

El tamaño en px se lee de `@page{size:Wpx Hpx}` del HTML, que debe coincidir con lienzo(W, H);
si no coinciden, el PNG queda con franjas blancas o recortado.
"""
import argparse, re, shutil, subprocess, sys
from pathlib import Path

CANDIDATOS = [
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
]


def navegador():
    for c in CANDIDATOS:
        if Path(c).exists():
            return c
    for n in ('google-chrome', 'chromium', 'chromium-browser', 'chrome', 'msedge', 'microsoft-edge'):
        if shutil.which(n):
            return shutil.which(n)
    sys.exit('No se encontró Chrome ni Edge; instala uno o agrega su ruta a CANDIDATOS.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('html')
    ap.add_argument('salida', help='ruta sin extensión; se escriben .png y .pdf')
    ap.add_argument('--ancho-mm', type=float, default=180, help='ancho de impresión (180 mm = doble columna)')
    ap.add_argument('--dpi', type=float, default=600)
    a = ap.parse_args()
    html = Path(a.html).resolve()
    m = re.search(r'@page\s*\{\s*size:\s*(\d+)px\s+(\d+)px', html.read_text(encoding='utf-8'))
    if not m:
        sys.exit('El HTML necesita @page{size:Wpx Hpx;margin:0} con el mismo tamaño del lienzo.')
    w, h = int(m[1]), int(m[2])
    escala = round(a.dpi * a.ancho_mm / 25.4 / w, 3)
    salida = Path(a.salida).resolve(); salida.parent.mkdir(parents=True, exist_ok=True)
    base = [navegador(), '--headless=new', '--disable-gpu', '--hide-scrollbars', '--no-pdf-header-footer',
            '--allow-file-access-from-files', '--virtual-time-budget=5000']
    subprocess.run(base + [f'--force-device-scale-factor={escala}', f'--window-size={w},{h}',
                           f'--screenshot={salida.with_suffix(".png")}', html.as_uri()], check=True, capture_output=True)
    subprocess.run(base + [f'--print-to-pdf={salida.with_suffix(".pdf")}', html.as_uri()], check=True, capture_output=True)
    print(f'{salida.with_suffix(".png")}  ({round(w * escala)} x {round(h * escala)} px, {a.dpi:.0f} dpi a {a.ancho_mm:.0f} mm)')
    print(salida.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
