"""Public inputs that are downloaded as whole files, and the check of every input against data/insumos_sha256.json.

Downloads, into the folders the pipeline reads (RECURSOS_GENERALES and SIG_DIR):
  WorldClim 2.1 elevation, 30"             elevation/wc2.1_30s_elev.tif
  GHS-SMOD R2023A, 2020, 30"               GHSL_SMOD_E2020_R2023A/GHS_SMOD_E2020_GLOBE_R2023A_4326_30ss_V2_0.tif
  PISCOp v3.0, monthly                     PISCO_SENAMIH/PISCOp_m.nc
  SENAMHI 1981-2010 normals (WMO/NCEI)     SENAMHI_normales_OMM_1981-2010/WMO_Normals_Excel_Template-PERU.xls
  HydroBASINS v1c, South America           (SIG_DIR) Basin SouthAmerica/hybas_sa_lev01-12_v1c.zip
CHELSA, Bio-ORACLE, the 200-nautical-mile limit and MapBiomas are downloaded by descargar_chelsa.py,
ejecutar_marino.py --fase descargar and mascara_natural.py --fase descargar. Three groups cannot be downloaded by a script:
  manual    SERNANP cadastre, INEI boundaries and MINAM ecoregions; their terms do not allow redistributing the files
  licensed  IUCN Red List and BirdLife ranges, supplied on request for non-commercial use
Every input, downloaded or not, is then checked against the SHA-256 of the files used for the article. Files rebuilt by a
download service (netCDF from ERDDAP, rasters exported from Earth Engine) can carry new metadata and differ in their hash
while holding the same values; the check reports them and goes on.
Run: python descargar_insumos.py [--verificar]   (--verificar checks without downloading)
"""
import hashlib, json, os, shutil, sys, tempfile, zipfile
from pathlib import Path
import requests

RECURSOS = Path(os.environ.get('RECURSOS_GENERALES', Path.home() / 'Documents' / 'Recursos generales'))
SIG_DIR = Path(os.environ.get('SIG_DIR', r'C:\Archivos\SIG'))
MANIFIESTO = Path(__file__).resolve().parents[1] / 'data' / 'insumos_sha256.json'
RAICES = {'RECURSOS_GENERALES': RECURSOS, 'SIG_DIR': SIG_DIR}

# (root, destination, URL, member of the zip to extract or None)
DESCARGAS = [
    ('RECURSOS_GENERALES', 'elevation/wc2.1_30s_elev.tif',
     'https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_elev.zip', 'wc2.1_30s_elev.tif'),
    ('RECURSOS_GENERALES', 'GHSL_SMOD_E2020_R2023A/GHS_SMOD_E2020_GLOBE_R2023A_4326_30ss_V2_0.tif',
     'https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_SMOD_GLOBE_R2023A/GHS_SMOD_E2020_GLOBE_R2023A_4326_30ss/V2-0/'
     'GHS_SMOD_E2020_GLOBE_R2023A_4326_30ss_V2_0.zip', 'GHS_SMOD_E2020_GLOBE_R2023A_4326_30ss_V2_0.tif'),
    ('RECURSOS_GENERALES', 'PISCO_SENAMIH/PISCOp_m.nc', 'https://ndownloader.figshare.com/files/64968111', None),  # doi:10.6084/m9.figshare.32411886.v1
    ('RECURSOS_GENERALES', 'SENAMHI_normales_OMM_1981-2010/WMO_Normals_Excel_Template-PERU.xls',
     'https://www.ncei.noaa.gov/data/normals-old/WMO/1981-2010/RA-III/Peru/WMO_Normals_Excel_Template-PERU.xls', None),
    ('SIG_DIR', 'Basin SouthAmerica/hybas_sa_lev01-12_v1c.zip',
     'https://data.hydrosheds.org/file/HydroBASINS/standard/hybas_sa_lev01-12_v1c.zip', None),
]
DONDE = {
    'CHELSA/V2.1/1981-2010/cartografia': 'SERNANP, Catastro Oficial de Areas Naturales Protegidas (https://geo.sernanp.gob.pe), '
                                         'cadastre of 9 October 2026, as GeoJSON per tier; INEI departmental limit dissolved to Peru',
    'capas_sig/Peru': 'INEI departmental boundaries (https://ide.inei.gob.pe)',
    'N53_ECORREGIONES': 'MINAM (2017), Mapa de Ecorregiones del Peru (https://geoservidor.minam.gob.pe)',
    'IUCN': 'IUCN Red List spatial data, version 2026-1 (https://www.iucnredlist.org/resources/spatial-data-download)',
    'BIRDLIFE_DATAZONE': 'BirdLife International and HBW, bird species distribution maps 2025 (http://datazone.birdlife.org)',
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for bloque in iter(lambda: f.read(1 << 22), b''):
            h.update(bloque)
    return h.hexdigest()


def descargar(url, destino, miembro=None):
    destino.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destino.parent) as tmp:
        tmp = Path(tmp) / 'descarga'
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(tmp, 'wb') as f:
                for bloque in r.iter_content(1 << 22):
                    f.write(bloque)
        if miembro:
            with zipfile.ZipFile(tmp) as z, z.open(miembro) as src, open(destino, 'wb') as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.move(tmp, destino)


def verificar():
    esperado = json.loads(MANIFIESTO.read_text(encoding='utf-8'))
    faltan, distintos, n = {}, [], 0
    for raiz, grupos in esperado.items():
        for grupo, archivos in grupos.items():
            for rel, info in archivos.items():
                p = RAICES[raiz] / rel; n += 1
                if not p.exists():
                    clave = next((k for k in DONDE if k in rel), rel)
                    faltan.setdefault((grupo, clave), []).append(rel)
                elif sha256(p) != info['sha256']:
                    distintos.append(f'{raiz}/{rel}')
    print(f'{n - sum(map(len, faltan.values())) - len(distintos)} of {n} input files match the article')
    for f in distintos:
        print('  different hash:', f)
    for (grupo, clave), rels in faltan.items():
        print(f'  missing ({grupo}, {len(rels)} files): {DONDE.get(clave, clave)}')
    return not faltan and not distintos


if __name__ == '__main__':
    if '--verificar' not in sys.argv:
        for raiz, rel, url, miembro in DESCARGAS:
            destino = RAICES[raiz] / rel
            if destino.exists():
                print('present', rel)
                continue
            print('downloading', rel, flush=True)
            descargar(url, destino, miembro)
    sys.exit(0 if verificar() else 1)
