"""Lector minimo de shapefile poligonal (tipo 5) y DBF, sin dependencias de GDAL/OGR."""
import struct

from shapely.geometry import LinearRing, MultiPolygon, Polygon


def read_dbf(path, encoding='utf-8'):
    with open(path, 'rb') as f:
        n, hlen, rlen = struct.unpack('<xxxxIHH20x', f.read(32))
        fields = []
        while True:
            d = f.read(32)
            if d[0] == 0x0D:
                break
            fields.append((d[:11].split(b'\0')[0].decode(), d[16]))
        f.seek(hlen)
        rows = []
        for _ in range(n):
            rec, pos, row = f.read(rlen), 1, {}
            for name, size in fields:
                row[name] = rec[pos:pos+size].decode(encoding, 'replace').strip()
                pos += size
            rows.append(row)
    return rows


def read_shp(path, bbox=None):
    """Geometrias en orden de registro; None si el registro es nulo o su caja no corta bbox (x0, y0, x1, y1)."""
    geoms = []
    with open(path, 'rb') as f:
        f.seek(100)
        while (h := f.read(8)) and len(h) == 8:
            content = f.read(struct.unpack('>ii', h)[1]*2)
            if struct.unpack('<i', content[:4])[0] == 0:
                geoms.append(None)
                continue
            x0, y0, x1, y1 = struct.unpack('<4d', content[4:36])
            if bbox and (x1 < bbox[0] or x0 > bbox[2] or y1 < bbox[1] or y0 > bbox[3]):
                geoms.append(None)
                continue
            nparts, npts = struct.unpack('<ii', content[36:44])
            parts = list(struct.unpack(f'<{nparts}i', content[44:44+4*nparts])) + [npts]
            xy = struct.unpack(f'<{2*npts}d', content[44+4*nparts:44+4*nparts+16*npts])
            rings = [[(xy[2*j], xy[2*j+1]) for j in range(parts[k], parts[k+1])] for k in range(nparts)]
            rings = [r for r in rings if len(r) > 3]
            # Convencion shapefile: exterior horario, huecos antihorarios; cada hueco va al exterior que lo contiene.
            shells = [[Polygon(r), []] for r in rings if not LinearRing(r).is_ccw]
            for hole in (r for r in rings if LinearRing(r).is_ccw):
                for shell in shells:
                    if shell[0].contains(Polygon(hole).representative_point()):
                        shell[1].append(hole)
                        break
            polys = [Polygon(p.exterior.coords, hs) for p, hs in shells]
            geoms.append(polys[0] if len(polys) == 1 else MultiPolygon(polys))
    return geoms
