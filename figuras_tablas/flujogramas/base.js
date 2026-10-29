// Shared SVG helpers for the graphical workflow figures.
const NS = 'http://www.w3.org/2000/svg';
const INK = '#1f1f1d', SOFT = '#52514e', MUTED = '#8a8986', LINE = '#c9c8c2', CYAN = '#0096c7', RED = '#a50f15';
const REF = '#2a78d6';  // references of the example protected area
let svg;

function lienzo(w, h) {
  svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('width', w); svg.setAttribute('height', h); svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  svg.setAttribute('font-family', 'Arial, Helvetica, sans-serif');
  document.body.appendChild(svg);
  const defs = el('defs', {});
  for (const [id, col] of [['flecha', SOFT], ['flechaRoja', RED], ['flechaCian', CYAN]]) {
    const m = el('marker', {id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse'}, defs);
    el('path', {d: 'M0,0 L10,5 L0,10 z', fill: col}, m);
  }
  return svg;
}

function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  (parent || svg).appendChild(e);
  return e;
}

// Text with simple markup: x_{sub}, x^{sup}, **bold**, *italic*; '\n' gives new lines.
function txt(x, y, s, o = {}) {
  const size = o.size || 9, lh = o.lh || size * 1.25;
  const t = el('text', {x, y, 'font-size': size, fill: o.fill || '#2e2d2b', 'text-anchor': o.anchor || 'start',
                        'font-weight': o.weight || 'normal', 'font-style': o.style || 'normal'}, o.parent);
  s.split('\n').forEach((line, i) => {
    const row = el('tspan', {x, dy: i ? lh : 0}, t);
    if (!line.length) line = ' ';  // empty tspans do not advance the line
    if (!line.length) line = ' ';
    const re = /(_\{[^}]*\}|\^\{[^}]*\}|\*\*[^*]+\*\*|\*[^*]+\*)/g;
    let last = 0, m;
    while ((m = re.exec(line)) !== null) {
      if (m.index > last) row.appendChild(document.createTextNode(line.slice(last, m.index)));
      const tok = m[0];
      if (tok.startsWith('_{') || tok.startsWith('^{')) {
        const sp = el('tspan', {'font-size': size * .72, 'baseline-shift': tok[0] === '_' ? 'sub' : 'super'}, row);
        sp.textContent = tok.slice(2, -1);
      } else if (tok.startsWith('**')) {
        const sp = el('tspan', {'font-weight': 'bold'}, row); sp.textContent = tok.slice(2, -2);
      } else {
        const sp = el('tspan', {'font-style': 'italic'}, row); sp.textContent = tok.slice(1, -1);
      }
      last = m.index + tok.length;
    }
    if (last < line.length) row.appendChild(document.createTextNode(line.slice(last)));
  });
  return t;
}

function flecha(pts, o = {}) {
  const d = 'M' + pts.map(p => p.join(',')).join(' L');
  return el('path', {d, fill: 'none', stroke: o.color || SOFT, 'stroke-width': o.w || 1.4,
                     'stroke-dasharray': o.dash || 'none', 'marker-end': `url(#${o.marker || 'flecha'})`});
}

function curva(x1, y1, x2, y2, o = {}) {
  const mx = (x1 + x2) / 2;
  return el('path', {d: `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`, fill: 'none', stroke: o.color || SOFT,
                     'stroke-width': o.w || 1.3, 'marker-end': `url(#${o.marker || 'flecha'})`});
}

function paso(x, y, n, col = INK) {
  el('circle', {cx: x, cy: y, r: 9, fill: col});
  txt(x, y + 3.6, String(n), {size: 10, weight: 'bold', fill: '#fff', anchor: 'middle'});
}

function tarjeta(x, y, w, h, o = {}) {
  return el('rect', {x, y, width: w, height: h, rx: 7, fill: o.fill || '#fafaf8', stroke: o.stroke || '#e1e0db', 'stroke-width': 1});
}

function imagen(href, x, y, w, h) {
  return el('image', {href, x, y, width: w, height: h, preserveAspectRatio: 'xMidYMid meet'});
}

// Linear scale helper.
function escala(d0, d1, r0, r1) { return v => r0 + (v - d0) / (d1 - d0) * (r1 - r0); }

function ejes(x, y, w, h, o = {}) {
  el('line', {x1: x, y1: y + h, x2: x + w, y2: y + h, stroke: MUTED, 'stroke-width': .8});
  el('line', {x1: x, y1: y, x2: x, y2: y + h, stroke: MUTED, 'stroke-width': .8});
  if (o.xlab) txt(x + w / 2, y + h + (o.xoff || 22), o.xlab, {size: 7.5, anchor: 'middle', fill: SOFT});
  if (o.ylab) {
    const t = txt(0, 0, o.ylab, {size: 7.5, anchor: 'middle', fill: SOFT});
    t.setAttribute('transform', `translate(${x - (o.yoff || 10)},${y + h / 2}) rotate(-90)`);
    t.querySelectorAll(':scope > tspan').forEach(s => s.setAttribute('x', 0));
  }
}

function fmt(v, nd = 0) { return v.toFixed(nd).replace(/\B(?=(\d{3})+(?!\d))/g, ' '); }

// Two-dimensional climate space: grey national cloud, references of the area, queries with r*.
function espacio(x, y, w, h, E, o = {}) {
  // zoom on the references and the query pixels; the national cloud is clipped to the panel
  const foco = E.refs.concat([E.c1, E.r1, E.c2, E.r2]);
  const xs = foco.map(p => p[0]), ys = foco.map(p => p[1]);
  const pad = o.pad || .35;
  const cp = el('clipPath', {id: 'clip' + x}, svg.querySelector('defs'));
  el('rect', {x, y, width: w, height: h}, cp);
  const g = el('g', {'clip-path': `url(#clip${x})`});
  const [xa, xb] = [Math.min(...xs), Math.max(...xs)], [ya, yb] = [Math.min(...ys), Math.max(...ys)];
  const sx = escala(xa - pad * (xb - xa), xb + pad * (xb - xa), x, x + w), sy = escala(ya - pad * (yb - ya), yb + pad * (yb - ya), y + h, y);
  for (const p of E.nube) el('circle', {cx: sx(p[0]), cy: sy(p[1]), r: o.rg || 1.3, fill: '#cfcec8', opacity: .75}, g);
  for (const p of E.refs) el('circle', {cx: sx(p[0]), cy: sy(p[1]), r: o.rr || 2.1, fill: REF, opacity: .85}, g);
  const q = [['c1', 'r1', '#1b7837'], ...(o.dos === false ? [] : [['c2', 'r2', RED]])];
  for (const [c, r, col] of q) {
    el('line', {x1: sx(E[c][0]), y1: sy(E[c][1]), x2: sx(E[r][0]), y2: sy(E[r][1]), stroke: col, 'stroke-width': 1.4, 'stroke-dasharray': '3,2'});
    el('circle', {cx: sx(E[r][0]), cy: sy(E[r][1]), r: 3.6, fill: 'none', stroke: INK, 'stroke-width': 1.2});
    estrella(sx(E[c][0]), sy(E[c][1]), o.re || 6, col);
  }
  return {sx, sy};
}

function estrella(cx, cy, r, col) {
  const pts = [];
  for (let i = 0; i < 10; i++) {
    const a = Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * .45 : r;
    pts.push([cx + rr * Math.cos(a), cy - rr * Math.sin(a)]);
  }
  el('polygon', {points: pts.map(p => p.join(',')).join(' '), fill: col, stroke: '#fff', 'stroke-width': .8});
}

function leyendaD(x, y, o = {}) {
  const lab = ['0–0.05 nearly identical', '0.05–0.1 very similar', '0.1–0.2 similar', '0.2–0.3 moderate', '0.3–0.5 weak', '0.5–1 low similarity', '> 1 not analogous'];
  const cls = DATA.d_classes.concat([['#ffffff', 1, '']]);
  cls.forEach((c, i) => {
    el('rect', {x, y: y + i * (o.dy || 11), width: 12, height: 8, fill: c[0], 'fill-opacity': c[1], stroke: '#bdbdb5', 'stroke-width': .4});
    txt(x + 16, y + 7 + i * (o.dy || 11), lab[i], {size: o.size || 7, fill: SOFT});
  });
}
