// CUDA 讲义动画的共用小工具（依赖 anim.js 里的 el / txt / clear / $）
function R(g, x, y, w, h, o = {}) {
  const r = el('rect', {x, y, width: w, height: h, rx: o.rx ?? 4, fill: o.fill ?? 'var(--bg)',
    stroke: o.stroke ?? 'var(--line)', 'stroke-width': o.sw ?? 1}, g);
  if (o.op !== undefined) r.setAttribute('opacity', o.op);
  return r;
}
function T(g, s, x, y, o = {}) {
  const t = txt(String(s), {x, y, 'text-anchor': o.a ?? 'start', class: o.cls ?? ''}, g);
  if (o.fill) t.style.fill = o.fill;
  if (o.size) t.style.fontSize = o.size + 'px';
  if (o.bold) t.style.fontWeight = '600';
  if (o.mono) t.style.fontFamily = 'ui-monospace,Menlo,monospace';
  return t;
}
function L(g, x1, y1, x2, y2, o = {}) {
  const l = el('line', {x1, y1, x2, y2, stroke: o.stroke ?? 'var(--muted)', 'stroke-width': o.sw ?? 1}, g);
  if (o.arrow) l.setAttribute('marker-end', 'url(#ar)');
  if (o.dash) l.setAttribute('stroke-dasharray', o.dash);
  return l;
}
function arrowDefs(g, color = 'var(--muted)') {
  const d = el('defs', {}, g), m = el('marker', {id: 'ar', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto'}, d);
  el('path', {d: 'M0,0 L10,5 L0,10 z', fill: color}, m);
}
function cap(g, s, w = 680) { T(g, s, w / 2, 24, {a: 'middle', cls: 'cap'}); }
// 一组互斥按钮：opts = [{id, label, value}]，onPick(value)
function radio(rowId, opts, get, onPick) {
  const row = $(rowId);
  opts.forEach(o => {
    const b = document.createElement('button'); b.id = o.id; b.textContent = o.label;
    b.onclick = () => { onPick(o.value); sync(); };
    row.appendChild(b);
  });
  function sync() { opts.forEach(o => $(o.id).classList.toggle('on', get() === o.value)); }
  sync();
  return sync;
}
function stat(id, v) { const e = $(id); if (e) e.textContent = v; }
