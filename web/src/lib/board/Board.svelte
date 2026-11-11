<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import { app, prefersReducedMotion, type MenuItem } from '../state.svelte';
  import { get, post, patch, del, errText, fmtInt, type BoardObject, type NodeDetail, type Region, type VCell, type VEdge, type VNode, type Viewport } from '../api';
  import { Renderer, ShapeBatch, LineBatch } from './renderer';
  import Icon from '../ui/Icon.svelte';
  import Minimap from './Minimap.svelte';
  import BoardObjects from './BoardObjects.svelte';
  import { clickOutside } from '../ui/actions';

  // Palette: Fylume greys, with Yaob's ember for selection and gold for "needs attention".
  const C = {
    white: 0xf7f7f2ff,
    muted: 0xabababff,
    dim: 0x6e6e6eff,
    line: 0x303030ff,
    regionFill: 0x0b0b0bff,
    domain: 0x262626ff,
    domain2: 0x1d1d1dff,
    void: 0x050505ff,
    ember: 0xf34b27ff,
    gold: 0xf7b547ff,
    edge: 0x8a8a8a59,
    edgeNear: 0xf7f7f2cc,
    cell: 0xf7f7f28c,
  };
  const MAX_ZOOM = 14;
  const DETAIL_PX = 30; // node radius on screen from which cells are drawn
  const SANS = '"Space Grotesk", system-ui, sans-serif';
  const MONO = '"IBM Plex Mono", ui-monospace, monospace';

  let host: HTMLDivElement;
  let glCanvas: HTMLCanvasElement;
  let textCanvas: HTMLCanvasElement;
  let worldLayer = $state<HTMLDivElement | null>(null);
  let minimap = $state<ReturnType<typeof Minimap> | null>(null);
  let ctx: CanvasRenderingContext2D | null = null;
  let renderer: Renderer | null = null;

  const cam = { x: 0, y: 0, zoom: 0.2 };
  let W = 1;
  let H = 1;
  let dpr = 1;
  let viewReady = false;

  // ---- data shown on the board (plain fields: mutated per frame, never rendered through Svelte)
  let nodes: VNode[] = [];
  let nodeById = new Map<string, VNode>();
  let cells: VCell[] = [];
  let cellSize = 0;
  let edges: VEdge[] = [];
  let domains: Region[] = [];
  const overrides = new Map<string, { x: number; y: number }>();
  let hoverId: string | null = null;
  let dirtyNodes = true;
  let dirtyRegions = true;
  let dirtyEdges = true;
  let lastDetail = false;
  let geomRev = 0;

  let level = $state<'nodes' | 'cells' | null>(null);
  let totalInView = $state(0);
  let glError = $state<string | null>(null);
  let loadError = $state<string | null>(null);
  let firstLoad = $state(true);
  let canBack = $state(false);
  let canFwd = $state(false);
  let showBundles = $state(true);
  let viewsOpen = $state(false);
  let keyOpen = $state(false);
  let viewName = $state('');
  let viewBusy = $state(false);
  let objects = $state.raw<BoardObject[]>([]);
  let selectedObject = $state<string | null>(null);
  let band = $state<{ x: number; y: number; w: number; h: number } | null>(null);
  let tip = $state<{ x: number; y: number; text: string } | null>(null);
  let connectSrc = $state.raw<VNode | null>(null);
  let cursor = $state('grab');

  const stats = { nodes: 0, labels: 0, edges: 0, cells: 0, regions: 0, level: '', requests: [] as { ms: number; serverMs: number; bytes: number; level: string; items: number }[], frames: 0 };

  const spacing = () => app.board?.spacing ?? 26;
  const nodeR = () => spacing() * 0.3;
  const nodeRpx = () => Math.max(nodeR() * cam.zoom, 1.6);
  const sx = (x: number) => (x - cam.x) * cam.zoom + W / 2;
  const sy = (y: number) => (y - cam.y) * cam.zoom + H / 2;
  const wx = (px: number) => (px - W / 2) / cam.zoom + cam.x;
  const wy = (py: number) => (py - H / 2) / cam.zoom + cam.y;
  const posOf = (n: VNode) => overrides.get(n.id) ?? n;

  // ---------------------------------------------------------------- frame loop (runs only when something changed)
  let raf = 0;
  let lastT = 0;
  function requestFrame() {
    if (!raf) raf = requestAnimationFrame(frame);
  }
  function frame(t: number) {
    raf = 0;
    const dt = lastT ? Math.min(64, t - lastT) : 16;
    lastT = t;
    const moving = stepCamera(t, dt);
    if (!renderer) return;
    const detail = nodeRpx() >= DETAIL_PX;
    if (detail !== lastDetail) {
      lastDetail = detail;
      dirtyNodes = true;
      if (detail) queueDetails();
    }
    if (dirtyRegions) buildRegions();
    if (dirtyNodes) buildNodes();
    if (dirtyEdges) buildEdges();
    renderer.setAlpha('bundles', bundleAlpha());
    // far out, nodes are texture under the region names; closer in they are the subject
    renderer.setAlpha('nodes', Math.max(0.55, Math.min(1, 0.55 + (nodeR() * cam.zoom - 1.6) * 0.45)));
    renderer.draw(cam.x, cam.y, cam.zoom, ['bundles', 'mpn', 'domain', 'edges', 'cells', 'nodes'], nodeRpx());
    drawLabels();
    if (worldLayer) {
      worldLayer.style.transform = `translate(${W / 2 - cam.x * cam.zoom}px, ${H / 2 - cam.y * cam.zoom}px) scale(${cam.zoom})`;
      worldLayer.style.setProperty('--z', String(cam.zoom));
    }
    minimap?.draw(cam, W, H);
    stats.frames++;
    if (moving) requestFrame();
    else lastT = 0;
  }

  // ---------------------------------------------------------------- camera
  interface View {
    x: number;
    y: number;
    zoom: number;
  }
  let anim: { from: View; to: View; t0: number; dur: number } | null = null;
  let zoomTo: { zoom: number; px: number; py: number; wx: number; wy: number } | null = null;

  function minZoom() {
    const b = app.board?.bounds;
    if (!b) return 0.02;
    return Math.min(W / (b[2] - b[0]), H / (b[3] - b[1])) * 0.4;
  }
  const clampZoom = (z: number) => Math.min(MAX_ZOOM, Math.max(minZoom(), z));

  function stepCamera(t: number, dt: number): boolean {
    let moving = false;
    if (anim) {
      const k = Math.min(1, (t - anim.t0) / anim.dur);
      const e = 1 - Math.pow(1 - k, 3);
      cam.zoom = Math.exp(Math.log(anim.from.zoom) + (Math.log(anim.to.zoom) - Math.log(anim.from.zoom)) * e);
      cam.x = anim.from.x + (anim.to.x - anim.from.x) * e;
      cam.y = anim.from.y + (anim.to.y - anim.from.y) * e;
      if (k >= 1) anim = null;
      else moving = true;
      cameraChanged();
    }
    if (zoomTo) {
      const k = 1 - Math.exp(-dt / 55);
      let z = cam.zoom * Math.pow(zoomTo.zoom / cam.zoom, k);
      if (Math.abs(Math.log(zoomTo.zoom / z)) < 0.004) z = zoomTo.zoom;
      cam.zoom = z;
      cam.x = zoomTo.wx - (zoomTo.px - W / 2) / z;
      cam.y = zoomTo.wy - (zoomTo.py - H / 2) / z;
      if (z === zoomTo.zoom) zoomTo = null;
      else moving = true;
      cameraChanged();
    }
    return moving;
  }

  let settleTimer: ReturnType<typeof setTimeout> | null = null;
  let lastLoadAt = 0;
  function cameraChanged() {
    const now = performance.now();
    if (now - lastLoadAt > 320) loadViewport();
    else scheduleLoad(110);
    if (settleTimer) clearTimeout(settleTimer);
    settleTimer = setTimeout(settled, 500);
    if (tip) tip = null;
  }
  function settled() {
    app.zoom = cam.zoom;
    recordGesture();
    app.savePanels(50);
    if (nodeRpx() >= DETAIL_PX) queueDetails();
  }
  function setCam(x: number, y: number, zoom: number) {
    cam.x = x;
    cam.y = y;
    cam.zoom = clampZoom(zoom);
    cameraChanged();
    requestFrame();
  }
  function animateTo(v: View, dur = 220) {
    zoomTo = null;
    v = { ...v, zoom: clampZoom(v.zoom) };
    if (prefersReducedMotion() || dur <= 0) {
      anim = null;
      setCam(v.x, v.y, v.zoom);
      return;
    }
    anim = { from: { ...cam }, to: v, t0: performance.now(), dur };
    requestFrame();
  }
  function zoomAt(px: number, py: number, factor: number, smooth: boolean) {
    anim = null;
    const base = zoomTo ? zoomTo.zoom : cam.zoom;
    const target = clampZoom(base * factor);
    const wxx = zoomTo ? zoomTo.wx : wx(px);
    const wyy = zoomTo ? zoomTo.wy : wy(py);
    if (!smooth || prefersReducedMotion()) {
      zoomTo = null;
      const ax = wx(px);
      const ay = wy(py);
      cam.zoom = target;
      cam.x = ax - (px - W / 2) / target;
      cam.y = ay - (py - H / 2) / target;
      cameraChanged();
    } else {
      zoomTo = { zoom: target, px, py, wx: wxx, wy: wyy };
    }
    requestFrame();
  }

  // ---- view history (back / forward)
  let hist: View[] = [];
  let hi = -1;
  const near = (a: View, b: View) => Math.abs(a.x - b.x) * b.zoom < 3 && Math.abs(a.y - b.y) * b.zoom < 3 && Math.abs(Math.log(a.zoom / b.zoom)) < 0.01;
  function pushHistory(v: View) {
    if (hi >= 0 && near(hist[hi], v)) return;
    hist = hist.slice(0, hi + 1);
    hist.push({ ...v });
    if (hist.length > 60) hist.shift();
    hi = hist.length - 1;
    canBack = hi > 0;
    canFwd = false;
  }
  function recordGesture() {
    if (anim || zoomTo) return;
    if (hi < 0) return pushHistory({ ...cam });
    const h = hist[hi];
    const far = Math.abs(h.x - cam.x) * cam.zoom > W * 0.25 || Math.abs(h.y - cam.y) * cam.zoom > H * 0.25 || Math.abs(Math.log(h.zoom / cam.zoom)) > 0.22;
    if (far) pushHistory({ ...cam });
  }
  function jump(v: View) {
    pushHistory({ ...cam });
    v = { ...v, zoom: clampZoom(v.zoom) };
    pushHistory(v);
    animateTo(v);
  }
  function back() {
    if (hi <= 0) return;
    if (!near(hist[hi], { ...cam })) {
      // the current, unrecorded position becomes the forward target
      pushHistory({ ...cam });
    }
    hi--;
    canBack = hi > 0;
    canFwd = true;
    animateTo(hist[hi]);
  }
  function forward() {
    if (hi >= hist.length - 1) return;
    hi++;
    canBack = hi > 0;
    canFwd = hi < hist.length - 1;
    animateTo(hist[hi]);
  }

  function fitBounds(minx: number, miny: number, maxx: number, maxy: number, pad = 48, maxZoom = MAX_ZOOM): View {
    const bw = Math.max(maxx - minx, 1);
    const bh = Math.max(maxy - miny, 1);
    const plain = Math.min((W - pad * 2) / bw, (H - pad * 2) / bh, maxZoom);
    // keep clear of the tools (top) and the overview map (right) when that costs little zoom
    const top = 40;
    const right = W > 700 ? 208 : 0;
    const inset = Math.min((W - pad * 2 - right) / bw, (H - pad * 2 - top) / bh, maxZoom);
    if (inset >= plain * 0.82) {
      const z = clampZoom(inset);
      return { x: (minx + maxx) / 2 + right / 2 / z, y: (miny + maxy) / 2 - top / 2 / z, zoom: z };
    }
    return { x: (minx + maxx) / 2, y: (miny + maxy) / 2, zoom: clampZoom(plain) };
  }
  function fitAllView(): View {
    const b = app.board?.bounds ?? [-1000, -1000, 1000, 1000];
    return fitBounds(b[0], b[1], b[2], b[3], 40);
  }
  function fitAll() {
    jump(fitAllView());
  }
  function fitSelection() {
    const pts = app.selection.map((id) => nodeById.get(id)).filter(Boolean) as VNode[];
    if (!pts.length) return;
    let a = Infinity,
      b = Infinity,
      c = -Infinity,
      d = -Infinity;
    for (const n of pts) {
      const p = posOf(n);
      a = Math.min(a, p.x);
      b = Math.min(b, p.y);
      c = Math.max(c, p.x);
      d = Math.max(d, p.y);
    }
    const m = spacing() * 1.5;
    jump(fitBounds(a - m, b - m, c + m, d + m, 80, 6));
  }
  function fitRegion(id: string) {
    const r = app.board?.mpns.find((m) => m.id === id);
    if (r) jump(fitBounds(r.x - r.r, r.y - r.r, r.x + r.r, r.y + r.r, 28));
  }

  // ---------------------------------------------------------------- viewport loading (bounded, cancellable, cached)
  const cache = new Map<string, Viewport>();
  let loadTimer: ReturnType<typeof setTimeout> | null = null;
  let ctrl: AbortController | null = null;
  let shownKey = '';
  function scheduleLoad(delay: number) {
    if (loadTimer) clearTimeout(loadTimer);
    loadTimer = setTimeout(loadViewport, delay);
  }
  async function loadViewport() {
    if (loadTimer) clearTimeout(loadTimer);
    loadTimer = null;
    if (!app.board || !viewReady) return;
    lastLoadAt = performance.now();
    // the scale sent to the server is quantised so that nearby zoom levels share cache entries
    const step = Math.pow(2, Math.round(Math.log2(cam.zoom) * 4) / 4);
    const hw = (W / 2 / cam.zoom) * 1.3;
    const hh = (H / 2 / cam.zoom) * 1.3;
    const q = (Math.max(W, H) / step) * 0.2;
    const bbox = [Math.floor((cam.x - hw) / q) * q, Math.floor((cam.y - hh) / q) * q, Math.ceil((cam.x + hw) / q) * q, Math.ceil((cam.y + hh) / q) * q];
    const wantEdges = spacing() * step >= 12;
    const key = `${app.head}|${geomRev}|${step.toFixed(4)}|${bbox.map((v) => Math.round(v)).join(',')}|${wantEdges ? 1 : 0}`;
    if (key === shownKey) return;
    const hit = cache.get(key);
    if (hit) {
      cache.delete(key);
      cache.set(key, hit);
      ctrl?.abort();
      ctrl = null;
      show(hit, key);
      return;
    }
    ctrl?.abort();
    const mine = (ctrl = new AbortController());
    const t0 = performance.now();
    try {
      const res = await fetch('api/viewport', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ bbox, scale: step, cap: 2500, labels: 160, edges: wantEdges, edge_cap: 1500 }),
        signal: mine.signal,
      });
      const text = await res.text();
      if (!res.ok) throw new Error(JSON.parse(text)?.error ?? `The server answered ${res.status}.`);
      const vp = JSON.parse(text) as Viewport;
      stats.requests.push({ ms: performance.now() - t0, serverMs: vp.ms, bytes: text.length, level: vp.level, items: (vp.nodes ?? vp.cells ?? []).length });
      if (stats.requests.length > 400) stats.requests.shift();
      cache.set(key, vp);
      if (cache.size > 16) cache.delete(cache.keys().next().value!);
      if (mine.signal.aborted) return;
      if (ctrl === mine) ctrl = null;
      loadError = null;
      show(vp, key);
      if (vp.head !== app.head) app.loadStatus();
    } catch (e: any) {
      if (e?.name === 'AbortError') return;
      loadError = errText(e);
    }
  }
  function show(vp: Viewport, key: string) {
    shownKey = key;
    nodes = vp.nodes ?? [];
    nodeById = new Map(nodes.map((n) => [n.id, n]));
    cells = vp.cells ?? [];
    cellSize = vp.cell_size ?? 0;
    edges = vp.edges ?? [];
    domains = vp.regions.filter((r) => r.kind === 'domain');
    if (drag?.kind !== 'move') overrides.clear();
    level = vp.level;
    totalInView = vp.total_in_view;
    firstLoad = false;
    dirtyNodes = dirtyRegions = dirtyEdges = true;
    if (nodeRpx() >= DETAIL_PX) queueDetails();
    requestFrame();
  }
  function reload(geometryChanged = false) {
    if (geometryChanged) {
      geomRev++;
      cache.clear();
    }
    shownKey = '';
    loadViewport();
  }

  // ---- node records for the cells drawn at high zoom (bounded: only what is on screen, small cache)
  let detailBusy = 0;
  function queueDetails() {
    if (level !== 'nodes') return;
    const want: string[] = [];
    for (const n of nodes) {
      const p = posOf(n);
      const px = sx(p.x);
      const py = sy(p.y);
      if (px < -60 || py < -60 || px > W + 60 || py > H + 60) continue;
      if (!app.nodeCache.has(n.id)) want.push(n.id);
      if (want.length >= 40) break;
    }
    for (const id of want) fetchDetail(id);
  }
  const pending = new Set<string>();
  async function fetchDetail(id: string): Promise<NodeDetail | null> {
    const have = app.nodeCache.get(id);
    if (have) return have;
    if (pending.has(id) || detailBusy >= 6) return null;
    pending.add(id);
    detailBusy++;
    try {
      const n = await get<NodeDetail>(`api/nodes/${id}`);
      app.nodeCache.set(id, n);
      if (app.nodeCache.size > 300) app.nodeCache.delete(app.nodeCache.keys().next().value!);
      dirtyNodes = true;
      requestFrame();
      return n;
    } catch {
      return null;
    } finally {
      pending.delete(id);
      detailBusy--;
      if (detailBusy < 3 && nodeRpx() >= DETAIL_PX) queueDetails();
    }
  }

  // ---------------------------------------------------------------- geometry batches
  function buildRegions() {
    dirtyRegions = false;
    if (!renderer || !app.board) return;
    const b = new ShapeBatch(64);
    for (const m of app.board.mpns) {
      const off = m.enabled === false || m.installed === false;
      b.push(m.x, m.y, m.r, 0, 0, 0, C.regionFill, off ? 0x4a4a4aff : C.line, 1, 0, 0, 0, off ? 5 : 0);
    }
    renderer.setShapes('mpn', b);
    const d = new ShapeBatch(128);
    for (const r of domains) d.push(r.x, r.y, r.r, 0, 0, 0, 0, (r.depth ?? 1) <= 1 ? C.domain : C.domain2, 1);
    renderer.setShapes('domain', d);
    stats.regions = b.n + d.n;

    const lb = new LineBatch(256);
    const byId = new Map(app.board.mpns.map((m) => [m.id, m]));
    let max = 1;
    for (const bd of app.board.bundles) max = Math.max(max, bd.edges + bd.shared);
    for (const bd of app.board.bundles) {
      const a = byId.get(bd.a);
      const z = byId.get(bd.b);
      if (!a || !z) continue;
      const dx = z.x - a.x;
      const dy = z.y - a.y;
      const len = Math.hypot(dx, dy);
      if (len <= a.r + z.r + 1) continue;
      const t = (bd.edges + bd.shared) / max;
      const alpha = Math.round(255 * (0.05 + 0.4 * t * t));
      lb.push(a.x + (dx / len) * a.r, a.y + (dy / len) * a.r, z.x - (dx / len) * z.r, z.y - (dy / len) * z.r, 0.5 + 3.5 * t, (0xababab00 | alpha) >>> 0);
    }
    renderer.setLines('bundles', lb);
  }
  function bundleAlpha() {
    if (!showBundles || !app.board?.mpns.length) return 0;
    let r = 0;
    for (const m of app.board.mpns) r += m.r;
    const px = (r / app.board.mpns.length) * cam.zoom;
    return Math.max(0, Math.min(1, (420 - px) / 260));
  }
  function segsOf(id: string) {
    return app.nodeCache.get(id)?.schema.roles.length ?? 0;
  }
  function buildNodes() {
    dirtyNodes = false;
    if (!renderer) return;
    const sel = new Set(app.selection);
    const hl = new Set(app.highlight);
    const off = new Set((app.board?.mpns ?? []).filter((m) => m.enabled === false || m.installed === false).map((m) => m.id));
    const b = new ShapeBatch(Math.max(64, nodes.length));
    const detail = nodeRpx() >= DETAIL_PX;
    const R = nodeR();
    const add = (n: VNode, isSel: boolean) => {
      const p = posOf(n);
      const disabled = n.s === 'disabled';
      let fill = isSel ? C.ember : off.has(n.m) ? C.dim : C.white;
      let stroke = 0;
      let sw = 0;
      if (disabled) {
        fill = C.void;
        stroke = isSel ? C.ember : C.muted;
        sw = 1.5;
      }
      let ring = 0;
      let rw = 0;
      if (isSel || hl.has(n.id)) {
        ring = C.ember;
        rw = 1.5;
      } else if (n.learned === 'stale') {
        ring = C.gold;
        rw = 1.5;
      } else if (n.id === hoverId || n.id === connectSrc?.id) {
        ring = C.white;
        rw = 1;
      }
      b.push(p.x, p.y, R, 1.6, 0, n.t === 'procedure' ? 1 : 0, fill, stroke, sw, ring, rw, detail ? segsOf(n.id) : 0);
    };
    for (const n of nodes) if (!sel.has(n.id)) add(n, false);
    for (const n of nodes) if (sel.has(n.id)) add(n, true);
    renderer.setShapes('nodes', b);
    stats.nodes = b.n;

    const cb = new ShapeBatch(Math.max(64, cells.length));
    let max = 1;
    for (const c of cells) max = Math.max(max, c.n);
    for (const c of cells) cb.push(c.x, c.y, cellSize * 0.5 * Math.max(0.2, Math.sqrt(c.n / max)), 1.5, 0, 0, C.cell);
    renderer.setShapes('cells', cb);
    stats.cells = cb.n;
    stats.level = level ?? '';
  }
  function buildEdges() {
    dirtyEdges = false;
    if (!renderer) return;
    const sel = new Set(app.selection);
    const lb = new LineBatch(Math.max(64, edges.length));
    const selId = app.selectedEdge?.id;
    for (const e of edges) {
      const a = nodeById.get(e.s);
      const z = nodeById.get(e.d);
      if (!a || !z) continue;
      const p = posOf(a);
      const q = posOf(z);
      const isSel = e.id === selId;
      const touch = sel.has(e.s) || sel.has(e.d);
      lb.push(p.x, p.y, q.x, q.y, isSel ? 2 : touch ? 1.25 : 1, isSel ? C.ember : touch ? C.edgeNear : C.edge, 1);
    }
    renderer.setLines('edges', lb);
    stats.edges = lb.n;
  }

  // ---------------------------------------------------------------- labels (2D canvas, bounded)
  const widths = new Map<string, number>();
  function tw(font: string, s: string) {
    const k = font + '|' + s;
    let w = widths.get(k);
    if (w === undefined) {
      ctx!.font = font;
      w = ctx!.measureText(s).width;
      if (widths.size > 4000) widths.clear();
      widths.set(k, w);
    }
    return w;
  }
  function fitText(font: string, s: string, max: number) {
    if (tw(font, s) <= max) return s;
    let lo = 0;
    let hiN = s.length;
    while (lo < hiN) {
      const mid = (lo + hiN + 1) >> 1;
      if (tw(font, s.slice(0, mid) + '…') <= max) lo = mid;
      else hiN = mid - 1;
    }
    return lo > 0 ? s.slice(0, lo).trimEnd() + '…' : '';
  }
  function wrap2(font: string, s: string, max: number): string[] {
    if (tw(font, s) <= max) return [s];
    const words = s.split(' ');
    let a = '';
    let i = 0;
    for (; i < words.length; i++) {
      const t = a ? a + ' ' + words[i] : words[i];
      if (tw(font, t) > max && a) break;
      a = t;
    }
    const rest = words.slice(i).join(' ');
    return rest ? [a, fitText(font, rest, max)] : [fitText(font, a, max)];
  }
  // label placement: a coarse occupancy grid over the screen (6 px cells), reset every frame
  const OC = 6;
  let occ = new Uint8Array(1);
  let ocw = 1;
  let och = 1;
  function occReset() {
    ocw = Math.ceil(W / OC) + 1;
    och = Math.ceil(H / OC) + 1;
    if (occ.length !== ocw * och) occ = new Uint8Array(ocw * och);
    else occ.fill(0);
  }
  function free(x: number, y: number, w: number, h: number) {
    const x0 = Math.max(0, Math.floor(x / OC));
    const y0 = Math.max(0, Math.floor(y / OC));
    const x1 = Math.min(ocw - 1, Math.floor((x + w) / OC));
    const y1 = Math.min(och - 1, Math.floor((y + h) / OC));
    for (let j = y0; j <= y1; j++) for (let i = x0; i <= x1; i++) if (occ[j * ocw + i]) return false;
    return true;
  }
  function take(x: number, y: number, w: number, h: number) {
    const x0 = Math.max(0, Math.floor(x / OC));
    const y0 = Math.max(0, Math.floor(y / OC));
    const x1 = Math.min(ocw - 1, Math.floor((x + w) / OC));
    const y1 = Math.min(och - 1, Math.floor((y + h) / OC));
    for (let j = y0; j <= y1; j++) for (let i = x0; i <= x1; i++) occ[j * ocw + i] = 1;
  }
  function text(s: string, x: number, y: number, font: string, color: string, align: CanvasTextAlign = 'left', halo = true) {
    const c = ctx!;
    c.font = font;
    c.textAlign = align;
    if (halo) {
      c.strokeStyle = '#050505';
      c.lineWidth = 3.5;
      c.strokeText(s, x, y);
    }
    c.fillStyle = color;
    c.fillText(s, x, y);
  }

  function drawLabels() {
    const c = ctx;
    if (!c) return;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, W, H);
    c.textBaseline = 'middle';
    c.lineJoin = 'round';
    occReset();
    let drawn = 0;
    const F13 = `500 13px ${SANS}`;
    const F12 = `400 12px ${SANS}`;
    const F12M = `500 12px ${SANS}`;

    // MPN regions: name and real node count
    for (const m of app.board?.mpns ?? []) {
      const px = sx(m.x);
      const py = sy(m.y);
      const r = m.r * cam.zoom;
      if (px + r < 0 || py + r < 0 || px - r > W || py - r > H || r < 24) continue;
      const state = m.installed === false ? 'not installed' : m.enabled === false ? 'disabled' : '';
      const count = fmtInt(m.count) + (state ? ', ' + state : '');
      if (r < 150) {
        const font = r < 56 ? F12M : F13;
        const lines = wrap2(font, m.name, Math.max(60, Math.min(r * 1.75, 200)));
        const lh = 15;
        const top = py - ((lines.length + 1) * lh) / 2 + lh / 2;
        lines.forEach((ln, i) => text(ln, px, top + i * lh, font, '#f7f7f2', 'center'));
        text(count, px, top + lines.length * lh, F12, '#ababab', 'center');
        const wmax = Math.max(...lines.map((l) => tw(font, l)));
        take(px - wmax / 2 - 4, top - 9, wmax + 8, (lines.length + 1) * lh + 4);
        drawn++;
      } else {
        const cnt = count.replace(/^([\d,]+)/, '$1 nodes');
        const nameW = tw(F13, m.name);
        const total = nameW + 8 + tw(F12, cnt);
        const lx = Math.max(12 + total / 2, Math.min(W - 12 - total / 2, px));
        const ly = Math.max(54, py - r + 18);
        if (Math.hypot(lx - px, ly - py) > r - 10) continue;
        c.fillStyle = 'rgba(5,5,5,0.86)';
        c.fillRect(lx - total / 2 - 6, ly - 11, total + 12, 22);
        text(m.name, lx - total / 2, ly, F13, '#f7f7f2', 'left', false);
        text(cnt, lx - total / 2 + nameW + 8, ly, F12, '#ababab', 'left', false);
        take(lx - total / 2 - 6, ly - 11, total + 12, 22);
        drawn++;
      }
    }
    // domains
    for (const d of domains) {
      const r = d.r * cam.zoom;
      if (r < 46) continue;
      const px = sx(d.x);
      const py = sy(d.y) - r + 13;
      if (px < -100 || px > W + 100 || py < 40 || py > H) continue;
      const name = fitText(F12, d.name, Math.max(40, r * 1.5));
      const cnt = ' (' + fmtInt(d.count) + ')';
      const w = tw(F12, name) + tw(F12, cnt);
      if (!free(px - w / 2 - 3, py - 8, w + 6, 16)) continue;
      take(px - w / 2 - 3, py - 8, w + 6, 16);
      text(name, px - w / 2, py, F12, '#ababab', 'left');
      text(cnt, px - w / 2 + tw(F12, name), py, F12, '#8a8a8a', 'left');
      drawn++;
    }
    // nodes
    const rp = nodeRpx();
    if (level === 'nodes' && rp >= 2.4) {
      const detail = rp >= DETAIL_PX;
      const sel = new Set(app.selection);
      let budget = 180;
      if (!detail && nodes.length <= 900)
        for (const n of nodes) {
          const p = posOf(n);
          const px = sx(p.x);
          const py = sy(p.y);
          if (px > -rp && py > -rp && px < W + rp && py < H + rp) take(px - rp, py - rp, rp * 2, rp * 2);
        }
      for (const n of nodes) {
        if (budget <= 0) break;
        const p = posOf(n);
        const px = sx(p.x);
        const py = sy(p.y);
        if (px < -rp - 200 || py < -rp - 30 || px > W + rp || py > H + rp + 30) continue;
        const det = detail ? app.nodeCache.get(n.id) : undefined;
        const title = n.title ?? det?.title;
        if (detail && det) {
          const roles = det.schema.roles;
          const k = roles.length;
          const sq = n.t === 'procedure';
          const showCells = rp >= (k <= 3 ? 46 : 80);
          if (showCells) {
            const hollow = n.s === 'disabled';
            const col = hollow ? '#f7f7f2' : '#050505';
            const f = rp >= 70 ? F13 : F12M;
            for (let i = 0; i < k; i++) {
              let lx: number, ly: number, maxW: number;
              if (sq) {
                const rr = rp * 0.9;
                lx = px;
                ly = py - rr + ((i + 0.5) * 2 * rr) / k;
                maxW = rr * 1.8;
              } else {
                const ang = -Math.PI / 2 + ((i + 0.5) * 2 * Math.PI) / k;
                const rad = k === 1 ? 0 : rp * 0.58;
                lx = px + Math.cos(ang) * rad;
                ly = py + Math.sin(ang) * rad;
                maxW = k <= 3 ? rp * 0.95 : rp * 0.7;
              }
              const s = fitText(f, roles[i].label, maxW);
              if (s) text(s, lx, ly, f, col, 'center', false);
            }
          }
        }
        if (!title) continue;
        if (detail) {
          const s = fitText(F13, title, Math.max(120, rp * 2.6));
          const w = tw(F13, s);
          const ly = py + rp + 14;
          if (!free(px - w / 2 - 3, ly - 9, w + 6, 18)) continue;
          take(px - w / 2 - 3, ly - 9, w + 6, 18);
          text(s, px, ly, F13, '#f7f7f2', 'center');
        } else {
          const isSel = sel.has(n.id);
          const f = isSel ? F12M : F12;
          const s = fitText(f, title, 170);
          const w = tw(f, s);
          const lx = px + rp + 6;
          if (!isSel && !free(lx, py - 6, w + 2, 12)) continue;
          take(lx - 2, py - 7, w + 6, 14);
          text(s, lx, py, f, isSel ? '#f7f7f2' : '#d6d6d2', 'left');
        }
        drawn++;
        budget--;
      }
      // relation type on the selected edge and on edges of the selection, when there is room to read it
      if (detail || app.selectedEdge) {
        const FM = `400 11px ${MONO}`;
        let n = 0;
        for (const e of edges) {
          const on = e.id === app.selectedEdge?.id || (detail && (sel.has(e.s) || sel.has(e.d)));
          if (!on) continue;
          const a = nodeById.get(e.s);
          const z = nodeById.get(e.d);
          if (!a || !z) continue;
          const p = posOf(a);
          const q = posOf(z);
          const mx = sx((p.x + q.x) / 2);
          const my = sy((p.y + q.y) / 2);
          if (mx < 0 || my < 0 || mx > W || my > H) continue;
          text(e.t, mx, my, FM, e.id === app.selectedEdge?.id ? '#f7f7f2' : '#ababab', 'center');
          drawn++;
          if (++n > 24) break;
        }
      }
    }
    // connect rubber line
    if (drag?.kind === 'connect' || (connectSrc && pointer)) {
      const src = drag?.kind === 'connect' ? drag.src : connectSrc!;
      const p = posOf(src);
      const tx = drag?.kind === 'connect' ? drag.x : pointer!.x;
      const ty = drag?.kind === 'connect' ? drag.y : pointer!.y;
      c.strokeStyle = '#f34b27';
      c.lineWidth = 1.5;
      c.setLineDash([5, 4]);
      c.beginPath();
      c.moveTo(sx(p.x), sy(p.y));
      c.lineTo(tx, ty);
      c.stroke();
      c.setLineDash([]);
    }
    stats.labels = drawn;
  }

  // ---------------------------------------------------------------- hit testing
  function hitNode(px: number, py: number): VNode | null {
    if (level !== 'nodes') return null;
    const rp = Math.max(nodeRpx(), 5) + 2;
    let best: VNode | null = null;
    let bd = rp * rp;
    for (const n of nodes) {
      const p = posOf(n);
      const dx = sx(p.x) - px;
      const dy = sy(p.y) - py;
      const d = dx * dx + dy * dy;
      if (d <= bd) {
        bd = d;
        best = n;
      }
    }
    return best;
  }
  function hitEdge(px: number, py: number): VEdge | null {
    let best: VEdge | null = null;
    let bd = 5;
    for (const e of edges) {
      const a = nodeById.get(e.s);
      const z = nodeById.get(e.d);
      if (!a || !z) continue;
      const p = posOf(a);
      const q = posOf(z);
      const x0 = sx(p.x),
        y0 = sy(p.y),
        x1 = sx(q.x),
        y1 = sy(q.y);
      const dx = x1 - x0,
        dy = y1 - y0;
      const l2 = dx * dx + dy * dy;
      if (l2 < 1) continue;
      const t = Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / l2));
      const d = Math.hypot(px - (x0 + t * dx), py - (y0 + t * dy));
      if (d < bd) {
        bd = d;
        best = e;
      }
    }
    return best;
  }

  // ---------------------------------------------------------------- pointer interaction
  type Drag =
    | { kind: 'pan'; px: number; py: number; cx: number; cy: number; moved: boolean; clear: boolean }
    | { kind: 'band'; x0: number; y0: number; add: boolean }
    | { kind: 'move'; ids: string[]; wx0: number; wy0: number; orig: Map<string, { x: number; y: number }>; moved: boolean; clickId: string; shift: boolean; px: number; py: number }
    | { kind: 'connect'; src: VNode; x: number; y: number; moved: boolean };
  let drag: Drag | null = null;
  let pointer: { x: number; y: number } | null = null;
  let spaceDown = false;
  const touches = new Map<number, { x: number; y: number }>();
  let pinch: { d: number; zoom: number } | null = null;

  function local(e: { clientX: number; clientY: number }) {
    const r = host.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }
  function onPointerDown(e: PointerEvent) {
    host.focus({ preventScroll: true });
    app.menu = null;
    selectedObject = null;
    const p = local(e);
    if (e.pointerType === 'touch') {
      touches.set(e.pointerId, p);
      if (touches.size === 2) {
        const [a, b] = [...touches.values()];
        pinch = { d: Math.hypot(a.x - b.x, a.y - b.y), zoom: cam.zoom };
        drag = null;
        return;
      }
    }
    if (e.button === 2) return;
    textCanvas.setPointerCapture(e.pointerId);
    anim = null;
    zoomTo = null;
    const pan = (clear: boolean): Drag => ({ kind: 'pan', px: p.x, py: p.y, cx: cam.x, cy: cam.y, moved: false, clear });
    if (e.button === 1 || spaceDown) {
      drag = pan(false);
      cursor = 'grabbing';
      return;
    }
    const hit = hitNode(p.x, p.y);
    if (app.tool === 'connect') {
      if (hit) {
        if (connectSrc && connectSrc.id !== hit.id) {
          askConnect(connectSrc, hit);
          connectSrc = null;
          return;
        }
        drag = { kind: 'connect', src: hit, x: p.x, y: p.y, moved: false };
        dirtyNodes = true;
      } else drag = pan(false);
      requestFrame();
      return;
    }
    if (hit) {
      if (e.shiftKey) {
        app.selection = app.selection.includes(hit.id) ? app.selection.filter((i) => i !== hit.id) : [...app.selection, hit.id];
        app.selectedEdge = null;
      } else if (!app.selection.includes(hit.id)) {
        app.select([hit.id], false);
      }
      const ids = app.selection.filter((id) => nodeById.has(id));
      const orig = new Map(ids.map((id) => [id, { ...posOf(nodeById.get(id)!) }]));
      drag = { kind: 'move', ids, wx0: wx(p.x), wy0: wy(p.y), orig, moved: false, clickId: hit.id, shift: e.shiftKey, px: p.x, py: p.y };
      return;
    }
    const eh = hitEdge(p.x, p.y);
    if (eh) {
      app.selection = [];
      app.selectedEdge = eh;
      dirtyEdges = true;
      requestFrame();
      drag = pan(false);
      return;
    }
    if (app.tool === 'select' || e.shiftKey) {
      drag = { kind: 'band', x0: p.x, y0: p.y, add: e.shiftKey };
      band = { x: p.x, y: p.y, w: 0, h: 0 };
    } else {
      drag = pan(true);
      cursor = 'grabbing';
    }
  }
  function onPointerMove(e: PointerEvent) {
    const p = local(e);
    pointer = p;
    if (e.pointerType === 'touch' && touches.has(e.pointerId)) {
      touches.set(e.pointerId, p);
      if (pinch && touches.size === 2) {
        const [a, b] = [...touches.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        const mx = (a.x + b.x) / 2;
        const my = (a.y + b.y) / 2;
        zoomAt(mx, my, (pinch.zoom * (d / pinch.d)) / cam.zoom, false);
        return;
      }
    }
    if (!drag) {
      const hit = hitNode(p.x, p.y);
      const id = hit?.id ?? null;
      if (id !== hoverId) {
        hoverId = id;
        dirtyNodes = true;
        requestFrame();
        hoverTip(hit, p);
      }
      cursor = hit ? 'pointer' : app.tool === 'pan' ? (spaceDown ? 'grab' : 'grab') : 'crosshair';
      if (connectSrc) requestFrame();
      return;
    }
    if (drag.kind === 'pan') {
      const dx = p.x - drag.px;
      const dy = p.y - drag.py;
      if (!drag.moved && Math.hypot(dx, dy) < 4) return;
      drag.moved = true;
      cam.x = drag.cx - dx / cam.zoom;
      cam.y = drag.cy - dy / cam.zoom;
      cameraChanged();
      requestFrame();
    } else if (drag.kind === 'band') {
      band = { x: Math.min(drag.x0, p.x), y: Math.min(drag.y0, p.y), w: Math.abs(p.x - drag.x0), h: Math.abs(p.y - drag.y0) };
    } else if (drag.kind === 'move') {
      if (!drag.moved && Math.hypot(p.x - drag.px, p.y - drag.py) < 4) return;
      if (!drag.moved) {
        drag.moved = true;
        app.hint = 'Moving changes the board layout only. The model is not changed.';
        cursor = 'grabbing';
      }
      const dx = wx(p.x) - drag.wx0;
      const dy = wy(p.y) - drag.wy0;
      for (const [id, o] of drag.orig) overrides.set(id, { x: o.x + dx, y: o.y + dy });
      dirtyNodes = dirtyEdges = true;
      requestFrame();
    } else if (drag.kind === 'connect') {
      drag.x = p.x;
      drag.y = p.y;
      if (Math.hypot(p.x - sx(posOf(drag.src).x), p.y - sy(posOf(drag.src).y)) > 6) drag.moved = true;
      requestFrame();
    }
  }
  function onPointerUp(e: PointerEvent) {
    touches.delete(e.pointerId);
    if (touches.size < 2) pinch = null;
    const d = drag;
    drag = null;
    if (!d) return;
    const p = local(e);
    cursor = app.tool === 'pan' ? 'grab' : 'crosshair';
    if (d.kind === 'pan') {
      if (!d.moved && d.clear) {
        if (app.selection.length || app.selectedEdge) {
          app.selection = [];
          app.selectedEdge = null;
        }
      }
    } else if (d.kind === 'band') {
      const b = band;
      band = null;
      if (b && (b.w > 3 || b.h > 3)) {
        if (level !== 'nodes') {
          app.toast('Zoom in until single nodes are drawn, then select them.');
          return;
        }
        const ids: string[] = [];
        for (const n of nodes) {
          const q = posOf(n);
          const x = sx(q.x);
          const y = sy(q.y);
          if (x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) ids.push(n.id);
        }
        app.select(d.add ? [...new Set([...app.selection, ...ids])] : ids, true);
      } else if (!d.add) {
        app.selection = [];
        app.selectedEdge = null;
      }
    } else if (d.kind === 'move') {
      app.hint = null;
      if (!d.moved) {
        if (!d.shift) app.select([d.clickId]);
        return;
      }
      const moves = d.ids.map((id) => ({ id, ...overrides.get(id)! }));
      commitMove(moves);
    } else if (d.kind === 'connect') {
      const hit = hitNode(p.x, p.y);
      if (hit && hit.id !== d.src.id) {
        connectSrc = null;
        askConnect(d.src, hit);
      } else if (!d.moved) {
        connectSrc = d.src;
        app.hint = 'Click the node to connect to. Escape cancels.';
      }
      dirtyNodes = true;
      requestFrame();
    }
  }
  async function commitMove(moves: { id: string; x: number; y: number }[]) {
    try {
      const r = await post('api/layout/move', { moves });
      for (const m of moves) {
        const n = nodeById.get(m.id);
        if (n) {
          n.x = m.x;
          n.y = m.y;
          n.pin = 1;
        }
      }
      geomRev++;
      cache.clear();
      shownKey = '';
      app.nodeCache.clear();
      app.toast(`Moved ${r.moved === 1 ? '1 node' : r.moved + ' nodes'}. ${r.note}`);
    } catch (e) {
      overrides.clear();
      dirtyNodes = dirtyEdges = true;
      requestFrame();
      app.toast(errText(e), 'error');
    }
  }
  function onWheel(e: WheelEvent) {
    e.preventDefault();
    const p = local(e);
    if (e.ctrlKey) {
      // touchpad pinch arrives as ctrl+wheel with small deltas
      zoomAt(p.x, p.y, Math.exp(-e.deltaY * 0.012), false);
    } else {
      const unit = e.deltaMode === 1 ? 33 : 1;
      zoomAt(p.x, p.y, Math.exp(-e.deltaY * unit * 0.0022), true);
    }
  }
  function onDblClick(e: MouseEvent) {
    const p = local(e);
    if (hitNode(p.x, p.y)) return;
    zoomAt(p.x, p.y, 2, true);
  }

  let tipTimer: ReturnType<typeof setTimeout> | null = null;
  function hoverTip(n: VNode | null, p: { x: number; y: number }) {
    if (tipTimer) clearTimeout(tipTimer);
    tip = null;
    if (!n || n.title || nodeRpx() >= DETAIL_PX) return;
    tipTimer = setTimeout(async () => {
      const d = await fetchDetail(n.id);
      if (d && hoverId === n.id) tip = { x: p.x + 12, y: p.y + 14, text: d.title };
    }, 220);
  }

  // ---------------------------------------------------------------- actions
  function titleOf(n: VNode) {
    return n.title ?? app.nodeCache.get(n.id)?.title ?? `node ${n.id}`;
  }
  function askConnect(a: VNode, b: VNode) {
    app.hint = null;
    app.dialog = { kind: 'connect', src: a.id, dst: b.id, srcTitle: titleOf(a), dstTitle: titleOf(b) };
  }
  function startConnect(src?: string) {
    app.tool = 'connect';
    connectSrc = src ? (nodeById.get(src) ?? null) : null;
    app.hint = connectSrc ? 'Click the node to connect to. Escape cancels.' : 'Drag from one node to another, or click the two nodes in turn.';
    dirtyNodes = true;
    requestFrame();
  }
  async function setStatus(ids: string[], status: 'live' | 'disabled') {
    try {
      let last: any;
      for (const id of ids) last = await patch(`api/nodes/${id}`, { status });
      await app.afterEdit(last);
      app.toast(`${ids.length === 1 ? 'Node' : ids.length + ' nodes'} ${status === 'live' ? 'enabled' : 'disabled'}.`);
    } catch (e) {
      app.toast(errText(e), 'error');
    }
  }
  async function duplicate(id: string, mode: 'canonical' | 'reference', position?: { x: number; y: number }) {
    try {
      const r = await post(`api/nodes/${id}/duplicate`, { mode, ...(position ? { position } : {}) });
      if (mode === 'reference') {
        await app.loadBoard();
        app.toast(r.note);
      } else {
        await app.afterEdit(r);
        app.select([r.id]);
        app.toast(r.note);
      }
    } catch (e) {
      app.toast(errText(e), 'error');
    }
  }
  async function paste(at: { x: number; y: number }) {
    const ids = app.clipboard;
    let i = 0;
    const made: string[] = [];
    try {
      let last: any;
      for (const id of ids) {
        last = await post(`api/nodes/${id}/duplicate`, { mode: 'canonical', position: { x: at.x + i * spacing(), y: at.y } });
        made.push(last.id);
        i++;
      }
      await app.afterEdit(last);
      app.select(made);
      app.toast(`Pasted ${made.length === 1 ? '1 node' : made.length + ' nodes'} as new ${made.length === 1 ? 'node' : 'nodes'}.`);
    } catch (e) {
      app.toast(errText(e), 'error');
    }
  }
  async function disconnect(e: VEdge) {
    try {
      const r = await del(`api/edges/${e.id}`);
      app.selectedEdge = null;
      await app.afterEdit(r);
      app.toast('Disconnected.');
    } catch (err) {
      app.toast(errText(err), 'error');
    }
  }

  function onContextMenu(e: MouseEvent) {
    e.preventDefault();
    const p = local(e);
    const at = { x: wx(p.x), y: wy(p.y) };
    const hit = hitNode(p.x, p.y);
    let items: (MenuItem | 'sep')[];
    if (hit) {
      if (!app.selection.includes(hit.id)) app.select([hit.id], false);
      const ids = app.selection.length > 1 ? app.selection : [hit.id];
      const many = ids.length > 1;
      items = [
        { label: 'Open', run: () => app.openNode(hit.id) },
        { label: 'Connect to…', kbd: 'C', run: () => startConnect(hit.id) },
        { label: 'Copy', kbd: 'Ctrl C', run: () => copy(ids) },
        { label: 'Duplicate as new node', run: () => duplicate(hit.id, 'canonical') },
        { label: 'Add reference here', run: () => duplicate(hit.id, 'reference') },
        hit.s === 'disabled' ? { label: many ? `Enable ${ids.length} nodes` : 'Enable', run: () => setStatus(ids, 'live') } : { label: many ? `Disable ${ids.length} nodes` : 'Disable', run: () => setStatus(ids, 'disabled') },
        'sep',
        { label: many ? `Delete ${ids.length} nodes…` : 'Delete…', kbd: 'Del', danger: true, run: () => (app.dialog = { kind: 'delete', ids }) },
      ];
    } else {
      const eh = hitEdge(p.x, p.y);
      if (eh) {
        app.selection = [];
        app.selectedEdge = eh;
        items = [{ label: 'Disconnect', kbd: 'Del', danger: true, run: () => disconnect(eh) }];
      } else {
        const n = app.clipboard.length;
        items = [
          { label: 'New node here…', kbd: 'N', run: () => (app.dialog = { kind: 'newNode', x: at.x, y: at.y }) },
          { label: 'New MPN…', run: () => (app.dialog = { kind: 'newMpn' }) },
          n ? { label: n === 1 ? 'Paste' : `Paste ${n} nodes`, kbd: 'Ctrl V', run: () => paste(at) } : { label: 'Paste', disabled: true, why: 'Nothing copied' },
        ];
      }
    }
    app.openMenu(e.clientX, e.clientY, items);
  }
  function copy(ids: string[]) {
    app.clipboard = ids;
    app.toast(`Copied ${ids.length === 1 ? '1 node' : ids.length + ' nodes'}. Paste makes new nodes.`);
  }

  // ---------------------------------------------------------------- keyboard
  function editable(t: EventTarget | null) {
    const el = t as HTMLElement | null;
    return !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable);
  }
  function onKeyDown(e: KeyboardEvent) {
    if (editable(e.target) || app.dialog || app.searchOpen) return;
    const el = e.target as HTMLElement;
    // letters, arrows and Delete act on the board only while focus is on it (or nowhere in particular)
    const onBoard = el === document.body || !!el.closest?.('.board');
    const k = e.key;
    if (e.altKey && k === 'ArrowLeft') {
      e.preventDefault();
      return back();
    }
    if (e.altKey && k === 'ArrowRight') {
      e.preventDefault();
      return forward();
    }
    if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === 'c' && app.selection.length && !window.getSelection()?.toString()) {
      e.preventDefault();
      return copy(app.selection);
    }
    if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === 'v' && app.clipboard.length) {
      e.preventDefault();
      return void paste(pointer ? { x: wx(pointer.x), y: wy(pointer.y) } : { x: cam.x, y: cam.y });
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (k === 'Escape') {
      if (app.menu) app.menu = null;
      else if (connectSrc || app.tool === 'connect') {
        connectSrc = null;
        app.tool = 'pan';
        app.hint = null;
      } else if (selectedObject) selectedObject = null;
      else {
        app.selection = [];
        app.selectedEdge = null;
      }
      dirtyNodes = true;
      requestFrame();
      return;
    }
    if (!onBoard) return;
    if (k === ' ') {
      if (el.tagName === 'BUTTON') return;
      spaceDown = true;
      e.preventDefault();
      return;
    }
    const step = 90 / cam.zoom;
    if (k === 'ArrowLeft') setCam(cam.x - step, cam.y, cam.zoom);
    else if (k === 'ArrowRight') setCam(cam.x + step, cam.y, cam.zoom);
    else if (k === 'ArrowUp') setCam(cam.x, cam.y - step, cam.zoom);
    else if (k === 'ArrowDown') setCam(cam.x, cam.y + step, cam.zoom);
    else if (k === '+' || k === '=') zoomAt(W / 2, H / 2, 1.3, true);
    else if (k === '-' || k === '_') zoomAt(W / 2, H / 2, 1 / 1.3, true);
    else if (k === '0') fitAll();
    else if (k === 'f' || k === 'F') fitSelection();
    else if (k === 'n' || k === 'N') app.dialog = { kind: 'newNode', x: pointer ? wx(pointer.x) : cam.x, y: pointer ? wy(pointer.y) : cam.y };
    else if (k === 'c' || k === 'C') {
      if (app.tool === 'connect') {
        app.tool = 'pan';
        connectSrc = null;
        app.hint = null;
      } else startConnect(app.selection.length === 1 ? app.selection[0] : undefined);
    } else if (k === 'v' || k === 'V') app.tool = 'pan';
    else if (k === 's' || k === 'S') app.tool = 'select';
    else if (k === 'Delete' || k === 'Backspace') {
      if (selectedObject) removeObject(selectedObject);
      else if (app.selectedEdge) disconnect(app.selectedEdge);
      else if (app.selection.length) app.dialog = { kind: 'delete', ids: app.selection };
    } else return;
    e.preventDefault();
  }
  function onKeyUp(e: KeyboardEvent) {
    if (e.key === ' ') spaceDown = false;
  }

  // ---------------------------------------------------------------- board objects (groups, notes, references)
  async function addObject(kind: 'note' | 'group') {
    const z = cam.zoom;
    let body: any;
    if (kind === 'group' && app.selection.length) {
      const pts = app.selection.map((id) => nodeById.get(id)).filter(Boolean) as VNode[];
      if (pts.length) {
        const xs = pts.map((n) => posOf(n).x);
        const ys = pts.map((n) => posOf(n).y);
        const m = spacing();
        body = { kind, x: Math.min(...xs) - m, y: Math.min(...ys) - m, w: Math.max(...xs) - Math.min(...xs) + 2 * m, h: Math.max(...ys) - Math.min(...ys) + 2 * m, props: { label: 'Group' } };
      }
    }
    if (!body) {
      const w = (kind === 'note' ? 220 : 320) / z;
      const h = (kind === 'note' ? 130 : 220) / z;
      body = { kind, x: cam.x - w / 2, y: cam.y - h / 2, w, h, props: kind === 'note' ? { text: '', s: 1 / z } : { label: 'Group' } };
    }
    try {
      const r = await post('api/board/objects', body);
      objects = [...objects, { id: r.id, ...body }];
      selectedObject = r.id;
    } catch (e) {
      app.toast(errText(e), 'error');
    }
  }
  let patchTimers = new Map<string, ReturnType<typeof setTimeout>>();
  function changeObject(o: BoardObject, final: boolean) {
    objects = objects.map((x) => (x.id === o.id ? o : x));
    const prev = patchTimers.get(o.id);
    if (prev) clearTimeout(prev);
    const send = () => {
      patchTimers.delete(o.id);
      patch(`api/board/objects/${o.id}`, { x: o.x, y: o.y, w: o.w, h: o.h, props: o.props }).catch((e) => app.toast(errText(e), 'error'));
    };
    if (final) send();
    else patchTimers.set(o.id, setTimeout(send, 400));
  }
  async function removeObject(id: string) {
    try {
      await del(`api/board/objects/${id}`);
      objects = objects.filter((o) => o.id !== id);
      if (selectedObject === id) selectedObject = null;
    } catch (e) {
      app.toast(errText(e), 'error');
    }
  }
  function objectMenu(o: BoardObject, e: MouseEvent) {
    const items: (MenuItem | 'sep')[] = [];
    if (o.kind === 'reference') {
      items.push({ label: 'Open', run: () => app.openNode(o.props.node_id) }, { label: 'Go to the node', run: () => app.goToNode(o.props.node_id) }, 'sep', { label: 'Remove reference', danger: true, run: () => removeObject(o.id) });
    } else {
      items.push({ label: o.kind === 'note' ? 'Delete note' : 'Delete group', kbd: 'Del', danger: true, run: () => removeObject(o.id) });
    }
    app.openMenu(e.clientX, e.clientY, items);
  }

  // ---------------------------------------------------------------- saved views
  async function saveView() {
    const name = viewName.trim();
    if (!name) return;
    viewBusy = true;
    try {
      await post('api/board/views', { name, x: cam.x, y: cam.y, zoom: cam.zoom });
      viewName = '';
      await app.loadBoard();
    } catch (e) {
      app.toast(errText(e), 'error');
    }
    viewBusy = false;
  }
  async function deleteView(id: string) {
    try {
      await del(`api/board/views/${id}`);
      await app.loadBoard();
    } catch (e) {
      app.toast(errText(e), 'error');
    }
  }

  // ---------------------------------------------------------------- lifecycle
  function resize() {
    const r = host.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return;
    W = r.width;
    H = r.height;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    renderer?.resize(W, H, dpr);
    textCanvas.width = Math.round(W * dpr);
    textCanvas.height = Math.round(H * dpr);
    if (!viewReady && app.board) initView();
    else if (viewReady) {
      scheduleLoad(60);
      requestFrame();
    }
  }
  function initView() {
    const v = app.board!.view;
    const ok = typeof v?.x === 'number' && typeof v?.y === 'number' && typeof v?.zoom === 'number' && v.zoom > 0;
    const start = ok ? { x: v.x!, y: v.y!, zoom: v.zoom! } : fitAllView();
    cam.x = start.x;
    cam.y = start.y;
    cam.zoom = clampZoom(start.zoom);
    app.zoom = cam.zoom;
    viewReady = true;
    pushHistory({ ...cam });
    loadViewport();
    requestFrame();
  }

  onMount(() => {
    ctx = textCanvas.getContext('2d');
    try {
      renderer = new Renderer(glCanvas);
    } catch (e) {
      glError = errText(e);
    }
    const lost = (e: Event) => e.preventDefault();
    const restored = () => {
      renderer?.init();
      dirtyNodes = dirtyRegions = dirtyEdges = true;
      requestFrame();
    };
    glCanvas.addEventListener('webglcontextlost', lost);
    glCanvas.addEventListener('webglcontextrestored', restored);
    const ro = new ResizeObserver(resize);
    ro.observe(host);
    resize();
    document.fonts?.ready.then(() => {
      widths.clear();
      requestFrame();
    });
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    app.boardApi = {
      jumpTo: (x, y, zoom) => jump({ x, y, zoom: zoom ?? cam.zoom }),
      fitAll,
      fitSelection,
      fitRegion,
      reload,
      view: () => ({ ...cam }),
      center: () => ({ x: cam.x, y: cam.y }),
      startConnect,
    };
    // test and measurement hook (read-only view of what is drawn)
    (window as any).__mpn = {
      stats,
      cam,
      size: () => ({ w: W, h: H }),
      toScreen: (x: number, y: number) => {
        const r = host.getBoundingClientRect();
        return { x: sx(x) + r.left, y: sy(y) + r.top };
      },
      nodes: () => nodes.map((n) => ({ ...n, ...posOf(n) })),
      selection: () => app.selection,
      edges: () => edges,
      regions: () => app.board?.mpns ?? [],
      level: () => level,
      setView: (x: number, y: number, zoom: number) => setCam(x, y, zoom),
      idle: () => !raf && !anim && !zoomTo && !ctrl && !loadTimer,
    };
    return () => {
      ro.disconnect();
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      glCanvas.removeEventListener('webglcontextlost', lost);
      glCanvas.removeEventListener('webglcontextrestored', restored);
      app.boardApi = null;
    };
  });

  // board data arrived or changed
  $effect(() => {
    const b = app.board;
    if (!b) return;
    untrack(() => {
      objects = b.objects;
      dirtyRegions = true;
      if (!viewReady && W > 2) initView();
      requestFrame();
    });
  });
  // model revision changed: what is drawn may be out of date
  $effect(() => {
    app.head;
    untrack(() => {
      if (viewReady) reload();
    });
  });
  // selection, run highlight
  $effect(() => {
    app.selection;
    app.highlight;
    app.selectedEdge;
    untrack(() => {
      dirtyNodes = dirtyEdges = true;
      requestFrame();
    });
  });
  $effect(() => {
    const t = app.tool;
    untrack(() => {
      cursor = t === 'pan' ? 'grab' : 'crosshair';
      if (t !== 'connect' && connectSrc) {
        connectSrc = null;
        app.hint = null;
        dirtyNodes = true;
        requestFrame();
      }
    });
  });
  $effect(() => {
    showBundles;
    untrack(requestFrame);
  });

  const zoomPct = $derived(Math.round(app.zoom * 100));
  const empty = $derived(!!app.board && !firstLoad && totalInView === 0 && (app.status?.counts?.canonical_live_nodes ?? 1) === 0);
</script>

<div class="board" bind:this={host} tabindex="0" role="application" aria-label="Board: {app.board?.name ?? 'model network'}" style:cursor>
  <canvas class="gl" bind:this={glCanvas}></canvas>
  <canvas class="text" bind:this={textCanvas} onpointerdown={onPointerDown} onpointermove={onPointerMove} onpointerup={onPointerUp} onpointercancel={onPointerUp} onwheel={onWheel} ondblclick={onDblClick} oncontextmenu={onContextMenu} onpointerleave={() => ((pointer = null), (tip = null))}></canvas>

  <div class="world" bind:this={worldLayer} onwheel={onWheel}>
    <BoardObjects {objects} zoom={() => cam.zoom} radius={nodeR()} bind:selected={selectedObject} onchange={changeObject} onmenu={objectMenu} onopen={(id) => app.openNode(id)} />
  </div>

  {#if band}
    <div class="band" style="left:{band.x}px;top:{band.y}px;width:{band.w}px;height:{band.h}px"></div>
  {/if}
  {#if tip}
    <div class="tip" style="left:{tip.x}px;top:{tip.y}px">{tip.text}</div>
  {/if}

  <div class="tools" role="toolbar" aria-label="Board tools">
    <div class="seg" role="radiogroup" aria-label="Pointer tool">
      <button role="radio" aria-checked={app.tool === 'pan'} title="Move the board (V). Shift-drag selects." onclick={() => (app.tool = 'pan')}><Icon name="pan" /><span>Move</span></button>
      <button role="radio" aria-checked={app.tool === 'select'} title="Drag a box to select nodes (S)" onclick={() => (app.tool = 'select')}><Icon name="select" /><span>Select</span></button>
      <button role="radio" aria-checked={app.tool === 'connect'} title="Connect two nodes (C)" onclick={() => (app.tool === 'connect' ? (app.tool = 'pan') : startConnect())}><Icon name="connect" /><span>Connect</span></button>
    </div>
    <button class="btn" title="Add a note at the centre of the view" onclick={() => addObject('note')}><Icon name="note" />Note</button>
    <button class="btn" title={app.selection.length ? 'Draw a group around the selection' : 'Add a group at the centre of the view'} onclick={() => addObject('group')}><Icon name="group" />Group</button>
    <div class="anchor" use:clickOutside={() => (keyOpen ? ((keyOpen = false), true) : false)}>
      <button class="btn" aria-expanded={keyOpen} onclick={() => (keyOpen = !keyOpen)}>Key</button>
      {#if keyOpen}
        <div class="pop key">
          <div class="k"><svg width="16" height="16"><circle cx="8" cy="8" r="5" fill="#f7f7f2" /></svg>Knowledge node</div>
          <div class="k"><svg width="16" height="16"><rect x="3.5" y="3.5" width="9" height="9" fill="#f7f7f2" /></svg>Procedure node</div>
          <div class="k"><svg width="16" height="16"><circle cx="8" cy="8" r="4.5" fill="none" stroke="#ababab" stroke-width="1.5" /></svg>Disabled node (hollow)</div>
          <div class="k"><svg width="16" height="16"><circle cx="8" cy="8" r="3.5" fill="#f7f7f2" /><circle cx="8" cy="8" r="6.5" fill="none" stroke="#f7b547" stroke-width="1.5" /></svg>Trained state out of date (ring)</div>
          <div class="k"><svg width="16" height="16"><circle cx="8" cy="8" r="3.5" fill="#f34b27" /><circle cx="8" cy="8" r="6.5" fill="none" stroke="#f34b27" stroke-width="1.5" /></svg>Selected</div>
          <div class="k"><svg width="16" height="16"><circle cx="8" cy="8" r="3.5" fill="#f7f7f2" /><circle cx="8" cy="8" r="6.5" fill="none" stroke="#f34b27" stroke-width="1.5" /></svg>Read in the last run (ring)</div>
          <div class="k"><svg width="16" height="16"><circle cx="8" cy="8" r="6.5" fill="none" stroke="#6e6e6e" stroke-width="1" stroke-dasharray="3 2.5" /></svg>Disabled MPN (dashed)</div>
          <label class="k chk"><input type="checkbox" bind:checked={showBundles} />Show relations between MPNs</label>
          <p class="muted">Line width between two MPNs is their number of edges plus shared nodes.</p>
        </div>
      {/if}
    </div>
  </div>

  <div class="nav">
    <Minimap bind:this={minimap} mpns={app.board?.mpns ?? []} bounds={app.board?.bounds ?? null} onmove={(x, y) => setCam(x, y, cam.zoom)} />
    <div class="navrow" role="toolbar" aria-label="View">
      <button class="btn icon" aria-label="Back to the previous view" title="Back (Alt+Left)" disabled={!canBack} onclick={back}><Icon name="back" /></button>
      <button class="btn icon" aria-label="Forward to the next view" title="Forward (Alt+Right)" disabled={!canFwd} onclick={forward}><Icon name="forward" /></button>
      <span class="gap"></span>
      <button class="btn icon" aria-label="Zoom out" title="Zoom out (-)" onclick={() => zoomAt(W / 2, H / 2, 1 / 1.3, true)}><Icon name="minus" /></button>
      <span class="pct num" title="Zoom">{zoomPct}%</span>
      <button class="btn icon" aria-label="Zoom in" title="Zoom in (+)" onclick={() => zoomAt(W / 2, H / 2, 1.3, true)}><Icon name="plus" /></button>
    </div>
    <div class="navrow" role="toolbar" aria-label="Fit and saved views">
      <button class="btn" title="Fit the whole network (0)" onclick={fitAll}>Fit all</button>
      <button class="btn" title="Fit the selected nodes (F)" disabled={!app.selection.length} onclick={fitSelection}>Fit selection</button>
      <div class="anchor" use:clickOutside={() => (viewsOpen ? ((viewsOpen = false), true) : false)}>
        <button class="btn icon" aria-label="Saved views" title="Saved views" aria-expanded={viewsOpen} onclick={() => (viewsOpen = !viewsOpen)}><Icon name="views" /></button>
        {#if viewsOpen}
          <div class="pop views">
            {#if app.board?.saved_views.length}
              <ul>
                {#each app.board.saved_views as v (v.id)}
                  <li>
                    <button class="jump ellipsis" onclick={() => (jump({ x: v.x, y: v.y, zoom: v.zoom }), (viewsOpen = false))}>{v.name}</button>
                    <button class="btn icon quiet" aria-label="Delete the saved view {v.name}" title="Delete this saved view" onclick={() => deleteView(v.id)}><Icon name="close" size={14} /></button>
                  </li>
                {/each}
              </ul>
            {:else}
              <p class="muted none">No saved views yet.</p>
            {/if}
            <form class="row" onsubmit={(e) => (e.preventDefault(), saveView())}>
              <input class="field" placeholder="Name this view" aria-label="Name for the current view" bind:value={viewName} maxlength="80" />
              <button class="btn" type="submit" disabled={!viewName.trim() || viewBusy}>Save</button>
            </form>
          </div>
        {/if}
      </div>
    </div>
  </div>

  {#if glError}
    <div class="state"><p>The board cannot be drawn: {glError}</p></div>
  {:else if app.boardError && !app.board}
    <div class="state">
      <p>The board could not be loaded. {app.boardError}</p>
      <button class="btn" onclick={() => app.loadBoard()}>Try again</button>
    </div>
  {:else if !app.board || firstLoad}
    <div class="state quiet"><p class="muted">Loading the board</p></div>
  {:else if empty}
    <div class="state">
      <p>This model has no nodes yet.</p>
      <button class="btn primary" onclick={() => (app.dialog = { kind: 'newNode', x: cam.x, y: cam.y })}>New node</button>
    </div>
  {/if}
  {#if loadError && app.board}
    <div class="loaderr error-line">The view could not be refreshed. {loadError} <button class="btn" onclick={() => reload()}>Try again</button></div>
  {/if}
</div>

<style>
  .board {
    position: relative;
    flex: 1 1 auto;
    min-width: 0;
    min-height: 0;
    overflow: hidden;
    background: var(--void);
    touch-action: none;
    user-select: none;
  }
  .board:focus-visible {
    outline: none;
    box-shadow: inset 0 0 0 1px #5a5a5a;
  }
  canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    display: block;
  }
  .world {
    position: absolute;
    left: 0;
    top: 0;
    width: 0;
    height: 0;
    transform-origin: 0 0;
    pointer-events: none;
    will-change: transform;
  }
  .band {
    position: absolute;
    border: 1px solid var(--white);
    background: rgba(247, 247, 242, 0.06);
    pointer-events: none;
  }
  .tip {
    position: absolute;
    padding: 2px 6px;
    background: var(--surface);
    border: 1px solid var(--line);
    pointer-events: none;
    white-space: nowrap;
    max-width: 320px;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .tools {
    position: absolute;
    left: 8px;
    top: 8px;
    display: flex;
    gap: 6px;
    align-items: center;
  }
  .tools > .btn,
  .tools .anchor > .btn,
  .tools .seg,
  .nav .btn {
    background: var(--surface);
  }
  .nav .btn:hover:not(:disabled),
  .tools .btn:hover:not(:disabled) {
    background: #1c1c1c;
  }
  .anchor {
    position: relative;
  }
  .key {
    position: absolute;
    left: 0;
    top: 34px;
    width: 264px;
    padding: 8px 10px;
    z-index: 5;
  }
  .key .k {
    display: flex;
    align-items: center;
    gap: 8px;
    height: 22px;
  }
  .key .chk {
    margin-top: 6px;
    padding-top: 8px;
    height: 30px;
    border-top: 1px solid var(--line);
    cursor: pointer;
  }
  .key p {
    margin-top: 2px;
    font-size: 12px;
    line-height: 16px;
  }
  .nav {
    position: absolute;
    right: 8px;
    bottom: 8px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    align-items: stretch;
    width: 196px;
  }
  .navrow {
    display: flex;
    gap: 4px;
    align-items: center;
  }
  .navrow .gap {
    flex: 1;
  }
  .navrow .btn:not(.icon) {
    padding: 0 8px;
    flex: 1 1 auto;
  }
  .pct {
    min-width: 44px;
    text-align: center;
    background: var(--void);
    border-radius: 2px;
    line-height: 28px;
    color: var(--muted);
  }
  .views {
    position: absolute;
    right: 0;
    bottom: 34px;
    width: 260px;
    padding: 6px;
    z-index: 5;
  }
  .views ul {
    list-style: none;
    margin: 0 0 6px;
    padding: 0;
    max-height: 220px;
    overflow: auto;
  }
  .views li {
    display: flex;
    align-items: center;
    gap: 2px;
  }
  .views .jump {
    flex: 1;
    min-width: 0;
    height: 28px;
    border: 0;
    background: transparent;
    text-align: left;
    padding: 0 6px;
    cursor: pointer;
    border-radius: 2px;
  }
  .views .jump:hover {
    background: #262626;
  }
  .views .none {
    padding: 4px 6px 8px;
  }
  .state {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    gap: 12px;
    align-items: center;
    justify-content: center;
    background: rgba(5, 5, 5, 0.86);
    text-align: center;
    padding: 24px;
    font-size: 14px;
  }
  .state.quiet {
    background: transparent;
    pointer-events: none;
  }
  .loaderr {
    position: absolute;
    left: 50%;
    top: 8px;
    transform: translateX(-50%);
    display: flex;
    gap: 10px;
    align-items: center;
    max-width: 70%;
  }
  @media (max-width: 720px) {
    .tools .seg span,
    .tools > .btn:not(.icon) :global(svg) {
      display: none;
    }
    /* a narrow board keeps its view controls whole; the overview map would cover a third of it */
    .nav {
      width: auto;
      max-width: calc(100% - 16px);
    }
    .nav :global(.minimap) {
      display: none;
    }
    .navrow {
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .navrow .gap {
      display: none;
    }
  }
</style>
