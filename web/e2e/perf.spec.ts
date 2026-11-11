import { test, expect } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { openApp, resetView } from './helpers';

// Repeatable performance trace: a scripted ~10 s pan/zoom tour of the board, measured in the page with
// requestAnimationFrame. Point it at another store with MPN_URL, e.g.
//   MPN_URL=http://127.0.0.1:8840 pnpm perf
const here = dirname(fileURLToPath(import.meta.url));
const pct = (a: number[], p: number) => {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))];
};
const r2 = (n: number) => Math.round(n * 100) / 100;

test('scripted pan and zoom: frame times, viewport requests, drawn counts', async ({ page, request, baseURL }) => {
  test.setTimeout(120_000);
  await resetView(request);
  await page.setViewportSize({ width: 1920, height: 1080 });
  await openApp(page);
  const status = await (await request.get('api/status')).json();

  const trace = await page.evaluate(async (durationMs) => {
    const m = window.__mpn;
    const regions = [...m.regions()].sort((a: any, b: any) => b.count - a.count);
    const A = regions[0];
    // the nearest other region gives a pan that crosses loaded and unloaded area
    const B = regions.slice(1).sort((p: any, q: any) => Math.hypot(p.x - A.x, p.y - A.y) - Math.hypot(q.x - A.x, q.y - A.y))[0] ?? A;
    const { w, h } = m.size();
    const start = { x: m.cam.x, y: m.cam.y, zoom: m.cam.zoom };
    const fitA = Math.min(w, h) / (A.r * 2.2);
    const keys = [
      { t: 0.0, ...start },
      { t: 0.18, x: A.x, y: A.y, zoom: fitA },
      { t: 0.36, x: A.x + A.r * 0.3, y: A.y, zoom: 3 },
      { t: 0.6, x: B.x - B.r * 0.3, y: B.y, zoom: 3 },
      { t: 0.74, x: B.x, y: B.y, zoom: 8 },
      { t: 0.82, x: B.x + 30, y: B.y + 20, zoom: 8 },
      { t: 1.0, ...start },
    ];
    const frames: number[] = [];
    const drawn = { nodes: 0, labels: 0, edges: 0, cells: 0 };
    const levels: Record<string, number> = {};
    m.stats.requests.length = 0;
    const smooth = (u: number) => u * u * (3 - 2 * u);
    await new Promise<void>((resolve) => {
      let t0 = 0;
      let last = 0;
      const tick = (now: number) => {
        if (!t0) t0 = now;
        else frames.push(now - last);
        last = now;
        const u = Math.min(1, (now - t0) / durationMs);
        let i = 0;
        while (i < keys.length - 2 && u > keys[i + 1].t) i++;
        const a = keys[i];
        const b = keys[i + 1];
        const k = smooth(Math.min(1, Math.max(0, (u - a.t) / (b.t - a.t))));
        m.setView(a.x + (b.x - a.x) * k, a.y + (b.y - a.y) * k, Math.exp(Math.log(a.zoom) + (Math.log(b.zoom) - Math.log(a.zoom)) * k));
        const s = m.stats;
        drawn.nodes = Math.max(drawn.nodes, s.nodes);
        drawn.labels = Math.max(drawn.labels, s.labels);
        drawn.edges = Math.max(drawn.edges, s.edges);
        drawn.cells = Math.max(drawn.cells, s.cells);
        levels[s.level] = (levels[s.level] ?? 0) + 1;
        if (u < 1) requestAnimationFrame(tick);
        else resolve();
      };
      requestAnimationFrame(tick);
    });
    const gl = document.createElement('canvas').getContext('webgl2');
    const ext = gl?.getExtension('WEBGL_debug_renderer_info');
    return {
      frames,
      drawn,
      levels,
      requests: m.stats.requests.map((r: any) => ({ ...r })),
      renderer: gl ? String(gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER)) : 'none',
      dpr: window.devicePixelRatio,
      canvas: m.size(),
    };
  }, 10_000);

  const f = trace.frames;
  const req = trace.requests as { ms: number; serverMs: number; bytes: number; level: string; items: number }[];
  const result = {
    when: new Date().toISOString(),
    url: baseURL,
    store: { live_nodes: status.counts.canonical_live_nodes, edges: status.counts.edges, mpns: status.counts.mpns, head: status.counts.rev },
    window: { width: 1920, height: 1080, board: trace.canvas, dpr: trace.dpr },
    renderer: trace.renderer,
    duration_s: r2(f.reduce((a, b) => a + b, 0) / 1000),
    frames: {
      count: f.length,
      p50_ms: r2(pct(f, 50)),
      p95_ms: r2(pct(f, 95)),
      p99_ms: r2(pct(f, 99)),
      max_ms: r2(Math.max(...f)),
      over_20ms: f.filter((x) => x > 20).length,
      over_33ms: f.filter((x) => x > 33.4).length,
      fps_mean: r2(1000 / (f.reduce((a, b) => a + b, 0) / f.length)),
    },
    viewport_requests: {
      count: req.length,
      p50_ms: r2(pct(req.map((r) => r.ms), 50)),
      p95_ms: r2(pct(req.map((r) => r.ms), 95)),
      max_ms: r2(Math.max(0, ...req.map((r) => r.ms))),
      server_p95_ms: r2(pct(req.map((r) => r.serverMs), 95)),
      payload_bytes_p50: pct(req.map((r) => r.bytes), 50),
      payload_bytes_p95: pct(req.map((r) => r.bytes), 95),
      payload_bytes_max: Math.max(0, ...req.map((r) => r.bytes)),
      max_items: Math.max(0, ...req.map((r) => r.items)),
      by_level: { nodes: req.filter((r) => r.level === 'nodes').length, cells: req.filter((r) => r.level === 'cells').length },
    },
    drawn_max: trace.drawn,
    frames_by_level: trace.levels,
  };
  const dir = join(here, 'results');
  mkdirSync(dir, { recursive: true });
  const host = new URL(baseURL!).host.replace(/[^\w.-]/g, '_');
  writeFileSync(join(dir, `perf-${host}-${result.store.live_nodes}nodes.json`), JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(result, null, 2));
  expect(f.length).toBeGreaterThan(100);
  expect(result.viewport_requests.max_items).toBeLessThanOrEqual(2500);
});
