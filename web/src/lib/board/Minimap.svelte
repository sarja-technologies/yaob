<script lang="ts">
  import type { Region } from '../api';

  let { mpns, bounds, onmove }: { mpns: Region[]; bounds: [number, number, number, number] | null; onmove: (x: number, y: number) => void } = $props();
  const S = 196;
  const HGT = 132;
  let canvas: HTMLCanvasElement;
  let last = { x: 0, y: 0, zoom: 1, w: 1, h: 1 };
  let dragging = false;

  function fit() {
    const b = bounds ?? [-1000, -1000, 1000, 1000];
    const pad = 6;
    const k = Math.min((S - pad * 2) / (b[2] - b[0]), (HGT - pad * 2) / (b[3] - b[1]));
    return { k, ox: S / 2 - ((b[0] + b[2]) / 2) * k, oy: HGT / 2 - ((b[1] + b[3]) / 2) * k };
  }
  /** Called by the board on every frame it draws. */
  export function draw(cam: { x: number; y: number; zoom: number }, w: number, h: number) {
    last = { x: cam.x, y: cam.y, zoom: cam.zoom, w, h };
    const c = canvas?.getContext('2d');
    if (!c) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== S * dpr) {
      canvas.width = S * dpr;
      canvas.height = HGT * dpr;
    }
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, S, HGT);
    const { k, ox, oy } = fit();
    c.lineWidth = 1;
    for (const m of mpns) {
      c.beginPath();
      c.arc(m.x * k + ox, m.y * k + oy, Math.max(1.5, m.r * k), 0, Math.PI * 2);
      c.fillStyle = '#1c1c1c';
      c.fill();
      c.strokeStyle = m.enabled === false || m.installed === false ? '#3a3a3a' : '#5a5a5a';
      c.stroke();
    }
    const vw = (w / cam.zoom) * k;
    const vh = (h / cam.zoom) * k;
    const vx = cam.x * k + ox - vw / 2;
    const vy = cam.y * k + oy - vh / 2;
    c.strokeStyle = '#f7f7f2';
    c.lineWidth = 1;
    const x0 = Math.max(0.5, vx);
    const y0 = Math.max(0.5, vy);
    const x1 = Math.min(S - 0.5, vx + vw);
    const y1 = Math.min(HGT - 0.5, vy + vh);
    if (x1 - x0 < 5 || y1 - y0 < 5) {
      // too small to read as a rectangle: mark the position with a cross
      const cx = Math.max(4, Math.min(S - 4, vx + vw / 2));
      const cy = Math.max(4, Math.min(HGT - 4, vy + vh / 2));
      c.beginPath();
      c.moveTo(cx - 4, cy);
      c.lineTo(cx + 4, cy);
      c.moveTo(cx, cy - 4);
      c.lineTo(cx, cy + 4);
      c.stroke();
    } else {
      c.fillStyle = 'rgba(247,247,242,0.07)';
      c.fillRect(x0, y0, x1 - x0, y1 - y0);
      c.strokeRect(Math.round(x0) + 0.5, Math.round(y0) + 0.5, Math.round(x1 - x0) - 1, Math.round(y1 - y0) - 1);
    }
  }
  function go(e: PointerEvent) {
    const r = canvas.getBoundingClientRect();
    const { k, ox, oy } = fit();
    onmove((e.clientX - r.left - ox) / k, (e.clientY - r.top - oy) / k);
  }
  function key(e: KeyboardEvent) {
    const step = (last.w / last.zoom) * 0.25;
    const d: Record<string, [number, number]> = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
    const m = d[e.key];
    if (!m) return;
    e.preventDefault();
    e.stopPropagation();
    onmove(last.x + m[0], last.y + m[1]);
  }
  $effect(() => {
    mpns;
    bounds;
    draw(last, last.w, last.h);
  });
</script>

<canvas
  bind:this={canvas}
  class="minimap"
  style="width:{S}px;height:{HGT}px"
  tabindex="0"
  role="slider"
  aria-label="Overview map. Arrow keys move the view."
  aria-valuenow={Math.round(last.x)}
  onpointerdown={(e) => {
    dragging = true;
    canvas.setPointerCapture(e.pointerId);
    go(e);
  }}
  onpointermove={(e) => dragging && go(e)}
  onpointerup={() => (dragging = false)}
  onpointercancel={() => (dragging = false)}
  onkeydown={key}
></canvas>

<style>
  .minimap {
    display: block;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 2px;
    cursor: crosshair;
    touch-action: none;
  }
</style>
