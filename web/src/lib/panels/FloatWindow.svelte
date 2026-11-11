<script lang="ts">
  import { app, panelTitle, panelHome, FLOAT_MIN_W, FLOAT_MIN_H, type PanelId } from '../state.svelte';
  import Icon from '../ui/Icon.svelte';
  import PanelBody from './PanelBody.svelte';

  let { id }: { id: PanelId } = $props();
  const p = $derived(app.panels.items[id]);
  const z = $derived(30 + Math.max(0, app.panels.order.indexOf(id)));
  type Dir = 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw';
  const DIRS: Dir[] = ['n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw'];
  const home = $derived(panelHome(id));

  // Pointer capture keeps the gesture on the handle, so the board underneath never pans or zooms.
  function drag(e: PointerEvent, mode: 'move' | Dir) {
    if (e.button !== 0) return;
    if (mode === 'move' && (e.target as HTMLElement).closest('button')) return;
    e.preventDefault();
    e.stopPropagation();
    const el = e.currentTarget as HTMLElement;
    el.setPointerCapture(e.pointerId);
    app.raise(id);
    const sx = e.clientX;
    const sy = e.clientY;
    const o = { x: p.x, y: p.y, w: p.w, h: p.h };
    const W = window.innerWidth;
    const H = window.innerHeight;
    const move = (ev: PointerEvent) => {
      const dx = ev.clientX - sx;
      const dy = ev.clientY - sy;
      const it = app.panels.items[id];
      if (mode === 'move') {
        it.x = Math.max(80 - o.w, Math.min(W - 80, o.x + dx));
        it.y = Math.max(40, Math.min(H - 56, o.y + dy));
        return;
      }
      if (mode.includes('e')) it.w = Math.max(FLOAT_MIN_W, Math.min(W - o.x, o.w + dx));
      if (mode.includes('s')) it.h = Math.max(FLOAT_MIN_H, Math.min(H - o.y, o.h + dy));
      if (mode.includes('w')) {
        const w = Math.max(FLOAT_MIN_W, o.w - dx);
        it.x = o.x + o.w - w;
        it.w = w;
      }
      if (mode.includes('n')) {
        const h = Math.max(FLOAT_MIN_H, Math.min(o.y + o.h - 40, o.h - dy));
        it.y = o.y + o.h - h;
        it.h = h;
      }
    };
    const up = () => {
      el.removeEventListener('pointermove', move);
      el.removeEventListener('pointerup', up);
      el.removeEventListener('pointercancel', up);
      app.savePanels();
    };
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', up);
    el.addEventListener('pointercancel', up);
  }
  function key(e: KeyboardEvent) {
    if (e.target !== e.currentTarget) return;
    const step = e.shiftKey ? 32 : 8;
    const d: Record<string, [number, number]> = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
    const m = d[e.key];
    if (!m) return;
    e.preventDefault();
    e.stopPropagation();
    const it = app.panels.items[id];
    if (e.altKey) {
      it.w = Math.max(FLOAT_MIN_W, it.w + m[0]);
      it.h = Math.max(FLOAT_MIN_H, it.h + m[1]);
    } else {
      it.x += m[0];
      it.y += m[1];
    }
    app.clampFloat(id);
    app.savePanels();
  }
</script>

<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<section class="win pop" style="left:{p.x}px;top:{p.y}px;width:{p.w}px;height:{p.h}px;z-index:{z}" aria-label="{panelTitle(id)} window" data-window={id} onpointerdowncapture={() => app.raise(id)} onwheel={(e) => e.stopPropagation()}>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
  <header onpointerdown={(e) => drag(e, 'move')} onkeydown={key} tabindex="0" aria-label="{panelTitle(id)}: arrow keys move, Alt+arrows resize" title="Drag to move">
    <h2 class="ellipsis">{panelTitle(id)}</h2>
    <button class="btn icon quiet" aria-label="Dock {panelTitle(id)} to the {home}" title="Dock to the {home}" onclick={() => app.dockPanel(id)}><Icon name={home === 'left' ? 'dockleft' : 'dock'} size={14} /></button>
    <button class="btn icon quiet" aria-label="Close the {panelTitle(id)} window" title="Close (reopen from Panels)" onclick={() => app.closePanel(id)}><Icon name="close" size={14} /></button>
  </header>
  <PanelBody {id} />
  {#each DIRS as d (d)}<span class="h {d}" data-resize={d} onpointerdown={(e) => drag(e, d)}></span>{/each}
</section>

<style>
  .win {
    position: fixed;
    display: flex;
    flex-direction: column;
    min-width: 320px;
    min-height: 240px;
  }
  header {
    display: flex;
    align-items: center;
    gap: 2px;
    height: 30px;
    flex: none;
    padding: 0 2px 0 10px;
    border-bottom: 1px solid var(--line);
    cursor: move;
    touch-action: none;
    user-select: none;
  }
  header:focus-visible {
    outline-offset: -2px;
  }
  h2 {
    flex: 1;
    font-weight: 500;
  }
  .h {
    position: absolute;
    touch-action: none;
    z-index: 2;
  }
  .h.n,
  .h.s {
    left: 10px;
    right: 10px;
    height: 7px;
    cursor: ns-resize;
  }
  .h.e,
  .h.w {
    top: 10px;
    bottom: 10px;
    width: 7px;
    cursor: ew-resize;
  }
  .h.n {
    top: -4px;
  }
  .h.s {
    bottom: -4px;
  }
  .h.e {
    right: -4px;
  }
  .h.w {
    left: -4px;
  }
  .h.ne,
  .h.nw,
  .h.se,
  .h.sw {
    width: 12px;
    height: 12px;
  }
  .h.ne {
    right: -4px;
    top: -4px;
    cursor: nesw-resize;
  }
  .h.sw {
    left: -4px;
    bottom: -4px;
    cursor: nesw-resize;
  }
  .h.nw {
    left: -4px;
    top: -4px;
    cursor: nwse-resize;
  }
  .h.se {
    right: -4px;
    bottom: -4px;
    cursor: nwse-resize;
  }
</style>
