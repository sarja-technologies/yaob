<script lang="ts">
  import { app, panelTitle, type Side } from '../state.svelte';
  import Icon from '../ui/Icon.svelte';
  import PanelBody from './PanelBody.svelte';

  let { side }: { side: Side } = $props();
  const dock = $derived(app.panels[side]);
  const ids = $derived(app.docked(side));
  const active = $derived(dock.active && ids.includes(dock.active) ? dock.active : (ids[0] ?? null));
  const name = side === 'left' ? 'Left' : 'Right';

  function resize(e: PointerEvent) {
    const el = e.currentTarget as HTMLElement;
    el.setPointerCapture(e.pointerId);
    e.preventDefault();
    const start = e.clientX;
    const w0 = dock.width;
    const max = Math.max(260, Math.min(640, window.innerWidth - 360));
    const move = (ev: PointerEvent) => {
      const d = side === 'left' ? ev.clientX - start : start - ev.clientX;
      dock.width = Math.round(Math.max(220, Math.min(max, w0 + d)));
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
  function resizeKey(e: KeyboardEvent) {
    const grow = side === 'left' ? 'ArrowRight' : 'ArrowLeft';
    const shrink = side === 'left' ? 'ArrowLeft' : 'ArrowRight';
    if (e.key !== grow && e.key !== shrink) return;
    e.preventDefault();
    e.stopPropagation();
    dock.width = Math.max(220, Math.min(640, dock.width + (e.key === grow ? 16 : -16)));
    app.savePanels();
  }
  function setOpen(v: boolean) {
    dock.open = v;
    if (v && app.narrow) app.panels[side === 'left' ? 'right' : 'left'].open = false;
    app.savePanels();
  }
  function tabKey(e: KeyboardEvent) {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    e.stopPropagation();
    const i = ids.indexOf(active!);
    const next = ids[(i + (e.key === 'ArrowRight' ? 1 : ids.length - 1)) % ids.length];
    dock.active = next;
    app.savePanels();
    queueMicrotask(() => document.getElementById(`tab-${next}`)?.focus());
  }
</script>

{#if ids.length}
  {#if dock.open && active}
    <aside class="dock {side}" class:overlay={app.narrow} style="width:{dock.width}px" aria-label="{name} panels">
      <div class="head">
        <div class="tabs" role="tablist" aria-label="{name} panels">
          {#each ids as id (id)}
            <button id="tab-{id}" role="tab" aria-selected={id === active} tabindex={id === active ? 0 : -1} onclick={() => ((dock.active = id), app.savePanels())} onkeydown={tabKey}>{panelTitle(id)}</button>
          {/each}
        </div>
        <button class="btn icon quiet" aria-label="Detach {panelTitle(active)} into a window" title="Detach into a window" onclick={() => app.floatPanel(active, { x: side === 'left' ? dock.width + 24 : window.innerWidth - dock.width - 480, y: 76 })}><Icon name="detach" size={14} /></button>
        <button class="btn icon quiet" aria-label="Close the {panelTitle(active)} panel" title="Close panel (reopen from Panels)" onclick={() => app.closePanel(active)}><Icon name="close" size={14} /></button>
        <button class="btn icon quiet" aria-label="Collapse the {side} dock" title="Collapse" onclick={() => setOpen(false)}><Icon name={side === 'left' ? 'collapseL' : 'collapseR'} size={14} /></button>
      </div>
      <PanelBody id={active} />
      <!-- svelte-ignore a11y_no_noninteractive_tabindex a11y_no_noninteractive_element_interactions -->
      <div class="resizer" role="separator" aria-orientation="vertical" aria-label="Resize the {side} dock" aria-valuenow={dock.width} aria-valuemin={220} aria-valuemax={640} tabindex="0" onpointerdown={resize} onkeydown={resizeKey}></div>
    </aside>
  {:else}
    <div class="rail {side}">
      <button class="btn icon quiet" aria-label="Expand the {side} dock" title="Show {ids.map(panelTitle).join(', ')}" onclick={() => setOpen(true)}><Icon name={side === 'left' ? 'collapseR' : 'collapseL'} size={14} /></button>
    </div>
  {/if}
{/if}

<style>
  .dock {
    position: relative;
    flex: none;
    display: flex;
    flex-direction: column;
    min-height: 0;
    max-width: calc(100vw - 80px);
    background: var(--surface);
  }
  .dock.left {
    border-right: 1px solid var(--line);
  }
  .dock.right {
    border-left: 1px solid var(--line);
  }
  .dock.overlay {
    position: absolute;
    top: 0;
    bottom: 0;
    z-index: 20;
  }
  .dock.overlay.left {
    left: 0;
  }
  .dock.overlay.right {
    right: 0;
  }
  .head {
    display: flex;
    align-items: center;
    height: 32px;
    flex: none;
    padding: 0 2px 0 0;
    border-bottom: 1px solid var(--line);
  }
  .tabs {
    display: flex;
    flex: 1 1 auto;
    min-width: 0;
    height: 100%;
    overflow-x: auto;
    scrollbar-width: none;
  }
  .tabs button {
    border: 0;
    background: transparent;
    color: var(--muted);
    padding: 0 10px;
    height: 100%;
    font-weight: 500;
    white-space: nowrap;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    border-top: 2px solid transparent;
  }
  .tabs button:hover {
    color: var(--white);
  }
  .tabs button[aria-selected='true'] {
    color: var(--white);
    border-bottom-color: var(--white);
  }
  .tabs button:focus-visible {
    outline-offset: -2px;
  }
  .resizer {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 7px;
    cursor: col-resize;
    touch-action: none;
    z-index: 3;
  }
  .left .resizer {
    right: -4px;
  }
  .right .resizer {
    left: -4px;
  }
  .resizer:hover,
  .resizer:focus-visible {
    background: rgba(247, 247, 242, 0.18);
    outline: none;
  }
  .rail {
    flex: none;
    width: 30px;
    padding-top: 2px;
    display: flex;
    flex-direction: column;
    align-items: center;
    background: var(--surface);
  }
  .rail.left {
    border-right: 1px solid var(--line);
  }
  .rail.right {
    border-left: 1px solid var(--line);
  }
</style>
