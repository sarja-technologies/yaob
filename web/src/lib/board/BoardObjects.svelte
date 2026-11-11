<script lang="ts">
  // Board-only objects drawn in world space: groups (labelled frames), notes and references
  // (a second occurrence of an existing node). They are few, so they are DOM; nodes never are.
  import { app } from '../state.svelte';
  import { get, type BoardObject, type NodeDetail } from '../api';

  let {
    objects,
    zoom,
    radius,
    selected = $bindable(null),
    onchange,
    onmenu,
    onopen,
  }: {
    objects: BoardObject[];
    zoom: () => number;
    radius: number;
    selected?: string | null;
    onchange: (o: BoardObject, final: boolean) => void;
    onmenu: (o: BoardObject, e: MouseEvent) => void;
    onopen: (nodeId: string) => void;
  } = $props();

  type Dir = 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw';
  const DIRS: Dir[] = ['n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw'];
  let editing = $state<string | null>(null);
  let refs = $state<Record<string, { title: string; type: string; status: string } | null>>({});

  $effect(() => {
    for (const o of objects) {
      if (o.kind !== 'reference') continue;
      const id = String(o.props.node_id);
      if (id in refs) continue;
      refs[id] = null;
      const have = app.nodeCache.get(id);
      (have ? Promise.resolve(have) : get<NodeDetail>(`api/nodes/${id}`))
        .then((n) => (refs[id] = { title: n.title, type: n.type, status: n.status }))
        .catch(() => (refs[id] = { title: `node ${id} (not found)`, type: 'knowledge', status: 'tombstone' }));
    }
  });
  // a node edit may have renamed a referenced node
  $effect(() => {
    app.head;
    refs = {};
  });

  function drag(e: PointerEvent, o: BoardObject, mode: 'move' | Dir, click?: () => void) {
    if (e.button !== 0) return;
    e.stopPropagation();
    const el = e.currentTarget as HTMLElement;
    el.setPointerCapture(e.pointerId);
    selected = o.id;
    const z = zoom();
    const sx = e.clientX;
    const sy = e.clientY;
    const minW = 48 / z;
    const minH = 32 / z;
    let cur = o;
    let moved = false;
    const move = (ev: PointerEvent) => {
      const dx = (ev.clientX - sx) / z;
      const dy = (ev.clientY - sy) / z;
      if (!moved && Math.hypot(ev.clientX - sx, ev.clientY - sy) < 3) return;
      moved = true;
      let { x, y, w, h } = o;
      if (mode === 'move') {
        x += dx;
        y += dy;
      } else {
        if (mode.includes('e')) w = Math.max(minW, o.w + dx);
        if (mode.includes('s')) h = Math.max(minH, o.h + dy);
        if (mode.includes('w')) {
          w = Math.max(minW, o.w - dx);
          x = o.x + o.w - w;
        }
        if (mode.includes('n')) {
          h = Math.max(minH, o.h - dy);
          y = o.y + o.h - h;
        }
      }
      cur = { ...o, x, y, w, h };
      onchange(cur, false);
    };
    const up = () => {
      el.removeEventListener('pointermove', move);
      el.removeEventListener('pointerup', up);
      el.removeEventListener('pointercancel', up);
      if (moved) onchange(cur, true);
      else click?.();
    };
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', up);
    el.addEventListener('pointercancel', up);
  }
  function key(e: KeyboardEvent, o: BoardObject) {
    if ((e.target as HTMLElement).tagName === 'TEXTAREA' || (e.target as HTMLElement).tagName === 'INPUT') {
      e.stopPropagation();
      if (e.key === 'Escape' || (e.key === 'Enter' && (e.target as HTMLElement).tagName === 'INPUT')) (e.target as HTMLElement).blur();
      return;
    }
    const step = 8 / zoom();
    const d: Record<string, [number, number]> = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
    const m = d[e.key];
    if (m) {
      e.preventDefault();
      e.stopPropagation();
      selected = o.id;
      onchange({ ...o, x: o.x + m[0], y: o.y + m[1] }, false);
    } else if (e.key === 'Enter' && o.kind !== 'reference') {
      e.preventDefault();
      editing = o.id;
    } else if (e.key === 'Enter') {
      onopen(String(o.props.node_id));
    }
  }
  function setProp(o: BoardObject, k: string, v: string) {
    if (o.props[k] === v) return;
    onchange({ ...o, props: { ...o.props, [k]: v } }, true);
  }
  function focusNow(node: HTMLElement) {
    queueMicrotask(() => {
      node.focus();
      if (node instanceof HTMLInputElement) node.select();
    });
  }
  const menu = (e: MouseEvent, o: BoardObject) => {
    e.preventDefault();
    e.stopPropagation();
    selected = o.id;
    onmenu(o, e);
  };
</script>

{#each objects as o (o.id)}
  {#if o.kind === 'reference'}
    {@const r = refs[String(o.props.node_id)]}
    <div
      class="ref"
      class:sel={selected === o.id}
      style="left:{o.x}px;top:{o.y}px;--r:{radius}px"
      role="button"
      tabindex="0"
      aria-label="Reference to {r?.title ?? 'a node'}"
      title="A second place for the same node. Opening it opens the one node."
      onpointerdown={(e) => drag(e, o, 'move', () => onopen(String(o.props.node_id)))}
      onkeydown={(e) => key(e, o)}
      oncontextmenu={(e) => menu(e, o)}
    >
      <span class="glyph" class:square={r?.type === 'procedure'}></span>
      <span class="reftitle">{r?.title ?? ''}{" "}<span class="muted">(reference)</span></span>
    </div>
  {:else if o.kind === 'group'}
    <div class="obj group" class:sel={selected === o.id} style="left:{o.x}px;top:{o.y}px;width:{o.w}px;height:{o.h}px">
      {#if editing === o.id}
        <input class="tab edit" value={o.props.label ?? ''} aria-label="Group label" maxlength="80" use:focusNow onkeydown={(e) => key(e, o)} onblur={(e) => (setProp(o, 'label', e.currentTarget.value.trim() || 'Group'), (editing = null))} />
      {:else}
        <button class="tab" title="Drag to move. Double-click to rename." onpointerdown={(e) => drag(e, o, 'move')} ondblclick={() => (editing = o.id)} onkeydown={(e) => key(e, o)} oncontextmenu={(e) => menu(e, o)}>{o.props.label || 'Group'}</button>
      {/if}
      {#if selected === o.id}
        {#each DIRS as d (d)}<span class="h {d}" onpointerdown={(e) => drag(e, o, d)}></span>{/each}
      {/if}
    </div>
  {:else if o.kind === 'note'}
    <div
      class="obj note"
      class:sel={selected === o.id}
      style="left:{o.x}px;top:{o.y}px;width:{o.w}px;height:{o.h}px;--s:{o.props.s ?? 1}"
      role="note"
      tabindex="0"
      onpointerdown={(e) => editing !== o.id && drag(e, o, 'move')}
      ondblclick={() => (editing = o.id)}
      onkeydown={(e) => key(e, o)}
      oncontextmenu={(e) => menu(e, o)}
    >
      {#if editing === o.id}
        <textarea aria-label="Note text" value={o.props.text ?? ''} use:focusNow onpointerdown={(e) => e.stopPropagation()} onblur={(e) => (setProp(o, 'text', e.currentTarget.value), (editing = null))}></textarea>
      {:else if o.props.text}
        <div class="txt">{o.props.text}</div>
      {:else}
        <div class="txt muted">Double-click to write</div>
      {/if}
      {#if selected === o.id}
        {#each DIRS as d (d)}<span class="h {d}" onpointerdown={(e) => drag(e, o, d)}></span>{/each}
      {/if}
    </div>
  {/if}
{/each}

<style>
  .obj {
    position: absolute;
    --px: calc(1px / var(--z, 1));
  }
  .group {
    border: var(--px) solid #5a5a5a;
    pointer-events: none;
  }
  .group.sel {
    border-color: var(--white);
  }
  .tab {
    position: absolute;
    left: calc(-1 * var(--px));
    bottom: 100%;
    font-size: calc(12px / var(--z, 1));
    line-height: calc(18px / var(--z, 1));
    padding: 0 calc(6px / var(--z, 1));
    max-width: max(100%, calc(80px / var(--z, 1)));
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    background: #2a2a2a;
    color: var(--white);
    border: 0;
    border-radius: 0;
    font-weight: 500;
    cursor: move;
    pointer-events: auto;
  }
  .group.sel .tab {
    background: var(--white);
    color: var(--void);
  }
  input.tab.edit {
    width: calc(180px / var(--z, 1));
    max-width: none;
    cursor: text;
    background: var(--void);
    color: var(--white);
    outline: var(--px) solid var(--white);
  }
  .note {
    background: var(--surface);
    border: var(--px) solid #4a4a4a;
    pointer-events: auto;
    cursor: move;
    overflow: visible;
  }
  .note.sel {
    border-color: var(--white);
  }
  .note .txt,
  .note textarea {
    position: absolute;
    inset: 0;
    padding: calc(8px * var(--s));
    font-size: calc(13px * var(--s));
    line-height: calc(18px * var(--s));
    white-space: pre-wrap;
    overflow: hidden;
    overflow-wrap: anywhere;
    user-select: none;
  }
  .note textarea {
    width: 100%;
    height: 100%;
    border: 0;
    background: var(--void);
    color: var(--white);
    resize: none;
    outline: none;
    cursor: text;
    user-select: text;
    font-family: inherit;
  }
  .h {
    position: absolute;
    width: calc(9px / var(--z, 1));
    height: calc(9px / var(--z, 1));
    margin: calc(-4.5px / var(--z, 1));
    background: var(--void);
    border: var(--px) solid var(--white);
    pointer-events: auto;
    touch-action: none;
  }
  /* edge handles stretch along their edge so the whole edge can be grabbed */
  .h.n,
  .h.s {
    left: calc(9px / var(--z, 1));
    right: calc(9px / var(--z, 1));
    width: auto;
    margin-left: 0;
    margin-right: 0;
    cursor: ns-resize;
    background: transparent;
    border-color: transparent;
  }
  .h.e,
  .h.w {
    top: calc(9px / var(--z, 1));
    bottom: calc(9px / var(--z, 1));
    height: auto;
    margin-top: 0;
    margin-bottom: 0;
    cursor: ew-resize;
    background: transparent;
    border-color: transparent;
  }
  .h.n {
    top: 0;
  }
  .h.s {
    top: 100%;
  }
  .h.w {
    left: 0;
  }
  .h.e {
    left: 100%;
  }
  .h.n::after,
  .h.s::after,
  .h.e::after,
  .h.w::after {
    content: '';
    position: absolute;
    left: 50%;
    top: 50%;
    width: calc(9px / var(--z, 1));
    height: calc(9px / var(--z, 1));
    margin: calc(-4.5px / var(--z, 1));
    background: var(--void);
    border: var(--px) solid var(--white);
    box-sizing: border-box;
  }
  .h.nw {
    left: 0;
    top: 0;
    cursor: nwse-resize;
  }
  .h.se {
    left: 100%;
    top: 100%;
    cursor: nwse-resize;
  }
  .h.ne {
    left: 100%;
    top: 0;
    cursor: nesw-resize;
  }
  .h.sw {
    left: 0;
    top: 100%;
    cursor: nesw-resize;
  }

  .ref {
    position: absolute;
    width: 0;
    height: 0;
    pointer-events: auto;
    cursor: pointer;
  }
  .ref .glyph {
    position: absolute;
    left: calc(-1 * var(--r));
    top: calc(-1 * var(--r));
    width: calc(2 * var(--r));
    height: calc(2 * var(--r));
    border-radius: 50%;
    border: max(calc(1.5px / var(--z, 1)), calc(var(--r) * 0.16)) dashed var(--white);
    background: var(--void);
  }
  .ref .glyph.square {
    border-radius: 0;
  }
  .ref.sel .glyph {
    border-color: var(--ember);
  }
  .ref:focus-visible {
    outline: none;
  }
  .ref:focus-visible .glyph {
    outline: calc(2px / var(--z, 1)) solid var(--white);
    outline-offset: calc(2px / var(--z, 1));
  }
  .ref .reftitle {
    position: absolute;
    left: calc(var(--r) + 5px / var(--z, 1));
    top: calc(-9px / var(--z, 1));
    font-size: calc(12px / var(--z, 1));
    line-height: calc(18px / var(--z, 1));
    white-space: nowrap;
    padding: 0 calc(3px / var(--z, 1));
    background: rgba(5, 5, 5, 0.86);
  }
</style>
