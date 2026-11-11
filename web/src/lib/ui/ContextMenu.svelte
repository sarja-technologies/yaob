<script lang="ts">
  import { app } from '../state.svelte';
  import { clickOutside } from './actions';

  let el = $state<HTMLDivElement | null>(null);
  let pos = $state({ x: 0, y: 0 });

  $effect(() => {
    const m = app.menu;
    if (!m || !el) return;
    // keep the menu inside the window, also near the edges of a small window
    const r = el.getBoundingClientRect();
    pos = { x: Math.max(4, Math.min(m.x, window.innerWidth - r.width - 4)), y: Math.max(4, Math.min(m.y, window.innerHeight - r.height - 4)) };
    queueMicrotask(() => (el?.querySelector('button:not(:disabled)') as HTMLElement | null)?.focus());
  });
  function key(e: KeyboardEvent) {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    e.preventDefault();
    e.stopPropagation();
    const items = [...(el?.querySelectorAll('button:not(:disabled)') ?? [])] as HTMLElement[];
    const i = items.indexOf(document.activeElement as HTMLElement);
    items[(i + (e.key === 'ArrowDown' ? 1 : items.length - 1)) % items.length]?.focus();
  }
</script>

{#if app.menu}
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="pop menu ctx" role="menu" tabindex="-1" bind:this={el} style="left:{pos.x}px;top:{pos.y}px" use:clickOutside={() => { app.menu = null; }} onkeydown={key} oncontextmenu={(e) => e.preventDefault()}>
    {#each app.menu.items as it}
      {#if it === 'sep'}
        <hr />
      {:else}
        <button role="menuitem" class:danger={it.danger} disabled={it.disabled} onclick={() => ((app.menu = null), it.run?.())}>
          {it.label}
          {#if it.disabled && it.why}<span class="why">{it.why}</span>{:else if it.kbd}<span class="kbd">{it.kbd}</span>{/if}
        </button>
      {/if}
    {/each}
  </div>
{/if}

<style>
  .ctx {
    position: fixed;
    z-index: 90;
    min-width: 210px;
  }
  .danger:not(:disabled) {
    color: #ffb9a6;
  }
</style>
