<script lang="ts">
  import type { Snippet } from 'svelte';
  import { onMount } from 'svelte';
  let { title, onclose, children, width = 440 }: { title: string; onclose: () => void; children: Snippet; width?: number } = $props();
  let dlg: HTMLDialogElement;
  onMount(() => {
    dlg.showModal();
    return () => dlg.open && dlg.close();
  });
</script>

<dialog bind:this={dlg} class="pop" style="width:{width}px" aria-label={title} onclose={onclose} onclick={(e) => e.target === dlg && onclose()}>
  <div class="inner">
    <h2>{title}</h2>
    {@render children()}
  </div>
</dialog>

<style>
  dialog {
    padding: 0;
    color: var(--white);
    max-width: calc(100vw - 24px);
    max-height: calc(100vh - 48px);
    overflow: auto;
  }
  dialog::backdrop {
    background: rgba(5, 5, 5, 0.6);
  }
  .inner {
    padding: 14px 16px 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  h2 {
    font-size: 15px;
    line-height: 20px;
    font-weight: 500;
  }
</style>
