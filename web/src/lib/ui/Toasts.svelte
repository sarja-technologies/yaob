<script lang="ts">
  import { app } from '../state.svelte';
  import Icon from './Icon.svelte';
</script>

<div class="toasts" aria-live="polite">
  {#each app.toasts as t (t.id)}
    <div class="toast pop" class:err={t.kind === 'error'} role={t.kind === 'error' ? 'alert' : 'status'}>
      <span class="msg">{t.text}</span>
      {#if t.action}<button class="btn" onclick={() => (t.action!.run(), app.dismiss(t.id))}>{t.action.label}</button>{/if}
      <button class="btn icon quiet" aria-label="Dismiss" onclick={() => app.dismiss(t.id)}><Icon name="close" size={14} /></button>
    </div>
  {/each}
</div>

<style>
  .toasts {
    position: fixed;
    left: 50%;
    bottom: 36px;
    transform: translateX(-50%);
    display: flex;
    flex-direction: column;
    gap: 6px;
    align-items: center;
    z-index: 95;
    pointer-events: none;
    max-width: calc(100vw - 24px);
  }
  .toast {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 4px 4px 12px;
    min-height: 36px;
    pointer-events: auto;
    max-width: 640px;
  }
  .toast.err {
    border-color: var(--clay);
  }
  .msg {
    padding: 4px 0;
  }
</style>
