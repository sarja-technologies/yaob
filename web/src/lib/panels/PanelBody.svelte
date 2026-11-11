<script lang="ts">
  import type { PanelId } from '../state.svelte';
  import Library from './Library.svelte';
  import Inspector from './Inspector.svelte';
  import RunPanel from './RunPanel.svelte';
  import SchemaEditor from './SchemaEditor.svelte';
  import Jobs from './Jobs.svelte';
  import Budget from './Budget.svelte';
  import Runtime from './Runtime.svelte';

  let { id }: { id: PanelId } = $props();
</script>

<div class="body scroll" data-panel={id}>
  {#if id === 'library'}<Library />
  {:else if id === 'inspector'}<Inspector />
  {:else if id === 'run'}<RunPanel />
  {:else if id === 'schema'}<SchemaEditor />
  {:else if id === 'jobs'}<Jobs />
  {:else if id === 'budget'}<Budget />
  {:else if id === 'runtime'}<Runtime />{/if}
</div>

<style>
  .body {
    flex: 1 1 auto;
    min-height: 0;
    min-width: 0;
    display: flex;
    flex-direction: column;
    container-type: inline-size;
  }
</style>
