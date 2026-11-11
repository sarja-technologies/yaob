<script lang="ts">
  import { onMount } from 'svelte';
  import { app } from '../state.svelte';
  import { get, post, errText, fmtBytes, fmtInt, human, when } from '../api';

  let rt = $state.raw<any>(null);
  let error = $state<string | null>(null);
  let busy = $state<string | null>(null);
  let actErr = $state<string | null>(null);

  async function load() {
    try {
      rt = await get('api/runtime');
      error = null;
    } catch (e) {
      error = errText(e);
    }
  }
  onMount(() => {
    load();
    const t = setInterval(() => !document.hidden && load(), 4000);
    return () => clearInterval(t);
  });
  async function act(action: 'load' | 'unload' | 'reindex') {
    busy = action;
    actErr = null;
    try {
      await post(`api/runtime/${action}`);
      // read the real state back instead of assuming the action took effect
      await load();
      app.loadStatus();
      app.loadMpns();
    } catch (e) {
      actErr = errText(e);
    }
    busy = null;
  }
  const down = $derived(!!rt && rt.available === false);
  const flat = (o: any): [string, string][] =>
    Object.entries(o ?? {})
      .filter(([, v]) => typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean')
      .map(([k, v]) => [human(k), typeof v === 'number' ? fmtInt(v) : String(v)]);
</script>

<div class="rt">
  {#if error}
    <p class="error-line">{error}</p>
    <button class="btn" onclick={load}>Try again</button>
  {:else if !rt}
    <p class="muted">Asking the runtime</p>
  {:else if down}
    <p class="lead">The runtime is not running</p>
    <p>{rt.reason}.</p>
    <p class="muted">The board, editing and text search work without it. Chat, search by meaning and training need it.</p>
    <div class="row wrap">
      <button class="btn" disabled>Load core</button>
      <button class="btn" disabled>Unload</button>
      <button class="btn" disabled>Reindex</button>
    </div>
    <p class="muted">Start the runtime process first.</p>
  {:else}
    <dl>
      <dt>Core</dt>
      <dd>{rt.core?.name}{rt.core?.loaded === false ? ', not loaded' : ', loaded'}</dd>
      {#if rt.core?.repo}
        <dt>Repository</dt>
        <dd class="mono">{rt.core.repo}</dd>
      {/if}
      <dt>Revision</dt>
      <dd class="mono">{rt.core?.revision}</dd>
      <dt>Licence</dt>
      <dd>{rt.core?.license}</dd>
      <dt>Core parameters</dt>
      <dd class="num">{fmtInt(rt.core?.params_total)}</dd>
      {#each flat(rt.learned) as [k, v]}
        <dt>Learned: {k}</dt>
        <dd class="num">{v}</dd>
      {/each}
      {#if rt.encoder}
        <dt>Encoder</dt>
        <dd class="mono">{typeof rt.encoder === 'string' ? rt.encoder : (rt.encoder.name ?? JSON.stringify(rt.encoder))}</dd>
      {/if}
      <dt>Checkpoint</dt>
      <dd class="mono">{rt.checkpoint ? (typeof rt.checkpoint === 'string' ? rt.checkpoint : (rt.checkpoint.id ?? JSON.stringify(rt.checkpoint))) : 'none'}</dd>
      <dt>Requests</dt>
      <dd class="num">{fmtInt(rt.requests?.active)} active, {fmtInt(rt.requests?.queued)} queued</dd>
      <dt>Memory</dt>
      <dd class="num">process {fmtInt(rt.memory?.process_rss_mib)} MiB, tensors {fmtInt(rt.memory?.torch_allocated_mib)} MiB, peak {fmtInt(rt.memory?.torch_peak_mib)} MiB</dd>
      <dt>Index</dt>
      <dd class="num">{fmtInt(rt.index?.rows)} rows, {fmtInt(rt.index?.pending)} pending</dd>
    </dl>
    <div class="row wrap">
      <button class="btn" disabled={!!busy} onclick={() => act('load')}>{busy === 'load' ? 'Loading' : 'Load core'}</button>
      <button class="btn" disabled={!!busy} onclick={() => act('unload')}>{busy === 'unload' ? 'Unloading' : 'Unload'}</button>
      <button class="btn" disabled={!!busy} onclick={() => act('reindex')}>{busy === 'reindex' ? 'Reindexing' : 'Reindex'}</button>
    </div>
    {#if actErr}<p class="error-line">{actErr}</p>{/if}
    <section>
      <h3>Resident MPN state <span class="muted num">{fmtBytes(rt.residency?.cache_bytes ?? 0)} of {fmtBytes(rt.residency?.cache_limit_bytes ?? 0)}</span></h3>
      <ul>
        {#each rt.residency?.mpns ?? [] as m (m.mpn_id)}
          <li><span class="ellipsis grow">{m.name}</span><span class="muted num">{m.resident ? `resident, ${fmtBytes(m.bytes)}${m.last_used ? ', used ' + when(m.last_used) : ''}` : 'not resident'}</span></li>
        {/each}
      </ul>
    </section>
  {/if}
</div>

<style>
  .rt {
    padding: 10px 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .lead {
    font-size: 15px;
    font-weight: 500;
  }
  dl {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 2px 12px;
    margin: 0;
  }
  dt {
    color: var(--muted);
  }
  dd {
    margin: 0;
    overflow-wrap: anywhere;
  }
  .wrap {
    flex-wrap: wrap;
  }
  section {
    padding-top: 10px;
    border-top: 1px solid var(--line);
  }
  h3 {
    font-weight: 500;
    margin-bottom: 4px;
  }
  h3 .muted {
    font-weight: 400;
  }
  ul {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  li {
    display: flex;
    gap: 10px;
    line-height: 22px;
  }
</style>
