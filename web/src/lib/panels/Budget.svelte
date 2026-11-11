<script lang="ts">
  import { onMount } from 'svelte';
  import { app } from '../state.svelte';
  import { get, put, errText } from '../api';

  const FIELDS: { k: string; label: string; unit: string; step?: number; hint?: string }[] = [
    { k: 'total_gib', label: 'Total memory limit', unit: 'GiB', step: 0.5, hint: '0 derives it from available memory minus the reserve' },
    { k: 'reserve_gib', label: 'Reserve for the system', unit: 'GiB', step: 0.5 },
    { k: 'resident_cache_mib', label: 'Resident MPN cache', unit: 'MiB' },
    { k: 'max_resident_mpns', label: 'MPNs kept resident', unit: 'MPNs' },
    { k: 'max_active_mpns', label: 'MPNs consulted per request', unit: 'MPNs' },
    { k: 'retrieval_candidates', label: 'Retrieval candidates', unit: 'nodes' },
    { k: 'max_active_adapters', label: 'Specialists applied at once', unit: 'adapters', hint: '0 by default: in held-out evaluation one specialist lowered accuracy (81 % vs 85 %) and made the model answer questions it should decline' },
    { k: 'selected_nodes', label: 'Nodes selected', unit: 'nodes' },
    { k: 'text_nodes', label: 'Of those, also given as text', unit: 'nodes' },
    { k: 'traversal_depth', label: 'Traversal depth', unit: 'hops' },
    { k: 'context_tokens', label: 'Context', unit: 'tokens', hint: 'Question, instructions, conversation and node text plus the answer' },
    { k: 'output_tokens', label: 'Answer cap', unit: 'tokens', hint: 'Longest answer. A chat client’s own limit is honoured up to this' },
    { k: 'concurrent_requests', label: 'Concurrent requests', unit: 'requests' },
    { k: 'prefetch_mib', label: 'Prefetch', unit: 'MiB' },
    { k: 'disk_cache_gib', label: 'Disk cache', unit: 'GiB', step: 0.5 },
  ];
  const PRESETS = ['conservative', 'balanced', 'expanded', 'specialists (experimental)'] as const;
  // the same rule the server applies (mpnstudio/runtime/budget.py): the answer cap must leave 512 + 16 tokens for input
  const tokenProblem = $derived.by(() => {
    const c = Number(draft.context_tokens), o = Number(draft.output_tokens);
    if (!Number.isFinite(c) || !Number.isFinite(o) || c <= 0 || o <= 0) return 'Context and answer cap must be positive numbers.';
    if (o + 528 > c) return `An answer cap of ${o} leaves no room for the question in a ${c}-token context. The cap can be at most ${c - 528}.`;
    return null;
  });

  let data = $state.raw<{ budget: Record<string, any>; presets: Record<string, Record<string, number>>; memory: any; note: string } | null>(null);
  let draft = $state<Record<string, any>>({});
  let error = $state<string | null>(null);
  let saveErr = $state<string | null>(null);
  let busy = $state(false);
  let saved = $state(false);

  const changed = $derived(!!data && FIELDS.some((f) => Number(draft[f.k]) !== Number(data!.budget[f.k])));

  async function load() {
    try {
      data = await get('api/budget');
      draft = { ...data!.budget };
      error = null;
    } catch (e) {
      error = errText(e);
    }
  }
  onMount(load);

  async function send(body: Record<string, any>) {
    busy = true;
    saveErr = null;
    saved = false;
    try {
      await put('api/budget', body);
      await load();
      app.loadStatus();
      saved = true;
    } catch (e) {
      saveErr = errText(e);
    }
    busy = false;
  }
  function save() {
    if (!data) return;
    const body: Record<string, any> = { preset: 'custom' };
    for (const f of FIELDS) if (Number(draft[f.k]) !== Number(data.budget[f.k])) body[f.k] = Number(draft[f.k]);
    send(body);
  }
</script>

<div class="budget">
  {#if error}
    <p class="error-line">{error}</p>
    <button class="btn" onclick={load}>Try again</button>
  {:else if !data}
    <p class="muted">Loading the budget</p>
  {:else}
    <p>{data.memory.total_gib} GiB installed, {data.memory.available_gib} GiB available now. With {data.memory.reserve_gib} GiB reserved, {data.memory.usable_gib} GiB can be used.</p>
    <div class="presets" role="radiogroup" aria-label="Preset">
      {#each PRESETS as p (p)}
        <button role="radio" aria-checked={data.budget.preset === p} disabled={busy} onclick={() => send({ preset: p, apply_preset: true })}>
          <span class="pn">{p[0].toUpperCase() + p.slice(1)}</span>
          <span class="num muted">{data.presets[p].total_gib} GiB total</span>
          <span class="num muted">{data.presets[p].max_active_mpns} MPNs, {data.presets[p].selected_nodes} nodes per request</span>
        </button>
      {/each}
    </div>
    {#if data.budget.repair_note}<p class="error-line">{data.budget.repair_note}</p>{/if}
    <p class="muted">{data.budget.preset === 'custom' ? 'The values below were set by hand. ' : ''}{data.note}</p>

    <form onsubmit={(e) => (e.preventDefault(), save())}>
      {#each FIELDS as f (f.k)}
        <div class="frow">
          <label for="b-{f.k}" title={f.k}>{f.label}{#if f.hint}<span class="muted hint">{f.hint}</span>{/if}</label>
          <input id="b-{f.k}" class="field num" type="number" step={f.step ?? 1} min="0" bind:value={draft[f.k]} class:bad={Number(draft[f.k]) !== Number(data.budget[f.k])} />
          <span class="unit muted">{f.unit}</span>
        </div>
      {/each}
      {#if tokenProblem}<p class="error-line">{tokenProblem}</p>{/if}
      {#if saveErr}<p class="error-line">{saveErr}</p>{/if}
      <div class="row">
        <button class="btn primary" type="submit" disabled={!changed || busy || !!tokenProblem}>Save budget</button>
        <button class="btn" type="button" disabled={!changed || busy} onclick={() => (draft = { ...data!.budget })}>Revert</button>
        {#if saved && !changed}<span class="muted">Saved.</span>{/if}
      </div>
    </form>
  {/if}
</div>

<style>
  .budget {
    padding: 10px 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .presets {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 6px;
  }
  .presets button {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 1px;
    padding: 6px 8px;
    border: 1px solid var(--line);
    border-radius: 2px;
    background: transparent;
    text-align: left;
    cursor: pointer;
  }
  .presets button:hover:not(:disabled) {
    background: var(--raise);
  }
  .presets button[aria-checked='true'] {
    border-color: var(--white);
    background: #1e1e1e;
  }
  .pn {
    font-weight: 500;
  }
  form {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding-top: 10px;
    border-top: 1px solid var(--line);
  }
  .frow {
    display: grid;
    grid-template-columns: 1fr 92px 62px;
    gap: 8px;
    align-items: center;
  }
  .frow label {
    min-width: 0;
  }
  .hint {
    display: block;
    font-size: 12px;
    line-height: 15px;
  }
  .frow input {
    text-align: right;
  }
  .row {
    margin-top: 8px;
  }
</style>
