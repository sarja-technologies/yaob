<script lang="ts">
  import { app } from '../state.svelte';
  import { get, post, errText, fmtInt, fmtBytes, plural, type Domain, type Mpn } from '../api';
  import Icon from '../ui/Icon.svelte';

  type Action = 'disable' | 'enable' | 'uninstall' | 'install';
  interface Impact {
    action: Action;
    nodes_only_here: number;
    shared_nodes: number;
    shared_examples: string[];
    cross_mpn_edges: number;
    trained_nodes: number;
    explanation: string;
    entangled: boolean;
  }
  const VERB: Record<Action, string> = { disable: 'Disable', enable: 'Enable', uninstall: 'Uninstall', install: 'Install' };

  let open = $state<string | null>(null);
  let domains = $state.raw<Record<string, Domain[]>>({});
  let impact = $state.raw<(Impact & { id: string }) | null>(null);
  let busy = $state(false);
  let error = $state<string | null>(null);
  let filter = $state('');

  const shown = $derived(app.mpns.filter((m) => !filter.trim() || m.name.toLowerCase().includes(filter.trim().toLowerCase())));
  const total = $derived(app.status?.counts?.canonical_live_nodes ?? null);
  const anyTrained = $derived(app.mpns.some((m) => m.training !== 'untrained'));

  // the state words that matter for a decision; a fully ordinary MPN shows none
  function words(m: Mpn): string {
    const w: string[] = [];
    if (!m.installed) w.push('not installed');
    else if (!m.enabled) w.push('disabled');
    if (m.resident) w.push(`resident (${fmtBytes(m.resident_bytes)})`);
    if (m.training === 'partly trained') w.push(`partly trained (${fmtInt(m.trained_nodes)} of ${fmtInt(m.primary_nodes)})`);
    else if (m.training === 'trained') w.push('trained');
    if (m.validated) w.push('validated');
    return w.join(', ');
  }
  async function toggle(m: Mpn) {
    impact = null;
    error = null;
    if (open === m.id) {
      open = null;
      return;
    }
    open = m.id;
    app.boardApi?.fitRegion(m.id);
    loadDomains(m.id);
  }
  async function loadDomains(id: string) {
    try {
      const r = await get<{ domains: Domain[] }>(`api/mpns/${id}`);
      domains = { ...domains, [id]: r.domains };
    } catch (e) {
      error = errText(e);
    }
  }
  $effect(() => {
    app.head;
    if (open) loadDomains(open);
  });
  function tree(list: Domain[]): { d: Domain; depth: number }[] {
    const kids = new Map<string | null, Domain[]>();
    for (const d of list) {
      const k = d.parent_id ?? null;
      if (!kids.has(k)) kids.set(k, []);
      kids.get(k)!.push(d);
    }
    const out: { d: Domain; depth: number }[] = [];
    const walk = (p: string | null, depth: number) => {
      for (const d of kids.get(p) ?? []) {
        out.push({ d, depth });
        walk(d.id, depth + 1);
      }
    };
    walk(null, 0);
    return out;
  }
  async function preview(m: Mpn, action: Action) {
    error = null;
    try {
      const r = await get<Impact>(`api/mpns/${m.id}/impact?action=${action}`);
      impact = { ...r, id: m.id };
    } catch (e) {
      error = errText(e);
    }
  }
  async function confirm(m: Mpn) {
    if (!impact) return;
    const action = impact.action;
    busy = true;
    error = null;
    try {
      const r = await post(`api/mpns/${m.id}/${action}`);
      impact = null;
      await app.afterEdit(r);
      const now = r.mpn;
      // report the state the server holds now, not the one that was asked for
      app.toast(`${m.name} is now ${now.installed ? (now.enabled ? 'installed and enabled' : 'disabled') : 'not installed'} (revision ${r.rev}).`);
    } catch (e) {
      error = errText(e);
    }
    busy = false;
  }
</script>

<div class="lib">
  <div class="model" data-brand>
    <img src="brand/yaob/yaob-wordmark-cream.svg" width="104" height="47" alt="Yaob" />
    <div class="cfg">
      <span class="cfgname">{app.modelName}</span>
      {#if total != null}<span class="muted num">{fmtInt(total)} nodes, {app.mpns.length} MPNs</span>{/if}
    </div>
  </div>

  <div class="bar">
    <input class="field" placeholder="Filter MPNs" aria-label="Filter MPNs by name" bind:value={filter} />
    <button class="btn" onclick={() => (app.dialog = { kind: 'newMpn' })}>New MPN</button>
  </div>
  {#if app.mpns.length && !anyTrained}<p class="muted note">No MPN has trained state yet.</p>{/if}

  {#if !app.mpns.length}
    <p class="muted note">{app.offline ? 'The MPN list could not be loaded.' : 'Loading MPNs'}</p>
  {:else if !shown.length}
    <p class="muted note">No MPN name contains “{filter.trim()}”.</p>
  {/if}

  <ul class="mpns">
    {#each shown as m (m.id)}
      {@const w = words(m)}
      <li class:open={open === m.id} class:off={!m.enabled || !m.installed}>
        <button class="mrow" aria-expanded={open === m.id} title="Show on the board" onclick={() => toggle(m)}>
          <span class="chev"><Icon name={open === m.id ? 'chevron' : 'chevronR'} size={12} /></span>
          <span class="nm">
            <span class="ellipsis">{m.name}</span>
            {#if w}<span class="w ellipsis">{w}</span>{/if}
          </span>
          <span class="cnt num">{fmtInt(m.node_count)}</span>
        </button>
        {#if open === m.id}
          <div class="detail">
            {#if m.description}<p class="muted">{m.description}</p>{/if}
            <p>{plural(m.node_count, 'node')}, {fmtInt(m.shared_count)} shared with other MPNs. Trained state for {fmtInt(m.trained_nodes)} of {fmtInt(m.primary_nodes)} primary nodes.</p>

            {#if domains[m.id]?.length}
              <ul class="tree" aria-label="Domains of {m.name}">
                {#each tree(domains[m.id]) as t (t.d.id)}
                  <li style="padding-left:{t.depth * 14}px"><span class="ellipsis">{t.d.name}</span><span class="num muted">{fmtInt(t.d.node_count)}</span></li>
                {/each}
              </ul>
            {:else if domains[m.id]}
              <p class="muted">No domains.</p>
            {/if}

            {#if impact?.id === m.id}
              <div class="impact" role="group" aria-label="What {VERB[impact.action].toLowerCase()} would do">
                <p>{impact.explanation}</p>
                <dl>
                  <dt>Nodes only in this MPN</dt>
                  <dd class="num">{fmtInt(impact.nodes_only_here)}</dd>
                  <dt>Shared nodes</dt>
                  <dd class="num">{fmtInt(impact.shared_nodes)}</dd>
                  <dt>Edges to other MPNs</dt>
                  <dd class="num">{fmtInt(impact.cross_mpn_edges)}</dd>
                  <dt>Nodes with trained state</dt>
                  <dd class="num">{fmtInt(impact.trained_nodes)}</dd>
                </dl>
                <div class="row">
                  <button class="btn {impact.action === 'uninstall' ? 'danger' : 'primary'}" disabled={busy} onclick={() => confirm(m)}>{VERB[impact.action]} {m.name}</button>
                  <button class="btn" disabled={busy} onclick={() => (impact = null)}>Cancel</button>
                </div>
              </div>
            {:else}
              <div class="row acts">
                {#if !m.installed}
                  <button class="btn" onclick={() => preview(m, 'install')}>Install…</button>
                {:else}
                  <button class="btn" onclick={() => preview(m, m.enabled ? 'disable' : 'enable')}>{m.enabled ? 'Disable…' : 'Enable…'}</button>
                  <button class="btn" onclick={() => preview(m, 'uninstall')}>Uninstall…</button>
                {/if}
              </div>
            {/if}
            {#if error}<p class="error-line">{error}</p>{/if}
          </div>
        {/if}
      </li>
    {/each}
  </ul>
</div>

<style>
  .lib {
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .model {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 12px 12px 10px 6px;
  }
  .model img {
    display: block;
    flex: none;
  }
  .cfg {
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  .cfgname {
    font-size: 16px;
    line-height: 20px;
    font-weight: 500;
    letter-spacing: -0.01em;
  }
  .bar {
    display: flex;
    gap: 6px;
    padding: 0 12px 8px;
  }
  .note {
    padding: 0 12px 8px;
  }
  .mpns {
    list-style: none;
    margin: 0;
    padding: 0 0 12px;
  }
  .mrow {
    display: flex;
    align-items: center;
    gap: 6px;
    width: 100%;
    min-height: 30px;
    padding: 4px 12px 4px 8px;
    border: 0;
    background: transparent;
    text-align: left;
    cursor: pointer;
  }
  .mrow:hover {
    background: var(--raise);
  }
  .mrow:focus-visible {
    outline-offset: -2px;
  }
  li.open > .mrow {
    background: var(--raise);
  }
  .chev {
    display: inline-flex;
    color: var(--muted);
    flex: none;
  }
  .nm {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
  }
  .w {
    color: var(--muted);
    font-size: 12px;
    line-height: 16px;
  }
  li.off .nm > :first-child {
    color: var(--muted);
  }
  .cnt {
    color: var(--muted);
    flex: none;
  }
  .detail {
    padding: 6px 12px 12px 26px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    background: var(--raise);
  }
  .tree {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .tree li {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    line-height: 20px;
  }
  .impact {
    border: 1px solid #6a5220;
    border-radius: 2px;
    background: #17130a;
    padding: 8px 10px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  dl {
    display: grid;
    grid-template-columns: 1fr max-content;
    gap: 1px 12px;
    margin: 0;
  }
  dt {
    color: var(--muted);
  }
  dd {
    margin: 0;
    text-align: right;
  }
  .row {
    flex-wrap: wrap;
  }
</style>
