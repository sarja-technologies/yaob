<script lang="ts">
  import { untrack } from 'svelte';
  import { app } from '../state.svelte';
  import { insp } from './inspector.svelte';
  import { get, post, put, del, patch, errText, fmtBytes, fmtInt, human, plural, when, type NodeEdge } from '../api';
  import Field from './Field.svelte';
  import Icon from '../ui/Icon.svelte';

  const n = $derived(insp.node);
  const d = $derived(insp.draft);
  const deleted = $derived(n?.status === 'tombstone');
  const many = $derived(app.selection.length > 1);
  let addMpn = $state('');
  let memErr = $state<string | null>(null);
  let history = $state.raw<any[] | null>(null);
  let histOpen = $state(false);
  let busy = $state(false);

  $effect(() => {
    const id = app.inspected;
    untrack(() => {
      history = null;
      histOpen = false;
      memErr = null;
      insp.load(id);
    });
  });
  // the model moved on (an edit here, an undo, or another client): re-read, keeping an unsaved draft
  let seen = 0;
  $effect(() => {
    const h = app.head;
    untrack(() => {
      if (seen && h !== seen && app.inspected) {
        insp.load(app.inspected, insp.dirty);
        if (histOpen) loadHistory();
      }
      seen = h;
    });
  });

  const freeMpns = $derived(app.mpns.filter((m) => !n?.memberships.some((x) => x.mpn_id === m.id)));

  async function act(fn: () => Promise<any>, done?: string) {
    busy = true;
    try {
      const r = await fn();
      await app.afterEdit(r);
      if (done) app.toast(done);
      return r;
    } catch (e) {
      app.toast(errText(e), 'error');
    } finally {
      busy = false;
    }
  }
  async function addMembership() {
    if (!n || !addMpn) return;
    memErr = null;
    try {
      const r = await put(`api/nodes/${n.id}/memberships/${addMpn}`, { role: 'shared' });
      addMpn = '';
      await app.afterEdit(r);
    } catch (e) {
      memErr = errText(e);
    }
  }
  async function removeMembership(mpnId: string) {
    if (!n) return;
    memErr = null;
    try {
      const r = await del(`api/nodes/${n.id}/memberships/${mpnId}`);
      await app.afterEdit(r);
      app.toast('Membership removed. The node and its other memberships are unchanged.');
    } catch (e) {
      // the API explains when this is the node's only membership
      memErr = errText(e);
    }
  }
  async function disconnect(e: NodeEdge) {
    await act(() => del(`api/edges/${e.id}`), 'Disconnected.');
  }
  async function loadHistory() {
    if (!n) return;
    try {
      history = (await get<{ versions: any[] }>(`api/nodes/${n.id}/history`)).versions;
    } catch (e) {
      app.toast(errText(e), 'error');
    }
  }
  function menu(e: MouseEvent) {
    if (!n) return;
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const id = n.id;
    app.openMenu(r.right - 210, r.bottom + 4, [
      { label: 'Show on the board', run: () => app.goToNode(id, n.position) },
      { label: 'Connect to…', run: () => (app.goToNode(id, n.position), app.boardApi?.startConnect(id)) },
      { label: 'Duplicate as new node', run: () => act(() => post(`api/nodes/${id}/duplicate`, { mode: 'canonical' })).then((r) => r && (app.select([r.id]), app.toast(r.note))) },
      {
        label: 'Add reference on the board',
        run: async () => {
          try {
            const r = await post(`api/nodes/${id}/duplicate`, { mode: 'reference' });
            await app.loadBoard();
            app.toast(r.note);
          } catch (err) {
            app.toast(errText(err), 'error');
          }
        },
      },
      n.status === 'disabled' ? { label: 'Enable', run: () => act(() => patch(`api/nodes/${id}`, { status: 'live' }), 'Node enabled.') } : { label: 'Disable', run: () => act(() => patch(`api/nodes/${id}`, { status: 'disabled' }), 'Node disabled.') },
      'sep',
      { label: 'Delete…', danger: true, run: () => (app.dialog = { kind: 'delete', ids: [id] }) },
    ]);
  }
  const isUrl = (s: string) => /^https?:\/\//i.test(s);
  const statusWord = (s: string) => (s === 'tombstone' ? 'deleted' : s);
</script>

{#if many}
  <div class="pad">
    <h2 class="h">{app.selection.length} nodes selected</h2>
    <div class="row wrap">
      <button class="btn" onclick={() => app.boardApi?.fitSelection()}>Fit selection</button>
      <button class="btn danger" onclick={() => (app.dialog = { kind: 'delete', ids: app.selection })}>Delete…</button>
    </div>
    <p class="muted">Drag any selected node to move them together. Click one node to open it here.</p>
  </div>
{:else if !app.inspected}
  <div class="empty">
    <p class="lead">No node is open</p>
    <p class="muted">Click a node on the board, or find one by name.</p>
    <div class="row">
      <button class="btn" onclick={() => (document.querySelector('.search input') as HTMLElement | null)?.focus()}>Search nodes</button>
      <button class="btn" onclick={() => (app.dialog = { kind: 'newNode', x: null, y: null })}>New node</button>
    </div>
  </div>
{:else if insp.error}
  <div class="empty">
    <p class="error-line">{insp.error}</p>
    <button class="btn" onclick={() => insp.load(app.inspected)}>Try again</button>
  </div>
{:else if !n || !d}
  <div class="empty"><p class="muted">Loading the node</p></div>
{:else}
  <div class="insp">
    <div class="top">
      <input class="title" aria-label="Title" bind:value={d.title} disabled={deleted} />
      <button class="btn icon quiet" aria-label="More actions for this node" title="More actions" disabled={deleted} onclick={menu}><Icon name="more" /></button>
    </div>
    <div class="meta">
      <select class="field tight" aria-label="Type" bind:value={d.type} disabled={deleted}>
        <option value="knowledge">Knowledge</option>
        <option value="procedure">Procedure</option>
      </select>
      <select class="field tight" aria-label="Validation" bind:value={d.validation} disabled={deleted}>
        <option value="unverified">Unverified</option>
        <option value="source_backed">Source backed</option>
        <option value="checked">Checked</option>
        <option value="failed">Failed</option>
      </select>
      <span class="state" class:off={n.status !== 'live'}>{statusWord(n.status)}</span>
      <span class="mono id" title="Node id, uid {n.uid}">{n.id}</span>
    </div>

    {#if deleted}
      <div class="problem banner">
        <span>This node is deleted. It is out of search and routing, and can be restored.</span>
        <button class="btn" disabled={busy} onclick={() => act(() => post(`api/nodes/${n.id}/restore`), 'Node restored.')}>Restore</button>
      </div>
    {/if}

    {#if insp.dirty || insp.saveError || insp.problems.length}
      <div class="savebar">
        <button class="btn primary" disabled={insp.saving || !insp.dirty} onclick={() => insp.save()}>{insp.saving ? 'Saving' : 'Save'}</button>
        <button class="btn" disabled={insp.saving || !insp.dirty} onclick={() => insp.revert()}>Revert</button>
        <span class="muted">Unsaved changes</span>
      </div>
    {/if}
    {#if insp.saveError}<p class="error-line">{insp.saveError}</p>{/if}
    {#each insp.problems as p}<p class="problem">{p}</p>{/each}
    {#if insp.effects && !insp.dirty}
      <div class="effects" role="status">
        <p>Saved as revision <span class="mono">{insp.effects.rev}</span>.</p>
        <dl>
          <dt>Live now</dt>
          <dd>{insp.effects.live_now.join(', ')}</dd>
          {#if insp.effects.after_indexing.length}
            <dt>After indexing</dt>
            <dd>{insp.effects.after_indexing.join(', ')}</dd>
          {/if}
          {#if insp.effects.after_training.length}
            <dt>After training</dt>
            <dd>{insp.effects.after_training.join(', ')}</dd>
          {/if}
        </dl>
      </div>
    {/if}
    {#if n.problems.length && !insp.problems.length}
      {#each n.problems as p}<p class="problem">{p}</p>{/each}
    {/if}

    <div class="sec">
      <label class="fl" for="insp-summary">Summary</label>
      <textarea id="insp-summary" class="field" rows="2" bind:value={d.summary} disabled={deleted}></textarea>
    </div>

    <section class="sec">
      <h3>Member of</h3>
      <ul class="list">
        {#each n.memberships as m (m.mpn_id)}
          <li>
            <button class="linkbtn ellipsis" title="Show this MPN on the board" onclick={() => app.boardApi?.fitRegion(m.mpn_id)}>{m.mpn_name}</button>
            <span class="muted">{m.role}{m.mpn_installed ? (m.mpn_enabled ? '' : ', MPN disabled') : ', MPN not installed'}</span>
            <button class="btn icon quiet" aria-label="Remove the membership in {m.mpn_name}" title="Remove this membership (the node stays)" disabled={deleted} onclick={() => removeMembership(m.mpn_id)}><Icon name="close" size={14} /></button>
          </li>
        {/each}
      </ul>
      {#if memErr}<p class="error-line">{memErr}</p>{/if}
      {#if !deleted && freeMpns.length}
        <form class="row" onsubmit={(e) => (e.preventDefault(), addMembership())}>
          <select class="field" aria-label="MPN to add this node to" bind:value={addMpn}>
            <option value="">Add to another MPN</option>
            {#each freeMpns as m (m.id)}<option value={m.id}>{m.name}</option>{/each}
          </select>
          <button class="btn" type="submit" disabled={!addMpn}>Add</button>
        </form>
      {/if}
    </section>

    {#each n.schema.roles as role (role.key)}
      <section class="sec cell" data-cell={role.key}>
        <h3 title={role.description}>{role.label}{#if role.label.toLowerCase() !== role.runtime_role && role.runtime_role !== 'none'}{" "}<span class="muted">read as {role.runtime_role}</span>{/if}</h3>
        {#each role.fields as field (field.key)}
          {#if !(n.type !== 'procedure' && ['impl', 'params', 'formula', 'unit'].includes(field.key) && d.type !== 'procedure' && d.cells[role.key]?.[field.key] == null)}
            <Field {field} bind:cell={d.cells[role.key]} sources={n.sources} disabled={deleted} uid="c-{role.key}" />
          {/if}
        {/each}
      </section>
    {/each}

    <section class="sec">
      <h3>Sources</h3>
      {#if n.sources.length}
        <ul class="list src">
          {#each n.sources as s (s.id)}
            <li>
              <div class="grow">
                {#if isUrl(s.url)}<a href={s.url} target="_blank" rel="noreferrer">{s.title}</a>{:else}<span>{s.title}</span>{/if}
                <div class="muted small">
                  {#if s.license}Licence {#if isUrl(s.license_url)}<a href={s.license_url} target="_blank" rel="noreferrer">{s.license}</a>{:else}{s.license}{/if}{/if}
                  {#if s.revision}
                    {s.license ? ', ' : ''}revision
                    {#if isUrl(s.url)}<a class="mono" href={s.url} target="_blank" rel="noreferrer">{s.revision}</a>{:else}<span class="mono">{s.revision}</span>{/if}
                  {/if}
                  {#if s.locator}{" "}<span class="mono">{s.locator}</span>{/if}
                </div>
              </div>
            </li>
          {/each}
        </ul>
      {:else}
        <p class="muted">No sources are linked to this node.</p>
      {/if}
    </section>

    <section class="sec">
      <h3>Links <span class="muted num">{fmtInt(n.edge_count)}</span></h3>
      {#if insp.edges.length}
        <ul class="list">
          {#each insp.edges as e (e.id)}
            <li>
              <span class="dir" title={e.direction === 'out' ? 'From this node' : 'To this node'}><Icon name={e.direction === 'out' ? 'arrowR' : 'arrowL'} size={14} /></span>
              <span class="mono rel">{e.type}</span>
              <button class="linkbtn ellipsis" title="Go to this node on the board" onclick={() => app.goToNode(e.other.id)}>{e.other.title}</button>
              <button class="btn icon quiet" aria-label="Disconnect from {e.other.title}" title="Disconnect" disabled={deleted || busy} onclick={() => disconnect(e)}><Icon name="close" size={14} /></button>
            </li>
          {/each}
        </ul>
        {#if insp.edges.length < n.edge_count}<button class="btn add" onclick={() => insp.moreEdges()}>Show more ({fmtInt(n.edge_count - insp.edges.length)} left)</button>{/if}
      {:else}
        <p class="muted">Not connected to other nodes.</p>
      {/if}
    </section>

    <section class="sec">
      <h3>Learned state</h3>
      <p>{n.state.explain}</p>
      <dl class="kv">
        <dt>Index</dt>
        <dd>{n.state.index}</dd>
        <dt>Trained state</dt>
        <dd>{n.state.learned}{n.state.rows.length ? `, ${plural(n.state.rows.length, 'row')}` : ''}</dd>
        {#if n.state.checkpoint}
          <dt>Checkpoint</dt>
          <dd class="mono">{n.state.checkpoint}</dd>
        {/if}
        <dt>Cost</dt>
        <dd>
          record {fmtBytes(n.cost.record_bytes)}, about {fmtInt(n.cost.text_tokens_estimate)} text tokens{#if n.cost.learned_parameters}, {fmtInt(n.cost.learned_parameters)} learned routing parameters of its own ({fmtBytes(n.cost.learned_bytes_f32)}); its content is learned by its MPN's shared specialist{:else}, no learned parameters yet{/if}
        </dd>
      </dl>
    </section>

    <details class="sec" bind:open={histOpen} ontoggle={() => histOpen && !history && loadHistory()}>
      <summary><h3>History</h3></summary>
      {#if history}
        <ul class="list hist">
          {#each history as v (v.rev_from)}
            <li>
              <span class="mono">rev {v.rev_from}</span>
              <span class="grow ellipsis" title={v.message}>{v.message}</span>
              <span class="muted small">{v.actor}, {when(v.created)}</span>
            </li>
          {/each}
        </ul>
      {:else}
        <p class="muted">Loading</p>
      {/if}
    </details>
  </div>
{/if}

<style>
  .pad,
  .insp {
    padding: 10px 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .h {
    font-size: 15px;
    font-weight: 500;
  }
  .wrap {
    flex-wrap: wrap;
  }
  .empty {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 24px;
    text-align: center;
  }
  .lead {
    font-size: 15px;
    font-weight: 500;
  }
  .top {
    display: flex;
    gap: 4px;
    align-items: center;
  }
  .title {
    flex: 1;
    min-width: 0;
    font-size: 16px;
    line-height: 22px;
    font-weight: 500;
    letter-spacing: -0.01em;
    padding: 2px 6px;
    margin-left: -7px;
    border: 1px solid transparent;
    border-radius: 2px;
    background: transparent;
    color: var(--white);
  }
  .title:hover:not(:disabled) {
    border-color: var(--line);
  }
  .title:focus-visible {
    border-color: var(--line);
    background: var(--void);
  }
  .meta {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }
  .tight {
    width: auto;
  }
  .state {
    color: var(--muted);
  }
  .state.off {
    color: var(--gold);
  }
  .id {
    margin-left: auto;
    color: var(--muted);
  }
  .banner {
    display: flex;
    align-items: center;
    gap: 8px;
    justify-content: space-between;
  }
  .savebar {
    position: sticky;
    top: 0;
    z-index: 2;
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    margin: 0 -12px;
    background: var(--surface);
    border-bottom: 1px solid var(--line);
  }
  .effects {
    border: 1px solid var(--line);
    border-radius: 2px;
    padding: 4px 8px;
    background: var(--raise);
  }
  dl {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 1px 10px;
    margin: 2px 0 0;
  }
  dt {
    color: var(--muted);
  }
  dd {
    margin: 0;
    min-width: 0;
    overflow-wrap: anywhere;
  }
  .sec {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding-top: 10px;
    border-top: 1px solid var(--line-soft);
  }
  h3 {
    font-weight: 500;
    font-size: 13px;
  }
  h3 .muted {
    font-weight: 400;
  }
  .fl {
    color: var(--muted);
  }
  .list {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .list li {
    display: flex;
    align-items: center;
    gap: 8px;
    min-height: 26px;
    min-width: 0;
  }
  .list li > .muted {
    flex: none;
  }
  .src li {
    align-items: flex-start;
    padding: 2px 0;
  }
  .small {
    font-size: 12px;
  }
  .linkbtn {
    flex: 1 1 auto;
    min-width: 0;
    border: 0;
    background: transparent;
    padding: 0;
    text-align: left;
    text-decoration: underline;
    text-decoration-color: #5a5a5a;
    text-underline-offset: 2px;
    cursor: pointer;
  }
  .linkbtn:hover {
    text-decoration-color: var(--white);
  }
  .dir {
    display: inline-flex;
    color: var(--muted);
    flex: none;
  }
  .rel {
    color: var(--muted);
    flex: none;
  }
  .add {
    align-self: flex-start;
  }
  summary {
    cursor: pointer;
    list-style-position: inside;
  }
  summary h3 {
    display: inline;
  }
  .hist li {
    gap: 10px;
  }
  .hist .muted {
    flex: none;
  }
</style>
