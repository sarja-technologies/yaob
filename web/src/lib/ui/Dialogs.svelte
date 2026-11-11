<script lang="ts">
  import { untrack } from 'svelte';
  import { app } from '../state.svelte';
  import { get, post, del, errText, ApiError, plural, human, type SchemaEntry } from '../api';
  import Modal from './Modal.svelte';

  const d = $derived(app.dialog);
  const close = () => (app.dialog = null);

  // ---------------------------------------------------------------- new node
  let nn = $state({ title: '', type: 'knowledge', schema: '3cn@1', statement: '', mpns: [''] as string[] });
  let schemas = $state.raw<SchemaEntry[]>([]);
  let busy = $state(false);
  let error = $state<string | null>(null);
  let problems = $state<string[]>([]);

  function regionAt(x: number | null, y: number | null): string {
    const ms = app.board?.mpns ?? [];
    if (x != null && y != null) {
      const hit = ms.filter((m) => Math.hypot(m.x - x, m.y - y) <= m.r).sort((a, b) => a.r - b.r)[0];
      if (hit) return hit.id;
    }
    return '';
  }
  $effect(() => {
    const d = app.dialog;
    untrack(() => openNewNode(d));
  });
  function openNewNode(d: typeof app.dialog) {
    if (d?.kind === 'newNode') {
      nn = { title: '', type: 'knowledge', schema: nn.schema, statement: '', mpns: [regionAt(d.x, d.y)] };
      error = null;
      problems = [];
      get<{ schemas: SchemaEntry[] }>('api/schemas')
        .then((r) => {
          schemas = r.schemas;
          if (!r.schemas.some((s) => `${s.id}@${s.version}` === nn.schema) && r.schemas[0]) nn.schema = `${r.schemas[0].id}@${r.schemas[0].version}`;
        })
        .catch(() => {});
    }
  }
  async function createNode() {
    if (d?.kind !== 'newNode') return;
    const [sid, sv] = nn.schema.split('@');
    const mems = [...new Set(nn.mpns.filter(Boolean))].map((mpn_id) => ({ mpn_id }));
    busy = true;
    error = null;
    problems = [];
    try {
      const body: any = {
        title: nn.title.trim(),
        type: nn.type,
        schema: { id: sid, version: Number(sv) },
        cells: { knowledge: { statements: nn.statement.trim() ? [{ text: nn.statement.trim() }] : [] }, thinking: {}, reasoning: {} },
        memberships: mems,
      };
      if (d.x != null && d.y != null) body.position = { x: d.x, y: d.y };
      const r = await post('api/nodes', body);
      close();
      await app.afterEdit(r);
      await app.goToNode(r.id, r.node?.position);
      const fx = r.effects;
      app.toast(`Created “${r.node.title}”. Live now: ${fx.live_now.join(', ')}. After indexing: ${fx.after_indexing.join(', ')}.`);
    } catch (e) {
      error = errText(e);
      problems = e instanceof ApiError ? (e.detail?.problems ?? []) : [];
    }
    busy = false;
  }

  // ---------------------------------------------------------------- new MPN
  let nm = $state({ name: '', description: '' });
  $effect(() => {
    const d = app.dialog;
    untrack(() => {
      if (d?.kind === 'newMpn') {
        nm = { name: '', description: '' };
        error = null;
      }
    });
  });
  async function createMpn() {
    busy = true;
    error = null;
    try {
      const r = await post('api/mpns', { name: nm.name.trim(), description: nm.description.trim() });
      close();
      await app.afterEdit(r);
      app.boardApi?.fitRegion(r.mpn.id);
      app.toast(`Created the MPN “${r.mpn.name}”. It is empty: add nodes to it from a node's memberships or with New node.`);
    } catch (e) {
      error = errText(e);
    }
    busy = false;
  }

  // ---------------------------------------------------------------- delete (preview first, recoverable)
  interface Preview {
    node: { id: string; title: string };
    memberships: { mpn_name: string; role: string }[];
    edges: number;
    learned_rows: any[];
    checkpoint: string | null;
    note: string;
  }
  const MAX_DELETE = 60;
  let previews = $state.raw<Preview[] | null>(null);
  $effect(() => {
    const d = app.dialog;
    untrack(() => {
      if (d?.kind !== 'delete') return;
      previews = null;
      error = null;
      if (d.ids.length > MAX_DELETE) return;
      Promise.all(d.ids.map((id) => get<Preview>(`api/nodes/${id}/delete-preview`)))
        .then((p) => (previews = p))
        .catch((e) => (error = errText(e)));
    });
  });
  async function doDelete() {
    if (d?.kind !== 'delete' || !previews) return;
    const ids = d.ids;
    busy = true;
    try {
      let last: any;
      for (const id of ids) last = await del(`api/nodes/${id}`);
      close();
      app.selection = [];
      await app.afterEdit(last);
      app.toast(`Deleted ${ids.length === 1 ? '“' + previews[0].node.title + '”' : ids.length + ' nodes'}. Deleted nodes can be restored.`, 'info', { label: 'Restore', run: () => restore(ids) });
    } catch (e) {
      error = errText(e);
    }
    busy = false;
  }
  async function restore(ids: string[]) {
    try {
      let last: any;
      for (const id of ids) last = await post(`api/nodes/${id}/restore`);
      await app.afterEdit(last);
      app.select(ids, ids.length === 1);
      app.toast(`Restored ${ids.length === 1 ? 'the node' : ids.length + ' nodes'}.`);
    } catch (e) {
      app.toast(errText(e), 'error');
    }
  }

  // ---------------------------------------------------------------- connect
  let types = $state.raw<string[]>([]);
  let rel = $state('related_to');
  let flip = $state(false);
  $effect(() => {
    const d = app.dialog;
    untrack(() => {
      if (d?.kind !== 'connect') return;
      error = null;
      flip = false;
      get<{ types: string[] }>('api/edge-types')
        .then((r) => {
          types = r.types;
          if (!r.types.includes(rel)) rel = r.types[0] ?? 'related_to';
        })
        .catch((e) => (error = errText(e)));
    });
  });
  async function connect() {
    if (d?.kind !== 'connect') return;
    busy = true;
    error = null;
    try {
      const r = await post('api/edges', flip ? { src: d.dst, dst: d.src, type: rel } : { src: d.src, dst: d.dst, type: rel });
      close();
      app.tool = 'pan';
      await app.afterEdit(r);
      app.toast(`Connected with “${human(rel)}”. This is a model edit (revision ${r.rev}).`);
    } catch (e) {
      error = errText(e);
    }
    busy = false;
  }
</script>

{#if d?.kind === 'newNode'}
  <Modal title="New node" onclose={close} width={460}>
    <form onsubmit={(e) => (e.preventDefault(), createNode())}>
      <div>
        <label class="lbl" for="nn-title">Title</label>
        <!-- svelte-ignore a11y_autofocus -->
        <input id="nn-title" class="field" bind:value={nn.title} autofocus required maxlength="200" />
      </div>
      <div class="two">
        <div>
          <label class="lbl" for="nn-type">Type</label>
          <select id="nn-type" class="field" bind:value={nn.type}>
            <option value="knowledge">Knowledge</option>
            <option value="procedure">Procedure</option>
          </select>
        </div>
        <div>
          <label class="lbl" for="nn-schema">Schema</label>
          <select id="nn-schema" class="field" bind:value={nn.schema}>
            {#each schemas as s (s.id + s.version)}<option value="{s.id}@{s.version}">{s.name} v{s.version}</option>{/each}
            {#if !schemas.length}<option value={nn.schema}>{nn.schema.replace('@', ' v')}</option>{/if}
          </select>
        </div>
      </div>
      <div>
        <label class="lbl" for="nn-st">First statement</label>
        <textarea id="nn-st" class="field" rows="2" bind:value={nn.statement} placeholder="One sentence this node states"></textarea>
      </div>
      <fieldset>
        <legend class="lbl">Member of</legend>
        {#each nn.mpns as _, i (i)}
          <div class="row">
            <select class="field" aria-label="MPN {i + 1}" bind:value={nn.mpns[i]} required={i === 0}>
              <option value="" disabled>Choose an MPN</option>
              {#each app.mpns as m (m.id)}<option value={m.id}>{m.name}{i === 0 ? '' : ' (shared)'}</option>{/each}
            </select>
            {#if nn.mpns.length > 1}<button type="button" class="btn" onclick={() => (nn.mpns = nn.mpns.filter((_, j) => j !== i))}>Remove</button>{/if}
          </div>
        {/each}
        <button type="button" class="btn add" onclick={() => (nn.mpns = [...nn.mpns, ''])}>Add another MPN</button>
      </fieldset>
      {#if error}<p class="error-line">{error}</p>{/if}
      {#each problems as p}<p class="problem">{p}</p>{/each}
      <div class="actions">
        <button type="button" class="btn" onclick={close}>Cancel</button>
        <button type="submit" class="btn primary" disabled={busy || !nn.title.trim() || !nn.mpns[0]}>Create node</button>
      </div>
    </form>
  </Modal>
{:else if d?.kind === 'newMpn'}
  <Modal title="New MPN" onclose={close} width={420}>
    <form onsubmit={(e) => (e.preventDefault(), createMpn())}>
      <div>
        <label class="lbl" for="nm-name">Name</label>
        <!-- svelte-ignore a11y_autofocus -->
        <input id="nm-name" class="field" bind:value={nm.name} autofocus required maxlength="120" />
      </div>
      <div>
        <label class="lbl" for="nm-desc">Description</label>
        <textarea id="nm-desc" class="field" rows="2" bind:value={nm.description}></textarea>
      </div>
      {#if error}<p class="error-line">{error}</p>{/if}
      <div class="actions">
        <button type="button" class="btn" onclick={close}>Cancel</button>
        <button type="submit" class="btn primary" disabled={busy || !nm.name.trim()}>Create MPN</button>
      </div>
    </form>
  </Modal>
{:else if d?.kind === 'delete'}
  <Modal title={d.ids.length === 1 ? 'Delete this node?' : `Delete ${d.ids.length} nodes?`} onclose={close} width={460}>
    {#if d.ids.length > MAX_DELETE}
      <p>Select at most {MAX_DELETE} nodes to delete at once, so each can be previewed.</p>
      <div class="actions"><button class="btn" onclick={close}>Close</button></div>
    {:else if error && !previews}
      <p class="error-line">{error}</p>
      <div class="actions"><button class="btn" onclick={close}>Close</button></div>
    {:else if !previews}
      <p class="muted">Reading what the deletion affects</p>
    {:else}
      {@const edges = previews.reduce((a, p) => a + p.edges, 0)}
      {@const rows = previews.reduce((a, p) => a + p.learned_rows.length, 0)}
      {@const mems = previews.reduce((a, p) => a + p.memberships.length, 0)}
      <div class="what">
        {#if previews.length === 1}
          <p class="strong">{previews[0].node.title}</p>
          <p>Member of {previews[0].memberships.map((m) => `${m.mpn_name} (${m.role})`).join(', ')}.</p>
        {:else}
          <p class="strong">{previews.slice(0, 4).map((p) => p.node.title).join(', ')}{previews.length > 4 ? ` and ${previews.length - 4} more` : ''}</p>
          <p>{plural(mems, 'membership')} in total.</p>
        {/if}
        <p>{plural(edges, 'edge')} to other nodes and {plural(rows, 'learned row')} are affected.</p>
        <p class="muted">{previews[0].note}</p>
      </div>
      {#if error}<p class="error-line">{error}</p>{/if}
      <div class="actions">
        <button class="btn" onclick={close}>Cancel</button>
        <button class="btn danger" disabled={busy} onclick={doDelete}>{previews.length === 1 ? 'Delete node' : `Delete ${previews.length} nodes`}</button>
      </div>
    {/if}
  </Modal>
{:else if d?.kind === 'connect'}
  <Modal title="Connect nodes" onclose={close} width={420}>
    <form onsubmit={(e) => (e.preventDefault(), connect())}>
      <div class="pair">
        <span class="ellipsis strong">{flip ? d.dstTitle : d.srcTitle}</span>
        <select class="field mono" aria-label="Relation type" bind:value={rel}>
          {#each types as t (t)}<option value={t}>{t}</option>{/each}
          {#if !types.length}<option value={rel}>{rel}</option>{/if}
        </select>
        <span class="ellipsis strong">{flip ? d.srcTitle : d.dstTitle}</span>
      </div>
      <p class="muted">This adds an edge to the model and creates a revision.</p>
      {#if error}<p class="error-line">{error}</p>{/if}
      <div class="actions">
        <button type="button" class="btn left" onclick={() => (flip = !flip)}>Swap direction</button>
        <button type="button" class="btn" onclick={close}>Cancel</button>
        <button type="submit" class="btn primary" disabled={busy}>Connect</button>
      </div>
    </form>
  </Modal>
{/if}

<style>
  form {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .two {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  fieldset {
    border: 0;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-width: 0;
  }
  legend {
    padding: 0;
  }
  .add {
    align-self: flex-start;
  }
  .actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    margin-top: 2px;
  }
  .actions .left {
    margin-right: auto;
  }
  .what {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .strong {
    font-weight: 500;
  }
  .pair {
    display: grid;
    grid-template-columns: 1fr;
    gap: 6px;
  }
</style>
