<script lang="ts">
  // One typed field of a node cell, rendered as an editor chosen by the schema's field type.
  import { get, errText, type NodeDetail, type NodeSource, type SchemaField } from '../api';
  import { app } from '../state.svelte';
  import Icon from '../ui/Icon.svelte';

  let { field, cell = $bindable(), sources, disabled = false, uid }: { field: SchemaField; cell: Record<string, any>; sources: NodeSource[]; disabled?: boolean; uid: string } = $props();

  const k = $derived(field.key);
  const label = $derived(field.key.replace(/_/g, ' '));
  const codeLike = $derived(['impl', 'formula', 'unit'].includes(field.key));
  // reading never writes; handlers create the list on first use
  const items = $derived<any[]>(Array.isArray(cell[k]) ? cell[k] : []);
  const arr = (): any[] => (Array.isArray(cell[k]) ? cell[k] : (cell[k] = []));
  const isStrings = (v: any) => Array.isArray(v) && v.every((x) => typeof x === 'string');

  function add(item: any) {
    arr().push(item);
  }
  function remove(i: number) {
    arr().splice(i, 1);
  }
  function grow(e: Event) {
    const t = e.currentTarget as HTMLTextAreaElement;
    t.style.height = 'auto';
    t.style.height = t.scrollHeight + 2 + 'px';
  }
  function autosize(node: HTMLTextAreaElement) {
    queueMicrotask(() => {
      node.style.height = 'auto';
      node.style.height = node.scrollHeight + 2 + 'px';
    });
  }
  const sourceName = (id: any) => sources.find((s) => s.id === String(id))?.title;

  // params: an object of name -> type, edited as rows
  let paramRows = $state<{ n: string; t: string }[]>([]);
  let paramsFrom: any = undefined;
  $effect(() => {
    if (field.type !== 'params') return;
    const v = cell[k];
    if (v === paramsFrom) return;
    paramsFrom = v;
    paramRows = Object.entries(v ?? {}).map(([n, t]) => ({ n, t: typeof t === 'string' ? t : JSON.stringify(t) }));
  });
  function commitParams() {
    const o: Record<string, string> = {};
    for (const r of paramRows) if (r.n.trim()) o[r.n.trim()] = r.t;
    paramsFrom = o;
    cell[k] = o;
  }

  // free-form JSON for objects and lists that are not plain strings
  let json = $state('');
  let jsonErr = $state<string | null>(null);
  let jsonFrom: any = undefined;
  const useJson = $derived(field.type === 'object' || (field.type === 'list' && cell[k] != null && !isStrings(cell[k])));
  $effect(() => {
    if (!useJson) return;
    const v = cell[k];
    if (v === jsonFrom) return;
    jsonFrom = v;
    json = v == null ? '' : JSON.stringify(v, null, 1);
    jsonErr = null;
  });
  function commitJson() {
    if (!json.trim()) {
      delete cell[k];
      jsonFrom = undefined;
      jsonErr = null;
      return;
    }
    try {
      const v = JSON.parse(json);
      jsonFrom = v;
      cell[k] = v;
      jsonErr = null;
    } catch {
      jsonErr = 'This is not valid JSON, so it is not saved yet.';
    }
  }

  // referrals are added by node id; the API stores the portable uid
  let refId = $state('');
  let refErr = $state<string | null>(null);
  async function addReferral() {
    const id = refId.trim();
    if (!id) return;
    refErr = null;
    try {
      const n = await get<NodeDetail>(`api/nodes/${id}`);
      add({ relation: 'refers_to', node: n.uid, title: n.title, id: n.id });
      refId = '';
    } catch (e) {
      refErr = errText(e);
    }
  }
</script>

<div class="f" data-field={k}>
  {#if field.type === 'statements'}
    <div class="fl">{label}{#if field.required}{" "}<span class="req">required</span>{/if}</div>
    {#each items as s, i (i)}
      <div class="st">
        <textarea class="field" rows="1" aria-label="Statement {i + 1}" bind:value={s.text} {disabled} oninput={grow} use:autosize></textarea>
        <div class="row">
          <select class="field src" aria-label="Source of statement {i + 1}" bind:value={s.source} {disabled}>
            <option value={undefined}>No source</option>
            {#each sources as so (so.id)}<option value={so.id}>{so.title}</option>{/each}
            {#if s.source != null && !sourceName(s.source)}<option value={s.source}>source {s.source}</option>{/if}
          </select>
          <button class="btn icon quiet" aria-label="Remove statement {i + 1}" title="Remove statement" {disabled} onclick={() => remove(i)}><Icon name="close" size={14} /></button>
        </div>
      </div>
    {/each}
    <button class="btn add" {disabled} onclick={() => add({ text: '', ...(sources[0] ? { source: sources[0].id } : {}) })}>Add statement</button>
  {:else if field.type === 'facts'}
    <div class="fl">{label}</div>
    {#if items.length}
      <table>
        <thead><tr><th>Property</th><th>Value</th><th>Source</th><th></th></tr></thead>
        <tbody>
          {#each items as f, i (i)}
            <tr>
              <td><input class="field" aria-label="Property {i + 1}" bind:value={f.property} {disabled} /></td>
              <td><input class="field" aria-label="Value {i + 1}" value={f.value} {disabled} oninput={(e) => (f.value = typeof f.value === 'number' && e.currentTarget.value.trim() !== '' && isFinite(Number(e.currentTarget.value)) ? Number(e.currentTarget.value) : e.currentTarget.value)} />
                {#if f.as_of || f.from || f.until}<span class="period">{f.as_of ? `as of ${f.as_of}` : f.from && f.until ? `${f.from} to ${f.until}` : f.from ? `from ${f.from}` : `until ${f.until}`}</span>{/if}
              </td>
              <td>
                <select class="field" aria-label="Source of fact {i + 1}" bind:value={f.source} {disabled}>
                  <option value={undefined}>None</option>
                  {#each sources as so (so.id)}<option value={so.id}>{so.title}</option>{/each}
                </select>
              </td>
              <td><button class="btn icon quiet" aria-label="Remove fact {i + 1}" title="Remove fact" {disabled} onclick={() => remove(i)}><Icon name="close" size={14} /></button></td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
    <button class="btn add" {disabled} onclick={() => add({ property: '', value: '', ...(sources[0] ? { source: sources[0].id } : {}) })}>Add fact</button>
  {:else if field.type === 'constraints' || field.type === 'checks'}
    <div class="fl">{label}</div>
    {#each items as c, i (i)}
      <div class="kv">
        <input class="field kind" aria-label="{label} {i + 1} kind" placeholder="kind" bind:value={c.kind} {disabled} />
        <input class="field" aria-label="{label} {i + 1} text" bind:value={c.text} {disabled} />
        <button class="btn icon quiet" aria-label="Remove {label} {i + 1}" title="Remove" {disabled} onclick={() => remove(i)}><Icon name="close" size={14} /></button>
      </div>
      {#if c.args !== undefined || c.expect !== undefined}
        <p class="mono extra">{c.args !== undefined ? 'args ' + JSON.stringify(c.args) : ''} {c.expect !== undefined ? 'expects ' + JSON.stringify(c.expect) : ''}</p>
      {/if}
    {/each}
    <button class="btn add" {disabled} onclick={() => add({ kind: '', text: '' })}>Add {field.type === 'checks' ? 'check' : 'constraint'}</button>
  {:else if field.type === 'referrals'}
    <div class="fl">{label}</div>
    {#each items as r, i (i)}
      <div class="kv">
        <span class="mono rel">{r.relation ?? 'refers_to'}</span>
        {#if r.id}
          <button class="linkbtn ellipsis" title="Go to this node on the board" onclick={() => app.goToNode(r.id)}>{r.title ?? r.node}</button>
        {:else}
          <span class="ellipsis grow" title="This referral does not resolve to a node in this model">{r.title ?? r.node}</span>
        {/if}
        <button class="btn icon quiet" aria-label="Remove referral {i + 1}" title="Remove referral" {disabled} onclick={() => remove(i)}><Icon name="close" size={14} /></button>
      </div>
    {/each}
    <form class="kv" onsubmit={(e) => (e.preventDefault(), addReferral())}>
      <input class="field" aria-label="Node id to refer to" placeholder="Node id" inputmode="numeric" bind:value={refId} {disabled} />
      <button class="btn" type="submit" disabled={disabled || !refId.trim()}>Add referral</button>
    </form>
    {#if refErr}<p class="error-line">{refErr}</p>{/if}
  {:else if field.type === 'params'}
    <div class="fl">{label}</div>
    {#each paramRows as p, i (i)}
      <div class="kv">
        <input class="field mono kind" aria-label="Parameter {i + 1} name" placeholder="name" bind:value={p.n} {disabled} oninput={commitParams} />
        <input class="field mono" aria-label="Parameter {i + 1} type" placeholder="type" bind:value={p.t} {disabled} oninput={commitParams} />
        <button class="btn icon quiet" aria-label="Remove parameter {i + 1}" title="Remove parameter" {disabled} onclick={() => (paramRows.splice(i, 1), commitParams())}><Icon name="close" size={14} /></button>
      </div>
    {/each}
    <button class="btn add" {disabled} onclick={() => paramRows.push({ n: '', t: '' })}>Add parameter</button>
  {:else if useJson}
    <label class="fl" for="{uid}-{k}">{label}</label>
    <textarea id="{uid}-{k}" class="field mono" rows="3" bind:value={json} {disabled} oninput={commitJson}></textarea>
    {#if jsonErr}<p class="problem">{jsonErr}</p>{/if}
  {:else if field.type === 'list'}
    <label class="fl" for="{uid}-{k}">{label} <span class="muted">one per line</span></label>
    <textarea
      id="{uid}-{k}"
      class="field"
      rows={Math.max(1, (cell[k] ?? []).length)}
      value={(cell[k] ?? []).join('\n')}
      {disabled}
      oninput={(e) => {
        const v = e.currentTarget.value.split('\n');
        cell[k] = v.length === 1 && v[0] === '' ? [] : v;
        grow(e);
      }}
      onblur={() => Array.isArray(cell[k]) && (cell[k] = cell[k].map((s: string) => s.trim()).filter(Boolean))}
    ></textarea>
  {:else if field.type === 'number'}
    <label class="fl" for="{uid}-{k}">{label}{#if field.min != null && field.max != null}{" "}<span class="muted">{field.min} to {field.max}</span>{/if}</label>
    <input id="{uid}-{k}" class="field short" type="number" step="any" min={field.min} max={field.max} value={cell[k] ?? ''} {disabled} oninput={(e) => (e.currentTarget.value === '' ? delete cell[k] : (cell[k] = Number(e.currentTarget.value)))} />
  {:else if field.type === 'boolean'}
    <label class="chk"><input type="checkbox" checked={!!cell[k]} {disabled} onchange={(e) => (cell[k] = e.currentTarget.checked)} />{label}</label>
  {:else}
    <label class="fl" for="{uid}-{k}">{label}</label>
    {#if field.key === 'description'}
      <textarea id="{uid}-{k}" class="field" rows="2" value={cell[k] ?? ''} {disabled} oninput={(e) => (e.currentTarget.value === '' ? delete cell[k] : (cell[k] = e.currentTarget.value))}></textarea>
    {:else}
      <input id="{uid}-{k}" class="field" class:mono={codeLike} value={cell[k] ?? ''} {disabled} oninput={(e) => (e.currentTarget.value === '' ? delete cell[k] : (cell[k] = e.currentTarget.value))} />
    {/if}
  {/if}
  {#if field.description && (field.type === 'text' || field.type === 'list') && field.key === 'impl'}<p class="muted desc">{field.description}</p>{/if}
</div>

<style>
  .period {
    display: block;
    margin-top: 2px;
    font-size: 12px;
    color: var(--muted);
  }
  .f {
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 0;
  }
  .fl {
    color: var(--muted);
  }
  .req {
    color: var(--faint);
  }
  .st {
    display: flex;
    flex-direction: column;
    gap: 3px;
    padding-bottom: 5px;
  }
  .st textarea {
    resize: none;
    overflow: hidden;
  }
  .src {
    flex: 1;
    color: var(--muted);
  }
  .add {
    align-self: flex-start;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    table-layout: fixed;
  }
  th {
    text-align: left;
    font-weight: 400;
    color: var(--faint);
    padding: 0 0 2px;
    font-size: 12px;
  }
  th:last-child,
  td:last-child {
    width: 28px;
  }
  td {
    padding: 0 4px 3px 0;
  }
  td:last-child {
    padding-right: 0;
  }
  .kv {
    display: flex;
    gap: 4px;
    align-items: center;
    min-width: 0;
  }
  .kind {
    flex: 0 0 34%;
    width: 34%;
  }
  .rel {
    flex: none;
    color: var(--muted);
  }
  .extra {
    color: var(--muted);
    overflow-wrap: anywhere;
  }
  .short {
    width: 120px;
  }
  .chk {
    display: flex;
    gap: 8px;
    align-items: center;
  }
  .linkbtn {
    flex: 1;
    min-width: 0;
    border: 0;
    background: transparent;
    padding: 0;
    text-align: left;
    text-decoration: underline;
    text-underline-offset: 2px;
    cursor: pointer;
  }
  .desc {
    font-size: 12px;
  }
</style>
