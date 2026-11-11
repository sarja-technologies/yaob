<script lang="ts" module>
  import type { SchemaEntry, SchemaSpec } from '../api';
  // kept at module level so a draft survives docking, floating and tab switches
  const S = $state({
    list: [] as SchemaEntry[],
    roles: [] as string[],
    types: [] as string[],
    key: '' as string,
    draft: null as SchemaSpec | null,
    newKeys: [] as string[],
  });
</script>

<script lang="ts">
  import { untrack } from 'svelte';
  import { app } from '../state.svelte';
  import { get, post, put, qs, errText, fmtInt, plural, human } from '../api';
  import Icon from '../ui/Icon.svelte';

  interface Change {
    kind: string;
    cell: string;
    field?: string;
    breaking: boolean;
    effect: string;
  }
  interface Diff {
    changes: Change[];
    breaking: boolean;
    migration_required: boolean;
    retrain_required: boolean;
    summary: string;
  }
  let loadErr = $state<string | null>(null);
  let val = $state.raw<{ problems: string[]; diff: Diff | null; compared_with: { id: string; version: number } | null } | null>(null);
  let valErr = $state<string | null>(null);
  let saving = $state(false);
  let saveErr = $state<string | null>(null);
  let saved = $state<string | null>(null);

  const keyOf = (e: { id: string; version: number }) => `${e.id}@${e.version}`;
  const entry = $derived(S.list.find((e) => keyOf(e) === S.key) ?? null);
  const clone = <T,>(v: T): T => JSON.parse(JSON.stringify(v));
  const changed = $derived(!!entry && !!S.draft && JSON.stringify(S.draft) !== JSON.stringify(entry.spec));
  const nextVersion = $derived(entry ? Math.max(...S.list.filter((e) => e.id === entry.id).map((e) => e.version)) + 1 : 1);
  const relabels = $derived(val?.diff?.changes.filter((c) => c.kind === 'relabelled') ?? []);
  const structural = $derived(val?.diff?.changes.filter((c) => c.kind !== 'relabelled') ?? []);
  const inUse = $derived((entry?.nodes ?? 0) > 0);

  async function load(keep = true) {
    try {
      const r = await get<{ schemas: SchemaEntry[]; runtime_roles: string[]; field_types: string[] }>('api/schemas');
      S.list = r.schemas;
      S.roles = r.runtime_roles;
      S.types = r.field_types;
      loadErr = null;
      if (!S.list.some((e) => keyOf(e) === S.key)) pick(S.list[0]);
      else if (!keep || !S.draft) pick(S.list.find((e) => keyOf(e) === S.key)!);
    } catch (e) {
      loadErr = errText(e);
    }
  }
  function pick(e: SchemaEntry | undefined) {
    if (!e) return;
    S.key = keyOf(e);
    S.draft = clone(e.spec);
    S.newKeys = [];
    saved = saveErr = null;
    mig.from = keyOf(e);
    const other = S.list.filter((x) => keyOf(x) !== keyOf(e));
    mig.to = keyOf(other.find((x) => x.id === e.id && x.version > e.version) ?? other[0] ?? e);
    preview = null;
  }
  $effect(() => {
    app.head;
    untrack(() => load());
  });

  // live validation of the draft
  let timer: ReturnType<typeof setTimeout> | null = null;
  let ctrl: AbortController | null = null;
  $effect(() => {
    const body = S.draft ? JSON.stringify(S.draft) : null;
    if (timer) clearTimeout(timer);
    if (!body) return;
    timer = setTimeout(async () => {
      ctrl?.abort();
      const mine = (ctrl = new AbortController());
      try {
        const r = await post('api/schemas/validate', JSON.parse(body), mine.signal);
        val = r;
        valErr = null;
      } catch (e: any) {
        if (e?.name !== 'AbortError') valErr = errText(e);
      }
    }, 200);
  });

  function addCell() {
    if (!S.draft) return;
    let i = S.draft.roles.length + 1;
    let key = `cell_${i}`;
    while (S.draft.roles.some((r) => r.key === key)) key = `cell_${++i}`;
    S.draft.roles.push({ key, label: 'New cell', runtime_role: 'none', description: '', fields: [] });
    S.newKeys.push(key);
  }
  function addField(ri: number) {
    if (!S.draft) return;
    const role = S.draft.roles[ri];
    let i = role.fields.length + 1;
    let key = `field_${i}`;
    while (role.fields.some((f) => f.key === key)) key = `field_${++i}`;
    role.fields.push({ key, type: 'text', required: false });
    S.newKeys.push(role.key + '.' + key);
  }
  const isNewCell = (k: string) => S.newKeys.includes(k) || !entry?.spec.roles.some((r) => r.key === k);
  const isNewField = (cell: string, k: string) => !entry?.spec.roles.find((r) => r.key === cell)?.fields.some((f) => f.key === k);

  async function save(asNext: boolean) {
    if (!S.draft || !entry) return;
    const spec = clone(S.draft);
    spec.version = asNext ? nextVersion : entry.version;
    saving = true;
    saveErr = saved = null;
    try {
      await put('api/schemas', spec);
      S.key = keyOf(spec);
      await load(false);
      saved = asNext ? `Saved as ${spec.name} v${spec.version}. Existing nodes stay on v${entry.version} until they are migrated.` : `Saved ${spec.name} v${spec.version}.`;
      app.nodeCache.clear();
    } catch (e) {
      saveErr = errText(e);
    }
    saving = false;
  }

  // ---------------------------------------------------------------- migration
  let mig = $state({ from: '', to: '' });
  let preview = $state.raw<any | null>(null);
  let migErr = $state<string | null>(null);
  let migBusy = $state(false);
  const split = (k: string) => {
    const [id, v] = k.split('@');
    return { id, version: Number(v) };
  };
  async function previewMigration() {
    migErr = null;
    preview = null;
    const f = split(mig.from);
    const t = split(mig.to);
    try {
      preview = await get('api/schemas/migration' + qs({ from_id: f.id, from_version: f.version, to_id: t.id, to_version: t.version }));
    } catch (e) {
      migErr = errText(e);
    }
  }
  async function migrate(ids?: string[]) {
    const f = split(mig.from);
    const t = split(mig.to);
    migBusy = true;
    migErr = null;
    try {
      const r = await post('api/schemas/migrate', { from_id: f.id, from_version: f.version, to_id: t.id, to_version: t.version, ...(ids ? { node_ids: ids } : {}) });
      preview = null;
      await app.afterEdit(r);
      await load();
      app.toast(`Migrated ${plural(r.migrated, 'node')}${r.skipped ? `, skipped ${r.skipped} that would not validate` : ''} (revision ${r.rev}).`);
    } catch (e) {
      migErr = errText(e);
    }
    migBusy = false;
  }
</script>

<div class="schema">
  {#if loadErr}
    <p class="error-line">{loadErr}</p>
    <button class="btn" onclick={() => load()}>Try again</button>
  {:else if !S.list.length}
    <p class="muted">Loading schemas</p>
  {:else}
    <ul class="versions" aria-label="Schemas and versions">
      {#each S.list as e (keyOf(e))}
        <li>
          <button class:on={keyOf(e) === S.key} aria-pressed={keyOf(e) === S.key} onclick={() => pick(e)}>
            <span class="ellipsis"><span class="vn">{e.name}</span> <span class="mono">{e.id} v{e.version}</span></span>
            <span class="muted num">{e.cells} cells, {plural(e.nodes, 'node')}</span>
          </button>
        </li>
      {/each}
    </ul>

    {#if S.draft && entry}
      <div class="grid2">
        <div>
          <label class="lbl" for="sc-name">Name</label>
          <input id="sc-name" class="field" bind:value={S.draft.name} />
        </div>
        <div>
          <label class="lbl" for="sc-desc">Description</label>
          <input id="sc-desc" class="field" bind:value={S.draft.description} />
        </div>
      </div>

      {#each S.draft.roles as role, ri (ri)}
        <section class="cell" data-cell={role.key}>
          <div class="cellhead">
            <div class="grow">
              <label class="lbl" for="sc-l-{ri}">Cell label</label>
              <input id="sc-l-{ri}" class="field" bind:value={role.label} />
            </div>
            <div class="rr">
              <label class="lbl" for="sc-r-{ri}">Runtime role</label>
              <select id="sc-r-{ri}" class="field" bind:value={role.runtime_role}>
                {#each S.roles as r (r)}<option value={r}>{r}</option>{/each}
              </select>
            </div>
            <button class="btn icon quiet rm" aria-label="Remove the cell {role.label}" title="Remove cell" onclick={() => S.draft!.roles.splice(ri, 1)}><Icon name="close" size={14} /></button>
          </div>
          <div class="keyline">
            {#if isNewCell(role.key)}
              <label class="muted" for="sc-k-{ri}">Key</label>
              <input id="sc-k-{ri}" class="field mono keyin" bind:value={role.key} />
            {:else}
              <span class="muted">Key <span class="mono">{role.key}</span></span>
            {/if}
          </div>
          <label class="sr-only" for="sc-d-{ri}">Description of {role.label}</label>
          <textarea id="sc-d-{ri}" class="field" rows="2" placeholder="What this cell holds and how the runtime reads it" bind:value={role.description}></textarea>
          {#if role.fields.length}
            <table>
              <thead><tr><th>Field</th><th>Type</th><th>Required</th><th></th></tr></thead>
              <tbody>
                {#each role.fields as f, fi (fi)}
                  <tr>
                    <td>
                      {#if isNewField(role.key, f.key) || S.newKeys.includes(role.key + '.' + f.key)}
                        <input class="field mono" aria-label="Field key" bind:value={f.key} />
                      {:else}
                        <span class="mono">{f.key}</span>
                      {/if}
                    </td>
                    <td>
                      <select class="field" aria-label="Type of {f.key}" bind:value={f.type}>
                        {#each S.types as t (t)}<option value={t}>{t}</option>{/each}
                      </select>
                    </td>
                    <td><input type="checkbox" aria-label="{f.key} is required" bind:checked={f.required} /></td>
                    <td><button class="btn icon quiet" aria-label="Remove the field {f.key}" title="Remove field" onclick={() => role.fields.splice(fi, 1)}><Icon name="close" size={14} /></button></td>
                  </tr>
                {/each}
              </tbody>
            </table>
          {/if}
          <button class="btn add" onclick={() => addField(ri)}>Add field</button>
        </section>
      {/each}
      <button class="btn add" onclick={addCell}>Add cell</button>

      <section class="check" aria-live="polite">
        <h3>What this draft changes</h3>
        {#if valErr}<p class="error-line">{valErr}</p>{/if}
        {#each val?.problems ?? [] as p}<p class="problem">{p}</p>{/each}
        {#if val && !val.problems.length}
          {#if !val.diff || !val.diff.changes.length}
            <p class="muted">{changed ? 'Only names or descriptions differ. Runtime behaviour and stored nodes are unchanged.' : `No changes compared with ${entry.id} v${entry.version}.`}</p>
          {:else}
            <p>{val.diff.summary}{val.compared_with ? ` Compared with ${val.compared_with.id} v${val.compared_with.version}.` : ''}</p>
            {#if relabels.length}
              <div class="grp">
                <h4>Relabel only: runtime unchanged</h4>
                <ul>{#each relabels as c}<li><span class="mono">{c.cell}</span> {c.effect}</li>{/each}</ul>
              </div>
            {/if}
            {#if structural.length}
              <div class="grp struct">
                <h4>Structural changes</h4>
                <ul>
                  {#each structural as c}<li><span class="mono">{c.cell}{c.field ? '.' + c.field : ''}</span> {c.effect}{#if c.breaking}{" "}<span class="brk">Needs a new version.</span>{/if}</li>{/each}
                </ul>
                {#if val.diff.migration_required}<p>Nodes on the current version must be migrated to use it.</p>{/if}
                {#if val.diff.retrain_required}<p>Learned state depends on a changed cell: affected nodes need retraining.</p>{/if}
              </div>
            {/if}
          {/if}
        {/if}
        {#if saveErr}<p class="error-line">{saveErr}</p>{/if}
        {#if saved}<p class="okline">{saved}</p>{/if}
        <div class="row wrap">
          {#if val?.diff?.breaking && inUse}
            <button class="btn" disabled title="{fmtInt(entry.nodes)} nodes use this version">Save v{entry.version}</button>
            <span class="muted">In use: structural change</span>
          {:else}
            <button class="btn" disabled={!changed || saving || !!val?.problems.length} onclick={() => save(false)}>Save v{entry.version}</button>
          {/if}
          <button class="btn primary" disabled={!changed || saving || !!val?.problems.length} onclick={() => save(true)}>Save as v{nextVersion}</button>
          <button class="btn" disabled={!changed || saving} onclick={() => pick(entry)}>Discard draft</button>
        </div>
      </section>

      <section class="mig">
        <h3>Migrate nodes</h3>
        <div class="row wrap">
          <select class="field auto" aria-label="Migrate from" bind:value={mig.from} onchange={() => (preview = null)}>
            {#each S.list as e (keyOf(e))}<option value={keyOf(e)}>{e.id} v{e.version} ({plural(e.nodes, 'node')})</option>{/each}
          </select>
          <span class="muted">to</span>
          <select class="field auto" aria-label="Migrate to" bind:value={mig.to} onchange={() => (preview = null)}>
            {#each S.list as e (keyOf(e))}<option value={keyOf(e)}>{e.id} v{e.version}</option>{/each}
          </select>
          <button class="btn" disabled={mig.from === mig.to} onclick={previewMigration}>Preview</button>
        </div>
        {#if migErr}<p class="error-line">{migErr}</p>{/if}
        {#if preview}
          <div class="prev">
            <p>{plural(preview.nodes, 'node')} on {preview.from.id} v{preview.from.version}. {fmtInt(preview.sampled)} were tried against {preview.to.id} v{preview.to.version}: {preview.sample_failing ? fmtInt(preview.sample_failing) + ' would not validate' : 'all validate'}.</p>
            <p class="muted">{preview.note}</p>
            {#if preview.diff?.changes?.length}
              <ul class="changes">{#each preview.diff.changes as c}<li><span class="mono">{c.cell}{c.field ? '.' + c.field : ''}</span> {c.effect}</li>{/each}</ul>
            {/if}
            {#if preview.example_failure}
              <p class="problem">Node {preview.example_failure.node_id}: {preview.example_failure.problems.join('; ')}</p>
            {/if}
            <div class="row wrap">
              <button class="btn primary" disabled={!preview.can_apply || migBusy || !preview.nodes} onclick={() => migrate()}>Migrate all {fmtInt(preview.nodes)}</button>
              {#if app.selection.length}
                <button class="btn" disabled={!preview.can_apply || migBusy} onclick={() => migrate(app.selection)}>Migrate the {app.selection.length} selected</button>
              {:else}
                <button class="btn" disabled>Migrate selected</button><span class="muted">No nodes selected</span>
              {/if}
            </div>
          </div>
        {/if}
      </section>
    {/if}
  {/if}
</div>

<style>
  .schema {
    padding: 10px 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .versions {
    list-style: none;
    margin: 0 -12px;
    padding: 0 0 8px;
    border-bottom: 1px solid var(--line-soft);
  }
  .versions button {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    width: 100%;
    min-height: 28px;
    align-items: center;
    padding: 2px 12px;
    border: 0;
    background: transparent;
    text-align: left;
    cursor: pointer;
    flex-wrap: wrap;
  }
  .versions button:hover {
    background: var(--raise);
  }
  .versions button.on {
    background: #262626;
  }
  .versions button:focus-visible {
    outline-offset: -2px;
  }
  .vn {
    font-weight: 500;
  }
  .versions .mono {
    color: var(--muted);
  }
  .grid2 {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 8px;
  }
  .cell {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding-top: 10px;
    border-top: 1px solid var(--line-soft);
  }
  .cellhead {
    display: flex;
    gap: 8px;
    align-items: flex-end;
  }
  .rr {
    flex: 0 0 132px;
  }
  .keyline {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .keyin {
    width: 160px;
  }
  table {
    border-collapse: collapse;
    width: 100%;
  }
  th {
    text-align: left;
    font-weight: 400;
    color: var(--faint);
    font-size: 12px;
    padding: 0 6px 2px 0;
  }
  td {
    padding: 0 6px 3px 0;
  }
  td:nth-child(2) {
    width: 124px;
  }
  td:nth-child(3),
  th:nth-child(3) {
    width: 60px;
  }
  td:last-child {
    width: 28px;
    padding-right: 0;
  }
  .add {
    align-self: flex-start;
  }
  .check,
  .mig {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding-top: 10px;
    border-top: 1px solid var(--line);
  }
  h3 {
    font-weight: 500;
  }
  h4 {
    margin: 0;
    font-weight: 500;
    font-size: 13px;
  }
  .grp ul,
  .changes {
    margin: 2px 0 0;
    padding-left: 16px;
  }
  .grp li,
  .changes li {
    margin: 2px 0;
  }
  .grp .mono,
  .changes .mono {
    color: var(--muted);
  }
  .struct {
    border: 1px solid #6a5220;
    background: #17130a;
    padding: 6px 8px;
  }
  .brk {
    color: var(--gold);
  }
  .okline {
    border: 1px solid var(--line);
    padding: 4px 8px;
    background: var(--raise);
  }
  .wrap {
    flex-wrap: wrap;
  }
  .auto {
    width: auto;
    flex: 0 1 auto;
    max-width: 100%;
  }
  .prev {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
</style>
