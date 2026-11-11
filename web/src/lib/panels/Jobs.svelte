<script lang="ts">
  import { onMount } from 'svelte';
  import { app } from '../state.svelte';
  import { get, post, errText, fmtInt, human, when, type Job } from '../api';

  const KINDS: { v: string; label: string }[] = [
    { v: 'train_bank', label: 'Train bank' },
    { v: 'evaluate', label: 'Evaluate' },
    { v: 'reindex', label: 'Reindex' },
    { v: 'package', label: 'Package' },
    { v: 'sleep', label: 'Sleep (test job)' },
  ];
  const kindLabel = (k: string) => KINDS.find((x) => x.v === k)?.label ?? human(k);
  const ACTIVE = ['queued', 'running', 'cancelling'];

  let jobs = $state.raw<Job[]>([]);
  let memory = $state.raw<any>(null);
  let checkpoints = $state.raw<any[]>([]);
  let error = $state<string | null>(null);
  let loaded = $state(false);
  let open = $state<string | null>(null);
  let log = $state('');
  let logNext = 0;
  let logFor: string | null = null;
  let logEl = $state<HTMLPreElement | null>(null);

  let kind = $state('sleep');
  let sleep = $state({ steps: 20, step_s: 0.5 });
  let gpu = $state(false);
  let mem = $state(0);
  let extra = $state<{ k: string; v: string }[]>([]);
  let startErr = $state<string | null>(null);
  let starting = $state(false);
  let busyId = $state<string | null>(null);

  $effect(() => {
    // defaults follow the kind: the test job needs no GPU or memory, real jobs default to the server's own assumptions
    if (kind === 'sleep') {
      gpu = false;
      mem = 0;
    } else {
      gpu = true;
      mem = 12;
    }
  });

  async function load() {
    try {
      const r = await get<{ jobs: Job[]; memory: any }>('api/jobs');
      jobs = r.jobs;
      memory = r.memory;
      error = null;
      loaded = true;
    } catch (e) {
      error = errText(e);
    }
  }
  async function loadCheckpoints() {
    try {
      checkpoints = (await get<{ checkpoints: any[] }>('api/checkpoints')).checkpoints;
    } catch {
      /* the jobs error line covers an unreachable server */
    }
  }
  async function tail() {
    const id = open;
    if (!id) return;
    if (logFor !== id) {
      logFor = id;
      log = '';
      logNext = -16384;
    }
    try {
      const r = await get<{ text: string; next: number; size: number }>(`api/jobs/${id}/log?offset=${logNext}`);
      if (open !== id) return;
      if (r.text) {
        const stick = !logEl || logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 8;
        log = (log + r.text).slice(-60000);
        if (stick) queueMicrotask(() => logEl?.scrollTo({ top: logEl.scrollHeight }));
      }
      logNext = r.next;
    } catch {
      /* next poll retries */
    }
  }
  onMount(() => {
    load();
    loadCheckpoints();
    let n = 0;
    const t = setInterval(() => {
      if (document.hidden) return;
      n++;
      const active = jobs.some((j) => ACTIVE.includes(j.state));
      if (active || n % 5 === 0) load();
      if (open && (active || n % 5 === 0)) tail();
      if (n % 10 === 0) loadCheckpoints();
    }, 1200);
    return () => clearInterval(t);
  });

  function parse(v: string): any {
    try {
      return JSON.parse(v);
    } catch {
      return v;
    }
  }
  async function start() {
    const params: Record<string, any> = kind === 'sleep' ? { steps: Number(sleep.steps), step_s: Number(sleep.step_s) } : {};
    for (const r of extra) if (r.k.trim()) params[r.k.trim()] = parse(r.v.trim());
    params.gpu = gpu;
    params.memory_estimate_gib = Number(mem);
    starting = true;
    startErr = null;
    try {
      const r = await post<{ id: string }>('api/jobs', { kind, params });
      await load();
      toggle(r.id, true);
      app.loadStatus();
    } catch (e) {
      startErr = errText(e);
    }
    starting = false;
  }
  function toggle(id: string, force = false) {
    open = open === id && !force ? null : id;
    if (open) tail();
  }
  async function act(j: Job, what: 'cancel' | 'resume') {
    busyId = j.id;
    try {
      const r = await post(`api/jobs/${j.id}/${what}`);
      await load();
      // say what the server now reports, not what was asked
      if (what === 'cancel') app.toast(`Job ${j.id} is ${r.state}.`);
      else {
        app.toast(`Queued job ${r.id} to continue ${j.id}.`);
        toggle(r.id, true);
      }
    } catch (e) {
      app.toast(errText(e), 'error');
    }
    busyId = null;
  }
  async function activate(id: string) {
    try {
      await post(`api/checkpoints/${id}/activate`);
      await loadCheckpoints();
      app.toast(`Checkpoint ${id} is active.`);
    } catch (e) {
      app.toast(errText(e), 'error');
    }
  }
  function progress(j: Job): { text: string; frac: number | null } {
    const p = j.progress ?? {};
    if (j.state === 'queued' && p.waiting) return { text: p.waiting, frac: null };
    if (typeof p.step === 'number' && typeof p.total === 'number' && p.total > 0) return { text: `step ${fmtInt(p.step)} of ${fmtInt(p.total)}`, frac: Math.min(1, p.step / p.total) };
    const bits = Object.entries(p)
      .filter(([k, v]) => k !== 'waiting' && (typeof v === 'number' || typeof v === 'string'))
      .slice(0, 3)
      .map(([k, v]) => `${human(k)} ${typeof v === 'number' ? fmtInt(v) : v}`);
    return { text: bits.join(', '), frac: null };
  }
</script>

<div class="jobs">
  <form class="start" onsubmit={(e) => (e.preventDefault(), start())}>
    <div class="row">
      <select class="field kind" aria-label="Job kind" bind:value={kind}>
        {#each KINDS as k (k.v)}<option value={k.v}>{k.label}</option>{/each}
      </select>
      <button class="btn primary" type="submit" disabled={starting}>Start job</button>
    </div>
    <div class="params">
      {#if kind === 'sleep'}
        <label>Steps<input class="field" type="number" min="1" max="100000" bind:value={sleep.steps} /></label>
        <label>Seconds per step<input class="field" type="number" min="0" step="0.1" bind:value={sleep.step_s} /></label>
      {/if}
      <label>Memory estimate, GiB<input class="field" type="number" min="0" step="0.5" bind:value={mem} /></label>
      <label class="chk"><input type="checkbox" bind:checked={gpu} />Uses the GPU</label>
    </div>
    {#each extra as r, i (i)}
      <div class="row">
        <input class="field mono" placeholder="parameter" aria-label="Parameter name {i + 1}" bind:value={r.k} />
        <input class="field mono" placeholder="value" aria-label="Parameter value {i + 1}" bind:value={r.v} />
        <button type="button" class="btn" onclick={() => extra.splice(i, 1)}>Remove</button>
      </div>
    {/each}
    <button type="button" class="btn add" onclick={() => extra.push({ k: '', v: '' })}>Add parameter</button>
    {#if memory}<p class="muted">{memory.available_gib} GiB of memory is available; {memory.reserve_gib} GiB is reserved, which leaves {memory.usable_gib} GiB for jobs.</p>{/if}
    {#if startErr}<p class="error-line">{startErr}</p>{/if}
  </form>

  <section>
    <h3>Jobs</h3>
    {#if error}<p class="error-line">{error}</p>{/if}
    {#if loaded && !jobs.length}<p class="muted">No jobs have been started.</p>{/if}
    <ul class="list">
      {#each jobs as j (j.id)}
        {@const p = progress(j)}
        <li class:open={open === j.id} data-job={j.id} data-state={j.state}>
          <div class="jrow">
            <button class="jmain" aria-expanded={open === j.id} title="Show the log" onclick={() => toggle(j.id)}>
              <span class="k">{kindLabel(j.kind)}{j.attempt > 1 ? `, attempt ${j.attempt}` : ''}</span>
              <span class="s" class:attn={j.state === 'failed' || j.state === 'interrupted'}>{j.state}{p.text ? ', ' + p.text : ''}</span>
              <span class="muted mono">{j.id}</span>
            </button>
            {#if j.state === 'queued' || j.state === 'running'}
              <button class="btn" disabled={busyId === j.id} onclick={() => act(j, 'cancel')}>Cancel</button>
            {:else if ['cancelled', 'interrupted', 'failed'].includes(j.state)}
              <button class="btn" disabled={busyId === j.id} onclick={() => act(j, 'resume')}>Resume</button>
            {/if}
          </div>
          {#if p.frac != null && ACTIVE.includes(j.state)}<progress max="1" value={p.frac} aria-label="Progress of job {j.id}"></progress>{/if}
          {#if j.error}<p class="problem">{j.error}</p>{/if}
          {#if open === j.id}
            <p class="muted small">Created {when(j.created)}{j.started ? ', started ' + when(j.started) : ''}{j.finished ? ', finished ' + when(j.finished) : ''}{j.resume_of ? ', continues ' + j.resume_of : ''}</p>
            <pre class="log mono scroll" bind:this={logEl} aria-label="Log of job {j.id}">{log || 'No log output yet.'}</pre>
          {/if}
        </li>
      {/each}
    </ul>
  </section>

  <section>
    <h3>Checkpoints</h3>
    {#if !checkpoints.length}
      <p class="muted">No checkpoints yet. A training job writes them.</p>
    {:else}
      <ul class="list">
        {#each checkpoints as c (c.id)}
          <li>
            <div class="jrow">
              <div class="jmain static">
                <span class="k mono">{c.id}</span>
                <span class="s">{c.status}, {human(c.kind)}, step {fmtInt(c.step)}{c.rev != null ? ', revision ' + c.rev : ''}</span>
                <span class="muted small">{when(c.created)}</span>
              </div>
              <button class="btn" disabled={c.status === 'active'} onclick={() => activate(c.id)}>{c.status === 'active' ? 'Active' : 'Activate'}</button>
            </div>
          </li>
        {/each}
      </ul>
    {/if}
  </section>
</div>

<style>
  .jobs {
    padding: 10px 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .start {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .kind {
    flex: 1;
  }
  .params {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 8px;
    align-items: end;
  }
  .params label {
    display: flex;
    flex-direction: column;
    gap: 3px;
    color: var(--muted);
  }
  .params .chk {
    flex-direction: row;
    align-items: center;
    gap: 8px;
    height: 28px;
    color: var(--white);
  }
  .add {
    align-self: flex-start;
  }
  section {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding-top: 10px;
    border-top: 1px solid var(--line);
  }
  h3 {
    font-weight: 500;
  }
  .list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
  }
  .list li {
    padding: 5px 0;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .list li + li {
    border-top: 1px solid var(--line-soft);
  }
  .jrow {
    display: flex;
    gap: 8px;
    align-items: center;
  }
  .jmain {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    border: 0;
    background: transparent;
    padding: 0;
    text-align: left;
    cursor: pointer;
  }
  .jmain.static {
    cursor: default;
  }
  .k {
    font-weight: 500;
  }
  .s {
    color: var(--muted);
    overflow-wrap: anywhere;
  }
  .s.attn {
    color: var(--gold);
  }
  .small {
    font-size: 12px;
  }
  progress {
    appearance: none;
    width: 100%;
    height: 3px;
    border: 0;
    background: var(--line);
    display: block;
  }
  progress::-webkit-progress-bar {
    background: var(--line);
  }
  progress::-webkit-progress-value {
    background: var(--white);
  }
  progress::-moz-progress-bar {
    background: var(--white);
  }
  .log {
    margin: 0;
    padding: 6px 8px;
    background: var(--void);
    border: 1px solid var(--line);
    max-height: 220px;
    min-height: 60px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    line-height: 17px;
  }
</style>
