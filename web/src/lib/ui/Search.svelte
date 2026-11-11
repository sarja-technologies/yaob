<script lang="ts">
  import { app } from '../state.svelte';
  import { get, qs, errText, human, type SearchResult } from '../api';
  import { clickOutside } from './actions';
  import Icon from './Icon.svelte';

  type Mode = 'lexical' | 'semantic' | 'hybrid';
  const MODES: { v: Mode; label: string }[] = [
    { v: 'lexical', label: 'Text' },
    { v: 'semantic', label: 'Meaning' },
    { v: 'hybrid', label: 'Both' },
  ];
  const MATCH: Record<string, string> = { text: 'text', meaning: 'meaning', 'text+meaning': 'text and meaning', id: 'id' };
  const LIMIT = 20;

  let input: HTMLInputElement;
  let q = $state('');
  let mode = $state<Mode>('hybrid');
  let f = $state({ type: '', status: '', validation: '', origin: '', mpn_id: '' });
  let results = $state.raw<SearchResult[]>([]);
  let notice = $state<string | null>(null);
  let error = $state<string | null>(null);
  let busy = $state(false);
  let more = $state(false);
  let active = $state(0);
  let asked = $state(false);
  let ctrl: AbortController | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const hasFilter = $derived(Object.values(f).some(Boolean));

  export function focus() {
    app.searchOpen = true;
    queueMicrotask(() => {
      input.focus();
      input.select();
    });
  }
  function close() {
    app.searchOpen = false;
  }
  async function run(offset = 0) {
    ctrl?.abort();
    if (!q.trim() && !hasFilter) {
      results = [];
      notice = error = null;
      asked = false;
      more = false;
      busy = false;
      return;
    }
    const mine = (ctrl = new AbortController());
    busy = true;
    try {
      const r = await get<{ results: SearchResult[]; notice?: string }>('api/search' + qs({ q: q.trim(), mode, limit: LIMIT, offset, ...f }), mine.signal);
      if (mine.signal.aborted) return;
      results = offset ? [...results, ...r.results] : r.results;
      more = r.results.length === LIMIT;
      notice = r.notice ?? null;
      error = null;
      asked = true;
      if (!offset) active = 0;
    } catch (e: any) {
      if (e?.name === 'AbortError') return;
      error = errText(e);
      results = [];
    }
    busy = false;
  }
  function changed() {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => run(), 140);
  }
  $effect(() => {
    q;
    mode;
    f.type;
    f.status;
    f.validation;
    f.origin;
    f.mpn_id;
    changed();
  });
  // an edit elsewhere can change what matches
  $effect(() => {
    app.head;
    if (app.searchOpen) changed();
  });

  function go(r: SearchResult) {
    close();
    input.blur();
    app.goToNode(r.id, r);
  }
  function key(e: KeyboardEvent) {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      active = Math.min(results.length - 1, active + 1);
      scroll();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      active = Math.max(0, active - 1);
      scroll();
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (results[active]) go(results[active]);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      close();
      input.blur();
    }
  }
  function scroll() {
    queueMicrotask(() => document.getElementById('sr-' + active)?.scrollIntoView({ block: 'nearest' }));
  }
  function parts(s: string): { t: string; hit: boolean }[] {
    const out: { t: string; hit: boolean }[] = [];
    const re = /\[([^\]]*)\]/g;
    let last = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(s))) {
      if (m.index > last) out.push({ t: s.slice(last, m.index), hit: false });
      out.push({ t: m[1], hit: true });
      last = m.index + m[0].length;
    }
    if (last < s.length) out.push({ t: s.slice(last), hit: false });
    return out;
  }
  const statusWord = (s: string) => (s === 'tombstone' ? 'deleted' : s === 'live' ? '' : s);
</script>

<div class="search" use:clickOutside={close}>
  <div class="box">
    <Icon name="search" size={14} />
    <input
      bind:this={input}
      bind:value={q}
      type="text"
      role="combobox"
      aria-expanded={app.searchOpen}
      aria-controls="search-results"
      aria-label="Search nodes"
      aria-activedescendant={results.length ? 'sr-' + active : undefined}
      placeholder="Search nodes"
      autocomplete="off"
      spellcheck="false"
      onfocus={() => (app.searchOpen = true)}
      oninput={() => (app.searchOpen = true)}
      onclick={() => (app.searchOpen = true)}
      onkeydown={key}
    />
    {#if !q && !app.searchOpen}<kbd title="Press / or Ctrl+K to search">/</kbd>{/if}
  </div>
  {#if app.searchOpen}
    <div class="pop panel">
      <div class="controls">
        <div class="seg" role="radiogroup" aria-label="Search by">
          {#each MODES as m (m.v)}
            <button role="radio" aria-checked={mode === m.v} onclick={() => (mode = m.v)}>{m.label}</button>
          {/each}
        </div>
        <div class="filters">
          <select class="field" aria-label="Type" bind:value={f.type}>
            <option value="">Any type</option>
            <option value="knowledge">Knowledge</option>
            <option value="procedure">Procedure</option>
          </select>
          <select class="field" aria-label="Status" bind:value={f.status}>
            <option value="">Live and disabled</option>
            <option value="live">Live</option>
            <option value="disabled">Disabled</option>
            <option value="tombstone">Deleted</option>
          </select>
          <select class="field" aria-label="Validation" bind:value={f.validation}>
            <option value="">Any validation</option>
            <option value="unverified">Unverified</option>
            <option value="source_backed">Source backed</option>
            <option value="checked">Checked</option>
            <option value="failed">Failed</option>
          </select>
          <select class="field" aria-label="Origin" bind:value={f.origin}>
            <option value="">Any origin</option>
            <option value="reference">Reference</option>
            <option value="user">User</option>
            <option value="synthetic">Synthetic</option>
            <option value="diagnostic">Diagnostic</option>
          </select>
          <select class="field mpn" aria-label="MPN" bind:value={f.mpn_id}>
            <option value="">All MPNs</option>
            {#each app.mpns as m (m.id)}<option value={m.id}>{m.name}</option>{/each}
          </select>
        </div>
      </div>
      {#if notice}<p class="problem">{notice}</p>{/if}
      {#if error}<p class="error-line">{error}</p>{/if}
      <ul id="search-results" role="listbox" aria-label="Results" class="scroll">
        {#each results as r, i (r.id)}
          <!-- svelte-ignore a11y_click_events_have_key_events -->
          <li id="sr-{i}" role="option" aria-selected={i === active} class:on={i === active} onclick={() => go(r)} onpointermove={() => (active = i)}>
            <div class="l1">
              <span class="t ellipsis">{r.title}</span>
              {#if statusWord(r.status)}<span class="st">{statusWord(r.status)}</span>{/if}
              <span class="m">{r.match ? 'matched by ' + (MATCH[r.match] ?? r.match) : ''}</span>
            </div>
            <div class="l2 ellipsis">
              {#if r.snippet}{#each parts(r.snippet) as p}{#if p.hit}<mark>{p.t}</mark>{:else}{p.t}{/if}{/each}{:else}{r.summary}{/if}
            </div>
            <div class="l3 ellipsis">{human(r.type)}{r.mpn ? ' in ' + r.mpn : ''}</div>
          </li>
        {/each}
      </ul>
      {#if !results.length && !error}
        <p class="hint muted">
          {#if busy}Searching{:else if asked}No nodes match.{:else}Search titles, statements and cues. A number opens the node with that id. Filters alone list nodes, for example deleted ones.{/if}
        </p>
      {/if}
      {#if more}<button class="btn more" onclick={() => run(results.length)}>Show more</button>{/if}
    </div>
  {/if}
</div>

<style>
  .search {
    position: relative;
    flex: 0 1 460px;
    min-width: 90px;
  }
  .box {
    display: flex;
    align-items: center;
    gap: 6px;
    height: 28px;
    padding: 0 8px;
    border: 1px solid var(--line);
    border-radius: 2px;
    background: var(--void);
    color: var(--muted);
  }
  .box:focus-within {
    outline: 2px solid var(--white);
    outline-offset: -1px;
  }
  .box input {
    flex: 1;
    min-width: 0;
    border: 0;
    background: transparent;
    outline: none;
    color: var(--white);
    height: 100%;
  }
  .box input::placeholder {
    color: var(--faint);
  }
  kbd {
    font-family: var(--mono);
    font-size: 12px;
    line-height: 16px;
    padding: 0 5px;
    border: 1px solid var(--line);
    border-radius: 2px;
    color: var(--muted);
  }
  .panel {
    position: absolute;
    top: 33px;
    left: 0;
    width: max(100%, 560px);
    max-width: calc(100vw - 16px);
    z-index: 60;
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-height: min(560px, calc(100vh - 80px));
  }
  .controls {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .filters {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    flex: 1;
  }
  .filters .field {
    width: auto;
    flex: 0 1 auto;
    max-width: 170px;
  }
  ul {
    list-style: none;
    margin: 0;
    padding: 0;
    min-height: 0;
    flex: 1 1 auto;
  }
  li {
    padding: 5px 8px;
    border-radius: 2px;
    cursor: pointer;
  }
  li.on {
    background: #262626;
  }
  .l1 {
    display: flex;
    gap: 8px;
    align-items: baseline;
  }
  .t {
    font-weight: 500;
    min-width: 0;
  }
  .st {
    color: var(--gold);
    flex: none;
  }
  .m {
    margin-left: auto;
    color: var(--muted);
    font-size: 12px;
    flex: none;
  }
  .l2 {
    color: var(--muted);
  }
  .l3 {
    color: var(--faint);
    font-size: 12px;
  }
  mark {
    background: #3a3a3a;
    color: var(--white);
    padding: 0 1px;
  }
  .hint {
    padding: 4px 8px 6px;
    max-width: 62ch;
  }
  .more {
    align-self: flex-start;
  }
  @media (max-width: 720px) {
    .panel {
      position: fixed;
      left: 8px;
      right: 8px;
      top: 37px;
      width: auto;
    }
  }
</style>
