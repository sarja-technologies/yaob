import { test, expect } from '@playwright/test';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { cam, openApp, resetView, settle, showPanel } from './helpers';

// The dev server has no model runtime, so these two tests feed the UI the response shapes that
// docs/api.md specifies, through request interception. They prove that the UI renders those shapes;
// they do not prove anything about the real runtime.
const here = dirname(fileURLToPath(import.meta.url));
const sse = (events: [string, unknown][]) => events.map(([e, d]) => `event: ${e}\ndata: ${JSON.stringify(d)}\n\n`).join('');

test.beforeEach(async ({ request }) => {
  await resetView(request);
});

test('MOCKED run stream: tokens are shown and the evidence block lists what was consulted', async ({ page }) => {
  await page.route('**/api/run', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: sse([
        ['start', { request_id: 'r-mock', rev: 1, mode: 'yaob', core: { name: 'core-name-from-api', revision: 'abc1234' }, checkpoint: 'ck-mock' }],
        [
          'route',
          {
            ms: 12,
            mpns: [
              { id: '2', name: 'Mathematics and statistics', score: 0.91, active: true },
              { id: '4', name: 'Physics', score: 0.22, active: false },
            ],
            skipped: [{ id: '11', name: 'History', reason: 'disabled' }],
            shards: [{ mpn_id: '2', name: 'Mathematics and statistics', state: 'loaded now', bytes: 1048576, ms: 40 }],
            nodes: [{ id: '1201', title: 'Arithmetic mean', mpn_id: '2', mpn: 'Mathematics and statistics', score: 0.8, gate: 0.9, read_as: ['text', 'specialist'], via: 'router' }],
            adapters: [{ mpn_id: '2', name: 'Mathematics and statistics', bytes: 66060288, rank: 16, trainable_params: 16515072, validated: true }],
            procedures: [{ node_id: '1201', title: 'Arithmetic mean', impl: 'math.mean', args: { values: [2, 4, 9] }, ok: true, result: 5, unit: '' }],
            budget: { applied: {}, reductions: ['Text evidence was cut from 3 nodes to 1 to fit the context budget.'] },
          },
        ],
        ['token', { text: 'The mean ' }],
        ['token', { text: 'is 5 [1].' }],
        ['validation', { citations: [{ n: 1, node_id: '1201', title: 'Arithmetic mean', source: { title: 'MPN Studio procedure library', url: '', license: 'project source', revision: 'r9' } }], checks: [{ node_id: '1201', title: 'Arithmetic mean', kind: 'worked_example', ok: true, detail: 'matches the stored example' }] }],
        ['done', { text: 'The mean is 5 [1].', cancelled: false, usage: { prompt_tokens: 40, text_evidence_tokens: 30, completion_tokens: 9 }, timing: { route_ms: 12, load_ms: 40, ttft_ms: 180, total_ms: 600, tokens_per_s: 21.5 }, memory: { process_rss_mib: 5200, torch_allocated_mib: 3100, torch_peak_mib: 3300, estimate_mib: 3400, resident_cache_bytes: 1048576, system_available_gib: 20 } }],
      ]),
    }),
  );
  await openApp(page);
  await page.getByRole('tab', { name: 'Run' }).click();
  await page.getByLabel('Message').fill('What is the mean of 2, 4 and 9?');
  await page.getByLabel('Message').press('Enter');
  const msg = page.locator('.run .msg.assistant');
  await expect(msg.locator('.text')).toHaveText('The mean is 5 [1].');
  const ev = msg.locator('details.ev');
  await expect(ev.locator('summary')).toContainText('What was consulted');
  await expect(ev.locator('summary')).toContainText('1 MPN, 1 node, 1 procedure, 1 citation');
  await expect(ev.locator('.evbody')).toBeHidden();
  await ev.locator('summary').click();
  for (const t of ['MPNs consulted', 'MPNs skipped', 'History disabled', 'Nodes read', 'read as text and specialist, via router', 'Specialists applied', 'rank 16', 'math.mean({"values":[2,4,9]}) = 5', 'Citations', 'licence project source', 'Passed: Arithmetic mean', 'first token 180 ms', 'core-name-from-api']) {
    await expect(ev).toContainText(t);
  }
  await expect(page.locator('.run')).not.toContainText(/thought|reasoning trace/i);
  await page.screenshot({ path: join(here, 'screenshots', 'mocked-run-evidence-1440x900.png') });

  // a node named in the evidence jumps the board to it and opens it
  const before = await cam(page);
  await ev.locator('.grp', { hasText: 'Nodes read' }).getByRole('button', { name: 'Arithmetic mean' }).click();
  await expect(page.locator('.insp .title')).toHaveValue('Arithmetic mean');
  await settle(page);
  expect((await cam(page)).zoom).toBeGreaterThan(before.zoom);
});

test('MOCKED runtime status: the runtime panel shows core, residency, memory and index, and re-reads state after an action', async ({ page }) => {
  let loads = 0;
  await page.route('**/api/runtime', (route) =>
    route.fulfill({
      json: {
        available: true,
        core: { name: 'core-name-from-api', repo: 'org/core', revision: 'abc1234', license: 'apache-2.0', params_total: 3_800_000_000, loaded: loads > 0 },
        encoder: 'encoder-name',
        checkpoint: null,
        learned: { 'router parameters': 2_900_000, 'of which per node': 2_570_000, 'specialist adapters': 1, 'adapter parameters': 16_515_072, 'total learned': 19_415_072 },
        residency: { mpns: [{ mpn_id: '2', name: 'Mathematics and statistics', resident: true, bytes: 2_097_152, last_used: '2026-11-11T05:00:00Z' }, { mpn_id: '4', name: 'Physics', resident: false, bytes: 0, last_used: null }], cache_bytes: 2_097_152, cache_limit_bytes: 536_870_912 },
        requests: { active: 0, queued: 0 },
        memory: { process_rss_mib: 5200, torch_allocated_mib: 3100, torch_peak_mib: 3300 },
        index: { rows: 1240, pending: 3 },
      },
    }),
  );
  await page.route('**/api/runtime/load', (route) => {
    loads++;
    return route.fulfill({ json: { ok: true } });
  });
  await openApp(page);
  await showPanel(page, 'Runtime');
  const p = page.locator('.rt');
  for (const t of ['core-name-from-api, not loaded', 'abc1234', 'apache-2.0', '3,800,000,000', 'router parameters', '1,240 rows, 3 pending', 'Mathematics and statistics', 'resident, 2.0 MiB', 'not resident']) await expect(p).toContainText(t);
  await p.getByRole('button', { name: 'Load core' }).click();
  await expect(p).toContainText('core-name-from-api, loaded');
  await page.screenshot({ path: join(here, 'screenshots', 'mocked-runtime-panel-1440x900.png') });
});
