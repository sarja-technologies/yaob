import { test, expect, type APIRequestContext, type Page } from '@playwright/test';
import { results, waitNode, api, cam, level, openApp, resetView, screenOf, searchAndOpen, setView, settle, showPanel, stats, visibleNodes } from './helpers';

// Journeys run against a live MPN Studio API (the synthetic dev store by default). They edit the
// store through the UI and put back what they change, so the suite can be run repeatedly.

const uniq = (s: string) => `${s} ${Date.now().toString(36)}`;

async function makeNode(request: APIRequestContext, title: string, mpn_id: string, position: { x: number; y: number }) {
  const r = await request.post('api/nodes', {
    data: { title, cells: { knowledge: { statements: [{ text: `${title} statement.` }] }, thinking: {}, reasoning: {} }, memberships: [{ mpn_id }], position },
  });
  expect(r.ok(), await r.text()).toBeTruthy();
  return (await r.json()).id as string;
}
async function region(request: APIRequestContext, name: string) {
  const b = await api(request, 'api/board');
  return b.mpns.find((m: any) => m.name === name) as { id: string; x: number; y: number; r: number; name: string };
}
const errors: string[] = [];
function watch(page: Page) {
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => m.type() === 'error' && !/favicon|Failed to load resource/.test(m.text()) && errors.push(m.text()));
}

test.beforeEach(async ({ page, request }) => {
  await resetView(request);
  errors.length = 0;
  watch(page);
});
test.afterEach(() => {
  expect([...new Set(errors)].slice(0, 4), 'no uncaught page errors').toEqual([]);
});

test('the board opens directly and shows the MPN regions with real counts', async ({ page, request }) => {
  await openApp(page);
  const board = await api(request, 'api/board');
  const s = await stats(page);
  expect(s.regions).toBeGreaterThanOrEqual(board.mpns.length);
  expect(s.labels).toBeGreaterThanOrEqual(board.mpns.length);
  await expect(page.locator('.mrow')).toHaveCount(board.mpns.length);
  // real brand artwork, loaded
  for (const sel of ['header img[alt="Fylume"]', '.model img[alt="Yaob"]']) {
    expect(await page.locator(sel).evaluate((i: HTMLImageElement) => i.complete && i.naturalWidth > 0)).toBeTruthy();
  }
  const first = board.mpns[0];
  await expect(page.locator('.mrow', { hasText: first.name })).toContainText(first.count.toLocaleString('en-US'));
  // no horizontal overflow
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
});

test('zooming out gives density aggregates; zooming into an MPN draws nodes, labels and edges', async ({ page }) => {
  await openApp(page);
  const box = (await page.locator('.board').boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  for (let i = 0; i < 8; i++) await page.mouse.wheel(0, 400);
  await settle(page);
  expect(await level(page)).toBe('cells');
  expect((await stats(page)).cells).toBeGreaterThan(0);

  await page.locator('.mrow', { hasText: 'Mathematics and statistics' }).click();
  await settle(page);
  expect(await level(page)).toBe('nodes');
  const far = await stats(page);
  expect(far.nodes).toBeGreaterThan(50);
  // wheel zoom around the cursor until single nodes are large
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  for (let i = 0; i < 6; i++) await page.mouse.wheel(0, -300);
  await settle(page);
  const near = await stats(page);
  expect((await cam(page)).zoom).toBeGreaterThan(2);
  expect(near.nodes).toBeGreaterThan(0);
  expect(near.labels).toBeGreaterThan(0);
  expect(near.nodes).toBeLessThan(far.nodes);
});

test('search jumps to an unloaded node; a statement edit is saved, survives reload, and is undone', async ({ page, request }) => {
  await openApp(page);
  await page.locator('.mrow', { hasText: 'Biology' }).click();
  await settle(page);
  const before = await cam(page);
  expect((await page.evaluate(() => window.__mpn.nodes().some((n: any) => n.id === '12')))).toBeFalsy();

  await searchAndOpen(page, 'Synthetic topic 12', 'Synthetic topic 12');
  const after = await cam(page);
  expect(Math.hypot(after.x - before.x, after.y - before.y)).toBeGreaterThan(500);
  expect(await page.evaluate(() => window.__mpn.nodes().some((n: any) => n.id === '12'))).toBeTruthy();
  for (const h of ['Knowledge', 'Thinking', 'Reasoning', 'Sources', 'Learned state']) await expect(page.locator('.insp h3', { hasText: h }).first()).toBeVisible();
  await expect(page.locator('.insp')).toContainText('Development fixture (synthetic)');
  await expect(page.locator('.insp')).toContainText('findable by text search now');

  const original = (await api(request, 'api/nodes/12')).cells.knowledge.statements[0].text as string;
  const edited = `${original.replace(/ Edited.*$/, '')} Edited ${Date.now().toString(36)}.`;
  const st = page.getByLabel('Statement 1', { exact: true });
  await st.fill(edited);
  await page.locator('.savebar').getByRole('button', { name: 'Save' }).click();
  await expect(page.locator('.effects')).toContainText('Saved as revision');
  await expect(page.locator('.effects')).toContainText('After indexing');
  expect((await api(request, 'api/nodes/12')).cells.knowledge.statements[0].text).toBe(edited);

  await page.reload();
  await page.waitForFunction(() => window.__mpn && window.__mpn.level() !== null);
  await searchAndOpen(page, '12', 'Synthetic topic 12');
  await expect(page.getByLabel('Statement 1', { exact: true })).toHaveValue(edited);

  // a save the schema rejects shows the API's problems inline
  await page.getByLabel('Statement 1', { exact: true }).fill('');
  await page.getByLabel('Statement 2', { exact: true }).fill('');
  await page.locator('.savebar').getByRole('button', { name: 'Save' }).click();
  await expect(page.locator('.insp .problem').first()).toContainText('statements');
  await page.locator('.savebar').getByRole('button', { name: 'Revert' }).click();

  await page.getByRole('button', { name: 'Undo the last model edit' }).click();
  await expect(page.getByLabel('Statement 1', { exact: true })).toHaveValue(original);
  expect((await api(request, 'api/nodes/12')).cells.knowledge.statements[0].text).toBe(original);
});

test('a node with two memberships keeps existing when one is removed; the last one is refused; undo restores', async ({ page, request }) => {
  await openApp(page);
  const title = uniq('E2E shared node');
  const box = (await page.locator('.board').boundingBox())!;
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2, { button: 'right' });
  await page.getByRole('menuitem', { name: /New node here/ }).click();
  await page.getByLabel('Title').fill(title);
  await page.getByLabel('First statement').fill('Created by the end-to-end suite.');
  await page.getByLabel('MPN 1').selectOption({ label: 'Physics' });
  await page.getByRole('button', { name: 'Add another MPN' }).click();
  await page.getByLabel('MPN 2').selectOption({ label: 'Chemistry (shared)' });
  await page.getByRole('button', { name: 'Create node' }).click();
  await expect(page.locator('.insp .title')).toHaveValue(title);
  const id = (await page.locator('.insp .id').innerText()).trim();
  const members = page.locator('.insp section', { hasText: 'Member of' }).first().locator('ul.list li');
  await expect(members).toHaveCount(2);

  await page.getByRole('button', { name: 'Remove the membership in Chemistry' }).click();
  await expect(members).toHaveCount(1);
  let n = await api(request, `api/nodes/${id}`);
  expect(n.status).toBe('live');
  expect(n.memberships.map((m: any) => m.mpn_name)).toEqual(['Physics']);

  await page.getByRole('button', { name: 'Remove the membership in Physics' }).click();
  await expect(page.locator('.insp .error-line')).toContainText('only membership');
  expect((await api(request, `api/nodes/${id}`)).memberships).toHaveLength(1);

  await page.getByRole('button', { name: 'Undo the last model edit' }).click();
  await expect(members).toHaveCount(2);
  n = await api(request, `api/nodes/${id}`);
  expect(n.memberships).toHaveLength(2);
  await request.delete(`api/nodes/${id}`);
});

test('two nodes are connected by dragging in connect mode, then disconnected from the edge', async ({ page, request }) => {
  const r = await region(request, 'Physics');
  const a = await makeNode(request, uniq('E2E connect A'), r.id, { x: r.x - 40, y: r.y - r.r - 60 });
  const b = await makeNode(request, uniq('E2E connect B'), r.id, { x: r.x + 60, y: r.y - r.r - 60 });
  const pa = (await api(request, `api/nodes/${a}`)).position;
  const pb = (await api(request, `api/nodes/${b}`)).position;
  await openApp(page);
  await setView(page, (pa.x + pb.x) / 2, (pa.y + pb.y) / 2, 4);
  const sa = await screenOf(page, pa.x, pa.y);
  const sb = await screenOf(page, pb.x, pb.y);

  await page.getByRole('radio', { name: 'Connect' }).click();
  await page.mouse.move(sa.x, sa.y);
  await page.mouse.down();
  await page.mouse.move((sa.x + sb.x) / 2, (sa.y + sb.y) / 2, { steps: 6 });
  await page.mouse.move(sb.x, sb.y, { steps: 6 });
  await page.mouse.up();
  await expect(page.getByRole('dialog', { name: 'Connect nodes' })).toBeVisible();
  await page.getByLabel('Relation type').selectOption('depends_on');
  await page.getByRole('button', { name: 'Connect', exact: true }).click();
  await expect(page.locator('.toast')).toContainText('Connected');
  let e = await api(request, `api/nodes/${a}/edges`);
  expect(e.edges).toHaveLength(1);
  expect(e.edges[0]).toMatchObject({ type: 'depends_on', dst: b, direction: 'out' });

  await settle(page);
  await expect.poll(async () => (await stats(page)).edges).toBeGreaterThan(0);
  await page.mouse.click((sa.x + sb.x) / 2, (sa.y + sb.y) / 2, { button: 'right' });
  await page.getByRole('menuitem', { name: 'Disconnect' }).click();
  await expect(page.locator('.toast').last()).toContainText('Disconnected');
  e = await api(request, `api/nodes/${a}/edges`);
  expect(e.edges).toHaveLength(0);
  await request.delete(`api/nodes/${a}`);
  await request.delete(`api/nodes/${b}`);
});

test('delete shows a preview and is recoverable: Restore from the toast, and from the deleted filter', async ({ page, request }) => {
  const r = await region(request, 'History');
  const title = uniq('E2E delete me');
  const id = await makeNode(request, title, r.id, { x: r.x, y: r.y - r.r - 50 });
  await openApp(page);
  await searchAndOpen(page, title, title);

  await page.getByRole('button', { name: 'More actions for this node' }).click();
  await page.getByRole('menuitem', { name: 'Delete…' }).click();
  const dlg = page.getByRole('dialog');
  await expect(dlg).toContainText('Delete this node?');
  await expect(dlg).toContainText('Deletion is recoverable');
  await expect(dlg).toContainText('History (primary)');
  await dlg.getByRole('button', { name: 'Delete node' }).click();
  await expect.poll(async () => (await api(request, `api/nodes/${id}`)).status).toBe('tombstone');
  await page.locator('.toast').getByRole('button', { name: 'Restore' }).click();
  await expect.poll(async () => (await api(request, `api/nodes/${id}`)).status).toBe('live');

  // delete again with the keyboard, then find the tombstone through the status filter
  const p = (await api(request, `api/nodes/${id}`)).position;
  await waitNode(page, id);
  const s = await screenOf(page, p.x, p.y);
  await page.mouse.click(s.x, s.y);
  await page.keyboard.press('Delete');
  await page.getByRole('dialog').getByRole('button', { name: 'Delete node' }).click();
  await expect.poll(async () => (await api(request, `api/nodes/${id}`)).status).toBe('tombstone');
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await page.keyboard.press('/');
  await page.getByRole('combobox', { name: 'Search nodes' }).fill(title);
  await expect(page.locator('#search-results', { hasText: title })).toHaveCount(0);
  await page.getByLabel('Status').selectOption('tombstone');
  await expect(results(page).first()).toContainText(title);
  await expect(results(page).first()).toContainText('deleted');
  await results(page).first().click();
  await expect(page.locator('.insp .banner')).toContainText('This node is deleted');
  await page.locator('.insp .banner').getByRole('button', { name: 'Restore' }).click();
  await expect.poll(async () => (await api(request, `api/nodes/${id}`)).status).toBe('live');
  await expect(page.locator('.insp .banner')).toHaveCount(0);
  await request.delete(`api/nodes/${id}`);
});

test('an MPN is disabled and enabled again, each after the impact preview', async ({ page, request }) => {
  await openApp(page);
  const row = page.locator('.mpns > li', { hasText: 'History' });
  await row.locator('.mrow').click();
  await expect(row.locator('.tree li').first()).toBeVisible();
  const impact = await api(request, `api/mpns/${(await region(request, 'History')).id}/impact?action=disable`);
  await row.getByRole('button', { name: 'Disable…' }).click();
  await expect(row.locator('.impact')).toContainText(impact.explanation);
  await expect(row.locator('.impact')).toContainText(String(impact.nodes_only_here));
  await expect(row.locator('.impact')).toContainText(String(impact.cross_mpn_edges));
  await row.getByRole('button', { name: 'Disable History' }).click();
  await expect(row.locator('.mrow')).toContainText('disabled');
  let m = (await api(request, 'api/mpns')).mpns.find((x: any) => x.name === 'History');
  expect(m.enabled).toBe(false);

  await row.getByRole('button', { name: 'Enable…' }).click();
  await expect(row.locator('.impact')).toContainText('may consult History again');
  await row.getByRole('button', { name: 'Enable History' }).click();
  await expect(row.locator('.mrow')).not.toContainText('disabled');
  m = (await api(request, 'api/mpns')).mpns.find((x: any) => x.name === 'History');
  expect(m.enabled).toBe(true);
});

test('schema editor: a relabel validates as runtime-unchanged, a role change as structural; a version is saved; one node is migrated and undone', async ({ page, request }) => {
  await openApp(page);
  await showPanel(page, 'Schemas');
  const ed = page.locator('.schema');
  await ed.locator('.versions button', { hasText: 'Three Cell Node' }).first().click();
  const nodes = (await api(request, 'api/schemas')).schemas.find((s: any) => s.id === '3cn' && s.version === 1).nodes;
  await expect(ed.locator('.versions button', { hasText: 'Three Cell Node' }).first()).toContainText(nodes.toLocaleString('en-US'));

  await ed.locator('#sc-l-0').fill('Facts');
  await expect(ed.locator('.check')).toContainText('Relabel only: runtime unchanged');
  await expect(ed.locator('.check')).toContainText('Runtime semantics unchanged');
  await expect(ed.locator('.check .struct')).toHaveCount(0);
  await expect(ed.getByRole('button', { name: 'Save v1' })).toBeEnabled();

  await ed.locator('#sc-r-1').selectOption('none');
  await expect(ed.locator('.check .struct')).toContainText('This changes execution, not just the label');
  await expect(ed.getByRole('button', { name: 'Save v1' })).toBeDisabled();
  await expect(ed.locator('.check')).toContainText('In use: structural change');
  await ed.getByRole('button', { name: 'Discard draft' }).click();
  await expect(ed.locator('#sc-l-0')).toHaveValue('Knowledge');

  // an unused schema gets a field and is saved as its next version
  await ed.locator('.versions button', { hasText: 'Five Cell Node' }).last().click();
  const next = Math.max(...(await api(request, 'api/schemas')).schemas.filter((s: any) => s.id === '5cn').map((s: any) => s.version)) + 1;
  await ed.locator('section[data-cell="interface"]').getByRole('button', { name: 'Add field' }).click();
  await expect(ed.locator('.check')).toContainText("added to cell 'interface'");
  await ed.getByRole('button', { name: `Save as v${next}` }).click();
  await expect(ed.locator('.okline')).toContainText(`v${next}`);
  await expect(ed.locator('.versions')).toContainText(`5cn v${next}`);

  // migrate one selected node to the five-cell schema, see five cells, undo
  await searchAndOpen(page, 'Synthetic topic 40', 'Synthetic topic 40');
  await showPanel(page, 'Schemas');
  await ed.getByLabel('Migrate from').selectOption('3cn@1');
  await ed.getByLabel('Migrate to').selectOption('5cn@1');
  await ed.getByRole('button', { name: 'Preview' }).click();
  await expect(ed.locator('.prev')).toContainText('all validate');
  await ed.getByRole('button', { name: 'Migrate the 1 selected' }).click();
  await expect(page.locator('.toast')).toContainText('Migrated 1 node');
  expect((await api(request, 'api/nodes/40')).schema.id).toBe('5cn');
  await page.getByRole('tab', { name: 'Node' }).click();
  for (const h of ['Provenance', 'Interface']) await expect(page.locator('.insp h3', { hasText: h })).toBeVisible();
  await page.getByRole('button', { name: 'Undo the last model edit' }).click();
  await expect.poll(async () => (await api(request, 'api/nodes/40')).schema.id).toBe('3cn');
});

test('the run panel reports the runtime being unavailable with the sentence from the stream', async ({ page, request }) => {
  const rt = await api(request, 'api/runtime');
  test.skip(rt.available !== false, 'the runtime is running on this server');
  await openApp(page);
  await page.getByRole('tab', { name: 'Run' }).click();
  await expect(page.getByLabel('Run mode')).toHaveValue('yaob');
  await page.getByLabel('Run mode').selectOption('retrieval');
  await page.getByLabel('Message').fill('What is the arithmetic mean?');
  await page.getByRole('button', { name: 'Send' }).click();
  await expect(page.locator('.run .error-line')).toContainText('The runtime is not running, so nothing can be generated');
  await expect(page.locator('.run .msg.assistant .text')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Send' })).toBeVisible();
});

test('a sleep job is started, its log is tailed, and it is cancelled', async ({ page, request }) => {
  await openApp(page);
  await showPanel(page, 'Jobs');
  const jobs = page.locator('.jobs');
  await jobs.getByLabel('Job kind').selectOption('sleep');
  await jobs.getByLabel('Steps').fill('60');
  await jobs.getByLabel('Seconds per step').fill('0.5');
  await jobs.getByRole('button', { name: 'Start job' }).click();
  const row = jobs.locator('li[data-job]').first();
  await expect(row).toHaveAttribute('data-state', 'running', { timeout: 20_000 });
  await expect(row.locator('.log')).toContainText(/step \d+\/60/, { timeout: 15_000 });
  await expect(row.locator('progress')).toBeVisible();
  await row.getByRole('button', { name: 'Cancel' }).click();
  await expect(row).toHaveAttribute('data-state', 'cancelled', { timeout: 20_000 });
  const id = await row.getAttribute('data-job');
  expect((await api(request, `api/jobs/${id}`)).state).toBe('cancelled');
  await expect(row.getByRole('button', { name: 'Resume' })).toBeVisible();
});

test('a floating panel resizes from its edges without moving the board, keeps a minimum size, and docks again', async ({ page }) => {
  await openApp(page);
  await page.getByRole('tab', { name: 'Run' }).click();
  await page.getByRole('button', { name: 'Detach Run into a window' }).click();
  const win = page.locator('section[data-window="run"]');
  await expect(win).toBeVisible();
  const before = await cam(page);
  const b0 = (await win.boundingBox())!;

  const dragHandle = async (dir: string, dx: number, dy: number) => {
    const h = (await win.locator(`[data-resize="${dir}"]`).boundingBox())!;
    await page.mouse.move(h.x + h.width / 2, h.y + h.height / 2);
    await page.mouse.down();
    await page.mouse.move(h.x + h.width / 2 + dx, h.y + h.height / 2 + dy, { steps: 5 });
    await page.mouse.up();
  };
  await dragHandle('e', 90, 0);
  const b1 = (await win.boundingBox())!;
  expect(Math.round(b1.width - b0.width)).toBe(90);
  expect(Math.round(b1.x)).toBe(Math.round(b0.x));
  await dragHandle('n', 0, -30);
  const b2 = (await win.boundingBox())!;
  expect(Math.round(b2.height - b1.height)).toBe(30);
  expect(Math.round(b2.y - b1.y)).toBe(-30);
  // the title bar cannot be pushed under the top bar
  await dragHandle('n', 0, -300);
  expect((await win.boundingBox())!.y).toBeGreaterThanOrEqual(40);
  await dragHandle('w', 900, 0);
  const b3 = (await win.boundingBox())!;
  expect(Math.round(b3.width)).toBe(320);
  await dragHandle('s', 0, -900);
  const b4 = (await win.boundingBox())!;
  expect(Math.round(b4.height)).toBe(240);
  // content reflows inside the small window
  const ta = (await win.getByLabel('Message').boundingBox())!;
  expect(ta.x).toBeGreaterThanOrEqual(b4.x);
  expect(ta.x + ta.width).toBeLessThanOrEqual(b4.x + b4.width);
  expect(await cam(page)).toEqual(before);

  // wheel over the window scrolls the window, not the board
  await page.mouse.move(b4.x + 100, b4.y + 100);
  await page.mouse.wheel(0, -300);
  await page.waitForTimeout(200);
  expect(await cam(page)).toEqual(before);

  await win.getByRole('button', { name: 'Dock Run to the right' }).click();
  await expect(win).toHaveCount(0);
  await expect(page.getByRole('tab', { name: 'Run' })).toHaveAttribute('aria-selected', 'true');
  // close is not delete: the panel comes back from the Panels menu with its content
  await page.getByLabel('Message').fill('draft kept');
  await page.getByRole('button', { name: 'Close the Run panel' }).click();
  await expect(page.getByRole('tab', { name: 'Run' })).toHaveCount(0);
  await showPanel(page, 'Run');
  await expect(page.getByLabel('Message')).toHaveValue('draft kept');
});

test('the view and the panel arrangement survive a reload', async ({ page }) => {
  await openApp(page);
  await page.getByRole('button', { name: 'Detach Node into a window' }).click();
  const win = page.locator('section[data-window="inspector"]');
  const head = (await win.locator('header').boundingBox())!;
  await page.mouse.move(head.x + 60, head.y + 12);
  await page.mouse.down();
  await page.mouse.move(head.x + 60 - 180, head.y + 12 + 90, { steps: 6 });
  await page.mouse.up();
  await page.getByRole('button', { name: 'Collapse the left dock' }).click();
  await showPanel(page, 'Runtime');
  const saved = page.waitForResponse((r) => r.url().includes('api/board/view') && r.request().method() === 'PUT');
  await setView(page, 700, -380, 2.2);
  await saved;
  await page.waitForTimeout(700);
  const box = (await win.boundingBox())!;
  const view = await cam(page);

  await page.reload();
  await page.waitForFunction(() => window.__mpn && window.__mpn.level() !== null);
  await settle(page);
  const win2 = page.locator('section[data-window="inspector"]');
  await expect(win2).toBeVisible();
  const box2 = (await win2.boundingBox())!;
  expect(Math.round(box2.x)).toBe(Math.round(box.x));
  expect(Math.round(box2.y)).toBe(Math.round(box.y));
  expect(Math.round(box2.width)).toBe(Math.round(box.width));
  await expect(page.locator('.rail.left')).toBeVisible();
  await expect(page.getByRole('tab', { name: 'Runtime' })).toHaveAttribute('aria-selected', 'true');
  const view2 = await cam(page);
  expect(view2.x).toBeCloseTo(view.x, 1);
  expect(view2.y).toBeCloseTo(view.y, 1);
  expect(view2.zoom).toBeCloseTo(view.zoom, 3);
});

test('moving nodes changes geometry only; rubber-band selects; Escape clears', async ({ page, request }) => {
  await openApp(page);
  const r = await region(request, 'Chemistry');
  await setView(page, r.x, r.y, 2.4);
  const rev = (await api(request, 'api/health')).head;
  const n = (await visibleNodes(page))[0];
  await page.mouse.move(n.s.x, n.s.y);
  await page.mouse.down();
  await page.mouse.move(n.s.x + 30, n.s.y + 20, { steps: 4 });
  await expect(page.locator('footer .hint')).toContainText('layout only');
  await page.mouse.move(n.s.x + 72, n.s.y + 48, { steps: 4 });
  await page.mouse.up();
  await expect(page.locator('.toast')).toContainText('Geometry only. No model revision was created.');
  const pos = (await api(request, `api/nodes/${n.id}`)).position;
  expect(pos.x).toBeCloseTo(n.x + 72 / 2.4, 0);
  expect(pos.y).toBeCloseTo(n.y + 48 / 2.4, 0);
  expect((await api(request, 'api/health')).head).toBe(rev);
  // put it back
  await request.post('api/layout/move', { data: { moves: [{ id: n.id, x: n.x, y: n.y }] } });

  await page.getByRole('radio', { name: 'Select' }).click();
  const box = (await page.locator('.board').boundingBox())!;
  await page.mouse.move(box.x + 120, box.y + 120);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width - 260, box.y + box.height - 120, { steps: 5 });
  await page.mouse.up();
  await expect(page.locator('[data-panel="inspector"]')).toContainText(/\d+ nodes selected/);
  await expect(page.getByRole('button', { name: 'Fit selection' }).first()).toBeEnabled();
  await page.locator('.board').focus();
  await page.keyboard.press('Escape');
  await expect(page.locator('[data-panel="inspector"]')).not.toContainText('nodes selected');
});

test('notes and groups move, resize from any handle and persist; a reference opens the same node', async ({ page, request }) => {
  await openApp(page);
  const r = await region(request, 'Biology');
  await setView(page, r.x, r.y, 2);
  const before = (await api(request, 'api/board')).objects.length;
  await page.getByRole('button', { name: 'Note', exact: true }).click();
  const note = page.locator('.obj.note').last();
  await expect(note).toBeVisible();
  await note.dblclick();
  await note.locator('textarea').fill('Check these sources');
  await page.locator('.board').click({ position: { x: 300, y: 600 } });
  await expect(note.locator('.txt')).toHaveText('Check these sources');
  await note.click();
  const b0 = (await note.boundingBox())!;
  const view = await cam(page);
  // west edge: the left side moves, the right side stays
  const h = (await note.locator('.h.w').boundingBox())!;
  await page.mouse.move(h.x + h.width / 2, h.y + h.height / 2);
  await page.mouse.down();
  await page.mouse.move(h.x + h.width / 2 - 60, h.y + h.height / 2, { steps: 4 });
  await page.mouse.up();
  const b1 = (await note.boundingBox())!;
  expect(Math.round(b1.width - b0.width)).toBe(60);
  expect(Math.round(b1.x + b1.width)).toBe(Math.round(b0.x + b0.width));
  expect(await cam(page)).toEqual(view);
  await expect(note.locator('.h')).toHaveCount(8);
  await expect.poll(async () => (await api(request, 'api/board')).objects.length).toBe(before + 1);
  await expect.poll(async () => Math.round((await api(request, 'api/board')).objects.at(-1).w * 2)).toBe(Math.round(b1.width));

  await page.reload();
  await page.waitForFunction(() => window.__mpn && window.__mpn.level() !== null);
  await settle(page);
  const again = page.locator('.obj.note', { hasText: 'Check these sources' });
  await expect(again).toBeVisible();
  expect(Math.round((await again.boundingBox())!.width)).toBe(Math.round(b1.width));
  await again.click();
  await page.keyboard.press('Delete');
  await expect(again).toHaveCount(0);
  await expect.poll(async () => (await api(request, 'api/board')).objects.length).toBe(before);

  // reference: a second occurrence that opens the one node
  const n = (await visibleNodes(page))[0];
  await page.mouse.click(n.s.x, n.s.y, { button: 'right' });
  const refsBefore = await page.locator(".ref").count();
  await page.getByRole('menuitem', { name: 'Add reference here' }).click();
  await expect(page.locator('.ref')).toHaveCount(refsBefore + 1);
  const ref = page.locator('.ref').last();
  await expect(ref.locator('.glyph')).toBeVisible();
  const title = (await api(request, `api/nodes/${n.id}`)).title;
  await expect(ref).toContainText(title);
  await page.locator('.board').click({ position: { x: 300, y: 640 } });
  await ref.locator('.glyph').click();
  await expect(page.locator('.insp .id')).toHaveText(n.id);
  await ref.locator('.glyph').click({ button: 'right' });
  await page.getByRole('menuitem', { name: 'Remove reference' }).click();
  await expect(page.locator(".ref")).toHaveCount(refsBefore);
  expect((await api(request, `api/nodes/${n.id}`)).status).toBe('live');
});

test('keyboard and view navigation: search key, pan, zoom, fit, back and forward, saved views', async ({ page, request }) => {
  await openApp(page);
  const start = await cam(page);
  await page.keyboard.press('/');
  await expect(page.getByRole('combobox', { name: 'Search nodes' })).toBeFocused();
  await page.keyboard.press('Escape');
  await page.locator('.board').focus();
  await page.keyboard.press('ArrowRight');
  await settle(page);
  expect((await cam(page)).x).toBeGreaterThan(start.x);
  await page.keyboard.press('+');
  await settle(page);
  expect((await cam(page)).zoom).toBeGreaterThan(start.zoom);

  await page.locator('.mrow', { hasText: 'Physics' }).click();
  await settle(page);
  const physics = await cam(page);
  await page.locator('.mrow', { hasText: 'History' }).click();
  await settle(page);
  const history = await cam(page);
  expect(Math.abs(history.x - physics.x)).toBeGreaterThan(100);
  await page.locator('.board').focus();
  await page.keyboard.press('Alt+ArrowLeft');
  await settle(page);
  expect((await cam(page)).x).toBeCloseTo(physics.x, 1);
  await page.getByRole('button', { name: 'Forward to the next view' }).click();
  await settle(page);
  expect((await cam(page)).x).toBeCloseTo(history.x, 1);

  // saved views: create, jump, delete (through the API-backed list)
  const name = uniq('E2E view');
  await page.getByRole('button', { name: 'Saved views' }).click();
  await page.getByLabel('Name for the current view').fill(name);
  await page.getByRole('button', { name: 'Save', exact: true }).click();
  await expect(page.locator('.views li', { hasText: name })).toBeVisible();
  expect((await api(request, 'api/board')).saved_views.some((v: any) => v.name === name)).toBeTruthy();
  await page.keyboard.press('Escape');
  await page.locator('.board').focus();
  await page.keyboard.press('0');
  await settle(page);
  expect((await cam(page)).zoom).toBeLessThan(history.zoom / 2);
  await page.getByRole('button', { name: 'Saved views' }).click();
  await page.locator('.views li', { hasText: name }).locator('.jump').click();
  await settle(page);
  expect((await cam(page)).x).toBeCloseTo(history.x, 1);
  await page.getByRole('button', { name: 'Saved views' }).click();
  await page.getByRole('button', { name: `Delete the saved view ${name}` }).click();
  await expect(page.locator('.views li', { hasText: name })).toHaveCount(0);
  expect((await api(request, 'api/board')).saved_views.some((v: any) => v.name === name)).toBeFalsy();

  // the minimap moves the view
  const mm = (await page.locator('.minimap').boundingBox())!;
  await page.mouse.click(mm.x + 30, mm.y + 30);
  await settle(page);
  expect((await cam(page)).x).toBeLessThan(history.x);
});

test('budget: a preset is applied, a field is saved, and a value the server refuses shows its message', async ({ page, request }) => {
  const original = (await api(request, 'api/budget')).budget;
  await openApp(page);
  await showPanel(page, 'Memory budget');
  const b = page.locator('.budget');
  await b.getByRole('radio', { name: /Conservative/ }).click();
  await expect(b.getByRole('radio', { name: /Conservative/ })).toHaveAttribute('aria-checked', 'true');
  expect((await api(request, 'api/budget')).budget.max_active_mpns).toBe(2);
  await b.getByLabel('Output').fill('300');
  await b.getByRole('button', { name: 'Save budget' }).click();
  await expect.poll(async () => (await api(request, 'api/budget')).budget.output_tokens).toBe(300);
  expect((await api(request, 'api/budget')).budget.preset).toBe('custom');
  await b.getByLabel('Of those, also given as text').fill('50');
  await b.getByRole('button', { name: 'Save budget' }).click();
  await expect(b.locator('.error-line')).toContainText('text_nodes must be between 0 and 16');
  await request.put('api/budget', { data: { ...original, preset: original.preset, apply_preset: false } });
  await request.put('api/budget', { data: { preset: original.preset, apply_preset: true } });
});

test('the runtime panel states plainly that the runtime is not running and offers no working-looking actions', async ({ page, request }) => {
  const rt = await api(request, 'api/runtime');
  test.skip(rt.available !== false, 'the runtime is running on this server');
  await openApp(page);
  await showPanel(page, 'Runtime');
  const p = page.locator('.rt');
  await expect(p).toContainText('The runtime is not running');
  await expect(p).toContainText(rt.reason);
  for (const name of ['Load core', 'Unload', 'Reindex']) await expect(p.getByRole('button', { name })).toBeDisabled();
});

test('small window: 1024x700 has no horizontal overflow and the docks still work', async ({ page, request }) => {
  await page.setViewportSize({ width: 1024, height: 700 });
  await openApp(page);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
  const board = (await page.locator('.board').boundingBox())!;
  expect(board.width).toBeGreaterThan(300);
  await expect(page.locator('.mrow').first()).toBeVisible();
  await expect(page.getByRole('tab', { name: 'Node' })).toBeVisible();
  // context menu near the bottom-right corner of the window stays inside it
  await page.getByRole('button', { name: 'Collapse the right dock' }).click();
  const b = await api(request, 'api/board');
  await setView(page, b.bounds[0] - 900, b.bounds[1] - 900, 1);
  const wide = (await page.locator('.board').boundingBox())!;
  await page.mouse.click(wide.x + wide.width - 6, wide.y + wide.height - 190, { button: 'right' });
  const menu = (await page.locator('.ctx').boundingBox())!;
  expect(menu.x + menu.width).toBeLessThanOrEqual(1024);
  expect(menu.y + menu.height).toBeLessThanOrEqual(700);
  await expect(page.getByRole('menuitem', { name: 'Paste' })).toBeDisabled();
  await expect(page.getByRole('menuitem', { name: 'Paste' })).toContainText('Nothing copied');
});
