import { expect, type APIRequestContext, type Page } from '@playwright/test';

declare global {
  interface Window {
    __mpn: any;
  }
}

/** Forget the stored view and panel arrangement so a test starts from the default layout. */
export async function resetView(request: APIRequestContext) {
  const r = await request.put('api/board/view', { data: {} });
  expect(r.ok()).toBeTruthy();
}

export async function openApp(page: Page) {
  await page.goto('./');
  await page.waitForFunction(() => window.__mpn && window.__mpn.level() !== null, null, { timeout: 20_000 });
  await settle(page);
}

/** Wait until the board has no pending animation, request or frame. */
export async function settle(page: Page) {
  await page.waitForFunction(() => window.__mpn.idle(), null, { timeout: 15_000, polling: 50 });
}

export const stats = (page: Page) => page.evaluate(() => ({ ...window.__mpn.stats, requests: window.__mpn.stats.requests.length }));
export const cam = (page: Page) => page.evaluate(() => ({ ...window.__mpn.cam }));
export const level = (page: Page) => page.evaluate(() => window.__mpn.level());

export async function setView(page: Page, x: number, y: number, zoom: number) {
  await page.evaluate(([x, y, z]) => window.__mpn.setView(x, y, z), [x, y, zoom]);
  await settle(page);
}

export async function screenOf(page: Page, x: number, y: number) {
  return page.evaluate(([x, y]) => window.__mpn.toScreen(x, y), [x, y]);
}

/** Loaded nodes that are comfortably inside the visible board area, away from the overlays. */
export async function visibleNodes(page: Page) {
  return page.evaluate(() => {
    const m = window.__mpn;
    const r = document.querySelector('.board')!.getBoundingClientRect();
    return m
      .nodes()
      .map((n: any) => ({ ...n, s: m.toScreen(n.x, n.y) }))
      .filter((n: any) => n.s.x > r.left + 60 && n.s.x < r.right - 240 && n.s.y > r.top + 70 && n.s.y < r.bottom - 60) as any[];
  });
}

export async function api<T = any>(request: APIRequestContext, path: string): Promise<T> {
  const r = await request.get(path);
  expect(r.ok(), `${path} -> ${r.status()}`).toBeTruthy();
  return r.json();
}

export async function showPanel(page: Page, title: string) {
  await page.getByRole('button', { name: 'Panels' }).click();
  const item = page.getByRole('menuitemcheckbox', { name: new RegExp('^' + title + '(\\s|$)') });
  if ((await item.getAttribute('aria-checked')) !== 'true') await item.click();
  else await page.keyboard.press('Escape');
  await expect(page.getByRole('menu')).toHaveCount(0);
  const tab = page.getByRole('tab', { name: title });
  if (await tab.count()) await tab.click();
}

export const results = (page: Page) => page.locator('#search-results [role=option]');

export async function searchAndOpen(page: Page, query: string, title: string) {
  await page.keyboard.press('Control+k');
  const input = page.getByRole('combobox', { name: 'Search nodes' });
  await expect(input).toBeFocused();
  await input.fill(query);
  await expect(results(page).first()).toContainText(title);
  await input.press('Enter');
  await expect(page.locator('.insp .title')).toHaveValue(title);
  await settle(page);
}

/** Wait until the board has loaded a given node (after an edit the view is re-read asynchronously). */
export async function waitNode(page: Page, id: string, present = true) {
  await page.waitForFunction(([id, present]) => window.__mpn.nodes().some((n: any) => n.id === id) === present, [id, present] as const, { timeout: 15_000, polling: 50 });
  await settle(page);
}
