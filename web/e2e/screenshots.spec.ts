import { test, expect } from '@playwright/test';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { api, openApp, resetView, searchAndOpen, setView, settle, showPanel } from './helpers';

// Real captures of the running app at the three window sizes it has to hold at.
const here = dirname(fileURLToPath(import.meta.url));
const SIZES = [
  [1440, 900],
  [1920, 1080],
  [1024, 700],
] as const;

for (const [w, h] of SIZES) {
  test(`screenshots at ${w}x${h}`, async ({ page, request }) => {
    test.setTimeout(120_000);
    const shot = (name: string) => page.screenshot({ path: join(here, 'screenshots', `${w}x${h}-${name}.png`) });
    await resetView(request);
    await page.setViewportSize({ width: w, height: h });
    await openApp(page);
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(300);
    await shot('1-overview');

    await page.locator('.mrow', { hasText: 'Mathematics and statistics' }).click();
    await settle(page);
    await page.waitForTimeout(300);
    await shot('2-mid-zoom');

    await searchAndOpen(page, 'Arithmetic mean', 'Arithmetic mean');
    const pos = (await api(request, 'api/nodes/1201')).position;
    await setView(page, pos.x + 6, pos.y, 6.5);
    await page.waitForTimeout(900);
    await shot('3-node-zoom-inspector');

    await page.keyboard.press('Control+k');
    await page.getByRole('combobox', { name: 'Search nodes' }).fill('mean');
    await expect(page.locator('#search-results [role=option]').first()).toBeVisible();
    await shot('4-search');
    await page.keyboard.press('Escape');

    await page.locator('.mrow', { hasText: 'Mathematics and statistics' }).click();
    const row = page.locator('.mpns > li', { hasText: 'History' });
    await row.locator('.mrow').click();
    await row.getByRole('button', { name: 'Disable…' }).click();
    await expect(row.locator('.impact')).toBeVisible();
    await settle(page);
    await row.locator('.impact').scrollIntoViewIfNeeded();
    await shot('5-library-impact');
    await row.getByRole('button', { name: 'Cancel' }).click();

    await showPanel(page, 'Schemas');
    await page.locator('.schema #sc-l-0').fill('Facts');
    await expect(page.locator('.schema .check')).toContainText('Relabel only');
    await shot('6-schema-editor');
    await page.locator('.schema .check').scrollIntoViewIfNeeded();
    await shot('6b-schema-validation');
    await page.locator('.schema').getByRole('button', { name: 'Discard draft' }).click();
    await page.getByRole('button', { name: 'Close the Schemas panel' }).click();

    await page.getByRole('tab', { name: 'Run' }).click();
    await page.getByLabel('Message').fill('What is the arithmetic mean of 2, 4 and 9?');
    await page.getByRole('button', { name: 'Send' }).click();
    await expect(page.locator('.run .error-line')).toBeVisible();
    await shot('7-run-panel');

    await showPanel(page, 'Jobs');
    const jobs = page.locator('.jobs');
    await jobs.getByLabel('Steps').fill('30');
    await jobs.getByLabel('Seconds per step').fill('0.4');
    await jobs.getByRole('button', { name: 'Start job' }).click();
    const jrow = jobs.locator('li[data-job]').first();
    await expect(jrow.locator('.log')).toContainText(/step \d+\/30/, { timeout: 25_000 });
    await page.waitForTimeout(1500);
    await shot('8-jobs-panel');
    await jrow.getByRole('button', { name: 'Cancel' }).click();
    await expect(jrow).toHaveAttribute('data-state', 'cancelled', { timeout: 20_000 });
    await page.getByRole('button', { name: 'Close the Jobs panel' }).click();
  });
}
