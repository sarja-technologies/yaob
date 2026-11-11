// ad-hoc: open a node by search on the running app and screenshot. node e2e/look-node.mjs <query> <out.png> [w] [h]
import { chromium } from '@playwright/test';
const [q, out, w = '1440', h = '900'] = process.argv.slice(2);
const b = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-gpu', '--ignore-gpu-blocklist'] });
const p = await b.newPage({ viewport: { width: +w, height: +h } });
const logs = [];
p.on('pageerror', (e) => logs.push('PAGEERROR: ' + e.message));
p.on('console', (m) => m.type() === 'error' && logs.push('console: ' + m.text()));
await p.goto(process.env.MPN_URL ?? 'http://127.0.0.1:5174/');
await p.waitForFunction(() => window.__mpn && window.__mpn.level() !== null, null, { timeout: 20000 });
await p.keyboard.press('Control+k');
await p.getByRole('combobox', { name: 'Search nodes' }).fill(q);
await p.locator('#search-results [role=option]').first().waitFor();
logs.push('first result: ' + (await p.locator('#search-results [role=option]').first().innerText()).replace(/\n/g, ' | '));
await p.keyboard.press('Enter');
await p.waitForFunction(() => window.__mpn.idle(), null, { timeout: 15000, polling: 50 });
await p.waitForTimeout(800);
await p.screenshot({ path: out });
const facts = p.locator('.insp table');
if (await facts.count()) { await facts.first().scrollIntoViewIfNeeded(); await p.screenshot({ path: out.replace('.png', '-facts.png') }); }
console.log(logs.join('\n'));
await b.close();
