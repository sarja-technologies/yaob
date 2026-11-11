import { defineConfig } from '@playwright/test';

// MPN_URL selects the server under test: the Vite dev server, the API serving web/dist, or the real 10K store.
const baseURL = process.env.MPN_URL ?? 'http://127.0.0.1:5174';

export default defineConfig({
  testDir: './e2e',
  outputDir: './test-results',
  timeout: 90_000,
  expect: { timeout: 10_000 },
  workers: 1,
  fullyParallel: false,
  reporter: [['list']],
  use: {
    baseURL,
    channel: 'chrome',
    headless: true,
    viewport: { width: 1440, height: 900 },
    launchOptions: { args: ['--enable-gpu', '--ignore-gpu-blocklist', '--enable-webgl'] },
  },
  projects: [
    { name: 'e2e', testMatch: /(journeys|mocked)\.spec\.ts/ },
    { name: 'shots', testMatch: /screenshots\.spec\.ts/ },
    { name: 'perf', testMatch: /perf\.spec\.ts/ },
  ],
});
