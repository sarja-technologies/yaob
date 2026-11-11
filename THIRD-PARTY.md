# Third-party components

Every third-party component that MPN Studio and Yaob ship or require, with its licence. Checked on 12 November 2026 against the source listed. Fylume's terms (`LICENSE`, `LICENSE-WEIGHTS`) do not apply to these components and do not restrict any right their licences grant.

"Shipped" means the component, or part of it, is included in what Fylume distributes. "Required" means you obtain it yourself (for example with `pip`, `pnpm` or from Hugging Face) under its own licence.

## Base models

| Name | Revision | Licence | Source | Shipped? |
|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 (frozen core) | `cdbee75f17c01a7cc42f958dc650907174af0554` | Apache-2.0. LICENSE file at that revision: Apache License 2.0 with "Copyright 2024 Alibaba Cloud" | <https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/LICENSE> | Yes, unchanged, as a separate release asset (split into parts) with its LICENSE file. Not in the source repository. Required for the `core` and `yaob` modes |
| Qwen3-Embedding-0.6B (frozen encoder) | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | Apache-2.0, as declared in the model card metadata. The repository has no separate LICENSE file at this revision | <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B> | Yes, unchanged, as a separate release asset, with the Apache License 2.0 text added as LICENSE. Not in the source repository. Required |

## Data sources

| Name | Version | Licence | Source | Shipped? |
|---|---|---|---|---|
| English Wikipedia, article lead sections | one recorded revision per article (9,808 articles) | CC BY-SA 4.0 | <https://en.wikipedia.org/>; reuse terms: <https://en.wikipedia.org/wiki/Wikipedia:Reusing_Wikipedia_content> | Yes, as edited excerpts in the node data. See `DATA-LICENSE.md` |
| Wikipedia "Vital articles" lists, levels 4 and 5 | 32 list pages, one recorded revision each | CC BY-SA 4.0 | <https://en.wikipedia.org/wiki/Wikipedia:Vital_articles> | Titles and sub-domain labels only, in the node data |
| Wikidata structured data | retrieved November 2026 (9,808 items) | CC0 1.0 | <https://www.wikidata.org/wiki/Wikidata:Licensing> | Yes, as facts, relations, aliases and descriptions in the node data |

## Fonts

All three are in `web/public/brand/fonts/` with their OFL licence files, and are copied into the built interface (`web/dist/brand/fonts/`). They came unmodified from the Fylume-Yaob Brand Kit v2.0.

| Name | Font revision | Licence | Source | Shipped? |
|---|---|---|---|---|
| Space Grotesk (`SpaceGrotesk.woff2`) | 2.000 | OFL-1.1. Copyright 2020 The Space Grotesk Project Authors | <https://github.com/floriankarsten/space-grotesk> | Yes |
| Manrope (`Manrope.woff2`) | 4.505 | OFL-1.1. Copyright 2018 The Manrope Project Authors | <https://github.com/googlefonts/manrope> | Yes. The current stylesheet does not use it |
| IBM Plex Mono Regular (`IBMPlexMono-Regular.woff2`) | 2.003 | OFL-1.1. Copyright 2017 IBM Corp., Reserved Font Name "Plex" | <https://github.com/IBM/plex> | Yes |

Licence text: <https://openfontlicense.org/open-font-license-official-text/>. OFL fonts may be bundled and redistributed with other software, including commercially, but not sold on their own (OFL condition 1). Fylume's no-resale condition does not apply to them.

## JavaScript compiled into the built web interface

| Name | Version | Licence | Source | Shipped? |
|---|---|---|---|---|
| svelte (runtime) | 5.57.3 | MIT. Copyright (c) 2016-2025 Svelte Contributors | <https://github.com/sveltejs/svelte> | Yes, compiled into `web/dist/assets/*.js` |
| clsx (svelte runtime dependency) | 2.1.1 | MIT. Copyright (c) Luke Edwards | <https://github.com/lukeed/clsx> | Only if the interface uses dynamic class expressions; the current interface does not |
| esm-env (svelte runtime dependency) | 1.2.2 | MIT. Copyright 2022 Benjamin McCann | <https://github.com/benmccann/esm-env> | Build-time constants only; no code expected in the bundle |
| vite (module preload helper) | 8.3.7 | MIT. Copyright (c) 2019-present, VoidZero Inc. and Vite contributors | <https://github.com/vitejs/vite> | Yes, a small helper injected into the bundle |

The MIT License requires its copyright and permission notice to be included in all copies or substantial portions. A distribution of `web/dist/` should include these notices (see `NOTICE`); the full MIT texts are in each package's `LICENSE` file under `web/node_modules/`.

## JavaScript build and test tools (not shipped)

| Name | Version | Licence | Shipped? |
|---|---|---|---|
| @sveltejs/vite-plugin-svelte | 7.3.1 | MIT | No, build only |
| svelte-check | 4.7.8 | MIT | No, development only |
| typescript | 6.0.3 | Apache-2.0 | No, development only |
| @playwright/test | 1.65.0 | Apache-2.0 | No, tests only |
| @types/node | 26.6.6 | MIT | No, development only |

## Python dependencies (required, not shipped)

Fylume does not redistribute these. They are installed from PyPI under their own licences. Versions are those in the project's `.venv`. For the `model` extra they are the current PyPI releases, because the training machine's environment was not inspected here.

| Name | Version | Licence (package metadata) | Group |
|---|---|---|---|
| fastapi | 0.162.4 | MIT | core |
| starlette (via fastapi) | 1.10.0 | BSD-3-Clause | core |
| uvicorn | 0.54.0 | BSD-3-Clause | core |
| httpx | 0.28.1 | BSD-3-Clause | core |
| pydantic / pydantic-core | 2.14.3 / 2.50.4 | MIT | core |
| orjson | 3.13.0 | MPL-2.0 AND (Apache-2.0 OR MIT) | core |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | core |
| psutil | 7.2.2 | BSD-3-Clause | core |
| torch | 2.15.0 (PyPI) | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib | model |
| transformers | 5.22.2 (PyPI) | Apache-2.0 | model |
| accelerate | 1.15.0 (PyPI) | Apache-2.0 | model |
| safetensors | 0.8.0 (PyPI) | Apache-2.0 | model |
| peft | 0.21.2 (PyPI) | Apache-2.0 | model |
| huggingface_hub | 2.4.0 (PyPI) | Apache-2.0 | model |
| nvidia-ml-py | 13.615.71 (PyPI) | BSD | model |
| pytest | 9.2.0 | MIT | dev |

If a future release bundles any of these (for example in a container image or a frozen executable), their licence texts and notices must be included with it, and this table must be updated.
