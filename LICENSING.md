# Licensing of MPN Studio and Yaob

This is a plain-language summary. The binding terms are in `LICENSE`, `LICENSE-WEIGHTS` and `DATA-LICENSE.md`, and in the third-party licences listed in `THIRD-PARTY.md`. Where this summary and those files differ, those files apply. This summary is not legal advice.

## Which part is under which terms

| Part | Terms | File |
|---|---|---|
| Code written by Fylume: MPN Studio (web interface), the Yaob runtime, training, evaluation and corpus pipeline, schemas, documentation | Fylume Source-Available Terms 1.0: the text of the Apache License 2.0 with the Fylume No-Resale Condition 1.0 | `LICENSE` |
| Weights trained by Fylume: the learned router and the four experimental LoRA specialist adapters | Fylume Source-Available Terms 1.0 for weights: the same structure, with weight-specific definitions | `LICENSE-WEIGHTS` |
| Yaob N10K node data (statements, facts, canary and procedure nodes, structure) | CC BY-SA 4.0. The no-resale condition does **not** apply | `DATA-LICENSE.md` |
| Base models Qwen3-4B-Instruct-2507 and Qwen3-Embedding-0.6B | Apache-2.0, from Alibaba Cloud (Qwen). Redistributed unchanged as separate release assets; Fylume's terms do not apply to them | `THIRD-PARTY.md` |
| Fonts (Space Grotesk, Manrope, IBM Plex Mono) | SIL Open Font License 1.1 | `THIRD-PARTY.md` |
| Svelte and Vite code in the built web interface | MIT | `THIRD-PARTY.md` |
| Fylume and Yaob names, logos and brand artwork | Not licensed, except to be reproduced unchanged as part of the code | `LICENSE`, Section 7 of the condition |

## What you may do with Fylume's code and weights

Free of charge, without asking:

- download, run and use them for any purpose, including inside a company or other organisation and for its internal work;
- read and inspect the code and the weights;
- modify them, fine-tune or further train the weights, and build your own work on them;
- give copies away, modified or not, as long as you charge nothing for them and pass on the licence files and notices;
- charge for your own services that are not the software or weights themselves, such as consulting, support, training, integration or development;
- sell your own separate products that work with MPN Studio or Yaob but do not include them;
- use what the software and weights produce when you run them.

## What you may not do

- **Resell.** You may not provide the code or weights, or a modified version of them, to anyone for a fee or other consideration. This covers selling or licensing copies, including within a larger paid product or package, and offering them as a paid hosted or network service where access to them is all or a substantial part of what the customer pays for.
- **Remove attribution.** You must keep the copyright notices, the `NOTICE` file and the licence files of Fylume and of the third parties.
- **Claim authorship.** You may not claim or imply that you wrote the code or trained the weights.
- **Pass off a modified version as the original.** A modified version must say prominently that it is modified, by whom and approximately when, and must not be presented as an original Fylume release or as endorsed by Fylume.

To do something these terms do not allow, ask Fylume for a separate written agreement.

## Why this is source-available, not open source

The source and weights are public and may be used, studied, modified and shared, but not resold. That makes them **source-available**. They are not open source as the Open Source Initiative defines it (<https://opensource.org/osd>):

- Criterion 1, Free Redistribution: "The license shall not restrict any party from selling or giving away the software as a component of an aggregate software distribution containing programs from several different sources." The no-resale condition restricts selling.
- Criterion 6, No Discrimination Against Fields of Endeavor: "The license must not restrict anyone from making use of the program in a specific field of endeavor." The condition rules out one line of business: selling the software or weights, or offering them as a paid service.

For the same reason the weights are not "Open Source AI" under the OSI's Open Source AI Definition 1.0 (<https://opensource.org/ai/open-source-ai-definition>), which requires the freedom to use and share the system "for any purpose" and requires parameters to be made available "under OSI-approved terms". Fylume describes the weights as published under source-available terms. It does not call them open source or open weights.

The terms are also not the Apache License. They reuse its text unchanged and add a condition. The Apache Software Foundation states that a modified licence "is NOT the Apache License" and may not use "Apache" in its name (<https://www.apache.org/foundation/license-faq.html>).

## Upstream licences are preserved

Fylume's no-resale condition applies only to material in which Fylume holds rights. It does not take away any right that someone else has granted:

- The Qwen base models remain under Apache-2.0. Anyone may get them from Fylume's release assets or from Hugging Face and use, modify and sell them under that licence. A model that merges Qwen weights with Fylume's adapters contains both, so both sets of terms apply to it.
- The node data contains Wikipedia text under CC BY-SA 4.0. That licence forbids adding restrictions, so the whole node database is CC BY-SA 4.0 and may be sold by anyone who follows CC BY-SA 4.0.
- Wikidata facts are CC0 1.0.
- The fonts remain under the OFL 1.1, and Svelte and Vite under the MIT License.

## An open question about the weights

Three adapters (astronomy, chemistry, history) and the router were trained on node text derived from Wikipedia. Whether trained weights are an adaptation of the text they were trained on, in the sense of CC BY-SA 4.0, is not settled. Creative Commons writes that "if AI models or outputs are based on ShareAlike content and they will be shared publicly, following the ShareAlike condition would require AI developers to use the same CC license as the original works", and also that "CC licenses apply only when copyright permission is required" (<https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/>). `LICENSE-WEIGHTS` Section 7 therefore provides that, to whatever extent the weights are adaptations of CC BY-SA material, they are licensed under CC BY-SA 4.0, and the no-resale condition does not apply to that extent.

## How to tell an original Fylume release from a modified one

- Original releases are published by Fylume, through its GitHub organisation `sarja-technologies` (<https://github.com/sarja-technologies>) and channels that Fylume names there.
- Every MPN Studio bundle has a `manifest.json` that lists a SHA-256 checksum for each file. A release matches the original only if its checksums match the ones Fylume publishes for that release.
- A modified version must carry a notice saying it is modified, by whom and when. A copy whose files do not match the checksums Fylume publishes is not an original Fylume release, whether or not it carries such a notice.

## Files

- `LICENSE`: terms for Fylume's code
- `LICENSE-WEIGHTS`: terms for Fylume's learned weights, and the obligations for the Qwen base models
- `DATA-LICENSE.md`: terms and attribution for the node data
- `NOTICE`: copyright and attribution notices that must be kept
- `THIRD-PARTY.md`: every third-party component, its version, licence and source
