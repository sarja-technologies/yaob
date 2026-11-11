# Yaob N10K node data: licence

This file covers the Yaob N10K node data: the nodes, cells, statements, facts, edges, memberships and source records in a Yaob model store, and the same records when exported by MPN Studio (for example `nodes.jsonl` and `sources.jsonl` in an MPN package or model bundle).

It does not cover the code (see `LICENSE`) or the learned weights (see `LICENSE-WEIGHTS`).

## Licence

The node data, as a whole and in every part that Fylume distributes, is licensed under the **Creative Commons Attribution-ShareAlike 4.0 International licence (CC BY-SA 4.0)**:
<https://creativecommons.org/licenses/by-sa/4.0/legalcode>

Some parts also come with fewer restrictions, because their sources allow it (see the table below). Anyone may use those parts under the more permissive terms.

## What the data is made of

| Part | Count | Origin | Licence of the source | How it is licensed here |
|---|---|---|---|---|
| Statements in Knowledge cells, links and hatnote ("not to be confused with") relations | 9,808 knowledge nodes, one English Wikipedia article each | Lead sections of English Wikipedia articles at a recorded revision, split into sentences and in places shortened or edited | CC BY-SA 4.0 | CC BY-SA 4.0 |
| Topic selection and sub-domain labels | 32 list pages | Wikipedia "Vital articles" lists, levels 4 and 5, at a recorded revision | CC BY-SA 4.0 | CC BY-SA 4.0 |
| Facts, relations, aliases and descriptions | 9,808 Wikidata items | Wikidata structured data (`wbgetentities`) | CC0 1.0 | CC0 1.0 on their own; CC BY-SA 4.0 as part of the database |
| Canary nodes | 160 | Invented by Fylume for diagnostics; they describe nothing real | Fylume's own work | CC BY-SA 4.0 when distributed as part of the database |
| Procedure nodes | 192 | Written by Fylume: formula, parameters and worked example of a function in `mpnstudio/runtime/procedures.py` | Fylume's own work | CC BY-SA 4.0 for the node records. The Python implementation is code and is covered by `LICENSE` |
| Structure: node layout, edges, MPN memberships, cell roles | all | Built by Fylume's corpus pipeline | Fylume's own work | CC BY-SA 4.0 |

Wikidata licensing: <https://www.wikidata.org/wiki/Wikidata:Licensing> (structured data in the main, Property, Lexeme and EntitySchema namespaces is released under CC0 1.0).

## Why the database as a whole is CC BY-SA 4.0

1. The statements are taken from Wikipedia text and in places shortened or edited. Where copyright applies to them, they are Adapted Material under Section 1(a) of CC BY-SA 4.0.
2. Section 3(b) of CC BY-SA 4.0 (ShareAlike) requires Adapted Material to be shared under CC BY-SA 4.0, a later version, or a BY-SA Compatible License.
3. Section 4(b) provides that a database which includes all or a substantial portion of the contents of a licensed database is itself Adapted Material for these purposes.
4. Section 2(a)(5)(B) and Section 3(b)(3) forbid offering or imposing additional or different terms that restrict the rights the licence grants.

Fylume therefore distributes the combined database under CC BY-SA 4.0. Fylume's own contributions (canary nodes, procedure node records, structure) are included under the same licence, and Fylume's CC0 sources are not made more restrictive by being included.

**The Fylume No-Resale Condition does not apply to the node data.** Under CC BY-SA 4.0 anyone may copy, modify and redistribute the node data, including commercially and including for a fee, if they meet the licence conditions.

## Attribution

### Wikipedia text

Every Wikipedia-derived statement cites a source record. All 9,808 Wikipedia source records hold:

- the article title;
- the URL of the exact revision, for example `https://en.wikipedia.org/w/index.php?title=English_language&oldid=1384615058`;
- the revision id (`revision`);
- the retrieval time (`retrieved_at`);
- an attribution string: `Wikipedia contributors, "<article title>", https://en.wikipedia.org/wiki/<article title>`;
- the licence and its URL.

The 32 Vital articles list pages have the same kind of record, with their own revision ids.

The authors of each article are the contributors listed in that article's revision history, which the revision URL leads to. This follows the attribution method described in [Wikipedia:Reusing Wikipedia content](https://en.wikipedia.org/wiki/Wikipedia:Reusing_Wikipedia_content): a hyperlink or URL to the page reused, and a notice that the text is licensed under CC BY-SA.

Modification notice: **statements are excerpts from the lead section of the cited revision. They may have been split into sentences, shortened or edited, and a node may have been edited after import.** Edits made in MPN Studio are recorded in the store's revision history.

### Wikidata

CC0 1.0 does not require attribution. Each node still cites its Wikidata item (`https://www.wikidata.org/wiki/<QID>`).

### Fylume

For Fylume's contributions, Fylume asks to be credited as follows (Section 3(a)(1)(A)(i) of CC BY-SA 4.0):

> Yaob N10K node data, Fylume, 2026, licensed under CC BY-SA 4.0. Includes text from English Wikipedia (Wikipedia contributors, CC BY-SA 4.0) and data from Wikidata (CC0 1.0).

## If you share the data or an adaptation of it

CC BY-SA 4.0 requires you to:

- keep the source records (title, revision URL, attribution string, licence) with the statements they support, or otherwise give attribution in a reasonable manner for the medium you use (Section 3(a)(1) and 3(a)(2));
- state that the material is licensed under CC BY-SA 4.0 and include the licence text or its URL;
- indicate whether you modified the material, and keep the indication of earlier modifications, including the modification notice above;
- license your adaptation under CC BY-SA 4.0, a later version, or a BY-SA Compatible License, and not add terms or technical measures that restrict the rights that licence grants (Section 3(b));
- not suggest that Fylume, the Wikipedia contributors or the Wikimedia Foundation endorse you or your use (Section 2(a)(6)).

Trademark rights, including in "Wikipedia", "Wikidata", "Fylume" and "Yaob", are not licensed (Section 2(b)(2)).

Answers that quote or paraphrase node statements should carry the citation to the node's source, as MPN Studio's runtime does.

## Disclaimer

Fylume offers its contributions to the node data as-is and as-available, without warranty, as set out in Section 5 of CC BY-SA 4.0. Facts and statements may be wrong or out of date. Canary nodes are fictitious.

## Sources

- CC BY-SA 4.0 legal code: <https://creativecommons.org/licenses/by-sa/4.0/legalcode>
- CC0 1.0 legal code: <https://creativecommons.org/publicdomain/zero/1.0/legalcode>
- Wikipedia:Reusing Wikipedia content: <https://en.wikipedia.org/wiki/Wikipedia:Reusing_Wikipedia_content>
- Wikidata:Licensing: <https://www.wikidata.org/wiki/Wikidata:Licensing>
