# Developer handoff

Implement from the supplied masters. Do not redraw or regenerate these logos.

1. Use revision 2.0 assets throughout. The previous F-shaped Fylume symbol and solid Yaob lobes are retired. Read README.md and BRAND-MANUAL.md. Open START-HERE.html for the intended visual treatment.
2. Keep Fylume's site black with crisp glowing-white marks. Use Space Grotesk for Fylume typography and local font files. Do not apply Yaob's warm palette globally to the lab site.
3. Use the exact fylu.me wordmark SVG for the home hero. Use the Prism aperture mark in navigation, favicons and small brand positions. An optional fylume-name wordmark is supplied for editorial contexts.
4. Yaob has its own paper, ember and ink palette, Manrope typography and rounded geometry. In a Yaob section, state “A model family by Fylume” where provenance needs to be explicit. Use the custom path-based Yaob logo, not plain typed text.
5. Import web/brand.css and web/tokens.json. The CSS includes @font-face rules, button treatments and reduced-motion handling. It is a foundation, not a full page layout. Copy assets while preserving relative font paths or update those URLs.
6. For Fylume's halo, add class fylume-glow to the sharp white SVG image. The separate glow-on-black SVG is for presentations; use the clean transparent master on websites.
7. Use Ink on Paper or Ink on Ember for Yaob reading text. Use Ink buttons with Paper labels. See CONTRAST.md for measured combinations.
8. Use supplied favicons and touch icons from favicons/. Choose either the Fylume or Yaob set for the current experience. Example head tags are in web/favicon-example.html.
9. Preserve useful alt text and visible keyboard focus. Honour prefers-reduced-motion. Do not animate body text. No continuous pulsing is needed for the glow.
10. Treat the lab/model descriptions as positioning and research intent. Do not invent model downloads, benchmarks, team biographies, release dates or hardware compatibility. Use accurate supplied content or explicit unavailable states.

This archive is a brand kit, not a production website. START-HERE.html and OVERVIEW.html are offline identity previews.

Use logos/*/svg/*-symbol-micro-*.svg for tiny UI marks. The detailed Yaob symbol contains inner seeds that are intentionally omitted in the micro version. Do not invent an intermediate geometry. Root-level SVG masters use black for Fylume and ember for Yaob; white reversed variants are in their logos folders.
