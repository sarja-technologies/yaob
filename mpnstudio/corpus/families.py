"""The subject families (top-level MPNs) of the starter network and how list sections map to them.

A topic's place in Wikipedia's Vital Articles hierarchy decides its *primary* MPN.
Rules are ordered; the first rule whose words all occur in the topic's section
path wins. Shared memberships are derived later from links and structured data.
"""
from __future__ import annotations

FAMILIES = [
    ("language-linguistics", "Language and linguistics", "Languages, writing systems, grammar, phonetics and communication."),
    ("mathematics-statistics", "Mathematics and statistics", "Numbers, algebra, geometry, analysis, discrete mathematics, probability and statistics."),
    ("computing-ai", "Computing and AI", "Computer science, software, hardware, networks, algorithms and artificial intelligence."),
    ("physics", "Physics", "Mechanics, electromagnetism, thermodynamics, quantum and particle physics, measurement."),
    ("chemistry", "Chemistry", "Elements, compounds, reactions, bonding and analytical methods."),
    ("biology", "Biology", "Organisms, cells, genetics, evolution, anatomy, physiology and ecology."),
    ("earth-environment", "Earth, environment and climate", "Geology, atmosphere, oceans, climate, weather and natural hazards."),
    ("astronomy", "Astronomy and space", "Solar system, stars, galaxies, cosmology, observation and spaceflight."),
    ("medicine-health", "Medicine and health", "Diseases, treatments, drugs, public health, fitness and medical technology."),
    ("engineering-technology", "Engineering and technology", "Machines, energy, materials, transport, infrastructure, electronics and industry."),
    ("history", "History", "Periods, events, states, leaders and movements from prehistory to the present."),
    ("geography-geopolitics", "Geography and geopolitics", "Countries, regions, cities, physical features, borders, war and international bodies."),
    ("economics-business", "Economics and business", "Markets, money, finance, trade, companies and economic thought."),
    ("civics-law", "Civics, law and government", "Law, rights, crime, political systems, ideologies and institutions."),
    ("philosophy-religion", "Philosophy, logic and religion", "Philosophy, logic, ethics, scientific method, religions and mythology."),
    ("psychology-society", "Psychology and society", "Mind and behaviour, emotions, social structure, culture, education and family."),
    ("arts-literature-music", "Arts, literature and music", "Literature, music, visual and performing arts, architecture, film and media."),
    ("everyday-life", "Everyday life", "Food and drink, clothing, home, sport, games, recreation and festivals."),
]
FAMILY_SLUGS = [f[0] for f in FAMILIES]

# Compound section names contain words ("health", "mathematicians", "scientists") that would otherwise match
# rules meant for their sub-sections, so they are collapsed to single tokens before matching.
COMPOUND = [
    ("scientists, inventors, and mathematicians", "stem-people"),
    ("scientists, inventors and mathematicians", "stem-people"),
    ("biology and health sciences", "lifesci"),
    ("biological and health sciences", "lifesci"),
    ("health, medicine and disease", "healthcare"),
    ("philosophers, historians, political and social scientists", "thinkers"),
    ("philosophers, historians, and political and social scientists", "thinkers"),
    ("social scientists", "socsci"),
    ("history of science and technology", "science-history"),
    ("philosophy of science", "science-philosophy"),
]

# (first path segment must start with this, words that must all occur in the remaining segments, family).
# Ordered; the first match wins.
RULES: list[tuple[str, tuple[str, ...], str]] = [
    # --- people, by field
    ("people", ("stem-people", "computer scientist"), "computing-ai"),
    ("people", ("stem-people", "mathematician"), "mathematics-statistics"),
    ("people", ("stem-people", "physic"), "physics"),
    ("people", ("stem-people", "chemis"), "chemistry"),
    ("people", ("stem-people", "biolog"), "biology"),
    ("people", ("stem-people", "medicine"), "medicine-health"),
    ("people", ("stem-people", "astronom"), "astronomy"),
    ("people", ("stem-people", "earth"), "earth-environment"),
    ("people", ("stem-people", "inventors and engineers"), "engineering-technology"),
    ("people", ("stem-people", "pre-modern"), "history"),
    ("people", ("stem-people",), "physics"),
    ("people", ("thinkers", "economist"), "economics-business"),
    ("people", ("thinkers", "psycholog"), "psychology-society"),
    ("people", ("thinkers", "linguist"), "language-linguistics"),
    ("people", ("thinkers", "historian"), "history"),
    ("people", ("thinkers", "archaeolog"), "history"),
    ("people", ("thinkers", "political"), "civics-law"),
    ("people", ("thinkers", "jurist"), "civics-law"),
    ("people", ("thinkers", "legal"), "civics-law"),
    ("people", ("thinkers", "geograph"), "geography-geopolitics"),
    ("people", ("thinkers", "socsci"), "psychology-society"),
    ("people", ("thinkers",), "philosophy-religion"),
    ("people", ("religious figures",), "philosophy-religion"),
    ("people", ("politicians and leaders",), "history"),
    ("people", ("military",), "history"),
    ("people", ("rebels",), "history"),
    ("people", ("explorer",), "geography-geopolitics"),
    ("people", ("businesspeople",), "economics-business"),
    ("people", ("crim",), "civics-law"),
    ("people", ("law enforcement",), "civics-law"),
    ("people", ("sports",), "everyday-life"),
    ("people", ("chefs",), "everyday-life"),
    ("people", ("writers",), "arts-literature-music"),
    ("people", ("journalist",), "arts-literature-music"),
    ("people", ("musician",), "arts-literature-music"),
    ("people", ("artist",), "arts-literature-music"),
    ("people", ("director",), "arts-literature-music"),
    ("people", ("filmmaker",), "arts-literature-music"),
    ("people", ("entertainer",), "arts-literature-music"),
    ("people", (), "history"),
    ("arts", (), "arts-literature-music"),
    # --- biology and health
    ("lifesci", ("health",), "medicine-health"),
    ("lifesci", ("morbidity",), "medicine-health"),
    ("lifesci", (), "biology"),
    # --- everyday life
    ("everyday life", ("sexuality",), "psychology-society"),
    ("everyday life", ("family",), "psychology-society"),
    ("everyday life", ("stages of life",), "psychology-society"),
    ("everyday life", (), "everyday-life"),
    ("geography", (), "geography-geopolitics"),
    ("history", (), "history"),
    ("mathematics", ("theoretical computer science",), "computing-ai"),
    ("mathematics", (), "mathematics-statistics"),
    ("philosophy and religion", (), "philosophy-religion"),
    # --- physical sciences
    ("physical sciences", ("astronomy",), "astronomy"),
    ("physical sciences", ("chemistry",), "chemistry"),
    ("physical sciences", ("earth science",), "earth-environment"),
    ("physical sciences", ("science basics",), "philosophy-religion"),
    ("physical sciences", ("research methods",), "philosophy-religion"),
    ("physical sciences", (), "physics"),
    # --- society and social sciences
    ("society and social sciences", ("language",), "language-linguistics"),
    ("society and social sciences", ("communication",), "language-linguistics"),
    ("society and social sciences", ("business and economics",), "economics-business"),
    ("society and social sciences", ("companies",), "economics-business"),
    ("society and social sciences", ("law",), "civics-law"),
    ("society and social sciences", ("politics and government",), "civics-law"),
    ("society and social sciences", ("war and military",), "geography-geopolitics"),
    ("society and social sciences", ("organizations",), "geography-geopolitics"),
    ("society and social sciences", ("psychology",), "psychology-society"),
    ("society and social sciences", ("mass media",), "arts-literature-music"),
    ("society and social sciences", ("journalism",), "arts-literature-music"),
    ("society and social sciences", ("festivals",), "everyday-life"),
    ("society and social sciences", (), "psychology-society"),
    # --- technology
    ("technology", ("computing and information",), "computing-ai"),
    ("technology", ("medical technology",), "medicine-health"),
    ("technology", ("biotechnology",), "biology"),
    ("technology", ("space",), "astronomy"),
    ("technology", ("astronomical technology",), "astronomy"),
    ("technology", (), "engineering-technology"),
]


def segments(page: str, path: list[str]) -> list[str]:
    tail = page.split("/Level ", 1)[-1]
    tail = tail.split("/", 1)[1] if "/" in tail else ""
    segs = [x for x in tail.split("/") if x] + list(path)
    out = []
    for x in segs:
        x = x.lower().strip()
        for a, b in COMPOUND:
            x = x.replace(a, b)
        out.append(x)
    return out


def path_key(page: str, path: list[str]) -> str:
    return " > ".join(segments(page, path))


def family_of(page: str, path: list[str]) -> str | None:
    segs = segments(page, path)
    if not segs:
        return None
    rest = " > ".join(segs[1:])
    for first, words, fam in RULES:
        if segs[0].startswith(first) and all(w in rest for w in words):
            return fam
    return None


# Target number of knowledge nodes per family in Yaob N10K. Procedure nodes are added on top (see procedures.py)
# so that the release has exactly 10,000 canonical nodes. The targets widen the families that the level-4 list
# covers thinly (computing, psychology, economics, law, linguistics, mathematics) and trim the ones it covers
# heavily (animals, people, cities), so no family dominates and none is token.
TARGETS = {
    "language-linguistics": 380,
    "mathematics-statistics": 560,
    "computing-ai": 520,
    "physics": 520,
    "chemistry": 460,
    "biology": 760,
    "earth-environment": 460,
    "astronomy": 400,
    "medicine-health": 560,
    "engineering-technology": 680,
    "history": 980,
    "geography-geopolitics": 900,
    "economics-business": 420,
    "civics-law": 440,
    "philosophy-religion": 520,
    "psychology-society": 460,
    "arts-literature-music": 640,
    "everyday-life": 340,
}
