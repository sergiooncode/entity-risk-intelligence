"""
Synthetic corpus generation: 15,000 entities, 200,000 documents.

The two properties this generator exists to produce:

(a) STRUCTURE IN EMBEDDING SPACE.
    - Clusters: topic x jurisdiction x sector, reinforced by the fact that a
      document's vocabulary is drawn from topic-specific slot pools. Documents
      about maritime AIS gaps in the Gulf do not look like documents about
      labour transfers in Xinjiang.
    - Gradients: a latent `severity` score derived from hops_to_designated
      selects the register of the closing sentences, so documents about the
      same entity and topic spread along a continuous "how categorical is the
      sanctions claim" axis.
    - Near-duplicates: ~3% of documents are deliberate near-copies of another
      document (same entity, same template, one slot changed), which is what
      syndicated risk reporting actually looks like.

(b) FILTER FIELDS CORRELATED WITH CONTENT.
    jurisdiction -> topic -> sector -> vocabulary is a chain of conditional
    distributions, not independent draws. Publication year is conditioned on
    topic. That means a metadata filter is *also* a filter on a region of
    embedding space, which is the case where post-filtering falls apart: the
    ANN walk heads for the global nearest neighbours, and the filter deletes
    exactly those.

    With independent filters, a k*10 overfetch recovers most of the recall and
    the experiment shows nothing interesting. That is why this matters.
"""
from __future__ import annotations

import datetime as dt
import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from . import vocab
from .templates import TEMPLATES, SLOT_POOLS, SEVERITY_CLOSERS, CONTEXT_UNITS

N_ENTITIES = 15_000
N_DOCS = 200_000
SEED = 20260827

NEAR_COLLISION_RATE = 0.05   # entity names one token apart
FKA_RATE = 0.15              # aliases pointing at another entity's name pool
NEAR_DUPLICATE_RATE = 0.03   # documents that are near-copies of another

# P(designated | jurisdiction). Correlated, like everything else here.
DESIGNATION_RATE = {
    "RU": 0.14, "CN": 0.09, "HK": 0.06, "AE": 0.05, "TR": 0.04,
    "KZ": 0.04, "SG": 0.03, "NL": 0.02, "DE": 0.02, "US": 0.02,
}

# hops_to_designated for entities that are not themselves designated.
HOPS_WEIGHTS = {1: .08, 2: .14, 3: .16, 4: .12, 5: .08, 99: .42}

# Latent severity by graph distance. Drives the closing-sentence register.
SEVERITY_BY_HOPS = {0: 0.95, 1: 0.80, 2: 0.66, 3: 0.50, 4: 0.38, 5: 0.30, 99: 0.15}

_DOUBLE_STOP = re.compile(r"\.\.+")


@dataclass
class Entity:
    entity_id: int
    canonical_name: str
    aliases: list[str]
    jurisdiction: str
    sector: str
    is_designated: bool
    hops_to_designated: int


@dataclass
class Document:
    doc_id: int
    entity_id: int
    entity_name: str      # the surface form actually used in this document
    text: str
    jurisdiction: str
    topic: str
    sector: str
    published: str        # ISO date
    is_designated: bool
    hops_to_designated: int
    severity: float


# --------------------------------------------------------------------------
# Weighted sampling
# --------------------------------------------------------------------------

def _pick(rng: random.Random, weights: dict):
    """Weighted choice over a {value: weight} mapping."""
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def _topic_given(jurisdiction: str, sector: str) -> dict:
    """
    P(topic | jurisdiction, sector) proportional to
    P(topic | jurisdiction) * P(sector | topic).

    This is what ties an entity's documents to its sector without making the
    entity a single-topic monolith.
    """
    out = {}
    for topic, pj in vocab.TOPIC_GIVEN_JURISDICTION[jurisdiction].items():
        ps = vocab.SECTOR_GIVEN_TOPIC[topic].get(sector, 0.0)
        w = pj * ps
        if w > 0:
            out[topic] = w
    if not out:  # sector unreachable from this jurisdiction's topic mix
        return dict(vocab.TOPIC_GIVEN_JURISDICTION[jurisdiction])
    return out


# --------------------------------------------------------------------------
# Entities
# --------------------------------------------------------------------------

def generate_entities(rng: random.Random, n: int = N_ENTITIES) -> list[Entity]:
    # Pass 1: jurisdiction, sector, canonical name.
    raw = []
    for eid in range(n):
        juris = _pick(rng, vocab.JURISDICTION_WEIGHTS)
        primary_topic = _pick(rng, vocab.TOPIC_GIVEN_JURISDICTION[juris])
        sector = _pick(rng, vocab.SECTOR_GIVEN_TOPIC[primary_topic])
        name = vocab.make_canonical_name(rng, juris)
        raw.append([eid, name, juris, sector])

    # Pass 2: near-collisions. ~5% of entities are renamed to a one-token
    # variant of another entity in the same jurisdiction. These are the hard
    # cases for any name-based retrieval, and they put genuinely confusable
    # documents next to each other in embedding space.
    by_juris: dict[str, list[int]] = {}
    for i, (_, _, juris, _) in enumerate(raw):
        by_juris.setdefault(juris, []).append(i)

    n_collide = int(n * NEAR_COLLISION_RATE)
    for i in rng.sample(range(n), n_collide):
        juris = raw[i][2]
        pool = by_juris[juris]
        if len(pool) < 2:
            continue
        j = rng.choice(pool)
        if j == i:
            continue
        raw[i][1] = vocab.near_collision(rng, raw[j][1], juris)

    # Pass 3: aliases, designation, hops.
    entities = []
    for eid, name, juris, sector in raw:
        foreign = None
        if rng.random() < FKA_RATE:
            # "formerly known as <some other entity's name>" - deliberate
            # ambiguity: this surface form now points at two entities.
            foreign = raw[rng.randrange(n)][1]
        aliases = vocab.make_aliases(rng, name, juris, foreign)

        designated = rng.random() < DESIGNATION_RATE[juris]
        hops = 0 if designated else _pick(rng, HOPS_WEIGHTS)

        entities.append(Entity(
            entity_id=eid, canonical_name=name, aliases=aliases,
            jurisdiction=juris, sector=sector,
            is_designated=designated, hops_to_designated=hops,
        ))
    return entities


# --------------------------------------------------------------------------
# Document text
# --------------------------------------------------------------------------

def _severity_band(severity: float) -> str:
    if severity < 0.35:
        return "low"
    if severity < 0.70:
        return "mid"
    return "high"


# In forced_labour the sector names *are* commodities, so an unconstrained
# draw produces "cotton lint output enters the polysilicon supply chain".
# Bias the commodity slot toward the entity's sector most of the time.
SECTOR_COMMODITY = {
    "polysilicon": ["polysilicon ingots", "solar wafers"],
    "cotton processing": ["cotton lint", "combed yarn"],
    "textiles": ["viscose staple fibre", "spandex yarn", "combed yarn"],
    "agriculture": ["tomato paste", "processed seafood"],
    "electronics assembly": ["nitrile gloves", "PVC flooring"],
    "mining": ["calcium carbide", "aluminium extrusions"],
    "construction": ["PVC flooring", "aluminium extrusions"],
}


def _bind_slots(rng, template, ent, doc_ctx) -> dict:
    """Bind the template's 4-6 topic slots plus the shared slots."""
    topic = doc_ctx["_topic"]
    bindings = dict(doc_ctx)
    for slot in template["slots"]:
        bindings[slot] = rng.choice(SLOT_POOLS[topic][slot])

    if topic == "forced_labour" and "commodity" in bindings:
        matched = SECTOR_COMMODITY.get(doc_ctx["sector"])
        if matched and rng.random() < 0.7:
            bindings["commodity"] = rng.choice(matched)

    return bindings


def _sentence(template_str: str, bindings: dict) -> str:
    """
    Render one sentence unit.

    Two cosmetic fixes that matter for embedding quality: a slot value at the
    start of a sentence has to be capitalised, and an entity name ending in a
    legal suffix ('Co., Ltd.') must not produce a doubled full stop.
    """
    s = template_str.format_map(bindings)
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return _DOUBLE_STOP.sub(".", s)


def _render(rng, template, bindings, severity: float, min_words=100, max_words=200):
    """
    Emit the lead sentence plus a sampled subset of the remaining units, then
    1-2 severity-register closers, landing in the 100-200 word band.
    """
    lead = template["units"][0]
    rest = template["units"][1:]

    n_units = rng.randint(5, min(6, len(rest)))
    chosen_idx = sorted(rng.sample(range(len(rest)), n_units))
    chosen = [rest[i] for i in chosen_idx]

    # Light reordering: occasionally swap an adjacent pair. Keeps the paragraph
    # coherent while making the token sequence differ between documents that
    # happen to draw the same units.
    for i in range(len(chosen) - 1):
        if rng.random() < 0.25:
            chosen[i], chosen[i + 1] = chosen[i + 1], chosen[i]

    band = _severity_band(severity)
    closers = rng.sample(SEVERITY_CLOSERS[band], 2)
    context = rng.sample(CONTEXT_UNITS, rng.randint(2, 3))

    sentences = [lead] + chosen + context + closers
    render = lambda ss: " ".join(_sentence(s, bindings) for s in ss)
    text = render(sentences)

    # Grow the paragraph if it came out short, using units we did not pick.
    unused = [rest[i] for i in range(len(rest)) if i not in set(chosen_idx)]
    unused += [c for c in CONTEXT_UNITS if c not in context]
    while len(text.split()) < min_words and unused:
        extra = unused.pop(rng.randrange(len(unused)))
        sentences.insert(-len(closers), extra)
        text = render(sentences)

    # Trim if it overran.
    while len(text.split()) > max_words and len(sentences) > 4:
        del sentences[-len(closers) - 1]
        text = render(sentences)

    return text


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------

def generate_documents(rng: random.Random, entities: list[Entity],
                       n: int = N_DOCS) -> list[Document]:
    n_ent = len(entities)

    # Coverage is skewed: a minority of entities attract most of the reporting.
    # Lognormal weights give a long tail without any entity dominating.
    ent_weights = [rng.lognormvariate(0.0, 0.9) for _ in range(n_ent)]
    ent_index = list(range(n_ent))

    # Template popularity is skewed too, so some paragraph shapes are common
    # and others rare. That produces uneven cluster densities, which is what
    # HNSW graph quality actually varies over.
    template_weights = {
        t: [rng.lognormvariate(0.0, 0.5) for _ in TEMPLATES[t]]
        for t in vocab.TOPICS
    }

    docs: list[Document] = []
    # Keep recent (entity, template, bindings) around so near-duplicates can
    # be produced as genuine one-slot edits rather than random similar text.
    recent: list[tuple] = []

    picked_entities = rng.choices(ent_index, weights=ent_weights, k=n)

    for doc_id in range(n):
        make_dupe = docs and rng.random() < NEAR_DUPLICATE_RATE and recent

        if make_dupe:
            ent, template, bindings, topic, published, severity = rng.choice(recent)
            bindings = dict(bindings)
            # Change exactly one topic slot: the "same story, restated by a
            # second outlet" case.
            slot = rng.choice(template["slots"])
            bindings[slot] = rng.choice(SLOT_POOLS[topic][slot])
        else:
            ent = entities[picked_entities[doc_id]]
            topic = _pick(rng, _topic_given(ent.jurisdiction, ent.sector))

            year = _pick(rng, vocab.YEAR_WEIGHTS_GIVEN_TOPIC[topic])
            month = rng.randint(1, 12)
            day = rng.randint(1, 28)
            published = dt.date(year, month, day)

            base = SEVERITY_BY_HOPS[ent.hops_to_designated]
            severity = min(1.0, max(0.0, rng.gauss(base, 0.12)))

            tw = template_weights[topic]
            template = rng.choices(TEMPLATES[topic], weights=tw, k=1)[0]

            surface = (ent.canonical_name if rng.random() < 0.6 or not ent.aliases
                       else rng.choice(ent.aliases))

            bindings = {
                "_topic": topic,
                "entity": surface,
                "alias": rng.choice(ent.aliases) if ent.aliases else ent.canonical_name,
                "peer": entities[rng.randrange(n_ent)].canonical_name,
                "sector": ent.sector,
                "country": vocab.COUNTRY_NAME[ent.jurisdiction],
                "city": rng.choice(vocab.CITY[ent.jurisdiction]),
                "year": published.year,
                "month_year": f"{vocab.MONTHS[published.month - 1]} {published.year}",
                "designation_program": rng.choice(vocab.DESIGNATION_PROGRAMS),
            }
            bindings = _bind_slots(rng, template, ent, bindings)

            recent.append((ent, template, bindings, topic, published, severity))
            if len(recent) > 64:
                recent.pop(0)

        text = _render(rng, template, bindings, severity)

        docs.append(Document(
            doc_id=doc_id,
            entity_id=ent.entity_id,
            entity_name=bindings["entity"],
            text=text,
            jurisdiction=ent.jurisdiction,
            topic=topic,
            sector=ent.sector,
            published=published.isoformat(),
            is_designated=ent.is_designated,
            hops_to_designated=ent.hops_to_designated,
            severity=round(severity, 4),
        ))

    return docs


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def build(cache_dir: Path, seed: int = SEED, n_entities: int = N_ENTITIES,
          n_docs: int = N_DOCS, rebuild: bool = False):
    """
    Generate (or load) the corpus. Cached as JSONL so that a re-run produces
    byte-identical text, which is what makes the embedding cache safe to reuse.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    ent_path = cache_dir / f"entities_{seed}_{n_entities}.jsonl"
    doc_path = cache_dir / f"documents_{seed}_{n_entities}_{n_docs}.jsonl"

    if not rebuild and ent_path.exists() and doc_path.exists():
        with ent_path.open() as f:
            entities = [Entity(**json.loads(l)) for l in f]
        with doc_path.open() as f:
            docs = [Document(**json.loads(l)) for l in f]
        return entities, docs

    rng = random.Random(seed)
    entities = generate_entities(rng, n_entities)
    docs = generate_documents(rng, entities, n_docs)

    with ent_path.open("w") as f:
        for e in entities:
            f.write(json.dumps(asdict(e)) + "\n")
    with doc_path.open("w") as f:
        for d in docs:
            f.write(json.dumps(asdict(d)) + "\n")

    return entities, docs
