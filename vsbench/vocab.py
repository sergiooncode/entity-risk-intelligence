"""
Vocabulary pools, entity-name generation and alias noise.

Everything here is FICTITIOUS. Names are invented; the *shapes* (legal-suffix
conventions, transliteration variance, free-zone naming) are modelled on
publicly documented patterns so that alias noise is realistic rather than
random.

Two properties this module exists to create:

  1. Names collide. Regions and descriptors are drawn from small fixed pools,
     so "Kashgar Hengrui Technology Co., Ltd." and "Kashgar Hengrui Materials
     Co., Ltd." are both plausible and both present.
  2. Aliases are ambiguous. ~15% of entities carry a "formerly known as"
     pointing into a *different* entity's name pool, which means a surface form
     in the corpus does not uniquely identify an entity.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Jurisdictions
# --------------------------------------------------------------------------

JURISDICTIONS = ["CN", "RU", "HK", "AE", "SG", "TR", "KZ", "DE", "NL", "US"]

# Marginal P(jurisdiction). Deliberately skewed: a uniform corpus makes every
# single-attribute filter the same selectivity, which flattens E2.
JURISDICTION_WEIGHTS = {
    "CN": 0.22, "RU": 0.15, "HK": 0.11, "AE": 0.11, "SG": 0.08,
    "TR": 0.08, "KZ": 0.07, "DE": 0.07, "NL": 0.06, "US": 0.05,
}

COUNTRY_NAME = {
    "CN": "China", "RU": "Russia", "HK": "Hong Kong",
    "AE": "the United Arab Emirates", "SG": "Singapore", "TR": "Turkiye",
    "KZ": "Kazakhstan", "DE": "Germany", "NL": "the Netherlands",
    "US": "the United States",
}

CITY = {
    "CN": ["Urumqi", "Shenzhen", "Kashgar", "Ningbo", "Hefei", "Shihezi"],
    "RU": ["Moscow", "Yekaterinburg", "Novosibirsk", "Kazan", "Perm", "Rostov-on-Don"],
    "HK": ["Kowloon", "Sheung Wan", "Tsuen Wan", "Kwun Tong", "Wan Chai"],
    "AE": ["Dubai", "Sharjah", "Fujairah", "Jebel Ali", "Abu Dhabi", "Ajman"],
    "SG": ["Singapore", "Jurong", "Tuas", "Pasir Panjang"],
    "TR": ["Istanbul", "Izmir", "Mersin", "Gebze", "Bursa"],
    "KZ": ["Almaty", "Astana", "Karaganda", "Aktau", "Shymkent"],
    "DE": ["Hamburg", "Stuttgart", "Duesseldorf", "Bremen", "Leipzig"],
    "NL": ["Rotterdam", "Amsterdam", "Eindhoven", "Vlissingen", "Delft"],
    "US": ["Houston", "Newark", "Long Beach", "Miami", "Chicago"],
}

# --------------------------------------------------------------------------
# Topics and sectors
# --------------------------------------------------------------------------

TOPICS = [
    "forced_labour", "export_controls", "sanctions_evasion",
    "military_end_use", "ownership_change", "maritime", "procurement",
]

# P(topic | jurisdiction). This is the correlation that makes E2 meaningful:
# a jurisdiction filter is *also* a topic filter, and therefore also a filter
# on a region of embedding space.
TOPIC_GIVEN_JURISDICTION = {
    "CN": {"forced_labour": .34, "export_controls": .22, "military_end_use": .14,
           "procurement": .12, "ownership_change": .10, "maritime": .05,
           "sanctions_evasion": .03},
    "RU": {"sanctions_evasion": .26, "export_controls": .20, "military_end_use": .20,
           "maritime": .14, "procurement": .10, "ownership_change": .09,
           "forced_labour": .01},
    "HK": {"sanctions_evasion": .30, "ownership_change": .24, "export_controls": .18,
           "procurement": .12, "maritime": .10, "military_end_use": .05,
           "forced_labour": .01},
    "AE": {"sanctions_evasion": .32, "maritime": .30, "ownership_change": .14,
           "procurement": .10, "export_controls": .10, "military_end_use": .03,
           "forced_labour": .01},
    "SG": {"maritime": .34, "sanctions_evasion": .22, "export_controls": .16,
           "ownership_change": .12, "procurement": .10, "military_end_use": .05,
           "forced_labour": .01},
    "TR": {"sanctions_evasion": .30, "export_controls": .20, "maritime": .18,
           "procurement": .14, "ownership_change": .12, "military_end_use": .05,
           "forced_labour": .01},
    "KZ": {"sanctions_evasion": .34, "export_controls": .24, "procurement": .16,
           "ownership_change": .12, "maritime": .08, "military_end_use": .05,
           "forced_labour": .01},
    "DE": {"export_controls": .34, "procurement": .20, "ownership_change": .18,
           "sanctions_evasion": .14, "military_end_use": .08, "maritime": .05,
           "forced_labour": .01},
    "NL": {"export_controls": .32, "maritime": .20, "procurement": .16,
           "ownership_change": .16, "sanctions_evasion": .12, "military_end_use": .03,
           "forced_labour": .01},
    "US": {"procurement": .28, "export_controls": .26, "ownership_change": .20,
           "sanctions_evasion": .14, "military_end_use": .08, "maritime": .03,
           "forced_labour": .01},
}

# P(sector | topic). Sector words appear in the document text, so this is a
# second content/metadata correlation channel.
SECTOR_GIVEN_TOPIC = {
    "forced_labour": {"polysilicon": .22, "cotton processing": .22, "textiles": .20,
                      "agriculture": .12, "electronics assembly": .10, "mining": .08,
                      "construction": .06},
    "export_controls": {"semiconductors": .28, "machine tools": .20,
                        "aerospace components": .16, "electronics assembly": .14,
                        "surveillance technology": .12, "battery cells": .10},
    "sanctions_evasion": {"logistics": .26, "petrochemicals": .20, "banking": .18,
                          "shipping": .16, "metals trading": .12, "electronics assembly": .08},
    "military_end_use": {"aerospace components": .30, "machine tools": .22,
                         "semiconductors": .20, "surveillance technology": .14,
                         "metals trading": .08, "shipping": .06},
    "ownership_change": {"banking": .24, "mining": .20, "petrochemicals": .18,
                         "logistics": .16, "metals trading": .12, "construction": .10},
    "maritime": {"shipping": .46, "petrochemicals": .24, "logistics": .18,
                 "metals trading": .12},
    "procurement": {"construction": .22, "surveillance technology": .20,
                    "logistics": .18, "machine tools": .16, "banking": .12,
                    "aerospace components": .12},
}

# Publication-year skew per topic. Sanctions-evasion reporting is a post-2022
# phenomenon; forced-labour reporting peaks around UFLPA implementation.
YEAR_WEIGHTS_GIVEN_TOPIC = {
    "forced_labour":     {2018: .04, 2019: .06, 2020: .12, 2021: .18, 2022: .20,
                          2023: .16, 2024: .12, 2025: .08, 2026: .04},
    "export_controls":   {2018: .04, 2019: .06, 2020: .07, 2021: .09, 2022: .14,
                          2023: .17, 2024: .17, 2025: .16, 2026: .10},
    "sanctions_evasion": {2018: .02, 2019: .02, 2020: .03, 2021: .04, 2022: .17,
                          2023: .21, 2024: .21, 2025: .19, 2026: .11},
    "military_end_use":  {2018: .03, 2019: .04, 2020: .05, 2021: .06, 2022: .18,
                          2023: .21, 2024: .20, 2025: .15, 2026: .08},
    "ownership_change":  {2018: .09, 2019: .10, 2020: .10, 2021: .11, 2022: .14,
                          2023: .13, 2024: .12, 2025: .12, 2026: .09},
    "maritime":          {2018: .04, 2019: .05, 2020: .06, 2021: .07, 2022: .16,
                          2023: .19, 2024: .18, 2025: .16, 2026: .09},
    "procurement":       {2018: .08, 2019: .09, 2020: .10, 2021: .11, 2022: .13,
                          2023: .13, 2024: .13, 2025: .13, 2026: .10},
}

# --------------------------------------------------------------------------
# Name pools
# --------------------------------------------------------------------------

CN_REGION = ["Xinjiang", "Kashgar", "Aksu", "Hotan", "Karamay", "Shihezi",
             "Urumqi", "Bortala", "Changji", "Yili", "Turpan", "Tianshan",
             "Korla", "Kuqa", "Yarkant", "Altay", "Tacheng", "Hami"]
CN_DESCRIPTOR = ["Hengrui", "Jinlong", "Zhongtai", "Guangyuan", "Xingchen",
                 "Ruifeng", "Dazheng", "Yongtai", "Haoyu", "Chuangxin",
                 "Lianhe", "Shengda", "Tianhe", "Bocheng", "Weiye",
                 "Jiacheng", "Mingxin", "Zhenghe"]
CN_SUFFIX = ["Group Co., Ltd.", "Industrial Co., Ltd.", "Technology Co., Ltd.",
             "Holdings Ltd.", "Materials Co., Ltd.", "Manufacturing Co., Ltd.",
             "Textile Co., Ltd.", "New Energy Co., Ltd."]

HK_REGION = ["Kowloon", "Tsuen Wan", "Sha Tin", "Kwun Tong", "Aberdeen",
             "Causeway", "Sheung Wan", "Wan Chai", "Tsim Sha Tsui", "Yau Ma Tei"]
HK_SUFFIX = ["Trading Ltd.", "International Ltd.", "Enterprises Ltd.",
             "Holdings Ltd.", "Development Ltd.", "Industrial Ltd."]

SLAVIC_PREFIX = ["OAO", "OOO", "PAO", "AO", "ZAO"]
RU_STEM = ["Severstroy", "Uralmash-Tekhnika", "Novagaz", "Rostransmash",
           "Sibirtorg", "Metallresurs", "Kronshtadt-Invest", "Yuzhpromexport",
           "Volgotekh", "Baltpromsnab", "Nizhmash", "Tekhnopolis",
           "Energomash-Yug", "Promsyrye", "Vostokstal", "Tsentrmetall",
           "Khimprom-Sever", "Yamalgaz", "Krasnoyarsktrans", "Permsplav"]
KZ_STEM = ["Altynmash", "Karagandaresurs", "Tengiztrans", "Astanaprom",
           "Baikonurstroy", "Kazsplav", "Zhezkazgan-Invest", "Semeytorg",
           "Aktaulogistik", "Shymkenttekh"]

INTL_WORD_A = ["Meridian", "Halcyon", "Aventine", "Northgate", "Calder",
               "Brightwater", "Stellaris", "Quorum", "Vantage", "Ardent",
               "Kestrel", "Lumen", "Orrery", "Pinnacle", "Thalassa",
               "Corvid", "Harbinger", "Solace", "Verity", "Anchorage"]
INTL_WORD_B = ["Marine", "Commodities", "Resources", "Logistics", "Capital",
               "Petro", "Metals", "Systems", "Technologies", "Freight",
               "Chartering", "Industrial", "Materials", "Energy",
               "Bunkering", "Instruments"]

INTL_SUFFIX = {
    "AE": ["FZE", "DMCC", "FZ-LLC", "Trading LLC", "General Trading LLC"],
    "SG": ["Pte Ltd", "Pte. Ltd.", "Holdings Pte Ltd", "Shipping Pte Ltd"],
    "TR": ["A.S.", "Ltd. Sti.", "Dis Ticaret A.S.", "Sanayi A.S."],
    "DE": ["GmbH", "AG", "GmbH & Co. KG", "Handels GmbH"],
    "NL": ["B.V.", "Holding B.V.", "N.V.", "Trading B.V."],
    "US": ["Inc.", "LLC", "Corp.", "Holdings Inc."],
}

# Legal suffixes stripped when deriving the "short form" alias.
STRIPPABLE_SUFFIXES = sorted(
    set(CN_SUFFIX + HK_SUFFIX + [s for v in INTL_SUFFIX.values() for s in v]),
    key=len, reverse=True,
)

DESIGNATION_PROGRAMS = [
    "the OFAC SDN List (RUSSIA-EO14024)", "the OFAC SDN List (NPWMD)",
    "the BIS Entity List", "the UFLPA Entity List", "the DoD 1260H List",
    "the BIS Military End User List", "the EU consolidated list",
    "the UK OFSI consolidated list",
]

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


# --------------------------------------------------------------------------
# Name construction
# --------------------------------------------------------------------------

def make_canonical_name(rng, jurisdiction: str) -> str:
    """Build a canonical legal name in the style of the given jurisdiction."""
    if jurisdiction == "CN":
        return (f"{rng.choice(CN_REGION)} {rng.choice(CN_DESCRIPTOR)} "
                f"{rng.choice(CN_SUFFIX)}")
    if jurisdiction == "HK":
        return (f"{rng.choice(HK_REGION)} {rng.choice(CN_DESCRIPTOR)} "
                f"{rng.choice(HK_SUFFIX)}")
    if jurisdiction == "RU":
        return f"{rng.choice(SLAVIC_PREFIX)} {rng.choice(RU_STEM)}"
    if jurisdiction == "KZ":
        return f"{rng.choice(SLAVIC_PREFIX)} {rng.choice(KZ_STEM)}"
    return (f"{rng.choice(INTL_WORD_A)} {rng.choice(INTL_WORD_B)} "
            f"{rng.choice(INTL_SUFFIX[jurisdiction])}")


def strip_suffix(name: str) -> str:
    """Drop a trailing legal suffix, e.g. 'X Group Co., Ltd.' -> 'X'."""
    for suf in STRIPPABLE_SUFFIXES:
        if name.endswith(" " + suf):
            return name[: -(len(suf) + 1)].strip()
    return name


def transliterate(name: str) -> str:
    """
    The kh/h, y/i, ts/c variance you get when the same Cyrillic name is
    romanised by two different systems.
    """
    out = name.replace("kh", "h").replace("Kh", "H")
    out = out.replace("ts", "c").replace("Ts", "C")
    out = out.replace("y", "i").replace("Y", "I")
    return out


def legal_form_variant(name: str) -> str:
    """OAO -> JSC, OOO -> LLC, PAO -> PJSC: the same company, filed twice."""
    for src, dst in (("PAO", "PJSC"), ("OAO", "JSC"), ("ZAO", "CJSC"),
                     ("OOO", "LLC"), ("AO", "JSC")):
        if name.startswith(src + " "):
            return dst + " " + name[len(src) + 1:]
    return name


def initialism(name: str) -> str:
    """'Kashgar Hengrui Technology Co., Ltd.' -> 'KHT Group'."""
    core = strip_suffix(name)
    tokens = [t for t in core.split() if t[:1].isalpha()]
    if len(tokens) < 2:
        return core
    return "".join(t[0].upper() for t in tokens) + " Group"


def make_aliases(rng, canonical: str, jurisdiction: str, foreign_name: str | None):
    """
    2-4 aliases per entity.

    Candidate forms: suffix-stripped, whitespace-removed, initialism,
    transliteration / legal-form variant, and (when `foreign_name` is supplied,
    ~15% of entities) a "formerly known as" pointing at an unrelated entity's
    name. That last one is intentional: it means a surface form in the corpus
    does not uniquely resolve to an entity.
    """
    short = strip_suffix(canonical)
    candidates = [short, short.replace(" ", ""), initialism(canonical)]

    if jurisdiction in ("RU", "KZ"):
        candidates.append(transliterate(short))
        candidates.append(legal_form_variant(canonical))
    elif jurisdiction in ("CN", "HK"):
        candidates.append(short + " Co Ltd")
        # Pinyin syllable-splitting variance: "Hengrui" vs "Heng Rui".
        parts = short.split()
        if len(parts) >= 2 and len(parts[1]) > 4:
            mid = parts[1]
            candidates.append(" ".join(
                [parts[0], mid[: len(mid) // 2] + " " + mid[len(mid) // 2:]]
                + parts[2:]))
    else:
        candidates.append(short.upper())

    if foreign_name is not None:
        candidates.append(f"{short} (formerly {strip_suffix(foreign_name)})")

    seen, uniq = set(), []
    for c in candidates:
        c = c.strip()
        if c and c != canonical and c not in seen:
            seen.add(c)
            uniq.append(c)

    n = min(len(uniq), rng.randint(2, 4))
    return uniq[:n]


def near_collision(rng, name: str, jurisdiction: str) -> str:
    """
    Return a name differing from `name` by exactly one token, drawn from the
    same pool the original token came from. These are the hard cases: two
    distinct entities whose names differ by one word.
    """
    tokens = name.split()
    pools = []
    if jurisdiction == "CN":
        pools = [(0, CN_REGION), (1, CN_DESCRIPTOR)]
    elif jurisdiction == "HK":
        pools = [(0, HK_REGION), (1, CN_DESCRIPTOR)]
    elif jurisdiction in ("RU", "KZ"):
        stems = RU_STEM if jurisdiction == "RU" else KZ_STEM
        pools = [(0, SLAVIC_PREFIX), (1, stems)]
    else:
        pools = [(0, INTL_WORD_A), (1, INTL_WORD_B)]

    idx, pool = rng.choice(pools)
    if idx >= len(tokens):
        return name
    replacement = rng.choice([p for p in pool if p != tokens[idx]] or pool)
    tokens[idx] = replacement
    return " ".join(tokens)
