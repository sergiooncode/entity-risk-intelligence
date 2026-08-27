"""
The query workload: 48 analyst-style questions, fixed and shared by every
experiment so that numbers are comparable across strategies.

The set deliberately spans all seven topics and several jurisdictions. Because
the corpus correlates content with metadata, some (query, filter) pairs come
out *aligned* - the query's global nearest neighbours are mostly inside the
filter - and others come out *adversarial*, where the filter deletes precisely
the region the ANN walk heads for. Both cases are present on purpose, and the
experiments report the spread across queries rather than only the mean, because
the mean hides exactly the failure this benchmark is looking for.
"""
from __future__ import annotations

QUERIES: list[tuple[str, str]] = [
    # forced labour
    ("q01", "labour transfer programme supplying a polysilicon plant in Xinjiang"),
    ("q02", "cotton processing facility with restricted worker movement and withheld wages"),
    ("q03", "textile supplier detained at the border under a forced labour presumption"),
    ("q04", "dormitory co-located with a production campus and controlled egress"),
    ("q05", "state-brokered recruitment quotas assigned to a manufacturing enterprise"),
    ("q06", "downstream brand suspended orders pending supply chain traceability review"),
    ("q07", "solar wafer producer named on a forced labour entity list"),

    # export controls
    ("q08", "five-axis machine tools diverted to a listed end user"),
    ("q09", "semiconductor equipment re-exported through a Hong Kong freight forwarder"),
    ("q10", "licence denial for advanced chips under the foreign direct product rule"),
    ("q11", "distributor with no technical staff reselling controlled test equipment"),
    ("q12", "temporary denial order issued after a post-shipment verification failure"),
    ("q13", "dual-use goods routed through an Almaty consolidator"),
    ("q14", "ECCN 3A090 classification dispute over accelerator cards"),

    # sanctions evasion
    ("q15", "shell company incorporated weeks before its first shipment"),
    ("q16", "free zone trading entity with a shared mailbox address and no warehouse"),
    ("q17", "back-to-back invoicing used to obscure the ultimate payer"),
    ("q18", "correspondent bank exited the relationship after a documentary mismatch"),
    ("q19", "transhipment through a free trade zone to a designated procurement agent"),
    ("q20", "serial re-registration of a trading arm after designation"),
    ("q21", "under-valued invoices for electronics moving to a blocked conglomerate"),
    ("q22", "crypto settlement replacing letters of credit after a bank exit"),

    # military end use
    ("q23", "microcontrollers recovered from a downed reconnaissance drone"),
    ("q24", "supplier of gyroscopes to a state defence research institute"),
    ("q25", "components matching cruise missile guidance specifications"),
    ("q26", "military end-use language found in a technical annex"),
    ("q27", "consignee sharing an address with a weapons plant"),
    ("q28", "thermal imaging cores shipped under a civilian end-use certificate"),

    # ownership change
    ("q29", "majority stake transferred to a Cyprus holding after designation"),
    ("q30", "fifty percent rule aggregation across two minority shareholders"),
    ("q31", "nominee director appointed in a third country to obscure control"),
    ("q32", "divestment announced days after a sanctions listing"),
    ("q33", "beneficial ownership register does not identify the ultimate owner"),
    ("q34", "share pledge granted to a bank in a third country"),
    ("q35", "restructuring that left operational control unchanged"),

    # maritime
    ("q36", "tanker with a 46 hour AIS gap before a ship to ship transfer"),
    ("q37", "crude oil transferred at anchor off Fujairah"),
    ("q38", "vessel changed flag three times in fourteen months"),
    ("q39", "single ship company with unverifiable protection and indemnity cover"),
    ("q40", "shadow fleet operator acquiring ageing tankers above scrap value"),
    ("q41", "draught observations inconsistent with the declared cargo quantity"),

    # procurement
    ("q42", "single source award with a four day bid window"),
    ("q43", "tender specification copied verbatim from one vendor's datasheet"),
    ("q44", "surveillance camera contract awarded to a supplier with eleven employees"),
    ("q45", "port crane procurement by a state operator with one qualified bidder"),
    ("q46", "contract extended without re-tendering by a ministry procurement agency"),
    ("q47", "evaluation committee overlapping the winning bidder's advisory board"),

    # deliberately cross-cutting
    ("q48", "entity appearing under a transliterated alias in customs records"),
]


def query_ids() -> list[str]:
    return [q for q, _ in QUERIES]


def query_texts() -> list[str]:
    return [t for _, t in QUERIES]
