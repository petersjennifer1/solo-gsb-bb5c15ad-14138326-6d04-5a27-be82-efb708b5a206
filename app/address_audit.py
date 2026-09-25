"""标准地址自匹配巡检 (standard-address self-match audit).

Each standard address should be its own best match: querying the matcher
with the address's own full text ought to land on that address's ID.  This
module runs that check for **every** row of the current standard-address
table using the **default** rules (the configured thresholds -- the same
plain scoring as a regular match; the review-consensus memory is not
consulted):

* every address is scored against the **whole** table, not just the usual
  top-5 candidates;
* only rows that do **not** automatically hit their own ID are reported;
* for every reported row the response carries its own score and every other
  address whose score is at least ``own_score - MARGIN_THRESHOLD`` (the
  blocking candidates, never cut off at five);
* three verdict kinds are distinguished:

  - ``self_low``      自身分数不足 -- the address's own score is itself
    below the auto-hit threshold (corrupt/legacy data or an address whose
    tokens cannot reach an automatic decision);
  - ``top_shift``     首选偏移 -- the own score is sufficient, but another
    address ranks strictly above it (the default query would prefer that
    other address);
  - ``margin_ambiguous`` 分差歧义 -- the address is still the preferred
    candidate and its own score is sufficient, but some other address sits
    within the default score-margin threshold, so the decision is ambiguous.

Like the coverage-gaps manifest this is a pure read: nothing is written to
the match logs and no scoring rules are modified.  Re-opening the page (or
calling the API) after uploading a new table recomputes everything against
the current rows.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from . import db
from .config import Settings
from .matcher import candidate_to_dict, score_candidates

# Risk kinds, in presentation order (most severe first).
SELF_LOW = "self_low"
TOP_SHIFT = "top_shift"
MARGIN_AMBIGUOUS = "margin_ambiguous"
VALID_KINDS = (SELF_LOW, TOP_SHIFT, MARGIN_AMBIGUOUS)

KIND_ORDER = {SELF_LOW: 0, TOP_SHIFT: 1, MARGIN_AMBIGUOUS: 2}

KIND_LABELS = {
    SELF_LOW: "自身分数不足",
    TOP_SHIFT: "首选偏移",
    MARGIN_AMBIGUOUS: "分差歧义",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def compute_audit(conn: sqlite3.Connection, settings: Settings,
                  kind: str | None = None) -> dict[str, Any]:
    """Run the self-match audit against the current standard-address table.

    ``kind`` optionally restricts the returned records to one of
    :data:`VALID_KINDS`; the summary counts always cover every risk.  The
    audit is recomputed on every call, so a freshly uploaded table is
    reflected the next time the page is opened.
    """
    # One snapshot of the current table is shared by every self-match run
    # in this call; rows arrive in id order, which also fixes score ties.
    addresses = db.all_addresses(conn)

    thresholds = {
        "auto_threshold": settings.auto_threshold,
        "margin_threshold": settings.margin_threshold,
        "reject_threshold": settings.reject_threshold,
        "token_weight": settings.token_weight,
    }
    if not addresses:
        return {
            "generated_at": _now(),
            "total_addresses": 0,
            "total_risks": 0,
            SELF_LOW: 0,
            TOP_SHIFT: 0,
            MARGIN_AMBIGUOUS: 0,
            "thresholds": thresholds,
            "risks": [],
        }

    risks: list[dict[str, Any]] = []
    for row in addresses:
        own_id = row["id"]
        # Full, untruncated score list: blockers below the fifth rank must
        # not disappear behind the usual top-5 candidate cut-off.
        scored = score_candidates(row["full_text"], addresses, settings)

        own: Any = None
        rank = None
        for i, cand in enumerate(scored):
            if cand.address["id"] == own_id:
                own, rank = cand, i
                break
        # The address itself is one of the rows, so it must always score.
        assert own is not None, f"address {own_id} missing from its own scoring"

        own_score = own.score
        cutoff = round(own_score - settings.margin_threshold, 4)
        preferred = scored[0]
        # Preserve rank within the full table while collecting every
        # near-tied other address (no top-5 cut-off).
        blockers: list[tuple[int, Any]] = [
            (i, c) for i, c in enumerate(scored)
            if c.address["id"] != own_id and c.score >= cutoff
        ]

        if own_score < settings.auto_threshold:
            risk_kind = SELF_LOW
            reason = (
                f"自身分数 {own_score:.4f} 低于自动命中阈值 "
                f"{settings.auto_threshold:.4f}，用默认规则查询自身都无法自动命中"
            )
        elif preferred.address["id"] != own_id:
            risk_kind = TOP_SHIFT
            reason = (
                f"首选候选为 #{preferred.address['id']}（{preferred.score:.4f}）而非自身 "
                f"#{own_id}（{own_score:.4f}），自身落后首选 "
                f"{preferred.score - own_score:.4f} 分"
            )
        elif not blockers:
            continue
        else:
            # Own is the preferred candidate with an auto-hittable score;
            # a risk remains only when some blocker sits within the margin.
            risk_kind = MARGIN_AMBIGUOUS
            nearest = blockers[0][1]
            reason = (
                f"首选仍为自身 #{own_id}（{own_score:.4f}），但候选 "
                f"#{nearest.address['id']}（{nearest.score:.4f}）分差仅 "
                f"{own_score - nearest.score:.4f}，小于默认分差阈值 "
                f"{settings.margin_threshold:.4f}，存在歧义"
            )

        blocker_dicts = [
            {
                **candidate_to_dict(c),
                "rank": i + 1,
                "score_gap": round(own_score - c.score, 4),
                "is_preferred": c is preferred,
            }
            for i, c in blockers
        ]

        risks.append({
            "address_id": own_id,
            "address": {"id": own_id, "full_text": row["full_text"]},
            "full_text": row["full_text"],
            "external_id": row["external_id"],
            "query": row["full_text"],
            "kind": risk_kind,
            "kind_label": KIND_LABELS[risk_kind],
            "reason": reason,
            "own_score": own_score,
            "own_rank": rank + 1,
            "preferred_id": preferred.address["id"],
            "preferred_score": preferred.score,
            "blocker_threshold": cutoff,
            "blocker_count": len(blocker_dicts),
            "blockers": blocker_dicts,
            "self_candidate": candidate_to_dict(own),
        })

    summary = {
        "total_addresses": len(addresses),
        "total_risks": len(risks),
        SELF_LOW: sum(1 for r in risks if r["kind"] == SELF_LOW),
        TOP_SHIFT: sum(1 for r in risks if r["kind"] == TOP_SHIFT),
        MARGIN_AMBIGUOUS: sum(1 for r in risks if r["kind"] == MARGIN_AMBIGUOUS),
    }

    if kind is not None:
        risks = [r for r in risks if r["kind"] == kind]

    # Most severe kind first, then stable address-id order.
    risks.sort(key=lambda r: (KIND_ORDER[r["kind"]], r["address_id"]))

    return {
        "generated_at": _now(),
        **summary,
        "thresholds": thresholds,
        "risks": risks,
    }
