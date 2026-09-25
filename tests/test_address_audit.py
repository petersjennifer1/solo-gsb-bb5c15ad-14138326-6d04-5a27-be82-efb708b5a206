"""Tests for the standard-address self-match audit (``/api/address-audit``)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import address_audit, db, ingest  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402
from app.normalization import normalize  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _insert_duplicate_row(conn, full_text: str, external_id: str) -> int:
    """Insert an address row that normalises identically to another one,
    bypassing the upload-time duplicate guard (as a legacy/bulk import might).
    """
    n = normalize(full_text)
    aid = db.insert_address(
        conn,
        full_text=full_text,
        tokens=n.tokens,
        street_tokens=n.street_tokens,
        locality_tokens=n.locality_tokens,
        house=n.house,
        fields={"line1": None, "city": None, "state": None,
                "postal_code": None, "country": None,
                "external_id": external_id},
        raw={"full_text": full_text},
    )
    conn.commit()
    return aid


def _risk_map(data: dict) -> dict:
    return {r["address_id"]: r for r in data["risks"]}


# --------------------------------------------------------------------- basic


def test_sample_table_passes_self_match_audit(client):
    # The bundled sample addresses must all automatically hit themselves;
    # the near-duplicate pairs (50/52 Hillcrest, 26/99 SW 26th) differ enough
    # that nobody falls within the default score-margin threshold.
    r = client.get("/api/address-audit")
    assert r.status_code == 200
    data = r.json()
    assert data["total_addresses"] == 15
    assert data["total_risks"] == 0
    assert data["self_low"] == 0
    assert data["top_shift"] == 0
    assert data["margin_ambiguous"] == 0
    assert data["risks"] == []
    assert data["thresholds"]["auto_threshold"] == 0.80
    assert data["thresholds"]["margin_threshold"] == 0.06
    assert data["generated_at"]


# ------------------------------------------------------- ambiguity / shift


def test_exact_duplicate_rows_flag_top_shift_and_margin_ambiguous(client):
    from app.main import get_conn

    conn = get_conn()
    id_a = _insert_duplicate_row(
        conn, "2048 Mirrored Lane, Twinville 61001", "AUD-DUP-A")
    id_b = _insert_duplicate_row(
        conn, "2048 Mirrored Lane, Twinville 61001", "AUD-DUP-B")

    data = client.get("/api/address-audit").json()
    risks = _risk_map(data)
    # Score ties keep id order: the smaller id stays preferred (ambiguous),
    # the larger id is no longer its own first choice (top shift).
    ra, rb = risks[id_a], risks[id_b]
    assert ra["kind"] == "margin_ambiguous"
    assert ra["own_score"] == 1.0
    assert ra["own_rank"] == 1
    assert ra["preferred_id"] == id_a
    assert rb["kind"] == "top_shift"
    assert rb["preferred_id"] == id_a
    assert rb["own_rank"] == 2

    for risk, other_id in ((ra, id_b), (rb, id_a)):
        assert risk["query"] == risk["full_text"]
        assert risk["kind_label"] == address_audit.KIND_LABELS[risk["kind"]]
        assert risk["blocker_count"] == 1
        blocker = risk["blockers"][0]
        assert blocker["address"]["id"] == other_id
        assert blocker["score"] == 1.0
        assert blocker["score_gap"] == 0.0
        assert blocker["score"] >= risk["blocker_threshold"]
        assert blocker["address"]["id"] != risk["address_id"]
    assert ra["blockers"][0]["is_preferred"] is False
    assert rb["blockers"][0]["is_preferred"] is True


def test_near_tied_address_flagged_margin_ambiguous(client):
    from app.main import get_conn

    conn = get_conn()
    id_c = _insert_duplicate_row(
        conn, "303 Alpha Same Street, Alpha City", "AUD-NEAR-C")
    id_d = _insert_duplicate_row(
        conn, "303 Alpha Same Street, Alpha City Beta", "AUD-NEAR-D")

    data = client.get("/api/address-audit").json()
    risks = _risk_map(data)
    assert id_c in risks and id_d in risks
    for aid in (id_c, id_d):
        risk = risks[aid]
        assert risk["kind"] == "margin_ambiguous"
        assert risk["own_score"] >= risk["blocker_threshold"]
        # The blocker score is within the default margin threshold.
        gap = risk["own_score"] - risk["blockers"][0]["score"]
        assert 0.0 < gap < data["thresholds"]["margin_threshold"]
        assert risk["preferred_id"] == aid


# ---------------------------------------------------------------- self_low


def test_legacy_row_without_tokens_is_self_low(client):
    from app.main import get_conn

    conn = get_conn()
    legacy_id = db.insert_address(
        conn,
        full_text="9999 Legacy Hollow, Lost Town",
        tokens=[], street_tokens=[], locality_tokens=[], house=None,
        fields={"line1": None, "city": None, "state": None,
                "postal_code": None, "country": None, "external_id": None},
        raw={},
    )
    conn.commit()

    data = client.get("/api/address-audit").json()
    risks = _risk_map(data)
    assert legacy_id in risks
    risk = risks[legacy_id]
    assert risk["kind"] == "self_low"
    assert risk["own_score"] < data["thresholds"]["auto_threshold"]
    assert "自身分数" in risk["reason"]
    # Even though the self score is far below auto-hit, every other address
    # above own_score - margin is still listed (not cut off at five).
    assert risk["blocker_count"] == len(risk["blockers"])
    assert all(b["address"]["id"] != legacy_id for b in risk["blockers"])
    assert all(b["score"] >= risk["blocker_threshold"] for b in risk["blockers"])


def test_self_low_dominates_top_shift_classification(client):
    # An address whose own score is below the auto threshold is "self_low"
    # even when another address ranks above it.
    from app.main import get_conn

    conn = get_conn()
    lid = db.insert_address(
        conn,
        full_text="8888 Empty Legacy Row, Nowhere",
        tokens=[], street_tokens=[], locality_tokens=[], house=None,
        fields={"line1": None, "city": None, "state": None,
                "postal_code": None, "country": None, "external_id": None},
        raw={},
    )
    conn.commit()
    data = client.get("/api/address-audit").json()
    risk = _risk_map(data)[lid]
    assert risk["kind"] == "self_low"
    assert risk["own_rank"] > 1 or risk["own_score"] < risk["preferred_score"]


# ----------------------------------------------------- no top-5 truncation


def test_blockers_are_not_truncated_at_five(client):
    from app.main import get_conn

    conn = get_conn()
    ids = [
        _insert_duplicate_row(conn, "777 Septuplet Court, Crowton", f"AUD-S{i}")
        for i in range(7)
    ]
    data = client.get("/api/address-audit").json()
    risks = _risk_map(data)
    # The lowest-id row prefers itself but still has six tied siblings;
    # all six must be reported despite the regular matcher's top-5 cut-off.
    first = risks[ids[0]]
    assert first["blocker_count"] == 6
    assert len(first["blockers"]) == 6
    assert {b["address"]["id"] for b in first["blockers"]} == set(ids[1:])
    # Ranks come from the full-table scoring, not a five-row slice.
    last = risks[ids[-1]]
    assert len(last["blockers"]) == 6
    assert max(b["rank"] for b in last["blockers"]) >= 6


# ---------------------------------------------------------------- read-only


def test_audit_does_not_write_match_logs(client):
    from app.main import get_conn

    conn = get_conn()
    before = client.get("/api/matches?limit=500").json()["total"]
    for _ in range(2):
        r = client.get("/api/address-audit")
        assert r.status_code == 200
    after = client.get("/api/matches?limit=500").json()["total"]
    assert after == before
    assert conn.execute("SELECT COUNT(*) FROM match_logs").fetchone()[0] == before


def test_refresh_after_upload_recomputes(client):
    # A unique new address passes the audit; uploading it cannot create or
    # remove unrelated rows, and the totals reflect the new table size.
    r = client.get("/api/address-audit").json()
    total_before = r["total_addresses"]
    risks_before = r["total_risks"]

    csv_bytes = b"id,address,city,zip\nAUD-UNIQ,4242 Zzunique Promenade,Ztown,55512\n"
    up = client.post("/api/addresses/upload",
                     files={"file": ("a.csv", csv_bytes, "text/csv")})
    assert up.status_code == 200 and up.json()["inserted"] == 1

    data = client.get("/api/address-audit").json()
    assert data["total_addresses"] == total_before + 1
    assert not any("Zzunique" in r["full_text"] for r in data["risks"])
    assert data["total_risks"] == risks_before


# ----------------------------------------------------------------- filters


def test_kind_filter(client):
    data = client.get("/api/address-audit").json()
    for kind in ("self_low", "top_shift", "margin_ambiguous"):
        filtered = client.get("/api/address-audit", params={"kind": kind}).json()
        assert {r["kind"] for r in filtered["risks"]} <= {kind}
        assert len(filtered["risks"]) == data[kind]
        # Summary counts are independent of the filter.
        assert filtered["total_risks"] == data["total_risks"]


def test_invalid_kind_filter(client):
    r = client.get("/api/address-audit", params={"kind": "bogus"})
    assert r.status_code == 400


# ----------------------------------------------- isolated database, no API


def test_empty_table_yields_empty_audit():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(db_path)
    old = os.environ.get("ADDRESS_DB_PATH")
    os.environ["ADDRESS_DB_PATH"] = db_path
    conn = None
    try:
        conn = db.connect()
        db.init_db(conn)
        out = address_audit.compute_audit(conn, get_settings())
        assert out["risks"] == []
        assert out["total_addresses"] == 0
        assert out["total_risks"] == 0
        assert out["self_low"] == out["top_shift"] == out["margin_ambiguous"] == 0

        # Filling with just the sample rows gives a clean audit as well.
        sample = Path(get_settings().sample_csv)
        ingest.add_many(conn, ingest.parse_csv(sample.read_bytes()))
        out2 = address_audit.compute_audit(conn, get_settings())
        assert out2["total_addresses"] == 15
        assert out2["total_risks"] == 0
    finally:
        if conn is not None:
            conn.close()
        if old is None:
            os.environ.pop("ADDRESS_DB_PATH", None)
        else:
            os.environ["ADDRESS_DB_PATH"] = old
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(db_path + suffix)
            except OSError:
                pass
