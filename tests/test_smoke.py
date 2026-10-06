"""FieldDay smoke tests: scoring bounds, safe log paths, batch shape, fallbacks."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import explain, ExplainReq, grass_score, health
from tabpfn_predict import flare_risks, log_path


def test_health():
    assert health() == {"ok": True, "service": "fieldday"}


def test_grass_score_bounds():
    for t in (-10, 0, 21, 35, 45):
        for code in (0, 55, 65, 99):
            s = grass_score(t, 50, 10, 5, code)
            assert 0 <= s <= 100, (t, code, s)
    assert grass_score(21, 0, 5, 2, 0) > grass_score(35, 90, 30, 9, 99)


def test_log_path_rejects_traversal():
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    assert log_path("../../etc/x") == os.path.join(base, "logs.csv")
    assert log_path("") == os.path.join(base, "logs.csv")
    assert log_path(None) == os.path.join(base, "logs.csv")
    p = log_path("abc-123-XYZ")
    assert p.endswith("logs_abc-123-XYZ.csv")
    assert os.path.dirname(os.path.abspath(p)) == os.path.abspath(base)


def test_flare_risks_batch_shape():
    samples = [(9, 24.0, 2, 8.0), (18, 30.0, 4, 15.0)]
    out = flare_risks(samples, uid="smoke-test-uid")
    assert len(out) == 2
    for r in out:
        assert set(r) == {"risk", "source", "n_rows"}
        assert 0.0 <= r["risk"] <= 1.0
        assert r["n_rows"] >= 10
    # With a token configured this must be genuine inference, never silent heuristic.
    if os.environ.get("TABPFN_TOKEN"):
        assert {r["source"] for r in out} == {"tabpfn"}


def test_explain_template_fallback_without_backends():
    # No Ollama running here, no hosted keys -> honest template label, never a crash.
    out = explain(ExplainReq(city="Hanoi", best={"time": "2026-10-07T07:00", "combined": 97,
                                                 "flare_risk": 0.05}, risk_source="tabpfn"))
    assert out["source"] == "template-fallback"
    assert "97" in out["text"]
