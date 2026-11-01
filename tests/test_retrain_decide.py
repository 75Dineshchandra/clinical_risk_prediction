"""The retrain rule (retrain.py). Pure logic: no MLflow, no data files, no models."""
import json
import sys

import pytest

import retrain


def make_perf(version=2, clean_weeks=6, pooled=(0.234, 0.215, 0.258), ref=0.2224, scenario_weeks=0):
    """A performance_summary.json shaped like the real one (only the fields the rule reads)."""
    def week(n, scen):
        return {"week": n, "versions": [version], "scenarios": scen, "reading": "healthy",
                "drift": {"major": 0, "moderate": 0, "score_psi": 0.0, "top": "-"}}
    weeks = [week(i + 1, []) for i in range(clean_weeks)]
    weeks += [week(100 + i, ["new_categories"]) for i in range(scenario_weeks)]
    pool = {"n": 12000, "pr_auc": pooled[0], "pr_lo": pooled[1], "pr_hi": pooled[2]}
    return {"weeks": weeks, "by_version": [{"version": version, "family": "lightgbm",
                                            "reference_pr_auc": ref, "clean": pool, "all": pool}]}


def test_keep_when_pooled_range_is_not_below_reference():
    out = retrain.decide(make_perf(), 2, "clean", "r")
    assert out["decision"] == "keep"
    assert out["evidence"]["below_reference"] is False


def test_keep_when_upper_bound_just_reaches_reference():
    # the rule needs the OPTIMISTIC end under the reference; touching it is not enough
    out = retrain.decide(make_perf(pooled=(0.20, 0.18, 0.2224)), 2, "clean", "r")
    assert out["decision"] == "keep"


def test_retrain_when_whole_range_is_below_reference():
    out = retrain.decide(make_perf(pooled=(0.19, 0.17, 0.21)), 2, "clean", "r")
    assert out["decision"] == "retrain"
    assert out["evidence"]["below_reference"] is True


def test_keep_when_too_few_weeks_even_if_below():
    out = retrain.decide(make_perf(clean_weeks=3, pooled=(0.19, 0.17, 0.21)), 2, "clean", "r")
    assert out["decision"] == "keep"
    assert "at least 4" in out["reason"]


def test_keep_when_no_stays_for_the_live_version():
    out = retrain.decide(make_perf(version=2), 3, "clean", "r")
    assert out["decision"] == "keep"


def test_weeks_of_other_scope_are_not_counted():
    # 3 clean weeks plus 3 injected weeks: the clean scope has too few, the all scope has enough
    perf = make_perf(clean_weeks=3, scenario_weeks=3, pooled=(0.19, 0.17, 0.21))
    assert retrain.decide(perf, 2, "clean", "r")["decision"] == "keep"
    assert retrain.decide(perf, 2, "all", "r")["decision"] == "retrain"


def test_fix_data_first_when_unseen_categories_and_below(tmp_path, monkeypatch):
    monkeypatch.setattr(retrain, "DATA", tmp_path)
    drift = tmp_path / "drift" / "r"
    drift.mkdir(parents=True)
    (drift / "drift_summary.json").write_text(json.dumps({"weeks": [
        {"week": 2, "top": [{"feature": "discharge_disposition_id", "what": "__unseen__ 0.0% -> 5.0%"}]}]}))
    out = retrain.decide(make_perf(pooled=(0.19, 0.17, 0.21)), 2, "clean", "r")
    assert out["decision"] == "fix_data_first"
    assert out["weeks_with_unseen_categories"] == [2]


def test_unseen_categories_alone_do_not_trigger_a_retrain(tmp_path, monkeypatch):
    monkeypatch.setattr(retrain, "DATA", tmp_path)
    drift = tmp_path / "drift" / "r"
    drift.mkdir(parents=True)
    (drift / "drift_summary.json").write_text(json.dumps({"weeks": [
        {"week": 2, "top": [{"feature": "x", "what": "__unseen__ 0.0% -> 5.0%"}]}]}))
    out = retrain.decide(make_perf(), 2, "clean", "r")      # performance is fine
    assert out["decision"] == "keep"
    assert "fixing the data" in out["reason"]


def test_missing_reference_falls_back_to_the_gate_result(tmp_path, monkeypatch):
    (tmp_path / "evaluation.json").write_text(json.dumps({"results": {"lightgbm": {"test_pr_auc": 0.2224}}}))
    monkeypatch.setattr(retrain, "MODELS", tmp_path)
    perf = make_perf()
    perf["by_version"][0]["reference_pr_auc"] = None          # what a failed lookup in the performance report leaves behind
    out = retrain.decide(perf, 2, "clean", "r")
    assert out["decision"] == "keep"
    assert out["evidence"]["reference_pr_auc"] == 0.2224
    assert "evaluation.json" in out["evidence"]["reference_source"]


def test_unknown_family_in_the_summary_uses_the_champions_family(tmp_path, monkeypatch):
    (tmp_path / "evaluation.json").write_text(json.dumps({"results": {"lightgbm": {"test_pr_auc": 0.2224}}}))
    monkeypatch.setattr(retrain, "MODELS", tmp_path)
    perf = make_perf()
    perf["by_version"][0]["reference_pr_auc"] = None
    perf["by_version"][0]["family"] = "?"                     # what the performance report wrote for an unmapped version
    out = retrain.decide(perf, 2, "clean", "r", family="lightgbm")
    assert out["model_family"] == "lightgbm"
    assert out["evidence"]["reference_pr_auc"] == 0.2224
    assert out["decision"] == "keep"


def test_no_reference_anywhere_keeps_the_model_with_a_clear_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(retrain, "MODELS", tmp_path)          # no evaluation.json here
    perf = make_perf()
    perf["by_version"][0]["reference_pr_auc"] = None
    out = retrain.decide(perf, 2, "clean", "r")
    assert out["decision"] == "keep"
    assert "No reference PR-AUC" in out["reason"]


def test_force_retrain_is_recorded_as_forced(tmp_path, monkeypatch, capsys):
    perf_file = tmp_path / "performance_summary.json"
    perf_file.write_text(json.dumps(make_perf()))
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "champion.json").write_text(json.dumps({"version": 2}))
    monkeypatch.setattr(retrain, "DATA", tmp_path / "data")
    monkeypatch.setattr(retrain, "MODELS", tmp_path / "models")
    monkeypatch.setattr(sys, "argv", ["retrain.py", "decide", "--run-id", "r", "--performance", str(perf_file),
                                      "--id", "t", "--force-retrain"])
    retrain.main()
    trigger = json.loads((tmp_path / "data" / "retrain" / "t" / "trigger.json").read_text())
    assert trigger["decision"] == "retrain"
    assert trigger["forced"] is True
    assert trigger["evidence_decision"] == "keep"          # what the evidence actually said is never lost
    assert "DECISION: RETRAIN  (forced)" in capsys.readouterr().out