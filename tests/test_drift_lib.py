"""PSI functions shared by the drift report and the live exporter (monitoring/drift_lib.py)."""
import numpy as np
import pandas as pd

import drift_lib as d


def test_identical_data_has_zero_psi():
    base = pd.Series(np.arange(1000, dtype=float))
    prof = d.make_profile(base)
    assert d.psi(d.shares(prof, base), d.shares(prof, base)) < 1e-9


def test_shifted_numeric_feature_is_major_drift():
    base = pd.Series(np.arange(1000, dtype=float))
    prof = d.make_profile(base)
    value = d.psi(d.shares(prof, base), d.shares(prof, base + 400))
    assert d.level(value) == "major"


def test_never_seen_category_lands_in_the_unseen_bucket():
    base = pd.Series(["a"] * 500 + ["b"] * 500)
    prof = d.make_profile(base)
    new = pd.Series(["a"] * 500 + ["zzz"] * 500)
    shares = d.shares(prof, new)
    assert shares["__unseen__"] == 0.5
    assert d.level(d.psi(d.shares(prof, base), shares)) == "major"


def test_levels_follow_the_thresholds():
    assert d.level(0.05) == "stable"
    assert d.level(0.10) == "moderate"
    assert d.level(0.25) == "major"


def test_profile_survives_a_json_round_trip():
    base = pd.Series(["a"] * 600 + ["b"] * 400)
    prof = d.profile_with_shares(base)
    again = d.profile_from_json(d.profile_to_json(prof))
    new = pd.Series(["a"] * 300 + ["b"] * 700)
    assert abs(d.psi(prof["shares"], d.shares(prof, new)) - d.psi(again["shares"], d.shares(again, new))) < 1e-12
