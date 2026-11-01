"""
Shared drift (PSI) functions, used by drift_report.py (the weekly report), drift_exporter.py (live Prometheus metrics)
and serving/export_champion.py (which saves the reference profile next to the model).

Needs only pandas and numpy, so the small drift-exporter container stays small.

The idea, in one paragraph
    A PROFILE describes one feature as the reference data saw it: the bins (deciles) for a numeric feature, or the
    category list for a categorical one, plus the share of reference stays in each bin or category. Later data is put
    into the SAME bins and its shares are compared with the reference shares:
        PSI = sum over bins of (a - e) * ln(a / e)       a = share in the new data, e = share in the reference
    Rule of thumb: below 0.10 stable, 0.10 to 0.25 moderate, above 0.25 major.
    For categories two extra buckets exist: "__rare__" (seen in the reference but under 0.5%) and "__unseen__"
    (never seen in the reference), so a brand-new code shows up on its own.
"""

import numpy as np
import pandas as pd

EPS = 1e-4              # floor for empty bins, so ln() is defined
RARE_SHARE = 0.005      # reference categories under this share are pooled into "__rare__"
MAX_DISCRETE = 12       # numeric features with this many values or fewer are compared value by value
N_BINS = 10
MODERATE, MAJOR = 0.10, 0.25
SCORE = "__score__"


def _label(v) -> str:
    """One value as a category label. 3, 3.0 and "3" all become "3"; other text stays as it is."""
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f.is_integer() else str(f)


def _labels(series: pd.Series, numeric: bool) -> pd.Series:
    s = series.dropna()
    return s.map(_label) if numeric else s.astype(str)


def make_profile(series: pd.Series) -> dict:
    """Bins or categories taken from the reference data only (no shares yet)."""
    s = series.dropna()
    numeric = bool(pd.api.types.is_numeric_dtype(series))
    if numeric and s.nunique() > MAX_DISCRETE:
        edges = np.unique(np.quantile(s.astype(float), np.linspace(0, 1, N_BINS + 1)[1:-1]))
        return {"kind": "bins", "edges": edges, "numeric": True}
    labels = _labels(series, numeric)
    share = labels.value_counts(normalize=True)
    return {"kind": "cat", "keep": set(share[share >= RARE_SHARE].index), "all": set(share.index), "numeric": numeric}


def shares(profile: dict, series: pd.Series) -> pd.Series:
    """Share of stays per bin or category of the profile. Values outside the reference go to __unseen__."""
    if profile["kind"] == "bins":
        values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        idx = np.searchsorted(profile["edges"], values, side="right")
        counts = np.bincount(idx, minlength=len(profile["edges"]) + 1)
        return pd.Series(counts / max(1, counts.sum()), index=[f"bin{i}" for i in range(len(counts))])
    labels = _labels(series, profile["numeric"])
    mapped = np.where(labels.isin(profile["keep"]), labels,
                      np.where(labels.isin(profile["all"]), "__rare__", "__unseen__"))
    out = pd.Series(mapped).value_counts(normalize=True)
    cats = sorted(profile["keep"]) + ["__rare__", "__unseen__"]
    return out.reindex(cats).fillna(0.0)


def psi(expected: pd.Series, actual: pd.Series) -> float:
    e = np.clip(expected.to_numpy(dtype=float), EPS, None)
    a = np.clip(actual.reindex(expected.index).fillna(0).to_numpy(dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def level(v: float) -> str:
    return "major" if v >= MAJOR else ("moderate" if v >= MODERATE else "stable")


def describe(profile: dict, base: pd.Series, week: pd.Series, e: pd.Series, a: pd.Series) -> str:
    """One short sentence saying how the feature changed."""
    if profile["kind"] == "bins" or profile["numeric"]:
        text = f"mean {pd.to_numeric(base, errors='coerce').mean():.2f} -> {pd.to_numeric(week, errors='coerce').mean():.2f}"
        if profile["kind"] == "bins":
            return text
    else:
        text = ""
    diff = (a.reindex(e.index).fillna(0) - e).sort_values(key=lambda x: -x.abs())[:3]
    moves = ", ".join(f"{c} {e[c]:.1%} -> {a.get(c, 0.0):.1%}" for c in diff.index)
    return (text + "; " if text else "") + moves


# ---------------------------------------------------------------- saved reference profiles (JSON)
def profile_with_shares(series: pd.Series) -> dict:
    """A profile plus the reference shares, ready to be saved."""
    prof = make_profile(series)
    prof["shares"] = shares(prof, series)
    return prof


def profile_to_json(prof: dict) -> dict:
    out = {"kind": prof["kind"], "numeric": bool(prof["numeric"]),
           "shares": {str(k): float(v) for k, v in prof["shares"].items()}}
    if prof["kind"] == "bins":
        out["edges"] = [float(x) for x in prof["edges"]]
    else:
        out["keep"] = sorted(prof["keep"])
        out["all"] = sorted(prof["all"])
    return out


def profile_from_json(obj: dict) -> dict:
    prof = {"kind": obj["kind"], "numeric": bool(obj["numeric"]),
            "shares": pd.Series(obj["shares"], dtype=float)}
    if obj["kind"] == "bins":
        prof["edges"] = np.array(obj["edges"], dtype=float)
    else:
        prof["keep"] = set(obj["keep"])
        prof["all"] = set(obj["all"])
    return prof


def build_reference(features: pd.DataFrame, scores, meta: dict) -> dict:
    """The reference profile that serving/export_champion.py saves next to the model."""
    return {"format": 1, **meta, "reference_rows": int(len(features)),
            "features": {c: profile_to_json(profile_with_shares(features[c])) for c in features.columns},
            "score": profile_to_json(profile_with_shares(pd.Series(np.asarray(scores, dtype=float))))}
