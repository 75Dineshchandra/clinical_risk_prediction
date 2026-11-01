"""Input validation and schema reading in serving/app.py. Uses a tiny fitted pipeline, not the real model."""
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import HTTPException
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import app


def state():
    """The few fields build_frame reads from the model state."""
    return SimpleNamespace(
        required=["days", "disp"], numeric=["days"], categorical={"disp": "str"},
        categories={"disp": {"1", "2"}})


def reason_of(excinfo):
    return excinfo.value.detail["error"]


def test_valid_records_are_converted():
    df, ignored, unseen, ids = app.build_frame(
        [{"encounter_id": "A1", "days": "3", "disp": 1.0, "extra": "x"}], state())
    assert df.loc[0, "days"] == 3 and df["days"].dtype.kind in "if"
    assert df.loc[0, "disp"] == "1"            # 1, 1.0 and "1" mean the same category
    assert ids == ["A1"] and ignored == ["extra"] and unseen == {}


def test_unseen_category_is_reported_not_rejected():
    _, _, unseen, _ = app.build_frame([{"days": 2, "disp": "99"}], state())
    assert unseen == {"disp": ["99"]}


def test_missing_column_is_rejected():
    with pytest.raises(HTTPException) as e:
        app.build_frame([{"days": 2}], state())
    assert e.value.status_code == 422 and reason_of(e) == "missing_columns"


def test_null_is_rejected():
    with pytest.raises(HTTPException) as e:
        app.build_frame([{"days": None, "disp": "1"}], state())
    assert e.value.status_code == 422 and reason_of(e) == "nulls"


def test_text_in_a_numeric_column_is_rejected():
    with pytest.raises(HTTPException) as e:
        app.build_frame([{"days": "abc", "disp": "1"}], state())
    assert e.value.status_code == 422 and reason_of(e) == "not_numeric"


def test_inspect_pipeline_reads_the_contract_from_the_fitted_pipeline():
    train = pd.DataFrame({"days": [1, 2, 3, 4, 5, 6], "disp": ["1", "2", "1", "2", "1", "2"]})
    pipe = Pipeline([
        ("prep", ColumnTransformer([("num", StandardScaler(), ["days"]),
                                    ("cat", OneHotEncoder(handle_unknown="ignore"), ["disp"])])),
        ("model", LogisticRegression())]).fit(train, [0, 1, 0, 1, 1, 0])
    schema = app.inspect_pipeline(pipe)
    assert schema["required"] == ["days", "disp"]
    assert schema["numeric"] == ["days"]
    assert schema["categorical"] == {"disp": "str"}
    assert schema["categories"] == {"disp": {"1", "2"}}
    assert schema["feature_to_column"]["disp_1"] == "disp"      # one-hot name maps back to its source column
    assert schema["feature_to_column"]["days"] == "days"
