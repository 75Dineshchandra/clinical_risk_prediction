import pytest

from data_cleaning import icd9_group


@pytest.mark.parametrize("code,group", [
    ("250", "diabetes"), ("250.83", "diabetes"),
    ("428", "circulatory"), ("401", "circulatory"), ("785", "circulatory"),
    ("486", "respiratory"), ("786", "respiratory"),
    ("560", "digestive"), ("787", "digestive"),
    ("599", "genitourinary"), ("788", "genitourinary"),
    ("174", "neoplasms"),
    ("715", "musculoskeletal"),
    ("820", "injury"),
    ("38", "infectious"), ("8", "infectious"),     # leading zeros are missing in the source data
    ("V57", "supplementary"), ("E888", "external"),
    ("missing", "missing"), ("", "missing"),
    ("not-a-code", "other"), ("?", "other"),
])
def test_icd9_group(code, group):
    assert icd9_group(code) == group
