"""OpenF1 loader edge cases (no network)."""
from __future__ import annotations

import pandas as pd

from src.pitwall import openf1_data


class _Resp:
    def __init__(self, status: int, data):
        self.status_code, self._data = status, data

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


def test_empty_answers_are_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(openf1_data, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(openf1_data.time, "sleep", lambda *_: None)
    answers = iter([_Resp(404, None), _Resp(200, [{"lap_number": 1}])])
    monkeypatch.setattr(openf1_data.requests, "get", lambda *a, **k: next(answers))
    # before the session is published: empty, and nothing written to disk
    assert openf1_data._get("laps", session_key=1) == []
    assert not list(tmp_path.iterdir())
    # once published: the real data comes through and is cached
    assert openf1_data._get("laps", session_key=1) == [{"lap_number": 1}]
    assert len(list(tmp_path.iterdir())) == 1


def test_text_positions_are_treated_as_unclassified():
    sess = openf1_data.OpenF1Session.__new__(openf1_data.OpenF1Session)
    sess.name = "Qualifying"
    res = pd.DataFrame({
        "driver_number": [1, 2, 3],
        "position": [1, "RT", 2],  # OpenF1 status code in the position field
        "number_of_laps": [12, 3, 12],
        "dnf": [False, False, False], "dns": [False, False, False], "dsq": [False, False, False],
    })
    drivers = pd.DataFrame({
        "name_acronym": ["AAA", "BBB", "CCC"], "full_name": ["A", "B", "C"], "team_name": ["T1", "T2", "T3"],
        "team_colour": ["111111", "222222", "333333"], "headshot_url": [None, None, None],
    }, index=[1, 2, 3])
    out = sess._results(res, drivers, pd.DataFrame())
    pos = dict(zip(out["DriverNumber"].astype(int), out["Position"]))
    assert pos[1] == 1 and pos[3] == 2 and pos[2] == 3


def test_driver_names_use_proper_case():
    sess = openf1_data.OpenF1Session.__new__(openf1_data.OpenF1Session)
    sess.name = "Qualifying"
    res = pd.DataFrame({"driver_number": [1], "position": [1], "number_of_laps": [12], "dnf": [False], "dns": [False], "dsq": [False]})
    drivers = pd.DataFrame({
        "name_acronym": ["VER"], "full_name": ["Max VERSTAPPEN"], "first_name": ["Max"], "last_name": ["Verstappen"],
        "team_name": ["Red Bull Racing"], "team_colour": ["4781D7"], "headshot_url": [None],
    }, index=[1])
    out = sess._results(res, drivers, pd.DataFrame())
    assert out.loc[0, "FullName"] == "Max Verstappen"
