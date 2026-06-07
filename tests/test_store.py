import numpy as np

from parkingai.store import Store


def emb(seed):
    rng = np.random.default_rng(seed)
    v = rng.random(256).astype(np.float32)
    return v / np.linalg.norm(v)


def test_create_and_match_returning_vehicle(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    e = emb(1)
    vid, name, returning = s.match_or_create(e, "Red", reid_threshold=0.4)
    assert returning is False and name.startswith("Red")
    # same fingerprint again -> matched as returning, same id
    vid2, name2, returning2 = s.match_or_create(e, "Red", reid_threshold=0.4)
    assert vid2 == vid and returning2 is True


def test_distinct_vehicles_get_distinct_ids(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    v1, _, _ = s.match_or_create(emb(1), "Red", reid_threshold=0.01)
    v2, _, _ = s.match_or_create(emb(99), "Blue", reid_threshold=0.01)
    assert v1 != v2
    assert s.summary()["vehicles_known"] == 2


def test_session_lifecycle_updates_stats(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    vid, _, _ = s.match_or_create(emb(1), "Red", reid_threshold=0.4, now=1000.0)
    sid = s.open_session("1", vid, now=1000.0)
    dur = s.close_session(sid, now=1060.0, min_seconds=5.0)
    assert dur == 60.0
    veh = {v["id"]: v for v in s.list_vehicles()}[vid]
    assert veh["visits"] == 1 and veh["total_parked"] == 60.0
    summ = s.summary()
    assert summ["completed_sessions"] == 1 and summ["avg_park_seconds"] == 60.0


def test_short_session_is_discarded(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    sid = s.open_session("1", None, now=1000.0)
    assert s.close_session(sid, now=1002.0, min_seconds=5.0) is None
    assert s.summary()["completed_sessions"] == 0


def test_rename(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    vid, _, _ = s.match_or_create(emb(1), "Red", reid_threshold=0.4)
    assert s.rename(vid, "My Car") is True
    assert {v["id"]: v["name"] for v in s.list_vehicles()}[vid] == "My Car"
