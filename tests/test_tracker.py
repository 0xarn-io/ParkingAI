from parkingai.tracker import Tracker, iou


def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (0, 0, 10, 5)) - 0.5) < 1e-6


def test_stable_id_for_stationary_car():
    tr = Tracker(iou_threshold=0.3, max_misses=3)
    a = tr.update([(0, 0, 10, 10, 0.9)])
    b = tr.update([(1, 1, 11, 11, 0.9)])      # barely moved -> same id
    assert a[0].id == b[0].id


def test_new_id_for_new_car():
    tr = Tracker()
    first = tr.update([(0, 0, 10, 10, 0.9)])
    both = tr.update([(0, 0, 10, 10, 0.9), (100, 100, 120, 120, 0.9)])
    ids = {t.id for t in both}
    assert first[0].id in ids and len(ids) == 2


def test_track_drops_after_max_misses():
    tr = Tracker(max_misses=2)
    tr.update([(0, 0, 10, 10, 0.9)])
    tr.update([])           # miss 1
    tr.update([])           # miss 2
    visible = tr.update([])  # miss 3 -> dropped
    assert visible == []
