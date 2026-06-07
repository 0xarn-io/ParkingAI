from parkingai.engine import ZoneState
from parkingai.zones import Zone


def square(x, y, s=10):
    return [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]


def test_smoothing_requires_consecutive_frames():
    state = ZoneState(Zone(id="a", points=square(0, 0)))
    # one frame above threshold is not enough with smoothing=3
    state.update(0.9, threshold=0.15, smoothing=3)
    assert state.occupied is False
    state.update(0.9, threshold=0.15, smoothing=3)
    assert state.occupied is False
    state.update(0.9, threshold=0.15, smoothing=3)
    assert state.occupied is True


def test_flicker_does_not_flip_state():
    state = ZoneState(Zone(id="a", points=square(0, 0)))
    # alternating readings should never accumulate enough to flip
    for cov in [0.9, 0.0, 0.9, 0.0, 0.9, 0.0]:
        state.update(cov, threshold=0.15, smoothing=3)
    assert state.occupied is False


def test_state_clears_after_sustained_vacancy():
    state = ZoneState(Zone(id="a", points=square(0, 0)))
    for _ in range(3):
        state.update(0.9, threshold=0.15, smoothing=3)
    assert state.occupied is True
    for _ in range(3):
        state.update(0.0, threshold=0.15, smoothing=3)
    assert state.occupied is False
