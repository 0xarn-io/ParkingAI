from parkingai.zones import Zone, zone_coverage, load_zones, save_zones


def square(x, y, s=10):
    return [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]


def test_zone_area():
    z = Zone(id="a", points=square(0, 0, 10))
    assert z.area == 100.0


def test_full_coverage():
    z = Zone(id="a", points=square(0, 0, 10))
    # a box that fully contains the zone
    assert zone_coverage(z, [(-5, -5, 15, 15, 0.9)]) == 1.0


def test_partial_coverage():
    z = Zone(id="a", points=square(0, 0, 10))
    # box covers the left half
    cov = zone_coverage(z, [(0, 0, 5, 10, 0.9)])
    assert abs(cov - 0.5) < 1e-6


def test_no_overlap():
    z = Zone(id="a", points=square(0, 0, 10))
    assert zone_coverage(z, [(100, 100, 120, 120, 0.9)]) == 0.0


def test_best_of_multiple_boxes():
    z = Zone(id="a", points=square(0, 0, 10))
    cov = zone_coverage(z, [(0, 0, 2, 10, 0.9), (0, 0, 8, 10, 0.9)])
    assert abs(cov - 0.8) < 1e-6


def test_save_and_load_roundtrip(tmp_path):
    zones = [Zone(id="1", points=square(0, 0)), Zone(id="2", points=square(20, 20))]
    path = tmp_path / "zones.json"
    save_zones(path, zones)
    loaded = load_zones(path)
    assert [z.id for z in loaded] == ["1", "2"]
    assert loaded[0].points == square(0, 0)


def test_load_missing_file(tmp_path):
    assert load_zones(tmp_path / "nope.json") == []
