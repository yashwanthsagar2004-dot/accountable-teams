from accteams.provenance import stamp


def test_stamp_has_what_a_reader_needs():
    a, b = stamp({"window": 20, "seed": 0}), stamp({"seed": 0, "window": 20})
    assert a["config_hash"] == b["config_hash"] != stamp({"window": 10, "seed": 0})["config_hash"]
    assert {"commit", "command", "device", "versions", "seconds", "preregistered"} <= set(a)
