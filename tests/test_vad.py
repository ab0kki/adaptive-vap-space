import json
from adaptive_vap_space.vad import read_vad_segments, vad_segments_to_frames, frame_segments


def test_read_vad_segments(tmp_path):
    p = tmp_path / "vad.jsonl"
    p.write_text(json.dumps({"start": 0.1, "end": 0.3}) + "\nbad\n")
    segs, bad = read_vad_segments(p)
    assert segs == [(0.1, 0.3)]
    assert bad == 1


def test_vad_segments_to_frames():
    va = vad_segments_to_frames([(0.0, 0.1)], [(0.2, 0.3)], n_frames=20, frame_hz=10)
    assert va[:1, 0].any()
    assert va[2:3, 1].any()


def test_frame_segments():
    assert frame_segments([False, True, True, False, True]) == [(1, 3), (4, 5)]
