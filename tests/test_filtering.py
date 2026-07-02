from adaptive_vap_space.filtering import validate_interaction, SpeakerValidation


def sp(duration=200, vad_ratio=0.2):
    return SpeakerValidation(True, [], duration, 16000, 1, 10, duration*vad_ratio, vad_ratio, 0, 0)


def test_validate_interaction_balanced():
    cfg = {"filters": {"min_duration_sec": 10, "max_audio_duration_diff_sec": 2, "min_speaker_fraction": 0.2, "max_speaker_fraction": 0.8, "min_total_speech_ratio": 0.01, "max_overlap_ratio": 0.9, "max_silence_ratio": 0.99}}
    ok, reasons, stats = validate_interaction([(0, 10)], [(20, 30)], sp(), sp(), cfg)
    assert ok
    assert reasons == []
    assert stats["duration_sec"] == 200
