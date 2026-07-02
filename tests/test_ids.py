from adaptive_vap_space.ids import parse_file_id, interaction_key_from_file_id


def test_parse_numeric_participant():
    p = parse_file_id("V00_S0554_I00000462_P0844")
    assert p.interaction_key == "V00_S0554_I00000462"
    assert p.participant_id == "P0844"


def test_parse_alphanumeric_participant():
    p = parse_file_id("V00_S0554_I00000462_P0844A")
    assert p.interaction_key == "V00_S0554_I00000462"
    assert p.participant_id == "P0844A"


def test_interaction_key():
    assert interaction_key_from_file_id("V01_S0108_I00000128_P0001") == "V01_S0108_I00000128"
