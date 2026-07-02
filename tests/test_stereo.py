import numpy as np
import soundfile as sf
from adaptive_vap_space.audio import write_stereo_wav, audio_info


def test_write_stereo_wav(tmp_path):
    sr = 16000
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    sf.write(a, np.zeros(sr), sr)
    sf.write(b, np.ones(sr), sr)
    out = tmp_path / "stereo.wav"
    info = write_stereo_wav(a, b, out, sample_rate=sr)
    assert out.exists()
    assert info["channels"] == 2
    assert audio_info(out)["channels"] == 2
