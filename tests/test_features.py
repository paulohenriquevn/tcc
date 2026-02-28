"""Tests for feature extraction modules.

Tests cover:
- acoustic.py: extract_acoustic_features(), features_to_vector()
- ecapa.py: compute_cosine_similarity(), compute_speaker_similarity_baseline()
- ssl.py: extract_ssl_features() (model mocked)
- backbone.py: extract_backbone_features() (model mocked)

Categories:
- Shape tests: verify output dimensions match documented contracts.
- Determinism tests: same input produces identical output.
- Edge cases: zero vectors, single-speaker baselines, out-of-range layers.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch

from src.features.acoustic import (
    AcousticFeatures,
    extract_acoustic_features,
    features_to_vector,
)
from src.features.ecapa import (
    compute_cosine_similarity,
    compute_speaker_similarity_baseline,
)


def _mock_torchaudio_load(path, **kwargs):
    """Fake torchaudio.load that returns a mono 16kHz waveform tensor.

    Reads the WAV via soundfile (which works without torchcodec) and
    returns (waveform_tensor, sample_rate) matching torchaudio's API.
    """
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim == 1:
        waveform = torch.from_numpy(data).unsqueeze(0)  # (1, N)
    else:
        waveform = torch.from_numpy(data.T)  # (channels, N)
    return waveform, sr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(
    tmp_dir: Path,
    filename: str = "test.wav",
    sr: int = 16000,
    duration_s: float = 1.0,
    freq_hz: float = 440.0,
    seed: int = 42,
) -> Path:
    """Create a synthetic mono WAV file with a sine tone + small noise."""
    rng = np.random.RandomState(seed)
    n_samples = int(sr * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * freq_hz * t) + 0.01 * rng.randn(n_samples)
    signal = signal.astype(np.float32)

    wav_path = tmp_dir / filename
    sf.write(str(wav_path), signal, sr)
    return wav_path


def _make_stereo_wav(
    tmp_dir: Path,
    filename: str = "stereo.wav",
    sr: int = 16000,
    duration_s: float = 1.0,
    seed: int = 42,
) -> Path:
    """Create a synthetic stereo WAV file."""
    rng = np.random.RandomState(seed)
    n_samples = int(sr * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)

    ch1 = 0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.01 * rng.randn(n_samples)
    ch2 = 0.5 * np.sin(2 * np.pi * 880.0 * t) + 0.01 * rng.randn(n_samples)
    stereo = np.stack([ch1, ch2], axis=-1).astype(np.float32)

    wav_path = tmp_dir / filename
    sf.write(str(wav_path), stereo, sr)
    return wav_path


# ===========================================================================
# acoustic.py tests
# ===========================================================================


class TestExtractAcousticFeatures:
    """Tests for extract_acoustic_features()."""

    def test_output_type(self, tmp_path):
        """Return type must be AcousticFeatures."""
        wav = _make_wav(tmp_path, duration_s=1.0)
        result = extract_acoustic_features(wav, utt_id="utt_001")
        assert isinstance(result, AcousticFeatures)

    def test_mfcc_shape_default(self, tmp_path):
        """Default n_mfcc=13 produces (13,) mean and std vectors."""
        wav = _make_wav(tmp_path)
        result = extract_acoustic_features(wav, utt_id="utt_001")

        assert result.mfcc_mean.shape == (13,)
        assert result.mfcc_std.shape == (13,)

    def test_mfcc_shape_custom(self, tmp_path):
        """Custom n_mfcc value is respected in output shape."""
        wav = _make_wav(tmp_path)
        result = extract_acoustic_features(wav, utt_id="utt_001", n_mfcc=20)

        assert result.mfcc_mean.shape == (20,)
        assert result.mfcc_std.shape == (20,)

    def test_utt_id_preserved(self, tmp_path):
        """utt_id passed in matches utt_id in output."""
        wav = _make_wav(tmp_path)
        result = extract_acoustic_features(wav, utt_id="my_utterance")
        assert result.utt_id == "my_utterance"

    def test_duration_positive(self, tmp_path):
        """Duration must be a positive float."""
        wav = _make_wav(tmp_path, duration_s=2.0)
        result = extract_acoustic_features(wav, utt_id="utt_001")
        assert result.duration_s > 0
        assert result.duration_s == pytest.approx(2.0, abs=0.05)

    def test_pitch_nonnegative(self, tmp_path):
        """Pitch mean and std must be non-negative."""
        wav = _make_wav(tmp_path, freq_hz=300.0)
        result = extract_acoustic_features(wav, utt_id="utt_001")
        assert result.pitch_mean >= 0.0
        assert result.pitch_std >= 0.0

    def test_energy_nonnegative(self, tmp_path):
        """Energy (RMS) mean and std must be non-negative."""
        wav = _make_wav(tmp_path)
        result = extract_acoustic_features(wav, utt_id="utt_001")
        assert result.energy_mean >= 0.0
        assert result.energy_std >= 0.0

    def test_speech_rate_nonnegative(self, tmp_path):
        """Speech rate must be non-negative."""
        wav = _make_wav(tmp_path)
        result = extract_acoustic_features(wav, utt_id="utt_001")
        assert result.speech_rate >= 0.0

    def test_determinism(self, tmp_path):
        """Same WAV file extracted twice yields identical features."""
        wav = _make_wav(tmp_path, seed=99)
        r1 = extract_acoustic_features(wav, utt_id="utt_a")
        r2 = extract_acoustic_features(wav, utt_id="utt_a")

        np.testing.assert_array_equal(r1.mfcc_mean, r2.mfcc_mean)
        np.testing.assert_array_equal(r1.mfcc_std, r2.mfcc_std)
        assert r1.pitch_mean == r2.pitch_mean
        assert r1.pitch_std == r2.pitch_std
        assert r1.energy_mean == r2.energy_mean
        assert r1.energy_std == r2.energy_std
        assert r1.speech_rate == r2.speech_rate

    def test_different_frequencies_yield_different_pitch(self, tmp_path):
        """Two tones at different frequencies must have different pitch_mean."""
        wav_low = _make_wav(tmp_path, filename="low.wav", freq_hz=150.0, seed=1)
        wav_high = _make_wav(tmp_path, filename="high.wav", freq_hz=500.0, seed=2)

        r_low = extract_acoustic_features(wav_low, utt_id="low")
        r_high = extract_acoustic_features(wav_high, utt_id="high")

        # Both should detect pitch; they should differ meaningfully
        if r_low.pitch_mean > 0 and r_high.pitch_mean > 0:
            assert r_low.pitch_mean != r_high.pitch_mean

    def test_resampling_different_sr(self, tmp_path):
        """Audio at non-16kHz is resampled without error."""
        wav = _make_wav(tmp_path, sr=22050, filename="sr22k.wav")
        result = extract_acoustic_features(wav, utt_id="utt_resample", sr=16000)
        assert isinstance(result, AcousticFeatures)
        assert result.mfcc_mean.shape == (13,)


class TestFeaturesToVector:
    """Tests for features_to_vector()."""

    def test_default_vector_length(self, tmp_path):
        """Default n_mfcc=13 produces vector of length 31.

        Layout: mfcc_mean(13) + mfcc_std(13) + pitch(2) + energy(2) + speech_rate(1) = 31
        """
        wav = _make_wav(tmp_path)
        features = extract_acoustic_features(wav, utt_id="utt_001")
        vec = features_to_vector(features)

        assert vec.ndim == 1
        assert vec.shape == (31,)

    def test_custom_mfcc_vector_length(self, tmp_path):
        """Custom n_mfcc=20 produces vector of length 45.

        Layout: mfcc_mean(20) + mfcc_std(20) + pitch(2) + energy(2) + speech_rate(1) = 45
        """
        wav = _make_wav(tmp_path)
        features = extract_acoustic_features(wav, utt_id="utt_001", n_mfcc=20)
        vec = features_to_vector(features)

        expected_len = 20 * 2 + 2 + 2 + 1  # 45
        assert vec.shape == (expected_len,)

    def test_vector_contains_correct_values(self):
        """Vector segments correspond to the correct feature fields."""
        features = AcousticFeatures(
            utt_id="manual",
            mfcc_mean=np.array([1.0, 2.0, 3.0]),
            mfcc_std=np.array([0.1, 0.2, 0.3]),
            pitch_mean=100.0,
            pitch_std=10.0,
            energy_mean=0.5,
            energy_std=0.05,
            speech_rate=3.0,
            duration_s=2.0,
        )
        vec = features_to_vector(features)

        expected = np.array([
            1.0, 2.0, 3.0,       # mfcc_mean
            0.1, 0.2, 0.3,       # mfcc_std
            100.0, 10.0,         # pitch_mean, pitch_std
            0.5, 0.05,           # energy_mean, energy_std
            3.0,                 # speech_rate
        ])
        np.testing.assert_array_almost_equal(vec, expected)

    def test_vector_dtype_is_float(self, tmp_path):
        """Output vector must be floating-point."""
        wav = _make_wav(tmp_path)
        features = extract_acoustic_features(wav, utt_id="utt_001")
        vec = features_to_vector(features)
        assert np.issubdtype(vec.dtype, np.floating)

    def test_duration_not_in_vector(self, tmp_path):
        """duration_s is NOT included in the probe vector (31 dims, not 32)."""
        wav = _make_wav(tmp_path)
        features = extract_acoustic_features(wav, utt_id="utt_001")
        vec = features_to_vector(features)
        assert vec.shape == (31,)


# ===========================================================================
# ecapa.py tests (cosine similarity and baseline — no model needed)
# ===========================================================================


class TestComputeCosineSimilarity:
    """Tests for compute_cosine_similarity()."""

    def test_identical_vectors(self):
        """Cosine similarity of a vector with itself is 1.0."""
        emb = np.array([1.0, 2.0, 3.0, 4.0])
        assert compute_cosine_similarity(emb, emb) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        """Cosine similarity of a vector with its negation is -1.0."""
        emb = np.array([1.0, 2.0, 3.0])
        assert compute_cosine_similarity(emb, -emb) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self):
        """Cosine similarity of orthogonal vectors is 0.0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert compute_cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        """Edge case: zero vector should return 0.0, not NaN."""
        zero = np.zeros(5)
        nonzero = np.ones(5)

        assert compute_cosine_similarity(zero, nonzero) == 0.0
        assert compute_cosine_similarity(nonzero, zero) == 0.0
        assert compute_cosine_similarity(zero, zero) == 0.0

    def test_range_is_bounded(self):
        """Result must be in [-1, 1] for arbitrary vectors."""
        rng = np.random.RandomState(42)
        for _ in range(50):
            a = rng.randn(192)
            b = rng.randn(192)
            sim = compute_cosine_similarity(a, b)
            assert -1.0 <= sim <= 1.0

    def test_symmetry(self):
        """Cosine similarity is symmetric: sim(a, b) == sim(b, a)."""
        rng = np.random.RandomState(42)
        a = rng.randn(192)
        b = rng.randn(192)
        assert compute_cosine_similarity(a, b) == pytest.approx(
            compute_cosine_similarity(b, a)
        )

    def test_scale_invariance(self):
        """Cosine similarity is invariant to scaling of either vector."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        sim_original = compute_cosine_similarity(a, b)
        sim_scaled = compute_cosine_similarity(a * 100, b * 0.01)
        assert sim_original == pytest.approx(sim_scaled, abs=1e-7)

    def test_high_dim_192(self):
        """Works correctly with 192-dim vectors (ECAPA embedding size)."""
        rng = np.random.RandomState(42)
        a = rng.randn(192)
        sim = compute_cosine_similarity(a, a)
        assert sim == pytest.approx(1.0, abs=1e-6)


class TestComputeSpeakerSimilarityBaseline:
    """Tests for compute_speaker_similarity_baseline()."""

    def test_basic_output_structure(self):
        """Output dict has 'intra' and 'inter' keys with required subkeys."""
        rng = np.random.RandomState(42)
        speaker_embs = {
            "spk_1": [rng.randn(192) for _ in range(3)],
            "spk_2": [rng.randn(192) for _ in range(3)],
        }

        result = compute_speaker_similarity_baseline(speaker_embs)

        assert "intra" in result
        assert "inter" in result
        for key in ("mean", "std", "n_pairs", "values"):
            assert key in result["intra"], f"Missing 'intra.{key}'"
            assert key in result["inter"], f"Missing 'inter.{key}'"

    def test_intra_higher_than_inter(self):
        """Intra-speaker similarity should be higher than inter-speaker
        when speakers have distinct embeddings."""
        rng = np.random.RandomState(42)
        # Create distinct clusters per speaker
        base_spk1 = rng.randn(192)
        base_spk2 = rng.randn(192) + 5.0  # shift far away

        speaker_embs = {
            "spk_1": [base_spk1 + 0.01 * rng.randn(192) for _ in range(5)],
            "spk_2": [base_spk2 + 0.01 * rng.randn(192) for _ in range(5)],
        }

        result = compute_speaker_similarity_baseline(speaker_embs)
        assert result["intra"]["mean"] > result["inter"]["mean"]

    def test_intra_pair_count(self):
        """Intra pairs = sum of C(n_i, 2) for each speaker."""
        rng = np.random.RandomState(42)
        speaker_embs = {
            "spk_1": [rng.randn(192) for _ in range(4)],  # C(4,2)=6
            "spk_2": [rng.randn(192) for _ in range(3)],  # C(3,2)=3
        }

        result = compute_speaker_similarity_baseline(speaker_embs)
        assert result["intra"]["n_pairs"] == 6 + 3  # 9

    def test_inter_pair_count(self):
        """Inter pairs = number of speaker pairs (1 random emb per speaker)."""
        rng = np.random.RandomState(42)
        speaker_embs = {
            "spk_1": [rng.randn(192) for _ in range(3)],
            "spk_2": [rng.randn(192) for _ in range(3)],
            "spk_3": [rng.randn(192) for _ in range(3)],
        }

        result = compute_speaker_similarity_baseline(speaker_embs)
        # C(3,2) = 3 speaker pairs
        assert result["inter"]["n_pairs"] == 3

    def test_single_speaker_no_inter(self):
        """With only one speaker, inter-speaker produces fallback [0.0]."""
        rng = np.random.RandomState(42)
        speaker_embs = {
            "spk_1": [rng.randn(192) for _ in range(5)],
        }

        result = compute_speaker_similarity_baseline(speaker_embs)
        assert result["inter"]["n_pairs"] == 0
        assert result["inter"]["mean"] == pytest.approx(0.0)

    def test_single_utterance_per_speaker_no_intra(self):
        """Speakers with only 1 utterance produce no intra pairs."""
        rng = np.random.RandomState(42)
        speaker_embs = {
            "spk_1": [rng.randn(192)],
            "spk_2": [rng.randn(192)],
        }

        result = compute_speaker_similarity_baseline(speaker_embs)
        assert result["intra"]["n_pairs"] == 0

    def test_determinism_with_same_seed(self):
        """Same seed produces identical inter-speaker sampling."""
        rng = np.random.RandomState(42)
        speaker_embs = {
            f"spk_{i}": [rng.randn(192) for _ in range(3)]
            for i in range(10)
        }

        r1 = compute_speaker_similarity_baseline(speaker_embs, seed=42)
        r2 = compute_speaker_similarity_baseline(speaker_embs, seed=42)

        assert r1["inter"]["values"] == r2["inter"]["values"]
        assert r1["inter"]["mean"] == r2["inter"]["mean"]

    def test_different_seeds_may_differ(self):
        """Different seeds can produce different inter sampling."""
        rng = np.random.RandomState(42)
        speaker_embs = {
            f"spk_{i}": [rng.randn(192) for _ in range(5)]
            for i in range(10)
        }

        r1 = compute_speaker_similarity_baseline(speaker_embs, seed=42)
        r2 = compute_speaker_similarity_baseline(speaker_embs, seed=99)

        # With many speakers and different seeds, inter values should differ
        # (probabilistically; check that at least one value differs)
        assert r1["inter"]["values"] != r2["inter"]["values"]

    def test_similarity_values_in_range(self):
        """All similarity values must be in [-1, 1]."""
        rng = np.random.RandomState(42)
        speaker_embs = {
            f"spk_{i}": [rng.randn(192) for _ in range(3)]
            for i in range(5)
        }

        result = compute_speaker_similarity_baseline(speaker_embs)

        for val in result["intra"]["values"]:
            assert -1.0 <= val <= 1.0
        for val in result["inter"]["values"]:
            assert -1.0 <= val <= 1.0

    def test_intra_sampling_caps_pairs(self):
        """When a speaker has more pairs than max_intra_pairs_per_speaker,
        the number of intra pairs is capped at the limit."""
        rng = np.random.RandomState(42)
        n_utts = 50  # C(50,2) = 1225 possible pairs
        cap = 100
        speaker_embs = {
            "spk_1": [rng.randn(192) for _ in range(n_utts)],
            "spk_2": [rng.randn(192) for _ in range(3)],  # C(3,2)=3, below cap
        }

        result = compute_speaker_similarity_baseline(
            speaker_embs, max_intra_pairs_per_speaker=cap,
        )
        # spk_1: capped at 100, spk_2: exhaustive 3 → total 103
        assert result["intra"]["n_pairs"] == cap + 3

    def test_intra_sampling_deterministic(self):
        """Sampled intra pairs are deterministic with same seed."""
        rng = np.random.RandomState(42)
        speaker_embs = {
            "spk_1": [rng.randn(192) for _ in range(50)],
        }

        r1 = compute_speaker_similarity_baseline(
            speaker_embs, seed=42, max_intra_pairs_per_speaker=20,
        )
        r2 = compute_speaker_similarity_baseline(
            speaker_embs, seed=42, max_intra_pairs_per_speaker=20,
        )

        assert r1["intra"]["values"] == r2["intra"]["values"]

    def test_intra_no_limit_exhaustive(self):
        """max_intra_pairs_per_speaker=0 disables sampling (exhaustive)."""
        rng = np.random.RandomState(42)
        speaker_embs = {
            "spk_1": [rng.randn(192) for _ in range(10)],  # C(10,2)=45
        }

        result = compute_speaker_similarity_baseline(
            speaker_embs, max_intra_pairs_per_speaker=0,
        )
        assert result["intra"]["n_pairs"] == 45


# ===========================================================================
# ssl.py tests (WavLM model fully mocked)
# ===========================================================================


def _build_mock_ssl_model(hidden_dim: int = 1024, n_layers: int = 25):
    """Build a mock WavLM model that returns fake hidden states."""

    def _mock_forward(input_values, **kwargs):
        batch_size = input_values.shape[0]
        seq_len = input_values.shape[1] // 320  # WavLM downsamples ~320x
        seq_len = max(seq_len, 1)

        # hidden_states: tuple of (n_layers+1) tensors, each (batch, seq_len, hidden_dim)
        hidden_states = tuple(
            torch.randn(batch_size, seq_len, hidden_dim)
            for _ in range(n_layers + 1)
        )
        return SimpleNamespace(hidden_states=hidden_states)

    model = MagicMock()
    model.side_effect = _mock_forward
    model.eval.return_value = model
    model.to.return_value = model
    return model


def _build_mock_ssl_processor(target_sr: int = 16000):
    """Build a mock WavLM processor."""

    def _mock_call(audio, sampling_rate=16000, return_tensors="pt", **kwargs):
        if isinstance(audio, np.ndarray):
            n_samples = len(audio)
        else:
            n_samples = 16000  # fallback
        return SimpleNamespace(
            input_values=torch.randn(1, n_samples)
        )

    processor = MagicMock()
    processor.side_effect = _mock_call
    return processor


class TestExtractSSLFeatures:
    """Tests for extract_ssl_features() with mocked WavLM."""

    @pytest.fixture(autouse=True)
    def _reset_ssl_globals(self):
        """Reset module-level globals before each test."""
        import src.features.ssl as ssl_mod
        ssl_mod._ssl_model = None
        ssl_mod._ssl_processor = None
        ssl_mod._ssl_device = None
        yield
        ssl_mod._ssl_model = None
        ssl_mod._ssl_processor = None
        ssl_mod._ssl_device = None

    @patch("src.features.ssl.torchaudio.load", side_effect=_mock_torchaudio_load)
    @patch("src.features.ssl._get_ssl_model")
    def test_output_keys_match_requested_layers(self, mock_get, _mock_load, tmp_path):
        """Returned dict keys must match the requested layers."""
        hidden_dim = 1024
        mock_get.return_value = (
            _build_mock_ssl_model(hidden_dim=hidden_dim, n_layers=25),
            _build_mock_ssl_processor(),
            "cpu",
        )
        wav = _make_wav(tmp_path)

        from src.features.ssl import extract_ssl_features

        result = extract_ssl_features(wav, layers=[0, 6, 12, 24])

        assert set(result.keys()) == {0, 6, 12, 24}

    @patch("src.features.ssl.torchaudio.load", side_effect=_mock_torchaudio_load)
    @patch("src.features.ssl._get_ssl_model")
    def test_output_shape_per_layer(self, mock_get, _mock_load, tmp_path):
        """Each layer output must be (hidden_dim,) after mean_temporal pooling."""
        hidden_dim = 1024
        mock_get.return_value = (
            _build_mock_ssl_model(hidden_dim=hidden_dim, n_layers=25),
            _build_mock_ssl_processor(),
            "cpu",
        )
        wav = _make_wav(tmp_path)

        from src.features.ssl import extract_ssl_features

        result = extract_ssl_features(wav, layers=[0, 12, 24])

        for layer_idx, vec in result.items():
            assert vec.shape == (hidden_dim,), (
                f"Layer {layer_idx}: expected ({hidden_dim},), got {vec.shape}"
            )

    @patch("src.features.ssl.torchaudio.load", side_effect=_mock_torchaudio_load)
    @patch("src.features.ssl._get_ssl_model")
    def test_output_is_numpy(self, mock_get, _mock_load, tmp_path):
        """Output vectors must be numpy arrays."""
        mock_get.return_value = (
            _build_mock_ssl_model(),
            _build_mock_ssl_processor(),
            "cpu",
        )
        wav = _make_wav(tmp_path)

        from src.features.ssl import extract_ssl_features

        result = extract_ssl_features(wav, layers=[0])

        assert isinstance(result[0], np.ndarray)

    @patch("src.features.ssl.torchaudio.load", side_effect=_mock_torchaudio_load)
    @patch("src.features.ssl._get_ssl_model")
    def test_out_of_range_layer_skipped(self, mock_get, _mock_load, tmp_path):
        """Layer index beyond model capacity is skipped with a warning."""
        n_layers = 10
        mock_get.return_value = (
            _build_mock_ssl_model(n_layers=n_layers),
            _build_mock_ssl_processor(),
            "cpu",
        )
        wav = _make_wav(tmp_path)

        from src.features.ssl import extract_ssl_features

        # Request layer 50, which doesn't exist (model has 11 hidden states: 0..10)
        result = extract_ssl_features(wav, layers=[0, 50])

        assert 0 in result
        assert 50 not in result

    @patch("src.features.ssl.torchaudio.load", side_effect=_mock_torchaudio_load)
    @patch("src.features.ssl._get_ssl_model")
    def test_invalid_pooling_raises(self, mock_get, _mock_load, tmp_path):
        """Unknown pooling strategy must raise ValueError."""
        mock_get.return_value = (
            _build_mock_ssl_model(),
            _build_mock_ssl_processor(),
            "cpu",
        )
        wav = _make_wav(tmp_path)

        from src.features.ssl import extract_ssl_features

        with pytest.raises(ValueError, match="Unknown pooling"):
            extract_ssl_features(wav, layers=[0], pooling="max")

    @patch("src.features.ssl.torchaudio.load", side_effect=_mock_torchaudio_load)
    @patch("src.features.ssl._get_ssl_model")
    def test_empty_layers_returns_empty_dict(self, mock_get, _mock_load, tmp_path):
        """Requesting no layers returns empty dict."""
        mock_get.return_value = (
            _build_mock_ssl_model(),
            _build_mock_ssl_processor(),
            "cpu",
        )
        wav = _make_wav(tmp_path)

        from src.features.ssl import extract_ssl_features

        result = extract_ssl_features(wav, layers=[])
        assert result == {}

    @patch("src.features.ssl.torchaudio.load", side_effect=_mock_torchaudio_load)
    @patch("src.features.ssl._get_ssl_model")
    def test_stereo_input_handled(self, mock_get, _mock_load, tmp_path):
        """Stereo WAV is downmixed to mono without error."""
        mock_get.return_value = (
            _build_mock_ssl_model(),
            _build_mock_ssl_processor(),
            "cpu",
        )
        wav = _make_stereo_wav(tmp_path)

        from src.features.ssl import extract_ssl_features

        result = extract_ssl_features(wav, layers=[0])
        assert 0 in result
        assert result[0].ndim == 1


# ===========================================================================
# backbone.py tests (Qwen3-TTS model fully mocked)
# ===========================================================================


def _build_mock_backbone_with_layers(hidden_dim: int = 2048, n_layers: int = 28):
    """Build a mock Qwen3-TTS model with real nn.Module layers for hook testing.

    Creates nn.Linear layers that support register_forward_hook().
    The mock's generate_voice_clone triggers a forward pass through all layers,
    causing hooks to fire and capture hidden states.
    """
    import torch.nn as nn

    layers = nn.ModuleList([
        nn.Linear(hidden_dim, hidden_dim, bias=False)
        for _ in range(n_layers)
    ])

    model = MagicMock()

    # Set up attribute path for _find_talker_layers(): model.model.talker.layers
    talker_mock = MagicMock()
    talker_mock.layers = layers
    model_inner = MagicMock()
    model_inner.talker = talker_mock
    model.model = model_inner

    # generate_voice_clone triggers forward through all layers (fires hooks)
    def _trigger_forward(**kwargs):
        x = torch.randn(1, 10, hidden_dim)
        for layer in layers:
            x = layer(x)

    model.generate_voice_clone = MagicMock(side_effect=_trigger_forward)

    return model


class TestExtractBackboneFeatures:
    """Tests for extract_backbone_features() with mocked Qwen3-TTS.

    The backbone uses hook-based extraction on the talker's transformer
    layers. Tests use real nn.Module layers with mocked model loading.
    """

    @pytest.fixture(autouse=True)
    def _reset_backbone_globals(self):
        """Reset module-level globals before each test."""
        import src.features.backbone as bb_mod
        bb_mod._backbone_model = None
        bb_mod._backbone_device = None
        bb_mod._hooks = []
        bb_mod._captured_hidden_states = {}
        yield
        bb_mod._backbone_model = None
        bb_mod._backbone_device = None
        bb_mod._hooks = []
        bb_mod._captured_hidden_states = {}

    @patch("src.features.backbone._get_backbone_model")
    def test_output_keys_match_requested_layers(self, mock_get, tmp_path):
        """Returned dict keys must match the requested layers."""
        hidden_dim = 2048
        mock_get.return_value = _build_mock_backbone_with_layers(
            hidden_dim=hidden_dim, n_layers=28
        )
        wav = _make_wav(tmp_path)

        from src.features.backbone import extract_backbone_features

        result = extract_backbone_features(
            wav, text="Olá mundo", layers=[0, 4, 8, 12], device="cpu"
        )

        assert set(result.keys()) == {0, 4, 8, 12}

    @patch("src.features.backbone._get_backbone_model")
    def test_output_shape_per_layer(self, mock_get, tmp_path):
        """Each layer output must be (hidden_dim,) after mean_temporal pooling."""
        hidden_dim = 2048
        mock_get.return_value = _build_mock_backbone_with_layers(
            hidden_dim=hidden_dim, n_layers=28
        )
        wav = _make_wav(tmp_path)

        from src.features.backbone import extract_backbone_features

        result = extract_backbone_features(
            wav, text="Teste", layers=[0, 27], device="cpu"
        )

        for layer_idx, vec in result.items():
            assert vec.shape == (hidden_dim,), (
                f"Layer {layer_idx}: expected ({hidden_dim},), got {vec.shape}"
            )

    @patch("src.features.backbone._get_backbone_model")
    def test_output_is_numpy_float32(self, mock_get, tmp_path):
        """Output must be numpy float32."""
        mock_get.return_value = _build_mock_backbone_with_layers()
        wav = _make_wav(tmp_path)

        from src.features.backbone import extract_backbone_features

        result = extract_backbone_features(
            wav, text="Teste", layers=[0], device="cpu"
        )

        vec = result[0]
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32

    @patch("src.features.backbone._get_backbone_model")
    def test_out_of_range_layer_skipped(self, mock_get, tmp_path):
        """Layer beyond model capacity is skipped."""
        n_layers = 12
        mock_get.return_value = _build_mock_backbone_with_layers(n_layers=n_layers)
        wav = _make_wav(tmp_path)

        from src.features.backbone import extract_backbone_features

        result = extract_backbone_features(
            wav, text="Teste", layers=[0, 50], device="cpu"
        )

        assert 0 in result
        assert 50 not in result

    @patch("src.features.backbone._get_backbone_model")
    def test_invalid_pooling_raises(self, mock_get, tmp_path):
        """Unknown pooling must raise ValueError."""
        mock_get.return_value = _build_mock_backbone_with_layers()
        wav = _make_wav(tmp_path)

        from src.features.backbone import extract_backbone_features

        with pytest.raises(ValueError, match="Unknown pooling"):
            extract_backbone_features(
                wav, text="Teste", layers=[0], device="cpu", pooling="attention"
            )

    @patch("src.features.backbone._get_backbone_model")
    def test_empty_layers_returns_empty_dict(self, mock_get, tmp_path):
        """Requesting no layers returns empty dict."""
        mock_get.return_value = _build_mock_backbone_with_layers()
        wav = _make_wav(tmp_path)

        from src.features.backbone import extract_backbone_features

        result = extract_backbone_features(
            wav, text="Teste", layers=[], device="cpu"
        )
        assert result == {}

    @patch("src.features.backbone._get_backbone_model")
    def test_stereo_input_handled(self, mock_get, tmp_path):
        """Stereo WAV is passed to model without error (model handles audio)."""
        mock_get.return_value = _build_mock_backbone_with_layers()
        wav = _make_stereo_wav(tmp_path)

        from src.features.backbone import extract_backbone_features

        result = extract_backbone_features(
            wav, text="Teste estéreo", layers=[0], device="cpu"
        )
        assert 0 in result
        assert result[0].ndim == 1

    @patch("src.features.backbone._get_backbone_model")
    def test_different_hidden_dims(self, mock_get, tmp_path):
        """Model with different hidden_dim produces vectors of that size."""
        hidden_dim = 768
        mock_get.return_value = _build_mock_backbone_with_layers(
            hidden_dim=hidden_dim, n_layers=12
        )
        wav = _make_wav(tmp_path)

        from src.features.backbone import extract_backbone_features

        result = extract_backbone_features(
            wav, text="Teste", layers=[0, 6, 11], device="cpu"
        )

        for layer_idx, vec in result.items():
            assert vec.shape == (hidden_dim,)
