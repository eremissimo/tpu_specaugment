import torch
import torchaudio.functional as taf
import tpu_aug_impl as impl


def test_vocoder(batch=2, n_bins=65, n_frames=300):
    """Output comparison between impl.phase_vocoder and torchaudio.phase_vocoder"""
    hop_length = n_bins - 1
    spec_real = torch.randn((batch, n_bins, n_frames, 2))
    spec_clpx = torch.view_as_complex(spec_real)
    phase_advance = torch.linspace(0, torch.pi * hop_length, n_bins)[..., None]
    out_torch = taf.phase_vocoder(spec_clpx, 1.2, phase_advance)
    out_torch = torch.view_as_real(out_torch)
    out_impl = impl.phase_vocoder(spec_real, 1.2, phase_advance)
    assert torch.isclose(out_torch, out_impl).all()


def test_stft(batch=2, n_fft=1024, hop_length=256):
    """Output comparison between impl.STFTModule and torch.stft"""
    waveform = torch.randn((batch, 10*n_fft))
    # window = torch.hann_window(n_fft)
    window = barlett_window(n_fft)
    spec_torch = torch.stft(waveform, n_fft, hop_length=hop_length, window=window, return_complex=False, center=False)
    specmodule = impl.STFTModule(n_fft, hop_length=hop_length, window=window)
    spec_impl = specmodule(waveform)
    assert torch.isclose(spec_torch, spec_impl, atol=1e-4, rtol=1e-4).all()


def _check_out_istft(n_fft=1024, hop_length=256, window_fn=torch.hann_window, normalization="nola"):
    """Output comparison between impl.ISTFTModule(torch.stft(.)) and the original waveform"""
    batch = 1
    waveform = torch.randn((batch, 10*n_fft))
    # waveform = torch.ones((batch, 10*n_fft))
    window = window_fn(n_fft)
    spec = torch.stft(waveform, n_fft, hop_length=hop_length, window=window, return_complex=False, center=False)
    istft = impl.ISTFTModule(n_fft, hop_length=hop_length, window=window, normalization=normalization)
    wave_transformed = istft(spec)
    assert torch.isclose(waveform, wave_transformed, atol=1e-4, rtol=1e-4).sum() > (waveform.shape[-1] - 2*n_fft)


def test_istft():
    _check_out_istft(1024, 256, torch.hann_window, "nola")
    _check_out_istft(1024, 256, torch.hann_window, "cola")
    #_check_out_istft(1024, 512, torch.hann_window, "cola")      # this doesn't pass because hann_window doesn't satisfy
                                                                 # cola constraint at hop_length = n_fft//2
    _check_out_istft(1024, 256, barlett_window, "nola")
    # _check_out_istft(1024, 256, barlett_window, "cola")        # this is not cola-compliant too
    _check_out_istft(1024, 512, barlett_window, "nola")

    _check_out_istft(1024, 256, cosine_window, "nola")
    _check_out_istft(1024, 256, cosine_window, "cola")
    _check_out_istft(1024, 512, cosine_window, "cola")
    _check_out_istft(1024, 512, cosine_window, "nola")


def barlett_window(M, sym=True):
    start = -1
    constant = 2 / (M if not sym else M - 1)
    k = torch.linspace(start=start,
                       end=start + (M - 1) * constant,
                       steps=M,
                       dtype=torch.float32,
                       requires_grad=False)
    return 1 - torch.abs(k)


def cosine_window(M, sym=True):
    start = 0.5
    constant = torch.pi / (M + 1 if not sym and M > 1 else M)
    k = torch.linspace(start=start * constant,
                       end=(start + (M - 1)) * constant,
                       steps=M,
                       dtype=torch.float32,
                       requires_grad=False)
    return torch.sin(k)



