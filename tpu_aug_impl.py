import torch
import torch.nn as nn
import torch.nn.functional as ff
from torch.linalg import vector_norm
from torchaudio.functional import resample
import torch.fft as fft
import einops
import random
from typing import Optional, Union, Tuple
from fractions import Fraction


class STFTModule(nn.Module):
    """Basically it is a conv1d with the DFT matrix as a 1->n channel kernel.
    Although TPUs support basic complex-valued tensor computations, there are some complications in real(), imag()
    and casting to reals, so here and below all complex computations are emulated by real-valued tensors."""
    def __init__(self, n_fft: int,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 window: Optional[torch.Tensor] = None,
                 pad: int = 0,
                 center: bool = False,
                 pad_mode: str = "reflect"):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length or n_fft // 4      # as in librosa
        self.pad = pad
        if window is None:
            window = torch.hann_window(self.win_length)
        window = _window_pad(window, n_fft)
        self.center = center
        self.pad_mode = pad_mode
        # dft_kernel = _get_dft_matrix_manual(n_fft, window)
        dft_kernel = _get_dft_matrix(n_fft, window, return_complex=False)
        self.register_buffer("kernel", dft_kernel)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        # shape: [batch, len] or [len]
        if self.center:
            signal = _signal_pad(signal, self.n_fft, self.pad_mode)
        signal, expanded_batch = _expand_dims(signal)
        # stft
        signal = ff.conv1d(signal, self.kernel, stride=self.hop_length)
        # real and imag parts separation
        signal = einops.rearrange(signal, "... (ri freq) n -> ... freq n ri", ri=2)
        if expanded_batch:
            signal.squeeze_(0)
        # output shape: [batch, n_bins, n_frames, 2] or [n_bins, n_frames, 2]
        return signal


class ISTFTModule(nn.Module):
    """Basically it's a conv_transpose1d with a dft kernel. A dft kernel is not like a pure rfft(eye(n_fft)) matrix
    but a slightly modified version of it to accomodate symmetric negative frequencies."""
    def __init__(self, n_fft: int,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 window: Optional[torch.Tensor] = None,
                 pad: int = 0,
                 center: bool = False,
                 pad_mode: str = "reflect",
                 normalization: str = "cola"):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length or n_fft // 4      # as in librosa
        if window is None:
            window = torch.hann_window(self.win_length)
        window = _window_pad(window, n_fft)

        dft_kernel = _get_dft_matrix(n_fft, window, return_complex=False)  #  [2*freq, 1, n_fft]
        dft_kernel[1: (n_fft+1)//2, ...] *= 2.
        dft_kernel[(n_fft+2)//2:, ...] *= 2.
        dft_kernel /= n_fft
        self.register_buffer("kernel", dft_kernel)
        if normalization == "cola":
            self.normalization = COLA(n_fft, hop_length, window)
        elif normalization == "nola":
            self.normalization = NOLA(n_fft, hop_length, window)
        else:
            raise ValueError(f"Unexpected normalization specification <{normalization}>. Possible entries: cola, nola")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # spectrogram tensor shape: [batch, frequencies, frames, 2] or [frequencies, frames, 2]

        # concatenate the real and imaginary parts in freq dim
        x = einops.rearrange(x, "... freq n ri -> ... (ri freq) n", ri=2)       # [..., 2*freq, frames]
        # istft: fft + overlap-add
        x = ff.conv_transpose1d(x, self.kernel, stride=self.hop_length).squeeze(1)
        x = self.normalization(x)
        return x


class ISTFTNormalization(nn.Module):
    """Strategies of ISTFT normalization"""
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement me")


class COLA(ISTFTNormalization):
    """Divides the waveform by the COLA constant. This method is fast but presents attenuation artifacts
    (~n_fft samples at the beginning and the end of the waveform)"""
    def __init__(self, n_fft: int, hop_length: int, window: torch.Tensor):
        super().__init__()
        inv_cola_c = hop_length / (window ** 2).sum()      # inverse of COLA constant [tested only on Hann window]
        self.register_buffer("_coeff", inv_cola_c)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform * self._coeff


class NOLA(ISTFTNormalization):
    """This method is slower but reduces the attenuation."""
    def __init__(self, n_fft: int, hop_length: int, window: torch.Tensor):
        super().__init__()
        self.register_buffer("window_ker", window[None, None, :] ** 2)
        self.hop_length = hop_length
        self.n_fft = n_fft

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        n_frames = 1 + ((waveform.shape[-1] - self.n_fft) // self.hop_length)
        # analogue of librosa.filters.window_sumsquare()
        nola = ff.conv_transpose1d(torch.ones((1, n_frames)), self.window_ker, stride=self.hop_length)
        nola = torch.where(nola < 1e-12, 1., nola)
        waveform = waveform / nola
        return waveform


def _window_pad(window, n_fft):
    if dlw := (n_fft - len(window)) > 0:
        half_dlw = dlw // 2
        window = ff.pad(window, (half_dlw, dlw - half_dlw), value=0.)
    elif dlw < 0:
        half_dlw = dlw // 2
        window = window[half_dlw: half_dlw + n_fft]
    return window


def _signal_pad(signal, n_fft, pad_mode):
    signal_dim = signal.dim()
    extended_shape = [1] * (3 - signal_dim) + list(signal.size())
    pad = int(n_fft // 2)
    signal = ff.pad(signal.view(extended_shape), [pad, pad], pad_mode)
    signal = signal.view(signal.shape[-signal_dim:])
    return signal


def _expand_dims(input):
    expanded_batch = False
    if input.dim() == 1:
        expanded_batch = True
        input = input[None, None,:]
    elif input.dim() == 2:
        input = input.unsqueeze(1)       # added channel dim
    else:
        raise RuntimeError("expected a 1D or 2D tensor")
    return input, expanded_batch


def _get_dft_matrix(n_fft: int, window: Optional[torch.Tensor] = None,
                    device: Union[torch.device, str, None] = None, return_complex: bool = True):
    dftm = fft.rfft(torch.eye(n_fft, device=device)).T       # [n_bins, n_fft]
    if window is not None:
        dftm = dftm * window.unsqueeze(0)
    if not return_complex:
        dftm = torch.view_as_real(dftm)         # falls back to cpu when used on xla devices
        # the op below is equivalent to torch.cat((dftm[...,0], dftm[...,1]), dim=0),
        # vertical concatenation of real and imag parts
        dftm = einops.rearrange(dftm, "f n ri -> (ri f) n")
    dftm.unsqueeze_(1)    # [bins, 1, n_fft]
    return dftm


def _get_dft_matrix_manual(n_fft: int, window: Optional[torch.Tensor] = None,
                           device: Union[torch.device, str, None] = None):
    """If torch.fft doesn't work for whatever reason, use this function"""
    if window is not None:
        window = window.unsqueeze(0)
    n_bins = n_fft // 2 + 1     # number of frequency bins
    w = -(2. * torch.pi / n_fft) * torch.arange(n_bins, dtype=torch.float32, device=device)
    n = torch.arange(n_fft, dtype=torch.float32, device=device)
    ph = einops.einsum(w, n, "w, n -> w n")
    # convolution with this kernel gives a concatenation of real and imaginary parts of the complex stft
    dft_ri = torch.cat((torch.cos(ph), torch.sin(ph)), dim=0)
    if window is not None:
        dft_ri = dft_ri * window
    dft_ri.unsqueeze_(1)
    return dft_ri


def _angle(realimag: torch.Tensor) -> torch.Tensor:
    """An argument (phase) of the 'complex-like' tensor containing the real and imaginary parts.
    realimag is supposed to be a tensor of the shape [..., 2] where
    realimag[..., 0] is the real part and realimag[..., 1] is the imaginary part."""
    return torch.atan2(realimag[..., 1], realimag[..., 0])


def _abs(realimag: torch.Tensor) -> torch.Tensor:
    """A modulus (magnitude) of the 'complex-like' tensor containing the real and imaginary parts.
    realimag is supposed to be a tensor of the shape [..., 2] where
    realimag[..., 0] is the real part and realimag[..., 1] is the imaginary part."""
    return vector_norm(realimag, ord=2, dim=-1, keepdim=False)


def _polar(mag: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """A 'complex-like' tensor with modulus mag and argument angle.
    An equivalent of torch.polar."""
    return torch.stack((mag*torch.cos(angle), mag*torch.sin(angle)), dim=-1)


def phase_vocoder(real_specgrams: torch.Tensor, rate: Union[float, torch.FloatTensor],
                  phase_advance: torch.Tensor) -> torch.Tensor:
    """A rewritten implementation of the torchaudio's phase vocoder for real-typed spectrograms
    of the shape [..., freq, frames, 2]"""
    if rate == 1.0:
        return real_specgrams
    # pack batch
    shape = real_specgrams.size()
    real_specgrams = real_specgrams.reshape([-1] + list(shape[-3:]))
    time_steps = torch.arange(0, real_specgrams.size(-2), rate,
                              device=real_specgrams.device, dtype=real_specgrams.dtype)
    alphas = time_steps % 1.0
    phase_0 = _angle(real_specgrams[..., :1, :])       # [batch, freq, 1]
    # Time Padding
    real_specgrams = torch.nn.functional.pad(real_specgrams, [0, 0, 0, 2])

    rs_0 = real_specgrams.index_select(-2, time_steps.long())
    rs_1 = real_specgrams.index_select(-2, (time_steps + 1).long())

    angle_0 = _angle(rs_0)
    angle_1 = _angle(rs_1)

    norm_0 = _abs(rs_0)
    norm_1 = _abs(rs_1)

    phase = angle_1 - angle_0 - phase_advance           # [batch, freq, new_n]
    phase = phase - 2 * torch.pi * torch.round(phase / (2 * torch.pi))

    # Compute Phase Accum
    phase = phase + phase_advance
    phase = torch.cat([phase_0, phase[..., :-1]], dim=-1)
    phase_acc = torch.cumsum(phase, -1)

    mag = alphas * norm_1 + (1 - alphas) * norm_0
    real_specgrams = _polar(mag, phase_acc)

    # unpack batch
    real_specgrams_stretch = real_specgrams.reshape(shape[:-3] + real_specgrams.shape[1:])
    return real_specgrams_stretch


class RandomRateGenerator:
    """A generator of synchronized random rates for both TimeStretch and Resample"""
    def __init__(self, max_ts_rate: Optional[float] = None, max_ps_semitones: Optional[int] = None):
        super().__init__()
        self.max_ts_rate = max_ts_rate
        self.max_ps_semitones = max_ps_semitones
        self._ts_random_rate: Optional[float] = None
        self._ps_random_rate: Optional[Fraction] = None

    def set_ts_max_rate(self, max_rate: float):
        assert max_rate > 1.
        self.max_ts_rate = float(max_rate)

    def set_ps_max_semitones(self, max_semitones: int):
        assert max_semitones >= 1
        self.max_ps_semitones = int(max_semitones)

    def get_ts_random_rate(self) -> float:
        self._update_state()   # populates _ts_random_rate when needed
        if self._ts_random_rate is None:
            raise RuntimeError("Unexpected second attempt to consume the time stretch random rate before "
                               "the state update")
        val = self._ts_random_rate
        self._ts_random_rate = None
        return val

    def get_ps_random_rate(self) -> Tuple[int, int]:
        if self.max_ps_semitones is None:
            raise RuntimeError("Cannot generate rates for pitch shift without max_ps_semitones bounds."
                               "Set up the bounds with set_ps_max_semitones method.")
        self._update_state()   # populates _ps_random_rate when needed
        if self._ps_random_rate is None:
            raise RuntimeError("Unexpected second attempt to consume the pitch shift random rate before "
                               "the state update")
        val: Fraction = self._ps_random_rate
        self._ps_random_rate = None
        return val.numerator, val.denominator

    def _update_state(self):
        ts = self.max_ts_rate is not None       # time stretch in the list of transforms
        ps = self.max_ps_semitones is not None  # pitch shift in the list of transforms
        need_ts = self._ts_random_rate is None
        need_ps = self._ps_random_rate is None
        if ts and ps and need_ts and need_ps:
            ps_rate = self._sample_ps_rate()
            ts_rate = self._sample_ts_rate()
            ts_rate *= float(ps_rate)           # THIS! rate of time stretch is multiplied by the rate of pitch shift
            self._ps_random_rate = ps_rate
            self._ts_random_rate = ts_rate
        elif ts and not ps and need_ts:
            self._ts_random_rate = self._sample_ts_rate()
        elif not ts and ps and need_ts and need_ps:
            self._ps_random_rate = self._sample_ps_rate()
            self._ts_random_rate = float(self._ps_random_rate)

    def _sample_ts_rate(self) -> float:
        # logarithmically spaced random number in the interval [1/max_ts_rate, max_ts_rate]
        return self.max_ts_rate**(2.*random.random() - 1.)

    def _sample_ps_rate(self, max_denominator=10) -> Fraction:
        # logarithmically spaced discrete random number
        # in the interval [2**(-max_ps_semitones/12), 2**(max_ps_semitones/12)]
        i = random.randint(-self.max_ps_semitones, self.max_ps_semitones)
        rate = 2.0**(i/12.0)
        return Fraction(rate).limit_denominator(max_denominator)


class TimeStretchModule(nn.Module):
    def __init__(self, n_fft: int = 1024, hop_length: Optional[int] = None,
                 max_rate: float = 1.1, rate_gen: Optional[RandomRateGenerator] = None):
        super().__init__()
        if max_rate <= 1.:
            raise ValueError("max_rate is supposed to be >1.0 so as the actual rate "
                             "is sampled from the interval [1/max_rate, max_rate]")
        n_freq = n_fft // 2 + 1
        hop_length = hop_length or n_fft // 4           # as in librosa
        self.max_rate = float(max_rate)
        self.random_rate_gen = rate_gen
        if rate_gen is not None:
            self.random_rate_gen.set_ts_max_rate(self.max_rate)
        self.register_buffer("phase_advance", torch.linspace(0, torch.pi * hop_length, n_freq)[..., None])

    def device(self):
        return self.phase_advance.device

    def _random_rate(self) -> float:
        # a random number from the interval [1/max_rate, max_rate] randomized uniformly in logarithmic space
        if self.random_rate_gen is not None:
            # sync with pitch shift rate generation
            rate = self.random_rate_gen.get_ts_random_rate()
        else:
            # independent rate generation [not being used actually]
            rate = self.max_rate**(2.*random.random() - 1.)
        return rate

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        rate = self._random_rate()
        if abs(rate - 1.) < 1e-3:
            return spectrogram
        return phase_vocoder(spectrogram, rate, self.phase_advance)


class ResampleModule(nn.Module):
    def __init__(self, max_semitones: int = 1, rate_gen: Optional[RandomRateGenerator] = None):
        super().__init__()
        if max_semitones < 1:
            raise ValueError("max_semitones is supposed to be an integer >= 1 so as the actual rate "
                             "is sampled from the interval [2**(-max_semitones/12), 2**(max_semitones/12)]")
        self.max_semitones = int(max_semitones)
        self.random_rate_gen = rate_gen
        if rate_gen is not None:
            self.random_rate_gen.set_ps_max_semitones(max_semitones)

    def _random_rate(self) -> Tuple[int, int]:
        if self.random_rate_gen is not None:
            # sync with time stretch rate generation
            rate = self.random_rate_gen.get_ps_random_rate()
        else:
            # independent rate generation [not being used actually]
            max_denominator = 10
            i = random.randint(-self.max_semitones, self.max_semitones)
            rate = 2.0**(i/12.0)
            rate = Fraction(rate).limit_denominator(max_denominator).as_integer_ratio()
        return rate

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        num, denom = self._random_rate()
        waveform = resample(waveform, num, denom)
        return waveform


class FrequencyMaskingModule(nn.Module):
    def __init__(self, n_fft: int,  max_mask_len: int = 1, **ignore):
        super().__init__()
        self.max_mask_len = int(max_mask_len)
        if self.max_mask_len < 1:
            raise ValueError(f"max_mask_len should be >= 1. "
                             f"Why would you need FrequencyMasking with {max_mask_len} band at all?")
        self.n_freq = n_fft//2 + 1

    def _random_freq_bnds(self) -> Tuple[int, int]:
        mask_len = random.randint(0, self.max_mask_len)
        f0 = random.randint(0, self.n_freq - 1 - mask_len)
        return f0, f0 + mask_len

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        f0, f1 = self._random_freq_bnds()
        spectrogram[:, f0:f1, :, :] = 0.      #   [batch, frequencies, frames, real_imag]
        return spectrogram

