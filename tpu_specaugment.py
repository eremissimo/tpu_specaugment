"""A pytorch/xla-friendly implementation of stft, istft and spectral augmentations.
The native torch.stft doesn't work on TPUs due to a lack of support of operations
like unfold and tensor.as_strided."""

import torch
import torch.nn as nn
import tpu_aug_impl as impl
from dataclasses import dataclass
from typing import Optional, Iterable, List, Tuple, Dict, Union
from multipledispatch import dispatch
from abc import ABC


"""
    * Parameters of transforms *
    
    The purpose of these classes is to provide a user input parameters of the transforms to the
    SpecAugment.__init__
    
"""


class AbstractTransform(ABC):
    pass


@dataclass
class STFT(AbstractTransform):
    n_fft: int
    win_length: Optional[int] = None
    hop_length: Optional[int] = None
    window: Optional[torch.Tensor] = None


@dataclass
class ISTFT(AbstractTransform):
    n_fft: Optional[int] = None
    win_length: Optional[int] = None
    hop_length: Optional[int] = None
    window: Optional[torch.Tensor] = None
    normalization: str = "nola"             # possible values: 'cola', 'nola'


@dataclass
class SpectralTransform(AbstractTransform):
    n_fft: Optional[int] = None
    hop_length: Optional[int] = None
    # iid_params: bool = False    # whether to apply different random params to each example in a batch
                                    # NOT YET IMPLEMENTED


@dataclass
class FrequencyMasking(SpectralTransform):
    max_mask_len: int = 10     # max length (in bins) of the frequency mask


@dataclass
class PitchShift(SpectralTransform):
    max_semitones: int = 1        # max pitch shift in semitones


@dataclass
class TimeStretch(SpectralTransform):
    max_rate: float = 1.1


""" *  Additional params for algorithmic representation: """


@dataclass
class _TimeStretch(TimeStretch):
    rate_gen: Optional[impl.RandomRateGenerator] = None


@dataclass
class _Resample(AbstractTransform):
    max_semitones: int = 1
    rate_gen: Optional[impl.RandomRateGenerator] = None


"""
    * SpecAugment *
    
    The main nn.Module for the spectral augmentations.
"""


class SpecAugment(nn.Sequential):
    """The main module of a SpecAugment transforms.

    Example of usage:
    spec_augment = SpecAugment(
        STFT(n_fft=1024),
        TimeStretch(max_rate=1.2),
        PitchShift(max_semitones=4),
        ISTFT()     # takes params from STFT(...) by default unless overrided
    ).to(device)

    <... later in the training loop >
    for batch in dataloader:
        batch = batch.to(device)
        batch = spec_augment(batch)
    <...>
    """
    def __init__(self, *transform_params):
        if len(transform_params) == 1 and isinstance(transform_params[0], Iterable):
            transform_params = transform_params[0]
        transform_params = list(transform_params)
        _validate_params(transform_params)
        _params_surgery(transform_params)
        modules = map(params2modules, transform_params)
        super().__init__(*modules)


"""  
    *  Utils  *   
"""


def _validate_params(params: List):
    """Check the list of transform specifications"""
    # 1. STFT may or may not be at 0th place of the list, or could be in the last place after ISTFT
    if _instance_in_list(params[1:-1], STFT) or \
            (isinstance(params[-1], STFT) and not isinstance(params[-2], ISTFT)):
        raise ValueError("STFT must be the first transform used, so place it to the head of the list "
                         "(or dont use it at all if the input data is already transformed to spectrograms). \n"
                         "Also it could be the last transform used but only after ISTFT if you are forced to "
                         "transform the spectrograms to waveforms after PitchShift usage.")
    # 2. ISTFT may or may not be at -1th place of the list but never on 0,1...-2 places. Also it could be
    # at -2th  place if the last element is STFT
    if _instance_in_list(params[:-2], ISTFT) or \
            (isinstance(params[-2], ISTFT) and not isinstance(params[-1], STFT)):
        raise ValueError("ISTFT must be the last transform used, so place it to the last element of the list "
                         "(or dont use it at all if the output data is supposed to be a batch of spectrograms). \n"
                         "Also it could be the second to last transform used but only before STFT if you are forced to "
                         "transform the spectrograms to waveforms after PitchShift usage.")
    # 3. If PitchShift is in the list then the last transform should be ISTFT because of pitch
    # shifting algorithm currently being used. It time-stretches the spectrogram, then converts it to a
    # waveform with ISTFT, then resamples it.
    if _instance_in_list(params[:-1], PitchShift) and not _instance_in_list(params[-2:], ISTFT):
        raise ValueError("It's mandatory to use ISTFT when using PitchShift due to algorithmic reasons.")
    # 4. Single PitchShift
    if _n_instances_in_list(params, PitchShift) > 1:
        raise ValueError("Multiple instances of PitchShift found. This makes unnecessary "
                         "complications so it's not supported.")
    # 5. Single TimeStretch
    if _n_instances_in_list(params, TimeStretch) > 1:
        raise ValueError("Multiple instances of TimeStretch found. This makes unnecessary "
                         "complications so it's not supported.")
    # 6. If STFT is not present in the list (not the first element to be precise) then stft params needed by
    # modules must be provided (at least the n_fft value) and all provided values must be equal
    # (or None in the case of hop_length)
    if not isinstance(params[0], STFT):
        fst_stft_param = None
        for i, elem in enumerate(params[1:]):
            if isinstance(elem, SpectralTransform):
                if _has_default_args(elem):
                    raise ValueError(f"n_fft is not specified for params[{i+1}].")
                curr_stft_param = {"n_fft": elem.n_fft, "hop_length": elem.hop_length}
                if fst_stft_param is None:
                    fst_stft_param = curr_stft_param
                elif not _similar(fst_stft_param, curr_stft_param):
                    raise ValueError(f"Incompatible stft parameters found: params[0] vs params[{i+1}].")


def _params_surgery(params: List[AbstractTransform]):
    """A transformation of parameter list containing aforementioned dataclasses to better suit the underlying
    algorithms. In other words it's a map from the interface representation to the algorithmic representation.

    A PitchShift is split up into a Time Stretch and Resample, All TimeStretch instances are combined into a
    single instance with a product rate to do a single vocoder pass, Resample is applied after ISTFT."""
    _provide_stft_params_to_other(params)
    rate_gen = None
    ps_semitones = None
    idx_to_del = None
    ts_in_params = _instance_in_list(params, TimeStretch)
    for i, elem in enumerate(params):
        if isinstance(elem, TimeStretch):
            if rate_gen is None:
                rate_gen = impl.RandomRateGenerator()
            rate_gen.set_ts_max_rate(elem.max_rate)
            params[i] = _TimeStretch(rate_gen=rate_gen, **elem.__dict__)
        elif isinstance(elem, PitchShift):
            if rate_gen is None:
                rate_gen = impl.RandomRateGenerator()
            ps_semitones = elem.max_semitones
            rate_gen.set_ps_max_semitones(ps_semitones)
            if ts_in_params:
                idx_to_del = i
            else:
                params[i] = _TimeStretch(n_fft=elem.n_fft, hop_length=elem.hop_length,
                                         max_rate=2**(ps_semitones/12), rate_gen=rate_gen)
        elif isinstance(elem, ISTFT) and ps_semitones is not None:
            params.insert(i+1, _Resample(max_semitones=ps_semitones, rate_gen=rate_gen))
    if idx_to_del is not None:
        del params[idx_to_del]


def _provide_stft_params_to_other(params: List[AbstractTransform]) -> None:
    if not isinstance(params[0], STFT):
        return
    stft_full_kwargs = params[0].__dict__
    stft_reduced_kwargs = {k: stft_full_kwargs[k] for k in ("n_fft", "hop_length")}
    for elem in params[1:]:
        if isinstance(elem, (SpectralTransform, ISTFT)) and _has_default_args(elem):
            d = stft_reduced_kwargs if isinstance(elem, SpectralTransform) else stft_full_kwargs
            _assign_attrs_from_dict(elem, d)


def _assign_attrs_from_dict(p: AbstractTransform, d: Dict) -> None:
    p.__dict__.update(d)


def _has_default_args(p: Union[SpectralTransform, ISTFT]) -> bool:
    return p.n_fft is None


def _instance_in_list(lst, cl) -> bool:
    return any(isinstance(obj, cl) for obj in lst)


def _n_instances_in_list(lst, cl) -> int:
    return sum(isinstance(obj, cl) for obj in lst)


def _get_first_instance(lst, cl):
    for elem in lst:
        if isinstance(elem, cl):
            return elem
    else:
        raise ValueError(f"No objects of class <{cl.__name__}> are present in the list.")


def _similar(d1: Dict, d2: Dict):
    # substitute None with default value
    d1["hop_length"] = d1["hop_length"] or (d1["n_fft"] // 4)
    d2["hop_length"] = d2["hop_length"] or (d2["n_fft"] // 4)
    return d1 == d2


@dispatch(STFT)
def params2modules(param):
    return impl.STFTModule(**param.__dict__)


@dispatch(ISTFT)
def params2modules(param):
    return impl.ISTFTModule(**param.__dict__)


@dispatch(FrequencyMasking)
def params2modules(param):
    return impl.FrequencyMaskingModule(**param.__dict__)


@dispatch(_Resample)
def params2modules(param):
    return impl.ResampleModule(**param.__dict__)


@dispatch(_TimeStretch)
def params2modules(param):
    return impl.TimeStretchModule(**param.__dict__)


if __name__ == "__main__":
    module = SpecAugment(
        STFT(n_fft=1024, hop_length=512),
        PitchShift(max_semitones=6),
        ISTFT()
    )

    print("woah!")

