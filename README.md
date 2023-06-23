# TPU SpecAugment
## Description
This is a tiny library for spectral augmentations on xla devices. \
List of supported transforms:
* STFT -- Short Time Fourier Transform
* ISTFT -- Inverse Short Time Fourier Transform
* Frequency Masking -- randomly zeros out a span of $n$ frequency bins, $`n \in `$ [0,  max_mask_len]
* Time Stretch -- stretches the input spectrogram with a logarithmically spaced random rate from [1/max_rate, max_rate] 
* Pitch Shift -- randomly shifts the pitch by $n$ semitones, $`n \in `$ [-max_semitones, max_semitones]

The transforms instantiating is as simple as 
```python
import torch
import tpu_specaugment as tsa

n_fft = 1024
hop_length = n_fft//4
window = torch.hann_window(n_fft)

aug = tsa.SpecAugment(
    tsa.STFT(n_fft=n_fft, hop_length=hop_length, window=window),
    tsa.FrequencyMasking(max_mask_len=10),
    tsa.TimeStretch(max_rate=1.2),
    tsa.PitchShift(max_semitones=4),
    tsa.ISTFT()     # takes params from STFT
)
```
STFT is optional here (i.e. when the data loading pipeline already 
provides stft-spectrograms). But when it's not there it is mandatory to pass
the used stft params into specaugment transforms:
```python
aug = tsa.SpecAugment(
    tsa.FrequencyMasking(n_fft=n_fft, hop_length=hop_length, max_mask_len=10),
    tsa.TimeStretch(n_fft=n_fft, hop_length=hop_length, max_rate=1.2),
    tsa.PitchShift(n_fft=n_fft, hop_length=hop_length, max_semitones=4),
    tsa.ISTFT(n_fft=n_fft, hop_length=hop_length, window=window)
)
```

The `tsa.SpecAugment` is a `torch.nn.Module` so it could (and should) 
be placed on a device to perform a fast random spectral transformations 
of audio waveforms before feeding them into a neural network.

This is an example of usage:
```python
aug = aug.to(device)
# training loop
for waveforms, targets in dataloader:
    waveforms = waveforms.to(device)
    waveforms = aug(waveforms)
    ...
```
 
## Motivation
The `tsa.SpecAugment` provides support of spectral augmentations on TPUs.
There are several problems with direct use of built-in `torch` and `torchaudio` 
functions for that:
1. `torch.stft` doesn't work on TPU entirely because of `Tensor.as_strided()` limitations.\
   https://github.com/pytorch/xla/issues/2241
2. Several ops fall back into cpu (e.g. `torch.istft`)
3. Although complex tensor data type is supported by pytorch/xla for algebraic 
   computations and autograd, the basic methods that require conversion to real 
   dtype (e.g. `Tensor.angle`, `Tensor.abs`) don't perform the conversion. 
   And this breaks some `torchaudio.functional.phase_vocoder` internals,
   which is the basis for built-in torchaudio's TimeStretch and PitchShift. \
   UPD: `Tensor.angle` now returns real valued tensor, but `Tensor.abs` is still complex.
   
So basically to get a workaround, STFT and ISTFT are implemented here as a strided 
convolutions and a transposed convolutions respectively with the DFT matrices, 
phase vocoder is reimplemented (based on original torchaidio code) to emulate 
complex computations with real-valued tensors. 

## Dependencies
* pytorch
* torchaudio
* einops
* multipledispatch (probably will be discarded soon)

## Limitations
1. Although the implemented `STFT` and `ISTFT` transforms work on CPU and CUDA devices, 
   these are really slow compared to the native `torch.stft` and `torch.istft`. So all
   these are only useful only on TPUs as a workaround.
    
2. Currently Pitch shifting here is implemented as a combination of TimeStretch and 
   Resample modules. And Resample works on waveforms, not spectrograms, so it is 
   mandatory to use `ISTFT` when `PitchShift` is used. The Resample is called 
   after ISTFT under the hood. \
   Unfortunately, this means that if you want to get a spectrogram as an output, 
   the transforms have to be defined as this 
   ```python
    aug = tsa.SpecAugment(
        tsa.STFT(n_fft=n_fft, hop_length=hop_length, window=window),
        tsa.PitchShift(max_semitones=4),
        tsa.ISTFT(),
        tsa.STFT(n_fft=n_fft, hop_length=hop_length, window=window)    
    )
    ```
   which is ugly as hell.\
   TODO:
   When `PitchShift` is in the list of transforms and `ISTFT` isn't, a different pitch 
   shifting strategy must be performed i.e. that works on time-frequency domain directly.
   
## TODO list
- [ ] spectrogram-to-spectrogram pitch shifting
