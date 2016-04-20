#!/usr/bin/env python
"""
A set of basic Audio and DSP functions using audioread and numpy
Mostly the core dsp tools collected from librosa, but with fewer dependencies
"""
import os
import warnings
import math

import audioread
import numpy as np


def load(path, sr=22050, mono=True, offset=0.0, duration=None,
         dtype=np.float32):
    """
    Load an audio file as a numpy floating point time series.

    Args:
        path: string
            path to the input file.
            Any format supported by `audioread` will work.
        sr: number > 0 [scalar]
            target sampling rate
            'None' uses the native sampling rate
        mono: bool
            convert signal to mono
        offset: float
            start reading after this time (in seconds)
        duration: float
            only load up to this much audio (in seconds)
        dtype: numeric type
            data type of `y`
    Returns:
        y: np.ndarray [shape=(n,) or (2, n)]
            audio time series
        sr: number > 0 [scalar]
            sampling rate of `y`
    """
    y = []
    with audioread.audio_open(os.path.realpath(path)) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels

        start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            end = np.inf
        else:
            end = start + (int(np.round(sr_native * duration)) * n_channels)

        n = 0

        for frame in input_file:
            frame = buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < start:
                # offset is after the current frame
                # keep reading
                continue

            if end < n_prev:
                # we're off the end.  stop reading
                break

            if end < n:
                # the end is in this frame.  crop.
                frame = frame[:end - n_prev]

            if n_prev <= start <= n:
                # beginning is in this frame
                frame = frame[(start - n_prev):]

            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)

        if n_channels > 1:
            y = y.reshape((-1, 2)).T
            if mono:
                y = to_mono(y)

        if sr is not None:
            y = resample(y, sr_native, sr)

        else:
            sr = sr_native

    # Final cleanup for dtype and contiguity
    y = np.ascontiguousarray(y, dtype=dtype)

    return (y, sr)


def write_wav(path, y, sr, norm=True):
    """
    Output a time series as a .wav file
    Args:
        path : str
            path to save the output wav file
        y : np.ndarray [shape=(n,) or (2,n)]
            audio time series (mono or stereo)
        sr : int > 0 [scalar]
            sampling rate of `y`
        norm : boolean [scalar]
            enable amplitude normalization
    """
    # Validate the buffer.  Stereo is okay here.
    valid_audio(y, mono=False)
    # Normalize
    if norm:
        headroom = 0.00003
        y = y / (np.max(np.abs(y)) + headroom)
    else:
        y = y
    # Convert to 16bit int
    wav = buf_to_int(y)
    # Check for stereo
    if wav.ndim > 1 and wav.shape[0] == 2:
        wav = wav.T
    # Save
    from scipy.io import wavfile
    wavfile.write(path, sr, wav)


def to_mono(y):
    """
    Force an audio signal down to mono.
    Args:
        y : np.ndarray [shape=(2,n) or shape=(n,)]
            audio time series, either stereo or mono
    Returns:
        y_mono : np.ndarray [shape=(n,)]
            `y` as a monophonic time-series
    """
    valid_audio(y, mono=False)
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    return y


def resample(y, orig_sr, target_sr, fix=True, scale=False):
    """Resample a time series from orig_sr to target_sr
    Args:
        y : np.ndarray [shape=(n,) or shape=(2, n)]
            audio time series.  Can be mono or stereo.
        orig_sr : number > 0 [scalar]
            original sampling rate of `y`
        target_sr : number > 0 [scalar]
            target sampling rate
        fix : bool
            adjust the length of the resampled signal to be of size exactly
            `ceil(target_sr * len(y) / orig_sr)`
        scale : bool
            Scale the resampled signal so that `y` and `y_hat` have
            approximately equal total energy.
    Returns
        y_hat : np.ndarray [shape=(n * target_sr / orig_sr,)]
            `y` resampled from `orig_sr` to `target_sr`
    """
    if y.ndim > 1:
        return np.vstack([resample(yi, orig_sr, target_sr, fix=fix)
                          for yi in y])
    valid_audio(y, mono=True)
    if orig_sr == target_sr:
        return y
    ratio = float(target_sr) / orig_sr
    n_samples = int(np.ceil(y.shape[-1] * ratio))
    try:
        # Try to use scikits.samplerate if available
        import scikits.samplerate as samplerate
        y_hat = samplerate.resample(y.T, ratio, 'sinc_best').T
    except ImportError:
        warnings.warn('Could not import scikits.samplerate. '
                      'Falling back to scipy.signal')
        import scipy.signal
        y_hat = scipy.signal.resample(y, n_samples, axis=-1)
    if fix:
        y_hat = fix_length(y_hat, n_samples)
    if scale:
        y_hat /= np.sqrt(ratio)
    return np.ascontiguousarray(y_hat, dtype=y.dtype)


def get_duration(y=None, sr=22050):
    """
    Compute the duration (in seconds) of an audio time series.
    Args:
        y: np.ndarray [shape=(n,), (2, n)] or None
            audio time series
        sr: number > 0 [scalar]
            audio sampling rate of `y`
    Returns:
        d: float >= 0
            Duration (in seconds) of the input time series.
    """
    valid_audio(y, mono=False)
    if y.ndim == 1:
        n_samples = len(y)
    else:
        n_samples = y.shape[-1]
    return float(n_samples) / sr


def fix_length(data, size, axis=-1, **kwargs):
    """
    Fix the length of array `data` to exactly `size`.
    If `data.shape[axis] < n`, pad according to kwargs.mode

    Args:
        data: np.ndarray
            array to be length-adjusted
        size: int >= 0 [scalar]
            desired length of the array
        axis: int, <= data.ndim
            axis along which to fix length
        kwargs:
            Parameters to np.pad()
    Returns:
        np.ndarray [shape=data.shape]
            `data` either trimmed or padded to length `size`
            along the specified axis.
    """
    kwargs.setdefault('mode', 'constant')
    n = data.shape[axis]
    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[slices]
    elif n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths, **kwargs)
    return data


def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """
    Convert an integer buffer to floating point values.
    Used for loading integer-valued wav data into numpy arrays.
    Args:
        x : np.ndarray [dtype=int]
            The integer-valued data buffer
        n_bytes : int [1, 2, 4]
            The number of bytes per sample in `x`
        dtype : numeric type
            The target output type (default: 32-bit float)
    Returns:
        x_float : np.ndarray [dtype=float]
            The input data buffer cast to floating point
    """
    # Invert the scale of the data
    scale = 1./float(1 << ((8 * n_bytes) - 1))
    # Construct the format string
    fmt = '<i{:d}'.format(n_bytes)
    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)


def buf_to_int(x, n_bytes=2):
    """
    Convert a floating point buffer into integer values.
    This is primarily useful as an intermediate step in wav output.
    Args:
        x: np.ndarray [dtype=float]
            Floating point data buffer
        n_bytes: int [1, 2, 4]
            Number of bytes per output sample
    Returns:
        x_int : np.ndarray [dtype=int]
            The original buffer cast to integer type.
    """
    if n_bytes not in [1, 2, 4]:
        raise ValueError('n_bytes must be one of {1, 2, 4}')
    # What is the scale of the input data?
    scale = float(1 << ((8 * n_bytes) - 1))
    # Construct a format string
    fmt = '<i{:d}'.format(n_bytes)
    # Rescale and cast the data
    return (x * scale).astype(fmt)


def valid_audio(y, mono=True):
    """
    Validate whether a variable contains valid audio data.
        Args:
        y : np.ndarray
          The input data to validate
        mono : bool
      Wheth:
        valid : bool
            True if all tests pass
    Raises:
        ParameterError
            If `y` fails to meet the following criteria:
                - `type(y)` is `np.ndarray`
                - `mono == True` and `y.ndim` is not 1
                - `mono == False` and `y.ndim` is not 1 or 2
                - `np.isfinite(y).all()` is not True
    """
    if not isinstance(y, np.ndarray):
        raise ValueError('Data must be of type numpy.ndarray')
    if mono and y.ndim != 1:
        raise ValueError('Invalid shape for monophonic audio: '
                         'ndim={:d}, shape={}'.format(y.ndim, y.shape))
    elif y.ndim > 2:
        raise ValueError('Invalid shape for audio: '
                         'ndim={:d}, shape={}'.format(y.ndim, y.shape))
    if not np.isfinite(y).all():
        raise ValueError('Audio buffer is not finite everywhere')
    return True


def normalize(S, norm=np.inf, axis=0):
    """
    Normalize the columns or rows of a matrix
    Args:
        S : np.ndarray [shape=(d, n)]
            The matrix to normalize
        norm : {np.inf, -np.inf, 0, float > 0, None}
            - `np.inf`  : maximum absolute value
            - `-np.inf` : mininum absolute value
            - `0`    : number of non-zeros
            - float  : corresponding l_p norm.
                See `scipy.linalg.norm` for details.
            - None : no normalization is performed
        axis : int [scalar]
            Axis along which to compute the norm.
            `axis=0` will normalize columns, `axis=1` will normalize rows.
            `axis=None` will normalize according to the entire matrix.
    Returns:
        S_norm : np.ndarray [shape=S.shape]
            Normalized matrix
    """
    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S)

    if norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        length = np.sum(mag > 0, axis=axis, keepdims=True)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag ** norm, axis=axis, keepdims=True)**(1./norm)

    elif norm is None:
        return S

    else:
        raise ValueError('Unsupported norm: {}'.format(repr(norm)))

    # Avoid div-by-zero
    SMALL_FLOAT = 1e-20
    length[length < SMALL_FLOAT] = 1.0

    return S / length


def fractional_delay(y, time, sr, mode='conv'):
    """
    Prepend zeros to delay signal by time seconds
    Use linear interpolation through convolution for fractional delay
    """
    samples = time * sr
    if isinstance(samples, int) or samples.is_integer():
        return delay(y, samples)

    if mode == 'upsample':
        # fractional delay by upsampling
        decimals = sum(c != '0' for c in str(round(samples % 1, 4))[2:])
        new_sr = None
        if decimals > 0:
            new_sr = sr * (decimals + 1)
            samples = time * new_sr
            y = resample(y, sr, new_sr)
        y = delay(y, samples)
        if new_sr:
            y = resample(y, new_sr, sr)
    else:
        ref = np.max(np.abs(y))
        f, i = math.modf(samples)
        # integer delay
        y = delay(y, i)
        # linear interpolation for fractional part
        y = np.convolve(y, [f, i-f], "same")
        # normalize back go original max value
        y = y * ref / np.max(np.abs(y))
    return y


def delay(y, N, sr=None):
    """
    Prepend zeros to delay signal by N samples (or N seconds if sr provided)
    """
    if sr:
        N = N * sr
    y = np.pad(
        y,
        pad_width=[round(N), 0],
        mode='constant',
        constant_values=0
    )
    return y


def sum_signals(signals):
    """
    Sum together a list of mono signals
    append zeros to match the longest array
    """
    if not signals:
        return np.array([])
    max_length = max(len(sig) for sig in signals)
    y = np.zeros(max_length)
    for sig in signals:
        padded = np.zeros(max_length)
        padded[0:len(sig)] = sig
        y += padded
    return y


def channel_merge(channels):
    """
    vstack a list of mono signals
    append zeros to match the longest signal
    """
    longest = max(len(channel) for channel in channels)
    padded_channels = []
    for i, channel in enumerate(channels):
        if len(channel) < longest:
            padded = np.zeros(longest)
            padded[0:len(channel)] = channel
            channel = padded
        padded_channels.append(channel)
    y = np.vstack(padded_channels)
    return y
