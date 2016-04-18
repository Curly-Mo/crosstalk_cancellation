#!/usr/bin/env python
import argparse
import math

import numpy as np

import audio


def process_file(audio_path, output, spkr_to_spkr, lstnr_to_spkr, ear_to_ear):
    """
    Read stereo binaural audio file and write wav file with crosstalk 'removed'
    """
    y, sr = audio.load(audio_path, mono=False)
    left = y[0]
    right = y[1]

    delta_d, d1 = compute_distances(spkr_to_spkr, lstnr_to_spkr, ear_to_ear)
    print(delta_d)
    print(d1)
    l_left, l_right = cancel_crosstalk(left, delta_d, sr)
    r_right, r_left = cancel_crosstalk(left, delta_d, sr)
    print(l_left)
    print(l_left.shape)
    print(r_left)
    print(r_left.shape)
    left = sum_signals([l_left, r_left, left])
    right = sum_signals([l_right, r_right, right])

    y = channel_merge([left, right])
    audio.write_wav(output, y, sr, norm=False)


def cancel_crosstalk(signal, delta_d, sr):
    c = 343.2
    time_delay = delta_d / c
    print(time_delay)
    ref = np.max(signal)
    cancel_sigs = recursive_cancel(signal, ref, time_delay, sr)
    cancel_sigs = list(cancel_sigs)
    contralateral = sum_signals(cancel_sigs[0::2])
    ipsilateral = sum_signals(cancel_sigs[1::2])
    return ipsilateral, contralateral


def recursive_cancel(sig, ref, time, sr, threshold_db=-40):
    cancel = invert(delay(sig, time, sr)) * 0.5

    db = 20 * math.log10(np.max(cancel) / ref)
    print(db)
    if db < threshold_db:
        return cancel
    else:
        yield cancel
        yield from recursive_cancel(cancel, ref, time, sr)


def compute_distances(spkr_to_spkr, lstnr_to_spkr, ear_to_ear):
    S = spkr_to_spkr / 2
    L = lstnr_to_spkr
    r = ear_to_ear / 2
    theta = math.acos(S / (math.sqrt(L**2 + S**2)))
    print(theta)
    delta_d = r * (math.pi - 2*theta)
    d1 = math.sqrt(L**2 + (S-r)**2)

    return delta_d, d1


def invert(x):
    y = x * -1
    return y


def delay(y, time, sr):
    """
    Prepend zeros to delay signal by time seconds
    Upsample, delay, then resample if delay is a fractional # of samples
    """
    sampletime = time * sr
    decimals = sum(c != '0' for c in str(round(sampletime % 1, 4))[2:])
    new_sr = None
    if decimals > 0:
        new_sr = sr * (decimals + 1)
        sampletime = time * new_sr
        y = audio.resample(y, sr, new_sr)
    y = np.pad(
        y,
        pad_width=[round(sampletime), 0],
        mode='constant',
        constant_values=0
    )
    if new_sr:
        y = audio.resample(y, new_sr, sr)
    return y


def sum_signals(signals):
    max_length = max(len(sig) for sig in signals)
    y = np.zeros(max_length)
    for sig in signals:
        padded = np.zeros(max_length)
        padded[0:len(sig)] = sig
        y += padded
    return y


def channel_merge(channels):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Cancel crosstalk in a stereo binaural audio signal."
    )
    parser.add_argument('audio_path', type=str,
                        help='Path to input audio file')
    parser.add_argument('output', type=str,
                        help='Output file')
    parser.add_argument('-s', '--spkr_to_spkr', type=float, default=0.26,
                        help='Distance between speakers in meters')
    parser.add_argument('-l', '--lstnr_to_spkr', type=float, default=0.5,
                        help='Distance listener is from speakers in meters')
    parser.add_argument('-e', '--ear_to_ear', type=float, default=0.215,
                        help='Distance between ears (diameter of head) in  meters')
    args = parser.parse_args()

    process_file(**vars(args))
