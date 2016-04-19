#!/usr/bin/env python
import argparse
import math
import logging

import numpy as np

import audio

logger = logging.getLogger(__name__)


def process_file(audio_path, output, spkr_to_spkr, lstnr_to_spkr, ear_to_ear):
    """
    Read stereo binaural audio file and write wav file with crosstalk 'removed'
    """
    y, sr = audio.load(audio_path, mono=False)
    left = y[0]
    right = y[1]

    d1, d2 = compute_distances(spkr_to_spkr, lstnr_to_spkr, ear_to_ear)
    logger.debug(d1)
    logger.debug(d2)
    l_left, l_right = cancel_crosstalk(left, d1, d2, sr)
    r_right, r_left = cancel_crosstalk(left, d1, d2, sr)
    left = audio.sum_signals([l_left, r_left, left])
    right = audio.sum_signals([l_right, r_right, right])

    y = audio.channel_merge([left, right])
    audio.write_wav(output, y, sr, norm=False)


def cancel_crosstalk(signal, d1, d2, sr):
    c = 343.2
    delta_d = abs(d2 - d1)
    logger.debug(delta_d)
    time_delay = delta_d / c
    attenuation = (d1) / (d2)
    ref = np.max(signal)
    logger.debug(attenuation)
    cancel_sigs = recursive_cancel(signal, ref, time_delay, attenuation, sr)
    cancel_sigs = list(cancel_sigs)
    contralateral = audio.sum_signals(cancel_sigs[0::2])
    ipsilateral = audio.sum_signals(cancel_sigs[1::2])
    return ipsilateral, contralateral


def recursive_cancel(sig, ref, time, attenuation, sr, threshold_db=-10):
    cancel = invert(audio.delay(sig, time, sr)) * attenuation

    db = 20 * math.log10(np.max(cancel) / ref)
    logger.debug(db)
    if db < threshold_db:
        return cancel
    else:
        yield cancel
        yield from recursive_cancel(cancel, ref, time, attenuation, sr)


def compute_distances(spkr_to_spkr, lstnr_to_spkr, ear_to_ear):
    S = spkr_to_spkr / 2
    L = lstnr_to_spkr
    r = ear_to_ear / 2
    theta = math.acos(S / (math.sqrt(L**2 + S**2)))
    delta_d = r * (math.pi - 2*theta)
    d1 = math.sqrt(L**2 + (S-r)**2)
    d2 = d1 + delta_d

    return d1, d2


def invert(x):
    y = x * -1
    return y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Cancel crosstalk in a stereo binaural audio signal."
    )
    parser.add_argument('audio_path', type=str,
                        help='Path to input audio file')
    parser.add_argument('output', type=str,
                        help='Output file')
    parser.add_argument('-s', '--spkr_to_spkr', type=float, default=0.3048,
                        help='Distance between speakers in meters')
    parser.add_argument('-l', '--lstnr_to_spkr', type=float, default=0.5588,
                        help='Distance listener is from speakers in meters')
    parser.add_argument('-e', '--ear_to_ear', type=float, default=0.215,
                        help='Distance between ears (diameter of head) in  meters')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print debug messages to stdout')
    args = parser.parse_args()

    import logging.config
    logging.config.fileConfig('logging.ini', disable_existing_loggers=False)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose debugging activated")
    del args.verbose

    process_file(**vars(args))
