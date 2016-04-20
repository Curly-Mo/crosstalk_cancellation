#!/usr/bin/env python
import argparse
import math
import logging

import numpy as np
import scipy.signal

import audio

logger = logging.getLogger(__name__)


def process_file(audio_path, output, spkr_to_spkr, lstnr_to_spkr, ear_to_ear):
    """
    Read stereo binaural audio file and write wav file with crosstalk 'removed'
    """
    logger.info('Loading file into memory: {}'.format(audio_path))
    y, sr = audio.load(audio_path, mono=False, sr=44100)
    left = y[0]
    right = y[1]

    logger.info('Computing distance from speaker to each ear')
    d1, d2, theta = compute_geometry(spkr_to_spkr, lstnr_to_spkr, ear_to_ear)
    logger.debug('d1: {}'.format(d1))
    logger.debug('d2: {}'.format(d2))
    logger.debug('theta: {}'.format(theta))

    headshadow = headshadow_filter_coefficients(theta, ear_to_ear/2, sr)
    logger.debug('headshadow b: {} a: {}'.format(*headshadow))

    logger.info('Computing recursive crosstalk cancellation for left channel')
    l_left, l_right = cancel_crosstalk(left, d1, d2, headshadow, sr)
    logger.info('Computing recursive crosstalk cancellation for right channel')
    r_right, r_left = cancel_crosstalk(left, d1, d2, headshadow, sr)

    left = audio.sum_signals([l_left, r_left, left])
    right = audio.sum_signals([l_right, r_right, right])

    y = audio.channel_merge([left, right])
    logger.info('Writing output to: {}'.format(output))
    audio.write_wav(output, y, sr, norm=True)


def cancel_crosstalk(signal, d1, d2, headshadow, sr):
    c = 343.2
    delta_d = abs(d2 - d1)
    logger.debug('delta_d: {}'.format(delta_d))
    time_delay = delta_d / c
    attenuation = (d1) / (d2)
    # Reference max amplitude
    ref = np.max(np.abs(signal))
    logger.debug('attenuation factor: {}'.format(attenuation))
    logger.debug('delay amount: {}'.format(time_delay))
    cancel_sigs = recursive_cancel(signal, ref, time_delay, attenuation, headshadow, sr)
    cancel_sigs = list(cancel_sigs)
    contralateral = audio.sum_signals(cancel_sigs[0::2])
    ipsilateral = audio.sum_signals(cancel_sigs[1::2])
    return ipsilateral, contralateral


def recursive_cancel(sig, ref, time, attenuation, headshadow, sr, threshold_db=-70):
    # delay and invert
    cancel_sig = invert(audio.fractional_delay(sig, time, sr))
    # apply headshadow filter (lowpass based on theta)
    cancel_sig = scipy.signal.filtfilt(*headshadow, cancel_sig)
    # attenuate
    cancel_sig = cancel_sig * attenuation

    # Recurse until rms db is below threshold
    db = 20 * math.log10(np.max(np.abs(cancel_sig)) / ref)
    logger.debug('{} dB'.format(db))
    if db < threshold_db:
        return cancel_sig
    else:
        yield cancel_sig
        yield from recursive_cancel(cancel_sig, ref, time, attenuation, headshadow, sr)


def headshadow_filter_coefficients(theta, r, sr):
    theta = theta + math.pi/2
    theta0 = 2.618
    alpha_min = 0.5
    c = 343.2
    w0 = c / r
    alpha = 1 + alpha_min/2 + (1-alpha_min/2)*math.cos(theta*math.pi/theta0)
    b = [(alpha+w0/sr)/(1+w0/sr), (-alpha+w0/sr)/(1+w0/sr)]
    a = [1, -(1-w0/sr)/(1+w0/sr)]
    return b, a



def compute_geometry(spkr_to_spkr, lstnr_to_spkr, ear_to_ear):
    S = spkr_to_spkr / 2
    L = lstnr_to_spkr
    r = ear_to_ear / 2
    theta = math.acos(S / (math.sqrt(L**2 + S**2)))
    delta_d = r * (math.pi - 2*theta)
    d1 = math.sqrt(L**2 + (S-r)**2)
    d2 = d1 + delta_d

    # angle from center of head to speaker (used for computing headshadow)
    theta = math.atan(S / L)

    return d1, d2, theta


def rms(sig):
    return np.sqrt(np.mean(sig**2))


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
