# -*- coding: utf-8  -*-

from __future__ import unicode_literals

import numpy as np


def linear_to_mel(frequency):
    """
    Given a frequency in the linear scale, returns the corresponding frequency
    in the Mel scale.
    """
    return 1127.01048 * np.log(1.0 + frequency / 700.0)


def mel_to_linear(frequency):
    """
    Given a frequency in the Mel scale, returns the corresponding frequency
    in the linear scale.
    """
    return 700.0 * (np.exp(frequency / 1127.01048) - 1.0)


class MelFilter(object):
    """
    A single Mel-frequency filter.
    """

    def __init__(self, sample_frequency):
        self.sample_frequency = float(sample_frequency)

    def create_filter(self, filter_num, mel_filter_width, size):
        mel_min_freq = filter_num * mel_filter_width / 2.0
        mel_center_freq = mel_min_freq + mel_filter_width / 2.0
        mel_max_freq = mel_min_freq + mel_filter_width
        min_freq = mel_to_linear(mel_min_freq)
        center_freq = mel_to_linear(mel_center_freq)
        max_freq = mel_to_linear(mel_max_freq)
        self.generate_spectrum(min_freq, center_freq, max_freq, size)

    def generate_spectrum(self, min_freq, center_freq, max_freq, size):
        self.spectrum = np.zeros(size)
        min_pos = int(size * min_freq / self.sample_frequency)
        max_pos = int(size * max_freq / self.sample_frequency)
        max_pos = min(max_pos, size)
        for k in range(min_pos, max_pos + 1):
            current_freq = k * self.sample_frequency / size
            if current_freq < min_freq:
                continue
            if current_freq < center_freq:
                self.spectrum[k] = (current_freq - min_freq) / (center_freq - min_freq)
            elif current_freq < max_freq:
                self.spectrum[k] = (max_freq - current_freq) / (max_freq - center_freq)


class MelFilterBank(object):
    """
    A bank of several Mel-frequency filters.
    """

    def __init__(self, sample_frequency, size, mel_filter_width=200.0, bank_size=24):
        self.sample_frequency = sample_frequency
        self.size = size
        self.filters = []
        for i in range(bank_size):
            mel_filter = MelFilter(self.sample_frequency)
            mel_filter.create_filter(i, mel_filter_width, self.size)
            self.filters.append(mel_filter)
