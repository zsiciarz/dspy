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

    def __init__(self, sample_frequency, filter_num, mel_filter_width, size):
        self.sample_frequency = float(sample_frequency)
        self.nonzero_samples = 1
        mel_min_freq = filter_num * mel_filter_width / 2.0
        mel_center_freq = mel_min_freq + mel_filter_width / 2.0
        mel_max_freq = mel_min_freq + mel_filter_width
        self.min_freq = mel_to_linear(mel_min_freq)
        self.center_freq = mel_to_linear(mel_center_freq)
        self.max_freq = mel_to_linear(mel_max_freq)
        self.generate_spectrum(size)

    def generate_spectrum(self, size):
        self.spectrum = np.zeros(size)
        min_pos = int(size * self.min_freq / self.sample_frequency)
        max_pos = int(size * self.max_freq / self.sample_frequency)
        max_pos = min(max_pos, size - 1)
        if max_pos <= min_pos:
            return
        self.nonzero_samples = max_pos - min_pos
        ascending_range = self.center_freq - self.min_freq
        descending_range = self.max_freq - self.center_freq
        for k in range(min_pos, max_pos + 1):
            current_freq = k * self.sample_frequency / size
            if current_freq < self.min_freq:
                continue
            if current_freq < self.center_freq:
                self.spectrum[k] = (current_freq - self.min_freq) / ascending_range
            elif current_freq < self.max_freq:
                self.spectrum[k] = (self.max_freq - current_freq) / descending_range

    def apply(self, x):
        return self.spectrum.dot(x) / float(self.nonzero_samples)


class MelFilterBank(object):
    """
    A bank of several Mel-frequency filters.
    """

    def __init__(self, sample_frequency, size, mel_filter_width=200.0, bank_size=24):
        self.sample_frequency = sample_frequency
        self.size = size
        self.filters = []
        for i in range(bank_size):
            mel_filter = MelFilter(self.sample_frequency, i, mel_filter_width, self.size)
            self.filters.append(mel_filter)

    def apply(self, x):
        padded = np.zeros(self.size)
        padded[0:x.size] = x
        return np.array([mel_filter.apply(padded) for mel_filter in self.filters])
