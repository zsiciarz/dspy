# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import matplotlib.pyplot as plt
import numpy as np

from dspy.filters import MelFilterBank


def main():
    sample_frequency = 44100.0
    size = 2048
    # 1. generate and plot test signal
    dt = 1.0 / sample_frequency
    t = np.linspace(0, size * dt, size)
    signal_frequency = 200.0
    x = np.sin(2 * np.pi * signal_frequency * t) + 0.25 * np.random.rand(size)
    plt.plot(t, x)
    plt.xlabel('Time [s]')
    plt.ylabel('Sample value')
    plt.title('Input signal')
    plt.show()

    # 2. calculate and plot magnitude spectrum of the signal
    spectrum = np.fft.rfft(x)
    # calculate FFT, discard the Nyquist frequency and compute magnitude
    spectrum = np.abs(spectrum[:-1])
    frequencies = np.linspace(0, sample_frequency / 2, size / 2)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Spectrum magnitude')
    plt.title('Input signal spectrum')
    plt.plot(frequencies, spectrum)
    plt.show()

    # 3. apply Mel-frequency filters to the spectrum and plot output
    filter_bank = MelFilterBank(sample_frequency, size)
    output = filter_bank.apply(spectrum)
    center_frequencies = [mel_filter.center_freq for mel_filter in filter_bank.filters]
    plt.xlabel('Mel filter center frequencies [Hz]')
    plt.ylabel('filter output')
    plt.title('Filter output')
    plt.plot(center_frequencies, output, 'ro')
    plt.show()


if __name__ == '__main__':
    main()
