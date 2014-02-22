# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import matplotlib.pyplot as plt
import numpy as np

from dspy.filters import MelFilterBank


def main():
    sample_frequency = 44100
    size = 2048
    filter_bank = MelFilterBank(sample_frequency, size)
    frequencies = np.linspace(0, sample_frequency, size, endpoint=False)
    for mel_filter in filter_bank.filters:
        plt.plot(frequencies, mel_filter.spectrum)
    plt.xlim([0, sample_frequency / 2])
    plt.xlabel('Frequency (linear scale)')
    plt.ylabel('Filter spectrum magnitude')
    plt.title('Mel filter spectra')
    plt.show()


if __name__ == '__main__':
    main()
