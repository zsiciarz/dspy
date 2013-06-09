# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import matplotlib.pyplot as plt
import numpy as np

from dspy.filters import MelFilter


def main():
    sample_frequency = 44100
    size = 2048
    f = MelFilter(sample_frequency)
    f.create_filter(9, 200, size)
    frequencies = np.linspace(0, sample_frequency, size)
    plt.plot(frequencies, f.spectrum)
    plt.xlim([0, sample_frequency / 2])
    plt.show()


if __name__ == '__main__':
    main()
