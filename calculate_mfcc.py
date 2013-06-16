# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import matplotlib.pyplot as plt
import numpy as np

from dspy.features import mfcc


def main():
    sample_frequency = 44100.0
    size = 2048
    # 1. generate and plot test signal
    dt = 1.0 / sample_frequency
    t = np.linspace(0, size * dt, size)
    signal_frequency = 200.0
    x = np.sin(2 * np.pi * signal_frequency * t) + 0.25 * np.random.rand(size)

    # 2. calculate MFCC features
    features = mfcc(x, sample_frequency)
    print features


if __name__ == '__main__':
    main()
