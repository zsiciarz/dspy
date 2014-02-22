# -*- coding: utf-8  -*-

from __future__ import unicode_literals

import numpy as np
from scipy.fftpack import dct

from .filters import MelFilterBank


def mfcc(x, sample_frequency, num_features=12):
    spectrum = np.fft.fft(x)
    # discard the Nyquist frequency and compute magnitude
    spectrum = np.abs(spectrum[:-1])
    filter_bank = MelFilterBank(sample_frequency, x.size)
    filter_output = filter_bank.apply(spectrum)
    # Note: DCT truncating is not implemented in scipy yet, slicing DCT
    # output is an attempt at providing requested number of features;
    # later it may be possible to use the following, correct approach:
    # return dct(filter_output, type=2, n=num_features, norm='ortho')
    return dct(filter_output, type=2, norm='ortho')[:num_features]
