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
