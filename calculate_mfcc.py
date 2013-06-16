# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from dspy.features import mfcc


def main():
    frame_size = 2048
    num_features = 12
    filename = None
    try:
        filename = sys.argv[1]
        sample_frequency, x = wavfile.read(filename)
        sample_frequency = float(sample_frequency)
        num_frames = x.size // frame_size
    except IndexError:
        sample_frequency = 44100.0
        num_frames = 100
    size = frame_size * num_frames
    dt = 1.0 / sample_frequency
    if filename is None:
        t = np.linspace(0, size * dt, size)
        signal_frequency = 1000.0
        x = np.sin(2 * np.pi * signal_frequency * t) + 0.25 * np.random.rand(size)
    print 'Input signal: %d frames, %d samples at %0.0f Hz' % (
        num_frames,
        size,
        sample_frequency,
    )

    # 2. split signal into frames and calculate MFCC features for each frame
    features = np.zeros((num_frames, num_features))
    for i in range(num_frames):
        idx_begin = i * frame_size
        idx_end = (i + 1) * frame_size
        frame = x[idx_begin:idx_end]
        features[i, :] = mfcc(frame, sample_frequency, num_features=num_features)

    # 3. plot the calculated features (transposed so frame number/time is on X axis)
    x_scale = np.linspace(0, dt * size, num_frames)
    y_scale = np.arange(0, num_features + 1, 1)
    plt.xlim([0, dt * size])
    plt.ylim([0, num_features])
    plt.pcolormesh(x_scale, y_scale, features.T)
    plt.title('MFCC chart')
    plt.xlabel('Time [s]')
    plt.ylabel('Feature number')
    plt.show()



if __name__ == '__main__':
    main()
