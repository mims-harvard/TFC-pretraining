"Vectorized transformation functions for mobile sensor time series"
import itertools
import numpy as np
import scipy.interpolate

__author__ = "C. I. Tang"
__copyright__ = "Copyright (C) 2020 C. I. Tang"

"""
Based on work of Tang et al.: https://arxiv.org/abs/2011.11542
Contact: cit27@cl.cam.ac.uk
License: GNU General Public License v3.0

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

An re-implemention of
T. T. Um et al., “Data augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural networks,” in Proceedings of the 19th ACM International Conference on Multimodal Interaction, ser. ICMI 2017. New York, NY, USA: ACM, 2017, pp. 216–220.

https://dl.acm.org/citation.cfm?id=3136817

https://arxiv.org/abs/1706.00527

@inproceedings{TerryUm_ICMI2017, author = {Um, Terry T. and Pfister, Franz M. J. and Pichler, Daniel and Endo, Satoshi and Lang, Muriel and Hirche, Sandra and Fietzek, Urban and Kuli\'{c}, Dana}, title = {Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring Using Convolutional Neural Networks}, booktitle = {Proceedings of the 19th ACM International Conference on Multimodal Interaction}, series = {ICMI 2017}, year = {2017}, isbn = {978-1-4503-5543-8}, location = {Glasgow, UK}, pages = {216--220}, numpages = {5}, doi = {10.1145/3136755.3136817}, acmid = {3136817}, publisher = {ACM}, address = {New York, NY, USA}, keywords = {Parkinson\&#39;s disease, convolutional neural networks, data augmentation, health monitoring, motor state detection, wearable sensor}, }

"""

def noise_transform_vectorized(X, sigma=0.05):
    """
    Adding random Gaussian noise with mean 0
    """
    noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + noise

def scaling_transform_vectorized(X, sigma=0.1):
    """
    Scaling by a random factor
    """
    scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[0], 1, X.shape[2]))
    return X * scaling_factor

def rotation_transform_vectorized(X):
    """
    Applying a random 3D rotation
    """
    axes = np.random.uniform(low=-1, high=1, size=(X.shape[0], X.shape[2]))
    angles = np.random.uniform(low=-np.pi, high=np.pi, size=(X.shape[0]))
    matrices = axis_angle_to_rotation_matrix_3d_vectorized(axes, angles)

    return np.matmul(X, matrices)

def axis_angle_to_rotation_matrix_3d_vectorized(axes, angles):
    """
    Get the rotational matrix corresponding to a rotation of (angle) radian around the axes

    Reference: the Transforms3d package - transforms3d.axangles.axangle2mat
    Formula: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    """
    axes = axes / np.linalg.norm(axes, ord=2, axis=1, keepdims=True)
    x = axes[:, 0]; y = axes[:, 1]; z = axes[:, 2]
    c = np.cos(angles)
    s = np.sin(angles)
    C = 1 - c

    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC

    m = np.array([
        [ x*xC+c,   xyC-zs,   zxC+ys ],
        [ xyC+zs,   y*yC+c,   yzC-xs ],
        [ zxC-ys,   yzC+xs,   z*zC+c ]])
    matrix_transposed = np.transpose(m, axes=(2,0,1))
    return matrix_transposed

def negate_transform_vectorized(X):
    """
    Inverting the signals
    """
    return X * -1

def time_flip_transform_vectorized(X):
    """
    Reversing the direction of time
    """
    return X[:, ::-1, :]


def channel_shuffle_transform_vectorized(X):
    """
    Shuffling the different channels
    
    Note: it might consume a lot of memory if the number of channels is high
    """
    channels = range(X.shape[2])
    all_channel_permutations = np.array(list(itertools.permutations(channels))[1:])

    random_permutation_indices = np.random.randint(len(all_channel_permutations), size=(X.shape[0]))
    permuted_channels = all_channel_permutations[random_permutation_indices]
    X_transformed = X[np.arange(X.shape[0])[:, np.newaxis, np.newaxis], np.arange(X.shape[1])[np.newaxis, :, np.newaxis], permuted_channels[:, np.newaxis, :]]
    return X_transformed

def time_segment_permutation_transform_improved(X, num_segments=4):
    """
    Randomly scrambling sections of the signal
    """
    segment_points_permuted = np.random.choice(X.shape[1], size=(X.shape[0], num_segments))
    segment_points = np.sort(segment_points_permuted, axis=1)

    X_transformed = np.empty(shape=X.shape)
    for i, (sample, segments) in enumerate(zip(X, segment_points)):
        # print(sample.shape)
        splitted = np.array(np.split(sample, np.append(segments, X.shape[1])))
        np.random.shuffle(splitted)
        concat = np.concatenate(splitted, axis=0)
        X_transformed[i] = concat
    return X_transformed

def get_cubic_spline_interpolation(x_eval, x_data, y_data):
    """
    Get values for the cubic spline interpolation
    """
    cubic_spline = scipy.interpolate.CubicSpline(x_data, y_data)
    return cubic_spline(x_eval)


def time_warp_transform_improved(X, sigma=0.2, num_knots=4):
    """
    Stretching and warping the time-series
    """
    time_stamps = np.arange(X.shape[1])
    knot_xs = np.arange(0, num_knots + 2, dtype=float) * (X.shape[1] - 1) / (num_knots + 1)
    spline_ys = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[0] * X.shape[2], num_knots + 2))

    spline_values = np.array([get_cubic_spline_interpolation(time_stamps, knot_xs, spline_ys_individual) for spline_ys_individual in spline_ys])

    cumulative_sum = np.cumsum(spline_values, axis=1)
    distorted_time_stamps_all = cumulative_sum / cumulative_sum[:, -1][:, np.newaxis] * (X.shape[1] - 1)

    X_transformed = np.empty(shape=X.shape)
    for i, distorted_time_stamps in enumerate(distorted_time_stamps_all):
        X_transformed[i // X.shape[2], :, i % X.shape[2]] = np.interp(time_stamps, distorted_time_stamps, X[i // X.shape[2], :, i % X.shape[2]])
    return X_transformed

def time_warp_transform_low_cost(X, sigma=0.2, num_knots=4, num_splines=150):
    """
    Stretching and warping the time-series (low cost)
    """
    time_stamps = np.arange(X.shape[1])
    knot_xs = np.arange(0, num_knots + 2, dtype=float) * (X.shape[1] - 1) / (num_knots + 1)
    spline_ys = np.random.normal(loc=1.0, scale=sigma, size=(num_splines, num_knots + 2))

    spline_values = np.array([get_cubic_spline_interpolation(time_stamps, knot_xs, spline_ys_individual) for spline_ys_individual in spline_ys])

    cumulative_sum = np.cumsum(spline_values, axis=1)
    distorted_time_stamps_all = cumulative_sum / cumulative_sum[:, -1][:, np.newaxis] * (X.shape[1] - 1)

    random_indices = np.random.randint(num_splines, size=(X.shape[0] * X.shape[2]))

    X_transformed = np.empty(shape=X.shape)
    for i, random_index in enumerate(random_indices):
        X_transformed[i // X.shape[2], :, i % X.shape[2]] = np.interp(time_stamps, distorted_time_stamps_all[random_index], X[i // X.shape[2], :, i % X.shape[2]])
    return X_transformed