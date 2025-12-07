"Vectorized transformation functions for mobile sensor time series"
import itertools
import numpy as np
import scipy.interpolate

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


def time_segment_permutation_transform_improved(X):
    # X shape: (window, channels) or (channels, window)
    sample = X.copy()

    # ensure shape = (window, channels)
    if sample.shape[0] < sample.shape[1]:
        sample = sample.T

    T = sample.shape[0]

    # choose 5 random segment boundaries
    pts = np.sort(np.random.choice(np.arange(1, T), size=5, replace=False))
    pts = np.concatenate(([0], pts, [T]))

    parts = []
    for i in range(len(pts) - 1):
        start = pts[i]
        end = pts[i + 1]
        parts.append(sample[start:end, :])

    # shuffle segments
    np.random.shuffle(parts)

    out = np.concatenate(parts, axis=0)

    # if original was transposed, restore shape
    if X.shape[0] < X.shape[1]:
        out = out.T

    return out
