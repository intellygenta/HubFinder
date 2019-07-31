# -*- coding: utf-8 -*-

import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
###############################################################################


# Ignore floating point errors
np.seterr(divide='ignore', invalid='ignore')


def standardize(X, axis=None):
    """
    Standardize time series sequences.
    """
    X -= X.mean(axis=axis)
    X /= X.std(axis=axis)
    return X


def get_ed(X, Y, axis=None, shifting=False, rescaling=False):
    """
    Return Euclidean Distance (ED) between two time series sequences.
    """
    A = X.copy()
    B = Y.copy()
    if shifting:
        A = A - A.mean(axis=axis)
        B = B - B.mean(axis=axis)
    if rescaling:
        A = A / A.std(axis=axis)
        B = B / B.std(axis=axis)
    distance = np.linalg.norm(A - B, axis=axis)
    return distance


def get_mse(X, Y, axis=None):
    """
    Return Mean Square Error (MSE) between two time series sequences.
    """
    return ((X - Y) ** 2).mean(axis=axis)


def get_rmse(X, Y, axis=None):
    """
    Return Root Mean Square Error (RMSE) between two time series sequences.
    """
    return np.sqrt(get_mse(X, Y, axis=axis))
###############################################################################


def get_sliding_statistics(X, window_size):
    length, num_sensors = X.shape
    cumsum = np.concatenate(
            [np.zeros([1, num_sensors]), X.cumsum(axis=0)],
            axis=0)
    cumsum2 = np.concatenate(
            [np.zeros([1, num_sensors]), (X**2).cumsum(axis=0)],
            axis=0)
    norm2s = cumsum2[window_size:] - cumsum2[:-window_size]
    means = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    std2s = norm2s / window_size - means**2
    stds = std2s**0.5
    return means, stds, norm2s


def get_sliding_dot_products(X, Q, batch_size=2**16):
    length = len(X)
    window_size, num_sensors = Q.shape
    if batch_size < length:  # MASSv3
        products = []
        for begin in range(
                0, length - window_size + 1, batch_size - window_size + 1):
            X_batch = X[begin:begin + batch_size]
            products.append(get_sliding_dot_products(X_batch, Q))
        products = np.concatenate(products, axis=0)
    else:
        if window_size < 2 * length:  # MASSv2
            A = np.zeros([length, num_sensors])
            B = np.zeros([length, num_sensors])
        else:  # MASSv1
            A = np.zeros([2 * length, num_sensors])
            B = np.zeros([2 * length, num_sensors])
        A[:window_size] = Q[::-1]
        B[:length] = X
        A = np.fft.fft(A, axis=0)
        B = np.fft.fft(B, axis=0)
        products = np.fft.ifft(A * B, axis=0).real[window_size - 1:length]
    return products


def get_sliding_Euclidean_distances(products, window_size,
                                    means, stds, norm2s,
                                    mean, std, norm2,
                                    shifting=True, rescaling=True,
                                    ):
    if shifting:
        if rescaling:
            distance2s = 2 * window_size - 2 * (
                    products - window_size * means * mean) / (stds * std)
        else:
            distance2s = window_size * (
                    2 * means * mean + stds**2 + std**2) - 2 * products
    else:
        distance2s = norm2s + norm2 - 2 * products
    distance2s = distance2s.sum(axis=1)
    distances = distance2s ** 0.5
    distances[~np.isfinite(distances)] = np.inf
    return distances


def mass(X, Q,
         means=None, stds=None, norm2s=None,
         mean=None, std=None, norm2=None,
         shifting=True, rescaling=True,
         ):
    """MASS

    Compute Euclidean distances between all subsequences in time series X
    and a query subsequence Q using Mueen's Algorithm for Similarity Search (MASS)[#]_

    .. [#] http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html

    Args:
        X (list of numpy.ndarray): Time series
            2D array.
            The first dimension is time axis.
            The second dimension is sensor axis.
        Q (list of numpy.ndarray): Query (subsequence)
            2D array.
            The first dimension is time axis.
            The second dimension is sensor axis.
        shifting (bool)
        rescaling (bool)
    """
    window_size = len(Q)
    assert window_size <= X.shape[0]

    # Compute dot products
    products = get_sliding_dot_products(X, Q)

    # Compute statistics
    if shifting:
        if means is None or stds is None:
            means, stds, norm2s = get_sliding_statistics(X, window_size)
        if mean is None:
            mean = Q.mean(axis=0)
        if std is None:
            std = Q.std(axis=0)
    else:
        if norm2s is None:
            means, stds, norm2s = get_sliding_statistics(X, window_size)
        if norm2 is None:
            norm2 = (Q**2).sum(axis=0)

    # Compute Euclidean distances
    distances = get_sliding_Euclidean_distances(products, window_size,
                                                means, stds, norm2s,
                                                mean, std, norm2,
                                                shifting, rescaling)
    return distances
###############################################################################


def get_lengths_info(Xs):
    lengths = np.array([X.shape[0] for X in Xs])
    ends = lengths.cumsum()
    begins = ends - lengths
    sum_lengths = ends[-1]
    return lengths, begins, ends, sum_lengths


class MatrixProfile:
    def __init__(self, shifting=True, rescaling=True):
        self.shifting = shifting
        self.rescaling = rescaling

    def compute(self, Xs, window_size):
        """
        Compute matrix profile (MP) and matrix profile index (MPI)

        Args:
            Xs (list of numpy.ndarray): Time series
                list of 2D array.
                The first dimension is time axis.
                The second dimension is sensor axis.
            window_size (int): Subsequence length
        """
        self.prepare(Xs, window_size)
        self.compute_matrix_profile()
    ###########################################################################

    def prepare(self, Xs, window_size):
        self.Xs = Xs
        self.window_size = window_size

        # Get time seires shape information
        self.num_data = len(Xs)
        self.num_sensors = Xs[0].shape[1]

        # Get time series length information
        lengths, begins, ends, sum_lengths = get_lengths_info(Xs)
        self.lengths = lengths
        self.begins = begins
        self.ends = ends
        self.sum_lengths = sum_lengths

        # Prepare MP and MPI
        self.matrix_profile = np.full(sum_lengths, np.inf)
        self.matrix_profile_index = np.full(sum_lengths, -1)

        # Get subsequence information
        data = np.full(sum_lengths, -1)
        times = np.full(sum_lengths, -1)
        indices = []
        for datum, (length, begin) in enumerate(zip(lengths, begins)):
            num_windows = length - window_size + 1
            end = begin + num_windows
            data[begin:end] = datum
            times[begin:end] = np.arange(num_windows)
            indices.append(np.arange(num_windows) + begin)
        positions = np.array([data, times]).T
        indices = np.concatenate(indices)
        num_subsequences = len(indices)
        self.positions = positions
        self.indices = indices
        self.num_subsequences = num_subsequences

        # Compute statistics of subsequences
        sliding_statistics = [
                get_sliding_statistics(X, window_size) for X in Xs]
        means, stds, norm2s = zip(*sliding_statistics)
        self.means = list(means)
        self.stds = list(stds)
        self.norm2s = list(norm2s)

        # Initialize the number of iterations
        self.iteration = 0

        # Set the margin for trivial match definition
        self.margin = window_size // 2

    def compute_matrix_profile(
            self, num_iterations=None, num_iterations_rate=None, desc=None):
        """
        Args:
            num_iterations (int): Number of iterations
                If not None, the algorithm will be truncated and return the approximate MP.
                If None, the algorithm will not be truncated and return the exact MP.
            num_iterations_rate (float): Rate of the number of iterations to the maximum one
        """

        # Set the number of iterations
        if num_iterations_rate is not None:
            num_iterations = int(num_iterations_rate * self.num_subsequences)
        if num_iterations is None:
            num_iterations = self.num_subsequences

        # Set the description of progress bar
        if desc is None:
            desc = self.__class__.__name__

        # Compute iteratively
        begin = self.iteration
        end = begin + num_iterations
        for index in tqdm(self.indices[begin:end], desc=desc):
            self.compute_distance_profile(index)
            update = (self.distance_profile < self.matrix_profile)
            self.matrix_profile[update] = self.distance_profile[update]
            self.matrix_profile_index[update] = index
            self.iteration += 1

    def compute_distance_profile(self, index):
        self.distance_profile = self.get_distance_profile(index)
    ###########################################################################

    def get_index(self, datum, time):
        """
        Get a serial number (index) corresponding to a subsequence.

        Args:
            datum (int): which time series sequence?
            time (int): which subsequence in the time series?

        Returns:
            index (int): Serial number of a subsequence
        """
        index = self.begins[datum] + time
        return index

    def get_position(self, index):
        """
        Get a position of a subsequence

        Args:
            index (int): Serial number of a subsequence

        Returns:
            position (tuple of int): position of subsequence
                position = d, t means Xs[d][t:t+window_size]
        """
        position = self.positions[index]
        return position

    def get_subsequence(self, index):
        """
        Get a subsequence

        Args:
            index (int): Serial number of a subsequence

        Returns:
            X (numpy.ndarray): subsequence
        """
        datum, time = self.get_position(index)
        X = self.Xs[datum][time:time + self.window_size]
        return X

    def get_subsequences(self, indices):
        """
        Get subsequences
        """
        Xs = [self.get_subsequence(index) for index in indices]
        return Xs

    def get_distance_profile(self, index, ignore_trivial_matches=True):
        """
        Get DP for a specified subsequence.

        Args:
            index (int): Serial number of a subsequence
            ignore_trivial_matches (bool): Either ignore trivial matches or not
                If True, DP values around the specified subsequence are np.inf.
        """
        i, j = self.get_position(index)

        # Prepare DP
        distance_profile = np.full(self.sum_lengths, np.inf)

        # Compute DP
        for k, (X, begin, end) in enumerate(
                zip(self.Xs, self.begins, self.ends)):
            distances = self.get_distances(i, j, k)
            distance_profile[begin:end - self.window_size + 1] = distances

        # Modify DP values around the specified subsequence
        if ignore_trivial_matches:
            distance_profile = self.ignore_trivial_matches(
                    distance_profile, index)
        else:
            distance_profile[index] = 0.0

        return distance_profile

    def get_distances(self, i, j, k):
        """
        Get Euclidean distances between all subsequences in Xs[k] and a subsequence Q = Xs[i][j:j+window_size]
        """
        X = self.Xs[k]
        Q = self.Xs[i][j:j + self.window_size]

        # Compute dor products
        products = get_sliding_dot_products(X, Q)

        # Compute Euclidean distances
        distances = get_sliding_Euclidean_distances(
                products, self.window_size,
                self.means[k], self.stds[k], self.norm2s[k],
                self.means[i][j], self.stds[i][j], self.norm2s[i][j],
                self.shifting, self.rescaling,
                )

        return distances

    def ignore_trivial_matches(self, distance_profile, index):
            margin = self.margin
            begin = max(0, index - margin)
            end = index + margin + 1
            distance_profile[begin:end] = np.inf
            return distance_profile
    ###########################################################################

    def get_centered_profile(self, profile):
        margin = self.margin
        centered_profile = np.full(profile.shape, np.inf)
        centered_profile[margin:] = profile[:-margin]
        return centered_profile

    def get_centered_matrix_profile(self):
        return self.get_centered_profile(self.matrix_profile)

    def get_centered_distance_profile(self, **kwargs):
        return self.get_centered_profile(self.get_distance_profile(**kwargs))
    ###########################################################################

    def get_local_minima(self, profile, width=None, threshold=np.inf):
        """
        Get local minima of profile (DP or MP)

        Args:
            profile (numpy.ndarray): Profile (DP or MP)
            width (int): Width of window used to detect local minima
            threshold (float): If a profile value is larger than or equal to this threshold,
                it is not trerated as a local minimum.

        Returns:
            indices (list of int): List of serial numbers of local minima
        """
        thick_profile = self.get_thick_profile(profile, width)
        condition = (profile == thick_profile)
        condition[~np.isfinite(profile)] = False
        condition[profile >= threshold] = False
        indices = list(np.where(condition)[0])
        return indices

    def get_thick_profile(self, profile, width=None):
        if width is None:
            width = self.margin
        thick_profile = np.full([2 * width + 1, len(profile)], np.inf)
        thick_profile[0] = profile
        for i in range(1, width + 1):
            thick_profile[i][i:] = profile[:-i]
            thick_profile[i + width][:-i] = profile[i:]
        thick_profile = thick_profile.min(axis=0)
        thick_profile[~np.isfinite(profile)] = np.inf
        return thick_profile
    ###########################################################################

    def get_global_minima(self, profile, threshold=np.inf):
        """
        Get global minima of profile (DP or MP) in ascending order.

        Args:
            profile (numpy.ndarray): Profile (DP or MP)
            threshold (float): Truncate if the profile value is larger than or equal to this threshold

        Returns:
            indices (list of int): List of serial numbers of global minima
        """
        margin = self.margin
        mask = np.zeros(profile.shape)
        indices = []
        while True:
            masked_profile = profile + mask
            index = masked_profile.argmin()
            distance = masked_profile[index]
            if distance < threshold:
                indices.append(index)
                begin = max(0, index - margin)
                end = index + margin + 1
                mask[begin:end] = np.inf
            else:
                break
        return indices

    def get_global_minima_mp(self, threshold=np.inf):
        mp = self.matrix_profile
        return self.get_global_minima(mp, threshold)

    def get_global_minima_dp(self, index, threshold=np.inf):
        dp = self.get_distance_profile(index)
        return self.get_global_minima(dp, threshold)
    ###########################################################################

    def plot_matrix_profile(self, begin=None, end=None, scale=0.01, height=3):
        """
        Visualize MP
        """
        if begin is None:
            begin = 0
        if end is None:
            end = self.sum_lengths
        width = (end - begin) * scale
        fig, ax = plt.subplots(figsize=(width, height))
        ax.plot(self.matrix_profile)
        ax.set_ylabel('matrix profile')
        ax.set_xlim(begin, end)
        plt.show()
    ###########################################################################

    def plot_subsequence(self, index, sensors=None):
        """
        Visualize a subsequence
        """
        if sensors is None:
            sensors = range(self.num_sensors)
        fig = plt.figure(figsize=(4, 3*len(sensors)))
        gs = gridspec.GridSpec(len(sensors), 1)
        X = self.get_subsequence(index)
        for i, sensor in enumerate(sensors):
            ax = fig.add_subplot(gs[i])
            ax.plot(X[:, sensor])
            if i == 0:
                ax.set_title('index={}'.format(index))
            ax.set_ylabel('sensor={}'.format(sensor))
        plt.show()

    def plot_subsequences(self, indices, sensors=None):
        """
        Visualize subsequences
        """
        if sensors is None:
            sensors = range(self.num_sensors)
        fig = plt.figure(figsize=(4*len(indices), 3*len(sensors)))
        gs = gridspec.GridSpec(len(sensors), len(indices))
        for j, index in enumerate(indices):
            X = self.get_subsequence(index)
            for i, sensor in enumerate(sensors):
                ax = fig.add_subplot(gs[i, j])
                ax.plot(X[:, sensor])
                if i == 0:
                    ax.set_title('index={}'.format(index))
                if j == 0:
                    ax.set_ylabel('sensor={}'.format(sensor))
        plt.show()
    ###########################################################################

    def plot_subsequence_detail(self, index, sensors=None):
        """
        Visualize a subsequence in deital.
        # XXX Visualize k-nearest neighbors rather than the single nearest neighbor.
        """
        if sensors is None:
            sensors = range(self.num_sensors)
        fig = plt.figure(figsize=(4*3, 3*len(sensors)))
        gs = gridspec.GridSpec(len(sensors), 3)
        index0 = index
        try:
            index1 = self.matrix_profile_index[index]
            distance = self.matrix_profile[index]
        except AttributeError:
            distance_profile = self.get_distance_profile(index)
            index1 = distance_profile.argmin()
            distance = distance_profile[index1]
        X0 = self.get_subsequence(index0)
        X1 = self.get_subsequence(index1)
        datum0, time0 = self.get_position(index0)
        datum1, time1 = self.get_position(index1)
        begin0, end0 = time0, time0 + self.window_size
        begin1, end1 = time1, time1 + self.window_size
        for i, sensor in enumerate(sensors):
            # Visualize around a subsequence
            ax = fig.add_subplot(gs[i, 0])
            ax.plot(self.Xs[datum0][:, sensor], c='0.5')
            ax.plot(np.arange(begin0, end0), X0[:, sensor], c='tab:blue', lw=5)
            if i == 0:
                ax.set_title('data={}, range=[{}:{}]'.format(
                        datum0, begin0, end0))
            ax.set_ylabel('sensor={}'.format(sensor))
            # Visualize the subsequence and its nearest neighbor
            ax = fig.add_subplot(gs[i, 1])
            ax.plot(X0[:, sensor], c='tab:blue')
            ax.plot(X1[:, sensor], c='tab:green', ls='--')
            if i == 0:
                ax.set_title('index={}, distance={:g}'.format(
                        index0, distance))
            # Visualize around the nearest neighbor
            ax = fig.add_subplot(gs[i, 2])
            ax.plot(self.Xs[datum1][:, sensor], c='0.5')
            ax.plot(np.arange(begin1, end1), X1[:, sensor],
                    c='tab:green', lw=5)
            if i == 0:
                ax.set_title('data={}, range=[{}:{}]'.format(
                        datum1, begin1, end1))
        plt.show()

    def plot_subsequences_detail(self, indices, sensors=None):
        """
        Visualize subsequences in deital.
        """
        for index in indices:
            self.plot_subsequence_detail(index, sensors)
    ###########################################################################


class BruteForceMP(MatrixProfile):
    """
    Brute force algorithm to compute MP
    """
    def get_distances(self, i, j, k):
        X = self.Xs[k]
        Q = self.Xs[i][j:j + self.window_size]
        num_windows = self.lengths[k] - self.window_size + 1
        distances = []
        for l in range(num_windows):
            R = X[l:l + self.window_size]
            distance = get_ed(Q, R, axis=0,
                              shifting=self.shifting,
                              rescaling=self.rescaling,
                              )
            distances.append(distance)
        distances = np.array(distances)
        distances[~np.isfinite(distances)] = np.inf
        return distances


class STAMP(MatrixProfile):
    """
    Scalable Time series Anytime Matrix Profile (STAMP)
    """
    def prepare(self, Xs, window_size, shuffle=False):
        """
        Args:
            shuffle (bool): Either shuffle the order of DP computation or not
        """
        super().prepare(Xs, window_size)

        if shuffle:
            np.random.shuffle(self.indices)


class STOMP(MatrixProfile):
    """
    Scalable Time series Ordered-search Matrix Profile (STOMP)
    """
    def prepare(self, Xs, window_size):
        super().prepare(Xs, window_size)

        # Compute dot product between a leading subsequence of each time series and all subsequences
        self.product_profiles = []
        for i in range(self.num_data):
            product_profile = np.full([self.sum_lengths, self.num_sensors],
                                      np.inf)
            for k, (X, begin, end) in enumerate(
                    zip(self.Xs, self.begins, self.ends)):
                X = self.Xs[k]
                Q = self.Xs[i][:self.window_size]
                products = get_sliding_dot_products(X, Q)
                product_profile[begin:end - self.window_size + 1] = products
            self.product_profiles.append(product_profile.copy())

    def compute_distance_profile(self, index):
        i, j = self.get_position(index)
        window_size = self.window_size

        # Prepare DP
        distance_profile = np.full(self.sum_lengths, np.inf)

        # Compute DP for a leading subsequence of each time series
        if j == 0:
            self.product_profile = self.product_profiles[i].copy()
            for k, (X, begin, end) in enumerate(
                    zip(self.Xs, self.begins, self.ends)):
                products = self.product_profile[begin:end - window_size + 1]
                distances = get_sliding_Euclidean_distances(
                        products, window_size,
                        self.means[k], self.stds[k], self.norm2s[k],
                        self.means[i][0], self.stds[i][0], self.norm2s[i][0],
                        self.shifting, self.rescaling)
                distance_profile[begin:end - window_size + 1] = distances

        # Compute DP for a subsequece except the leading subsequence
        else:
            for k, (X, begin, end) in enumerate(
                    zip(self.Xs, self.begins, self.ends)):
                products = self.product_profile[begin:end - window_size]
                products -= (self.Xs[k][:-window_size] *
                             self.Xs[i][j - 1])
                products += (self.Xs[k][window_size:] *
                             self.Xs[i][j + window_size - 1])
                self.product_profile[begin + 1:end - window_size + 1] = products
                self.product_profile[begin] = self.product_profiles[k][index]
                products = self.product_profile[begin:end - window_size + 1]
                distances = get_sliding_Euclidean_distances(
                        products, window_size,
                        self.means[k], self.stds[k], self.norm2s[k],
                        self.means[i][j], self.stds[i][j], self.norm2s[i][j],
                        self.shifting, self.rescaling)
                distance_profile[begin:end - window_size + 1] = distances

        # Ignore trivial matches
        distance_profile = self.ignore_trivial_matches(distance_profile, index)

        self.distance_profile = distance_profile
###############################################################################
