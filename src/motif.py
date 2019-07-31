# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from sklearn import manifold
from tqdm import tqdm

import metric
###############################################################################


def get_purity(actuals, predictions):
    purity = 0
    for label in np.unique(predictions):
        if label >= 0:
            count = Counter(actuals[predictions == label])
            purity += max(count.values())
    purity /= len(predictions)
    return purity


class MultiDimensionalScaling:
    def __init__(self, markers=None, colors=None):
        if markers is None:
            markers = ['o', '^', 's', '*']
        if colors is None:
            cmap20 = plt.get_cmap('tab20')
            colors20 = cmap20.colors
            colors = colors20[::2] + colors20[1::2]
        self.markers = markers
        self.colors = colors
        self.num_markers = len(markers)
        self.num_colors = len(colors)

    def get_marker(self, i):
        marker = self.markers[i % self.num_markers]
        return marker

    def get_color(self, i):
        color = self.colors[i % self.num_colors]
        return color

    def fit(self, X, actuals=None):
        length, num_sensors = X.shape
        if actuals is None:
            actuals = np.full(length, 0)
        actual_list = np.unique(actuals)
        mds = manifold.MDS()
        Xmds = mds.fit_transform(X)
        self.X = X
        self.length = length
        self.num_sensors = num_sensors
        self.actuals = actuals
        self.actual_list = actual_list
        self.mds = mds
        self.Xmds = Xmds

    def plot(self, predictions=None,
             width=8, height=6,
             title=None,
             labels=None,
             filename=None,
             ticks=False,
             ):
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Visualize in grayscale
        if predictions is None:
            color = 'w'
            edgecolor = 'k'
            if labels is None:
                labels = {}
                for actual in self.actual_list:
                    labels[actual] = 'Cluster {}'.format(actual + 1)
            for actual in self.actual_list:
                marker = self.get_marker(actual)
                Xmds = self.Xmds[self.actuals == actual]
                ax.scatter(Xmds[:, 0], Xmds[:, 1],
                           marker=marker, label=labels[actual],
                           c=color, edgecolors=edgecolor)
            ax.legend()

        # Visualize with colors
        else:
            prediction_list = np.unique(predictions)
            for prediction in prediction_list:
                if prediction >= 0:
                    color = self.get_color(prediction)
                    edgecolor = 'face'
                else:
                    color = 'w'
                    edgecolor = 'k'
                for actual in self.actual_list:
                    marker = self.get_marker(actual)
                    Xmds = self.Xmds[(self.actuals == actual) & (
                            predictions == prediction)]
                    ax.scatter(Xmds[:, 0], Xmds[:, 1], marker=marker,
                               c=color, edgecolors=edgecolor)
        
        if title is not None:
            ax.set_title(title)
        if not ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches='tight')
###############################################################################


class MotifDiscovery:
    def __init__(self, shifting=True, rescaling=True):
        self.mp = metric.MatrixProfile(shifting, rescaling)

    def prepare(self, Xs, window_size):
        self.mp.prepare(Xs, window_size)

    def get_nearest_labels(self, indices, num_motifs=0, bound=None):
        if bound is None:
        #    bound = np.inf
        #    bound = self.mp.margin
            bound = self.mp.window_size
        if len(self.labels) == 0:
            return np.full(len(indices), -1)
        try:
            label = self.labels[num_motifs - 1]
        except IndexError:
            label = self.labels[-1]
        label_indices = np.where(label >= 0)[0].reshape([-1, 1])
        if len(label_indices) == 0:
            return np.full(len(indices), -1)
        distance_matrix = np.absolute(label_indices - indices)
        argmin = distance_matrix.argmin(axis=0)
        nearest_indices = label_indices[argmin].reshape([-1])
        nearest_distances = np.diag(distance_matrix[argmin])
        nearest_labels = label[nearest_indices]
        nearest_labels[nearest_distances > bound] = -1  # TODO >= bound
        return nearest_labels

    def get_majority_labels(self, indices, num_motifs=0, bound=None):
        pass

    def plot_motifs(self, num_motifs=0, sensors=None,
                    begin=None, end=None, xscale=0.01, height=4):
        mp = self.mp
        if sensors is None:
            sensors = range(mp.num_sensors)
        if begin is None:
            begin = mp.begins[0]
        if end is None:
            end = mp.ends[-1]
        width = (end - begin) * xscale
        try:
            label = self.labels[num_motifs - 1]
        except IndexError:
            return
        # Visualize each motif
        for i, motif in enumerate(self.motifs[:num_motifs]):
            print('\n{} motif = {}'.format(i + 1, motif))
            fig = plt.figure(figsize=(4, 3*len(sensors)))
            gs = gridspec.GridSpec(len(sensors), 1)
            # Visualize each sensor
            for j, sensor in enumerate(sensors):
                ax = fig.add_subplot(gs[j])
                ax.set_xlim(begin, end)
            #    ax.set_ylabel('sensor={}'.format(sensor))
            #    if j == 0:
            #        ax.set_title('index={}'.format(motif), loc='left')
                # The original time series
                ax.axvline(0, ls=':', c='0.5')
                for X, begin_, end_ in zip(mp.Xs, mp.begins, mp.ends):
                    if end_ < begin or end < begin_:
                        continue
                    ax.plot(range(begin_, end_), X[:, sensor], c='0.5', lw=1)
                    ax.axvline(end_, ls=':', c='0.5')
                # Emphasize labels
                for motif_ in np.where(label == i)[0]:
                    begin_ = motif_
                    end_ = motif_ + mp.window_size
                    if end_ < begin or end < begin_:
                        continue
                    X = mp.get_subsequence(motif_)
                    ax.plot(range(begin_, end_), X[:, sensor], c='tab:blue')
                # Emphasize motifs
                begin_ = motif
                end_ = motif + mp.window_size
                if end_ < begin or end < begin_:
                    continue
                X = mp.get_subsequence(motif)
                ax.plot(range(begin_, end_), X[:, sensor], c='tab:orange')
            plt.show()

    def plot_labels(self, num_motifs=0, title=None, filename=None,
                    xlabel=None, ylabel='motif',
                    begin=None, end=None,
                    xscale=0.01, yscale=1.0):
        mp = self.mp
        if begin is None:
            begin = mp.begins[0]
        if end is None:
            end = mp.ends[-1]
        width = (end - begin) * xscale
        height = num_motifs * yscale
        try:
            label = self.labels[num_motifs - 1]
        except IndexError:
            return
        fig, ax = plt.subplots(figsize=(width, height))
        # Separate each time series
        ax.axvline(0, ls=':', c='0.5')
        for end_ in mp.ends:
            ax.axvline(end_, ls=':', c='0.5')
        # Visualize each motif
        for i, motif in enumerate(self.motifs[:num_motifs]):
            indices = np.where(label == i)[0]
            position = num_motifs - i - 1
            ax.axhline(position, c='0.5')
            # Plot labels
            ax.plot(indices, np.full(len(indices), position),
                    'o', c='tab:blue')
            # Emphasize motifs
            ax.plot(motif, position, 'o', c='tab:orange')
        ax.set_xlim(begin, end)
        ax.set_ylim(-0.5, num_motifs - 0.5)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        ax.set_yticks(np.arange(num_motifs))
        ax.set_yticklabels(np.arange(num_motifs, 0, -1))
        if title is not None:
            ax.set_title(title)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches='tight')
###############################################################################


class ScanMK(MotifDiscovery):
    """ScanMK

    Enumerate closest-pair motifs.
    Utilize MP(MASS) for efficiency.
    Simplify the algoritm in https://arxiv.org/pdf/1407.3685.pdf
    """
    def compute(self, Xs, window_size, radius):
        # Prepare MP
        self.prepare(Xs, window_size)
        # Compute MP
        self.scan()
        # Enumerate motifs
        self.find(radius)

    def scan(self):
        self.mp.compute_matrix_profile(desc='ScanMK')

    def find(self, radius, max_motifs=np.inf, mask_trivial_matches=True):
        mp = self.mp

        # Initialize
        self.motifs = []
        self.labels = []
        label = np.full(mp.sum_lengths, -1)
        mask = np.full(mp.sum_lengths, np.inf)
        mask[mp.indices] = 0.0

        # Enumerate motifs
        while len(self.motifs) < max_motifs:
            # Get the next motif
            index1 = (mp.matrix_profile + mask).argmin()
            if (mp.matrix_profile + mask)[index1] > radius:
                break
            # Get the closest subsequence of the next motif
            index2 = mp.matrix_profile_index[index1]
            # Get DP of the next motif and the closest subsequence including trivial matches
            distance_profile1 = mp.get_distance_profile(
                    index1, ignore_trivial_matches=False)
            distance_profile2 = mp.get_distance_profile(
                    index2, ignore_trivial_matches=False)
            # Update label
            indices = np.where(
                    (distance_profile1 <= radius) &
                    (distance_profile2 <= radius)
                    )[0]
            label[indices] = len(self.motifs)
            self.labels.append(label.copy())
            # Add a motif
            self.motifs.append(index1)
            # Update mask
            mask[distance_profile1 <= 2 * radius] = np.inf
            mask[distance_profile2 <= 2 * radius] = np.inf
            if mask_trivial_matches:
                for i in range(1, mp.margin + 1):
                    mask[indices + i] = np.inf
                    mask[indices - i] = np.inf
###############################################################################


class SetFinder(MotifDiscovery):
    """SetFinder

    Enumerate range motifs.
    Utilize MP(MASS) for efficiency.
    Simplify the algorithm in https://arxiv.org/pdf/1407.3685.pdf
    """
    def compute(self, Xs, window_size, radius):
        # Prepare
        self.prepare(Xs, window_size)
        # Count the number of subsequences inside the sphere
        self.count([radius])
        # Enumerate motifs
        self.find(radius)

    def count(self, radiuses):
        mp = self.mp

        # Initialize
        counts = {}
        for radius in radiuses:
            counts[radius] = np.zeros(mp.sum_lengths, dtype=int)
        self.counts = counts

        # Count the number of subsequences inside the sphere
        for index in tqdm(mp.indices, desc='SetFinder'):
            # Compute DP and update MP
            mp.compute_distance_profile(index)
            update = (mp.distance_profile < mp.matrix_profile)
            mp.matrix_profile[update] = mp.distance_profile[update]
            mp.matrix_profile_index[update] = index
            mp.iteration += 1
            # Count the number of subsequences inside the sphere for each radius
            for radius in radiuses:
                self.counts[radius][mp.distance_profile <= radius] += 1

    def find(self, radius, max_motifs=np.inf, mask_trivial_matches=True):
        self.radius = radius
        mp = self.mp
        counts = self.counts[radius]

        # Initialize
        self.motifs = []
        self.labels = []
        label = np.full(mp.sum_lengths, -1)
        mask = np.full(mp.sum_lengths, -1)
        mask[mp.indices] = 1

        # Enumerate motifs
        while len(self.motifs) < max_motifs:
            # Get the next motif
            index = (counts * mask).argmax()
            if counts[index] <= 0:
                break
            # Get DP of the next motif including trivial matches
            distance_profile = mp.get_distance_profile(
                    index, ignore_trivial_matches=False)
            # Update label
            indices = np.where(distance_profile <= radius)[0]
            label[indices] = len(self.motifs)
            self.labels.append(label.copy())
            # Add a motif
            self.motifs.append(index)
            # Update mask
            mask[distance_profile <= 2 * radius] = -1
            if mask_trivial_matches:
                for i in range(1, mp.margin + 1):
                    mask[indices + i] = -1
                    mask[indices - i] = -1
###############################################################################


class HubFinder(MotifDiscovery):
    """HubFinder

    Enumerate hub motifs.
    Utilize MP(STOMP) for efficiency.
    """
    def __init__(self, shifting=True, rescaling=True):
        self.mp = metric.STOMP(shifting, rescaling)

    def compute(self, Xs, window_size, max_motifs=None, threshold=np.inf):
        # Prepare MP
        self.prepare(Xs, window_size)
        # Refine candidates into significant motifs
        self.refine(max_motifs)
        # Sort motifs in order of significance
        self.sort()
        # Extract label information
        self.extract(threshold)

    def refine(self, max_motifs=None, width=None):
        """
        Refine motif candidates into significant motifs
        """
        mp = self.mp
        if max_motifs is None:
            max_motifs = mp.window_size
        if width is None:
            width = mp.margin

        # Initialize
        self.candidates = []
        self.cost_profiles = []
        self.anchors = []
        matrix_profile = np.full(mp.sum_lengths, np.inf)
        distance_profiles = np.full(mp.sum_lengths, None)

        # Flag of the end of MP
        profile_ends = np.full(mp.sum_lengths, False)
        profile_ends[mp.ends - mp.window_size] = True

        # Refine motifs
        for index in tqdm(mp.indices, desc='HubFinder'):
            # Compute DP and update MP
            mp.compute_distance_profile(index)
            update = (mp.distance_profile < mp.matrix_profile)
            mp.matrix_profile[update] = mp.distance_profile[update]
            mp.matrix_profile_index[update] = index
            mp.iteration += 1
            # Compute MP online to detect local minima
            matrix_profile[index] = mp.distance_profile.min()
            # Store the latest DP
            distance_profiles[index] = mp.distance_profile
            # Skip the head of time series
            if index < width:
                continue
            # Detect local minimum
            center = index - width
            if profile_ends[index]:
                center += matrix_profile[index - width:index + 1].argmin()
            # Skip if not local minimum
            left = max(center - width, 0)
            right = center + width + 1
            if matrix_profile[left:right].argmin() != center - left:
                continue
            # Add local minimum to candidates
            candidate = center
            # Delete DPs no longer needed
            distance_profiles[:center] = None
            # Add local minimum to anchors
            self.anchors.append(candidate)
            # Skip if the cost cannot be improved any more
            distance_profile = distance_profiles[candidate]
            if len(self.cost_profiles) > 0:
                prev_profile = np.array(self.cost_profiles).min(axis=0)
                next_profile = np.array(
                        [prev_profile, distance_profile]).min(axis=0)
                prev_cost = self.get_cost(prev_profile)
                next_cost = self.get_cost(next_profile)
                if prev_cost <= next_cost:
                    continue
                self.prev_cost = prev_cost
            # Add to candidates
            self.candidates.append(candidate)
            # Store profile for computing the cost
            cost_profile = distance_profile.copy()
            cost_profile[candidate] = matrix_profile[candidate]
            self.cost_profiles.append(cost_profile)
            # If cadidates is larger than maximum number of motifs
            if len(self.candidates) > max_motifs:
                # Remove the least significant candidate
                position = self.get_delete_position()
                del self.candidates[position]
                del self.cost_profiles[position]

    def sort(self):
        """
        Sort motifs in order of significance
        """
        # Copy
        candidates = self.candidates[:]
        cost_profiles = self.cost_profiles[:]
        # Sort in ascending order
        self.motifs = []
        num_motifs = len(candidates)
        while num_motifs > 0:
            if num_motifs > 1:
                min_cost = np.inf
                for i in range(num_motifs):
                    profiles = cost_profiles[:i]
                    profiles += cost_profiles[i + 1:]
                    profile = np.array(profiles).min(axis=0)
                    cost = self.get_cost(profile)
                    if cost < min_cost:
                        position = i
                        min_cost = cost
            else:
                position = 0
            motif = candidates[position]
            self.motifs.append(motif)
            del candidates[position]
            del cost_profiles[position]
            num_motifs -= 1
        # Sort in descending order
        self.motifs.reverse()

    def extract(self, threshold=None):  # XXX associate (motifs and labels)
        mp = self.mp
        # Set threshold of motif set (labels)
        if threshold is None:
            # threshold = np.inf
            threshold = np.sqrt(self.mp.window_size)
        # Compute DP for each motif
        self.distance_profiles = []
        for motif in self.motifs:
            distance_profile = mp.get_distance_profile(
                    motif, ignore_trivial_matches=False)
            self.distance_profiles.append(distance_profile)

        # Extract label information
        sub_matrix_profile = np.full(mp.sum_lengths, np.inf)
        sub_matrix_profile_index = np.full(mp.sum_lengths, -1)
        label = np.full(mp.sum_lengths, -1)
        self.labels = []
        for i in range(len(self.motifs)):
            distance_profile = self.distance_profiles[i]
            update = (distance_profile < sub_matrix_profile)
            sub_matrix_profile[update] = distance_profile[update]
            sub_matrix_profile_index[
                    update & (distance_profile < threshold)] = i
            local_minima = mp.get_local_minima(sub_matrix_profile)
            local_minima = np.array(local_minima)
            local_minima = local_minima[
                    sub_matrix_profile_index[local_minima] == i]
            for index in local_minima:
                begin = max(0, index - mp.margin + 1)
                end = index + mp.margin
                label[begin:end] = -1
                label[index] = i
            self.labels.append(label.copy())
        # Extract label information  # XXX this is more simple but unconfirmed
        '''
        self.labels = []
        for num_motifs in range(1, len(self.motifs) + 1):
            distance_profiles = np.array(self.distance_profiles[:num_motifs])
            sub_matrix_profile = distance_profiles.min(axis=0)
            sub_matrix_profile_index = distance_profiles.argmin(axis=0)
            local_minima = mp.get_local_minima(sub_matrix_profile)
            label = np.full(mp.sum_lengths, -1)
            label[local_minima] = sub_matrix_profile_index[local_minima]
            self.labels.append(label)
            '''

    def get_delete_position(self):
        num_candidates = len(self.candidates)
        min_cost = self.prev_cost
        position = num_candidates - 1
        for i in range(num_candidates - 1):
            profiles = self.cost_profiles[:i]
            profiles += self.cost_profiles[i + 1:]
            profile = np.array(profiles).min(axis=0)
            cost = self.get_cost(profile)
            if cost < min_cost:
                position = i
                min_cost = cost
        return position

    def get_cost(self, profile):
        thick_profile = self.mp.get_thick_profile(profile)
        costs = thick_profile[self.anchors]
        cost = costs[np.isfinite(costs)].sum()
        return cost

    def plot_sub_matrix_profile(self, num_motifs=None,
                                begin=None, end=None, scale=0.01, height=3):
        if num_motifs is None:
            num_motifs = len(self.motifs)
        if begin is None:
            begin = 0
        if end is None:
            end = self.mp.sum_lengths
        width = (end - begin) * scale
        fig, ax = plt.subplots(figsize=(width, height))
        label = self.labels[num_motifs - 1]
        for i in range(num_motifs):
            indices = np.where(label == i)[0]
            ax.plot(self.distance_profiles[i])
            ax.plot(indices, self.distance_profiles[i][indices], 'v', c='k')
        ax.plot(self.mp.matrix_profile, c='k')
        ax.set_ylabel('sub matrix profile')
        ax.set_xlim(begin, end)
        plt.show()
###############################################################################
