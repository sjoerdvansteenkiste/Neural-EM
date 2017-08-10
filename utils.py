#!/usr/bin/env python
# coding=utf-8

from __future__ import (print_function, division, absolute_import, unicode_literals)

import os
import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.colors import hsv_to_rgb
from sklearn.metrics import adjusted_mutual_info_score


ACTIVATION_FUNCTIONS = {
    'sigmoid': tf.nn.sigmoid,
    'tanh': tf.nn.tanh,
    'relu': tf.nn.relu,
    'elu': tf.nn.elu,
    'linear': lambda x: x,
}


def parse_activation_function(name_list):
    return [ACTIVATION_FUNCTIONS[name] for name in name_list]


def save_image(filename, image_array):
    import scipy.misc
    if image_array.shape[2] == 1:
        if np.min(image_array) >= 0.:
            scipy.misc.toimage(image_array[:, :, 0], cmin=0.0, cmax=1.0).save(filename)
        else:
            scipy.misc.toimage(image_array[:, :, 0], cmin=-1.0, cmax=1.0).save(filename)
    else:
        scipy.misc.toimage(255*image_array).save(filename)


def delete_files(folder, recursive=False):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif recursive and os.path.isdir(file_path):
                delete_files(file_path, recursive)
                os.unlink(file_path)
        except Exception as e:
            print(e)


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def print_vars(_vars):
    total_n_vars = 0
    for var in _vars:
        sh = var.get_shape().as_list()
        total_n_vars += np.prod(sh)

        print(var.name, sh)

    print(total_n_vars, 'total variables')

    return total_n_vars


def evaluate_groups_seq(true_groups, predicted, weights):
    """ Compute the weighted AMI score and corresponding mean confidence for given gammas.
    :param true_groups: (T, B, 1, W, H, 1)
    :param predicted: (T, B, K, W, H, 1)
    :param weights: (T)
    :return: scores, confidences (B,)
    """
    w_scores, w_confidences = 0., 0.
    assert true_groups.ndim == predicted.ndim == 6, true_groups.shape

    for t in range(true_groups.shape[0]):
        scores, confidences = evaluate_groups(true_groups[t], predicted[t])

        w_scores += weights[t] * np.array(scores)
        w_confidences += weights[t] * np.array(confidences)

    norm = np.sum(weights)

    return w_scores/norm, w_confidences/norm


def evaluate_groups(true_groups, predicted):
    """ Compute the AMI score and corresponding mean confidence for given gammas.
    :param true_groups: (B, 1, W, H, 1)
    :param predicted: (B, K, W, H, 1)
    :return: scores, confidences (B,)
    """
    scores, confidences = [], []
    assert true_groups.ndim == predicted.ndim == 5, true_groups.shape
    batch_size, K = predicted.shape[:2]
    true_groups = true_groups.reshape(batch_size, -1)
    predicted = predicted.reshape(batch_size, K, -1)
    predicted_groups = predicted.argmax(1)
    predicted_conf = predicted.max(1)
    for i in range(batch_size):
        true_group = true_groups[i]
        idxs = np.where(true_group != 0.0)[0]
        scores.append(adjusted_mutual_info_score(true_group[idxs], predicted_groups[i, idxs]))
        confidences.append(np.mean(predicted_conf[i, idxs]))

    return scores, confidences


def tf_adjusted_rand_index(groups, gammas, iter_weights):
    """
    Inputs:
        groups: shape=(T, B, 1, W, H, 1)
            These are the masks as stored in the hdf5 files
        gammas: shape=(T, B, K, W, H, 1)
            These are the gammas as predicted by the network
    """
    with tf.name_scope('ARI'):
        # ignore first iteration
        groups = groups[1:]
        gammas = gammas[1:]
        # reshape gammas and convert to one-hot
        yshape = tf.shape(gammas)
        gammas = tf.reshape(gammas, shape=tf.stack([yshape[0] * yshape[1], yshape[2], yshape[3] * yshape[4] * yshape[5]]))
        Y = tf.one_hot(tf.argmax(gammas, axis=1), depth=yshape[2], axis=1)
        # reshape masks
        gshape = tf.shape(groups)
        groups = tf.reshape(groups, shape=tf.stack([gshape[0] * gshape[1], 1, gshape[3] * gshape[4] * gshape[5]]))
        G = tf.one_hot(tf.cast(groups[:, 0], tf.int32), depth=tf.cast(tf.reduce_max(groups) + 1, tf.int32), axis=1)
        # now Y and G both have dim (B*T, K, N) where N=W*H*C
        # mask entries with group 0
        M = tf.cast(tf.greater(groups, 0.5), tf.float32)
        n = tf.cast(tf.reduce_sum(M, axis=[1, 2]), tf.float32)
        DM = G * M
        YM = Y * M
        # contingency table for overlap between G and Y
        nij = tf.einsum('bij,bkj->bki', YM, DM)
        a = tf.reduce_sum(nij, axis=1)
        b = tf.reduce_sum(nij, axis=2)
        # rand index
        rindex = tf.cast(tf.reduce_sum(nij * (nij-1), axis=[1, 2]), tf.float32)
        aindex = tf.cast(tf.reduce_sum(a * (a-1), axis=1), tf.float32)
        bindex = tf.cast(tf.reduce_sum(b * (b-1), axis=1), tf.float32)
        expected_rindex = aindex * bindex / (n*(n-1) + 1e-6)
        max_rindex = (aindex + bindex) / 2
        ARI = (rindex - expected_rindex)/tf.clip_by_value(max_rindex - expected_rindex, 1e-6, 1e6)
        ARI = tf.reshape(ARI, shape=(yshape[0], yshape[1]))
        iter_weigths= tf.constant(np.array(iter_weights)[:, None], dtype=tf.float32)
        sum_iter_weights = tf.constant(np.sum(iter_weights), dtype=tf.float32)
        seq_ARI = tf.reduce_mean(tf.reduce_sum(ARI * iter_weigths, axis=0) / sum_iter_weights)
        last_ARI = tf.reduce_mean(ARI[-1])
        confidences = tf.reduce_sum(tf.reduce_max(gammas, axis=1, keep_dims=True) * M, axis=[1, 2]) / n
        confidences = tf.reshape(confidences, shape=(yshape[0], yshape[1]))
        seq_conf = tf.reduce_mean(tf.reduce_sum(confidences * iter_weigths, axis=0) / sum_iter_weights)
        last_conf = tf.reduce_mean(confidences[-1])
        return seq_ARI, last_ARI, seq_conf, last_conf


def color_spines(ax, color, lw=2):
    for sn in ['top', 'bottom', 'left', 'right']:
        ax.spines[sn].set_linewidth(lw)
        ax.spines[sn].set_color(color)
        ax.spines[sn].set_visible(True)


def get_gamma_colors(nr_colors):
    hsv_colors = np.ones((nr_colors, 3))
    hsv_colors[:, 0] = (np.linspace(0, 1, nr_colors, endpoint=False) + 2/3) % 1.0
    color_conv = hsv_to_rgb(hsv_colors)
    return color_conv


def overview_plot(i, gammas, preds, inputs, corrupted=None, **kwargs):
    T, B, K, W, H, C = gammas.shape
    T -= 1  # the initialization doesn't count as iteration
    corrupted = corrupted if corrupted is not None else inputs
    gamma_colors = get_gamma_colors(K)

    # restrict to sample i and get rid of useless dims
    inputs = inputs[:, i, 0]
    gammas = gammas[:, i, :, :, :, 0]
    if preds.shape[1] != B:
        preds = preds[:, 0]
    preds = preds[:, i]
    corrupted = corrupted[:, i, 0]

    # rescale input range to [0 - 1], assumes input data is [0, 1]
    inputs = np.clip(inputs, 0., 1.)
    preds = np.clip(preds, 0., 1.)
    corrupted = np.clip(corrupted, 0., 1.)

    def plot_img(ax, data, cmap='Greys_r', xlabel=None, ylabel=None, border_color=None):
        if data.shape[-1] == 1:
            ax.matshow(data[:, :, 0], cmap=cmap, vmin=0., vmax=1., interpolation='nearest')
        else:
            ax.imshow(data, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(xlabel, color=border_color or 'k') if xlabel else None
        ax.set_ylabel(ylabel, color=border_color or 'k') if ylabel else None
        if border_color:
            color_spines(ax, color=border_color)

    def plot_gamma(ax, gamma, xlabel=None, ylabel=None):
        gamma = np.transpose(gamma, [1, 2, 0])
        gamma = gamma.reshape(-1, gamma.shape[-1]).dot(gamma_colors).reshape(gamma.shape[:-1] + (3,))
        ax.imshow(gamma, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(xlabel) if xlabel else None
        ax.set_ylabel(ylabel) if ylabel else None

    # if inputs.shape[0] > 1:
    nrows, ncols = (K + 4, T + 1)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2 * ncols, 2 * nrows))

    axes[0, 0].set_visible(False)
    axes[1, 0].set_visible(False)
    plot_gamma(axes[2, 0], gammas[0], ylabel='Gammas')
    for k in range(K + 1):
        axes[k + 3, 0].set_visible(False)
    for t in range(1, T + 1):
        g = gammas[t]
        p = preds[t]
        reconst = np.sum(g[:, :, :, None] * p, axis=0)
        plot_img(axes[0, t], inputs[t])
        plot_img(axes[1, t], reconst)
        plot_gamma(axes[2, t], g)
        for k in range(K):
            plot_img(axes[k + 3, t], p[k], border_color=tuple(gamma_colors[k]),
                     ylabel=('mu_{}'.format(k) if t == 1 else None))
        plot_img(axes[K + 3, t], corrupted[t - 1])

    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    return fig


def curve_plot(values, y_range):
    fig, axes = plt.subplots()
    axes.plot(values)

    axes.set_xlabel('epochs')
    axes.axis([0, len(values), y_range[0], y_range[1]])

    return fig

