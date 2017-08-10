#!/usr/bin/env python
# coding=utf-8

from __future__ import (print_function, division, absolute_import, unicode_literals)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # no INFO/WARN logs from Tensorflow

import time
import utils
import threading
import numpy as np
import tensorflow as tf

from tensorflow.contrib import distributions as dist
from sacred import Experiment
from sacred.utils import get_by_dotted_path
from datasets import ds
from datasets import InputPipeLine
from nem_model import nem, static_nem_iterations, get_loss_step_weights


ex = Experiment("NEM", ingredients=[ds, nem])


# noinspection PyUnusedLocal
@ex.config
def cfg():
    noise = {
        'noise_type': 'data',                           # {data, bitflip, masked_uniform}
        'prob': 0.2,                                    # probability of annihilating the pixel
    }
    training = {
        'optimizer': 'adam',                            # {adam, sgd, momentum, adadelta, adagrad, rmsprop}
        'params': {
            'learning_rate': 0.001,                     # float
        },
        'max_patience': 10,                             # number of epochs to wait before early stopping
        'batch_size': 64,
        'max_epoch': 500,
        'clip_gradients': None,                         # maximum norm of gradients
        'debug_samples': [3, 37, 54],                   # sample ids to generate plots for (None, int, list)
        'save_epochs': [1, 5, 10, 20, 50, 100]          # at what epochs to save the model independent of valid loss
    }
    validation = {
        'batch_size': training['batch_size'],
        'debug_samples': [0, 1, 2]                      # sample ids to generate plots for (None, int, list)
    }

    log_dir = 'debug_out'                               # directory to dump logs and debug plots
    net_path = None                                     # path of to network file to initialize weights with

    # config to control run_from_file
    run_config = {
        'usage': 'test',                                # what dataset to use {training, validation, test}
        'AMI': True,                                    # whether to compute the AMI score (this is expensive)
        'batch_size': 100,
        'debug_samples': [0, 1, 2]                      # sample ids to generate plots for (None, int, list)
    }


@ex.capture
def add_noise(data, noise, dataset):
    noise_type = noise['noise_type']
    if noise_type in ['None', 'none', None]:
        return data
    if noise_type == 'data':
        noise_type = 'bitflip' if dataset['binary'] else 'masked_uniform'

    with tf.name_scope('input_noise'):
        shape = tf.stack([s.value if s.value is not None else tf.shape(data)[i]
                         for i, s in enumerate(data.get_shape())])

        if noise_type == 'bitflip':
            noise_dist = dist.Bernoulli(probs=noise['prob'], dtype=data.dtype)
            n = noise_dist.sample(shape)
            corrupted = data + n - 2 * data * n  # hacky way of implementing (data XOR n)
        elif noise_type == 'masked_uniform':
            noise_dist = dist.Uniform(low=0., high=1.)
            noise_uniform = noise_dist.sample(shape)

            # sample mask
            mask_dist = dist.Bernoulli(probs=noise['prob'], dtype=data.dtype)
            mask = mask_dist.sample(shape)

            # produce output
            corrupted = mask * noise_uniform + (1 - mask) * data
        else:
            raise KeyError('Unknown noise_type "{}"'.format(noise_type))

        corrupted.set_shape(data.get_shape())
        return corrupted


@ex.capture(prefix='training')
def set_up_optimizer(loss, optimizer, params, clip_gradients):
    opt = {
        'adam': tf.train.AdamOptimizer,
        'sgd': tf.train.GradientDescentOptimizer,
        'momentum': tf.train.MomentumOptimizer,
        'adadelta': tf.train.AdadeltaOptimizer,
        'adagrad': tf.train.AdagradOptimizer,
        'rmsprop': tf.train.RMSPropOptimizer
    }[optimizer](**params)

    # optionally clip gradients by norm
    grads_and_vars = opt.compute_gradients(loss)
    if clip_gradients is not None:
        grads_and_vars = [(tf.clip_by_norm(grad, clip_gradients), var)
                          for grad, var in grads_and_vars]

    return opt, opt.apply_gradients(grads_and_vars)


@ex.capture
def build_graph(features, groups, dataset):
    features_corrupted = add_noise(features)
    loss, thetas, preds, gammas, other_losses, upper_bound_losses = \
        static_nem_iterations(features_corrupted, features, dataset['binary'])
    graph = {
        'inputs': features,
        'groups': groups,
        'corrupted': features_corrupted,
        'loss': loss,
        'gammas': gammas,
        'thetas': thetas,
        'preds': preds,
        'other_losses': other_losses,
        'upper_bound_losses': upper_bound_losses,
        'ARI': utils.tf_adjusted_rand_index(groups, gammas, get_loss_step_weights())
    }
    return graph


def build_debug_graph(inputs):
    nr_iters = inputs['features'].shape[0]
    feature_shape = [s.value for s in inputs['features'].shape[2:]]
    groups_shape = [s.value for s in inputs['groups'].shape[2:]]
    with tf.name_scope('debug'):
        X_debug_shape = [nr_iters, None] + feature_shape
        G_debug_shape = [nr_iters, None] + groups_shape
        X_debug = tf.placeholder(tf.float32, shape=X_debug_shape)
        G_debug = tf.placeholder(tf.float32, shape=G_debug_shape)
        return build_graph(X_debug, G_debug)


@ex.capture
def build_graphs(train_inputs, valid_inputs):
    # Build Graph
    varscope = tf.get_variable_scope()
    with tf.name_scope("train"):
        train_graph = build_graph(train_inputs['features'], train_inputs['groups'])
        opt, train_op = set_up_optimizer(train_graph['loss'])

    varscope.reuse_variables()
    with tf.name_scope("valid"):
        valid_graph = build_graph(valid_inputs['features'], valid_inputs['groups'])

    debug_graph = build_debug_graph(valid_inputs)

    return train_op, train_graph, valid_graph, debug_graph


@ex.capture
def create_curve_plots(name, losses, y_range, log_dir):
    import matplotlib.pyplot as plt
    fig = utils.curve_plot(losses, y_range)
    fig.suptitle(name)
    fig.savefig(os.path.join(log_dir, name + '_curve.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


@ex.capture
def create_debug_plots(name, debug_out, debug_groups, sample_indices, log_dir):
    import matplotlib.pyplot as plt
    scores, confidencess = utils.evaluate_groups_seq(debug_groups[1:], debug_out['gammas'][1:], get_loss_step_weights())
    for i, nr in enumerate(sample_indices):
        fig = utils.overview_plot(i, **debug_out)
        fig.suptitle(name + ', sample {},  AMI Score: {:.3f} ({:.3f}) '.format(nr, scores[i], confidencess[i]))
        fig.savefig(os.path.join(log_dir, name + '_{}.png'.format(nr)), bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def populate_debug_out(session, debug_graph, pipe_line, debug_samples, name):
    idxs = debug_samples if isinstance(debug_samples, list) else [debug_samples]
    debug_data = pipe_line.get_debug_samples(idxs, out_list=['features', 'groups'])
    debug_out = session.run(debug_graph, feed_dict={debug_graph['inputs']: debug_data['features'],
                                                    debug_graph['groups']: debug_data['groups']})

    create_debug_plots(name, debug_out, debug_data['groups'], idxs)


def run_epoch(session, pipe_line, graph, debug_graph, debug_samples, debug_name, train_op=None):
    fetches = [graph['loss'], graph['other_losses'], graph['ARI'], graph['upper_bound_losses']]
    fetches.append(train_op) if train_op is not None else None

    losses, others, ari_scores, ub_losses = [], [], [], []
    # run through the epoch
    for b in range(pipe_line.get_n_batches()):
        # run batch
        out = session.run(fetches)

        # log out
        losses.append(out[0])
        others.append(out[1])
        ari_scores.append(out[2])
        ub_losses.append(out[3])

    if debug_samples is not None:
        populate_debug_out(session, debug_graph, pipe_line, debug_samples, debug_name)

    return float(np.mean(losses)), np.mean(others, axis=0), np.mean(ari_scores, axis=0), float(np.mean(ub_losses, axis=0)[-1])


@ex.capture
def add_log(key, value, _run):
    if 'logs' not in _run.info:
        _run.info['logs'] = {}
    logs = _run.info['logs']
    split_path = key.split('.')
    current = logs
    for p in split_path[:-1]:
        if p not in current:
            current[p] = {}
        current = current[p]

    final_key = split_path[-1]
    if final_key not in current:
        current[final_key] = []
    entries = current[final_key]
    entries.append(value)


@ex.capture
def get_logs(key, _run):
    logs = _run.info.get('logs', {})
    return get_by_dotted_path(logs, key)


def compute_AMI_scores(session, pipeline, graph, batch_size):
    losses, others, scores, confidences = [], [], [], []
    for b in range(pipeline.limit // batch_size):
        samples = list(range(b * batch_size, (b + 1) * batch_size))
        data = pipeline.get_debug_samples(samples, out_list=['features', 'groups'])

        # run batch
        out = session.run(graph, {graph['inputs']: data['features'], graph['groups']: data['groups']})

        # log out
        losses.append(out['loss'])
        others.append(out['other_losses'])

        # for each timestep compute the unweighted AMI (ignoring the first step)
        batch_scores, batch_confidences = [], []
        for t in range(1, out['gammas'].shape[0]):
            time_score, time_confidence = utils.evaluate_groups(data['groups'][t], out['gammas'][t])
            batch_scores.append(time_score)
            batch_confidences.append(time_confidence)

        scores.append(np.mean(batch_scores, axis=1))
        confidences.append(np.mean(batch_confidences, axis=1))

    return float(np.mean(losses)), np.mean(others, axis=0), np.mean(scores, axis=0), np.mean(confidences, axis=0)


@ex.command
def run_from_file(run_config, nem, log_dir, seed, net_path=None):
    tf.set_random_seed(seed)

    # load network weights (default is log_dir/best if net_path is not set)
    net_path = os.path.abspath(os.path.join(log_dir, 'best')) if net_path is None else net_path
    usage = run_config['usage']

    with tf.Graph().as_default() as g:
        # Set up Data
        nr_steps = nem['nr_steps'] + 1
        inputs = InputPipeLine(usage, shuffle=False, sequence_length=nr_steps, batch_size=run_config['batch_size'])

        # Build Graph
        _, _, graph, debug_graph = build_graphs(inputs.output, inputs.output)

        t = time.time()
        with tf.Session(graph=g) as session:
            saver = tf.train.Saver()
            saver.restore(session, net_path)

            # run a regular epoch
            if not run_config['AMI']:
                # launch pipeline
                coord = tf.train.Coordinator()
                enqueue_thread = threading.Thread(target=inputs.enqueue, args=[session, coord])
                enqueue_thread.start()

                # compute ARI and losses
                loss, others, scores, ub_loss_last = run_epoch(
                    session, inputs, graph, debug_graph, run_config['debug_samples'], "rff_{}_e{}".format(usage, -1))

                # shutdown pipeline
                coord.request_stop()
                session.run(inputs.queue.close(cancel_pending_enqueues=True))
                coord.join()

                print("    Evaluation Loss: %.3f, ARI: %.3f (conf: %0.3f), Last ARI: %.3f (conf: %.3f) took %.3fs" %
                      (loss, scores[0], scores[2], scores[1], scores[3], time.time() - t))
                print("    other Evaluation Losses: ({})".format(", ".join(["%.2f" % o for o in others.mean(0)])))
                print("    loss UB last: %.3f" % ub_loss_last)

            # compute AMI scores
            else:
                loss, others, scores, confidences = compute_AMI_scores(session, inputs, debug_graph, run_config['batch_size'])

                # weight across time
                weights = get_loss_step_weights()
                s_weights = np.sum(weights)
                w_score, w_confidence = np.sum(scores * weights)/s_weights, np.sum(confidences * weights)/s_weights

                print("    Evaluation Loss: %.3f, AMI: %.3f (conf: %0.3f), Last AMI: %.3f (conf: %.3f) took %.3fs" %
                      (loss, w_score, w_confidence, scores[-1], confidences[-1], time.time() - t))
                print("    other Evaluation Losses: ({})".format(", ".join(["%.2f" % o for o in others.mean(0)])))


@ex.automain
def run(net_path, training, validation, nem, seed, log_dir, _run):

    # clear debug dir
    if log_dir and net_path is None:
        utils.create_directory(log_dir)
        utils.delete_files(log_dir, recursive=True)

    # Set up data pipelines
    nr_iters = nem['nr_steps'] + 1
    train_inputs = InputPipeLine('training', shuffle=True, sequence_length=nr_iters, batch_size=training['batch_size'])
    valid_inputs = InputPipeLine('validation', shuffle=False, sequence_length=nr_iters, batch_size=validation['batch_size'])

    # Build Graph
    train_op, train_graph, valid_graph, debug_graph = build_graphs(train_inputs.output, valid_inputs.output)
    init = tf.global_variables_initializer()

    # print vars
    utils.print_vars(tf.trainable_variables())

    with tf.Session() as session:
        tf.set_random_seed(seed)

        # continue training from net_path if specified.
        saver = tf.train.Saver()
        if net_path is not None:
            saver.restore(session, net_path)
        else:
            session.run(init)

        # start training pipelines
        coord = tf.train.Coordinator()
        train_enqueue_thread = threading.Thread(target=train_inputs.enqueue, args=[session, coord])
        coord.register_thread(train_enqueue_thread)
        train_enqueue_thread.start()
        valid_enqueue_thread = threading.Thread(target=valid_inputs.enqueue, args=[session, coord])
        coord.register_thread(valid_enqueue_thread)
        valid_enqueue_thread.start()

        best_valid_loss = np.inf
        best_valid_epoch = 0
        for epoch in range(1, training['max_epoch'] + 1):

            t = time.time()
            train_loss, others, train_scores, train_ub_loss_last = run_epoch(
                session, train_inputs, train_graph, debug_graph, training['debug_samples'], "train_e{}".format(epoch),
                train_op=train_op)

            add_log('training.loss', train_loss)
            add_log('training.others', others)
            add_log('training.score', train_scores[0])
            add_log('training.score_last', train_scores[1])
            add_log('training.ub_loss_last', train_ub_loss_last)

            create_curve_plots('train_loss', get_logs('training.loss'), [0, 2000])

            print("Epoch: %d Train Loss: %.3f, ARI: %.3f (conf: %0.3f), Last ARI: %.3f (conf: %.3f) took %.3fs" %
                  (epoch, train_loss, train_scores[0], train_scores[2], train_scores[1], train_scores[3], time.time() - t))
            print("    Other Train Losses:      ({})".format(", ".join(["%.2f" % o for o in others.mean(0)])))
            print("    Train Loss UB last: %.2f" % train_ub_loss_last)

            t = time.time()
            valid_loss, others, valid_scores, valid_ub_loss_last = run_epoch(
                session, valid_inputs, valid_graph, debug_graph, validation['debug_samples'], "valid_e{}".format(epoch))

            # valid_scores = seq_ARI, last_ARI, seq_conf, last_conf
            add_log('validation.loss', valid_loss)
            add_log('validation.others', others)
            add_log('validation.score', valid_scores[0])
            add_log('validation.score_last', valid_scores[1])
            add_log('validation.ub_loss_last', valid_ub_loss_last)

            create_curve_plots('valid_loss', get_logs('validation.loss'), [0, 2000])
            create_curve_plots('valid_score', get_logs('validation.score'), [0, 1])
            create_curve_plots('valid_score_last', get_logs('validation.score_last'), [0, 1])

            print("    Validation Loss: %.3f, ARI: %.3f (conf: %0.3f), Last ARI: %.3f (conf: %.3f) took %.3fs" %
                  (valid_loss, valid_scores[0], valid_scores[2], valid_scores[1], valid_scores[3], time.time() - t))
            print("    Other Validation Losses: ({})".format(", ".join(["%.2f" % o for o in others.mean(0)])))
            print("    Valid Loss UB last: %.2f" % valid_ub_loss_last)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_valid_epoch = epoch
                _run.result = float(valid_scores[0]), float(valid_scores[1]), float(valid_loss)
                print("    Best validation loss improved to %.03f" % best_valid_loss)
                save_destination = saver.save(session, os.path.abspath(os.path.join(log_dir, 'best')))
                print("    Saved to:", save_destination)
            if epoch in training['save_epochs']:
                save_destination = saver.save(session, os.path.abspath(os.path.join(log_dir, 'epoch_{}'.format(epoch))))
                print("    Saved to:", save_destination)
            best_valid_loss = min(best_valid_loss, valid_loss)
            if best_valid_loss < np.min(get_logs('validation.loss')[-training['max_patience']:]):
                print('Early Stopping because validation loss did not improve for {} epochs'.format(training['max_patience']))
                break

            if np.isnan(valid_loss):
                print('Early Stopping because validation loss is nan')
                break

        # shutdown everything to avoid zombies
        coord.request_stop()
        session.run(train_inputs.queue.close(cancel_pending_enqueues=True))
        session.run(valid_inputs.queue.close(cancel_pending_enqueues=True))
        coord.join()

    return float(get_logs('validation.score')[best_valid_epoch - 1]), float(get_logs('validation.score_last')[best_valid_epoch - 1]), float(get_logs('validation.loss')[best_valid_epoch - 1])
