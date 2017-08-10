#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals, absolute_import
import os
import h5py
import numpy as np
import tensorflow as tf

from sacred import Ingredient

ds = Ingredient('dataset')


@ds.config
def cfg():
    name = 'shapes'
    path = './data'
    binary = True
    train_size = None           # subset of training set (None, int)
    valid_size = 1000           # subset of valid set (None, int)
    test_size = None            # subset of test set (None, int)
    queue_capacity = 100        # nr of batches in the queue

ds.add_named_config('shapes',
                    {'name': 'shapes',
                     'binary': True})
ds.add_named_config('flying_shapes',
                    {'name': 'flying_shapes',
                     'binary': True})
ds.add_named_config('flying_shapes_4',
                    {'name': 'flying_shapes_4',
                     'binary': True})
ds.add_named_config('flying_shapes_5',
                    {'name': 'flying_shapes_5',
                     'binary': True})
ds.add_named_config('flying_mnist_medium_20_2',
                    {'name': 'flying_mnist_medium20_2digits',
                     'binary': False})
ds.add_named_config('flying_mnist_medium_500_2',
                    {'name': 'flying_mnist_medium500_2digits',
                     'binary': False})
ds.add_named_config('flying_mnist_hard_2',
                    {'name': 'flying_mnist_hard_2digits',
                     'binary': False})
ds.add_named_config('flying_mnist_hard_2_51',
                    {'name': 'flying_mnist_hard_2digits_51',
                     'binary': False})
ds.add_named_config('flying_mnist_hard_3',
                    {'name': 'flying_mnist_hard_3digits',
                     'binary': False})
ds.add_named_config('multi_mnist_2d',
                   {'name': 'mnist_2digits_downsampled',
                    'binary': False})


class InputPipeLine(object):
    @ds.capture
    def _open_dataset(self, out_list, path, name, train_size, valid_size, test_size):
        # open dataset file
        self._hdf5_file = h5py.File(os.path.join(path, name + '.h5'), 'r')
        self._data_in_file = {
            data_name: self._hdf5_file[self.usage][data_name] for data_name in out_list
        }
        self.limit = ({'training': train_size, 'validation': valid_size, 'test': test_size}[self.usage] or
                      self._data_in_file['features'].shape[1])

        # fix shapes and datatypes
        input_seq_len = 1 if self._data_in_file['features'].shape[0] == 1 else self.sequence_length
        self.shapes = {
            data_name: (input_seq_len, self.batch_size, 1) + self._data_in_file[data_name].shape[-3:]
            for data_name, data in self._data_in_file.items()
        }
        self.shapes['idx'] = ()
        self._dtypes = {data_name: tf.float32 for data_name in out_list}
        self._dtypes['idx'] = tf.int32

        # set up placeholders for inserting data into queue
        self._data_in = {
            data_name: tf.placeholder(self._dtypes[data_name], shape=shape)
            for data_name, shape in self.shapes.items()
        }

    @ds.capture
    def __init__(self, usage, shuffle, batch_size, sequence_length, queue_capacity, _rnd, out_list=('features', 'groups')):
        self.usage = usage
        self.shuffle = shuffle
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self._rnd = _rnd
        self.samples_cache = {}

        with tf.name_scope("{}_queue".format(usage[:5])):

            self._open_dataset(out_list)

            # set up queue
            self.queue = tf.FIFOQueue(capacity=queue_capacity,
                                      dtypes=[v for k, v in sorted(self._dtypes.items(), key=lambda x: x[0])],
                                      shapes=[v for k, v in sorted(self.shapes.items(), key=lambda x: x[0])],
                                      names=[k for k in sorted(self._dtypes)])

            self._enqueue_op = self.queue.enqueue(self._data_in)

            # set up outputs of queue (inputs for the model)
            self.output = self.queue.dequeue()
            if self.shapes['features'][0] == 1 and self.sequence_length > 1:
                # if the dataset has sequence length 1 we need to repeat the data
                reshaped_output = {data_name: tf.tile(self.output[data_name], [self.sequence_length, 1, 1, 1, 1, 1])
                                   for data_name in out_list}
                reshaped_output['idx'] = self.output['idx']
                self.output = reshaped_output

    def get_feed_data(self, start_idx):
        feed_dict = {self._data_in[data_name]: ds[:self.sequence_length, start_idx:start_idx + self.batch_size][:, :, None]
                     for data_name, ds in self._data_in_file.items()}
        feed_dict[self._data_in['idx']] = start_idx
        return feed_dict

    def get_debug_samples(self, samples_list, out_list=None):
        samples_key = tuple(samples_list)
        if samples_key in self.samples_cache:
            return self.samples_cache[samples_key]

        out_list = self._data_in_file.keys() if out_list is None else out_list
        results = {}
        for data_name in out_list:
            data = self._hdf5_file[self.usage][data_name][:, samples_list][:, :, None]
            if data.shape[0] == 1 and self.sequence_length > 1:
                data = np.repeat(data, self.sequence_length, axis=0)
            elif data.shape[0] > self.sequence_length:
                data = data[:self.sequence_length]
            results[data_name] = data

        self.samples_cache[samples_key] = results
        return results

    def get_batch_start_indices(self):
        idxs = np.arange(0, self.limit - self.batch_size, step=self.batch_size)
        if self.shuffle:
            self._rnd.shuffle(idxs)
        return 0, idxs

    def enqueue(self, session, coord):
        i, idxs = self.get_batch_start_indices()
        try:
            while not coord.should_stop():
                if i >= len(idxs):
                    i, idxs = self.get_batch_start_indices()
                session.run(self._enqueue_op, feed_dict=self.get_feed_data(idxs[i]))
                i += 1
        except Exception as e:
            coord.request_stop(e)
        finally:
            self._hdf5_file.close()

    def get_n_batches(self):
        return self.limit // self.batch_size
