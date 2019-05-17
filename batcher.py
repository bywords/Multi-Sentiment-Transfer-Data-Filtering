# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to process data into batches"""

import os
from random import shuffle
import codecs
import json
import glob
import numpy as np
import tensorflow as tf
import data

FLAGS = tf.app.flags.FLAGS
class Example(object):
  """Class representing a train/val/test example for text summarization."""

  def __init__(self, headline, label, vocab, hps):

    self.hps = hps
    headline_words = headline.split()
    if len(headline_words) > hps.max_dec_steps: #:
        headline_words = headline_words[:hps.max_dec_steps]

    # list of word ids; OOVs are represented by the id for UNK token
    self.enc_input = [vocab.word2id(w) for w in headline_words]
    self.enc_len = len(headline_words)  # store the length after truncation but before padding
    self.label = int(label)
    self.headline = headline

  def pad_encoder_input(self, max_sen_len, pad_id):
    """Pad the encoder input sequence with pad_id up to max_len."""
    while len(self.enc_input) < max_sen_len:
        self.enc_input.append(pad_id)


class Batch(object):
  """Class representing a minibatch of train/val/test examples for text summarization."""

  def __init__(self, example_list, hps, vocab):
    """Turns the example_list into a Batch object.

    Args:
       example_list: List of Example objects
       hps: hyperparameters
       vocab: Vocabulary object
    """
    self.pad_id = vocab.word2id(data.PAD_TOKEN) # id of the PAD token used to pad sequences
    self.init_encoder_seq(example_list, hps)  # initialize the input to the encoder

  def init_encoder_seq(self, example_list, hps):
    for ex in example_list:
      ex.pad_encoder_input(hps.max_dec_steps, self.pad_id)

    self.enc_batch = np.zeros((hps.batch_size,hps.max_dec_steps), dtype=np.int32)
    self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
    self.enc_padding_mask = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)
    self.labels = np.zeros((hps.batch_size), dtype=np.int32)
    self.headlines = [ex.headline for ex in example_list]

    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.labels[i] = ex.label
      self.enc_batch[i,:] = np.array(ex.enc_input)[:]
      self.enc_lens[i] = ex.enc_len
      for j in range(ex.enc_len):
          self.enc_padding_mask[i][j] = 1


class TrainBatcher(object):

    def __init__(self, hps, vocab):
        self._vocab = vocab
        self._hps = hps

        self.example_queue = self.fill_example_queue(os.path.join(hps.data_dir,
                                                                  hps.train_file))
        self.batch = self.create_batches(shuffleis=True)

    def create_batches(self, shuffleis=True):
        all_batch = []

        num_batches = int(len(self.example_queue) / self._hps.batch_size)
        if shuffleis:
            shuffle(self.example_queue)

        for i in range(0, num_batches):
            batch = self.example_queue[i*self._hps.batch_size:i*self._hps.batch_size + self._hps.batch_size]
            all_batch.append(Batch(batch, self._hps, self._vocab))

        return all_batch

    def get_batches(self):
        shuffle(self.batch)
        return self.batch

    def fill_example_queue(self, data_path):
        new_queue = []

        filelist = glob.glob(data_path)  # get the list of datafiles
        assert filelist, ('Error: Empty filelist at %s' % data_path)  # check filelist isn't empty
        for f in filelist:
            reader = codecs.open(f, 'r', 'utf-8')
            while True:
                string_ = reader.readline()
                if not string_:
                    break

                dict_example = json.loads(string_)
                headline = dict_example["headline"]
                if headline.strip() == "":
                    continue

                score = int(dict_example["score"])
                example = Example(headline=headline, label=score, vocab=self._vocab, hps=self._hps)
                new_queue.append(example)
        return new_queue


class TestBatcher(object):

    def __init__(self, hps, vocab):
        self._vocab = vocab
        self._hps = hps

        self.example_queue = self.fill_example_queue(os.path.join(hps.data_dir,
                                                                  hps.test_file))
        self.batch = self.create_batches(shuffleis=True)

    def create_batches(self, shuffleis=True):
        all_batch = []

        num_batches = int(len(self.example_queue) / self._hps.batch_size)
        if shuffleis:
            shuffle(self.example_queue)

        for i in range(0, num_batches):
            batch = self.example_queue[i*self._hps.batch_size:i*self._hps.batch_size + self._hps.batch_size]
            all_batch.append(Batch(batch, self._hps, self._vocab))

        return all_batch

    def get_batches(self):
        shuffle(self.batch)
        return self.batch

    def fill_example_queue(self, data_path):
        new_queue = []

        filelist = glob.glob(data_path)  # get the list of datafiles
        assert filelist, ('Error: Empty filelist at %s' % data_path)  # check filelist isn't empty
        for f in filelist:
            reader = codecs.open(f, 'r', 'utf-8')
            while True:
                string_ = reader.readline()
                if not string_:
                    break

                dict_example = json.loads(string_)
                headline = dict_example["headline"]
                if headline.strip() == "":
                    continue

                score = int(dict_example["score"])
                example = Example(headline=headline, label=score, vocab=self._vocab, hps=self._hps)
                new_queue.append(example)
        return new_queue


