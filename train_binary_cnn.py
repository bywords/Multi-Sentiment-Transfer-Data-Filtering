import os, time
import util
import numpy as np
import tensorflow as tf
from data import Vocab
from batcher import TrainBatcher, TestBatcher
from classifier import CNN
from collections import namedtuple
import tensorflow as tf
#import argparse

# parser = argparse.ArgumentParser(description='train_binary_cnn.py')
# parser.add_argument('-train_file', default="train.txt", help='Training data file')
# parser.add_argument('-test_file', default="test.txt", help='Training data file')
# parser.add_argument('-data_dir', default="data", help='Directory path to data')

FLAGS = tf.app.flags.FLAGS

# Where to find data
tf.app.flags.DEFINE_string('data_dir', 'data', 'Directory path to data')
tf.app.flags.DEFINE_string('train_file', 'train.txt', 'Data file for training')
tf.app.flags.DEFINE_string('test_file', 'test.txt', 'Data file for testing')
tf.app.flags.DEFINE_string('vocab_path', 'data/vocab.txt', 'Path expression to text vocabulary file.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train',
                           'must be train')  # train-classification  train-sentiment  train-cnn-classificatin train-generator
tf.app.flags.DEFINE_integer('num_class', 2, 'Number of style classes')

# Where to save output
tf.app.flags.DEFINE_string('log_root', 'log', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', 'evaluation',
                           'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

tf.app.flags.DEFINE_integer('gpuid', 1, 'for gradient clipping')

tf.app.flags.DEFINE_integer('max_enc_seq_len', 20,
                            'max timesteps of encoder (max source text tokens)')  # for discriminator

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')  # for discriminator and generator
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')  # for discriminator and generator
tf.app.flags.DEFINE_integer('batch_size', 64, 'minibatch size')  # for discriminator and generator
# The values are specified by statistics
tf.app.flags.DEFINE_integer('max_enc_steps', 20,
                            'max timesteps of encoder (max source text tokens)')  # for generator
tf.app.flags.DEFINE_integer('max_dec_steps', 20,
                            'max timesteps of decoder (max summary tokens)')  # for generator
tf.app.flags.DEFINE_integer('min_dec_steps', 10,
                            'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('vocab_size', 10000,
                            'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')

tf.app.flags.DEFINE_float('lr', 0.6, 'learning rate')  # for discriminator and generator
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1,
                          'initial accumulator value for Adagrad')  # for discriminator and generator
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02,
                          'magnitude for lstm cells random uniform inititalization')  # for discriminator and generator
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4,
                          'std of trunc norm init, used for initializing everything else')  # for discriminator and generator
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')  # for discriminator and generator


def setup_training_cnnclassifier(model):
    """Does setup before starting training (run_training)"""
    train_dir = os.path.join(FLAGS.log_root, "train-cnnclassification")
    if not os.path.exists(train_dir): os.makedirs(train_dir)

    model.build_graph()  # build the graph

    saver = tf.train.Saver(max_to_keep=20)  # we use this to load checkpoints for decoding
    sess = tf.Session(config=util.get_config())
    init = tf.global_variables_initializer()
    sess.run(init)
    # util.load_ckpt(saver, sess, ckpt_dir="train-cnnclassification")

    return sess, saver, train_dir


def run_train_cnn_classifier(model, train_batcher, test_batcher, max_run_epoch, sess, saver, train_dir):
    tf.logging.info("starting train_cnn_classifier")
    for epoch in range(max_run_epoch):
        batches = train_batcher.get_batches()
        t0 = time.time()
        loss_window = 0.0

        for step, current_batch in enumerate(batches):
            results = model.run_train_step(sess, current_batch)
            loss = results['loss']
            loss_window += loss
            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")
            train_step = results['global_step']  # we need this to update our running average loss
            if train_step % 100 == 0:
                t1 = time.time()
                tf.logging.info('seconds for %d training cnn classifier step: %.3f ', train_step, (t1 - t0) / 100)
                t0 = time.time()
                tf.logging.info('loss: %f', loss_window / 100)  # print the loss to screen
                loss_window = 0.0
            if train_step % 10000 == 0:
                acc = run_test_cnn_classification(model, test_batcher, sess)
                tf.logging.info('cnn evaluate test acc: %.6f', acc)  # print the loss to screen
                saver.save(sess, train_dir + "/model", global_step=train_step)

        tf.logging.info("finished %d epoches", epoch+1)


def run_test_cnn_classification(model, batcher, sess):
    tf.logging.info("starting run testing cnn_classification")
    batches = batcher.get_batches()
    right = 0.0
    all = 0.0

    for step, current_batch in enumerate(batches):
        right_s, number, error_list, error_label = model.run_eval_step(sess, current_batch)
        all += number
        right += right_s

    return right / all


def main():
    tf.logging.set_verbosity(tf.logging.INFO)  # choose what level of logging you want
    tf.logging.info('Starting running in %s mode...', (FLAGS.mode))

    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
    if not os.path.exists(FLAGS.log_root):
        os.makedirs(FLAGS.log_root)

    config = {
        'n_epochs': 5,
        'kernel_sizes': [3, 4, 5],
        'edim': FLAGS.emb_dim,
        'n_words': FLAGS.vocab_size,
        'std_dev': 0.05,
        'sentence_len': FLAGS.max_enc_steps,
        'n_filters': 100,
        'batch_size': FLAGS.batch_size,
        'trunc_norm_init_std': FLAGS.trunc_norm_init_std,
        'rand_unif_init_mag': FLAGS.rand_unif_init_mag,
        'max_grad_norm': FLAGS.max_grad_norm,
        'num_classes': FLAGS.num_class
    }

    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)

    hparam_list = ['lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
                   'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'source_class', 'num_class']
    hps_dict = {}
    for key, val in FLAGS.__flags.items():  # for each flag
        if key in hparam_list:  # if it's in the list
            hps_dict[key] = val  # add it to the dict
    hps_discriminator = namedtuple("HParams", hps_dict.keys())(**hps_dict)

    tf.set_random_seed(111)

    # Batcher: 데이터 준비하는 부분
    train_batcher = TrainBatcher(hps_discriminator, vocab)
    test_batcher = TestBatcher(hps_discriminator, vocab)

    # CNN class 초기화
    cnn_classifier = CNN(config)
    # CNN model 초기화 (TF-level)
    sess_cnn_cls, saver_cnn_cls, train_dir_cnn_cls = setup_training_cnnclassifier(cnn_classifier)
    run_train_cnn_classifier(model=cnn_classifier,
                             train_batcher=train_batcher,
                             test_batcher=test_batcher,
                             max_run_epoch=20,
                             sess=sess_cnn_cls,
                             saver=saver_cnn_cls,
                             train_dir=train_dir_cnn_cls)


if __name__ == '__main__':
    tf.app.run()