#! /usr/bin/python
# -*- coding: utf8 -*-
"""Sequence to Sequence Learning for Twitter/Cornell Chatbot.

References
----------
http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/
"""
import time

import click
import numpy as np
import tensorflow as tf
import tensorlayer as tl

from tqdm import tqdm
from sklearn.utils import shuffle
from tensorlayer.layers import DenseLayer, EmbeddingInputlayer, Seq2Seq, retrieve_seq_length_op2

from data.twitter import data

sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

"""
Training model [optional args]
"""
@click.command()
@click.option('-bs', '--batch-size', default=32, help='Batch size for training on minibatches',)
@click.option('-n', '--num-epochs', default=1, help='Number of epochs for training',)
@click.option('-lr', '--learning-rate', default=0.001, help='Learning rate to use when training model',)
@click.option('-inf', '--inference-mode', is_flag=True, help='Flag for INFERENCE mode',)


def train(batch_size, num_epochs, learning_rate, inference_mode):
    # Load dataset
    trainX = []
    trainY = []
    count = 0
    word_dict = {}
    fout = open("./data/lp_train_r0.txt", "r")
    fin = open("./data/lp_kw_train_r0.txt", "r")
    for line in fout:
        Y = []
        for word in line.strip().split():
            if word not in word_dict:
                word_dict[word] = count
                count = count + 1
            Y.append(word_dict[word])
        trainY.append(Y)
    word_dict['unk'] = count
    word_dict['_'] = count + 1
    src_vocab_size = len(word_dict)
    start_id = src_vocab_size  # 8002
    end_id = src_vocab_size + 1  # 8003
    word_dict['start_id'] = start_id
    word_dict['end_id'] = end_id
    word_dict_rev = {v: k for k, v in word_dict.items()}
    emb_dim = 1024
    src_vocab_size = src_vocab_size + 2

    for line in fin:
        line = line.replace("\t", " ")
        X = []
        for word in line.strip().split():
            X.append(word_dict[word])
        trainX.append(X)

    fout.close()
    fin.close()
    src_len = len(trainX)
    n_step = src_len // batch_size

    """ A data for Seq2Seq should look like this:
    input_seqs : ['how', 'are', 'you', '<PAD_ID'>]
    decode_seqs : ['<START_ID>', 'I', 'am', 'fine', '<PAD_ID'>]
    target_seqs : ['I', 'am', 'fine', '<END_ID>', '<PAD_ID'>]
    target_mask : [1, 1, 1, 1, 0]
    """

    # Init Session
    tf.reset_default_graph()
    sess = tf.Session(config=sess_config)

    # Training Data Placeholders
    encode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="encode_seqs")
    decode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="decode_seqs")
    target_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_seqs")
    target_mask = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_mask") 

    net_out, _ = create_model(encode_seqs, decode_seqs, src_vocab_size, emb_dim, is_train=True, reuse=False)
    net_out.print_params(False)

    # Inference Data Placeholders
    encode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="encode_seqs")
    decode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="decode_seqs")

    net, net_rnn = create_model(encode_seqs2, decode_seqs2, src_vocab_size, emb_dim, is_train=False, reuse=True)
    y = tf.nn.softmax(net.outputs)

    # Loss Function
    loss = tl.cost.cross_entropy_seq_with_mask(logits=net_out.outputs, target_seqs=target_seqs, 
                                                input_mask=target_mask, return_details=False, name='cost')

    # Optimizer
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # Init Vars
    sess.run(tf.global_variables_initializer())

    # Load Model
    tl.files.load_and_assign_npz(sess=sess, name='model.npz', network=net)

    if inference_mode:
        print('Inference Mode')
        print('--------------')
        while True:
            input_seq = input('Enter Query: ')
            sentence = inference(input_seq)
            print(" >", ' '.join(sentence))
    else:
        for epoch in range(num_epochs):
            trainX, trainY = shuffle(trainX, trainY, random_state=0)
            total_loss, n_iter = 0, 0
            for X, Y in tqdm(tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=batch_size, shuffle=False),
                            total=n_step, desc='Epoch[{}/{}]'.format(epoch + 1, num_epochs), leave=False):

                X = tl.prepro.pad_sequences(X)
                _target_seqs = tl.prepro.sequences_add_end_id(Y, end_id=end_id)
                _target_seqs = tl.prepro.pad_sequences(_target_seqs)
                _decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id=start_id, remove_last=False)
                _decode_seqs = tl.prepro.pad_sequences(_decode_seqs)
                _target_mask = tl.prepro.sequences_get_mask(_target_seqs)
                _, loss_iter = sess.run([train_op, loss], {encode_seqs: X, decode_seqs: _decode_seqs,
                                target_seqs: _target_seqs, target_mask: _target_mask})
                total_loss += loss_iter
                n_iter += 1

            # printing average loss after every epoch
            print('Epoch [{}/{}]: loss {:.4f}'.format(epoch + 1, num_epochs, total_loss / n_iter))

            tl.files.save_npz(net.all_params, name='model.npz', sess=sess)
    
    sess.close()


def create_model(encode_seqs, decode_seqs, src_vocab_size, emb_dim, is_train=True, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        # for chatbot, you can use the same embedding layer,
        # for translation, you may want to use 2 seperated embedding layers
        with tf.variable_scope("embedding") as vs:
            net_encode = EmbeddingInputlayer(
                inputs = encode_seqs,
                vocabulary_size = src_vocab_size,
                embedding_size = emb_dim,
                name = 'seq_embedding')
            vs.reuse_variables()
            net_decode = EmbeddingInputlayer(
                inputs = decode_seqs,
                vocabulary_size = src_vocab_size,
                embedding_size = emb_dim,
                name = 'seq_embedding')
            
        net_rnn = Seq2Seq(net_encode, net_decode,
                cell_fn = tf.nn.rnn_cell.LSTMCell,
                n_hidden = emb_dim,
                initializer = tf.random_uniform_initializer(-0.1, 0.1),
                encode_sequence_length = retrieve_seq_length_op2(encode_seqs),
                decode_sequence_length = retrieve_seq_length_op2(decode_seqs),
                initial_state_encode = None,
                dropout = (0.5 if is_train else None),
                n_layer = 3,
                return_seq_2d = True,
                name = 'seq2seq')

        net_out = DenseLayer(net_rnn, n_units=src_vocab_size, act=tf.identity, name='output')
    return net_out, net_rnn


def main():
    try:
        train()
    except KeyboardInterrupt:
        print('Aborted!')

if __name__ == '__main__':
    main()
