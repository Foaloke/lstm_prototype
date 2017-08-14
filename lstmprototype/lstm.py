'''
Created on 7 Jul 2017

@author: mtonnicchi
'''

import os
import sourceutils as src
import pickle
import tensorflow as tf
import numpy as np
import random

class LSTM_Builder():

    def __init__(self):
        self.infer_sample = False

    def with_embedding_size(self, embedding_size):
        self.embedding_size = embedding_size
        return self
    
    def with_rnn_size(self, rnn_size):
        self.rnn_size = rnn_size
        return self
    
    def with_batch_size(self, batch_size):
        self.batch_size = batch_size
        return self
    
    def with_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        return self
    
    def with_training_sequence_length(self, training_sequence_length):
        self.training_sequence_length = training_sequence_length
        return self
    
    def with_vocabulary_size(self, vocabulary_size):
        self.vocabulary_size = vocabulary_size
        return self
    
    def with_infer_sample(self, infer_sample):
        self.infer_sample = infer_sample
        return self
    
    def build(self):
        return LSTM(self.embedding_size, self.rnn_size, self.batch_size, self.learning_rate, self.training_sequence_length, self.vocabulary_size, self.infer_sample)


class LSTM():

    def __init__(self, embedding_size, rnn_size, batch_size, learning_rate, training_sequence_length, vocabulary_size, infer_sample=False):

        if infer_sample:
            self.batch_size = 1
            self.training_sequence_length = 1
        else:
            self.batch_size = batch_size
            self.training_sequence_length = training_sequence_length

        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.vocabulary_size = vocabulary_size
        self.infer_sample = infer_sample
        self.learning_rate = learning_rate
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
        self.zero_state = self.lstm_cell.zero_state(self.batch_size, tf.float32)
        self.x_data = tf.placeholder(tf.int32, [self.batch_size, self.training_sequence_length])
        self.y_output = tf.placeholder(tf.int32, [self.batch_size, self.training_sequence_length])
        
        with tf.variable_scope('lstm_vars'):
            # Softmax
            W = tf.get_variable('W', [self.rnn_size, self.vocabulary_size], tf.float32, tf.random_normal_initializer())
            b = tf.get_variable('b', [self.vocabulary_size], tf.float32, tf.constant_initializer(0.0))
        
            # Embedding
            embedding_mat = tf.get_variable('embedding_mat', [self.vocabulary_size, self.embedding_size],
                                            tf.float32, tf.random_normal_initializer())
                                            
            embedding_output = tf.nn.embedding_lookup(embedding_mat, self.x_data)
            rnn_inputs = tf.split(axis=1, num_or_size_splits=self.training_sequence_length, value=embedding_output)
            rnn_inputs_trimmed = [tf.squeeze(x, [1]) for x in rnn_inputs]
        
        # Inferring
        def inferred_loop(prev, count):
            prev_transformed = tf.matmul(prev, W) + b
            prev_symbol = tf.stop_gradient(tf.argmax(prev_transformed, 1))
            output = tf.nn.embedding_lookup(embedding_mat, prev_symbol)
            return(output)
        
        decoder = tf.contrib.legacy_seq2seq.rnn_decoder
        outputs, last_state = decoder(rnn_inputs_trimmed,
                                      self.zero_state,
                                      self.lstm_cell,
                                      loop_function=inferred_loop if infer_sample else None)

        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.rnn_size])
        self.logit_output = tf.matmul(output, W) + b
        self.model_output = tf.nn.softmax(self.logit_output)
        
        loss_fun = tf.contrib.legacy_seq2seq.sequence_loss_by_example
        loss = loss_fun([self.logit_output],[tf.reshape(self.y_output, [-1])],
                [tf.ones([self.batch_size * self.training_sequence_length])],
                self.vocabulary_size)
        self.cost = tf.reduce_sum(loss) / (self.batch_size * self.training_sequence_length)
        self.final_state = last_state
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()), 4.5)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
        self.saver = tf.train.Saver(tf.global_variables())
        
    def sample(self, sess, index_to_lemma, lemma_to_index, num, prime_text):
        state = sess.run(self.lstm_cell.zero_state(1, tf.float32))
        word_list = prime_text.split()
        for word in word_list[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = lemma_to_index[word]
            feed_dict = {self.x_data: x, self.zero_state:state}
            [state] = sess.run([self.final_state], feed_dict=feed_dict)

        out_sentence = prime_text
        word = word_list[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = lemma_to_index[word]
            feed_dict = {self.x_data: x, self.zero_state:state}
            [model_output, state] = sess.run([self.model_output, self.final_state], feed_dict=feed_dict)
            sample = np.argmax(model_output[0])
            if sample == 0:
                break
            if sample > len(index_to_lemma):
                print('Malformed sample! Got: '+sample+' on an index of size '+len(index_to_lemma))
                word = '?'
            else:
                word = index_to_lemma[sample]
            out_sentence = out_sentence + ' ' + word
        return(out_sentence)
    
    def train(self, session, prime_texts, batches, epochs, save_interval, eval_interval, state_saver, index_to_lemma, lemma_to_index, test_lstm_model):
        # Train model
        train_loss = []
        iteration_count = 1
        num_batches = len(batches)
        for epoch in range(epochs):
            # Shuffle word indices
            random.shuffle(batches)
            # Create targets from shuffled batches
            targets = [np.roll(x, -1, axis=1) for x in batches]
            # Run a through one epoch
            print('Epoch #{} of {}.'.format(epoch+1, epochs))
            # Reset initial LSTM state every epoch
            state = session.run(self.zero_state)
            for ix, batch in enumerate(batches):
                training_dict = {self.x_data: batch, self.y_output: targets[ix]}
                c, h = self.zero_state
                training_dict[c] = state.c
                training_dict[h] = state.h
                
                temp_loss, state, _ = session.run([self.cost, self.final_state, self.train_op],
                                               feed_dict=training_dict)
                train_loss.append(temp_loss)
                
                # Print status every 10 gens
                if iteration_count % 10 == 0:
                    summary_nums = (iteration_count, epoch+1, ix+1, num_batches+1, temp_loss)
                    print('Iteration: {}, Epoch: {}, Batch: {} out of {}, Loss: {:.2f}'.format(*summary_nums))
                
                # Save state
                if iteration_count % save_interval == 0:
                    state_saver.save_state(session, iteration_count)
                
                if iteration_count % eval_interval == 0:
                    for sample in prime_texts:
                        print(test_lstm_model.sample(session, index_to_lemma, lemma_to_index, epochs, prime_text=sample))

                iteration_count += 1
        
        return train_loss
    
class LSTM_StateSaver(object):

    def __init__(self, cfg_data, lemma_to_index, index_to_lemma):
        self.saver = tf.train.Saver(tf.global_variables())
        self.model_dir = src.prepare_directory_and_subdirectory(cfg_data['source_dir'], cfg_data['model_dir'])
        self.lemma_to_index = lemma_to_index
        self.index_to_lemma = index_to_lemma

    def save_state(self, session, iteration_count):
        # Persist model
        model_file_name = os.path.join(self.model_dir, 'model')
        self.saver.save(session, model_file_name, global_step = iteration_count)
        print('Model Saved To: {}'.format(model_file_name))
        # Persist vocabulary
        dictionary_file = os.path.join(self.model_dir, 'vocab.pkl')
        with open(dictionary_file, 'wb') as dict_file_conn:
            pickle.dump([self.lemma_to_index, self.index_to_lemma], dict_file_conn)

