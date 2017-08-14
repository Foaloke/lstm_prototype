'''
Created on 7 Jul 2017

@author: mtonnicchi
'''

if __name__ == '__main__':
    pass

from tensorflow.python.framework import ops

import matplotlib.pyplot as plt
import tensorflow as tf

import lstm_config
import sourceutils as src
import wordprocessor as wp
import lstm

# Reset tensorflow graph and get a new session
ops.reset_default_graph()
session = tf.Session()

# Read the configuration
cfg = lstm_config.LSTM_Config('config.ini')
cfg_data = cfg.section('DATA')
cfg_rnn = cfg.section('RNN')

# Get prime texts
prime_texts = cfg_rnn['prime_texts'].split(',')

# Load source and clean it
raw_source_text = src.load_source(cfg_data['source_dir'], cfg_data['source_file'], cfg_data['source_url'])
source_text = src.clean_text(raw_source_text, cfg_data['incipit'], cfg_data['ending'], cfg_data['punctuation_whitelist'])

# Index lemmas
index_to_lemma, lemma_to_index = wp.index_lemmas(source_text, int(cfg_data['min_word_frequency']))
# Vectorise text
source_vector = wp.vectorise(source_text, lemma_to_index)

lstm_model = \
    lstm.LSTM_Builder() \
        .with_embedding_size(int(cfg_rnn['rnn_size'])) \
        .with_rnn_size(int(cfg_rnn['rnn_size'])) \
        .with_batch_size(int(cfg_rnn['batch_size'])) \
        .with_learning_rate(float(cfg_rnn['learning_rate'])) \
        .with_training_sequence_length(int(cfg_rnn['training_sequence_length'])) \
        .with_vocabulary_size(len(index_to_lemma) + 1) \
        .with_infer_sample(False) \
        .build()

# Reuse scope
with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    test_lstm_model = \
        lstm.LSTM_Builder() \
            .with_embedding_size(int(cfg_rnn['rnn_size'])) \
            .with_rnn_size(int(cfg_rnn['rnn_size'])) \
            .with_batch_size(int(cfg_rnn['batch_size'])) \
            .with_learning_rate(float(cfg_rnn['learning_rate'])) \
            .with_training_sequence_length(int(cfg_rnn['training_sequence_length'])) \
            .with_vocabulary_size(len(index_to_lemma) + 1) \
            .with_infer_sample(True) \
            .build()


# Create model saver
saver = tf.train.Saver(tf.global_variables())
batches = wp.create_batches(source_vector, int(cfg_rnn['batch_size']), int(cfg_rnn['training_sequence_length']))
num_batches = len(batches)

# Initialize all variables
init = tf.global_variables_initializer()
session.run(init)

# Train model
state_saver = lstm.LSTM_StateSaver(cfg_data, lemma_to_index, index_to_lemma)
train_loss = lstm_model \
                .train(
                    session, \
                    prime_texts, \
                    batches, \
                    int(cfg_rnn['epochs']), \
                    int(cfg_rnn['save_interval']), \
                    int(cfg_rnn['eval_interval']), \
                    state_saver, \
                    index_to_lemma, \
                    lemma_to_index, \
                    test_lstm_model)

print('Training done')

# Plot loss over time
plt.plot(train_loss, 'k-')
plt.title('Sequence to Sequence Loss')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()