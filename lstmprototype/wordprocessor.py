'''
Created on 7 Jul 2017

@author: mtonnicchi
'''

import collections
import numpy as np

def index_lemmas(text, min_word_frequency):
    word_counts = collections.Counter(text.split(' '))
    
    word_counts = {key:val for key, val in word_counts.items() if val>min_word_frequency}

    words = word_counts.keys()
    lemma_to_index = {key:(index+1) for index, key in enumerate(words)}

    lemma_to_index['?']=0

    index_to_lemma = {val:key for key,val in lemma_to_index.items()}

    print('Indexed {} lemmas'.format(len(index_to_lemma) + 1))
    assert(len(index_to_lemma) == len(lemma_to_index))
    
    return(index_to_lemma, lemma_to_index)

def vectorise(text, lemma_to_index):
    words = text.split(' ')
    word_vector = []
    for index, word in enumerate(words):
        try:
            word_vector.append(lemma_to_index[word])
        except:
            word_vector.append(0)
    return np.array(word_vector)

def create_batches(source_vector, batch_size, training_sequence_length):
    num_batches = int(len(source_vector)/(batch_size * training_sequence_length)) + 1
    batches = np.array_split(source_vector, num_batches)
    batches = [np.resize(x, [batch_size, training_sequence_length]) for x in batches]
    return batches
