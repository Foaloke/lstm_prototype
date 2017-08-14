'''
Created on 7 Jul 2017

@author: mtonnicchi
'''

import codecs
import os
import requests
import re
import string

def load_source(source_dir, source_file, source_url):
    print('Loading Source')

    if not os.path.isfile(os.path.join(source_dir, source_file)):
        print('Not found, downloading from '+source_url)

        response = requests.get(source_url)
        raw_source = response.content
        text = raw_source.decode('utf-8')

        prepare_directory(source_dir)
        
        with codecs.open(os.path.join(source_dir, source_file), 'w', 'utf-8') as out_conn:
            out_conn.write(text)
            out_conn.close()
        
        return text
        
    else:

        with codecs.open(os.path.join(source_dir, source_file), 'r', 'utf-8') as file_conn:
            return file_conn.read().replace('\n', '')

def clean_text(text, incipit, ending, punctuation_whitelist_string):

    incipit_index = text.find(incipit)
    ending_index = text.rfind(ending)
    
    if incipit_index == -1:
        raise ValueError("Given incipit ("+incipit+") is not contained in the source")
    if ending_index == -1:
        raise ValueError("Given ending ("+ending+") is not contained in the raw source")

    # Remove carriage returns
    text = text[incipit_index:ending_index]
    text = text.replace('\r\n', '')
    text = text.replace('\n', '')

    # Remove punctuation
    punctuation = string.punctuation
    punctuation_whitelist = read_symbol_list(punctuation_whitelist_string.split(','))
    punctuation = ''.join([x for x in punctuation if x not in punctuation_whitelist])
    text = re.sub(r'[{}]'.format(punctuation), ' ', text)
    text = re.sub('\s+', ' ', text ).strip().lower()
    
    return text

def prepare_directory_and_subdirectory(directory, sub):
    full_path = os.path.join(directory, sub)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path

def prepare_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
def read_symbol_list(symbol_list):
    return [char_code.decode('unicode_escape') for char_code in symbol_list]
    
    
    
    