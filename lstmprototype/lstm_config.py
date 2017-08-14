'''
Created on 7 Jul 2017

@author: mtonnicchi
'''

import configparser


class LSTM_Config(object):

    def __init__(self, config_path):
        self.cfg = configparser.ConfigParser()
        self.cfg.read(config_path,encoding='utf8')
        
    def section(self, section_name):
        parameters = {}
        options = self.cfg.options(section_name)
        for option in options:
            try:
                parameters[option] = self.cfg.get(section_name, option)
            except:
                print("An error occurred for option %s!" % option)
                parameters[option] = None
        return parameters
        