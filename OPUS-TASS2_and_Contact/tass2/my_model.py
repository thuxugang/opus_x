# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import os
import tensorflow as tf
from my_transformer import Transformer, create_padding_mask
from my_rnn import BiRNN_C4
from my_cnn import ResNet
from my_cnn2 import TRRosettaCNN

def clean_inputs(x, x_mask, dim_input):

    x_mask = tf.tile(x_mask[:,:,tf.newaxis], [1, 1, dim_input])
    x_clean = tf.where(tf.math.equal(x_mask, 0), x, x_mask-1)
    return x_clean

class Model(object):
    
    def __init__(self, params, name):
        
        self.params = params
        self.name = name
        
        self.transformer = Transformer(num_layers=self.params["transfomer_layers"],
                                       d_model=self.params["d_input"],
                                       num_heads=self.params["transfomer_num_heads"],
                                       rate=self.params["dropout_rate"])

        self.cnn = ResNet()
        self.trr_cnn = TRRosettaCNN(filter_num=64, 
                                    num_layers=61, 
                                    dropout=0.5)
        
        self.birnn = BiRNN_C4(num_layers=self.params["lstm_layers"],
                              units=self.params["lstm_units"],
                              rate=self.params["dropout_rate"],
                              ss8_output=self.params["d_ss8_output"],
                              ss3_output=self.params["d_ss3_output"],
                              phipsi_output=self.params["d_phipsi_output"],
                              csf_output=self.params["d_csf_output"],
                              asa_output=self.params["d_asa_output"])
        print ("use c4 model...")

    def inference(self, x, x_mask, x_trr, y, y_mask, training):

        encoder_padding_mask = create_padding_mask(x_mask)
        
        x_trr = self.trr_cnn(x_trr, training=training)
    
        x = tf.concat([x, x_trr], -1)

        x = clean_inputs(x, x_mask, self.params["d_input"])
        
        transformer_out = self.transformer(x, encoder_padding_mask, training=training)
        cnn_out = self.cnn(x, training=training)
        x = tf.concat((x, cnn_out, transformer_out), -1)
        
        x = clean_inputs(x, x_mask, 3*self.params["d_input"])
        
        ss8_predictions, ss3_predictions, phipsi_predictions, csf_predictions, asa_predictions = \
            self.birnn(x, x_mask, training=training) 
        
        return ss8_predictions, ss3_predictions, phipsi_predictions, csf_predictions, asa_predictions

    def load_model(self):
        print ("load model:", self.name, self.params["save_path"])
        self.transformer.load_weights(os.path.join(self.params["save_path"], self.name + '_trans_model_weight'))
        self.cnn.load_weights(os.path.join(self.params["save_path"], self.name + '_cnn_model_weight'))
        self.trr_cnn.load_weights(os.path.join(self.params["save_path"], self.name + '_trr_cnn_model_weight'))
        self.birnn.load_weights(os.path.join(self.params["save_path"], self.name + '_birnn_model_weight'))





