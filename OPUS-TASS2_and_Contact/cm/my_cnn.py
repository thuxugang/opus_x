# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 15:03:06 2020

@author: xugang
"""

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

class FirstBlock(keras.layers.Layer):

    def __init__(self, filter_num: int, kernel_size: int):
        
        super(FirstBlock, self).__init__()
        
        self.conv = keras.layers.Conv2D(filters=filter_num,
                                        kernel_size=kernel_size,
                                        padding="SAME")
        self.norm = tfa.layers.normalizations.InstanceNormalization()

    def call(self, inputs):
        
        inputs = self.conv(inputs)
        inputs = self.norm(inputs)
        inputs = tf.nn.elu(inputs)
        
        return inputs

class MyConv2d(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding, dilation_rate):
        
        super(MyConv2d, self).__init__()
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation_rate = [1, dilation_rate, dilation_rate,1]
    
    def build(self, input_shape):
        
        self.kernel = self.add_variable(
                name="my_kernel",
                shape=(self.kernel_size, self.kernel_size, self.filters, self.filters),
                initializer=tf.keras.initializers.GlorotUniform(),
                trainable=True,
                )
        self.bias = self.add_variable(
                name="my_bias",
                shape=(self.filters,),
                initializer=tf.zeros_initializer(),
                trainable=True,
        )
        self.built = True
        
    def call(self, inputs):
        return tf.compat.v1.nn.conv2d(inputs, filter=self.kernel, strides=[1,1,1,1], padding=self.padding,
                  dilations=self.dilation_rate) + self.bias
    
class DilatedResidualBlock(keras.layers.Layer):

    def __init__(self, filter_num: int, kernel_size: int, dilation: int, dropout: float):
        
        super(DilatedResidualBlock, self).__init__()

        self.conv1 = MyConv2d(filters=filter_num,
                              kernel_size=kernel_size,
                              padding="SAME",
                              dilation_rate=dilation)        
        self.norm1 = tfa.layers.normalizations.InstanceNormalization()
        
        self.dropout = keras.layers.Dropout(dropout)
        
        self.conv2 = MyConv2d(filters=filter_num,
                              kernel_size=kernel_size,
                              padding="SAME",
                              dilation_rate=dilation)           
        self.norm2 = tfa.layers.normalizations.InstanceNormalization()
        
    def call(self, inputs, training):
        
        shortcut = inputs
        
        inputs = self.conv1(inputs)
        inputs = self.norm1(inputs)
        inputs = tf.nn.elu(inputs)
        
        inputs = self.dropout(inputs, training=training)
        
        inputs = self.conv2(inputs)
        inputs = self.norm2(inputs)
        inputs = tf.nn.elu(inputs + shortcut)
        
        return inputs
    
def make_basic_block_layer(filter_num=64, num_layers=41, dropout=0.5):
    
    res_block = keras.Sequential()
    res_block.add(FirstBlock(filter_num=filter_num, kernel_size=5))
    
    dilation = 1
    for _ in range(num_layers):
        res_block.add(DilatedResidualBlock(filter_num=filter_num, kernel_size=3, dilation=dilation, dropout=dropout))
        dilation *= 2
        if dilation > 16:
            dilation = 1
            
    return res_block

def make_basic_block_layer2(filter_num=256, num_layers=21, dropout=0.5):
    
    res_block = keras.Sequential()
    dilation = 1
    for _ in range(num_layers):
        res_block.add(DilatedResidualBlock(filter_num=filter_num, kernel_size=3, dilation=dilation, dropout=dropout))
        dilation *= 2
        if dilation > 16:
            dilation = 1
            
    return res_block

class TRRosettaCNN(keras.Model):

    def __init__(self, filter_num=[64,256], num_layers=[41,21], dropout=0.5):
        
        super(TRRosettaCNN, self).__init__()


        self.feature_layer1 = make_basic_block_layer(filter_num=filter_num[0],
                                                     num_layers=num_layers[0],
                                                     dropout=dropout)
        self.feature_layer2 = make_basic_block_layer(filter_num=filter_num[0],
                                                     num_layers=num_layers[0],
                                                     dropout=dropout)
        self.feature_layer3 = make_basic_block_layer(filter_num=filter_num[0],
                                                     num_layers=num_layers[0],
                                                     dropout=dropout)
        self.feature_layer4 = make_basic_block_layer(filter_num=filter_num[0],
                                                     num_layers=num_layers[0],
                                                     dropout=dropout)
        
        self.feature_layer5 = make_basic_block_layer2(filter_num=filter_num[1],
                                                      num_layers=num_layers[1],
                                                      dropout=dropout)
        
        self.predict_theta = keras.layers.Conv2D(filters=25, kernel_size=1, padding='SAME')
        self.predict_phi = keras.layers.Conv2D(filters=13, kernel_size=1, padding='SAME')
        self.predict_dist = keras.layers.Conv2D(filters=37, kernel_size=1, padding='SAME')
        self.predict_omega = keras.layers.Conv2D(filters=25, kernel_size=1, padding='SAME')

    def call(self, x_cov, x_pre, x_plm, x_1d, training):
        
        cov = x_cov
        pre = x_pre
        plm = x_plm
        f1d = x_1d

        cov = self.feature_layer1(cov, training=training)
        pre = self.feature_layer2(pre, training=training)
        plm = self.feature_layer3(plm, training=training)
        f1d = self.feature_layer4(f1d, training=training)
    
        x = tf.concat([cov, pre, plm, f1d], -1)
        
        x = self.feature_layer5(x, training=training)
        
        # anglegrams for theta
        logits_theta = self.predict_theta(x)

        # anglegrams for phi
        logits_phi = self.predict_phi(x)

        # symmetrize
        sym_x = 0.5 * (x + tf.transpose(x, perm=[0, 2, 1, 3]))

        # distograms
        logits_dist = self.predict_dist(sym_x)

        # anglegrams for omega
        logits_omega = self.predict_omega(sym_x)        
 
    
        return logits_theta, logits_phi, logits_dist, logits_omega


    
