# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import os
import tensorflow as tf
from my_cnn import TRRosettaCNN

class Model(object):
    
    def __init__(self, params, name):
        
        self.params = params
        self.name = name
    
        self.cnn = TRRosettaCNN(filter_num=self.params["filter_num"], 
                                num_layers=self.params["num_layers"], 
                                dropout=self.params["dropout"])

    def inference(self, x_cov, x_pre, x_plm, x_1d, y=None, training=False):

        """
        "theta",0,25
        "phi",25,38
        "dist",38,75
        "omega",75,100
        """
            
        logits = {}
        logits_theta, logits_phi, logits_dist, logits_omega = self.cnn(x_cov, x_pre, x_plm, x_1d, training=training)
        
        logits["theta"]= logits_theta
        logits["phi"] = logits_phi
        logits["dist"] = logits_dist
        logits["omega"] = logits_omega
        
        return logits
    
    def load_model(self):
        print ("load model:", self.name, self.params["save_path"])
        self.cnn.load_weights(os.path.join(self.params["save_path"], self.name + '_model_weight'))



