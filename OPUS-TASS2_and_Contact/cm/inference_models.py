# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import tensorflow as tf
from my_model import Model

#============================Parameters====================================
params = {}

params["d_output"] = 100
params["dropout"] = 0.5

#============================Models====================================

params["filter_num"] = [64,256]
params["num_layers"] = [61,21]

model1 = Model(params=params, name="n4")
model1.params["save_path"] = "./cm/models/l1"
model1.load_model() 

model2 = Model(params=params, name="n4")
model2.params["save_path"] = "./cm/models/l2"
model2.load_model() 

params["filter_num"] = [64,256]
params["num_layers"] = [41,21]

model3 = Model(params=params, name="n4")
model3.params["save_path"] = "./cm/models/m1"
model3.load_model() 

model4 = Model(params=params, name="n4")
model4.params["save_path"] = "./cm/models/m2"
model4.load_model() 

model5 = Model(params=params, name="n4")
model5.params["save_path"] = "./cm/models/m3"
model5.load_model() 

model6 = Model(params=params, name="n4")
model6.params["save_path"] = "./cm/models/m4"
model6.load_model() 

model7 = Model(params=params, name="n4")
model7.params["save_path"] = "./cm/models/m5"
model7.load_model() 

def test_infer_step(x_cov, x_pre, x_plm, x_1d):
    
    logits_predictions = []
    
    logits = model1.inference(x_cov, x_pre, x_plm, x_1d, y=None, training=False) 
    logits_predictions.append(logits)

    logits = model2.inference(x_cov, x_pre, x_plm, x_1d, y=None, training=False) 
    logits_predictions.append(logits)
    
    logits = model3.inference(x_cov, x_pre, x_plm, x_1d, y=None, training=False) 
    logits_predictions.append(logits)
    
    logits = model4.inference(x_cov, x_pre, x_plm, x_1d, y=None, training=False) 
    logits_predictions.append(logits)
    
    logits = model5.inference(x_cov, x_pre, x_plm, x_1d, y=None, training=False) 
    logits_predictions.append(logits)

    logits = model6.inference(x_cov, x_pre, x_plm, x_1d, y=None, training=False) 
    logits_predictions.append(logits)

    logits = model7.inference(x_cov, x_pre, x_plm, x_1d, y=None, training=False) 
    logits_predictions.append(logits)
    
    return logits_predictions