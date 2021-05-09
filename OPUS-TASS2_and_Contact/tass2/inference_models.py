# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

from my_model import Model
import tensorflow as tf

#============================Parameters====================================
params = {}
params["d_input"] = 170
params["d_ss8_output"] = 8
params["d_ss3_output"] = 3
params["d_phipsi_output"] = 4
params["d_csf_output"] = 3
params["d_asa_output"] = 1
params["dropout_rate"] = 0.25
  
#parameters of transfomer model
params["transfomer_layers"] = 2
params["transfomer_num_heads"] = 2

#parameters of birnn model
params["lstm_layers"] = 4
params["lstm_units"] = 1024

#============================Models====================================

model1 = Model(params=params, name="c4")
model1.params["save_path"] = "./tass2/models/1"
model1.load_model()

model2 = Model(params=params, name="c4")
model2.params["save_path"] = "./tass2/models/2"
model2.load_model()

model3 = Model(params=params, name="c4")
model3.params["save_path"] = "./tass2/models/3"
model3.load_model()

model4 = Model(params=params, name="c4")
model4.params["save_path"] = "./tass2/models/4"
model4.load_model()

model5 = Model(params=params, name="c4")
model5.params["save_path"] = "./tass2/models/5"
model5.load_model()

model6 = Model(params=params, name="c4")
model6.params["save_path"] = "./tass2/models/6"
model6.load_model()

model7 = Model(params=params, name="c4")
model7.params["save_path"] = "./tass2/models/7"
model7.load_model()

model8 = Model(params=params, name="c4")
model8.params["save_path"] = "./tass2/models/8"
model8.load_model()

model9 = Model(params=params, name="c4")
model9.params["save_path"] = "./tass2/models/9"
model9.load_model()


def test_infer_step(x, x_mask, x_trr):
    
    ss8_predictions = []
    ss3_predictions = []
    phi_predictions = []
    psi_predictions = []
    asa_predictions = []
      
    ss8_prediction, ss3_prediction, phipsi_prediction, _, asa_prediction = \
        model1.inference(x, x_mask, x_trr, y=None, y_mask=None, training=False)        
    ss8_predictions.append(ss8_prediction)
    ss3_predictions.append(ss3_prediction)
    phi_predictions.append(phipsi_prediction[:,:,:2])
    psi_predictions.append(phipsi_prediction[:,:,2:])
    asa_predictions.append(asa_prediction)

    ss8_prediction, ss3_prediction, phipsi_prediction, _, asa_prediction = \
        model2.inference(x, x_mask, x_trr, y=None, y_mask=None, training=False)        
    ss8_predictions.append(ss8_prediction)
    ss3_predictions.append(ss3_prediction)
    phi_predictions.append(phipsi_prediction[:,:,:2])
    psi_predictions.append(phipsi_prediction[:,:,2:])
    asa_predictions.append(asa_prediction)
    
    ss8_prediction, ss3_prediction, phipsi_prediction, _, asa_prediction = \
        model3.inference(x, x_mask, x_trr, y=None, y_mask=None, training=False)        
    ss8_predictions.append(ss8_prediction)
    ss3_predictions.append(ss3_prediction)
    phi_predictions.append(phipsi_prediction[:,:,:2])
    psi_predictions.append(phipsi_prediction[:,:,2:])
    asa_predictions.append(asa_prediction)
    
    ss8_prediction, ss3_prediction, phipsi_prediction, _, asa_prediction = \
        model4.inference(x, x_mask, x_trr, y=None, y_mask=None, training=False)        
    ss8_predictions.append(ss8_prediction)
    ss3_predictions.append(ss3_prediction)
    phi_predictions.append(phipsi_prediction[:,:,:2])
    psi_predictions.append(phipsi_prediction[:,:,2:])
    asa_predictions.append(asa_prediction)
    
    ss8_prediction, ss3_prediction, phipsi_prediction, _, asa_prediction = \
        model5.inference(x, x_mask, x_trr, y=None, y_mask=None, training=False)        
    ss8_predictions.append(ss8_prediction)
    ss3_predictions.append(ss3_prediction)
    phi_predictions.append(phipsi_prediction[:,:,:2])
    psi_predictions.append(phipsi_prediction[:,:,2:])
    asa_predictions.append(asa_prediction)
    
    ss8_prediction, ss3_prediction, phipsi_prediction, _, asa_prediction = \
        model6.inference(x, x_mask, x_trr, y=None, y_mask=None, training=False)        
    ss8_predictions.append(ss8_prediction)
    ss3_predictions.append(ss3_prediction)
    phi_predictions.append(phipsi_prediction[:,:,:2])
    psi_predictions.append(phipsi_prediction[:,:,2:])
    asa_predictions.append(asa_prediction)
    
    ss8_prediction, ss3_prediction, phipsi_prediction, _, asa_prediction = \
        model7.inference(x, x_mask, x_trr, y=None, y_mask=None, training=False)        
    ss8_predictions.append(ss8_prediction)
    ss3_predictions.append(ss3_prediction)
    phi_predictions.append(phipsi_prediction[:,:,:2])
    psi_predictions.append(phipsi_prediction[:,:,2:])
    asa_predictions.append(asa_prediction)
    
    ss8_prediction, ss3_prediction, phipsi_prediction, _, asa_prediction = \
        model8.inference(x, x_mask, x_trr, y=None, y_mask=None, training=False)        
    ss8_predictions.append(ss8_prediction)
    ss3_predictions.append(ss3_prediction)
    phi_predictions.append(phipsi_prediction[:,:,:2])
    psi_predictions.append(phipsi_prediction[:,:,2:])
    asa_predictions.append(asa_prediction)
    
    ss8_prediction, ss3_prediction, phipsi_prediction, _, asa_prediction = \
        model9.inference(x, x_mask, x_trr, y=None, y_mask=None, training=False)        
    ss8_predictions.append(ss8_prediction)
    ss3_predictions.append(ss3_prediction)
    phi_predictions.append(phipsi_prediction[:,:,:2])
    psi_predictions.append(phipsi_prediction[:,:,2:])
    asa_predictions.append(asa_prediction)


    return ss8_predictions, ss3_predictions, phi_predictions, psi_predictions, asa_predictions