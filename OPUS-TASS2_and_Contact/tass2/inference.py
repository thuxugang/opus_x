# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""
import warnings
import tensorflow as tf

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import os
import time
import argparse
import json
from inference_utils import InputReader, get_ensemble_ouput, output_results
from inference_models import test_infer_step

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
      
    args = parser.parse_args()

    config_path = args.config_path
    with open(config_path,'r') as json_file:
        preparation_config = json.load(json_file)

    #==================================Model===================================
    
    start_time = time.time()
    print ("Run OPUS-TASS2...")
    
    test_reader = InputReader(data_list=preparation_config["data_list"], 
                              num_batch_size=preparation_config["tass2_batch_size"],
                              inputs_files_path=preparation_config["tmp_files_path"])
    
    for step, filenames_batch in enumerate(test_reader.dataset):

        filenames, x, x_mask, x_trr, inputs_total_len = \
            test_reader.read_file_from_disk(filenames_batch)
        
        ss8_predictions, ss3_predictions, phi_predictions, psi_predictions, asa_predictions = \
            test_infer_step(x, x_mask, x_trr)
            
        ss8_outputs, _ = \
            get_ensemble_ouput("SS", ss8_predictions, x_mask, inputs_total_len)
            
        ss3_outputs, _ = \
            get_ensemble_ouput("SS", ss3_predictions, x_mask, inputs_total_len)
            
        phi_outputs, _ = \
            get_ensemble_ouput("PP", phi_predictions, x_mask, inputs_total_len)    

        psi_outputs, _ = \
            get_ensemble_ouput("PP", psi_predictions, x_mask, inputs_total_len) 

        asa_outputs, _ = \
            get_ensemble_ouput("ASA", asa_predictions, x_mask, inputs_total_len) 
            
        assert len(filenames) == len(ss8_outputs) == len(ss3_outputs) == \
            len(phi_outputs) == len(psi_outputs) == len(asa_outputs)
            
        output_results(filenames, ss8_outputs, ss3_outputs, phi_outputs, psi_outputs, asa_outputs, preparation_config)
        
    run_time = time.time() - start_time
    print('OPUS-TASS2 prediction done..., time: %3.3f' % (run_time)) 
    #==================================Model===================================
    
    
    