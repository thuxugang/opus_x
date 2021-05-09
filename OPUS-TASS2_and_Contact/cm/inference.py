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
    print ("Run OPUS-Contact...")
    
    test_reader = InputReader(data_list=preparation_config["data_list"],
                              preparation_config=preparation_config)
    
    for step, files_batch in enumerate(test_reader.dataset):

        filename, x_cov, x_pre, x_plm, x_1d, inputs_len = files_batch
        
        logits_predictions = test_infer_step(x_cov, x_pre, x_plm, x_1d)
            
        trrosetta_outputs = get_ensemble_ouput("TrRosetta", logits_predictions)            
            
        output_results(filename, trrosetta_outputs, preparation_config, inputs_len)
        
    run_time = time.time() - start_time
    print('OPUS-Contact prediction done..., time: %3.3f' % (run_time)) 
    #==================================Model===================================
    
    
    