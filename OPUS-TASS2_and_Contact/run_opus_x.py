# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import os
import tensorflow as tf
import warnings
import time
import json
from tass2.inference_utils import read_fasta, write_fasta, get_hhmaln, make_tass2_inputs
from cm.inference_utils import get_cov, get_pre, get_plm

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
if __name__ == '__main__':

    #============================Parameters====================================
    fasta_path = './casp14.fasta'
    files = read_fasta(fasta_path) 
    
    preparation_config = {}
    preparation_config["base_path"] = os.path.abspath('.')
    preparation_config["tmp_files_path"] = os.path.join(os.path.abspath('.'), "tmp_files")
    preparation_config["output_path"] = os.path.join(os.path.abspath('.'), "predictions")
    preparation_config["num_threads"] = 56

    #tass2
    preparation_config["hhsuite3_path"] = '/work/home/xugang/software/hh-suite3'
    preparation_config["hhmake_path"] = os.path.join(preparation_config["hhsuite3_path"], "build/bin/hhmake")
    
    preparation_config["tass2_batch_size"] = 1
    
    #cm
    preparation_config["contact_batch_size"] = 1
    preparation_config["script_path"] = os.path.join(os.path.abspath('.'), "mk_2dfeatures")
    
    #config_path
    preparation_config_json_file = os.path.join(os.path.abspath('.'), "config.json")
    
    #============================Parameters====================================
    
    
    #============================Preparation===================================
    
    start_time = time.time()
    print('Run preparation (pssm hhm deepmsa)...') 
    
    data_list = []
    for file in files:
        
        filename = file[0].split('.')[0]
        
        fasta_filename = filename + '.fasta'
        if not os.path.exists(os.path.join(preparation_config["tmp_files_path"], fasta_filename)):
            write_fasta(file, preparation_config)
            
        pssm_filename = filename + '.pssm'
        if not os.path.exists(os.path.join(preparation_config["tmp_files_path"], pssm_filename)):
            print ("Missing pssm file: " + pssm_filename)
            exit(-1)
        
        hhm_filename = filename + '.hhm'
        if not os.path.exists(os.path.join(preparation_config["tmp_files_path"], hhm_filename)):
            print ("Missing hhm file: " + hhm_filename)
            exit(-1)

        aln_filename = filename + '.aln'
        if not os.path.exists(os.path.join(preparation_config["tmp_files_path"], aln_filename)):
            print ("Missing deepmsa file: " + aln_filename)
            exit(-1)

        trrosetta_filename = filename + '.npz'
        if not os.path.exists(os.path.join(preparation_config["tmp_files_path"], trrosetta_filename)):
            print ("Missing trrosetta file: " + trrosetta_filename)
            exit(-1)
            
        hhmaln_filename = filename + '.hhm_aln'
        if not os.path.exists(os.path.join(preparation_config["tmp_files_path"], hhmaln_filename)):
            get_hhmaln(file, preparation_config)   
            
        make_tass2_inputs(file, preparation_config)
        data_list.append(filename)
    
    preparation_config["data_list"] = data_list
    with open(preparation_config_json_file,'w') as json_file:
        json.dump(preparation_config, json_file)
         
    run_time = time.time() - start_time
    print('Preparation (pssm hhm deepmsa) done..., time: %3.3f' % (run_time))  
    #============================Preparation===================================

    #============================OPUS-TASS2===================================
    cmd1 = "export CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"] + " && " + \
        "python -u tass2/inference.py --config_path " + preparation_config_json_file
    print (cmd1)
    success = os.system(cmd1)
    if success != 0:
        exit(success)
    #============================OPUS-TASS2===================================
    
    #============================Preparation===================================
    start_time = time.time()
    print('Run preparation (cov pre plm)...') 
    
    for file in files:
        
        filename = file[0].split('.')[0]

        with open(os.path.join(preparation_config["tmp_files_path"], filename + ".fasta"),'r') as r:
            results = [i.strip() for i in r.readlines()]
        length = len(results[1])  
        
        get_cov(filename, preparation_config, length)
        get_pre(filename, preparation_config, length)       
        get_plm(filename, preparation_config, length)    

    run_time = time.time() - start_time
    print('Preparation (cov pre plm) done..., time: %3.3f' % (run_time))  
    #============================Preparation===================================    
    
    #============================OPUS-Contact===================================    
    cmd2 = "export CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"] + " && " + \
        "python -u cm/inference.py --config_path " + preparation_config_json_file
    print (cmd2)
    success = os.system(cmd2)
    if success != 0:
        exit(success)
    #============================OPUS-Contact===================================    
    