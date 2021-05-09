# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import shutil
from io import BytesIO

def check_exists(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return 1
    else:
        return 0

def blockshaped(arr,dim=21):
    p=arr.shape[0]//dim
    re=np.zeros([dim*dim,p,p])
    for i in range(p):
        for j in range(p):
            re[:,i,j]=arr[i*dim:i*dim+dim,j*dim:j*dim+dim].flatten()
    return re

def readfromsavefile(raw_path):
    lines=open(raw_path).readlines()
    L=0
    for i in range(len(lines)):
        if lines[i][0]=='#':
            L=i
            break
    print(L)
    precision=np.zeros([L*21,L*21])
    #first:
    for i in range(L):
        vec=np.genfromtxt(BytesIO(lines[i].encode()))
        for j in range(21):
            precision[i*21+j,i*21+j]=0
    count=0;
    for i in range(L):
        for k in range(i+1,L):
            count+=1
            for j in range(21):
                vec=np.genfromtxt(BytesIO(lines[L+22*(count-1)+1+j].encode())).reshape([1,-1])
                precision[i*21+j,k*21:k*21+21]=vec
                precision[k*21:k*21+21,i*21+j:i*21+j+1]=np.transpose(vec)
    #cleaning    
    os.remove(raw_path)
    return blockshaped(precision)

def get_cov(filename, preparation_config, length):
    
    cov_path = os.path.join(preparation_config["tmp_files_path"], filename + ".cov")
    if check_exists(cov_path):
        print("cov generated.....skip")
    else:
        cmd = os.path.join(preparation_config["script_path"], "cov21stats") + " " + \
            os.path.join(preparation_config["tmp_files_path"], filename + ".aln") + " " + \
            cov_path
        print (cmd)
        os.system(cmd)
        
        if check_exists(cov_path):
            print("cov generated successfully....")
        else:
            print("cov generation failed....")
        
    data = np.fromfile(cov_path, dtype=np.float32)
    assert data.shape[0] == length*length*441

def get_pre(filename, preparation_config, length):
    
    pre_path = os.path.join(preparation_config["tmp_files_path"], filename + ".pre")
    if check_exists(pre_path):
        print("pre generated.....skip")
    else:
        cmd1 = os.path.join(preparation_config["script_path"], "calNf_ly") + " " + \
            os.path.join(preparation_config["tmp_files_path"], filename + ".aln") + " 0.8 > " + \
            os.path.join(preparation_config["tmp_files_path"], filename+".weight")
        print (cmd1)
        os.system(cmd1)
        
        cmd2 = "python -W ignore " + os.path.join(preparation_config["script_path"], "generate_pre.py") + " " + \
            os.path.join(preparation_config["tmp_files_path"], filename + ".aln") + " " + \
            os.path.join(preparation_config["tmp_files_path"], filename)
        print (cmd2)
        os.system(cmd2)        

        os.system("rm " + os.path.join(preparation_config["tmp_files_path"], filename+".weight"))

        if check_exists(pre_path):
            print("pre generated successfully....")
        else:
            print("pre generation failed....")
            
    data = np.fromfile(pre_path, dtype=np.float32)
    assert data.shape[0] == length*length*441

def get_plm(filename, preparation_config, length):
    
    ccmpred_path = os.path.join(preparation_config["tmp_files_path"], filename + ".ccmpred")
    plm_path = os.path.join(preparation_config["tmp_files_path"], filename + ".plm.npy")
    if check_exists(ccmpred_path) and check_exists(plm_path):
        print("ccmpred&plm generated.....skip")
    else:
        cmd = os.path.join(preparation_config["script_path"], "ccmpred") + " -d 0 -r " + \
            os.path.join(preparation_config["tmp_files_path"], filename + ".raw") + " " + \
            os.path.join(preparation_config["tmp_files_path"], filename + ".aln") + " " + \
            ccmpred_path
        print (cmd)
        os.system(cmd)
    
        plm = readfromsavefile(os.path.join(preparation_config["tmp_files_path"], filename + ".raw"))
        np.save(os.path.join(preparation_config["tmp_files_path"], filename + ".plm"), plm.astype(np.float32))

    data = np.load(plm_path)
    assert data.shape == (441,length,length)
    data = read_ccmpred(ccmpred_path)
    assert data.shape == (length, length, 1)

#=============================================================================    
ss8_str = "CSTHGIEB"
ss8_dict = {}
for k,v in enumerate(ss8_str):
    ss8_dict[v] = k

ss3_str = "CHE"
ss3_dict = {}
for k,v in enumerate(ss3_str):
    ss3_dict[v] = k
    
def read_tass2_results(path):
    ss3s = ''
    ss8s = ''
    phis = []
    psis = []
    asas = []
    with open(path,'r') as r:
        for i in r.readlines():
            if i.strip().split()[0][0] == '#':
                continue
            else:
                context = i.strip().split()
                assert len(context) == 17
                phis.append(float(context[3]))
                psis.append(float(context[4]))
                asas.append(float(context[5]))
                ss3s += context[1]
                ss8s += context[2]
                
    tas1_results = []
    for ss3, ss8, phi, psi, asa in zip(ss3s, ss8s, phis, psis, asas):

        ss3_f = np.zeros(3)
        ss3_f[ss3_dict[ss3]] = 1
        
        ss8_f = np.zeros(8)
        ss8_f[ss8_dict[ss8]] = 1
        
        phi_f = np.array([np.sin(np.deg2rad(phi)), np.cos(np.deg2rad(phi))])    
        psi_f = np.array([np.sin(np.deg2rad(psi)), np.cos(np.deg2rad(psi))])   
    
        asa_f = np.array([asa/100])
        
        feature = np.concatenate([ss3_f, ss8_f, phi_f, psi_f, asa_f])
        
        tas1_results.append(feature)
        
    return np.array(tas1_results)

def read_ccmpred(fname):
    with open(fname,'r') as f:
        feat_data = pd.read_csv(f,delim_whitespace=True,header=None).values.astype(float)
    feat_data = feat_data[:,:,None] if np.ndim(feat_data) == 2 else feat_data
    return feat_data+np.transpose(feat_data,[1,0,2])

def read_inputs(filename, preparation_config, model_type):
    """
    2d inputs
    (len, len, 441)
    """
    if model_type == "cov":
        # print ("cov")
        cov_inputs_ = np.fromfile(os.path.join(preparation_config["tmp_files_path"], filename + ".cov"), dtype=np.float32)
        seq_len = (int)(np.sqrt(cov_inputs_.shape[0]/441))
        cov_inputs_ = cov_inputs_.reshape(441, seq_len, seq_len)
        cov_inputs_ = cov_inputs_.transpose((1,2,0))
        inputs_ = cov_inputs_
    elif model_type == "pre":
        # print ("pre")
        pre_inputs_ = np.fromfile(os.path.join(preparation_config["tmp_files_path"], filename + ".pre"), dtype=np.float32)
        seq_len = (int)(np.sqrt(pre_inputs_.shape[0]/441))
        pre_inputs_ = pre_inputs_.reshape(441, seq_len, seq_len)
        pre_inputs_ = pre_inputs_.transpose((1,2,0))
        inputs_ = pre_inputs_
    elif model_type == "plm":
        # print ("plm")
        plm_inputs_ = np.load(os.path.join(preparation_config["tmp_files_path"], filename + ".plm.npy"))
        plm_inputs_ = plm_inputs_.transpose((1,2,0))
        inputs_ = plm_inputs_     
    elif model_type == "1d":
        # print ("1d")
        tas_inputs1 = np.loadtxt(os.path.join(preparation_config["tmp_files_path"], filename + ".tass2_inputs"))
        tas_inputs1 = tas_inputs1[:,:76]
        tas_inputs1[:, 20:50] = (tas_inputs1[:, 20:50] - 5000)/1000
        
        tas1_results = read_tass2_results(os.path.join(preparation_config["output_path"], filename + ".tass2"))
        inputs_1d_ = np.concatenate([tas_inputs1, tas1_results], axis=-1)
        
        inputs_2d_ = read_ccmpred(os.path.join(preparation_config["tmp_files_path"], filename + '.ccmpred'))
        
        trr_inputs1 = np.load(os.path.join(preparation_config["tmp_files_path"], filename + ".npz"))
        trr_inputs1 = np.concatenate([trr_inputs1['theta'], trr_inputs1['phi'], 
                                  trr_inputs1["dist"], trr_inputs1['omega']], axis=-1)

        ncol = inputs_1d_.shape[0]
        inputs_1d_ = np.concatenate([np.tile(inputs_1d_[:,None,:], [1,ncol,1]), 
                np.tile(inputs_1d_[None,:,:], [ncol,1,1])], axis=-1)
        
        inputs_ = np.concatenate([inputs_1d_, inputs_2d_, trr_inputs1], axis=-1)       

    elif model_type == "trr":

        trr_inputs1 = np.load(os.path.join(preparation_config["tmp_files_path"], filename + ".npz"))
        inputs_ = np.concatenate([trr_inputs1['theta'], trr_inputs1['phi'], 
                                  trr_inputs1["dist"], trr_inputs1['omega']], axis=-1)
        
    inputs_len = inputs_.shape[0]
    if model_type == "1d": 
        assert inputs_.shape == (inputs_len, inputs_len, 285)
    else:
        assert inputs_.shape == (inputs_len, inputs_len, 441)
    
    return inputs_, inputs_len

class InputReader(object):

    def __init__(self, data_list, preparation_config):

        self.filenames = data_list
        
        self.preparation_config = preparation_config
        
        self.dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        
        self.dataset = self.dataset.map(lambda x: tf.py_function(func=self.read_file_from_disk,
            inp=[x], Tout=[tf.string, tf.float32, tf.float32, tf.float32,
            tf.float32, tf.int32]), num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
        
        self.dataset = self.dataset.batch(preparation_config["contact_batch_size"])
            
        self.dataset = self.dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        print ("Data Size:", len(self.filenames)) 

    def read_file_from_disk(self, filename):

        filename = bytes.decode(filename.numpy())

        inputs_cov, inputs_len_cov = \
            read_inputs(filename, self.preparation_config, "cov")
        inputs_pre, inputs_len_pre = \
            read_inputs(filename, self.preparation_config, "pre")
        inputs_plm, inputs_len_plm = \
            read_inputs(filename, self.preparation_config, "plm")
        inputs_1d, inputs_len_1d = \
            read_inputs(filename, self.preparation_config, "1d")
        
        assert inputs_len_cov == inputs_len_pre == inputs_len_plm == inputs_len_1d
        inputs_len = inputs_len_cov
        
        inputs_cov = tf.convert_to_tensor(inputs_cov, dtype=tf.float32)
        inputs_pre = tf.convert_to_tensor(inputs_pre, dtype=tf.float32)
        inputs_plm = tf.convert_to_tensor(inputs_plm, dtype=tf.float32)
        inputs_1d = tf.convert_to_tensor(inputs_1d, dtype=tf.float32)
        
        return filename, inputs_cov, inputs_pre, inputs_plm, inputs_1d, inputs_len
            
#=============================================================================    

def get_ensemble_ouput(name, logits_predictions):
    
    if name == "TrRosetta":
            
        trrosetta_outputs = {}

        for key in ["theta", "phi", "dist", "omega"]:
            
            softmax_prediction = tf.nn.softmax(logits_predictions[0][key])
            tmp = [softmax_prediction.numpy()]
            
            if len(logits_predictions) > 1:
                for i in logits_predictions[1:]:
                    tmp.append(
                        tf.nn.softmax(i[key]).numpy())
            
            trrosetta_outputs[key] = np.mean(tmp, axis=0)

        return trrosetta_outputs
    
def output_results(filename, trrosetta_outputs, preparation_config, inputs_len):
    
    inputs_len = inputs_len.numpy()[0]
    filename = filename.numpy()[0]
    filename = bytes.decode(filename)
    
    assert trrosetta_outputs['dist'].shape == (1, inputs_len, inputs_len, 37)
    assert trrosetta_outputs['omega'].shape == (1, inputs_len, inputs_len, 25)
    assert trrosetta_outputs['theta'].shape == (1, inputs_len, inputs_len, 25)
    assert trrosetta_outputs['phi'].shape == (1, inputs_len, inputs_len, 13)

    np.savez_compressed(os.path.join(preparation_config["output_path"], filename + ".contact"), 
                        dist=trrosetta_outputs['dist'][0].astype(np.float32), 
                        omega=trrosetta_outputs['omega'][0].astype(np.float32), 
                        theta=trrosetta_outputs['theta'][0].astype(np.float32), 
                        phi=trrosetta_outputs['phi'][0].astype(np.float32))
    
    
    
    