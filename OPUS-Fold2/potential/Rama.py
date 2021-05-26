# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:46:08 2016

@author: XuGang
"""

import tensorflow as tf
import numpy as np
from scipy import interpolate

def cal_RamaPotential(init_torsions, rama_consts):

    phipsis = tf.reshape(init_torsions, (rama_consts["n_samples"], 3))[:,:2]
    phipsis = tf.cast(phipsis, tf.float64)
    xa = tf.repeat(phipsis, rama_consts["n_dims"], axis=0)
    r = (tf.sqrt(tf.reduce_sum(tf.square(xa-rama_consts["total_xi"]), -1)))
    r = tf.reshape(r, (rama_consts["n_samples"], rama_consts["n_dims"]))
    r = tf.sqrt(tf.square(1.0/rama_consts["total_epsilon"]*r) + 1)
    
    rama_potential = tf.reduce_sum(r*rama_consts["total_nodes"], -1)     
    rama_potential = tf.reduce_mean(rama_potential)
    
    pro_index = rama_consts["pro_index"]
    case_pro = tf.squeeze(tf.where(pro_index == 1), -1)
    case_nonpro = tf.squeeze(tf.where(pro_index == 0), -1)
    assert pro_index.shape[0] == case_pro.shape[0] + case_nonpro.shape[0]
    
    omega = tf.reshape(init_torsions, (rama_consts["n_samples"], 3))[:,2]
    omega_pro = tf.gather(omega, case_pro)
    omega_nonpro = tf.gather(omega, case_nonpro)

    omega_pro_case1 = tf.squeeze(tf.where((-90<=omega_pro) & (omega_pro<=90)), -1)
    omega_pro_case2 = tf.squeeze(tf.where((omega_pro<-90) | (omega_pro>90)), -1)
    assert case_pro.shape[0] == omega_pro_case1.shape[0] + omega_pro_case2.shape[0]

    omega_nonpro_case1 = tf.squeeze(tf.where((-90<=omega_nonpro) & (omega_nonpro<=90)), -1)
    omega_nonpro_case2 = tf.squeeze(tf.where((omega_nonpro<-90) | (omega_nonpro>90)), -1)
    assert case_nonpro.shape[0] == omega_nonpro_case1.shape[0] + omega_nonpro_case2.shape[0]
    
    omega_potential = 0
    omega_potential += tf.reduce_sum(tf.math.cos(np.pi - 2*tf.gather(omega_pro, omega_pro_case1)/180*np.pi))
    omega_potential += tf.reduce_sum(2*tf.math.cos(np.pi - 2*tf.gather(omega_pro, omega_pro_case2)/180*np.pi) - 1)
    
    omega_potential += tf.reduce_sum(tf.math.cos(np.pi - 2*tf.gather(omega_nonpro, omega_nonpro_case1)/180*np.pi))
    omega_potential += tf.reduce_sum(5*tf.math.cos(np.pi - 2*tf.gather(omega_nonpro, omega_nonpro_case2)/180*np.pi) - 4)
    
    return tf.cast(rama_potential, tf.float32), omega_potential/pro_index.shape[0]

def readRama(path):
    
    ramas = []
    f = open(path, "r")
    for i in f.readlines():
        if i[0] != '#':
            content = i.strip().split()
            assert len(content) == 22
            ramas.append([float(i) for i in content])    
    f.close()
    ramas = np.array(ramas)
    
    rama_cons = {}
    n_dims = 36*36
    res_names = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
    for rama_id, res_name in zip(range(2, 22), res_names):
        
        phis = ramas[:,0]
        psis = ramas[:,1]
        energy = ramas[:,rama_id]
        
        rbf = interpolate.Rbf(phis, psis, energy)
        
        rama_cons[res_name] = {}
        rama_cons[res_name]["xi"] = np.array(rbf.xi.T, dtype=np.float64)
        rama_cons[res_name]["epsilon"] = np.array([rbf.epsilon]*n_dims, dtype=np.float64)
        rama_cons[res_name]["nodes"] = np.array(rbf.nodes, dtype=np.float64)
        rama_cons["n_dims"] = n_dims
    
    return rama_cons

def getRamaCons(rama_cons, fasta):
    
    rama_consts = {}
    
    total_xi = []
    total_epsilon = []
    total_nodes = []
    pro_index = []
    for seq in fasta:

        total_xi.append(rama_cons[seq]["xi"])
        total_epsilon.append(rama_cons[seq]["epsilon"])
        total_nodes.append(rama_cons[seq]["nodes"])
        
        if seq == "P":
            pro_index.append(1)
        else:
            pro_index.append(0)
            
    total_xi = np.concatenate(np.array(total_xi, dtype=np.float64), axis=0)
    total_epsilon = np.array(total_epsilon, dtype=np.float64)
    total_nodes = np.array(total_nodes, dtype=np.float64)
    pro_index = np.array(pro_index, dtype=np.int32)
    
    rama_consts["total_xi"] = total_xi
    rama_consts["total_epsilon"] = total_epsilon
    rama_consts["total_nodes"] = total_nodes
    rama_consts["n_dims"] = rama_cons["n_dims"]
    rama_consts["n_samples"] = len(fasta)
    rama_consts["pro_index"] = pro_index

    return rama_consts    
    