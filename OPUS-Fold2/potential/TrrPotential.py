# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:46:08 2016

@author: XuGang
"""

import tensorflow as tf
import numpy as np
from buildprotein import PeptideBuilder
from scipy import interpolate

bins_dist = np.array([ 0.  ,  2.25,  2.75,  3.25,  3.75,  4.25,  4.75,  5.25,  5.75,
                        6.25,  6.75,  7.25,  7.75,  8.25,  8.75,  9.25,  9.75, 10.25,
                       10.75, 11.25, 11.75, 12.25, 12.75, 13.25, 13.75, 14.25, 14.75,
                       15.25, 15.75, 16.25, 16.75, 17.25, 17.75, 18.25, 18.75, 19.25,
                       19.75])
bins_omega = np.array([-202.5, -187.5, -172.5, -157.5, -142.5, -127.5, -112.5,  -97.5,
                        -82.5,  -67.5,  -52.5,  -37.5,  -22.5,   -7.5,    7.5,   22.5,
                         37.5,   52.5,   67.5,   82.5,   97.5,  112.5,  127.5,  142.5,
                        157.5,  172.5,  187.5,  202.5])
bins_theta = np.array([-202.5, -187.5, -172.5, -157.5, -142.5, -127.5, -112.5,  -97.5,
                        -82.5,  -67.5,  -52.5,  -37.5,  -22.5,   -7.5,    7.5,   22.5,
                         37.5,   52.5,   67.5,   82.5,   97.5,  112.5,  127.5,  142.5,
                        157.5,  172.5,  187.5,  202.5])
bins_phi = np.array([-22.5,  -7.5,   7.5,  22.5,  37.5,  52.5,  67.5,  82.5,  97.5,
                     112.5, 127.5, 142.5, 157.5, 172.5, 187.5, 202.5])

def cal_TrrPotential(Trr_matrix, atoms_matrix):

    atoms_matrix = tf.cast(atoms_matrix, tf.float32)

    dist_i_cb = Trr_matrix["dist_i_cb"]
    dist_j_cb = Trr_matrix["dist_j_cb"]
    dist_p = Trr_matrix["dist_p"]
    dist_c = Trr_matrix["dist_c"]
    
    dist_i_cb_pos = tf.gather(atoms_matrix, dist_i_cb)
    dist_j_cb_pos = tf.gather(atoms_matrix, dist_j_cb)
    
    case_dist = tf.squeeze(tf.where(dist_p>=0.05), -1)
    
    dist_i_cb_pos = tf.gather(dist_i_cb_pos, case_dist)
    dist_j_cb_pos = tf.gather(dist_j_cb_pos, case_dist)
    dist_c = tf.gather(dist_c, case_dist)
    
    dij = tf.sqrt(tf.reduce_sum(tf.square(dist_i_cb_pos - dist_j_cb_pos), -1))

    dij = tf.where(dij<20, dij, 20)
    c_index = np.digitize(dij, bins_dist[1:]).astype(np.uint8)
    c_index = np.clip(c_index, 0, 35)
    c_x_base = bins_dist[c_index]
    num = c_index.shape[0]
    c_params = dist_c.numpy()[range(num), c_index, :]

    dist_potential = tf.reduce_mean(
        c_params[:,0]*tf.math.pow(dij-c_x_base,3) + c_params[:,1]*tf.math.pow(dij-c_x_base,2) + \
        c_params[:,2]*tf.math.pow(dij-c_x_base,1) + c_params[:,3], -1)

    omega_i_ca = Trr_matrix["omega_i_ca"]
    omega_i_cb = Trr_matrix["omega_i_cb"]
    omega_j_ca = Trr_matrix["omega_j_ca"]
    omega_j_cb = Trr_matrix["omega_j_cb"]
    omega_p = Trr_matrix["omega_p"]
    omega_c = Trr_matrix["omega_c"]

    omega_i_ca_pos = tf.gather(atoms_matrix, omega_i_ca)
    omega_i_cb_pos = tf.gather(atoms_matrix, omega_i_cb)
    omega_j_ca_pos = tf.gather(atoms_matrix, omega_j_ca)
    omega_j_cb_pos = tf.gather(atoms_matrix, omega_j_cb)    
    
    case_omega = tf.squeeze(tf.where(omega_p>=0.55), -1)
    
    omega_i_ca_pos = tf.gather(omega_i_ca_pos, case_omega)
    omega_i_cb_pos = tf.gather(omega_i_cb_pos, case_omega)
    omega_j_ca_pos = tf.gather(omega_j_ca_pos, case_omega)
    omega_j_cb_pos = tf.gather(omega_j_cb_pos, case_omega)
    omega_c = tf.gather(omega_c, case_omega)

    omegaij = PeptideBuilder.get_dihedral(omega_i_ca_pos, omega_i_cb_pos, omega_j_cb_pos, omega_j_ca_pos)

    c_index = np.digitize(omegaij, bins_omega[1:]).astype(np.uint8)
    c_x_base = bins_omega[c_index]
    num = c_index.shape[0]
    c_params = omega_c.numpy()[range(num), c_index, :]
    
    omega_potential = tf.reduce_mean(
        c_params[:,0]*tf.math.pow(omegaij-c_x_base,3) + c_params[:,1]*tf.math.pow(omegaij-c_x_base,2) + \
        c_params[:,2]*tf.math.pow(omegaij-c_x_base,1) + c_params[:,3], -1)    
    

    theta_i_n = Trr_matrix["theta_i_n"]
    theta_i_ca = Trr_matrix["theta_i_ca"]
    theta_i_cb = Trr_matrix["theta_i_cb"]
    theta_j_cb = Trr_matrix["theta_j_cb"]
    theta_p = Trr_matrix["theta_p"]
    theta_c = Trr_matrix["theta_c"]

    theta_i_n_pos = tf.gather(atoms_matrix, theta_i_n)
    theta_i_ca_pos = tf.gather(atoms_matrix, theta_i_ca)
    theta_i_cb_pos = tf.gather(atoms_matrix, theta_i_cb)
    theta_j_cb_pos = tf.gather(atoms_matrix, theta_j_cb)    
    
    case_theta = tf.squeeze(tf.where(theta_p>=0.55), -1)
    
    theta_i_n_pos = tf.gather(theta_i_n_pos, case_theta)
    theta_i_ca_pos = tf.gather(theta_i_ca_pos, case_theta)
    theta_i_cb_pos = tf.gather(theta_i_cb_pos, case_theta)
    theta_j_cb_pos = tf.gather(theta_j_cb_pos, case_theta)
    theta_c = tf.gather(theta_c, case_theta)

    thetaij = PeptideBuilder.get_dihedral(theta_i_n_pos, theta_i_ca_pos, theta_i_cb_pos, theta_j_cb_pos)

    c_index = np.digitize(thetaij, bins_theta[1:]).astype(np.uint8)
    c_x_base = bins_theta[c_index]
    num = c_index.shape[0]
    c_params = theta_c.numpy()[range(num), c_index, :]
    
    theta_potential = tf.reduce_mean(
        c_params[:,0]*tf.math.pow(thetaij-c_x_base,3) + c_params[:,1]*tf.math.pow(thetaij-c_x_base,2) + \
        c_params[:,2]*tf.math.pow(thetaij-c_x_base,1) + c_params[:,3], -1)  
        
    
    phi_i_ca = Trr_matrix["phi_i_ca"]
    phi_i_cb = Trr_matrix["phi_i_cb"]
    phi_j_cb = Trr_matrix["phi_j_cb"]
    phi_p = Trr_matrix["phi_p"]
    phi_c = Trr_matrix["phi_c"]

    phi_i_ca_pos = tf.gather(atoms_matrix, phi_i_ca)
    phi_i_cb_pos = tf.gather(atoms_matrix, phi_i_cb)
    phi_j_cb_pos = tf.gather(atoms_matrix, phi_j_cb)
    
    case_phi = tf.squeeze(tf.where(phi_p>=0.65), -1)
    
    phi_i_ca_pos = tf.gather(phi_i_ca_pos, case_phi)
    phi_i_cb_pos = tf.gather(phi_i_cb_pos, case_phi)
    phi_j_cb_pos = tf.gather(phi_j_cb_pos, case_phi)
    phi_c = tf.gather(phi_c, case_phi)

    phiij = PeptideBuilder.get_angle(phi_i_ca_pos, phi_i_cb_pos, phi_j_cb_pos)

    c_index = np.digitize(phiij, bins_phi[1:]).astype(np.uint8)
    c_x_base = bins_phi[c_index]
    num = c_index.shape[0]
    c_params = phi_c.numpy()[range(num), c_index, :]
    
    phi_potential = tf.reduce_mean(
        c_params[:,0]*tf.math.pow(phiij-c_x_base,3) + c_params[:,1]*tf.math.pow(phiij-c_x_base,2) + \
        c_params[:,2]*tf.math.pow(phiij-c_x_base,1) + c_params[:,3], -1)          
        
    return dist_potential, omega_potential, theta_potential, phi_potential

def init_Trr_matrix(residuesData, trr_cons):

    dist_cons = trr_cons["dist"]
    dist_i_cb = []
    dist_j_cb = []
    dist_p = []
    dist_c = []
    for i, j, p, c in dist_cons:
        res_i_cb = 5*(i+1) - 1
        res_j_cb = 5*(j+1) - 1
        dist_i_cb.append(res_i_cb)
        dist_j_cb.append(res_j_cb)
        dist_p.append(p)
        dist_c.append(c)

    omega_cons = trr_cons["omega"]
    omega_i_ca = []
    omega_i_cb = []
    omega_j_ca = []
    omega_j_cb = []
    omega_p = []
    omega_c = []
    for i, j, p, c in omega_cons:
        res_i_ca = 5*(i+1) - 4
        res_i_cb = 5*(i+1) - 1
        res_j_ca = 5*(j+1) - 4
        res_j_cb = 5*(j+1) - 1
        omega_i_ca.append(res_i_ca)
        omega_i_cb.append(res_i_cb)
        omega_j_ca.append(res_j_ca)
        omega_j_cb.append(res_j_cb)
        omega_p.append(p)
        omega_c.append(c)

    theta_cons = trr_cons["theta"]
    theta_i_n = []
    theta_i_ca = []
    theta_i_cb = []
    theta_j_cb = []
    theta_p = []
    theta_c = []
    for i, j, p, c in theta_cons:
        res_i_n = 5*(i+1) - 5
        res_i_ca = 5*(i+1) - 4
        res_i_cb = 5*(i+1) - 1
        res_j_cb = 5*(j+1) - 1
        theta_i_n.append(res_i_n)
        theta_i_ca.append(res_i_ca)
        theta_i_cb.append(res_i_cb)
        theta_j_cb.append(res_j_cb)
        theta_p.append(p)
        theta_c.append(c)              

    phi_cons = trr_cons["phi"]
    phi_i_ca = []
    phi_i_cb = []
    phi_j_cb = []
    phi_p = []
    phi_c = []
    for i, j, p, c in phi_cons:
        res_i_ca = 5*(i+1) - 4
        res_i_cb = 5*(i+1) - 1
        res_j_cb = 5*(j+1) - 1
        phi_i_ca.append(res_i_ca)
        phi_i_cb.append(res_i_cb)
        phi_j_cb.append(res_j_cb)
        phi_p.append(p)
        phi_c.append(c)  


    Trr_matrix = {}
    Trr_matrix["dist_i_cb"] = np.array(dist_i_cb, dtype=np.int32)
    Trr_matrix["dist_j_cb"] = np.array(dist_j_cb, dtype=np.int32)
    Trr_matrix["dist_p"] = np.array(dist_p, dtype=np.float32)
    Trr_matrix["dist_c"] = np.array(dist_c, dtype=np.float32)

    Trr_matrix["omega_i_ca"] = np.array(omega_i_ca, dtype=np.int32)
    Trr_matrix["omega_i_cb"] = np.array(omega_i_cb, dtype=np.int32)
    Trr_matrix["omega_j_ca"] = np.array(omega_j_ca, dtype=np.int32)
    Trr_matrix["omega_j_cb"] = np.array(omega_j_cb, dtype=np.int32)
    Trr_matrix["omega_p"] = np.array(omega_p, dtype=np.float32)
    Trr_matrix["omega_c"] = np.array(omega_c, dtype=np.float32)
    
    Trr_matrix["theta_i_n"] = np.array(theta_i_n, dtype=np.int32)
    Trr_matrix["theta_i_ca"] = np.array(theta_i_ca, dtype=np.int32)
    Trr_matrix["theta_i_cb"] = np.array(theta_i_cb, dtype=np.int32)
    Trr_matrix["theta_j_cb"] = np.array(theta_j_cb, dtype=np.int32)
    Trr_matrix["theta_p"] = np.array(theta_p, dtype=np.float32)
    Trr_matrix["theta_c"] = np.array(theta_c, dtype=np.float32)

    Trr_matrix["phi_i_ca"] = np.array(phi_i_ca, dtype=np.int32)
    Trr_matrix["phi_i_cb"] = np.array(phi_i_cb, dtype=np.int32)
    Trr_matrix["phi_j_cb"] = np.array(phi_j_cb, dtype=np.int32)
    Trr_matrix["phi_p"] = np.array(phi_p, dtype=np.float32)
    Trr_matrix["phi_c"] = np.array(phi_c, dtype=np.float32)
    
    return Trr_matrix
    
def readTrrCons(path):

    files = np.load(path)
    
    trr_consts = {}
    
    PCUT = 0.05
    DSTEP = 0.5
    DCUT = 19.5
    ALPHA = 1.57
    MEFF = 0.0001
    EBASE = -0.5
    EREP = [10, 3,    1.5,  0.5,  0]
    DREP = [0,  2.25, 2.75, 3.25, 3.75]
    
    dist = files['dist']
    trr_consts['dist'] = []
    ########################################################
    # dist: 0..20A
    ########################################################
    nres = dist.shape[0]
    bins = np.array([4.25+DSTEP*i for i in range(32)])
    prob = np.sum(dist[:,:,5:], axis=-1)
    bkgr = np.array((bins/DCUT)**ALPHA)
    attr = -np.log((dist[:,:,5:]+MEFF)/(dist[:,:,-1][:,:,None]*bkgr[None,None,:]))+EBASE
    repul = np.maximum(attr[:,:,0],np.zeros((nres,nres)))[:,:,None]+np.array(EREP)[None,None,:]
    dist = np.concatenate([repul,attr], axis=-1)
    bins = np.concatenate([DREP,bins])
    i,j = np.where(prob>PCUT)
    prob = prob[i,j]
    
    # 37 bin
    counter = 0
    for a,b,p in zip(i,j,prob):
        if b>a:
            trr_consts['dist'].append([a, b, p, interpolate.CubicSpline(bins, dist[a,b]).c.T])
            counter += 1
                
    print("dist restraints:  %d"%(counter))

    omega = files['omega']
    trr_consts['omega'] = []
    ########################################################
    # omega: -pi..pi
    ########################################################
    nbins = omega.shape[2] - 1 + 4
    bins = np.linspace(-202.5, 202.5, nbins)
    prob = np.sum(omega[:,:,1:], axis=-1)
    i,j = np.where(prob>PCUT)
    prob = prob[i,j]
    omega = -np.log((omega+MEFF)/(omega[:,:,-1]+MEFF)[:,:,None])
    # for better smooth
    omega = np.concatenate([omega[:,:,-2:],omega[:,:,1:],omega[:,:,1:3]],axis=-1)
    
    # 28 bin
    counter = 0
    for a,b,p in zip(i,j,prob):
        if b>a:
            trr_consts['omega'].append([a, b, p, interpolate.CubicSpline(bins, omega[a,b]).c.T])
            counter += 1
    print("omega restraints: %d"%(counter))
    
    theta = files['theta']
    trr_consts['theta'] = []
    ########################################################
    # theta: -pi..pi
    ########################################################
    prob = np.sum(theta[:,:,1:], axis=-1)
    i,j = np.where(prob>PCUT)
    prob = prob[i,j]
    theta = -np.log((theta+MEFF)/(theta[:,:,-1]+MEFF)[:,:,None])
    theta = np.concatenate([theta[:,:,-2:],theta[:,:,1:],theta[:,:,1:3]],axis=-1)
    
    # 28 bin
    counter = 0
    for a,b,p in zip(i,j,prob):
        if b!=a:
            trr_consts['theta'].append([a, b, p, interpolate.CubicSpline(bins, theta[a,b]).c.T])
            counter += 1
    print("theta restraints: %d"%(counter))
    
    phi = files['phi']
    trr_consts['phi'] = []
    ########################################################
    # phi: 0..pi
    ########################################################
    nbins = phi.shape[2] - 1 + 4
    bins = np.linspace(-22.5, 202.5, nbins)
    prob = np.sum(phi[:,:,1:], axis=-1)
    i,j = np.where(prob>PCUT)
    prob = prob[i,j]
    phi = -np.log((phi+MEFF)/(phi[:,:,-1]+MEFF)[:,:,None])
    phi = np.concatenate([np.flip(phi[:,:,1:3],axis=-1),phi[:,:,1:],np.flip(phi[:,:,-2:],axis=-1)], axis=-1)
    
    # 16 bin
    counter = 0
    for a,b,p in zip(i,j,prob):
        if b!=a:
            trr_consts['phi'].append([a, b, p, interpolate.CubicSpline(bins, phi[a,b]).c.T])
            counter += 1
    print("phi restraints: %d"%(counter))
    
    return trr_consts

        
    
    