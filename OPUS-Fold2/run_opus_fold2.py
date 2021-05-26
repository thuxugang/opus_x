# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:11:13 2016

@author: XuGang
"""

import os
from myclass import Residues, Myio
from buildprotein import RebuildStructure
from potential import TrrPotential, Rama, Potentials
import time
import tensorflow as tf
import numpy as np
import multiprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = ""

def run_script(multi_iter):
    
    fasta_path, trr_cons_path, init_torsions, rama_cons, outputs = multi_iter
    
    if not os.path.exists(outputs):
    
        init_torsions = np.reshape(init_torsions, -1)
        num_torsions = len(init_torsions)
        
        name, fasta = Myio.readFasta(fasta_path)
        print ("Modeling: " + name)
        print (len(fasta), fasta)
        
        residuesData = Residues.getResidueDataFromSequence(fasta) 
        geosData = RebuildStructure.getGeosData(residuesData)
    
        assert num_torsions == 3*len(residuesData)
        
        atoms_matrix_init = np.zeros((5*len(residuesData), 3)).astype(np.float32)
        
        trr_cons = TrrPotential.readTrrCons(trr_cons_path)
        Trr_matrix = TrrPotential.init_Trr_matrix(residuesData, trr_cons)

        rama_cons = Rama.getRamaCons(rama_cons, fasta)

        init_torsions = [tf.Variable(i) for i in init_torsions]
          
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.5,
            decay_steps=300,
            decay_rate=0.96,
            staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
        best_potential = 1e6
        best_atoms_matrix = None
        for epoch in range(1000):
            with tf.GradientTape() as tape:
                start = time.time()
                for torsions in init_torsions:
                    if torsions > 180:
                        torsions.assign_sub(360)
                    elif torsions < -180:
                        torsions.assign_add(360) 
                atoms_matrix = RebuildStructure.rebuild_main_chain(init_torsions, geosData, residuesData)
                assert atoms_matrix_init.shape[0] == len(atoms_matrix)
                
                loss = Potentials.get_potentials(Trr_matrix, atoms_matrix, init_torsions, rama_cons)
                print ("Epoch:", epoch, loss.numpy())
    
                print (time.time() - start)
    
            gradients = tape.gradient(loss, init_torsions, unconnected_gradients="zero")
            
            optimizer.apply_gradients(zip(gradients, init_torsions))
    
            if loss.numpy() < best_potential:
                best_atoms_matrix = atoms_matrix
                best_potential = loss.numpy()
            
        Myio.outputPDB(residuesData, best_atoms_matrix, outputs)
    
if __name__ == '__main__':

    lists = []
    f = open('./list_casp14.txt')
    for i in f.readlines():
        lists.append(i.strip())
    f.close()    

    multi_iters = []
    for filename in lists:
        
        fasta_path = os.path.join("../OPUS-TASS2_and_Contact/tmp_files", 
                                  filename + ".fasta")
        trr_cons_path = os.path.join("../OPUS-TASS2_and_Contact/predictions", 
                                     filename + ".contact.npz")
        
        init_torsions = Myio.readTASS(os.path.join("../OPUS-TASS2_and_Contact/predictions", 
                                                   filename + ".tass2"))

        rama_cons = Rama.readRama("./lib/ramachandran.txt")

        outputs = "./predictions/" + filename + ".fold2"
        
        multi_iters.append([fasta_path, trr_cons_path, init_torsions, rama_cons, outputs])

    pool = multiprocessing.Pool(30)
    pool.map(run_script, multi_iters)
    pool.close()
    pool.join()  




        

