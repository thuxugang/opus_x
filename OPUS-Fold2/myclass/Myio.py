# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:32:13 2015

@author: XuGang
"""

from myclass import Residues
import numpy as np    

def readTASS(path):
    
    init_torsions = []
    with open(path,'r') as r:
        for i in r.readlines():
            if i.strip().split()[0][0] == '#':
                continue
            else:
                context = i.strip().split()
                assert len(context) == 17
                phi = float(context[3])
                psi = float(context[4])
                omega = 180.0
                init_torsions.extend([phi, psi, omega])
                
    return np.array(init_torsions, dtype=np.float32)

def readFasta(path):
    
    with open(path,'r') as r:
        results = [i.strip() for i in r.readlines()]
    return results[0][1:], results[1]
        
def outputPDB(residuesData, atoms_matrix, pdb_path):
    
    atom_id = 1
    counter = 0
    f = open(pdb_path, 'w')
    for residue in residuesData:
        for idx, name1 in enumerate(["N", "CA", "C", "O", "CB"]):
            if residue.resname == "G" and name1 == "CB": 
                counter += 1
                continue
            atom_id2 = atom_id + idx
            string = 'ATOM  '
            id_len = len(list(str(atom_id2)))
            string = string + " "*(5-id_len) + str(atom_id2)
            string = string + " "*2
            name1_len = len(list(name1))
            string = string + name1 + " "*(3-name1_len)
            resname = Residues.triResname(residue.resname)
            resname_len = len(list(resname))
            string = string + " "*(4-resname_len) + resname
            string = string + " "*2
            resid = str(residue.resid)
            resid_len = len(list(resid))
            string = string + " "*(4-resid_len) + str(resid)
            string = string + " "*4
            x = format(atoms_matrix[counter][0],".3f")
            x_len = len(list(x))
            string = string + " "*(8-x_len) + x
            y = format(atoms_matrix[counter][1],".3f")
            y_len = len(list(y))
            string = string + " "*(8-y_len) + y
            z = format(atoms_matrix[counter][2],".3f")        
            z_len = len(list(z))
            string = string + " "*(8-z_len) + z  
            
            f.write(string)
            f.write("\n")
            
            counter += 1
        
        atom_id += residue.num_atoms
        
    assert len(atoms_matrix) == counter
    f.close()

