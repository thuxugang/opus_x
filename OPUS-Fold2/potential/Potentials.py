# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:46:08 2016

@author: XuGang
"""

from potential import TrrPotential, Rama

def get_potentials(Trr_matrix, atoms_matrix, init_torsions, rama_cons):
    
    potentials = 0
    
    dist_potential, omega_potential, theta_potential, phi_potential = \
        TrrPotential.cal_TrrPotential(Trr_matrix, atoms_matrix)
    potentials += (20*dist_potential + 8*omega_potential + 8*theta_potential + 8*phi_potential)
    
    rama_potential, o_potential = Rama.cal_RamaPotential(init_torsions, rama_cons)
    potentials += (0.1*rama_potential + 0.05*o_potential)
    
    print (20*dist_potential.numpy(), 8*omega_potential.numpy(), 8*theta_potential.numpy(), 8*phi_potential.numpy(),
           0.1*rama_potential.numpy(), 0.05*o_potential.numpy())    

    return potentials