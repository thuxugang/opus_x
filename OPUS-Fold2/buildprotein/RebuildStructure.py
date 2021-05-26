# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:15:25 2016

@author: XuGang
"""

from buildprotein import Geometry
from buildprotein import PeptideBuilder

def getGeosData(residuesData):
    
    geosData = []
    for residue in residuesData:
        geo = Geometry.geometry(residue.resname)
        geosData.append(geo)
        
    return geosData   

def rebuild_main_chain(torsions, geosData, residuesData):
    
    count = 0
    atoms_matrix = []
    assert len(residuesData) == len(geosData)
    
    length = len(residuesData)
    for idx in range(length):
        
        if idx == 0:
            atoms_matrix.extend(PeptideBuilder.get_mainchain(None, atoms_matrix, 
                                                             residuesData[idx], geosData[idx], None))
        else:
            # phi, psi, omega
            torsion = [torsions[count], torsions[count-2], torsions[count+2]]
            atoms_matrix.extend(PeptideBuilder.get_mainchain(torsion, atoms_matrix, 
                                                             residuesData[idx], geosData[idx], geosData[idx-1]))
            
        count += 3
    
    assert count == len(torsions)
    
    return atoms_matrix