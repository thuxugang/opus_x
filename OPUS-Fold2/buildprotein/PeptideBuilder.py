import tensorflow as tf
import numpy as np
from buildprotein import Geometry

def get_norm(v):
    return tf.norm(v, axis=-1)
        
def get_angle(p1, p2, p3):
    
    v1 = p1 - p2
    v2 = p3 - p2

    v1_norm = get_norm(v1)
    v2_norm = get_norm(v2)
    c = tf.reduce_sum(v1*v2, -1)/(v1_norm * v2_norm)

    c = tf.clip_by_value(c, -0.999999, 0.999999)
    
    return tf.math.acos(c)/np.pi*180

def get_angle2(v1, v2):
    
    v1_norm = get_norm(v1)
    v2_norm = get_norm(v2)
    c = tf.reduce_sum(v1*v2, -1)/(v1_norm * v2_norm)

    c = tf.clip_by_value(c, -0.999999, 0.999999)
    
    return tf.math.acos(c)/np.pi*180

def get_dihedral(p1, p2, p3, p4):
    
    c1 = p1 - p2
    c2 = p2 - p3
    c3 = p3 - p4
    
    v1 = tf.linalg.cross(c2, c1)
    v2 = tf.linalg.cross(c3, c2)
    v3 = tf.linalg.cross(v2, v1)
    
    return tf.sign(tf.reduce_sum(v3*c2,-1))*get_angle2(v1,v2)
        
def calculateCoordinates(c1, c2, c3, L, ang, di):

    d2 = tf.stack([L*tf.math.cos(ang/180*np.pi),
                   L*tf.math.cos(di/180*np.pi)*tf.math.sin(ang/180*np.pi),
                   L*tf.math.sin(di/180*np.pi)*tf.math.sin(ang/180*np.pi)])
    ab = c2 - c1
    bc = c3 - c2
    bc = bc/get_norm(bc)
    n = tf.linalg.cross(ab, bc)
    n = n/get_norm(n)
    ab = tf.linalg.cross(n, bc)
    
    mtr = tf.stack([-bc, ab, n])
    mtr = tf.transpose(mtr)

    bc = tf.experimental.numpy.dot(mtr, d2)
    cc = c3 + bc
    
    return tf.cast(cc, tf.float32)

geo_ala = Geometry.geometry('A')
def get_mainchain(torsions, atoms_matrix, residue, geo, geo_last):
    
    resid = residue.resid
    if resid == 1:
        N = np.array([geo.CA_N_length*np.cos(geo.N_CA_C_angle*(np.pi/180.0)),
                      geo.CA_N_length*np.sin(geo.N_CA_C_angle*(np.pi/180.0)),
                      0], dtype=np.float32)
        CA = np.array([0,0,0], dtype=np.float32)
        C = np.array([geo.CA_C_length,0,0], dtype=np.float32)
    else:
        _N = atoms_matrix[-5]
        _CA = atoms_matrix[-4]
        _C = atoms_matrix[-3]
        
        N = calculateCoordinates(_N, _CA, _C, geo.peptide_bond, geo.CA_C_N_angle, torsions[1])
        CA = calculateCoordinates(_CA, _C, N, geo.CA_N_length, geo.C_N_CA_angle, torsions[2])
        C = calculateCoordinates(_C, N, CA, geo.CA_C_length, geo.N_CA_C_angle, torsions[0])
        
    O = calculateCoordinates(N, CA, C, geo.C_O_length, geo.CA_C_O_angle, geo.N_CA_C_O_diangle)
    
    if residue.resname == 'G': geo = geo_ala
    CB = calculateCoordinates(C, N, CA, geo.CA_CB_length, geo.C_CA_CB_angle, geo.N_C_CA_CB_diangle)
    
    return [N, CA, C, O, CB]        

