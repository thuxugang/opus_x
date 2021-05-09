# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 16:30:45 2017

@author: lee
"""

import sys,os,re
from subprocess import Popen, PIPE, STDOUT
import numpy as np
from io import BytesIO

def blockshaped(arr,dim=21):
    p=arr.shape[0]//dim
    re=np.zeros([dim*dim,p,p])
    for i in range(p):
        for j in range(p):
            re[:,i,j]=arr[i*dim:i*dim+dim,j*dim:j*dim+dim].flatten()
    return re

def computeccm(alignment_file, ccmpred_file, filename, gpu):
    # exefile=os.path.join(os.path.dirname(__file__),'bin/ccmpred')
    cmd = './ccmpred -d ' + gpu + ' -r ' + filename+'.raw ' + alignment_file + ' ' + \
        ccmpred_file
    print (cmd, ccmpred_file)
    os.system(cmd)

def readfromsavefile(filename):
    lines=open(filename+'.raw').readlines()
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
    os.remove(filename+'.raw')
    return blockshaped(precision)

def computeplm(alignment_file, ccmpred_file, plm_file, filename, gpu):

    computeccm(alignment_file, ccmpred_file, filename, gpu)
    plm = readfromsavefile(filename)
    
    np.save(plm_file, plm.astype(np.float32))

if __name__ == '__main__':
    
    gpu = str(0)
    
    ccmpred_path = "/work/home/xugang/projects/opus_toolkit/data/2d_features/ccmpred2"
    plm_path = "/work/home/xugang/projects/opus_toolkit/data/2d_features/plm2"
    
    alignment_path = "/work/home/xugang/projects/opus_toolkit/data/DeepMSA/results"
    
    list_all = []
    list_path = r'./list_partaa'
    with open(list_path,'r') as r:
        list_all.extend([i.strip() for i in r.readlines()])
        
    print (len(list_all))
    
    for idx, filename in enumerate(list_all):
        print (idx, filename)
        alignment_file = os.path.join(alignment_path, filename + ".aln")
        ccmpred_file = os.path.join(ccmpred_path, filename+".ccmpred")
        plm_file = os.path.join(plm_path, filename+".plm")
        
        computeplm(alignment_file, ccmpred_file, plm_file, filename, gpu)
    
    



         
            
    
    
        