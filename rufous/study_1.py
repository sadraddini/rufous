#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 01:48:07 2019

@author: sadra
"""

import numpy as np

np.random.seed(0)
# Generate system
n=5
o=1
m=1
A=np.random.random((n,n))-0.2
B=np.random.random((n,m))
C=np.random.random((m,n))
# Generate data
N=1000
T=20
x,u,y={},{},{}
for i in range(N):
    x[i,0]=np.random.random((n,1))-0.5
    y[i,0]=np.dot(C,x[i,0])
    for t in range(T):
        u[i,t]=np.random.random((m,1))-0.5
        x[i,t+1]=np.dot(A,x[i,t])+np.dot(B,u[i,t])
        y[i,t+1]=np.dot(C,x[i,t+1])
# Now analyze the data
M_s,N_s,e={},{},{}
for t in range(T):
    y_next=np.hstack([y[i,t+1] for i in range(N)])
    Y_current=np.vstack([np.hstack([y[i,tau] for i in range(N)]) for tau in range(t+1)])
    U_current=np.vstack([np.hstack([u[i,tau] for i in range(N)]) for tau in range(t+1)])
    YU=np.vstack((Y_current,U_current))
    Q=np.dot(y_next,np.linalg.pinv(YU))
    M_s[t]=Q[:,:Y_current.shape[0]]
    N_s[t]=Q[:,Y_current.shape[0]:]
    # Look at the error terms
    e[t]=np.hstack((y_next-np.dot(Q,YU)))
    
import matplotlib.pyplot as plt
plt.plot([np.linalg.norm(e[t],ord=2) for t in range(T)])