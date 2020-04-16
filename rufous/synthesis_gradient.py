#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:01:22 2020

@author: sadra
"""

import numpy as np

T=S.T
o=S.o
m=S.m
M={}
N={}

for t in range(T+1):
    for i in range(S.T_windows[t]):
        M[t,t-i] = S.M[t] [: , o*(S.T_windows[t]-i-1): o*(S.T_windows[t]-i) ]
        N[t,t-i] = S.M[t] [: , m*(S.T_windows[t]-i-1): m*(S.T_windows[t]-i) ]
        
# *** phi_bar
phi_bar={}
for t in range(T+1):
    phi_bar[t,t]=np.eye(o)
    
# recursive eqns.
for t in range(T+1):
    for j in range(t+1):
        phi_bar[t+1,j]=np.zeros((o,o))
        for i in range(t-S.T_windows[t]+1,t+1):
            if i>=j:
                print(t,j,i)
                phi_bar[t+1,j]=+np.dot( M[t,i], phi_bar[i,j] )

print("computing alphas")
# *** Alpha and Beta
alpha,beta={},{}
for t in range(T+1):
    beta[t,t]=np.zeros((o,m))
    for j in range(t+1):
        alpha[t,t,j]=np.zeros((o,m))

# recursive eqns.
for t in range(T+1):
    for j in range(t+1):
        if j>t-S.T_windows[t]:
            print(t,j)
            beta[t+1,j] = N [ t , j ]
            for i in range(j,t+1):
                beta[t+1,j] += np.dot( M[t,i] , beta[i, j]  ) 
        else:
            beta[t+1,j] = np.zeros((o,m))

# recursive eqns for alpha.
for t in range(T+1):
    for j in range(t+1):
        for k in range(j,t+1):
            if k>t-S.T_windows[t]:
                print(t,k,j)
                alpha[t+1,k,j] = N [ t , k ]
                for i in range(j,t+1):
                    alpha[t+1,k,j] += np.dot( M[t,i] , alpha[i, k, j]  ) 
            else:
                alpha[t+1,k,j] = np.zeros((o,m))               