#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 13:05:44 2019

@author: sadra
"""

import numpy as np
import pickle
import time
import os

clock=time.clock()
f=str(int(clock*10**1))
try:
    os.makedirs("data/%s"%f)
except:
    pass



def generate_data(n,o,m,N,T,flag="training"):
    # Generate system
    np.random.seed(500)
    A=(np.random.random((n,n))-0.5)*1
    print(np.linalg.eigvals(A))
    B=np.random.random((n,m))
    C=np.random.random((o,n))
    fS=open("data/%s/S.pkl"%f,"wb")
    pickle.dump((A,B,C),fS)
    fS.close()
    # Generate data
    x,u,y={},{},{}
    v,w={},{}
    for i in range(N):
        x[i],u[i],y[i],v[i],w[i]={},{},{},{},{}
        v[i][0]=0.05*(np.random.random((o,1))-0.5)
        x[i][0]=np.random.random((n,1))-0.5
        y[i][0]=np.dot(C,x[i][0])+v[i][0]
        for t in range(T):
            w[i][t]=0.05*(np.random.random((n,1))-0.5)+1*0
            v[i][t+1]=0.08*(np.random.random((o,1))-0.5)
            u[i][t]=np.random.random((m,1))-0.5
            x[i][t+1]=np.dot(A,x[i][t])+np.dot(B,u[i][t])+w[i][t]
            y[i][t+1]=np.dot(C,x[i][t+1])+v[i][t+1]
    return y,u
        

def learn_model(y,u,w_reg=1):      
    # analyze the data
    M_s,N_s,e,e_s,E={},{},{},{},{}
    T,N=len(u[0]),len(u)
    y0=np.hstack([y[i][0] for i in range(N)])
    for t in range(T):
        y_next=np.hstack([y[i][t+1] for i in range(N)])
        Y_current=np.vstack([np.hstack([y[i][tau] for i in range(N)]) for tau in range(t+1)])
        U_current=np.vstack([np.hstack([u[i][tau] for i in range(N)]) for tau in range(t+1)])
        YU=np.vstack((Y_current,U_current))
        YU=np.vstack((Y_current,U_current,w_reg*np.ones((1,N))))
        Q=np.dot(np.dot(y_next,(YU.T)),np.linalg.inv(np.dot(YU,YU.T)+w_reg*np.eye(YU.shape[0])))
        M_s[t]=Q[:,:Y_current.shape[0]]
        N_s[t]=Q[:,Y_current.shape[0]:Q.shape[1]-1]
        e_s[t]=w_reg*Q[:,Q.shape[1]-1:Q.shape[1]]
        # Look at the error terms
        e[t]=y_next-np.dot(Q,YU)
    E[0]=e[0]
    for t in range(T):
        E[t+1]=np.vstack((y0,np.vstack([e[tau]+e_s[tau] for tau in range(t+1)]) ))
    for t in range(T+1):
        print(t,E[t].shape)
    f_all=open("data/%s/system.pkl"%f,"wb")
    pickle.dump((M_s,N_s,e,e_s,E),f_all)
    f_all.close()
    return M_s,N_s,e_s,e

def test_prediction(y,u,M,N,e_bar):
    import matplotlib.pyplot as plt
    T,N_data=len(u[0]),len(u)
    total_error={}
    for t in range(T):
        Y=np.vstack([np.hstack([y[i][tau] for i in range(N_data)]) for tau in range(t+1)])
        U=np.vstack([np.hstack([u[i][tau] for i in range(N_data)]) for tau in range(t+1)]) 
        y_next=np.dot(M[t],Y)+np.dot(N[t],U)+e_bar[t]
        total_error[t]=1/N_data**0.5*np.linalg.norm(y_next-np.hstack([y[i][t+1] for i in range(N_data)]),'fro')
        print(t,"-error is",total_error[t])
    plt.plot([t for t in range(T)],[total_error[t] for t in range(T)])
    
n,o,m,T=20,10,2,10
y,u=generate_data(n,o,m,1000,T)
M,N,e_bar,e=learn_model(y,u,w_reg=10**-2)
# Training Error
test_prediction(y,u,M,N,e_bar)
# Test Error
y_test,u_test=generate_data(n,o,m,100,T)
test_prediction(y_test,u_test,M,N,e_bar)

