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
import matplotlib.pyplot as plt


clock=time.clock()
f=str(int(clock*10**1))
try:
    os.makedirs("data/%s"%f)
except:
    pass


def generate_data(n,o,m,N,T,flag="training"):
    np.random.seed(0)
    # Generate system
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
        v[i][0]=0.03*(np.random.random((o,1))-0.5)
        x[i][0]=1*(np.random.random((n,1))-0.5)
        y[i][0]=np.dot(C,x[i][0])+v[i][0]
        for t in range(T):
            w[i][t]=0.03*(np.random.random((n,1))-0.5)+1*0
            v[i][t+1]=0.05*(np.random.random((o,1))-0.5)
            u[i][t]=np.random.random((m,1))-0.5
            x[i][t+1]=np.dot(A,x[i][t])+np.dot(B,u[i][t])+w[i][t]
            y[i][t+1]=np.dot(C,x[i][t+1])+v[i][t+1]
    return y,u
        
def model_least_squares(y,u,w_M=1,w_N=1):
    # Define the variables
    M,N,e_bar,e,E={},{},{},{},{}
    T,N_D=len(u[0]),len(u)
    print("Data Size is %d. Time step is %d"%(N_D,T))
    o,m=y[0][0].shape[0],u[0][0].shape[0]  
    y_stack,Y_stack,u_stack,U_stack={},{},{},{}
    for t in range(T+1):
        y_stack[t]=np.zeros((o,N_D))
        for i in range(N_D):
            y_stack[t][:,i:i+1]=y[i][t]
        Y_stack[t]=np.vstack([y_stack[tau] for tau in range(t+1)])
    for t in range(T):
        u_stack[t]=np.zeros((m,N_D))
        for i in range(N_D):
            u_stack[t][:,i:i+1]=u[i][t]
        U_stack[t]=np.vstack([u_stack[tau] for tau in range(t+1)])
    one=np.ones((N_D,1))
    for t in range(T):
        # First row
        X11=np.dot(Y_stack[t],Y_stack[t].T)+w_M*N_D/o/(t+1)*np.eye(Y_stack[t].shape[0])
        X12=np.dot(Y_stack[t],U_stack[t].T) 
        X13=np.dot(Y_stack[t],one) 
        # Second row
        X21=np.dot(U_stack[t],Y_stack[t].T) 
        X22=np.dot(U_stack[t],U_stack[t].T)+w_N*N_D/o/(t+1)*np.eye(U_stack[t].shape[0])
        X23=np.dot(U_stack[t],one)
        # Third row
        X31=np.dot(one.T,Y_stack[t].T) 
        X32=np.dot(one.T,U_stack[t].T)
        X33=np.array([N_D]).reshape(1,1)
        # Now put them together
        X=np.vstack((  np.hstack((X11,X12,X13)) , \
                     np.hstack((X21,X22,X23))  , \
                     np.hstack((X31,X32,X33))
                     ))
        # Now get the others:
        YU=np.vstack(( Y_stack[t], U_stack[t] , one.T ))
        # Get the computation:
        Q=np.dot(y_stack[t+1],np.dot(YU.T,np.linalg.inv(X)))
        assert Q.shape[1]==(o+m)*(t+1)+1
        assert Q.shape[0]==o
        M[t]=Q[:,:o*(t+1)]
        N[t]=Q[:,o*(t+1):(o+m)*(t+1)]
        e_bar[t]=Q[:,(o+m)*(t+1):(o+m)*(t+1)+1]
    # Look at the error terms
    for t in range(T):
        e[t]=y_stack[t+1]-np.dot(M[t],Y_stack[t])-np.dot(N[t],U_stack[t])
        E[t+1]=np.vstack((y_stack[0],np.vstack([e[tau] for tau in range(t+1)]) ))
    E[0]=y_stack[0]
    for t in range(T+1):
        print(t,E[t].shape)
    f_all=open("data/%s/system.pkl"%f,"wb")
    pickle.dump((M,N,e_bar,e,E),f_all)
    f_all.close()
    return M,N,e_bar,e,E




def learn_model(y,u,w_reg=1):  
    """
    Soon to be deprecated 
    """    
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
    E[0]=y0
    # WARNING: THIS SEEMS to BE WRONG
    for t in range(T):
        E[t+1]=np.vstack((y0,np.vstack([e[tau]+e_s[tau] for tau in range(t+1)]) ))
    for t in range(T+1):
        print(t,E[t].shape)
    f_all=open("data/%s/system2.pkl"%f,"wb")
    pickle.dump((M_s,N_s,e,e_s,E),f_all)
    f_all.close()
    return M_s,N_s,e_s,e

def test_prediction(y,u,M,N,e_bar,color='red'):
    T,N_data=len(u[0]),len(u)
    total_error={}
    for t in range(T):
        Y=np.vstack([np.hstack([y[i][tau] for i in range(N_data)]) for tau in range(t+1)])
        U=np.vstack([np.hstack([u[i][tau] for i in range(N_data)]) for tau in range(t+1)]) 
        y_next=np.dot(M[t],Y)+np.dot(N[t],U)+e_bar[t]
        total_error[t]=1/N_data**0.5*np.linalg.norm(y_next-np.hstack([y[i][t+1] for i in range(N_data)]),'fro')
        print(t,"-error is",total_error[t])
    plt.plot([t for t in range(T)],[total_error[t] for t in range(T)],color=color)
    
n,o,m,T=5,1,1,10
N_data=500
N_train=400
N_test=N_data-N_train
y,u=generate_data(n,o,m,N_data,T)
y_train={i:y[i] for i in range(N_train)}
y_test={i-N_train:y[i] for i in range(N_train,N_data)}
u_train={i:u[i] for i in range(N_train)}
u_test={i-N_train:u[i] for i in range(N_train,N_data)}
start_time=time.time()
for my_reg in [-3,-2,-1,0,1,2,3]:
    M,N,e_bar,e=learn_model(y_train,u_train,w_reg=10**my_reg)
    print(time.time()-start_time)
    start_time=time.time()
#    M2,N2,e_bar2,e,E=model_least_squares(y_train,u_train,w_M=10**-1,w_N=10**-5)
    print(time.time()-start_time)
    # Training Error
    test_prediction(y_train,u_train,M,N,e_bar,color=(1,0.5-my_reg/6,0.5-my_reg/6))
    #test_prediction(y_train,u_train,M2,N2,e_bar2,'orange')
    # Test Error
    test_prediction(y_test,u_test,M,N,e_bar,color=(0.5-my_reg/6,0.5-my_reg/6,1))
    #test_prediction(y_test,u_test,M2,N2,e_bar2,'cyan')

