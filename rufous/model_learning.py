#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:37:52 2020

@author: sadra
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt

class ARX_model:
    def __init__(self,M,N,e_bar,zeta):
        self.M=M
        self.N=N
        self.e_bar=e_bar
        self.zeta=zeta
        self._build()
        
    def _build(self):
        self.T=max(self.M.keys())
        assert self.T==max(self.N.keys())
        self.o,self.m=self.M[0].shape[0],self.N[0].shape[0]
        self.T_windows={}
        for t in range(self.T+1):
            # T_windows is the number of outputs in the past (including current) that is used for prediction 
            self.T_windows[t]=int(self.M[t].shape[1]/self.o)


def model_least_squares_old(y,u,w_M=1,w_N=1):
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
        print("model least squares for ",t)
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
    try:
        f_all=open("data/%s/system.pkl"%f,"wb")
    except:
        f_all=open("learned_system.pkl","wb")
    pickle.dump((M,N,e_bar,e,E),f_all)
    f_all.close()
    return M,N,e_bar,e,E

def model_least_squares(y,u,w_M=1,w_N=1,T_window={}):
    # Define the variables
    M,N,e_bar,e,E={},{},{},{},{}
    T,N_D=len(u[0]),len(u)
    print("Data Size is %d. Time step is %d"%(N_D,T))
    o,m=y[0][0].shape[0],u[0][0].shape[0]  
    y_stack,Y_stack,u_stack,U_stack={},{},{},{}
    # if T_window is not specificed, then it is max everywhere
    if T_window=={}:
        T_window={t:T for t in range(T+1)}
    for t in range(T+1):
        y_stack[t]=np.zeros((o,N_D))
        for i in range(N_D):
            y_stack[t][:,i:i+1]=y[i][t]
        Y_stack[t]=np.vstack([y_stack[tau] for tau in range(max(t-T_window[t],0),t+1)])
    for t in range(T):
        u_stack[t]=np.zeros((m,N_D))
        for i in range(N_D):
            u_stack[t][:,i:i+1]=u[i][t]
        U_stack[t]=np.vstack([u_stack[tau] for tau in range(max(t-T_window[t],0),t+1)])
    one=np.ones((N_D,1))
    for t in range(T):
        print("model least squares for ",t)
        # First row
        X11=np.dot(Y_stack[t],Y_stack[t].T)+w_M*N_D/Y_stack[t].shape[0]*np.eye(Y_stack[t].shape[0])
        X12=np.dot(Y_stack[t],U_stack[t].T) 
        X13=np.dot(Y_stack[t],one) 
        # Second row
        X21=np.dot(U_stack[t],Y_stack[t].T) 
        X22=np.dot(U_stack[t],U_stack[t].T)+w_N*N_D/U_stack[t].shape[0]*np.eye(U_stack[t].shape[0])
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
        assert Q.shape[1]==Y_stack[t].shape[0]+U_stack[t].shape[0]+1
        assert Q.shape[0]==o
        M[t]=Q[:,:Y_stack[t].shape[0]]
        N[t]=Q[:,Y_stack[t].shape[0]:Y_stack[t].shape[0]+U_stack[t].shape[0]]
        e_bar[t]=Q[:,Y_stack[t].shape[0]+U_stack[t].shape[0]:Y_stack[t].shape[0]+U_stack[t].shape[0]+1]
    zeta={}
    zeta[0]=y_stack[0]
    # Look at the error terms
    for t in range(T):
        e[t]=y_stack[t+1]-np.dot(M[t],Y_stack[t])-np.dot(N[t],U_stack[t])
        zeta[t+1]=e[t]
        E[t+1]=np.vstack((y_stack[0],np.vstack([e[tau] for tau in range(t+1)]) ))
    E[0]=y_stack[0]
    return ARX_model(M,N,e_bar,zeta)




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
    try:
        f_all=open("data/%s/system.pkl"%f,"wb")
    except:
        f_all=open("learned_system.pkl","wb")
    pickle.dump((M_s,N_s,e,e_s,E),f_all)
    f_all.close()
    return M_s,N_s,e_s,e

def test_prediction_old(y,u,M,N,e_bar,color='red'):
    T,N_data=len(u[0]),len(u)
    total_error={}
    for t in range(T):
        Y=np.vstack([np.hstack([y[i][tau] for i in range(N_data)]) for tau in range(t+1)])
        U=np.vstack([np.hstack([u[i][tau] for i in range(N_data)]) for tau in range(t+1)]) 
        y_next=np.dot(M[t],Y)+np.dot(N[t],U)+e_bar[t]
        total_error[t]=1/N_data**0.5*np.linalg.norm(y_next-np.hstack([y[i][t+1] for i in range(N_data)]),'fro')
        print(t,"-error is",total_error[t])
    plt.plot([t for t in range(T)],[total_error[t] for t in range(T)],color=color)
    
def test_prediction(y,u,S,color='red'):
    """
    Inputs:
        * y: test data for outputs  y[n][t]=numpy.array((o,1))
        * u: test data for inputs   u[n][t]=numpy.array((m,1))
    """
    T,N_data=len(u[0]),len(u)
    total_error={}
    for t in range(T):
        Y=np.vstack([np.hstack([y[i][tau] for i in range(N_data)]) for tau in range(t-S.T_windows[t]+1,t+1)])
        U=np.vstack([np.hstack([u[i][tau] for i in range(N_data)]) for tau in range(t-S.T_windows[t]+1,t+1)]) 
        y_next=np.dot(S.M[t],Y)+np.dot(S.N[t],U)+S.e_bar[t]
        total_error[t]=1/N_data**0.5*np.linalg.norm(y_next-np.hstack([y[i][t+1] for i in range(N_data)]),'fro')
        print(t,"-error is",total_error[t])
    plt.plot([t for t in range(T)],[total_error[t] for t in range(T)],color=color)