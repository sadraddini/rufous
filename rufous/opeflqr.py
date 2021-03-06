w#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 19:20:10 2020

@author: sadra
"""

import numpy as np
import pickle

my_file=open("data/%s/system.pkl"%f,"rb")
(M,N,e_bar,e,E)=pickle.load(my_file)
T=len(N)
o,m=N[0].shape
N_data=e[0].shape[1]

import time


def LME(list_of_AB,C):
    assert list_of_AB[0][0].shape==(1,1)
    start_time=time.time()
    print("size of AB list",len(list_of_AB),"\t B size",list_of_AB[0][1].shape)
    N=list_of_AB[0][1].shape[0]
    A=np.zeros((N,N))
    for (e,f) in list_of_AB:
        A=A+f.T*e[0,0]
    print("sum=",time.time()-start_time)
    start_time=time.time()
    X=np.linalg.solve(A, C.T)
    print("solve=",time.time()-start_time)
    return X

def LME_greedy(list_of_AB,C):
    A,B=list_of_AB[0]
    X=np.random.random((A.shape[1],B.shape[0]))*10
    for t in range(100):
        h=np.random.random()
        D=np.random.random((A.shape[1],B.shape[0]))*h
        print("iteration",t)
        JX=np.linalg.norm(sum([np.linalg.multi_dot([e,X,f]) for (e,f) in list_of_AB])-C)
        JXD=np.linalg.norm(sum([np.linalg.multi_dot([e,X+D,f]) for (e,f) in list_of_AB])-C)
        print("JX",JX/np.sqrt(C.shape[0]*C.shape[1]),"JXD",JXD/np.sqrt(C.shape[0]*C.shape[1]))
        if JX>=JXD:
            X=X+D
        else:
            X=X-D
    return X
    
    
        

def opeflqr(M,N,y_ref,u_ref,Q,R,zeta,sigma):
    # I and Gamma
    I,Gamma={},{}
    for t in range(T+1):
        I[t]=np.zeros((o,o*(T+1)))
        I[t][:,o*t:o*(t+1)]=np.eye(o)
    for t in range(T):
        Gamma[t,"u"]=np.zeros((T+o*int(T*(T+1)/2),1))
        Gamma[t,"u"][t:(t+1),:]=np.eye(1)
        Gamma[t,"theta"]=np.zeros((T+o*int(T*(T+1)/2),o*(T+1)))
        start=T+o*int(t*(t+1)/2)
        end=start+o*(t+1)
        Gamma[t,"theta"][start:end,:o*(t+1)]=np.eye(o*(t+1))
    
    start_time=time.time()
    alpha,phi={},{}
    # Alpha and phi_bar
    phi["bar",0]=I[0]
    alpha[0,0]=np.zeros((o,m))
    for t in range(T):
        for k in range(t+1):
            alpha[k,t+1]=alpha[0,0]+sum([np.dot(M[t][:,o*tau:o*(tau+1)],alpha[k,tau]) for tau in range(k+1,t+1) ])+N[t][:,m*k:m*(k+1)]
        phi["bar",t+1]=sum([np.dot(M[t][:,o*tau:o*(tau+1)],phi["bar",tau]) for tau in range(0,t+1) ]) + I[t+1]    
    print("alpha and phi_bar=",time.time()-start_time)
    
    start_time=time.time()
    L=T+o*int(T*(T+1)/2)
    GuGu={}
    for tau in range(T):
        for k in range(T):
            GuGu[tau,k]=np.zeros((L,L))
            GuGu[tau,k][tau,k]=1
            
    GtZGu={}
    for tau in range(T):
        for k in range(T):
            GtZGu[tau,k]=np.zeros((L,L))
            start=T+o*int(tau*(tau+1)/2)
            end=start+(tau+1)*o
            GtZGu[tau,k][start:end,k:k+1]=zeta[0:(tau+1)*o,:]

    GtSGt={}
    for tau in range(T):
        for k in range(T):
            GtSGt[tau,k]=np.zeros((L,L))
            start_row=T+o*int(tau*(tau+1)/2)
            end_row=start_row+(tau+1)*o
            start_column=T+o*int(k*(k+1)/2)
            end_column=start_column+(k+1)*o
            GtSGt[tau,k][start_row:end_row,start_column:end_column]=sigma[0:(tau+1)*o,0:(k+1)*o]
    
    print("for loop terms=",time.time()-start_time)
    start_time=time.time()
            
    terms_x=[(np.linalg.multi_dot([alpha[k,t].T,Q[t],alpha[tau,t]]), \
                GuGu[tau,k] + GtZGu[tau,k] + GtZGu[k,tau].T + GtSGt[tau,k] ) \
                for t in range(T+1) if np.linalg.norm(Q[t])!=0 for k in range(t) for tau in range(t)]
    terms_u=[(R[t], GuGu[t,t]+GtZGu[t,t]+GtZGu[t,t].T+GtSGt[t,t] ) 
                for t in range(T)]
    
    terms=terms_x+terms_u

    print("collecting term lists=",time.time()-start_time)
    print(len(terms))
    start_time=time.time()
    # Class II term 1
    G_x=-sum([np.linalg.multi_dot([alpha[k,t].T,Q[t],\
            np.linalg.multi_dot([phi["bar",t],zeta,Gamma[k,"u"].T])+
            np.linalg.multi_dot([-y_ref[t],Gamma[k,"u"].T])+
            np.linalg.multi_dot([phi["bar",t],sigma,Gamma[k,"theta"].T])+
            np.linalg.multi_dot([-y_ref[t],zeta.T,Gamma[k,"theta"].T])
                                ]) \
         for t in range(T+1) for k in range(t)])
        
    G_u=-sum([np.linalg.multi_dot([R[t],-u_ref[t],Gamma[t,"u"].T])+\
         np.linalg.multi_dot([R[t],-u_ref[t],zeta.T,Gamma[t,"theta"].T])
         for t in range(T)])
    print("G=",time.time()-start_time)
    # The controls now!
    G=G_x+G_u
    print("the devil is here")
    X=LME(terms,G)
    start_time=time.time()
    pi=X.reshape(T+o*int(T*(T+1)/2),m).T
    ubar,theta={},{}
    k=0
    for t in range(T):
        ubar[t]=pi[:,k:k+1]
        k+=1
    for t in range(T):
        x=pi[:,k:k+o*(t+1)]
        theta[t]=x.reshape(m,o*(t+1))
        k+=o*(t+1)
    print("extraction=",time.time()-start_time)
    return pi,ubar,theta


my_file=open("data/%s/system.pkl"%f,"rb")
(M,N,e_bar,e,E)=pickle.load(my_file)
T=len(N)
o,m=N[0].shape
N_data=e[0].shape[1]

zeta=np.mean(E[T],1).reshape(o*(T+1),1)
sigma=np.dot(E[T],E[T].T)

y_target=np.ones((o,1))*15
R,Q,y_ref,u_ref={},{},{},{}
for t in range(T):
    y_ref[t]=np.zeros((o,1))
    u_ref[t]=np.zeros((m,1))
    R[t]=np.eye(m)*1
    Q[t]=np.eye(o)*1
Q[T]=np.eye(o)*100
y_ref[T]=y_target

start_time=time.time()
pi_ope,ubar,theta=opeflqr(M,N,y_ref,u_ref,Q,R,zeta,sigma)
print(time.time()-start_time)