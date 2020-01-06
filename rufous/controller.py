#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 21:11:37 2019

@author: sadra
"""

import numpy as np
import pickle

#clock=214
#f=str(clock)    
f_all=open("data/%s/system.pkl"%f,"rb")
(M,N,e,e_bar,E)=pickle.load(f_all)

T=len(N)
o,m=N[0].shape
N_data=e[0].shape[1]

   
def unit(n,i):
    x=np.zeros((n,1))
    x[i,0]=1
    return x



def synthesis_LS(y_target,system_file):
    pass    


print("Now doing Synthesis")    

# The model for L and J and I_0
L,J={},{}
I_0=np.kron(unit(T+1,0).T,np.eye(o))
for t in range(T):
    L[t]=np.kron(unit(T+1,t+1).T,np.eye(o))
for t in range(T+1):
    X=np.zeros(( o*(t+1),o*(T+1) ))
    X[:,:o*(t+1)]=np.eye( o*(t+1) )
    J[t]=X
    
# Now hard part: alpha and phi
phi,alpha={},{}
phi["bar",0]=I_0
alpha[0,0]=np.zeros((o,m))
for t in range(T):
    for k in range(t+1):
        alpha[k,t+1]=alpha[0,0]+sum([np.dot(M[t][:,o*tau:o*(tau+1)],alpha[k,tau]) for tau in range(k+1,t+1) ])+N[t][:,m*k:m*(k+1)]
    phi["bar",t+1]=sum([np.dot(M[t][:,o*tau:o*(tau+1)],phi["bar",tau]) for tau in range(0,t+1) ]) + L[t]

K={}
k=0
for tau in range(T):
    for l in range(m):
        k+=o*tau+o*(l>0)
        K[tau,l]=k
length=int(T*(T+1)/2*m*o)
SS=np.zeros((o,length+m*T,o*(T+1)))
# Fill
for i in range(o*(T+1)):
    for tau in range(T):
        for l in range(m):
            if i<o*(tau+1):
                SS[:,m*T+K[tau,l]+i,i]=alpha[tau,T][:,l]


Su=np.hstack([alpha[t,T] for t in range(T)])
I_u=np.hstack((np.eye(m*T), np.zeros((m*T,length)) ))
Su_final=np.dot(Su,I_u)


# Writing things as min||R + QX||^2_2+w||X||^2_2
y_target=np.array([45,20,20,10,10,10,10,10,10,-20]).reshape(10,1)
L_reg=10**-2
R,Q={},{}
for j in range(N_data):
    R[j]=np.dot(phi["bar",T],E[T][:,j].reshape(E[T].shape[0],1))-y_target
    Q[j]=Su_final/L_reg+np.dot(SS,E[T][:,j].reshape(E[T].shape[0],1)).reshape(SS.shape[0],SS.shape[1])

R["all"]=np.vstack([R[j] for j in range(N_data)])
Q["all"]=np.vstack([Q[j] for j in range(N_data)])
pi=-np.dot(np.linalg.inv(np.dot(Q["all"].T,Q["all"])+L_reg*np.eye(Q["all"].shape[1])),\
          np.dot(Q["all"].T,R["all"]))
# Error analysis
print("average final error=",np.linalg.norm(R["all"]+np.dot(Q["all"],pi),'fro')/N_data**0.5)

# Extract policy
def extract_policy(pi,T,m,o,L_reg):
    assert len(pi)==m*T*(1+(T+1)/2*o)
    ubar,theta={},{}
    k=0
    for t in range(T):
#        print("k",k)
        ubar[t]=pi[k:k+m,:]/L_reg
        k+=m
    for t in range(T):
        x=pi[k:k+m*o*(t+1),:]
        theta[t]=x.reshape(m,o*(t+1))
        k+=m*o*(t+1)
    return ubar,theta
    
ubar,theta=extract_policy(pi,T,m,o,L_reg)

#def simulate_my_controller(sys,x_0,T,A,B,C):
fS=open("data/%s/S.pkl"%f,"rb")
(A,B,C)=pickle.load(fS)
n=A.shape[0]
np.random.seed(None)
x,u,v,w,y={},{},{},{},{}
U,Y,xi,e_sim={},{},{},{}
v[0]=0.01*(np.random.random((o,1))-0.5)
x[0]=(np.random.random((n,1))-0.5)*1
y[0]=np.dot(C,x[0])+v[0]
for t in range(T):
    w[t]=0.05*(np.random.random((n,1))-0.5)+1*0
    v[t+1]=0.05*(np.random.random((o,1))-0.5)
    # Controls
    if t==0:
        u[t]=ubar[t]+np.dot(theta[0],y[0])
    else:
        Y[t-1]=np.vstack([y[tau] for tau in range(t)])
        U[t-1]=np.vstack([u[tau] for tau in range(t)])
        e_sim[t-1]=y[t]-np.dot(M[t-1],Y[t-1])-np.dot(N[t-1],U[t-1])
        xi[t]=np.vstack([y[0]]+[e_sim[tau] for tau in range(t)])
        u[t]=ubar[t]+(1.5)*np.dot(theta[t],xi[t])
    x[t+1]=np.dot(A,x[t])+np.dot(B,u[t])+w[t]
    y[t+1]=np.dot(C,x[t+1])+v[t+1]
    Y[t]=np.vstack([y[tau] for tau in range(t+1)])
    U[t]=np.vstack([u[tau] for tau in range(t+1)])
    e_sim[t]=y[t+1]-np.dot(M[t],Y[t])-np.dot(N[t],U[t])
    xi[t+1]=np.vstack([y[0]]+[e_sim[tau] for tau in range(t+1)])
    
phi[T]=phi["bar",T]+sum([np.dot(alpha[tau,T],np.dot(theta[tau],J[tau])) for tau in range(T)])
y_bar=sum([np.dot(alpha[tau,T],ubar[tau]) for tau in range(T)])
y_new=y_bar+np.dot(phi[T],xi[T])
#            
#    x,y,u,e,xi={},{},{},{},{}
#    x[0]=x_0
#    Y,U={},{}
#    for t in range(T+1):
#        print("simulating time:",t)
#        y[t]=np.dot(C,x[t])+v[t]
#        if t==T:
#            return x,y,u,x
#        if t==0:
#            xi[0]=y[0]
#        else:
#            Y[t-1]=np.vstack([y[tau] for tau in range(t)])
#            U[t-1]=np.vstack([u[tau] for tau in range(t)])
#            e[t-1]=y[t]-np.dot(sys.M[t-1],Y[t-1])-np.dot(sys.N[t-1],U[t-1])
#            xi[t]=np.vstack([y[0]]+[e[tau] for tau in range(t)])
#        u[t]=ubar[t]+np.dot(theta[t],xi[t])
#        x[t+1]=np.dot(A,x[t])+np.dot(B[t],u[t])+w[t]