#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 20:04:21 2020

@author: sadra
"""

import numpy as np
import pickle

my_file=open("data/%s/system.pkl"%f,"rb")
(M,N,e_bar,e,E)=pickle.load(my_file)
T=len(N)
o,m=N[0].shape
N_data=e[0].shape[1]


zeta_T=np.mean(E[T],1).reshape(o*(T+1),1)
Sigma_T=np.dot(E[T],E[T].T)#*np.random.random((12,12))

y_target=np.ones((o,1))*15
R,Q,y_ref,u_ref={},{},{},{}
for t in range(T):
    y_ref[t]=np.zeros((o,1))
    u_ref[t]=np.zeros((m,1))
    R[t]=np.eye(m)*1
    Q[t]=np.eye(o)*1
Q[T]=np.eye(o)*100
y_ref[T]=y_target

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
    print(t,start,end,"\t",end-start)
    Gamma[t,"theta"][start:end,:o*(t+1)]=np.eye(o*(t+1))

alpha,phi={},{}
# Alpha
phi["bar",0]=I[0]
alpha[0,0]=np.zeros((o,m))
for t in range(T):
    for k in range(t+1):
        alpha[k,t+1]=alpha[0,0]+sum([np.dot(M[t][:,o*tau:o*(tau+1)],alpha[k,tau]) for tau in range(k+1,t+1) ])+N[t][:,m*k:m*(k+1)]
    phi["bar",t+1]=sum([np.dot(M[t][:,o*tau:o*(tau+1)],phi["bar",tau]) for tau in range(0,t+1) ]) + I[t+1]    

# Sanity check for phi and alpha's
#for t in range(T+1):
#    phi[t]=phi["bar",t]+sum([np.linalg.multi_dot([alpha[tau,t],pi_program,Gamma[tau,"theta"]]) for tau in range(t)])
#    phi["new",t]=phi["bar",t]+sum([np.linalg.multi_dot([alpha[tau,t],np.hstack((theta[tau],np.zeros((m,Gamma[tau,'theta'].shape[1]-theta[tau].shape[1])) ))]) for tau in range(t)])

# ============ The E, F list for J_x
# Term 1
#E_u_1=[np.linalg.multi_dot([alpha[k,t].T,Q[t],alpha[tau,t]]) \
#     for t in range(T+1) for k in range(t) for tau in range(t)]
#F_u_1=[np.linalg.multi_dot([Gamma[tau,"u"],Gamma[k,"u"].T])
#    for t in range(T+1) for k in range(t) for tau in range(t)]
## Term 2
#E_u_2=[np.linalg.multi_dot([alpha[k,t].T,Q[t],alpha[tau,t]]) \
#     for t in range(T+1) for k in range(t) for tau in range(t)]
#F_u_2=[np.linalg.multi_dot([Gamma[tau,"theta"],zeta_T,Gamma[k,"u"].T])
#    for t in range(T+1) for k in range(t) for tau in range(t)]
## Term 3
#E_u_3=[np.linalg.multi_dot([alpha[k,t].T,Q[t],alpha[tau,t]]) \
#     for t in range(T+1) for k in range(t) for tau in range(t)]
#F_u_3=[np.linalg.multi_dot([Gamma[tau,"u"],zeta_T.T,Gamma[k,"theta"].T])
#    for t in range(T+1) for k in range(t) for tau in range(t)]
## Term 4
#E_u_4=[np.linalg.multi_dot([alpha[k,t].T,Q[t],alpha[tau,t]]) \
#     for t in range(T+1) for k in range(t) for tau in range(t)]
#F_u_4=[np.linalg.multi_dot([Gamma[tau,"theta"],Sigma_T,Gamma[k,"theta"].T])
#    for t in range(T+1) for k in range(t) for tau in range(t)]
## Sum all
#E_x=E_u_1+E_u_2+E_u_3+E_u_4
#F_x=F_u_1+F_u_2+F_u_3+F_u_4
#
#E_x_2=[np.linalg.multi_dot([alpha[k,t].T,Q[t],alpha[tau,t]]) \
#     for t in range(T+1) for k in range(t) for tau in range(t)]
#F_x_2=[np.linalg.multi_dot([Gamma[tau,"u"],Gamma[tau,"u"].T])+\
#       np.linalg.multi_dot([Gamma[tau,"theta"],zeta_T,Gamma[k,"u"].T])+\
#       np.linalg.multi_dot([Gamma[tau,"u"],zeta_T.T,Gamma[tau,"theta"].T])+\
#       np.linalg.multi_dot([Gamma[tau,"theta"],Sigma_T,Gamma[k,"theta"].T])
#    for t in range(T+1) for k in range(t) for tau in range(t)]

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
        GtZGu[tau,k][start:end,k:k+1]=zeta_T[0:(tau+1)*o,:]

GtSGt={}
for tau in range(T):
    for k in range(T):
        GtSGt[tau,k]=np.zeros((L,L))
        start_row=T+o*int(tau*(tau+1)/2)
        end_row=start_row+(tau+1)*o
        start_column=T+o*int(k*(k+1)/2)
        end_column=start_column+(k+1)*o
        GtSGt[tau,k][start_row:end_row,start_column:end_column]=Sigma_T[0:(tau+1)*o,0:(k+1)*o]
        
terms_1=[(np.linalg.multi_dot([alpha[k,t].T,Q[t],alpha[tau,t]]), \
            GuGu[tau,k] ) \
            for t in range(T+1) for k in range(t) for tau in range(t)]
terms_2=[(np.linalg.multi_dot([alpha[k,t].T,Q[t],alpha[tau,t]]), \
            GtZGu[tau,k] ) \
            for t in range(T+1) for k in range(t) for tau in range(t)]
terms_3=[(np.linalg.multi_dot([alpha[k,t].T,Q[t],alpha[tau,t]]), \
            GtZGu[k,tau].T ) \
            for t in range(T+1) for k in range(t) for tau in range(t)]
terms_4=[(np.linalg.multi_dot([alpha[k,t].T,Q[t],alpha[tau,t]]), \
            GtSGt[tau,k] ) \
            for t in range(T+1) for k in range(t) for tau in range(t)]
terms_5=[(R[t], GuGu[t,t]+GtZGu[t,t]+GtZGu[t,t].T+GtSGt[t,t] ) 
            for t in range(T)]
terms=terms_1+terms_2+terms_3+terms_4+terms_5
# Class II term 1
G_x=-sum([np.linalg.multi_dot([alpha[k,t].T,Q[t],\
        np.linalg.multi_dot([phi["bar",t],zeta_T,Gamma[k,"u"].T])+
        np.linalg.multi_dot([-y_ref[t],Gamma[k,"u"].T])+
        np.linalg.multi_dot([phi["bar",t],Sigma_T,Gamma[k,"theta"].T])+
        np.linalg.multi_dot([-y_ref[t],zeta_T.T,Gamma[k,"theta"].T])
                            ]) \
     for t in range(T+1) for k in range(t)])
    
# ============ The E, F list for J_u
E_u=[R[t] for t in range(T)]
F_u=[np.linalg.multi_dot([Gamma[t,"u"],Gamma[t,"u"].T])+\
     np.linalg.multi_dot([Gamma[t,"theta"],zeta_T,Gamma[t,"u"].T])+\
     np.linalg.multi_dot([Gamma[t,"u"],zeta_T.T,Gamma[t,"theta"].T])+\
     np.linalg.multi_dot([Gamma[t,"theta"],Sigma_T,Gamma[t,"theta"].T])
     for t in range(T)]
G_u=-sum([np.linalg.multi_dot([R[t],-u_ref[t],Gamma[t,"u"].T])+\
     np.linalg.multi_dot([R[t],-u_ref[t],zeta_T.T,Gamma[t,"theta"].T])
     for t in range(T)])

# The controls now!
#E=E_x+E_u
#F=F_x+F_u
G=G_x+G_u
#assert(len(E)==len(F))
#AX=sum([np.kron(F[i].T,E[i]) for i in range(len(E))])
AX=sum([np.kron(ff.T,ee) for (ee,ff) in terms])
bX=G.T.flatten()
X=np.linalg.solve(AX, bX)
#X,res,rank,singulars=np.linalg.lstsq(AX,bX)
pi_oflqr=X.reshape(T+o*int(T*(T+1)/2),m).T
