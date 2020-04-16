#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:15:05 2020

@author: sadra
"""
import numpy as np
import pickle
try:
    import torch as tch
except:
    print("pytorch already loaded")

#f=open("my_system_8.pkl","rb")
#S=pickle.load(f)
T=S.T
o=S.o
m=S.m
M={}
N={}

mu={}
for t in range(T+1):
    _z=np.vstack( [S.zeta[tau] for tau in range(t+1)] )
    _z_2=np.dot(_z,_z.T)
    _z_mean=np.mean(_z,1).reshape(o*(t+1),1)
    mu[1,t]=tch.from_numpy ( _z_mean )
    mu[2,t]=tch.from_numpy ( _z_2 )

for t in range(T+1):
    for i in range(S.T_windows[t]):
        M[t,t-i] = tch.from_numpy ( S.M[t] [: , o*(S.T_windows[t]-i-1): o*(S.T_windows[t]-i) ] )
        N[t,t-i] = tch.from_numpy ( S.N[t] [: , m*(S.T_windows[t]-i-1): m*(S.T_windows[t]-i) ] )


def cost(ubar2,theta2,eta):
    theta,ubar={},{}
    for t in range(T):
        ubar[t] = tch.from_numpy(ubar2[t])
        ubar[t].requires_grad=True
        for j in range(t+1):        
            theta[t,j] = tch.from_numpy(theta2[t,j])
            theta[t,j].requires_grad=True
        
    ybar={}
    y_ref={}
    Q={}
    for t in range(T+1):
        ybar[t]=tch.from_numpy ( np.zeros((o,1)) )
        y_ref[t]=tch.from_numpy ( y_target )
        Q[t]=tch.from_numpy ( np.eye((o)) )
        
        
    for t in range(T):
        for i in range(t-S.T_windows[t]+1,t+1):
            ybar[t+1] = tch.mm( M[t,i] , ybar[i] ) + tch.mm( N[t,i] , ubar[i].double() )
    
    phi={}
    for t in range(T+1):
        phi[t,t] = tch.from_numpy ( np.eye((o)) )
    for t in range(T):
        for j in range(t+1):
            for i in range(max(j,t-S.T_windows[t]+1),t+1):
                phi[t+1,j] = tch.mm( M[t,i] , phi[i,j] ) + tch.mm( N[t,i] , theta[i,j].double() )
    
    # Cost function?
    
    # ybar-y-ref
    r={}
    J=0
    for t in range(T+1):
        r[t]=  ybar[t]  -  y_ref[t]
        phi[t]=tch.cat ( [phi[t,j] for j in range(t+1)] , dim=1 )        
        J += tch.chain_matmul( r[t].transpose(0,1) , Q[t], r[t] )
        J += 2*tch.chain_matmul( r[t].transpose(0,1) , Q[t], phi[t] , mu[1,t]  )  
        J += tch.trace ( tch.chain_matmul( phi[t].transpose(0,1) , Q[t], phi[t] , mu[2,t]  )  )     
        
        
    J.backward()
    grad={}
    for t in range(T):
        grad["u",t] = ubar[t].grad
        for j in range(t+1):     
            grad["theta",t,j] = theta[t,j].grad
            
    print("J=",J[0,0])
    
    ubar2,theta2={},{}
    for t in range(T):
        ubar2[t] = (ubar[t] - grad["u",t] * eta).detach().numpy()
        for j in range(t+1): 
            theta2[t,j] = (theta[t,j] - grad["theta",t,j] * eta).detach().numpy()
             
    return ubar2,theta2,J
    

# Decleration
#theta,ubar={},{}
#for t in range(T):
#    ubar[t] = tch.ones(m,1,requires_grad=True)
#    for j in range(t+1):        
#        theta[t,j] = tch.zeros(m, o, requires_grad=True)

theta2,ubar2={},{}
for t in range(T):
    ubar2[t] = np.zeros((m,1))
    for j in range(t+1):        
        theta2[t,j] = np.zeros((m,o))
    
eta=2*10**-17
for k in range(200):
    print("iteration",k)
#    eta/=1.1
    ubar2,theta2,J=cost(ubar2,theta2,eta)
    
if True:
    for t in range(T):
        theta_adjusted=np.abs(theta2[t,t])/np.max((np.abs(theta2[t,t])))*256
#        u_adjusted=np.multiply(np.mean(S.zeta[t+1],1).reshape(o,1),theta_adjusted.T)
#        u_adjusted= np.abs(u_adjusted)/np.max(np.abs(u_adjusted))*256
        K_image=Image.fromarray(theta_adjusted.reshape(my_dpi,my_dpi))
        K_image=K_image.convert('L')
        K_image=K_image.resize((my_dpi*50,my_dpi*50))
        K_image.save("theta_%d.png"%t)
        
ubar=ubar2
theta={}
for t in range(T):
    theta[t]=np.hstack([theta2[t,j] for j in range(t+1)])