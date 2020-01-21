#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 21:11:37 2019

@author: sadra
"""

import numpy as np
import pickle

try:
    import pydrake.solvers.mathematicalprogram as MP
    import pydrake.solvers.gurobi as Gurobi_drake
    global gurobi_solver,OSQP_solver
    gurobi_solver=Gurobi_drake.GurobiSolver()
    import pydrake.symbolic as sym
except:
    print("Error in loading Drake Mathematical Program")

def triangular_stack(A,B):
    q=B.shape[1]-A.shape[1]
    if q>=0:
        return np.vstack((np.hstack((A,np.zeros((A.shape[0],q)))),B))
    else:
        return np.vstack((A,np.hstack((B,np.zeros((B.shape[0],-q))))))

def triangular_stack_list(list_of_matrices):
    N=len(list_of_matrices)
    if N==0:
        raise NotImplementedError
    elif N==1:
        return list_of_matrices[0]
    else:
        J=triangular_stack(list_of_matrices[0],list_of_matrices[1])
        for t in range(2,N):
            J=triangular_stack(J,list_of_matrices[t])
        return J
    
    
def synthesis_program(M,N,e,e_bar,E,y_ref,u_ref,Q,R):
    o,m,T=M[0].shape[0],N[0].shape[1],len(M)
    prog=MP.MathematicalProgram()
    # Add Variables
    phi,theta,Phi,Theta={},{},{},{}
    y_tilde,u_tilde={},{} 
    Y_tilde,U_tilde={},{}
    # Initial Condition
    y_tilde[0]=np.zeros((o,1))
    phi[0]=np.eye(o)
    # Main variables
    for t in range(T):
        theta[t]=prog.NewContinuousVariables(m,o*(t+1),"theta%d"%t)
        u_tilde[t]=prog.NewContinuousVariables(m,1,"u_tilde%d"%t)
    # Now we the dynamics
    Phi[0],Theta[0]=phi[0],theta[0]
    for t in range(T):
        Y_tilde[t]=np.vstack([y_tilde[tau] for tau in range(t+1)])
        U_tilde[t]=np.vstack([u_tilde[tau] for tau in range(t+1)])
        y_tilde[t+1]=np.dot(M[t],Y_tilde[t])+np.dot(N[t],U_tilde[t])
        phi[t+1]=np.hstack(( np.dot(M[t],Phi[t])+np.dot(N[t],Theta[t]),np.eye(o) ))
        Phi[t+1]=triangular_stack(Phi[t],phi[t+1])
        if t!=T-1:
            Theta[t+1]=triangular_stack(Theta[t],theta[t+1])
    # Cost X
    J=0
    for t in range(T+1):
        Sigma=np.dot(E[t],E[t].T)
        E_bar=np.mean(E[t],1).reshape(E[t].shape[0],1)
        J+= np.trace(np.linalg.multi_dot([Sigma,phi[t].T,Q[t],phi[t]]))+\
            np.linalg.multi_dot([y_tilde[t].T-y_ref[t].T,Q[t],y_tilde[t]-y_ref[t]])[0,0]+\
            2*np.linalg.multi_dot([E_bar.T,phi[t].T,Q[t],y_tilde[t]-y_ref[t]])[0,0]
    for t in range(T):
        Sigma=np.dot(E[t],E[t].T)
        E_bar=np.mean(E[t],1).reshape(E[t].shape[0],1)
        print(t,E_bar.shape,theta[t].shape,R[t].shape,u_tilde[t].shape)
        print(t,Sigma.shape,R[t].shape,theta[t].shape)
        J+= np.trace(np.linalg.multi_dot([Sigma,theta[t].T,R[t],theta[t]]))+\
            np.linalg.multi_dot([u_tilde[t].T-u_ref[t].T,R[t],u_tilde[t]-u_ref[t]])[0,0]+\
            2*np.linalg.multi_dot([E_bar.T,theta[t].T,R[t],u_tilde[t]-u_ref[t]])[0,0]
    prog.AddQuadraticCost(J)
    print("Now solving the QP")
    result=gurobi_solver.Solve(prog,None,None)
    if result.is_success():
        print("Synthesis Success!","\n"*5)
#        print "D=",result.GetSolution(D)
        theta_n={t:result.GetSolution(theta[t]).reshape(theta[t].shape) for t in range(0,T)}
        u_tilde_n={t:result.GetSolution(u_tilde[t]).reshape(u_tilde[t].shape) for t in range(0,T)}
#        phi_n={t:sym.Evaluate(result.GetSolution(phi[t])).reshape(phi[t].shape) for t in range(0,T+1)}
        return u_tilde_n,theta_n
    else:
        print("Synthesis Failed!")            
            
        
    
#clock=214
#f=str(clock)    
my_file=open("data/%s/system.pkl"%f,"rb")
(M,N,e_bar,e,E)=pickle.load(my_file)
T=len(N)
o,m=N[0].shape
N_data=e[0].shape[1]

   
def unit(n,i):
    x=np.zeros((n,1))
    x[i,0]=1
    return x


def extract_policy(pi,T,m,o,L_reg):
    assert len(pi)==m*T*(1+(T+1)/2*o)
    ubar,theta={},{}
    k=0
    for t in range(T):
        ubar[t]=pi[k:k+m,:]/L_reg
        k+=m
    for t in range(T):
        x=pi[k:k+m*o*(t+1),:]
        theta[t]=x.reshape(m,o*(t+1))
        k+=m*o*(t+1)
    return ubar,theta

def extract_policy_simple(pi,T,m,o):
    assert pi.shape[0]==m
    assert pi.shape[1]==int(T*(1+(T+1)/2*o))
    ubar,theta={},{}
    k=0
    for t in range(T):
        ubar[t]=pi[:,k:k+1]
        k+=1
    for t in range(T):
        x=pi[:,k:k+o*(t+1)]
        theta[t]=x.reshape(m,o*(t+1))
        k+=o*(t+1)
    return ubar,theta


def synthesis_LS(y_target,M,N,e,e_bar,E,L_reg):
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
    ubar,theta=extract_policy(pi,T,m,o,L_reg)
    return ubar,theta,phi,alpha,J
    

print("Now doing Synthesis")    

# The model for L and J and I_0
#L,J={},{}
#I_0=np.kron(unit(T+1,0).T,np.eye(o))
#for t in range(T):
#    L[t]=np.kron(unit(T+1,t+1).T,np.eye(o))
#for t in range(T+1):
#    X=np.zeros(( o*(t+1),o*(T+1) ))
#    X[:,:o*(t+1)]=np.eye( o*(t+1) )
#    J[t]=X
    
# Now hard part: alpha and phi
#phi,alpha={},{}
#phi["bar",0]=I_0
#alpha[0,0]=np.zeros((o,m))
#for t in range(T):
#    for k in range(t+1):
#        alpha[k,t+1]=alpha[0,0]+sum([np.dot(M[t][:,o*tau:o*(tau+1)],alpha[k,tau]) for tau in range(k+1,t+1) ])+N[t][:,m*k:m*(k+1)]
#    phi["bar",t+1]=sum([np.dot(M[t][:,o*tau:o*(tau+1)],phi["bar",tau]) for tau in range(0,t+1) ]) + L[t]
#K={}
#k=0
#for tau in range(T):
#    for l in range(m):
#        k+=o*tau+o*(l>0)
#        K[tau,l]=k
#length=int(T*(T+1)/2*m*o)
#SS=np.zeros((o,length+m*T,o*(T+1)))
## Fill
#for i in range(o*(T+1)):
#    for tau in range(T):
#        for l in range(m):
#            if i<o*(tau+1):
#                SS[:,m*T+K[tau,l]+i,i]=alpha[tau,T][:,l]
#
#
#Su=np.hstack([alpha[t,T] for t in range(T)])
#I_u=np.hstack((np.eye(m*T), np.zeros((m*T,length)) ))
#Su_final=np.dot(Su,I_u)
#
#
## Writing things as min||R + QX||^2_2+w||X||^2_2
#y_target=np.ones((o,1))*2
#L_reg=10**-1
#R,Q={},{}
#for j in range(N_data):
#    R[j]=np.dot(phi["bar",T],E[T][:,j].reshape(E[T].shape[0],1))-y_target
#    Q[j]=Su_final/L_reg+np.dot(SS,E[T][:,j].reshape(E[T].shape[0],1)).reshape(SS.shape[0],SS.shape[1])
#
#R["all"]=np.vstack([R[j] for j in range(N_data)])
#Q["all"]=np.vstack([Q[j] for j in range(N_data)])
#pi=-np.dot(np.linalg.inv(np.dot(Q["all"].T,Q["all"])+L_reg*np.eye(Q["all"].shape[1])),\
#          np.dot(Q["all"].T,R["all"]))
## Error analysis
#print("average final error=",np.linalg.norm(R["all"]+np.dot(Q["all"],pi),'fro')/N_data**0.5)
#ubar,theta=extract_policy(pi,T,m,o,L_reg)
#
#
#K={}
#k=0
#for tau in range(T):
#    for l in range(m):
#        k+=o*tau+o*(l>0)
#        K[tau,l]=k
#length=int(T*(T+1)/2*m*o)
#SS=np.zeros((o,length+m*T,o*(T+1)))
## Fill
#for i in range(o*(T+1)):
#    for tau in range(T):
#        for l in range(m):
#            if i<o*(tau+1):
#                SS[:,m*T+K[tau,l]+i,i]=alpha[tau,T][:,l]
#
#
#Su=np.hstack([alpha[t,T] for t in range(T)])
#I_u=np.hstack((np.eye(m*T), np.zeros((m*T,length)) ))
#Su_final=np.dot(Su,I_u)


y_target=np.ones((o,1))*15
ubar,theta,phi,alpha,J=synthesis_LS(y_target,M,N,e,e_bar,E,L_reg=10**0)
pi_regular=np.hstack([ubar[t] for t in range(T)]+[theta[t] for t in range(T)])


# Synthesis program
R,Q,y_ref,u_ref={},{},{},{}
for t in range(T):
    y_ref[t]=np.zeros((o,1))
    u_ref[t]=np.zeros((m,1))
    R[t]=np.eye(m)*1
    Q[t]=np.eye(o)*1
Q[T]=np.eye(o)*100
y_ref[T]=y_target
ubar,theta=synthesis_program(M,N,e,e_bar,E,y_ref,u_ref,Q,R)
pi_program=np.hstack([ubar[t] for t in range(T)]+[theta[t] for t in range(T)])

ubar,theta=extract_policy_simple(pi_new,T,m,o)

#def simulate_my_controller(sys,x_0,T,A,B,C):
fS=open("data/%s/S.pkl"%f,"rb")
(A,B,C)=pickle.load(fS)
n=A.shape[0]
N_test=100
delta=np.empty(N_test)
for i in range(N_test):
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
            u[t]=ubar[t]+(1)*np.dot(theta[t],xi[t])
        x[t+1]=np.dot(A,x[t])+np.dot(B,u[t])+w[t]
        y[t+1]=np.dot(C,x[t+1])+v[t+1]
        Y[t]=np.vstack([y[tau] for tau in range(t+1)])
        U[t]=np.vstack([u[tau] for tau in range(t+1)])
        e_sim[t]=y[t+1]-np.dot(M[t],Y[t])-np.dot(N[t],U[t])
        xi[t+1]=np.vstack([y[0]]+[e_sim[tau] for tau in range(t+1)])
        
    phiT=phi["bar",T]+sum([np.dot(alpha[tau,T],np.dot(theta[tau],J[tau])) for tau in range(T)])
    y_bar=sum([np.dot(alpha[tau,T],ubar[tau]) for tau in range(T)])
    y_new=y_bar+np.dot(phiT,xi[T])
        
    delta[i]=np.linalg.norm(y_new-y_target,2)
    delta[i]=np.linalg.norm(y[T]-y_target,2)

print(np.mean(delta))