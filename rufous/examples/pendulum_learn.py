#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:42:07 2020

@author: sadra
"""

import numpy as np
import pickle
from PIL import Image,ImageOps
#from ..model_learning import model_least_squares,learn_model

my_dpi=18
f=open("training_data_%d.pkl"%my_dpi,"rb")
x,y,u=pickle.load(f)

f_test=open("training_test_%d.pkl"%my_dpi,"rb")
x_test,y_test,u_test=pickle.load(f_test)

S=model_least_squares(y,u,w_M=20000*10**-2,w_N=20000*10**-2,T_window={t:4 for t in range(10)})
#M2,N2,e_bar2,e2=learn_model(y,u,w_reg=10**-1)
test_prediction(y_test,u_test,S,color='red')
test_prediction(y,u,S,color='blue')
#test_prediction(y_test,u_test,M2,N2,e_bar2,color='orange')
#test_prediction(y,u,M2,N2,e_bar2,color='cyan')

raise 1

if True:
    n=7
    t=3
    y_current=np.vstack([y_test[n][tau] for tau in range(t+1)])
    u_current=np.vstack([np.array(u_test[n][tau])  for tau in range(t+1)]) 
    y_prediction_t=e_bar[t]+np.dot(M[t],y_current)+np.dot(N[t],u_current)
    prediction_image=Image.fromarray(y_prediction_t.reshape(my_dpi,my_dpi))
    prediction_image=prediction_image.convert('L')
    prediction_image=prediction_image.resize((my_dpi*50,my_dpi*50))
    prediction_image.save("prediction.jpg")
    actual_image=Image.fromarray(y_test[n][t+1].reshape(my_dpi,my_dpi))
    actual_image=actual_image.convert("RGB")
    actual_image=actual_image.resize((my_dpi*50,my_dpi*50))
    actual_image.save("actual.png")
    now_image=Image.fromarray(y_test[n][t].reshape(my_dpi,my_dpi))
    now_image=now_image.convert("RGB")
    now_image=now_image.resize((my_dpi*50,my_dpi*50))
    now_image.save("now.png")
    print("torque",u_test[n][t][0,0])
    error=abs(y_prediction_t-y_test[n][t+1])
    error_now=abs(y_prediction_t-y_test[n][t])
    error_next=abs(y_prediction_t-y_test[n][t+2])
    print("error of prediction",np.linalg.norm(error,2))
    print("error of current image",np.linalg.norm(error_now,2))
    print("error of next image",np.linalg.norm(error_next,2))
    error_image=Image.fromarray(error.reshape(my_dpi,my_dpi))
    error_image=error_image.convert('L')
    error_image=error_image.resize((my_dpi*50,my_dpi*50))
    error_image.save("error.png")

# Multi-step Prediction    
if True:
    n=7
    t=1
    T_prediction=7
    y_prediction={}
    # Initials
    for tau in range(t+1):
        actual_image=Image.fromarray(y_test[n][tau].reshape(my_dpi,my_dpi))
        actual_image=actual_image.convert("RGB")
        actual_image=actual_image.resize((my_dpi*50,my_dpi*50))
        actual_image.save("multi_actual_pre_%d.png"%tau)
    # Predictions    
    for k in range(T_prediction):
        y_current=np.vstack([y_test[n][tau] for tau in range(t+1)]+[y_prediction[tau] for tau in range(k)])
        u_current=np.vstack([np.array(u_test[n][tau]) for tau in range(t+1)]+[u_test[n][t+1+tau] for tau in range(k)]) 
        y_prediction[k]=e_bar[t+k]+np.dot(M[t+k],y_current)+np.dot(N[t+k],u_current)
        prediction_image=Image.fromarray(y_prediction[k].reshape(my_dpi,my_dpi))
        prediction_image=prediction_image.convert('L')
        prediction_image=prediction_image.resize((my_dpi*50,my_dpi*50))
        prediction_image.save("multi_prediction_%d.jpg"%k)
    
        actual_image=Image.fromarray(y_test[n][t+k+1].reshape(my_dpi,my_dpi))
        actual_image=actual_image.convert("RGB")
        actual_image=actual_image.resize((my_dpi*50,my_dpi*50))
        actual_image.save("multi_actual_%d.png"%k)
        

a,b={},{}  
for n in range(20):
    for t in range(7):
        y_current=np.vstack([y_test[n][tau] for tau in range(t+1)])
        u_current=np.vstack([np.array(u_test[n][tau])  for tau in range(t+1)]) 
        y_prediction_t=e_bar[t]+np.dot(M[t],y_current)+np.dot(N[t],u_current)
        error=np.linalg.norm(y_prediction_t-y_test[n][t+1],2)
        error_previous=np.linalg.norm(y_prediction_t-y_test[n][t],2)
        error_next=np.linalg.norm(y_prediction_t-y_test[n][t+2],2)
        a[n,t]=error_next/error
        b[n,t]=error_previous/error
aa=np.array(list(a.values()))
bb=np.array(list(b.values()))



def image_pendulum(theta,t=0):
    L=1.0
#    R=0.7
    R=0.3
    fig,ax=plt.subplots()
    xc=L*np.sin(theta)
    yc=L*(np.cos(theta))
#    ax.plot([0,xc],[0,yc],LineWidth=0.1)
    ax.plot([0,xc],[0,yc],LineWidth=0.5)
#    ax.plot([0,xc],[0,yc],LineWidth=1.5)
    c=plt.Circle((xc,yc),R)
    ax.add_artist(c)
    fig.set_size_inches((1,1))
    plt.axis("equal")
    plt.axis('off')
    ax.set_xlim([-0.8,0.8])
    ax.set_ylim([-0.2,1.5])
    s="images/a_test_pendulum_%d.png"%t
    fig.savefig(s,dpi=my_dpi)
    a=Image.open(s)
    a=ImageOps.grayscale(a)
    b=np.array(a)
    fig.clf()
    return b

raise 1


T=5
o=my_dpi**2
m=1
zeta=np.mean(E[T],1).reshape(o*(T+1),1)
sigma=np.dot(E[T],E[T].T)
y_target=image_pendulum(0,0).reshape(my_dpi**2,1)

y0=np.mean(E[0],1).reshape(o,1)
R,Q,y_ref,u_ref={},{},{},{}
for t in range(T):
    y_ref[t]=y_target
    u_ref[t]=np.ones((m,1))*0
    R[t]=np.eye(m)*5*10**(2+0*(t==0))
    Q[t]=np.eye(o)*t
Q[T]=np.eye(o)*T
y_ref[T]=y_target

goal_image=Image.fromarray(y_target.reshape(my_dpi,my_dpi))
goal_image=goal_image.convert("RGB")
goal_image=goal_image.resize((my_dpi*50,my_dpi*50))
goal_image.save("goal.png")



pi,ubar,theta=opeflqr(M,N,y_ref,u_ref,Q,R,zeta,sigma,T)

if True:
    for t in range(T):
        theta_adjusted=np.abs(theta[t][:,theta[t].shape[1]-my_dpi**2:])/np.max((np.abs(theta[t][:,theta[t].shape[1]-my_dpi**2:])))*256
        u_adjusted=np.multiply(np.mean(e[t],1).reshape(o,1),theta_adjusted.T)
        u_adjusted= np.abs(u_adjusted)/np.max(np.abs(u_adjusted))*256
        K_image=Image.fromarray(u_adjusted.reshape(my_dpi,my_dpi))
        K_image=K_image.convert('L')
        K_image=K_image.resize((my_dpi*50,my_dpi*50))
        K_image.save("K_%d.png"%t)
    
# Now a run of the system
import matplotlib.pyplot as plt

dt=0.2
g=10

N_simulate=150
x_final={}
x_run={}
for j in range(N_simulate):
    print(j,)
    x_run[j]={}
    if True:
        y_run={}
        u_run={}
        Y_run={}
        U_run={}
        error_run,zeta_run={},{}
        y_run_prediction={}
        x_run[j][0,0]=0.4*(np.random.random()-0.5)
        x_run[j][1,0]=2*(np.random.random()-0.5)
        for t in range(T):
            y_run[t]=image_pendulum(x_run[j][0,t],t).reshape(my_dpi**2,1)+0*(np.random.random((my_dpi**2,1))-0.5)
            if t!=0:
                error_run[t-1]=y_run[t]-y_run_prediction[t]
#                print(t-1,"-error=",np.linalg.norm(error_run[t-1]-e_bar[t-1]))
            if t==0:
                zeta_run[0]=y_run[0]
            else:
                zeta_run[t]=np.vstack([y_run[0]]+[error_run[tau] for tau in range(t)])
            u_run[t]=ubar[t]+np.dot(theta[t],zeta_run[t])
            u_run[t]*=1
            Y_run[t]=np.vstack([y_run[tau] for tau in range(t+1)])
            U_run[t]=np.vstack([u_run[tau] for tau in range(t+1)])
            y_run_prediction[t+1]=np.dot(M[t],Y_run[t])+np.dot(N[t],U_run[t])
            w=0.2*(np.random.random()-0.5)
            x_run[j][0,t+1]=x_run[j][0,t]+dt*x_run[j][1,t]+0.01*(np.random.random()-0.5)
            x_run[j][1,t+1]=x_run[j][1,t]+g*dt*np.sin(x_run[j][0,t])+u_run[t][0,0]*dt+w
        y_run[T]=image_pendulum(x_run[j][0,T],T).reshape(my_dpi**2,1)
    x_final[j]=x_run[j][0,T]
    
#if True:
#    """
#    This is LQR Version
#    """
#    x_final={}
#    x_run={}
#    for j in range(N_simulate):
#        print(j,)
#        x_run[j]={}
#        u_run={}
#        U_run={}
#        error_run,zeta_run={},{}
#        y_run_prediction={}
#        x_run[j][0,0]=2*(np.random.random()-0.5)
#        x_run[j][1,0]=100*(np.random.random()-0.5)
#        for t in range(T):
#            u[t]=np.array(-K[0,0]*x_run[j][0,t]-K[0,1]*x_run[j][1,t]).reshape(1,1)
#            w=0.2*(np.random.random()-0.5)
#            x_run[j][0,t+1]=x_run[j][0,t]+dt*x_run[j][1,t]+0.01*(np.random.random()-0.5)
#            x_run[j][1,t+1]=x_run[j][1,t]+g*dt*np.sin(x_run[j][0,t])+u[t][0,0]*dt+w
#    x_final[j]=x_run[j][0,T]
    
import matplotlib.pyplot as myplot
xT=np.array(list(x_final.values()))
fig,ax=myplot.subplots()
ax.hist(xT)
fig.savefig("final_histogram.png")

fig2,ax2=myplot.subplots()
for j in range(N_simulate):
    ax2.plot([x_run[j][0,t] for t in range(T+1)],color='red')
    ax2.set_xlabel(r"time")
    ax2.set_ylabel(r"$\theta$")
    ax2.set_title(r"Inverted Pendulum Trajectories")
    ax2.set_ylim([-2.5,2.5])
    fig2.savefig("trajectories.png")
    
fig3,ax3=myplot.subplots()
#    ax2.plot([x_run[j][0,t] for t in range(T+1)],color='red')
ax3.plot([x_run[j][0,0] for j in range(N_simulate)],[x_run[j][1,0] for j in range(N_simulate)],'o',color='red')
ax3.plot([x_run[j][0,T] for j in range(N_simulate)],[x_run[j][1,T] for j in range(N_simulate)],'o',color='green')
ax3.set_xlabel(r"$\theta$")
ax3.set_ylabel(r"$\dot{\theta}$")
ax3.set_title(r"Inverted Pendulum Trajectories")
#ax3.set_ylim([-2.5,2.5])
#ax3.set_xlim([-10,10])
fig3.savefig("start_final.png")

fig3,ax3=myplot.subplots()    
for j in range(N_simulate):
    if abs(x_final[j])<0.2:
        ax3.plot([x_run[j][0,t] for t in range(1)],[x_run[j][1,t] for t in range(1)],'*',color='green')
    else:
        ax3.plot([x_run[j][0,t] for t in range(1)],[x_run[j][1,t] for t in range(1)],'.',color='red')
    ax3.set_ylabel(r"$\dot{\theta}$")
    ax3.set_xlabel(r"$\theta$")
    ax3.set_title(r"Inverted Pendulum Region of Attraction")
    ax3.set_ylim([-1,1])
    ax3.set_xlim([-0.5,0.5])
    fig3.savefig("ROA.png")
