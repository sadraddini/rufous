#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:25:36 2020

@author: sadra
"""

import numpy as np
import matplotlib.pyplot as plt
import pydrake
import pydrake.systems
from pydrake import systems as PS
#import pydrake.systems.controllers as PCS
import pickle
plt.ioff()

L=1.0
R=0.3
dt=0.2
g=10

from PIL import Image,ImageOps
my_dpi=11

def image_pendulum(theta,t=0,i=0):
    fig,ax=plt.subplots()
    xc=L*np.sin(theta)
    yc=L*(np.cos(theta))
#    ax.plot([0,xc],[0,yc],LineWidth=0.1)
#    ax.plot([0,xc],[0,yc],LineWidth=1.5)
    ax.plot([0,xc],[0,yc],LineWidth=0.5)    
    c=plt.Circle((xc,yc),R)
    ax.add_artist(c)
    fig.set_size_inches((1,1))
    ax.set_facecolor("orange")
    plt.axis("equal")
    plt.axis('off')
    ax.set_xlim([-0.8,0.8])
    ax.set_ylim([-0.2,1.5])
    s="images/my_pendulum_%d.png"%(t)
    fig.savefig(s,dpi=my_dpi)
    a=Image.open(s)
    a=ImageOps.grayscale(a)
    b=np.array(a)
    fig.clf()
    return b

#y=image_pendulum(-0.3)
#noise=np.random.randint(-50,50,size=y.shape)
#y_new=y+noise
#y_new=np.minimum(np.ones(y.shape)*255,y_new)
#y_new=np.maximum(np.zeros(y.shape),y_new)
#new_image=Image.fromarray(y_new)
#new_image.show()
A=np.array([[1,dt],[g*dt,1]])
B=np.array([0,dt]).reshape(2,1)
import scipy.linalg as sl
X=sl.solve_discrete_are(A,B,np.eye(2)*1,np.eye(1))
Y=np.linalg.inv(R+np.linalg.multi_dot([B.T,X,B]))
K=np.linalg.multi_dot([Y,B.T,X,A])
print(np.linalg.eigvals(A-np.dot(B,K)))
#K,Q=PS.controllers.DiscreteTimeLinearQuadraticRegulator(A,B,np.eye(2)*100,np.eye(1))

def get_a_run(T,i):
    x={}
    y={}
    u={}
    x[0,0]=1.5*(np.random.random()-0.5)
    x[1,0]=3*(np.random.random()-0.5)
    for t in range(T):
        y[t]=image_pendulum(x[0,t],t,i).reshape(my_dpi**2,1)
        u[t]=np.array(-K[0,0]*x[0,t]-K[0,1]*x[1,t]+5*(np.random.random()-0.5)).reshape(1,1)
        w=0.2*(np.random.random()-0.5)
        x[0,t+1]=x[0,t]+dt*x[1,t]+0.01*(np.random.random()-0.5)
        x[1,t+1]=x[1,t]+g*dt*np.sin(x[0,t])+u[t][0,0]*dt+w
    y[T]=image_pendulum(x[0,T],T,i).reshape(my_dpi**2,1)
    return x,u,y

T=8
# Training
if True:
    x={}
    y={}
    u={}
    for i in range(900):
        print("i",i)
        x[i],u[i],y[i]=get_a_run(T,i)
    f=open("training_data_%d.pkl"%my_dpi,"wb")
    pickle.dump((x,y,u),f)
    f.close()

# test
if True:
    x={}
    y={}
    u={}
    for i in range(100):
        print("i",i)
        x[i],u[i],y[i]=get_a_run(T,i)
    f=open("training_test_%d.pkl"%my_dpi,"wb")
    pickle.dump((x,y,u),f)
    f.close()