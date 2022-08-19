import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve,bicgstab
from scipy.interpolate import LinearNDInterpolator,griddata
import copy
import time
import pandas as pd
import math
import solverFunctions as sf


def CalcMomentumResiduals(u,v,p,prob):
    # Calculates residuals of the momentum equations
    dt=np.inf
    Au,bu,_ = sf.MomentumEqU(u,v,p,prob)
    Av,bv,_ = sf.MomentumEqV(u,v,p,prob)
    u_vector = u[1:-1,1:-1].reshape(Au.shape[0],1)
    v_vector = v[1:-1,1:-1].reshape(Av.shape[0],1)
    u_res = np.square(Au.dot(u_vector)-bu).mean()
    v_res = np.square(Av.dot(v_vector)-bv).mean()
    return u_res,v_res

def CalcMomentumResiduals_A(u,v,Au,Av,bu,bv):
    # Calculates residuals of the momentum equations if the linear problem is given
    # with Au,Av,bu,bv
    u_vector = u[1:-1,1:-1].reshape(Au.shape[0],1)
    v_vector = v[1:-1,1:-1].reshape(Av.shape[0],1)
    u_res = np.square(Au.dot(u_vector)-bu).mean()
    v_res = np.square(Av.dot(v_vector)-bv).mean()
    return u_res,v_res

def CalcContinuityResiduals(u,v,prob):
    # Calcualtes the condinuity residuals
    # Calcualtes the mass mean square mass flow that does not satisfy the continuity
    # equation across all the pressure control volumes
    mp_e = prob.rho*prob.dy_p*u[1:,1:-1]
    mp_w = prob.rho*prob.dy_p*u[:-1,1:-1]
    mp_n = prob.rho*prob.dx_p*v[1:-1,1:]
    mp_s = prob.rho*prob.dx_p*v[1:-1,:-1]
    dmp = mp_w - mp_e + mp_s - mp_n
    con_res = np.square(dmp).mean()
    return con_res

class Log():
    # Structure for loging values through during the solving
    def __init__(self,n):
        self.dif = np.zeros(n)
        self.res = np.zeros(n)
        self.u_res = np.zeros(n)
        self.v_res = np.zeros(n)
        self.con_res = np.zeros(n)
        self.dt = np.zeros(n)
        self.itt = 0
        
def PlotFlow(u,v,prob):
    # Plots the flow-field
    X_n,Y_n = np.meshgrid(prob.xc_u,prob.yc_v)
    u_n = (u[:,:-1]+u[:,1:])/2
    v_n = (v[:-1,:]+v[1:,:])/2
    u_n = np.transpose(u_n)
    v_n = np.transpose(v_n)
    X,Y = np.meshgrid(np.linspace(0,1,u_n.shape[0]*2),np.linspace(0,1,u_n.shape[1]*2))
    points_n = np.concatenate([X_n.reshape(X_n.size,1),Y_n.reshape(Y_n.size,1)],axis=1)
    values_n = np.concatenate([u_n.reshape(u_n.size,1),v_n.reshape(v_n.size,1)],axis=1)
    interp = LinearNDInterpolator(points_n, values_n)
    values_i = interp(X,Y)
    u_c = values_i[:,:,0]
    v_c = values_i[:,:,1]
    plt.figure(figsize=(8,8))
    axes = plt.axes()
    axes.set_ylim([0,1])
    axes.set_xlim([0,1])
    plt.contourf(X,Y,np.sqrt(u_c**2+v_c**2))
    plt.colorbar()
    plt.streamplot(X,Y,u_c,v_c,color="k",linewidth=2,density=3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def PlotConvergence(log):
    plt.figure(figsize=(10,10))
    ax = plt.subplot(2,2,1)
    ax.plot(log.dif[:log.itt])
    ax.set_yscale("log")
    ax.title.set_text("difference")
    ax = plt.subplot(2,2,2)
    ax.plot(log.con_res[:log.itt])
    ax.set_yscale("log")
    ax.title.set_text("continuity residuals")
    ax = plt.subplot(2,2,3)
    ax.plot(log.u_res[:log.itt])
    ax.set_yscale("log")
    ax.title.set_text("u-momentum residuals")
    ax = plt.subplot(2,2,4)
    ax.plot(log.v_res[:log.itt])
    ax.set_yscale("log")
    ax.title.set_text("v-momentum residuals")
    plt.show()
    
    

