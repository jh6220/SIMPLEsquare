import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import pandas as pd
import solverFunctions as sf
from tqdm import tqdm 


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
        
def PlotFlow(u,v,prob,title=""):
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
    plt.title(title)
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
    
def CompareWithBenchmark(u,v,prob,Re):
    if Re>1000:
        NU = 20000
        nn = 10201
    else:
        NU = 5000
        nn = 2601
    df = pd.read_csv(r"benchmark_data\Re{}\{}NUcav.plt".format(Re,NU),delim_whitespace=True)
    points = df.iloc[NU:NU+nn,0:2].to_numpy()
    df = pd.read_csv(r"benchmark_data\Re{}\{}NUcav.var".format(Re,NU),delim_whitespace=True,header=None)
    vals = df.to_numpy()
    interp = LinearNDInterpolator(points, vals)
    
    # x-middle slice u-velocity
    ix_mid = int(np.floor(u.shape[0]/2))
    x_slice = prob.X_u[ix_mid,:]
    y_slice = prob.Y_u[ix_mid,:]
    u_slice = u[ix_mid,:]
    u_slice_interp = interp(x_slice,y_slice)[:,0]
    plt.plot(y_slice,u_slice)
    plt.plot(y_slice,u_slice_interp,'x')
    plt.legend(["Ghia et al 1982","N="+str(prob.nx)])
    plt.xlabel("y")
    plt.ylabel("u")
    plt.show()
    
    # y-middle slice v_velocity
    iy_mid = int(np.floor(v.shape[1]/2))
    x_slice = prob.X_v[:,iy_mid]
    y_slice = prob.Y_v[:,iy_mid]
    v_slice = v[:,iy_mid]
    v_slice_interp = interp(x_slice,y_slice)[:,1]
    plt.plot(x_slice,v_slice)
    plt.plot(x_slice,v_slice_interp,'x')
    plt.plot(x_slice,v_slice)
    plt.plot(x_slice,v_slice_interp,'x')
    plt.legend(["Ghia et al 1982","N="+str(prob.nx)])
    plt.xlabel("x")
    plt.ylabel("v")
    plt.show()
    
def SubplotFlow(a,b,u,v,prob,show=True):
    n = min(a*b,u.shape[0])
    plt.figure(figsize=(14,14))
    for i in tqdm(range(n)):
        X_n,Y_n = np.meshgrid(prob.xc_u,prob.yc_v)
        u_n = (u[i,:,:-1]+u[i,:,1:])/2
        v_n = (v[i,:-1,:]+v[i,1:,:])/2
        u_n = np.transpose(u_n)
        v_n = np.transpose(v_n)
        X,Y = np.meshgrid(np.linspace(0,1,u_n.shape[0]*2),np.linspace(0,1,u_n.shape[1]*2))
        points_n = np.concatenate([X_n.reshape(X_n.size,1),Y_n.reshape(Y_n.size,1)],axis=1)
        values_n = np.concatenate([u_n.reshape(u_n.size,1),v_n.reshape(v_n.size,1)],axis=1)
        interp = LinearNDInterpolator(points_n, values_n)
        values_i = interp(X,Y)
        u_c = values_i[:,:,0]
        v_c = values_i[:,:,1]
        ax = plt.subplot(a,b,i+1)
        ax.set_ylim([0,1])
        ax.set_xlim([0,1])
        c = ax.pcolormesh(X,Y,np.sqrt(u_c**2+v_c**2))
        plt.colorbar(c)
        ax.streamplot(X,Y,u_c,v_c,color="k",linewidth=2,density=1)
        plt.gca().set_aspect('equal', adjustable='box')
    if show:
        plt.show()


def PlotX(a,b,x_train,prob):
    n = min(a*b,x_train.shape[0])
    plt.figure(figsize=(14,14))
    for i in tqdm(range(n)):
        X_n,Y_n = prob.X_p,prob.Y_p
        x_lim = (prob.xc_p[0],prob.xc_p[-1])
        y_lim = (prob.yc_p[0],prob.yc_p[-1])
        u_n = x_train[i,:,:,0]
        v_n = x_train[i,:,:,1]
        # u_n = np.transpose(u_n)
        # v_n = np.transpose(v_n)
        X,Y = np.meshgrid(np.linspace(x_lim[0],y_lim[-1],u_n.shape[0]*2),np.linspace(y_lim[0],y_lim[-1],u_n.shape[1]*2))
        points_n = np.concatenate([X_n.reshape(X_n.size,1),Y_n.reshape(Y_n.size,1)],axis=1)
        values_n = np.concatenate([u_n.reshape(u_n.size,1),v_n.reshape(v_n.size,1)],axis=1)
        interp = LinearNDInterpolator(points_n, values_n)
        values_i = interp(X,Y)
        u_c = values_i[:,:,0]
        v_c = values_i[:,:,1]
        ax = plt.subplot(a,b,i+1)
        ax.set_ylim(x_lim)
        ax.set_xlim(y_lim)
        c = ax.pcolormesh(X,Y,np.sqrt(u_c**2+v_c**2))
        plt.colorbar(c)
        ax.streamplot(X,Y,u_c,v_c,color="k",linewidth=2,density=1)
        plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
