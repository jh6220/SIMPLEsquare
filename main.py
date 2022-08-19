import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve,bicgstab
from scipy.interpolate import LinearNDInterpolator
import copy
import time
import pandas as pd
import solverFunctions as sf
import postProcessingFunctions as ppf

# constants definition
n = 30
rho = 1
mu = 0.01
nx = ny = n
dt = np.inf
alpha_uv = 0.07
alpha_p = 0.02
CDS = True
order = 1

# get the 1D discritization grid in x and y direction
x = sf.get1Dgrid(n+1,order)
y = x

# define boundary condition for lid driven cavity flow
# axis=1 is the bc values along the wall; axis=0 differentiates the walls:
# 0 - south; 1 - west; 2 - north; 3 - south
BCu = np.zeros((4,n+2))
BCv = np.zeros((4,n+2))
BCu[2,:] = 1 # u-velocity of 1 on the north boundary wall
# BCu,BCv,u,v = sf.GetBC2(nx)

# define the structure that hold all the constants of the CFD problem
prob = sf.CFDproblem(x,y, rho, mu, BCu, BCv,dt,alpha_uv,alpha_p,CDS)

# initialize arrays storing the fluid properties
# u,v = np.zeros((prob.nx+1,prob.ny+2)),np.zeros((prob.nx+2,prob.ny+1)) # if reshaped/flattened aixs=1 (y) will remain together
# u_star,v_star = np.zeros((prob.nx+1,prob.ny+2)),np.zeros((prob.nx+2,prob.ny+1)) # intermediate step in the calculation
# u_vector,v_vector = u[1:-1,1:-1].flatten(),v[1:-1,1:-1].flatten()
# p,p_prime,p_vector = np.zeros((prob.nx,prob.ny)),np.zeros((prob.nx,prob.ny)),np.zeros(prob.nx*prob.ny)

# define the iteration and convergence parameters 
dif = 1
dif_tolerance = 10**(-4)
itt_max = 2000
# define structure for storing the iteretion history 
log = ppf.Log(itt_max)

u,v = sf.ApplyVelBC(u,v,prob)

while (dif>dif_tolerance and log.itt<itt_max):
    # Iterate the fluid properties until the convergence criteria is satisfied or
    # the maximum number of iterations is reached
    
    
    Au,bu,Au_p = sf.MomentumEqU(u,v,p,prob)
    Av,bv,Av_p = sf.MomentumEqV(u,v,p,prob)
    log.u_res[log.itt],log.v_res[log.itt] = ppf.CalcMomentumResiduals_A(u,v,Au,Av,bu,bv)
    
    u_star[:,:] = u[:,:]
    v_star[:,:] = v[:,:]
    u_star,u_vector = sf.SolveMomentum(Au,bu,u_star)
    v_star,v_vector = sf.SolveMomentum(Av,bv,v_star)
    u_star = u + prob.alpha_uv*(u_star-u)
    v_star = v + prob.alpha_uv*(v_star-v)
    u_star,v_star = sf.ApplyVelBC(u_star,v_star,prob)
    
    log.con_res[log.itt] = ppf.CalcContinuityResiduals(u_star,v_star,prob)
    log.res[log.itt] = log.u_res[log.itt]+log.v_res[log.itt]+log.con_res[log.itt]
    
    Ap,bp,dmp = sf.pressureCorrectionCoff(u_star,v_star,p,Au_p,Av_p,prob)
    p_prime,p_vector = sf.SolvePressure(Ap,bp,prob)
    u_prime,v_prime = sf.VelocityCorrection(p_prime,Au_p,Av_p,u,v,prob)
    u_star[1:-1,1:-1] -= alpha_uv*u_prime
    v_star[1:-1,1:-1] -= alpha_uv*v_prime
    p -= alpha_p*p_prime
    
    dif = np.linalg.norm(u-u_star)+np.linalg.norm(v-v_star)+np.linalg.norm(p_prime)
    log.dif[log.itt] = dif
    
    u[:,:] = u_star[:,:]
    v[:,:] = v_star[:,:]
    u,v = sf.ApplyVelBC(u,v,prob)
    
    log.itt+=1
    print('\rdif={:.4g}; res={:.4g}; itt={}\t\t'.format(dif,log.res[log.itt-1],log.itt), end="")

print("\ndone")

u_n = (u[:,:-1]+u[:,1:])/2
v_n = (v[:-1,:]+v[1:,:])/2

ppf.PlotConvergence(log)
ppf.PlotFlow(u,v,prob)