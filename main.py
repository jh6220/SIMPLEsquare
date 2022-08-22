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
import analysisFunctions as af

# constants definition
n = 30
rho = 1
mu = 0.01
nx = ny = n
dt = np.inf
alpha_uv = 0.5
alpha_p = 0.2
CDS = True
order = 1
dif_tolerance = 10**(-4)
max_iterations = 2000

# get the 1D discritization grid in x and y direction
x = sf.get1Dgrid(n+1,order)
y = x

# define random boundary condition for lid driven cavity flow
order_bc = 5
BCu,BCv,u_bc,v_bc,u_bc_cor,v_bc_cor = sf.GetBC2(x,order=order_bc,continuous=False)

# define the structure that hold all the constants of the CFD problem
prob = sf.CFDproblem(x,y, rho, mu, BCu, BCv,dt,alpha_uv,alpha_p,CDS,dif_tolerance,max_iterations)

# solve the CFD problem
u,v,p,log = sf.SolveProblem(prob,True)

u_n = (u[:,:-1]+u[:,1:])/2
v_n = (v[:-1,:]+v[1:,:])/2

af.PlotConvergence(log)
af.PlotFlow(u,v,prob)